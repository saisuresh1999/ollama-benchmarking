import torch
import gc
from transformers import StoppingCriteria, StoppingCriteriaList
from src.inference.helpers.hf_constants import REVIEWER_PROMPT

class StopOnStringCriteria(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        super().__init__()
        self.stop_string = stop_string
        self.tokenizer = tokenizer
        self.stop_tokens = tokenizer.encode(stop_string, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        for sequence in input_ids:
            if len(sequence) >= len(self.stop_tokens):
                if list(sequence[-len(self.stop_tokens):].cpu().numpy()) == self.stop_tokens:
                    return True
        return False

def hf_response(paper, temperature, model, tokenizer, model_name=None):
    # If paper is a single prompt, make it a list for batching
    if isinstance(paper, str):
        papers = [paper]
    else:
        papers = paper

    messages_list = [
        [
            {"role": "system", "content": REVIEWER_PROMPT.strip()},
            {"role": "user", "content": p}
        ]
        for p in papers
    ]
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    input_token_count = inputs["input_ids"].shape[1]

    # stop_string = "<|endofanalysis|>"
    # stopping_criteria = StoppingCriteriaList([StopOnStringCriteria(stop_string, tokenizer)])

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                temperature=temperature,
                max_length=131072,
                pad_token_id=tokenizer.eos_token_id
            )
        responses = []
        for i, output in enumerate(outputs):
            gen_token_ids = output[input_token_count:]
            review_part = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
            # Optionally, truncate at stop_string if the model generates extra stuff after it
            if stop_string in review_part:
                review_part = review_part.split(stop_string)[0].strip()
            responses.append({
                "review": review_part,
                "input_token_count": input_token_count
            })
        # If only one, return a single dict for backward compatibility
        if len(responses) == 1:
            return responses[0]
        return responses
    except torch.cuda.OutOfMemoryError as e:
        print("[ERROR] CUDA OOM! Skipping this review.", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()
