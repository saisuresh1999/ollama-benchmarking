# src/inference/dispatcher.py

from src.inference.hf_infer import hf_response

# You can add more backends here (e.g., openai, vllm, etc.)
LLM_DISPATCH = {
    "llama3.1:405b": hf_response,
    "llama3.1:70b": hf_response,
    "llama3.1:8b": hf_response,
    "qwq": hf_response,
    "qwen-qwq32b": hf_response,
    "deepseek-r1-8b": hf_response,
    "deepseek": hf_response,
    "deepseek-r1-32b": hf_response,
    "deepseek-qwen": hf_response,
    "deepseek-qwen-32b": hf_response,
    "mistral-small:22b": hf_response,
    "mistral:8b": hf_response,
    # Add more model aliases here if needed
}

def generate_response(paper_content, model_name, temperature, model=None, tokenizer=None):
    if model_name not in LLM_DISPATCH:
        raise ValueError(f"Unsupported model: {model_name}")
    # Call the correct backend (currently all hf_response)
    return LLM_DISPATCH[model_name](
        paper_content,
        temperature,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name
    )
