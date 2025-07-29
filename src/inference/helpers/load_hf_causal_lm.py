# src/inference/helpers/load_hf_causal_lm.py

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_hf_causal_lm(model_id, cache_dir=None, dtype="auto", model_max_length=4096):
    """
    Loads a HuggingFace causal language model and tokenizer.

    Args:
        model_id: Model name or path (string)
        cache_dir: Optional cache directory
        dtype: torch dtype, e.g., torch.bfloat16 or "auto"
        model_max_length: (int) tokenizer max length

    Returns:
        model, tokenizer
    """
    print(f"[load_hf_causal_lm] Loading {model_id} (cache_dir={cache_dir})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="left"  # Important for decoder-only models when batching
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=dtype if dtype != "auto" else None,
        device_map="cuda"  # Auto-assign to all available CUDA devices (GPUs)
    )
    model.eval()
    return model, tokenizer
