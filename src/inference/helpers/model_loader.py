# src/inference/helpers/model_loader.py

from src.inference.helpers.load_hf_causal_lm import load_hf_causal_lm
from src.inference.helpers.hf_constants import MODEL_MAP

def get_model_loader(model_key):
    """
    Returns a loader function that loads the correct model/tokenizer.
    Uses Hugging Face's generic loader by default.
    """
    model_key_lower = model_key.lower()
    for prefix, model_id in MODEL_MAP.items():
        if model_key_lower.startswith(prefix):
            print(f"[model_loader] Using loader 'load_hf_causal_lm' for model_key '{model_key}' (prefix='{prefix}')", flush=True)
            return lambda *_args, **_kwargs: load_hf_causal_lm(model_id, *_args, **_kwargs)
    # Fallback: load using the provided key directly
    print(f"[model_loader] Using fallback loader for '{model_key}'", flush=True)
    return lambda *_args, **_kwargs: load_hf_causal_lm(model_key, *_args, **_kwargs)
