# src/inference/helpers/hf_constants.py

import os

# ==== Model Aliases ====
MODEL_MAP = {
    # Qwen
    "qwq": "Qwen/QwQ-32B",
    "qwen-qwq32b": "Qwen/QwQ-32B",
    # DeepSeek
    "deepseek-r1-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # Llama
    "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1:70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama3.1:405b": "RedHatAI/Meta-Llama-3.1-405B-Instruct-quantized.w4a16",
    # Mistral
    "mistral-small:22b": "mistralai/Mistral-Small-Instruct-2409",
    "mistral:8b": "mistralai/Ministral-8B-Instruct-2410",
}

# ==== Prompt Template (example) ====
REVIEWER_PROMPT = """
You are an expert tax accountant. Given the following journal entry (possibly suspicious), explain whether it is anomalous or not.
Give your answer as a JSON with two fields:
  "explanation": your reasoning (1-2 lines),
  "anomaly": 1 if anomalous, 0 if not.
Only output valid JSON. Here is the entry:
"""

# ==== HuggingFace Token (optional) ====
HF_TOKEN = os.getenv("HF_TOKEN")

def setup_hf_env():
    """
    Ensure HuggingFace cache/home environment variables are set.
    Sets sensible defaults if not already present.
    """
    os.environ.setdefault("HF_HOME", "/netscratch/smacharla/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/netscratch/smacharla/hf/transformers")
    os.environ.setdefault("HF_HUB_CACHE", "/netscratch/smacharla/hf/hub")
