import os
import json
import argparse
import torch
from src.inference.helpers.model_loader import get_model_loader
from src.inference.dispatcher import generate_response
import csv
import re
from src.prompts.prompt_templates import anomaly_prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key (e.g., llama3.1:8b)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--data", default="data/data_for_llms.json")
    parser.add_argument("--batch_size", type=int, default=60, help="Batch size for LLM inference")
    return parser.parse_args()

def extract_json_response(llm_output):
    try:
        match = re.search(r'\{[\s\S]+?\}', llm_output)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None

def load_existing_posting_ids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    posting_ids = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            posting_ids.add(row["posting_id"])
    return posting_ids

def write_results_append(results, csv_path):
    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["posting_id", "model", "explanation", "anomaly"])
        if write_header:
            writer.writeheader()
        writer.writerows(results)

def main():
    args = parse_args()
    model_tag = args.model.replace(":", "_")
    OUTPUT_CSV = f"outputs/llm_anomaly_reviews_{model_tag}.csv"
    existing_posting_ids = load_existing_posting_ids(OUTPUT_CSV)

    cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/netscratch/smacharla/hf/transformers")
    model_loader = get_model_loader(args.model)
    model, tokenizer = model_loader(
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        model_max_length=128000
    )

    with open(args.data) as f:
        data = json.load(f)

    BATCH_SIZE = args.batch_size
    unprocessed = [
        (posting_id, lines)
        for posting_id, lines in data.items()
        if posting_id not in existing_posting_ids
    ]

    for i in range(0, len(unprocessed), BATCH_SIZE):
        batch = unprocessed[i:i+BATCH_SIZE]
        batch_ids = [pid for pid, _ in batch]
        batch_prompts = [anomaly_prompt(lines) for _, lines in batch]
        responses = generate_response(
            batch_prompts,
            args.model,
            args.temperature,
            model=model,
            tokenizer=tokenizer
        )
        # Make sure responses is a list (even if batch size = 1)
        if isinstance(responses, dict):
            responses = [responses]

        results = []
        for (posting_id, _), resp in zip(batch, responses):
            llm_out = resp["review"] if "review" in resp else resp.get("text", "")
            parsed = extract_json_response(llm_out)
            if parsed is not None and "anomaly" in parsed:
                results.append({
                    "posting_id": posting_id,
                    "model": args.model,
                    "explanation": parsed["explanation"],
                    "anomaly": parsed["anomaly"]
                })
            else:
                print(f"[WARN] No valid JSON for {posting_id}: {llm_out[:60]}", flush=True)

        write_results_append(results, OUTPUT_CSV)
        print(f"Appended {len(results)} results to {OUTPUT_CSV} (batch {i//BATCH_SIZE + 1})", flush=True)

if __name__ == "__main__":
    main()
