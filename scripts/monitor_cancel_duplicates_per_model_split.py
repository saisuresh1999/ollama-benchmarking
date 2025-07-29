import subprocess
import time
import os

SLEEP_INTERVAL = 10  # seconds
MODEL_NAMES = [
    "llama3.1:70b",
    "llama3.1:8b",
    "qwq",
    "deepseek-r1-8b",
    "deepseek-qwen-32b",
    "mistral:8b",
    "mistral-small:22b",
    # Add more if needed
]

print(f"[INFO] Monitoring and canceling duplicate jobs for model names: {MODEL_NAMES}")

def get_jobs():
    output = os.popen(f"squeue -u $USER --format='%A %j %T'").read().strip().split("\n")[1:]
    jobs = []
    for line in output:
        parts = line.strip().split(None, 2)  # Job ID, Name, State
        if len(parts) == 3:
            jobs.append(tuple(parts))
    return jobs

def extract_model_and_split(jobname):
    for model in MODEL_NAMES:
        # Example: "llama3.1:70b-A100-80GB-split1"
        if jobname.startswith(model):
            tokens = jobname.split('-')
            split_id = None
            for t in tokens:
                if t.startswith("split"):
                    split_id = t.replace("split", "")
                    break
            return model, split_id
    return None, None

while True:
    jobs = get_jobs()
    # Group jobs by (model, split_id)
    job_map = {}
    for job_id, name, state in jobs:
        model, split_id = extract_model_and_split(name)
        if model and split_id:
            group_key = (model, split_id)
            if group_key not in job_map:
                job_map[group_key] = []
            job_map[group_key].append((job_id, state, name))
    for group_key, job_list in job_map.items():
        running = [j for j in job_list if j[1] in ("RUNNING", "R")]
        pending = [j for j in job_list if j[1] in ("PENDING", "PD")]
        if running:
            keep_id = running[0][0]
            to_cancel = [j[0] for j in job_list if j[0] != keep_id]
            for jid in to_cancel:
                print(f"⛔ Canceling duplicate job {jid} for model {group_key[0]} split {group_key[1]} (since {keep_id} is already RUNNING)")
                subprocess.run(f"scancel {jid}", shell=True)
            print(f"[INFO] Keeping RUNNING job {keep_id} for model {group_key[0]} split {group_key[1]}: {running[0][2]}")
        else:
            print(f"[INFO] No RUNNING job for model {group_key[0]} split {group_key[1]}; letting all pending jobs stay.")

    print(f"[INFO] Sleeping for {SLEEP_INTERVAL}s...")
    time.sleep(SLEEP_INTERVAL)
