import time
from openai import OpenAI
from pathlib import Path

def wait_for(job_id, sleep_sec=15):
    client = OpenAI()
    while True:
        r = client.responses.retrieve(job_id)
        if r.status == "completed":
            return r.output_text
        if r.status == "failed":
            raise RuntimeError(f"{job_id} failed")
        time.sleep(sleep_sec)

markdown_report = wait_for("resp_686c69249240819aa6ce274ba6c04f71004e0a045a94812a")
Path("draft.md").write_text(markdown_report)
print("Saved to draft.md")