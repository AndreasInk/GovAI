"""Utilities for retrieving and storing the result of an OpenAI job."""

import time
from openai import OpenAI
from pathlib import Path

def wait_for(job_id: str, sleep_sec: float = 15) -> str:
    """Poll the API until the given job finishes and return its output."""

    client = OpenAI()
    while True:
        r = client.responses.retrieve(job_id)
        if r.status == "completed":
            return r.output_text
        if r.status == "failed":
            raise RuntimeError(f"{job_id} failed")
        time.sleep(sleep_sec)

job_id = "resp_6879aa5c05b4819b8dee3e3ea514f22606a9775ba3735112"
markdown_report = wait_for(job_id)
Path(f"draft-{job_id}.md").write_text(markdown_report)
print("Saved to draft.md")
