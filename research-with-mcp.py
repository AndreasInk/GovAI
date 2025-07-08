#!/usr/bin/env python3
"""
research-mcp.py – Kick‑off a Deep‑Research job against the HOA MCP server
========================================================================

This script invokes **o4-mini-deep-research** to generate a ≤50‑page
consolidated summary of all HOA governing documents served by our
`doc-mcp.py` instance.

Quick‑start
-----------
1.  Start the MCP server in another terminal::

        python doc-mcp.py  # listens on http://localhost:8000/mcp

2.  Export your OpenAI key::

        export OPENAI_API_KEY=sk-...

3.  Run this driver::

        python research-mcp.py

Behaviour
---------
* Runs in **background** mode so it returns immediately with a request‑ID.
* Uses the MCP as *sole* data‑source (no public web search).
* Requests a reasoning trace summary for easier debugging.
* Sets `max_tool_calls` to 200 – adjust if you hit cost/time limits.
* Polls for completion if `--wait` is given.

The resulting summary will contain inline citations like `[C-221]` where
`221` refers to the `chunk_id` provided by the MCP.  Save the `output_text`
to `draft.md`, then run the Streamlit drift‑checker to flag issues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

MODEL = "o4-mini-deep-research"
MCP_LABEL = "hoa_docs"
MCP_URL = os.getenv("HOA_MCP_URL", "https://0d9cfd389bc5.ngrok-free.app/mcp")
OUT_FILE = Path("draft.md")

# ---------------------------------------------------------------------------#
# Deep‑research prompt                                                       #
# ---------------------------------------------------------------------------#
PROMPT = """
Document Consolidation Outline – The Plantation at Ponte Vedra Beach
===================================================================

You have full access to the **hoa_docs** MCP, which contains every governing
PDF (articles, bylaws, declaration, rules, enforcement & financial policies).

### Task
1. Write a concise **Executive Summary** (≤ 600 words, narrative paragraphs)
   capturing the core themes, rights, duties, and restrictions across all
   governing documents.

2. Follow the summary with a **three‑level Markdown OUTLINE** –
   *Part › Article › Section* – that captures **every distinct right,
   obligation, power, restriction, or procedure**.  Each leaf bullet
   ≤ 25 words.

Your deliverable should be **comprehensive** — aim for **≈ 25–30 pages
(~12 000–15 000 words)** when rendered, which typically means **300–400 leaf
bullets** after the executive summary.

### Citation requirement
After **every** leaf bullet append the stable chunk ID you fetched, e.g.:

    • Suspension of member privileges after 30‑day delinquency. [C-Rules_Regulations_10_0]

If a sentence summarises multiple chunks, list each token:
`[C-Bylaws_4_2][C-Bylaws_4_3]`.

### Scope
Include: Articles of Incorporation, Declaration of Covenants, Bylaws, all
facility‑specific rules, enforcement & financial policies.  Omit cover pages,
signature blocks, notarisation seals.

### Guidelines
* Quote verbatim where nuance matters (stay within the 25‑word bullet limit).
* **Do not** perform any external web research.
* Return only the **executive summary + outline** Markdown — no extra commentary, no appendix.
"""


# ---------------------------------------------------------------------------#
# Helper – wait for job completion                                           #
# ---------------------------------------------------------------------------#
def _poll_job(client: OpenAI, job_id: str, every_sec: int = 15) -> str:
    print(f"ℹ Waiting for job {job_id} to complete …", file=sys.stderr)
    while True:
        job = client.responses.retrieve(job_id)
        if getattr(job, "status", "") == "completed":
            return getattr(job, "output_text", "")
        if getattr(job, "status", "") == "failed":
            raise RuntimeError(f"Deep‑research job {job_id} failed")
        time.sleep(every_sec)


# ---------------------------------------------------------------------------#
# Main                                                                       #
# ---------------------------------------------------------------------------#
def main(wait: bool = False, out: Optional[Path] = None) -> None:
    client = OpenAI(timeout=3600)

    print("🚀 Launching deep‑research request …", file=sys.stderr)
    resp = client.responses.create(
        model=MODEL,
        reasoning={"summary": "auto"},
        max_tool_calls=200,
        input=PROMPT.strip(),
        tools=[
            {
                "type": "mcp",
                "server_label": MCP_LABEL,
                "server_url": MCP_URL,
                "require_approval": "never"
            },
            # Uncomment if you want public web search as a secondary source
            # {"type": "web_search_preview"},
        ]
    )

    job_id = resp.id
    print(f"🆔 Job ID: {job_id}")
    print(f"📊 Status: {getattr(resp, 'status', 'submitted')}")
    if not wait:
        print("Run again with --wait to poll for completion.")
        return

    output = _poll_job(client, job_id)
    if not out:
        out = OUT_FILE
    out.write_text(output)
    print(f"✅ Report saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HOA deep‑research job")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the job finishes and write output to draft.md",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Custom output markdown file (requires --wait)",
    )
    args = parser.parse_args()
    main(wait=args.wait, out=args.out)