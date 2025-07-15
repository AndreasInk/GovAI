#!/usr/bin/env python3
"""
research-mcp.py – Kick‑off a Deep‑Research job against the HOA MCP server
========================================================================

This script invokes **o3-deep-research** to generate a ≤50‑page
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
import os
import sys
import time
from pathlib import Path
from typing import Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

MODEL = "o4-mini-deep-research"
MCP_LABEL = "hoa_docs"
MCP_URL = os.getenv("HOA_MCP_URL", "https://cdcd024ac046.ngrok-free.app/mcp")
OUT_FILE = Path("draft.md")

# Standard 4‑part breakdown inspired by the consolidated master document
DEFAULT_TOPICS: list[str] = [
    "Foundational Documents",          # Part I
    "Community‑Wide Rules",            # Part II
    "Amenities & Facilities",          # Part III
    "Governance & Enforcement",        # Part IV
]

# ---------------------------------------------------------------------------#
# Deep‑research prompt                                                       #
# ---------------------------------------------------------------------------#
PROMPT = """
## Role
You are the **Plantation HOA Deep‑Research Analyst**.  
Your job is to condense the ≈ 500‑page HOA governing‑document corpus (in `/docs`) into a single **30 – 40 page equivalent** and return the result as a machine‑friendly JSON object.

---
### Task
1. Use the MCP **search** tool to discover which documents exist. Start broad (`bylaws`, `declaration`, `rules`, `enforcement`, `policies`) and then refine with context terms (`parking`, `pets`, `guest`, etc.).
2. Write an **executive_summary** (≤ 600 words, narrative paragraphs) that captures the core themes, rights, duties, restrictions, and enforcement mechanisms across everything you fetched.
3. Build a sections array with **≥ 150 entries** (roughly 4-5 per 30-40 pages).:  
   • `source_document` (string) PDF file name  
   • `source_page`  (int)  page number inside that PDF  
   • `source_text`  (string) verbatim clause or paragraph you fetched  
   • `summary_text`  (string) one‑sentence paraphrase of the clause
4. Order the array roughly by document hierarchy (Bylaws → CC&Rs → Rules → Policies).

---
### Required JSON Output (format)
```jsonc
{
  "executive_summary": "<600‑word narrative …>",
  "sections": [
    {
      "source_document": "Bylaws Approved 08.27.2024.pdf",
      "source_page": 94,
      "source_text": "§6.3 Suspension of member privileges …",
      "summary_text": "Board may suspend any membership privilege after 30‑day delinquency."
    }
    // …more entries…
  ]
}
```

---
### Important Rules
* **Every** fact in `summary_text` must be supported by the corresponding `source_text`.
* Internal stable chunk IDs are **not** required; the metadata above suffices.
* Quote brittle language verbatim in `source_text` to preserve nuance.
* Fetch only what you intend to cite—avoid unnecessary tool calls.
* No external web research; answer strictly from fetched chunks.
* If a relevant document is truly missing, mention that in `executive_summary`.
* Keep `summary_text` concise (≈ 1 sentence) so downstream embeddings can measure similarity against `source_text`.
"""

KICKOFF_PROMPT = """
### HOA Deep‑Research Kickoff

The Plantation at Ponte Vedra Beach is an **equity residential community** in Ponte Vedra Beach, Florida.  
Home ownership automatically includes membership to a championship golf course, private beach club, tennis & pickleball center, croquet lawns, fitness center, and a clubhouse offering dining and social events.

We are consolidating **20 + governing documents** into one master reference for easier member access and navigation.

**Current document families**
• Foundational documents – Articles of Incorporation, Bylaws, Declaration of Covenants  
• Facility‑specific rules – golf, tennis, racquet sports, croquet, fitness center, clubhouse, beach house  
• Policy manuals – organizational, financial, architectural design, covenant enforcement  
• Committee charters and other governance docs  

---
**Goal**: produce the JSON digest (executive_summary + sections) described in the main prompt.

**Steps**
1. Run broad **search** queries (`bylaws`, `declaration`, `rules`, `policies`, `enforcement`).
2. Refine with context words (`parking`, `pets`, `guest`, etc.) and **fetch** only the chunks you plan to cite.
3. Assemble the required JSON output.

**Search cheat‑sheet**

*Foundational* `bylaws`, `articles`, `declaration`, `covenants`, `CC&Rs`  
*Governance* `board`, `committee charter`, `quorum`, `voting`  
*Money* `assessment`, `dues`, `fee schedule`, `fine`, `lien`, `delinquency`  
*Amenities* `golf`, `tennis`, `pickleball`, `croquet`, `beach house`, `fitness`, `clubhouse`  
*Property use* `architectural`, `construction`, `renovation`, `landscaping`, `signage`, `noise`, `pets`, `parking`, `guest`  
*Enforcement* `violation`, `notice`, `hearing`, `suspension`, `sanction`  

> Combine broad + specific terms, e.g. `golf guest`, `architectural tree removal`, or `assessment late fee`.  

Use only the `hoa-docs-mcp` tools (`search`, `fetch`). No external web.
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
# Helper – kick off a single topic‑focused job                               #
# ---------------------------------------------------------------------------#
def _run_topic(topic: str, wait: bool, out_dir: Path = Path(".")) -> None:
    """
    Launch a deep‑research job constrained to a specific TOPIC (e.g. 'parking',
    'pets', 'amenities') and, if wait=True, write its JSON result to
    draft-<topic>.json.
    """
    client = OpenAI(timeout=3600)

    instructions_text = PROMPT.strip() + (
        "\n\n---\n### Topic Focus\n"
        f"Only research clauses, rules, fees, and policies relevant to the topic: **{topic}**.\n"
        "Ignore unrelated parts of the corpus.\n"
    )

    resp = client.responses.create(
        model=MODEL,
        reasoning={"summary": "auto"},
        max_tool_calls=200,
        instructions=instructions_text,
        input=KICKOFF_PROMPT.strip(),
        tools=[
            {
                "type": "mcp",
                "server_label": MCP_LABEL,
                "server_url": MCP_URL,
                "require_approval": "never",
            }
        ],
    )

    job_id = resp.id
    print(f"🏷️  Topic '{topic}' ➜ Job {job_id}")

    if not wait:
        return

    output = _poll_job(client, job_id)
    import re
    safe_topic = re.sub(r"\\W+", "_", topic.lower())
    out_path = out_dir / f"draft-{safe_topic}.json"
    out_path.write_text(output)
    print(f"✅ Topic '{topic}' saved to {out_path}")


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
        instructions=PROMPT.strip(),
        input=KICKOFF_PROMPT.strip(),
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
    parser.add_argument(
        "--std-chunks",
        "--parts",
        dest="std_chunks",
        action="store_true",
        help="Kick off the standard 4 part‑based research chunks (Foundational, Rules, Amenities, Governance)"
    )
    parser.add_argument(
        "--topics",
        help="Comma‑separated list of custom research topics to fan‑out in parallel. "
             "Use --std-chunks to run the 4 standard parts instead."
    )
    args = parser.parse_args()
    # Determine fan‑out topics from CLI options
    topics_param = getattr(args, "topics", None)
    topics: list[str] | None = None
    if topics_param:
        topics = [t.strip() for t in topics_param.split(",") if t.strip()]
    elif getattr(args, "std_chunks", False):
        topics = DEFAULT_TOPICS.copy()

    if topics:
        if not topics:
            print("⚠️  No valid topics parsed.", file=sys.stderr)
            sys.exit(0)

        max_workers = min(8, len(topics))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_topic, topic, args.wait, Path(".")): topic
                for topic in topics
            }
            for future in as_completed(futures):
                topic = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"❌ Topic '{topic}' errored: {exc}", file=sys.stderr)
        sys.exit(0)
    main(wait=args.wait, out=args.out)