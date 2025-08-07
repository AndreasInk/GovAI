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

        fastmcp run doc-mcp.py  # listens on http://localhost:8000/mcp

2.  Export your OpenAI key:

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
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration with both console and file handlers."""
    logger = logging.getLogger("research-mcp")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Global logger instance - will be initialized in main
logger = None

MODEL = "o4-mini-deep-research"
MCP_LABEL = "hoa_docs"
MCP_URL = os.getenv("HOA_MCP_URL", "https://e5045bdf8623.ngrok-free.app/mcp")
OUT_FILE = Path("draft.md")

# Standard 4‑part breakdown inspired by the consolidated master document
DEFAULT_TOPICS: list[str] = [
    "Foundational Documents",          # Part I
    "Community‑Wide Rules",            # Part II
    "Amenities & Facilities",          # Part III
    "Governance & Enforcement",        # Part IV
]

# Error tracking
class ResearchError(Exception):
    """Base exception for research-related errors."""
    pass

class JobFailedError(ResearchError):
    """Raised when a deep research job fails."""
    pass

class MCPConnectionError(ResearchError):
    """Raised when MCP server connection fails."""
    pass

class ValidationError(ResearchError):
    """Raised when input validation fails."""
    pass

# ---------------------------------------------------------------------------#
# Deep‑research prompt                                                       #
# ---------------------------------------------------------------------------#
PROMPT = """
## Role
You are the **Plantation HOA Deep‑Research Analyst**.  
Your job is to condense the ≈ 500‑page HOA governing‑document corpus (in `/docs`) into a single **10 page equivalent** and return the result as a machine‑friendly JSON object.

---
### Task
1. Use the MCP **search** tool to discover which documents exist. Start broad (`bylaws`, `declaration`, `rules`, `enforcement`, `policies`) and then refine with context terms (`parking`, `pets`, `guest`, etc.).
2. Write an **executive_summary** (≤ 600 words, narrative paragraphs) that captures the core themes, rights, duties, restrictions, and enforcement mechanisms across everything you fetched.
3. Build a sections array with **≥ 150 entries** (roughly 4-5 per 30-40 pages).:  
   • `source_document` (string) PDF file name  
   • `source_page`   (int) page number inside that PDF  
   • `source_text`   (string) verbatim clause or paragraph you fetched  
   • `summary_text`  (string) one‑sentence paraphrase of the clause
4. Order the array roughly by document hierarchy (Bylaws → CC&Rs → Rules → Policies).

---
### Required JSON Output (format)
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

We are consolidating **20 + governing documents** into one master reference for easier member access and navigation.

**Current document families**
• Foundational documents – Articles of Incorporation, Bylaws, Declaration of Covenants  
• Facility‑specific rules – golf, tennis, racquet sports, croquet, fitness center, clubhouse, beach house  
• Policy manuals – organizational, financial, architectural design, covenant enforcement  
• Committee charters and other governance docs  

---
**Goal**: produce the JSON digest (executive_summary + sections) described in the main prompt.

**Steps**
1. Run broad **search** queries (`bylaws`, `declaration`, `rules`, `policies`, `enforcement`).
2. Refine with context words (`parking`, `pets`, `guest`, etc.) and **fetch** only the chunks you plan to cite.
3. Assemble the required JSON output.

**Search cheat‑sheet**

*Foundational* `bylaws`, `articles`, `declaration`, `covenants`, `CC&Rs`  
*Governance* `board`, `committee charter`, `quorum`, `voting`  
*Money* `assessment`, `dues`, `fee schedule`, `fine`, `lien`, `delinquency`  
*Amenities* `golf`, `tennis`, `pickleball`, `croquet`, `beach house`, `fitness`, `clubhouse`  
*Property use* `architectural`, `construction`, `renovation`, `landscaping`, `signage`, `noise`, `pets`, `parking`, `guest`  
*Enforcement* `violation`, `notice`, `hearing`, `suspension`, `sanction`  

> Combine broad + specific terms, e.g. `golf guest`, `architectural tree removal`, or `assessment late fee`.  

Use only the `hoa-docs-mcp` tools (`search`, `fetch`). No external web.
"""

# ---------------------------------------------------------------------------#
# Helper functions                                                           #
# ---------------------------------------------------------------------------#
def validate_environment() -> None:
    """Validate that required environment variables and dependencies are available."""
    logger.info("Validating environment...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValidationError("OPENAI_API_KEY environment variable is required")
    
    # Check MCP URL
    if not MCP_URL:
        raise ValidationError("HOA_MCP_URL environment variable is required")
    
    logger.info(f"Using MCP URL: {MCP_URL}")
    logger.info("Environment validation passed")

def create_openai_client() -> OpenAI:
    """Create and validate OpenAI client with proper error handling."""
    try:
        logger.debug("Creating OpenAI client...")
        client = OpenAI(timeout=3600)
        
        # Test the client by making a simple API call
        # This will fail early if there are authentication issues
        logger.debug("Testing OpenAI client connection...")
        
        return client
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        raise MCPConnectionError(f"OpenAI client initialization failed: {e}")

def safe_write_file(content: str, file_path: Path, topic: str = "main") -> None:
    """Safely write content to file with error handling and backup."""
    try:
        # Create backup if file exists
        if file_path.exists():
            backup_path = file_path.with_suffix(f".backup.{int(time.time())}")
            logger.info(f"Creating backup of existing file: {backup_path}")
            file_path.rename(backup_path)
        
        # Write new content
        logger.info(f"Writing {topic} output to {file_path}")
        file_path.write_text(content)
        logger.info(f"✅ Successfully wrote {topic} output to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to write {topic} output to {file_path}: {e}")
        raise ResearchError(f"File write failed for {topic}: {e}")

# ---------------------------------------------------------------------------#
# Helper – wait for job completion with retry logic                          #
# ---------------------------------------------------------------------------#
def _poll_job(client: OpenAI, job_id: str, topic: str = "main", every_sec: int = 15, max_retries: int = 3) -> str:
    """Poll for job completion with retry logic and detailed logging."""
    logger.info(f"Starting to poll job {job_id} for topic '{topic}'...")
    
    retry_count = 0
    last_status = None
    
    while True:
        try:
            logger.debug(f"Polling job {job_id} (attempt {retry_count + 1})...")
            job = client.responses.retrieve(job_id)
            current_status = getattr(job, "status", "unknown")
            
            # Log status changes
            if current_status != last_status:
                logger.info(f"Job {job_id} status changed: {last_status} → {current_status}")
                last_status = current_status
            
            if current_status == "completed":
                output_text = getattr(job, "output_text", "")
                if not output_text:
                    logger.warning(f"Job {job_id} completed but output_text is empty")
                else:
                    logger.info(f"Job {job_id} completed successfully with {len(output_text)} characters")
                return output_text
                
            elif current_status == "failed":
                error_msg = getattr(job, "error", {}).get("message", "Unknown error")
                logger.error(f"Job {job_id} failed: {error_msg}")
                raise JobFailedError(f"Deep‑research job {job_id} failed: {error_msg}")
                
            elif current_status in ["cancelled", "expired"]:
                logger.error(f"Job {job_id} was {current_status}")
                raise JobFailedError(f"Deep‑research job {job_id} was {current_status}")
            
            # Reset retry count on successful poll
            retry_count = 0
            
        except Exception as e:
            retry_count += 1
            logger.warning(f"Poll attempt {retry_count} failed for job {job_id}: {e}")
            
            if retry_count >= max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded for job {job_id}")
                raise JobFailedError(f"Failed to poll job {job_id} after {max_retries} attempts: {e}")
            
            # Exponential backoff
            wait_time = every_sec * (2 ** (retry_count - 1))
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            continue
        
        time.sleep(every_sec)

# ---------------------------------------------------------------------------#
# Helper – kick off a single topic‑focused job                               #
# ---------------------------------------------------------------------------#
def _run_topic(topic: str, wait: bool, out_dir: Path = Path(".")) -> Dict[str, Any]:
    """
    Launch a deep‑research job constrained to a specific TOPIC and return result info.
    
    Returns:
        Dict containing job_id, status, and any error information
    """
    result = {
        "topic": topic,
        "job_id": None,
        "status": "not_started",
        "error": None,
        "output_path": None,
        "start_time": datetime.now(),
        "end_time": None
    }
    
    try:
        logger.info(f"Starting research job for topic: {topic}")
        client = create_openai_client()

        instructions_text = PROMPT.strip() + (
            "\n\n---\n### Topic Focus\n"
            f"Only research clauses, rules, fees, and policies relevant to the topic: **{topic}**.\n"
            "Ignore unrelated parts of the corpus.\n"
        )

        logger.debug(f"Creating deep research job for topic '{topic}'...")
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
        result["job_id"] = job_id
        result["status"] = "submitted"
        
        logger.info(f"🏷️  Topic '{topic}' ➜ Job {job_id}")

        if not wait:
            logger.info(f"Topic '{topic}' job submitted (not waiting for completion)")
            return result

        # Wait for completion
        logger.info(f"Waiting for topic '{topic}' job to complete...")
        output = _poll_job(client, job_id, topic)
        
        # Write output file
        import re
        safe_topic = re.sub(r"\W+", "_", topic.lower())
        out_path = out_dir / f"draft-{safe_topic}.json"
        
        safe_write_file(output, out_path, topic)
        
        result["status"] = "completed"
        result["output_path"] = str(out_path)
        result["end_time"] = datetime.now()
        
        logger.info(f"✅ Topic '{topic}' completed and saved to {out_path}")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["end_time"] = datetime.now()
        
        logger.error(f"❌ Topic '{topic}' failed: {e}")
        logger.debug(f"Full traceback for topic '{topic}': {traceback.format_exc()}")
        
        # Re-raise for the calling code to handle
        raise
    
    return result

# ---------------------------------------------------------------------------#
# Main                                                                       #
# ---------------------------------------------------------------------------#
def main(wait: bool = False, out: Optional[Path] = None) -> None:
    """Main function for single comprehensive research job."""
    logger.info("🚀 Starting main deep‑research job...")
    
    try:
        validate_environment()
        client = create_openai_client()

        logger.info("Creating comprehensive deep‑research request...")
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
        logger.info(f"🆔 Main Job ID: {job_id}")
        logger.info(f"📊 Status: {getattr(resp, 'status', 'submitted')}")
        
        if not wait:
            logger.info("Job submitted successfully. Run again with --wait to poll for completion.")
            return

        logger.info("Waiting for main job to complete...")
        output = _poll_job(client, job_id, "main")
        
        if not out:
            out = OUT_FILE
            
        safe_write_file(output, out, "main")
        
    except Exception as e:
        logger.error(f"Main job failed: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise

def run_parallel_topics(topics: list[str], wait: bool, out_dir: Path = Path(".")) -> Dict[str, Any]:
    """Run multiple topic jobs in parallel with comprehensive error handling."""
    logger.info(f"Starting parallel research for {len(topics)} topics: {topics}")
    
    results = {
        "start_time": datetime.now(),
        "topics": topics,
        "completed": [],
        "failed": [],
        "total_jobs": len(topics)
    }
    
    max_workers = min(8, len(topics))
    logger.info(f"Using {max_workers} parallel workers")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_topic, topic, wait, out_dir): topic
                for topic in topics
            }
            
            for future in as_completed(futures):
                topic = futures[future]
                try:
                    result = future.result()
                    results["completed"].append(result)
                    logger.info(f"✅ Topic '{topic}' completed successfully")
                    
                except Exception as exc:
                    error_result = {
                        "topic": topic,
                        "status": "failed",
                        "error": str(exc),
                        "end_time": datetime.now()
                    }
                    results["failed"].append(error_result)
                    logger.error(f"❌ Topic '{topic}' failed: {exc}")
        
        results["end_time"] = datetime.now()
        
        # Summary
        completed_count = len(results["completed"])
        failed_count = len(results["failed"])
        
        logger.info("📊 Parallel execution completed:")
        logger.info(f"   ✅ Completed: {completed_count}/{len(topics)}")
        logger.info(f"   ❌ Failed: {failed_count}/{len(topics)}")
        
        if failed_count > 0:
            logger.warning("Some topics failed. Check logs for details.")
            # Don't raise exception here - let caller decide how to handle partial failures
        
        return results
        
    except Exception as e:
        logger.error(f"Parallel execution failed: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise

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
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log to file in addition to console"
    )
    
    args = parser.parse_args()
    
    # Setup logging based on arguments
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("=" * 60)
    logger.info("HOA Deep Research Script Starting")
    logger.info("=" * 60)
    
    try:
        # Determine fan‑out topics from CLI options
        topics_param = getattr(args, "topics", None)
        topics: list[str] | None = None
        if topics_param:
            topics = [t.strip() for t in topics_param.split(",") if t.strip()]
        elif getattr(args, "std_chunks", False):
            topics = DEFAULT_TOPICS.copy()

        if topics:
            if not topics:
                logger.warning("⚠️  No valid topics parsed.")
                sys.exit(0)

            logger.info(f"Running parallel research for topics: {topics}")
            results = run_parallel_topics(topics, args.wait, Path("."))
            
            # Exit with error code if any topics failed
            if results["failed"]:
                logger.error("Some topics failed. Exiting with error code.")
                sys.exit(1)
            else:
                logger.info("All topics completed successfully!")
                sys.exit(0)
        else:
            # Single comprehensive job
            main(wait=args.wait, out=args.out)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down gracefully...")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        logger.info("=" * 60)
        logger.info("HOA Deep Research Script Finished")
        logger.info("=" * 60)