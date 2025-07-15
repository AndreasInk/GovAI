#!/usr/bin/env python3
"""
doc-mcp – HOA document retrieval MCP server
===========================================

A *minimal* remote MCP server exposing **search** and **fetch** tools so
OpenAI Deep‑Research (or any other agent) can browse the Plantation HOA legal
corpus.

Design goals
------------
* Zero heavy dependencies – uses pure‑Python regex search (no vector DB yet).
* Lazy indexing: PDFs are extracted and chunked on first request, then cached.
* Stable IDs: each chunk is addressed as  "<file_id>_<page_no>_<chunk_idx>"  so you can
  embed `[C-<chunk_idx>]` tags in summaries.

Directory layout
----------------
docs/
  ├── L3HhaocBJ54kke7E39spK7.pdf   # Bylaws Approved 08‑27‑2024
  ├── TYEHpDRQoBke7TS85cs7vx.pdf   # SeventhDeclaration.pdf
  ├── 6Ffj7QA3iSSdzMqBH6WoRj.pdf   # ADB PROPERTY ORG MGMT
  └── … (other PDFs)

The file names match the *file IDs* you supplied so citation IDs stay stable
even if the visible title changes.

Chunking rules
--------------
* 400‑token (~1,600‑char) windows, sentence‑boundary aligned.
* The PDF page number is recorded for citation UI helpers.

Example ID
----------
    "L3HhaocBJ54kke7E39spK7_94_02"   →  file‑id ✚ page‑no ✚ chunk‑idx

Returned search hits follow the Deep‑Research MCP schema:

```json
{
  "id":  "L3HhaocBJ54kke7E39spK7_94_02",
  "document_name": "Bylaws Approved 08.27.2024.pdf",
  "page_number": 94,
  "content": "§6.3 Suspension ..."
}
```
"""
from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import List

import pypdf                           # lightweight PDF text extraction

from fastmcp import FastMCP

DOC_DIR = Path(__file__).parent / "docs"
TOKEN_LIMIT = 400                      # ≈ 400 tokens ≈ 1,600 chars
CHUNK_OVERLAP = 0                      # no overlap for deterministic IDs

mcp = FastMCP("hoa-docs-mcp")

# ---------------------------------------------------------------------------#
#                       PDF → [chunks]  lazy indexer                         #
# ---------------------------------------------------------------------------#
class _Indexer:
    def __init__(self):
        self.cache: dict[str, list[dict]] = {}          # file_id → list[chunks]

    def _pdf_text(self, pdf_path: Path) -> List[tuple[int, str]]:
        """Return list[(page_no, text)] for each page."""
        reader = pypdf.PdfReader(str(pdf_path))
        out: List[tuple[int, str]] = []
        for i, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            out.append((i, text))
        return out

    def _chunk_page(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        buf = []
        char_count = 0
        for sent in sentences:
            buf.append(sent)
            char_count += len(sent)
            if char_count >= TOKEN_LIMIT * 4:   # rough char→token
                chunks.append(" ".join(buf).strip())
                buf, char_count = [], 0
        if buf:
            chunks.append(" ".join(buf).strip())
        return chunks

    def _load_pdf(self, file_id: str) -> list[dict]:
        """Extract + chunk a PDF, return list[chunk dict]."""
        file_id = file_id.rstrip()  # trim accidental trailing spaces
        pdf_path = DOC_DIR / f"{file_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        chunks = []
        for page_no, page_text in self._pdf_text(pdf_path):
            for chunk_idx, chunk in enumerate(self._chunk_page(page_text)):
                chunks.append(
                    {
                        "id": f"{file_id}_{page_no}_{chunk_idx}",
                        "document_name": pdf_path.name,
                        "page_number": page_no,
                        "content": chunk,
                    }
                )
        return chunks

    def chunks_for(self, file_id: str) -> list[dict]:
        if file_id not in self.cache:
            self.cache[file_id] = self._load_pdf(file_id)
        return self.cache[file_id]


index = _Indexer()

# Simple inverted‑index for fast regex search (word → [id,...])
inv_index: defaultdict[str, set[str]] = defaultdict(set)


def _index_if_needed(file_id: str):
    for ch in index.chunks_for(file_id):
        for word in re.findall(r"\w+", ch["content"].lower()):
            inv_index[word].add(ch["id"])


def _search_chunks(query: str, top_k: int = 8) -> List[dict]:
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return []

    # Simple AND requirement: candidate ids that contain all terms
    candidate_ids = set.intersection(*(inv_index[t] for t in terms if t in inv_index)) \
        if all(t in inv_index for t in terms) else set()

    # Fallback: OR if AND yields nothing
    if not candidate_ids:
        candidate_sets = [inv_index[t] for t in terms if t in inv_index]
        candidate_ids = set().union(*candidate_sets) if candidate_sets else set()

    # Rank by naive TF (term freq sum)
    scored = []
    for cid in candidate_ids:
        chunk = _id_to_chunk(cid)
        score = sum(chunk["content"].lower().count(t) for t in terms)
        scored.append((score, chunk))
    scored.sort(key=lambda tup: tup[0], reverse=True)

    return [c for _, c in scored[:top_k]]


def _id_to_chunk(cid: str) -> dict:
    """
    Resolve a stable chunk ID "<file_id>_<page>_<idx>" to the cached chunk dict.

    The file_id may itself contain underscores, so we split off the **last two**
    underscore‑separated parts (page_no and chunk_idx) and join the rest back
    to recover the original file_id.
    """
    try:
        file_id, _, _ = cid.rsplit("_", 2)  # keep everything before last 2 "_"
    except ValueError:                       # not enough segments
        raise KeyError(cid)

    for ch in index.chunks_for(file_id):
        if ch["id"] == cid:
            return ch
    raise KeyError(cid)

# ---------------------------------------------------------------------------#
#                               MCP tools                                    #
# ---------------------------------------------------------------------------#
@mcp.tool()
async def search(query: str) -> dict:
    """
    Full‑text search across **all** PDFs.

    Returns a dict with key "results" -> list (up to 8 items). Each result is a dict with:
        - id: chunk id string
        - title: string, e.g. "Bylaws Approved 08.27.2024.pdf (p.94)"
        - text: short snippet (single line, max ~200 chars)
        - url: string, e.g. "mcp://<chunk_id>"
    """
    # ensure all docs are indexed once
    for pdf in DOC_DIR.glob("*.pdf"):
        _index_if_needed(pdf.stem)

    TOP_K = 8  # Deep‑Research expects at most 8 hits
    hits = _search_chunks(query, TOP_K)
    results = []
    for ch in hits:
        # Create a short, single‑line snippet for the UI
        snippet = textwrap.shorten(
            ch["content"].strip().replace("\n", " "),
            width=200,
            placeholder="…",
        )
        results.append(
            {
                "id": ch["id"],
                "title": f"{ch['document_name']} (p.{ch['page_number']})",
                "text": snippet,
                "url": f"mcp://{ch['id']}",
            }
        )
    return {"results": results}


@mcp.tool()
async def fetch(id: str) -> dict:
    """
    Fetch full chunk content for *id*.

    *id* must match the "<file_id>_<page_no>_<chunk_idx>" pattern returned by
    `search` (e.g. "L3HhaocBJ54kke7E39spK7_94_02").
    """
    chunk = _id_to_chunk(id)
    return chunk


# ---------------------------------------------------------------------------#
# CLI entry‑point
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    print("✔ HOA docs MCP server ready at 'hoa-docs-mcp'")
    mcp.run()