#!/usr/bin/env python3
"""
ingest.py – Pre‑processing pipeline for HOA rewrite project
===========================================================

Usage
-----
$ python ingest.py docs/ --out-dir data/
# (optional) add a draft (PDF/DOCX/MD) to compute drift flags immediately
$ python ingest.py docs/ --draft draft.json --out-dir data/

The script:
1) extracts raw text from one or more *source* documents (PDF or Word),
2) splits the text into **token‑bounded chunks** (default ≈400 tokens),
3) fetches OpenAI embeddings for every chunk and saves them to `chunk_vecs.npy`,
4) writes the plain‑text chunks to `chunks.json`,
5) *(optional)* compares each *summary* sentence in `draft.md` against its cited
   source chunks and emits `flags.json`.

Dependencies
------------
pip install pypdf python-docx tiktoken numpy scipy openai

`ai.py` (shipped with this repo) must be importable and have `embed()`.

Environment
-----------
Require `OPENAI_API_KEY`.

Notes
-----
• Chunk IDs are "<fileid>_<page>_<chunk>", matching doc‑mcp.py.
• The cosine‑similarity threshold for drift detection defaults to **0.85**
  but can be tweaked via `--threshold`.
• LLM judge drift detection can be enabled with `--use-llm-judge`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cosine
from tiktoken import get_encoding
from pydantic import BaseModel

# Third‑party extraction libs
from pypdf import PdfReader

try:
    import docx
except ImportError:  # pragma: no cover
    docx = None  # type: ignore

# Local helper (OpenAI wrapper)
import ai  # noqa: E402

ENC = get_encoding("cl100k_base")

# ---------------------------------------------------------------------------#
#                              Pydantic Models                               #
# ---------------------------------------------------------------------------#
class DriftJudgment(BaseModel):
    """Pydantic model for LLM drift judgment response."""
    is_drift: bool
    confidence: float
    reasoning: str

# ---------------------------------------------------------------------------#
#                               Text Extraction                              #
# ---------------------------------------------------------------------------#
def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return " ".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(path: Path) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed – cannot read .docx files")
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(files: Iterable[Path]) -> str:
    parts: List[str] = []
    for fp in files:
        if fp.suffix.lower() == ".pdf":
            parts.append(_extract_pdf(fp))
        elif fp.suffix.lower() in {".docx", ".doc"}:
            parts.append(_extract_docx(fp))
        else:
            sys.exit(f"Unsupported file type: {fp.name}")
    return "\n".join(parts)

# ---------------------------------------------------------------------------#
#                                 Chunking                                   #
# ---------------------------------------------------------------------------#
def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for word in words:
        current.append(word)
        current_tokens += len(ENC.encode(word + " "))
        if current_tokens >= max_tokens:
            chunks.append(" ".join(current).strip())
            current, current_tokens = [], 0

    if current:
        chunks.append(" ".join(current).strip())
    return chunks

# ---------------------------------------------------------------------------#
#                    Stable‑ID chunk generator per source file               #
# ---------------------------------------------------------------------------#
def file_chunks(fp: Path, max_tokens: int = 400) -> List[Tuple[str, str]]:
    """
    Return a list of (chunk_id, chunk_text) for *one* source file.
    The ID format matches doc‑mcp.py:  "<fileid>_<page>_<chunk_idx>".

    Currently supports PDFs only – DOCX fallback uses page_no=0.
    """
    # Normalise filename → lowercase, alphanumeric + underscores only
    file_id = re.sub(r"[^a-z0-9]+", "_", fp.stem.lower())  # collapse any run of non‑alnum chars
    file_id = re.sub(r"_+", "_", file_id).strip("_")       # squeeze repeated "_" and trim edges
    chunks: List[Tuple[str, str]] = []

    if fp.suffix.lower() == ".pdf":
        reader = PdfReader(str(fp))
        for page_no, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            for chunk_idx, chunk in enumerate(chunk_text(page_text, max_tokens)):
                cid = f"{file_id}_{page_no}_{chunk_idx}"
                chunks.append((cid, chunk))

    elif fp.suffix.lower() in {".docx", ".doc"}:
        if docx is None:
            raise RuntimeError("python-docx required for DOCX ingestion")
        text = _extract_docx(fp)
        # treat whole doc as page_no 0
        for chunk_idx, chunk in enumerate(chunk_text(text, max_tokens)):
            cid = f"{file_id}_0_{chunk_idx}"
            chunks.append((cid, chunk))
    else:
        sys.exit(f"Unsupported file type: {fp.name}")

    return chunks


# ---------------------------------------------------------------------------#
#                              Drift Detection                               #
# ---------------------------------------------------------------------------#
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def load_draft_sentences(draft_path: Path):
    """Extract sentences from the *draft* file with extra cleanup.

    For JSON files: returns a list of (summary, source) tuples.
    For other formats: returns a list of summary strings.
    """
    suffix = draft_path.suffix.lower()

    if suffix == ".json":
        return _load_json_sentences(draft_path)
    elif suffix == ".pdf":
        text = _extract_pdf(draft_path)
    elif suffix in {".docx", ".doc"}:
        text = _extract_docx(draft_path)
    else:
        # Fallback: treat as text/markdown – decode with utf‑8
        text = draft_path.read_text(encoding="utf-8", errors="ignore")

    # Normalise whitespace so line‑breaks don't split sentences
    text = re.sub(r"\s+", " ", text).strip()

    # Initial sentence split
    raw_parts = _SENT_SPLIT_RE.split(text)

    # Merge tiny fragments (≤9 words) into the previous sentence
    sentences: List[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if sentences and len(part.split()) < 10:
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)

    return sentences

def _load_json_sentences(json_path: Path):
    """Extract (summary, source) pairs from JSON deep research output files.
    
    For JSON input, we extract the raw source text as-is without chunking,
    preserving the original context and structure from the deep research output.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError(f"Failed to parse JSON file {json_path}: {e}")
    
    pairs: list = []
    
    # Add executive summary if present (no source)
    if "executive_summary" in data and data["executive_summary"]:
        summary = data["executive_summary"].strip()
        if summary:
            pairs.append((summary, None))
    
    # Add summary_text and source_text from each section
    if "sections" in data and isinstance(data["sections"], list):
        for section_idx, section in enumerate(data["sections"]):
            if (
                isinstance(section, dict)
                and "summary_text" in section
                and "source_text" in section
            ):
                summary_text = section["summary_text"].strip()
                source_text = section["source_text"].strip()
                
                if summary_text:
                    # Preserve additional source context if available
                    source_context = {
                        "raw_source": source_text,
                        "section_index": section_idx,
                        "source_document": section.get("source_document", "Unknown"),
                        "source_page": section.get("source_page", "Unknown"),
                        "source_lines": section.get("source_lines", []),
                        "source_chunks": section.get("source_chunks", [])
                    }
                    
                    # If we have source_lines, use those as the primary source
                    if source_context["source_lines"]:
                        # Join source lines with newlines to preserve structure
                        primary_source = "\n".join(source_context["source_lines"])
                    else:
                        # Fall back to source_text
                        primary_source = source_text
                    
                    pairs.append((summary_text, primary_source))
    
    if not pairs:
        raise RuntimeError(f"No valid summary/source pairs found in JSON file {json_path}")
    
    print(f"   → Extracted {len(pairs)} summary/source pairs from JSON file")
    print(f"   → Using raw source lines/context instead of chunked text")
    
    return pairs

def _llm_judge_drift(summary: str, source: str) -> Tuple[bool, float, str]:
    """
    Use LLM to judge if there's semantic drift between summary and source.
    
    Returns:
        - is_drift: bool (True if drift detected)
        - confidence: float (0.0-1.0, LLM's confidence in the judgment)
        - reasoning: str (LLM's explanation)
    """
    request = {
        "messages": [
            {
                "role": "user",
                "content": f"""You are an expert legal document reviewer. Your task is to determine if a summary sentence accurately represents the source text without introducing factual errors, omissions, or misleading interpretations.

SOURCE TEXT:
{source}

SUMMARY SENTENCE:
{summary}

Evaluate whether the summary sentence:
1. Accurately represents the key facts and requirements from the source
2. Does not add information not present in the source
3. Does not omit critical information that would mislead readers
4. Maintains the same legal meaning and intent

Examples of drift:
- Adding requirements not in source ("must" vs "may")
- Omitting critical exceptions or conditions
- Changing numerical values or timeframes
- Misrepresenting who has authority or responsibility
- Adding or removing penalties/consequences

Examples of acceptable paraphrasing:
- Restating in clearer language
- Reorganizing information for better flow
- Using synonyms for legal terms
- Condensing while preserving all key points"""
            }
        ],
        "system_prompt": "You are an expert legal document reviewer. Analyze the summary sentence against the source text and provide a structured judgment about semantic drift."
    }

    try:
        judgment = ai.extract(request, DriftJudgment)
        return (
            judgment.is_drift,
            judgment.confidence,
            judgment.reasoning
        )
    except Exception as e:
        # Fallback on error
        return (False, 0.5, f"LLM evaluation failed: {str(e)}")

# Accept `[C-foo_bar]` or `[foo_bar]` **inside square brackets only**
_CIT_RE = re.compile(r"(?:\[|【)C-([^】\]]+)(?:\]|】)", re.IGNORECASE)

# Helper: fuzzy resolve citation IDs by normalising zero‑padding and dash/underscore
def _fuzzy_resolve(cid: str, id_map: dict[str, int]) -> int | None:
    """
    Return the embedding‑index for `cid`, tolerating:
        • zero‑padding differences  (page 02  → 2)
        • dash ↔ underscore swaps   (file-name → file_name)
    """
    alt = cid.replace("-", "_")
    # Drop leading zeros within each numeric segment
    alt = "_".join(re.sub(r"^0+(\d)", r"\1", part) for part in alt.split("_"))
    # Further normalise: drop any remaining non‑alphanumeric chars, squeeze repeats, trim
    alt = re.sub(r"[^a-z0-9]+", "_", alt)
    alt = re.sub(r"_+", "_", alt).strip("_")
    return id_map.get(alt)

def make_flags(
    sentences,
    id_to_idx: dict[str, int],
    chunk_vecs: np.ndarray,
    threshold: float = 0.85,
    use_llm_judge: bool = False,
) -> list:
    """
    Generate drift flags using either vector similarity or LLM judge.
    
    If sentences is a list of (summary, source) pairs (JSON input):
        - Compare summary to raw source text directly (if source is not None)
        - Uses original source lines/context from JSON, not chunked text
        - If source is None, flag as 0.0 similarity
    If sentences is a list of strings (other formats):
        - Use citation tag logic as before
    """
    flags = []
    
    if sentences and isinstance(sentences[0], tuple):
        # JSON input: (summary, source) pairs
        for summary, source in sentences:
            if not source:
                flags.append((0.0, summary, [], "No source text available"))
                continue
                
            if use_llm_judge:
                # Use LLM judge for semantic drift detection with raw source
                is_drift, confidence, reasoning = _llm_judge_drift(summary, source)
                if is_drift:
                    # Convert confidence to similarity score (inverse relationship)
                    sim_score = 1.0 - confidence
                    flags.append((round(sim_score, 4), summary, [source], reasoning))
            else:
                # Use vector similarity with raw source text
                s_vec = ai.embed(summary)
                src_vec = ai.embed(source)
                sim = 1 - cosine(s_vec, src_vec)
                if sim < threshold:
                    flags.append((round(float(sim), 4), summary, [source], "Vector similarity below threshold (raw source)"))
        return flags
    
    # Else: legacy string input
    for s in sentences:
        src_ids = [x.lower() for x in _CIT_RE.findall(s)]
        idxs = []
        for cid in src_ids:
            if cid in id_to_idx:
                idxs.append(id_to_idx[cid])
            else:
                alt_idx = _fuzzy_resolve(cid, id_to_idx)
                if alt_idx is not None:
                    idxs.append(alt_idx)
        if not src_ids:
            flags.append((0.0, s, [], "No citation tags found"))
            continue
        if not idxs:
            flags.append((0.0, s, src_ids, "Citation tags not found in chunks"))
            continue
            
        if use_llm_judge:
            # For legacy input, we can't use LLM judge since we don't have direct source text
            # Fall back to vector similarity
            s_vec = ai.embed(s)
            worst = min(1 - cosine(s_vec, chunk_vecs[i]) for i in idxs)
            if worst < threshold:
                flags.append((round(float(worst), 4), s, src_ids, "Vector similarity below threshold (LLM judge not available for citation-based input)"))
        else:
            # Use vector similarity
            s_vec = ai.embed(s)
            worst = min(1 - cosine(s_vec, chunk_vecs[i]) for i in idxs)
            if worst < threshold:
                flags.append((round(float(worst), 4), s, src_ids, "Vector similarity below threshold"))
    
    return flags

# ---------------------------------------------------------------------------#
#                     Helper – expand directories to files                   #
# ---------------------------------------------------------------------------#
def collect_files(paths: Iterable[Path]) -> List[Path]:
    """Return a flat list of PDF/DOCX files.

    Any path that is a *directory* is searched recursively for ``*.pdf``,
    ``*.docx`` and ``*.doc`` files.  Non‑matching files are ignored.
    """
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            for ext in ("*.pdf", "*.docx", "*.doc"):
                out.extend(p.rglob(ext))
        else:
            out.append(p)
    return out

# ---------------------------------------------------------------------------#
#                                   CLI                                      #
# ---------------------------------------------------------------------------#
def main() -> None:
    p = argparse.ArgumentParser(description="Pre‑process source docs → embeddings.")
    p.add_argument("files", nargs="+", type=Path, help="PDF/DOCX file(s) *or* directory/ies containing them")
    p.add_argument("--draft", type=Path, help="Path to draft file (PDF, DOCX, or MD) for drift flagging")
    p.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory")
    p.add_argument("--chunk-tokens", type=int, default=400, help="Tokens per chunk")
    p.add_argument("--threshold", type=float, default=0.2, help="Cosine similarity threshold")
    p.add_argument("--use-llm-judge", action="store_true", help="Use LLM judge for semantic drift detection (recommended for JSON drafts)")
    args = p.parse_args()

    # Expand any directories into actual file paths
    file_paths = collect_files(args.files)
    if not file_paths:
        sys.exit("❌ No PDF/DOCX files found in the provided path(s).")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📝 Extracting text …")
    _ = extract_text(file_paths)  # We don't need the combined text anymore

    print("✂️  Chunking text …")
    all_chunks: List[str] = []
    id_to_idx: dict[str, int] = {}

    for fp in file_paths:
        for cid, ctext in file_chunks(fp, max_tokens=args.chunk_tokens):
            id_to_idx[cid.lower()] = len(all_chunks)
            all_chunks.append(ctext)

    (out_dir / "chunks.json").write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2))
    (out_dir / "id_to_idx.json").write_text(json.dumps(id_to_idx, indent=2))
    print(f"   → {len(all_chunks)} chunks")

    print("🔮 Generating embeddings …")
    chunk_vecs = ai.embed(all_chunks)  # shape (N, D)
    np.save(out_dir / "chunk_vecs.npy", chunk_vecs)
    print("   Embeddings saved.")

    # Optional drift‑flag generation
    if args.draft:
        print("🚦 Running drift detection …")
        
        # Check if this is a JSON draft
        is_json_draft = args.draft.suffix.lower() == ".json"
        if is_json_draft:
            print("   JSON draft detected - using raw source text from deep research output")
            print("   This preserves original context and structure without chunking")
        
        if args.use_llm_judge:
            print("   Using LLM judge for semantic drift detection")
        else:
            print("   Using vector similarity for drift detection")
            
        sentences = load_draft_sentences(args.draft)
        flags = make_flags(sentences, id_to_idx, chunk_vecs, threshold=args.threshold, use_llm_judge=args.use_llm_judge)
        flags_path = out_dir / "flags.json"
        flags_path.write_text(json.dumps(flags, ensure_ascii=False, indent=2))
        print(f"   {len(flags)} potential drift flags written to {flags_path}")

    print("✅ Done.")

if __name__ == "__main__":
    main()