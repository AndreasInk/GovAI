#!/usr/bin/env python3
"""
ingest.py – Pre‑processing pipeline for HOA rewrite project
===========================================================

Usage
-----
$ python ingest.py docs/ --out-dir data/
# (optional) add a draft (PDF/DOCX/MD) to compute drift flags immediately
$ python ingest.py docs/ --draft draft.md --out-dir data/

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
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from tiktoken import get_encoding

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

def load_draft_sentences(draft_path: Path) -> List[str]:
    """Extract sentences from the *draft* file with extra cleanup.

    – Collapses all runs of whitespace (incl. newlines) to single spaces
      so hard line‑breaks in PDFs don’t create orphaned fragments.
    – Splits on punctuation (.!? ) via `_SENT_SPLIT_RE`.
    – Fuses any leftover pieces shorter than 10 words into the
      preceding sentence to avoid 1–2‑word fragments.
    """
    suffix = draft_path.suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf(draft_path)
    elif suffix in {".docx", ".doc"}:
        text = _extract_docx(draft_path)
    else:
        # Fallback: treat as text/markdown – decode with utf‑8
        text = draft_path.read_text(encoding="utf-8", errors="ignore")

    # Normalise whitespace so line‑breaks don’t split sentences
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
    sentences: List[str],
    id_to_idx: dict[str, int],
    chunk_vecs: np.ndarray,
    threshold: float = 0.85,
) -> List[Tuple[float, str, List[str]]]:
    """
    Return a list of drift flags.

    • If a sentence has **no citation tags** → similarity is set to 0.0 and it is flagged.
    • If citation tags are present but **none resolve** to a chunk → similarity 0.0 and flagged.
    • If at least one valid citation resolves, use cosine‑similarity and flag when below `threshold`.
    """
    flags: List[Tuple[float, str, List[str]]] = []

    for s in sentences:
        # Collect citation IDs (case‑insensitive)
        src_ids = [x.lower() for x in _CIT_RE.findall(s)]
        idxs = []
        for cid in src_ids:
            if cid in id_to_idx:
                idxs.append(id_to_idx[cid])
            else:
                alt_idx = _fuzzy_resolve(cid, id_to_idx)
                if alt_idx is not None:
                    idxs.append(alt_idx)

        # Case 1: No citations at all
        if not src_ids:
            flags.append((0.0, s, []))
            continue

        # Case 2: Citation(s) present but none matched the corpus
        if not idxs:
            flags.append((0.0, s, src_ids))
            continue

        # Case 3: Normal similarity check
        s_vec = ai.embed(s)  # (D,)
        worst = min(1 - cosine(s_vec, chunk_vecs[i]) for i in idxs)
        if worst < threshold:
            flags.append((round(float(worst), 4), s, src_ids))

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
    p.add_argument("--threshold", type=float, default=0.4, help="Cosine similarity threshold")
    args = p.parse_args()

    # Expand any directories into actual file paths
    file_paths = collect_files(args.files)
    if not file_paths:
        sys.exit("❌ No PDF/DOCX files found in the provided path(s).")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📝 Extracting text …")
    _ = extract_text(file_paths)  # We don’t need the combined text anymore

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
        sentences = load_draft_sentences(args.draft)
        flags = make_flags(sentences, id_to_idx, chunk_vecs, threshold=args.threshold)
        flags_path = out_dir / "flags.json"
        flags_path.write_text(json.dumps(flags, ensure_ascii=False, indent=2))
        print(f"   {len(flags)} potential drift flags written to {flags_path}")

    print("✅ Done.")

if __name__ == "__main__":
    main()