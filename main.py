# review_app.py
"""
⚖️  HOA Drift-Checker – Streamlit UI
-----------------------------------
Flags low-similarity summary sentences, lets the board edit them,
and commits accepted edits back to GitHub as a pull request.

Quick-start
-----------
1.  pip install streamlit PyGithub streamlit-diff-viewer openai tiktoken
2.  export OPENAI_API_KEY=…
    export GITHUB_TOKEN=ghp_xxx
    export GITHUB_REPO=username/bylaws-rewrite   #   org/repo
3.  Have three companion files in the same folder:
    • chunks.json        – list[str]   chunk_id → source text
    • chunk_vecs.npy     – NumPy array of embeddings (same order)
    • flags.json         – list[tuple(similarity, sentence, [ids], reasoning)]
"""
from __future__ import annotations

import os
import json
import time
import re
from pathlib import Path
from typing import List, Tuple, Dict

from scipy.spatial.distance import cosine

import numpy as np
import streamlit as st
from st_diff_viewer import diff_viewer
from github import Github, InputGitAuthor

import ai  # your helper wrapper
from tiktoken import get_encoding           # just for token count display

# PDF generation
from fpdf import FPDF

# ------------------- Custom PDF class with branded header and footer -------------------
class PDFReport(FPDF):
    """Custom PDF class with branded header and footer."""
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Plantation Governance Report", ln=1, align="C")
        self.set_draw_color(100, 100, 100)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.y, self.w - self.r_margin, self.y)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.1)
        self.line(self.l_margin, self.y, self.w - self.r_margin, self.y)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
import io
import unicodedata

# -------------------------
# 🌐  Streamlit page config
# -------------------------
st.set_page_config(page_title="Plantation Governance Report Drift Checker", page_icon="📜", layout="wide")

# ---------------------------------
# 🔒  Simple API key based auth gate
# ---------------------------------
def _get_allowed_keys() -> set[str]:
    keys: set[str] = set()
    # Single or comma-separated env vars
    k1 = os.getenv("GOVAI_APP_KEY", "").strip()
    if k1:
        keys.add(k1)
    klist = os.getenv("GOVAI_APP_KEYS", "").strip()
    if klist:
        for item in klist.split(","):
            item = item.strip()
            if item:
                keys.add(item)
    # Streamlit secrets support
    try:
        if "auth_key" in st.secrets:
            v = str(st.secrets["auth_key"]).strip()
            if v:
                keys.add(v)
        if "auth_keys" in st.secrets:
            for v in st.secrets["auth_keys"]:
                v = str(v).strip()
                if v:
                    keys.add(v)
    except Exception:
        pass
    return keys


def require_auth() -> None:
    # Global toggle: if not explicitly enabled, auth is bypassed
    require_flag = os.getenv("GOVAI_REQUIRE_AUTH", "0").strip().lower() in {"1", "true", "yes"}
    if not require_flag:
        return

    allowed = _get_allowed_keys()
    if not allowed:
        # No keys configured → open access
        return

    # Query param auto-login support (?key=...)
    try:
        qp_key = None
        qp = st.query_params
        if isinstance(qp, dict):
            qp_key = qp.get("key")
        if qp_key and qp_key in allowed:
            st.session_state["auth_ok"] = True
    except Exception:
        pass

    if st.session_state.get("auth_ok"):
        return

    st.title("🔒 Access required")
    st.caption("Enter the access key to use this app.")
    with st.form("auth_form", clear_on_submit=False):
        key_input = st.text_input("Access key", type="password")
        submit = st.form_submit_button("Sign in")
    if submit:
        if key_input in allowed:
            st.session_state["auth_ok"] = True
            st.experimental_rerun()
        else:
            st.error("Invalid access key. Please try again.")
    st.stop()


# Enforce auth before loading any data
require_auth()

# Optional shared folder link (e.g., Google Drive) for docs/data
DRIVE_URL = os.getenv(
    "GOVAI_DRIVE_URL",
    "https://drive.google.com/drive/folders/1aEFZOXLcd0H1O6EdBpaiA3MWRE6UVJf1?usp=drive_link",
)

# ------------------------------------------------------------------
# ⬇️  Load pre-computed artefacts  (produced by the notebook prototype)
# ------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

# Validate required artefacts exist
required_files = [
    DATA_DIR / "chunks.json",
    DATA_DIR / "chunk_vecs.npy",
    DATA_DIR / "flags.json",
]
missing = [p.name for p in required_files if not p.exists()]
if missing:
    st.error(
        "Missing required data files: " + ", ".join(missing) +
        ". Run: `python ingest.py docs/ --out-dir data/` and optionally `--draft draft.json` to generate flags."
    )
    st.stop()

chunks: List[str] = json.loads((DATA_DIR / "chunks.json").read_text())
embeddings = np.load(DATA_DIR / "chunk_vecs.npy")
flags: List[Tuple[float, str, List[int], str]] = json.loads((DATA_DIR / "flags.json").read_text())
# Store flags in session_state, sorted, for possible re-flagging
if "flags" not in st.session_state:
    flags.sort(key=lambda tup: tup[0])
    st.session_state.flags = flags

# Mapping from chunk_id (str) -> integer index in chunks list
ID2IDX_PATH = DATA_DIR / "id_to_idx.json"
id_to_idx: dict[str, int] = {}
if ID2IDX_PATH.exists():
    id_to_idx = json.loads(ID2IDX_PATH.read_text())
    # normalise keys to lower-case for robustness
    id_to_idx = {k.lower(): v for k, v in id_to_idx.items()}

def _cid_to_idx(cid: str | int) -> int | None:
    """Return integer index for a chunk ID (str or int)."""
    if isinstance(cid, int):
        return cid if 0 <= cid < len(chunks) else None
    return id_to_idx.get(str(cid).lower())

ENC = get_encoding("cl100k_base")

# ---------------------------------------------
# 🔐  GitHub client  (lazy-init on first commit)
# ---------------------------------------------
def _gh_client() -> Github:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        st.error("GITHUB_TOKEN env var missing – cannot push PRs.")
        st.stop()
    return Github(token)


# ----------------------------------------------------------
# 🛠️  Utility – open a PR with the edited draft.md content
# ----------------------------------------------------------
def create_or_update_pr(new_content: str, user_name: str = "HOA Reviewer") -> None:
    repo_full = os.getenv("GITHUB_REPO")
    if not repo_full:
        st.error("Set GITHUB_REPO (e.g. 'user/repo') to enable PRs.")
        return

    gh = _gh_client()
    repo = gh.get_repo(repo_full)

    base = repo.get_branch("main")
    branch_name = f"hoa-drift-fix/{int(time.time())}"
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)

    commit_message = "HOA summary edits via Streamlit reviewer"
    author = InputGitAuthor(user_name, f"{user_name.replace(' ','.').lower()}@example.com")
    # Ensure file exists in branch: if not, create it first
    try:
        current = repo.get_contents("draft.md", ref=branch_name)
        repo.update_file(
            path="draft.md",
            message=commit_message,
            content=new_content,
            sha=current.sha,
            branch=branch_name,
            author=author,
        )
    except Exception:
        repo.create_file(
            path="draft.md",
            message=commit_message,
            content=new_content,
            branch=branch_name,
            author=author,
        )
    pr = repo.create_pull(
        title="🏷️ HOA drift fixes",
        body="Auto-generated by Streamlit reviewer; please squash-merge.",
        head=branch_name,
        base="main",
    )
    st.success(f"✅ Pull Request created: {pr.html_url}")


st.title("📜 Plantation Governance Report Drift Checker")

st.sidebar.markdown(f"**{len(st.session_state.flags)} flags** loaded · Source chunks: **{len(chunks)}**")

# -----------------------------
# 📦 Data Setup (drag-and-drop)
# -----------------------------
with st.sidebar.expander("Data Setup", expanded=False):
    st.caption("Drop PDFs or a ZIP; then build embeddings. All files are saved under `docs/` and `data/`.")
    if DRIVE_URL:
        st.markdown(f"[Open shared folder]({DRIVE_URL})")
    uploaded = st.file_uploader("Upload PDFs or ZIP", type=["pdf", "zip"], accept_multiple_files=True)
    colA, colB = st.columns(2)
    with colA:
        build = st.button("Build embeddings", use_container_width=True)
    with colB:
        gen_flags = st.button("Generate flags from draft", use_container_width=True)

    def _save_uploads(files: list) -> list[Path]:
        saved: list[Path] = []
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        import zipfile, io
        for f in files:
            name = f.name
            if name.lower().endswith(".pdf"):
                out = docs_dir / name
                out.write_bytes(f.read())
                saved.append(out)
            elif name.lower().endswith(".zip"):
                buf = io.BytesIO(f.read())
                with zipfile.ZipFile(buf) as zf:
                    for zi in zf.infolist():
                        if zi.filename.lower().endswith(".pdf"):
                            target = docs_dir / Path(zi.filename).name
                            with zf.open(zi) as src:
                                target.write_bytes(src.read())
                            saved.append(target)
        return saved

    if uploaded:
        saved_paths = _save_uploads(uploaded)
        if saved_paths:
            st.success(f"Saved {len(saved_paths)} file(s) to `docs/`.")
        else:
            st.info("No PDFs found in uploads.")

    if build:
        import subprocess, sys
        st.info("Running ingestion to build embeddings…")
        try:
            proc = subprocess.run([sys.executable, "ingest.py", "docs/", "--out-dir", "data/"], capture_output=True, text=True)
            if proc.returncode == 0:
                st.success("Embeddings built. Reload the app to pick up new data.")
            else:
                st.error("Ingestion failed. See logs below.")
                st.code(proc.stdout + "\n" + proc.stderr)
        except Exception as e:
            st.error(f"Failed to run ingestion: {e}")

    if gen_flags:
        import subprocess, sys
        draft_opt = None
        # Prefer a local draft.json, fallback to draft.md if present
        if Path("draft.json").exists():
            draft_opt = "draft.json"
        elif Path("draft.md").exists():
            draft_opt = "draft.md"
        if not draft_opt:
            st.warning("No draft.json or draft.md found in project root.")
        else:
            st.info(f"Generating flags from {draft_opt}…")
            try:
                proc = subprocess.run([sys.executable, "ingest.py", "docs/", "--draft", draft_opt, "--out-dir", "data/"], capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("Flags generated. Reload the app to pick up new data.")
                else:
                    st.error("Flag generation failed. See logs below.")
                    st.code(proc.stdout + "\n" + proc.stderr)
            except Exception as e:
                st.error(f"Failed to run flag generation: {e}")

# Global "edited draft" buffer (one long string)
if "draft_buffer" not in st.session_state:
    st.session_state.draft_buffer = Path("draft.md").read_text()

# --------------------------------------------------------------------
# 📚  Chunk Browser Functions
# --------------------------------------------------------------------
def parse_chunk_id(chunk_id: str) -> Dict[str, str]:
    """Parse chunk ID to extract document, page, and chunk number."""
    # Format: "document_page_chunk" (e.g., "bylaws_5_3")
    parts = chunk_id.split('_')
    if len(parts) >= 3:
        # Reconstruct document name (might contain underscores)
        doc_name = '_'.join(parts[:-2])
        page_num = parts[-2]
        chunk_num = parts[-1]
        return {
            'document': doc_name,
            'page': page_num,
            'chunk': chunk_num,
            'full_id': chunk_id
        }
    return {
        'document': 'unknown',
        'page': '0',
        'chunk': '0',
        'full_id': chunk_id
    }

def get_document_list() -> List[str]:
    """Get list of unique document names from chunk IDs."""
    documents = set()
    for chunk_id in id_to_idx.keys():
        doc_info = parse_chunk_id(chunk_id)
        documents.add(doc_info['document'])
    return sorted(list(documents))

def search_chunks(query: str, limit: int = 50) -> List[Tuple[int, str, float]]:
    """Search chunks using semantic similarity."""
    if not query.strip():
        return []
    
    query_vec = ai.embed(query)
    similarities = []
    
    for i, chunk in enumerate(chunks):
        chunk_vec = embeddings[i]
        sim = 1 - cosine(query_vec, chunk_vec)
        similarities.append((i, chunk, sim))
    
    # Sort by similarity and return top results
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:limit]

def filter_chunks_by_document(document: str) -> List[Tuple[int, str, Dict[str, str]]]:
    """Filter chunks by document name."""
    filtered = []
    for chunk_id, idx in id_to_idx.items():
        doc_info = parse_chunk_id(chunk_id)
        if doc_info['document'] == document:
            filtered.append((idx, chunks[idx], doc_info))
    
    # Sort by page, then by chunk number
    filtered.sort(key=lambda x: (int(x[2]['page']), int(x[2]['chunk'])))
    return filtered

def filter_chunks_by_page(document: str, page: str) -> List[Tuple[int, str, Dict[str, str]]]:
    """Filter chunks by document and page."""
    filtered = []
    for chunk_id, idx in id_to_idx.items():
        doc_info = parse_chunk_id(chunk_id)
        if doc_info['document'] == document and doc_info['page'] == page:
            filtered.append((idx, chunks[idx], doc_info))
    
    # Sort by chunk number
    filtered.sort(key=lambda x: int(x[2]['chunk']))
    return filtered

# --------------------------------------------------------------------
# ---------- Re-flagging helper (uses cached chunk embeddings) ----------
# Accept IDs like `[C-123]`, `[C-Bylaws_5_3]`, or `[Bylaws_5_3]`
_CIT_RE = re.compile(r"(?:C-)?([\w\-]+)")

def _make_flags(draft_text: str, threshold: float = 0.85):
    """Return fresh flags list from *draft_text* (markdown)."""
    sentences = re.split(r"(?<=[.!?])\s+", draft_text)
    new_flags = []
    for s in sentences:
        src_ids = [x.lower() for x in _CIT_RE.findall(s)]
        if not src_ids:
            continue
        s_vec = ai.embed(s)  # (D,)
        idxs = [_cid_to_idx(cid) for cid in src_ids if _cid_to_idx(cid) is not None]
        if not idxs:
            continue  # no valid matching chunks
        worst = min(1 - cosine(s_vec, embeddings[i]) for i in idxs)
        if worst < threshold:
            new_flags.append((round(float(worst), 4), s, src_ids, "Vector similarity below threshold"))
    new_flags.sort(key=lambda tup: tup[0])
    return new_flags

def _to_latin1(text):
    """Replace common Unicode punctuation with ASCII equivalents and remove other non-latin-1 chars."""
    if not isinstance(text, str):
        return text
    # Replace curly quotes, dashes, ellipsis, etc.
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u2012': '-',
        '\u2010': '-', '\u2011': '-', '\u00a0': ' ',
    }
    for uni, ascii_ in replacements.items():
        text = text.replace(uni, ascii_)
    # Remove any remaining non-latin-1 chars
    return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1')

 

# --------------------------------------------------------------------
# 🚦  Review Drift Flags  (restored standalone viewer)
# --------------------------------------------------------------------
st.header("🚦 Review Drift Flags")

min_sim_flags = st.slider("Min similarity for flags", 0.0, 1.0, 0.0, key="flag_sim")
filter_text_flags = st.text_input("Filter flags by text…", key="flag_text")

# Filter flags
flag_entries = [
    f for f in st.session_state.flags
    if f[0] >= min_sim_flags and filter_text_flags.lower() in f[1].lower()
]

for idx, flag_data in enumerate(flag_entries, 1):
    # Support both 3‑tuple and 4‑tuple flag formats
    if len(flag_data) == 3:
        sim, sent, ids = flag_data
        reasoning = "No reasoning provided"
    else:
        sim, sent, ids, reasoning = flag_data

    with st.expander(f"({idx}/{len(flag_entries)}) Similarity {sim:.2f}  |  {sent[:80]}…"):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### ✏️ **Edit summary sentence**")
            edited = st.text_area(
                "Sentence", value=sent, key=f"edit-flag-{idx}", height=80, label_visibility="collapsed"
            )
            token_len = len(ENC.encode(edited))
            st.caption(f"{token_len} tokens")

            # Display LLM reasoning if available
            if reasoning and reasoning != "No reasoning provided":
                st.markdown("##### 🤖 **LLM Reasoning**")
                st.info(reasoning)

        with col2:
            st.markdown("##### 📖 **Source chunk(s)**")
            if len(ids) == 1 and isinstance(ids[0], str) and ids[0] not in id_to_idx:
                st.text_area("Source Text", value=ids[0], height=200, label_visibility="collapsed", disabled=True)
            elif not ids:
                st.warning("⚠️ No source chunk(s) found for this flag.")
            else:
                for cid in ids:
                    idx2 = _cid_to_idx(cid)
                    if idx2 is not None:
                        st.write(chunks[idx2])
                        st.divider()
                    else:
                        st.warning(f"⚠️ Source chunk '{cid}' not found.")

        # Diff viewer (only show if edited differs)
        if edited.strip() != sent.strip():
            st.markdown("##### 🔍 Diff")
            diff_viewer(sent, edited, lang="md")

        # Store edits
        if "edits" not in st.session_state:
            st.session_state.edits = {}
        st.session_state.edits[sent] = edited


# ------------------------------------------------------------------
# 📥  Download JSON Draft as PDF
# ------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.markdown("### Download JSON Draft as PDF")

# Try to load draft.json as default
default_draft_data = None
try:
    if Path("draft.json").exists():
        with open("draft.json", "r") as f:
            default_draft_data = json.load(f)
        st.sidebar.success("✅ Loaded draft.json as default")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not load draft.json: {e}")

json_draft_file = st.sidebar.file_uploader("Upload JSON draft", type=["json"], key="json-draft-upload")

# Use default data if no file is uploaded
if json_draft_file is not None:
    try:
        draft_data = json.load(json_draft_file)
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded file: {e}")
        draft_data = None
elif default_draft_data is not None:
    draft_data = default_draft_data
    st.sidebar.info("📄 Using draft.json as default")
else:
    draft_data = None
if draft_data is not None:
    pairs = []
    executive_summary = None
    # Check for executive_summary and sections (GovAI draft format)
    if isinstance(draft_data, dict) and "sections" in draft_data:
        executive_summary = draft_data.get("executive_summary")
        for section in draft_data["sections"]:
            summary = section.get("summary_text", "")
            source = section.get("source_text", "")
            doc = section.get("source_document", "")
            page = section.get("source_page", "")
            pairs.append({
                "summary": summary,
                "source": source,
                "document": doc,
                "page": page
            })
    else:
        # Fallback to previous logic
        if isinstance(draft_data, list):
            for item in draft_data:
                if isinstance(item, dict) and "summary" in item and "source_text" in item:
                    pairs.append({"summary": item["summary"], "source": item["source_text"], "document": "", "page": ""})
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs.append({"summary": item[0], "source": item[1], "document": "", "page": ""})
        elif isinstance(draft_data, dict) and "summary" in draft_data and "source_text" in draft_data:
            pairs.append({"summary": draft_data["summary"], "source": draft_data["source_text"], "document": "", "page": ""})
        else:
            st.sidebar.warning("Could not parse summary/source pairs from JSON.")
            pairs = []

    # Apply edits to JSON draft and offer download
    if pairs or executive_summary:
        # Apply edits to JSON draft and offer download
        if "edits" in st.session_state:
            # Update pairs with edited summaries
            for pair in pairs:
                original = pair["summary"]
                if original in st.session_state.edits:
                    pair["summary"] = st.session_state.edits[original]
        # Build revised JSON structure
        revised = {}
        if executive_summary is not None:
            revised["executive_summary"] = executive_summary
        revised["sections"] = [
            {
                "summary_text": p["summary"],
                "source_text": p["source"],
                "source_document": p["document"],
                "source_page": p["page"]
            }
            for p in pairs
        ]
        revised_json_str = json.dumps(revised, indent=2)
        st.sidebar.download_button(
            label="💾 Download revised draft.json",
            data=revised_json_str,
            file_name="draft.json",
            mime="application/json"
        )

    # ------------------------------------------------------------------
    # 📥  Download JSON Draft as PDF
    # ------------------------------------------------------------------
    try:
        # Generate PDF in memory using custom branded PDF class
        pdf = PDFReport()
        pdf.set_auto_page_break(auto=True, margin=15)

        from datetime import datetime

        # Cover page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 28)
        pdf.ln(45)  # vertical spacing
        pdf.multi_cell(0, 14, _to_latin1("Plantation Governance Report"), align="C")
        pdf.set_font("Helvetica", "", 14)
        pdf.ln(8)
        pdf.multi_cell(0, 10, datetime.now().strftime("Generated on %B %d, %Y"), align="C")
        pdf.add_page()

        # Optional Executive Summary
        if executive_summary:
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 12, _to_latin1("Executive Summary"))
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 12)
            pdf.multi_cell(0, 8, _to_latin1(executive_summary))

        # Main Sections
        for idx, pair in enumerate(pairs, 1):
            # Automatic page break to avoid overflow
            if pdf.get_y() > pdf.h - pdf.b_margin - 20:
                pdf.add_page()

            # Separator between sections
            if idx > 1:
                pdf.set_draw_color(200, 200, 200)
                pdf.set_line_width(0.2)
                y = pdf.get_y()
                pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
                pdf.ln(4)

            # Section heading with context
            pdf.set_font("Helvetica", "B", 16)
            header_text = f"Section {idx}"
            if pair["document"] or pair["page"]:
                header_text += f" – {pair['document']} (Page {pair['page']})"
            pdf.multi_cell(0, 12, _to_latin1(header_text))
            pdf.ln(4)

            # Document context
            if pair["document"] or pair["page"]:
                pdf.set_font("Helvetica", "I", 11)
                context_line = " · ".join(filter(None, [pair["document"], f"Page {pair['page']}"]))
                pdf.multi_cell(0, 8, _to_latin1(context_line))
                pdf.ln(1)

            # Summary
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 9, _to_latin1("Key Insights"))
            pdf.set_font("Helvetica", "", 12)
            pdf.multi_cell(0, 8, _to_latin1(pair["summary"]))
            pdf.ln(1)

            # Source
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 9, _to_latin1("Source Excerpt"))
            pdf.set_font("Helvetica", "I", 11)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 8, _to_latin1(pair["source"]))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_buffer = io.BytesIO(pdf_bytes)
        st.sidebar.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="draft_summaries.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.sidebar.error(f"Failed to generate PDF: {e}")
    if not (pairs or executive_summary):
        st.sidebar.info("No summary/source pairs found in JSON.")