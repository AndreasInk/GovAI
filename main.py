# review_app.py
"""Streamlit UI to review drift flags and open PRs with edits."""
from __future__ import annotations

import os
import math
import json
import time
import io
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
from st_diff_viewer import diff_viewer
from github import Github, InputGitAuthor

from tiktoken import get_encoding           # just for token count display
from fpdf import FPDF


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

# Title first so page renders even if data is missing
st.title("📜 Plantation Governance Report Drift Checker")

# -----------------------------
# 📦 Data Setup (always visible)
# -----------------------------
st.sidebar.markdown("### Data Setup")
st.sidebar.caption("Drop PDFs or a ZIP; files are saved under `docs/` and `data/`.")
if DRIVE_URL:
    st.sidebar.markdown(f"[Open shared folder]({DRIVE_URL})")
uploaded = st.sidebar.file_uploader(
    "Upload PDFs/ZIP (docs or data)", type=["pdf", "zip"], accept_multiple_files=True
)

def _save_uploads(files: list) -> tuple[list[Path], list[Path]]:
    saved_docs: list[Path] = []
    saved_data: list[Path] = []
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    import zipfile
    for f in files:
        name = f.name
        if name.lower().endswith(".pdf"):
            out = docs_dir / name
            out.write_bytes(f.read())
            saved_docs.append(out)
        elif name.lower().endswith(".zip"):
            buf = io.BytesIO(f.read())
            with zipfile.ZipFile(buf) as zf:
                for zi in zf.infolist():
                    base = Path(zi.filename).name
                    # Docs
                    if base.lower().endswith(".pdf"):
                        target = docs_dir / base
                        with zf.open(zi) as src:
                            target.write_bytes(src.read())
                        saved_docs.append(target)
                    # Data artefacts
                    elif base in {"chunks.json", "id_to_idx.json", "flags.json", "chunk_vecs.npy"}:
                        target = data_dir / base
                        with zf.open(zi) as src:
                            target.write_bytes(src.read())
                        saved_data.append(target)
                    # Draft files – save to repo root for app defaults
                    elif base in {"draft.json", "draft.md"}:
                        target = Path(base)
                        with zf.open(zi) as src:
                            target.write_bytes(src.read())
    return saved_docs, saved_data

if uploaded:
    docs_saved, data_saved = _save_uploads(uploaded)
    if docs_saved:
        st.sidebar.success(f"Saved {len(docs_saved)} doc(s) to `docs/`.")
    if data_saved:
        st.sidebar.success(f"Saved {len(data_saved)} data file(s) to `data/`.")
    if not docs_saved and not data_saved:
        st.sidebar.info("No PDFs or data files found in uploads.")

# No build or flag-generation buttons; uploading data ZIPs is sufficient

# ------------------------------------------------------------------
# ⬇️  Load pre-computed artefacts  (produced by the notebook prototype)
# ------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

# Validate required artefacts exist; allow page to render if missing so users can upload data
required_files = [
    DATA_DIR / "chunks.json",
    DATA_DIR / "chunk_vecs.npy",
    DATA_DIR / "flags.json",
]
missing = [p.name for p in required_files if not p.exists()]
if missing:
    st.warning(
        "Missing data: " + ", ".join(missing) +
        ". Use the sidebar 'Data Setup' to upload a ZIP with data files."
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

# -----------------------------
# 🧩 Cohesive report helpers
# -----------------------------
# Build reverse mapping for chunk IDs where possible
IDX_TO_ID: dict[int, str] = {}
if id_to_idx:
    try:
        IDX_TO_ID = {v: k for k, v in id_to_idx.items()}
    except Exception:
        IDX_TO_ID = {}


def _to_latin1(text: str) -> str:
    """Simplify punctuation and strip non latin-1 for PDF output."""
    if not isinstance(text, str):
        return text
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": '-', "\u2014": '-', "\u2026": '...', "\u2012": '-',
        "\u2010": '-', "\u2011": '-', "\u00a0": ' ',
    }
    for uni, ascii_ in replacements.items():
        text = text.replace(uni, ascii_)
    try:
        import unicodedata as _ud
        return _ud.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1')
    except Exception:
        return text


def _clean_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s[-1] not in '.!?':
        s = s + '.'
    s = s.replace(' ,', ',').replace(' .', '.')
    return s


def build_cohesive_sections() -> list[tuple[str, list[str]]]:
    """Build a simple, cohesive narrative grouped by document.

    Returns: list of (document_title, paragraphs)
    """
    entries: list[tuple[str, int, int, str]] = []  # (doc, page, chunk, text)
    seen = set()
    for flag in st.session_state.flags:
        if len(flag) == 3:
            _sim, sent, ids = flag
        else:
            _sim, sent, ids, _reason = flag
        text = st.session_state.get('edits', {}).get(sent, sent) if isinstance(sent, str) else str(sent)
        text = _clean_sentence(text)
        doc_name = 'General'
        page = 0
        chunk_no = 0
        if isinstance(ids, (list, tuple)) and ids:
            for cid in ids:
                cid_str = None
                if isinstance(cid, str):
                    cid_str = cid
                elif isinstance(cid, int) and cid in IDX_TO_ID:
                    cid_str = IDX_TO_ID[cid]
                if cid_str:
                    info = parse_chunk_id(cid_str)
                    doc_name = info.get('document') or doc_name
                    try:
                        page = int(info.get('page') or 0)
                        chunk_no = int(info.get('chunk') or 0)
                    except Exception:
                        pass
                    break
        key = (doc_name, page, chunk_no, text)
        if key in seen:
            continue
        seen.add(key)
        entries.append((doc_name, page, chunk_no, text))

    from collections import defaultdict
    groups: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for doc, page, chunk_no, text in entries:
        groups[doc].append((page, chunk_no, text))
    for doc in groups:
        groups[doc].sort(key=lambda x: (x[0], x[1]))

    result: list[tuple[str, list[str]]] = []
    for doc in sorted(groups.keys(), key=lambda k: (k != 'General', str(k).lower())):
        paras: list[str] = []
        cur_page = None
        buf: list[str] = []
        for page, _chunk, text in groups[doc]:
            if cur_page is None:
                cur_page = page
            if page != cur_page and buf:
                paras.append(' '.join(buf))
                buf = []
                cur_page = page
            buf.append(text)
        if buf:
            paras.append(' '.join(buf))
        # Deduplicate simple
        uniq: list[str] = []
        seen_p = set()
        for p in paras:
            key = p.strip().lower()
            if key in seen_p:
                continue
            seen_p.add(key)
            uniq.append(p)
        title = doc.replace('_', ' ').strip().title() if doc else 'General'
        result.append((title, uniq))
    return result


def generate_pdf_bytes(sections: list[tuple[str, list[str]]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(20)
    pdf.multi_cell(0, 12, _to_latin1("Plantation Governance Report"), align="C")
    pdf.set_font("Helvetica", "", 11)
    from datetime import datetime
    pdf.ln(4)
    pdf.multi_cell(0, 8, datetime.now().strftime("Generated on %B %d, %Y"), align="C")
    # Sections
    for title, paras in sections:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, _to_latin1(title))
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 12)
        for p in paras:
            pdf.multi_cell(0, 8, _to_latin1(p))
            pdf.ln(2)
    return pdf.output(dest='S').encode('latin-1')

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


st.sidebar.markdown(f"**{len(st.session_state.flags)} flags** loaded · Source chunks: **{len(chunks)}**")

# Global "edited draft" buffer (one long string)
if "draft_buffer" not in st.session_state:
    default_md: str | None = None
    for cand in (Path("draft.md"), Path("docs+data") / "draft.md"):
        try:
            if cand.exists():
                default_md = cand.read_text(encoding="utf-8", errors="ignore")
                break
        except Exception:
            pass
    st.session_state.draft_buffer = default_md or ""

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

# (Re-flagging helper removed; this UI reviews precomputed flags only.)

 

 

tab_review, tab_chat = st.tabs(["Review Flags", "Search Docs"])

with tab_review:
    st.header("🚦 Review Drift Flags")

    # Show flags at or below this similarity (default aligns with detection threshold)
    max_sim_flags = st.slider(
        "Max similarity to show (lower = worse match)", 0.0, 1.0, 0.85, key="flag_sim"
    )
    filter_text_flags = st.text_input("Filter flags by text…", key="flag_text")

    # Filter flags
    flag_entries = [
        f for f in st.session_state.flags
        if f[0] <= max_sim_flags and filter_text_flags.lower() in f[1].lower()
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

with tab_chat:
    st.header("🔎 Search the Documents")
    st.caption("Ask a question to find relevant rules and passages across the uploaded documents.")

    # Lazy-precompute normalized embeddings for semantic search
    if "_emb_norm" not in st.session_state:
        try:
            emb = embeddings.astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
            st.session_state._emb_norm = emb / norms
        except Exception:
            st.session_state._emb_norm = None

    # Keyword index + IDF for fallback/hybrid search (built once)
    if "_kw_index" not in st.session_state:
        from collections import defaultdict as _dd
        idx = _dd(set)
        for i, text in enumerate(chunks):
            tokens = "".join(c if c.isalnum() else " " for c in text.lower()).split()
            for w in set(tokens):
                if w:
                    idx[w].add(i)
        # Compute IDF
        N = float(len(chunks)) if chunks else 1.0
        idf = {w: math.log((N + 1.0) / (len(posts) + 1.0)) + 1.0 for w, posts in idx.items()}
        st.session_state._kw_index = idx
        st.session_state._kw_idf = idf

    # Display prior chat
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])

    use_ctx = st.checkbox("Use chat history for context", value=True, key="chat_use_ctx")
    alpha = st.slider("Semantic weight", 0.0, 1.0, 0.7, 0.05, key="chat_alpha")

    user_query = st.chat_input("Ask about HOA rules, e.g., 'parking overnight' or 'fines'.")
    if user_query:
        st.session_state.chat.append({"role": "user", "content": user_query})

        # Build contextualized query text
        ctx_text = ""
        if use_ctx and st.session_state.chat:
            # Include up to last 4 turns (user/assistant) excluding the just-appended user message
            history = st.session_state.chat[:-1][-4:]
            bits = []
            for m in history:
                role = m.get("role", "user")
                content = str(m.get("content", "")).strip()
                if content:
                    bits.append(f"{role}: {content}")
            if bits:
                ctx_text = "\n".join(bits)
        query_text = user_query if not ctx_text else f"{user_query}\n\nContext:\n{ctx_text}"

        # Try semantic search via OpenAI embeddings if available
        q_vec = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                import importlib
                ai_mod = importlib.import_module("ai")
                q_vec = ai_mod.embed(query_text)  # shape (D,)
                # normalize
                q_vec = (q_vec / (np.linalg.norm(q_vec) + 1e-8)).astype(np.float32)
            except Exception:
                q_vec = None

        hits: list[tuple[int, float]] = []  # (idx, score)
        # Hybrid ranking: combine cosine similarity (if available) with TF‑IDF
        terms = [t for t in "".join(c if c.isalnum() else " " for c in query_text.lower()).split() if t]
        candidates = set()
        for t in terms:
            candidates.update(st.session_state._kw_index.get(t, set()))

        tfidf_scores: dict[int, float] = {}
        idf_map = st.session_state.get("_kw_idf", {})
        for i in candidates:
            txt_tokens = "".join(c if c.isalnum() else " " for c in chunks[i].lower()).split()
            if not txt_tokens:
                continue
            # simple term frequency
            score = 0.0
            for t in terms:
                if not t:
                    continue
                tf = txt_tokens.count(t)
                if tf:
                    score += tf * float(idf_map.get(t, 1.0))
            if score:
                tfidf_scores[int(i)] = score
        # Normalize TF‑IDF to 0..1
        if tfidf_scores:
            max_tfidf = max(tfidf_scores.values()) or 1.0
            for k in list(tfidf_scores.keys()):
                tfidf_scores[k] = tfidf_scores[k] / max_tfidf

        if q_vec is not None and st.session_state._emb_norm is not None:
            sims = st.session_state._emb_norm @ q_vec  # cosine similarity
            # If we have candidates from keywords, restrict to their union; else use all
            if candidates:
                indices = np.fromiter((int(i) for i in candidates), dtype=int)
            else:
                indices = np.arange(sims.shape[0])
            combined = []
            for i in indices:
                cos = float(sims[int(i)])
                kw = tfidf_scores.get(int(i), 0.0)
                combined.append((int(i), alpha * cos + (1.0 - alpha) * kw))
            combined.sort(key=lambda x: x[1], reverse=True)
            hits = combined[:10]
        else:
            # Keyword‑only ranking
            hits = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # Compose assistant reply with top matches
        if not hits:
            answer = "I couldn't find a relevant passage. Try rephrasing with different keywords."
            st.session_state.chat.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        else:
            lines = ["Here are the most relevant passages:"]
            for rank, (i, score) in enumerate(hits, 1):
                cid = IDX_TO_ID.get(i, f"idx:{i}")
                meta = parse_chunk_id(cid) if isinstance(cid, str) else {"document": "", "page": ""}
                doc = meta.get("document", "").replace("_", " ") or "Document"
                page = meta.get("page", "?")
                snippet = chunks[i].strip().replace("\n", " ")
                if len(snippet) > 320:
                    snippet = snippet[:320].rstrip() + "…"
                score_str = f"{score:.2f} (hybrid)" if q_vec is not None else f"{score:.2f} (kw)"
                lines.append(f"{rank}. {doc} (p.{page}) – {snippet} [id: {cid}]  • score {score_str}")

            answer = "\n".join(lines)
            st.session_state.chat.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

# ------------------------------------------------------------------
# 📄  Cohesive Report Export
# ------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.markdown("### Export Cohesive PDF")
if st.sidebar.button("Download cohesive report PDF", use_container_width=True):
    try:
        sections = build_cohesive_sections()
        if not sections:
            st.sidebar.warning("Nothing to export yet.")
        else:
            pdf_bytes = generate_pdf_bytes(sections)
            st.sidebar.download_button(
                label="Download PDF",
                data=io.BytesIO(pdf_bytes),
                file_name="governance_report.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        st.sidebar.error(f"Failed to build PDF: {e}")
