# HOA Rule Summariser & Drift Checker

This project processes a set of HOA documents and generates a summary while flagging sentences whose embeddings diverge from their cited sources. The workflow revolves around creating embeddings for PDF documents, producing a draft summary and reviewing low-similarity sentences.

## Installation
1. Ensure **Python 3.11+** is available.
2. Install dependencies defined in `pyproject.toml`:
   ```bash
   pip install -e .
   ```
3. Set the required environment variables:
   - `OPENAI_API_KEY` – needed for embedding generation.
   - `GITHUB_TOKEN` and `GITHUB_REPO` (optional) – to create pull requests from the Streamlit app.

## Preprocessing Documents
Use `ingest.py` to extract text and build embeddings from the PDFs in `docs/`:
```bash
python ingest.py docs/ --out-dir data/
```
Providing the `--draft` option will also compute drift flags for the supplied markdown draft:
```bash
python ingest.py docs/ --draft draft.md --out-dir data/
```
The script produces `chunks.json`, `chunk_vecs.npy`, `id_to_idx.json` and, if a draft is provided, `flags.json` in the chosen output directory.

## Reviewing Flags
Launch the Streamlit interface to review and edit flagged sentences:
```bash
streamlit run main.py
```
The app displays each flagged sentence alongside its source text. Accepted edits can be committed back to GitHub as a pull request when `GITHUB_TOKEN` and `GITHUB_REPO` are configured.

## Regenerating Drafts
The `regenerate_all.sh` helper script rebuilds embeddings, launches the document MCP server and an ngrok tunnel, and generates a new draft using the research script. Review `draft_new.md` before replacing `draft.md`.

# GovAI Utilities

This repo contains helper scripts for document processing and AI-powered
analysis.  Embeddings retrieved from OpenAI are now cached on disk in
`data/embed_cache.json` to avoid recomputation when rerunning the tools.
