# HOA Rule Summarizer & Drift Checker

This project processes a set of HOA documents and generates a consolidated summary while flagging sentences whose content diverges from their cited sources. It includes:

- **Preprocessing**: chunk + embed PDFs/DOCX from `docs/` with `ingest.py`
- **Deep research**: optional MCP-powered content generation via `research-with-mcp.py`
- **Review UI**: Streamlit app (`main.py`) to review flags, edit summaries, and open PRs
- **PDF export**: Generate a branded PDF from a JSON draft

## Installation
1. Ensure **Python 3.10+** is available.
2. Install dependencies defined in `pyproject.toml`:
   ```bash
   pip install -e .
   ```
3. Copy `.env.example` to `.env` and set required variables:
   - `OPENAI_API_KEY` – required for embeddings and LLM features
   - `GITHUB_TOKEN` and `GITHUB_REPO` (optional) – to create pull requests from the Streamlit app

## Preprocessing Documents
Place your source PDFs/DOCX in `docs/` (this folder is git-ignored). Use `ingest.py` to extract text and build embeddings:
```bash
python ingest.py docs/ --out-dir data/
```
Providing the `--draft` option will also compute drift flags for the supplied markdown draft:
```bash
python ingest.py docs/ --draft draft.md --out-dir data/
```
The script produces `chunks.json`, `chunk_vecs.npy`, `id_to_idx.json` and, if a draft is provided, `flags.json` in the chosen output directory.

Notes:
- Default drift threshold is 0.85 (cosine similarity). Override with `--threshold`.
- JSON drafts (from deep research) are supported and preserve original source text. You can enable an **LLM judge** mode with `--use-llm-judge`.

## Reviewing Flags
Launch the Streamlit interface to review and edit flagged sentences:
```bash
streamlit run main.py
```
The app displays each flagged sentence alongside its source text. Accepted edits can be committed back to GitHub as a pull request when `GITHUB_TOKEN` and `GITHUB_REPO` are configured.

If you want to run with a fresh draft JSON, generate it first with the research workflow, then run `ingest.py` with `--draft` to create `flags.json`.

## Drag-and-drop setup (no CLI)
If you don't want to run the CLI, you can use the Streamlit app to provision data:

1) Start the app and sign in (see auth in this README)
2) Use the sidebar "Data Setup" panel to drag-and-drop your `docs/` PDFs or a ZIP containing PDFs
3) Click "Build embeddings" to generate `data/chunks.json`, `data/chunk_vecs.npy`, and `data/id_to_idx.json`
4) Optionally upload a `draft.json`/`draft.md` to generate `data/flags.json`

## Regenerating Drafts
The `regenerate_all.sh` helper script rebuilds embeddings, launches the document MCP server and an ngrok tunnel, and generates a new draft using the research script. Review `draft_new.json` before replacing `draft.json`.

## Environment

Create a `.env` file or export the following variables:

```
OPENAI_API_KEY=...
GITHUB_TOKEN=...
GITHUB_REPO=org/repo
HOA_MCP_URL=https://<your-ngrok>.ngrok-free.app/mcp
```

Alternatively, place your OpenAI key in `~/.openai_key` for the shell scripts.

## Repository Structure

- `ingest.py`: preprocess PDFs/DOCX → chunks, embeddings, optional drift flags
- `main.py`: Streamlit review UI (edit, diff, PR)
- `doc-mcp.py`: minimal MCP server for PDF retrieval (`search`, `fetch`)
- `research-with-mcp.py`: driver to create deep-research jobs via MCP
- `ai.py`: OpenAI helper layer (embeddings, responses, simple cache)
- `data/`: generated artefacts (`chunks.json`, `chunk_vecs.npy`, `id_to_idx.json`, `flags.json`)
- `docs/`: input PDFs/DOCX

## Development

- Format/lint: project uses Ruff defaults (line length 120).
- Test the LLM judge prompt quickly with:
  ```bash
  python test_llm_judge.py
  ```

# GovAI Utilities

This repo contains helper scripts for document processing and AI-powered
analysis.  Embeddings retrieved from OpenAI are now cached on disk in
`data/embed_cache.json` to avoid recomputation when rerunning the tools.
