### Agents and Conventions for GovAI

This repository holds a prototype application that summarises HOA documents and flags sentences whose embeddings differ from their cited source.

Agent integration points:

- `doc-mcp.py`: Minimal MCP server exposing `search` and `fetch` over the `docs/` corpus using stable chunk IDs `<file_id>_<page_no>_<chunk_idx>`.
- `research-with-mcp.py`: Orchestrates OpenAI deep‑research jobs pointed at the MCP server, producing a JSON digest (`draft.json`).

Conventions:
- Target Python 3.10+; line length 120; prefer type hints and clear names.
- Run checks before committing:
  ```bash
  ruff check .
  python -m compileall -q .
  ```
- Keep `docs/` for input PDFs/DOCX; `data/` for generated artefacts; `draft.json`/`draft.md` for current draft content.
