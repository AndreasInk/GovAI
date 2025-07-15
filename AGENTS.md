# Coding Conventions for GovAI

This repository holds a prototype application that summarises HOA documents and flags sentences whose embeddings differ from their cited source. The following rules apply to all files in this repo and are intended for future Codex agents.

## Commit Messages
- The first line should be a concise summary in the imperative mood (max 72 characters).
- Use additional paragraphs for context if needed.

## Code Style
- Target Python 3.11 or newer.
- Follow PEP8 with a max line length of 120 characters.
- Use type hints and docstrings for all public functions.
- Prefer clear variable names over comments.

## Programmatic Checks
Run these commands before committing:

```bash
ruff check .
python -m compileall -q .
```

Address any lint or compile errors reported by these checks.

## Repository Notes
- `docs/` contains original PDF documents. Avoid adding large binaries without discussion.
- `data/` stores generated embeddings and JSON artefacts.
- `draft.md` is the current HOA summary. Keep it in sync with the source documents.

## Pull Requests
- Summarise notable changes and mention any new documents or dependencies.
- Ensure all programmatic checks pass and include their results in the PR description.
