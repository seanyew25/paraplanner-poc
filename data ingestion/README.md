# Policy ingestion

This processor sends each PDF under `Policies/` directly to Gemini, then stores:

- deterministic policy metadata in SQLite (`chunker/policies.db`)
- context-preserving clause chunks in ChromaDB (`chunker/chroma_db/`)

## Run

Install dependencies. The processor automatically loads `API_KEY` from the repository
root `.env` file:

```powershell
cd chunker
python -m pip install -r requirements.txt
python ingest_policies.py
```

The command processes all PDFs recursively, with up to four Gemini requests running
at once. Reduce concurrency when rate limits require it:

```powershell
python ingest_policies.py --workers 2
```

A failed file is reported and does not
prevent the remaining files from being processed; the command exits with status 1
when any file fails.

Use `--policies-dir`, `--database`, and `--chroma-dir` to override the defaults.
