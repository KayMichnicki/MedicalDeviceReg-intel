# Demo: generate data, test the app, show your work

Use this checklist for assignments, portfolios, or stakeholder demos.

## 1. Environment

```bash
cd fda-search-app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **`OPENAI_API_KEY`** — required for Streamlit search, protocol checker, incident summarization, and draft assistant (classification + answers).
- **`OPENFDA_API_KEY`** — optional; raises OpenFDA rate limits for recall/MAUDE fetches.
- **`EMBEDDING_BACKEND=local`** — optional; uses Hugging Face embeddings instead of OpenAI when building/running offline demos (`sentence-transformers/all-MiniLM-L6-v2`).

## 2. Generate / refresh corpus data

**Option A — already committed PDFs**

Your repo includes CFR/USCODE/CDRH PDFs under `fda_docs/`. Nothing else required for indexing.

**Option B — fetch live FDA guidance (CDRH)**

From `fda-search-app`:

```bash
export EMBEDDING_BACKEND=local   # optional, for scripted checks without OpenAI embeddings
python scripts/demo_generate_and_verify.py --fetch-guidance --fetch-max 10
```

Or use the Streamlit **Live updates** tab: centers → **CDRH only**, **Fetch latest**, optionally **Save fetched PDFs into fda_docs**.

**Naming tip:** Files prefixed with `CDRH_` get `regulatory_branch=device` metadata for stricter filtering; others may be tagged `general` but still retrieve.

## 3. Automated verification (good “lab notebook” artifact)

Runs OpenFDA ping + builds FAISS + prints top retrieval snippets:

```bash
cd fda-search-app
export EMBEDDING_BACKEND=local
python scripts/demo_generate_and_verify.py
```

Save terminal output or redirect to a log file for submission:

```bash
python scripts/demo_generate_and_verify.py 2>&1 | tee demo_run_log.txt
```

## 4. Manual app test (screenshots)

**Streamlit**

```bash
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

Suggested screenshots:

1. **Search guidance** — question about Part 820 / design controls + source expander.
2. **Device incidents → guidance** — OpenFDA recall or MAUDE fetch → **Match incident to CDRH guidance** + answer + sources.
3. **Live updates** — CDRH fetch or update monitor summary.

**FastAPI**

```bash
uvicorn web_server:app --reload --app-dir .
```

Exercise `POST /api/match-sop` or `POST /api/match-incident` (e.g. via curl or Swagger at `/docs`).

## 5. What to cite in write-ups

- **Data pipeline & retrieval math:** [`METHOD_AND_DATA_PIPELINE.md`](METHOD_AND_DATA_PIPELINE.md)
- **Repo overview:** root [`README.md`](../../README.md) and [`README.md`](../README.md) (run/deploy).

## 6. Honest limitations (good for grading rubrics)

Results depend on corpus coverage, PDF text extraction quality, and LLM grounding; outputs are **research/education aids**, not regulatory advice.
