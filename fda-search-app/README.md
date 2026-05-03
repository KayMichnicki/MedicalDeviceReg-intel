# FDA Guidance Search & Protocol Checker (Streamlit)

## Local Run
```bash
python -m venv venv && source venv/bin/activate
pip install -r fda-search-app/requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run fda-search-app/app.py
```

## Streamlit Community Cloud
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app pointing to `fda-search-app/app.py`.
3. Add secret `OPENAI_API_KEY` under Settings → Secrets:
```
[openai]
api_key = "sk-..."
```

## Render.com (recommended)
Option A: One-click via Blueprint
- Connect repo → New + → Blueprint → pick `render.yaml`.
- Add env var `OPENAI_API_KEY`.

Option B: Manual Web Service
- Build Command: `pip install -r fda-search-app/requirements.txt`
- Start Command: `streamlit run fda-search-app/app.py --server.port $PORT --server.address 0.0.0.0`
- Env Var: `OPENAI_API_KEY`

## Docker
```bash
docker build -t fda-search-app -f fda-search-app/Dockerfile .
docker run -e OPENAI_API_KEY=sk-... -p 8501:8501 fda-search-app
```
Navigate to http://localhost:8501

## Hugging Face Spaces
- Create a new Space (Streamlit), upload this project.
- Make sure `requirements.txt` is at `fda-search-app/requirements.txt` or copy it to root.
- Set `OPENAI_API_KEY` secret in Space settings.

## Notes
- PDFs for the persistent corpus live in `fda-search-app/fda_docs/`.
- **Medical device workflow**: Live updates default to **CDRH only**; use the **Device incidents → guidance** tab to pull OpenFDA device recalls / MAUDE events and match narratives to your CDRH guidance index. Optional env: `OPENFDA_API_KEY` (higher OpenFDA rate limits).
- FastAPI: `POST /api/match-incident` with form fields `text`, `incident_kind` mirrors that pipeline.
- Technical write-up (data pipeline, retrieval math, freshness): [`docs/METHOD_AND_DATA_PIPELINE.md`](docs/METHOD_AND_DATA_PIPELINE.md).
- Use the "Live updates" tab to fetch the latest CDER/CDRH guidance and optionally save to `fda_docs/`.
- Use "Update monitor" to detect newly published/updated FDA PDFs against local sync state (`.fda_sync_state.json`).
- Configure auto-check interval (daily/weekly/monthly) to run update checks automatically when the app opens.
- Default automation is set to daily checks, auto-save of new/updated PDFs, and automatic index rebuild after auto-save.
- Model and API key are read from Streamlit Secrets or `OPENAI_API_KEY` env var.
- "Draft assistant" asks guided regulatory questions (device/drug/etc.) and generates a first draft protocol plan grounded in matched FDA guidance.
