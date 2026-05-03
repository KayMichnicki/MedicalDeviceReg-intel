import typing as t
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from loader import (
    extract_text_from_docx_bytes,
    extract_text_from_pdf_bytes,
    extract_text_from_txt_bytes,
    load_documents,
)
from regulatory_core import (
    get_llm,
    match_incident_to_regulations,
    match_sop_to_regulations,
    openai_configured,
)
from vector_store import build_vector_store

APP_DIR = Path(__file__).resolve().parent
FDA_DOCS_DIR = APP_DIR / "fda_docs"
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        docs = load_documents(str(FDA_DOCS_DIR))
        if not docs:
            raise RuntimeError(
                f"No PDFs in {FDA_DOCS_DIR}. Add guidance PDFs or use Streamlit Live updates to fetch FDA documents."
            )
        _vector_store = build_vector_store(docs)
    return _vector_store


app = FastAPI(title="FDA Regulation Matcher", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, "has_key": openai_configured(), "docs_dir": str(FDA_DOCS_DIR)},
    )


def _extract_sop_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    if name.endswith(".docx"):
        return extract_text_from_docx_bytes(data)
    if name.endswith(".txt"):
        return extract_text_from_txt_bytes(data)
    raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")


def _parse_branches_form(raw: t.Optional[str]) -> t.Optional[list[str]]:
    if not raw or not raw.strip():
        return None
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    allowed = {"drug", "device", "biologic"}
    out = [p for p in parts if p in allowed]
    return out if out else None


@app.post("/api/match-sop")
async def api_match_sop(
    file: UploadFile = File(...),
    branches: t.Optional[str] = Form(None),
):
    if not openai_configured():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is required for classification and answers.",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        text = _extract_sop_text(file.filename or "", data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read document: {exc}") from exc
    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="No extractable text in file.")

    try:
        vs = get_vector_store()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    override = _parse_branches_form(branches)
    result = match_sop_to_regulations(
        sop_text=text,
        vector_store=vs,
        llm=get_llm(),
        branches_override=override,
        top_k=8,
    )
    return result


@app.post("/api/match-incident")
async def api_match_incident(
    text: str = Form(...),
    incident_kind: str = Form("pasted_incident"),
):
    """Map device recall / MAUDE-style text to CDRH guidance chunks (same pipeline as Streamlit incident tab)."""
    if not openai_configured():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is required for summarization and retrieval.",
        )
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty incident text.")

    try:
        vs = get_vector_store()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = match_incident_to_regulations(
        incident_text=text,
        vector_store=vs,
        llm=get_llm(),
        incident_kind=incident_kind or "pasted_incident",
        top_k=8,
    )
    return result


@app.post("/api/reload-index")
async def reload_index():
    global _vector_store
    _vector_store = None
    try:
        get_vector_store()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "message": "Vector index rebuilt from fda_docs."}
