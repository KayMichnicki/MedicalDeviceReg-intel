import os
import io
import typing as t
import fitz  # PyMuPDF
from docx import Document as DocxDocument


def metadata_from_center_code(center_code: t.Optional[str]) -> dict:
    code = (center_code or "UNKNOWN").strip().upper() or "UNKNOWN"
    branch = {"CDER": "drug", "CDRH": "device", "CBER": "biologic"}.get(code, "general")
    return {"center_code": code, "regulatory_branch": branch}


def infer_metadata_from_filename(filename: str) -> dict:
    upper = os.path.basename(filename).upper()
    for prefix, code in (
        ("CDER_", "CDER"),
        ("CDRH_", "CDRH"),
        ("CBER_", "CBER"),
    ):
        if upper.startswith(prefix):
            return metadata_from_center_code(code)
    return metadata_from_center_code("UNKNOWN")


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    stream = io.BytesIO(file_bytes)
    doc = DocxDocument(stream)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def extract_text_from_txt_bytes(file_blocks: bytes) -> str:
    return file_blocks.decode("utf-8", errors="ignore")


def _index_max_documents_from_env() -> t.Optional[int]:
    raw = (os.environ.get("FDA_INDEX_MAX_DOCS") or "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return None


def load_documents(folder: str, max_documents: t.Optional[int] = None) -> list[dict]:
    docs: list[dict] = []
    if not os.path.isdir(folder):
        return docs
    cap = max_documents if max_documents is not None else _index_max_documents_from_env()
    pdf_filenames = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
    if cap is not None:
        pdf_filenames = pdf_filenames[:cap]
    for filename in pdf_filenames:
        path = os.path.join(folder, filename)
        text = ""
        with fitz.open(path) as pdf:
            for page in pdf:
                text += page.get_text()
        meta = infer_metadata_from_filename(filename)
        docs.append({"content": text, "name": filename, **meta})
    return docs
