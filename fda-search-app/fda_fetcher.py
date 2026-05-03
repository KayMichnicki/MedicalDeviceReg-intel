import os
import re
import time
import json
import hashlib
import typing as t
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from loader import extract_text_from_pdf_bytes


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}
BASE_URL = "https://www.fda.gov"
SEARCH_URL = f"{BASE_URL}/regulatory-information/search-fda-guidance-documents"

CENTER_NAME_TO_CODE = {
    "Center for Drug Evaluation and Research": "CDER",
    "Center for Devices and Radiological Health": "CDRH",
    "Center for Biologics Evaluation and Research": "CBER",
    "Center for Food Safety and Applied Nutrition": "CFSAN",
    "Center for Tobacco Products": "CTP",
}


@dataclass
class GuidanceDoc:
    title: str
    center_code: str
    pdf_url: str
    pdf_bytes: bytes
    text: str
    detail_url: t.Optional[str] = None


def _http_get(url: str, *, timeout: int = 20) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp


def _abs_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return f"{BASE_URL}{href}"


def _detect_center_from_page(soup: BeautifulSoup) -> t.Optional[str]:
    # Look for center label text
    text = soup.get_text(" ")
    for name, code in CENTER_NAME_TO_CODE.items():
        if name in text:
            return code
    return None


def _extract_pdf_url_from_detail(soup: BeautifulSoup) -> t.Optional[str]:
    # Prefer direct PDF anchors
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            return _abs_url(href)
    return None


def _sanitize_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name[:160]


def fetch_guidance_pdfs(
    *,
    centers: t.Sequence[str] = ("CDER", "CDRH"),
    query: t.Optional[str] = None,
    max_docs: int = 30,
    sleep_seconds: float = 0.8,
) -> list[GuidanceDoc]:
    results: list[GuidanceDoc] = []
    page = 0
    params_base = {"items_per_page": 50}
    if query:
        params_base["search"] = query

    while len(results) < max_docs and page < 6:
        params = dict(params_base)
        params["page"] = page
        r = _http_get(SEARCH_URL, timeout=25)
        # Note: some servers ignore query params unless appended manually
        if params:
            query_parts = [f"{k}={v}" for k, v in params.items()]
            url = SEARCH_URL + "?" + "&".join(query_parts)
            r = _http_get(url, timeout=25)
        soup = BeautifulSoup(r.text, "html.parser")

        # Collect listing links
        listing_links: list[tuple[str, str]] = []  # (title, href)
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            title = a.get_text(strip=True)
            if not title or len(title) < 5:
                continue
            if "/regulatory-information/search-fda-guidance-documents/" in href:
                listing_links.append((title, _abs_url(href)))

        if not listing_links:
            break

        for title, detail_url in listing_links:
            try:
                detail_resp = _http_get(detail_url, timeout=25)
                detail_soup = BeautifulSoup(detail_resp.text, "html.parser")
                center_code = _detect_center_from_page(detail_soup)
                if centers and center_code and center_code not in centers:
                    continue
                pdf_url = _extract_pdf_url_from_detail(detail_soup)
                if not pdf_url:
                    continue
                pdf_resp = _http_get(pdf_url, timeout=60)
                pdf_bytes = pdf_resp.content
                text = extract_text_from_pdf_bytes(pdf_bytes)
                if not text.strip():
                    continue
                results.append(
                    GuidanceDoc(
                        title=title,
                        center_code=center_code or "UNKNOWN",
                        pdf_url=pdf_url,
                        pdf_bytes=pdf_bytes,
                        text=text,
                        detail_url=detail_url,
                    )
                )
                if len(results) >= max_docs:
                    break
                time.sleep(sleep_seconds)
            except Exception:
                # Skip failures and continue
                continue
        page += 1
        time.sleep(sleep_seconds)
    return results


def persist_guidance_pdfs(docs: list[GuidanceDoc], folder: str) -> list[str]:
    os.makedirs(folder, exist_ok=True)
    saved: list[str] = []
    for d in docs:
        base = _sanitize_filename(f"{d.center_code}_{d.title}.pdf") or "guidance.pdf"
        path = os.path.join(folder, base)
        try:
            with open(path, "wb") as f:
                f.write(d.pdf_bytes)
            saved.append(path)
        except Exception:
            continue
    return saved


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _doc_key(doc: GuidanceDoc) -> str:
    # PDF URL is typically stable enough for document identity.
    return doc.pdf_url.strip()


def load_sync_state(state_path: str) -> dict[str, t.Any]:
    if not os.path.exists(state_path):
        return {"documents": {}, "last_sync_epoch": None}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"documents": {}, "last_sync_epoch": None}
            data.setdefault("documents", {})
            data.setdefault("last_sync_epoch", None)
            return data
    except Exception:
        return {"documents": {}, "last_sync_epoch": None}


def save_sync_state(state_path: str, state: dict[str, t.Any]) -> None:
    parent = os.path.dirname(state_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def check_for_guidance_updates(
    *,
    state_path: str,
    centers: t.Sequence[str] = ("CDER", "CDRH"),
    query: t.Optional[str] = None,
    max_docs: int = 40,
) -> dict[str, t.Any]:
    """
    Fetch latest guidance docs and compare against local sync state.
    Returns dict with keys: new_docs, changed_docs, fetched_count.
    """
    current = fetch_guidance_pdfs(centers=centers, query=query, max_docs=max_docs)
    state = load_sync_state(state_path)
    known: dict[str, dict[str, t.Any]] = state.get("documents", {})

    new_docs: list[GuidanceDoc] = []
    changed_docs: list[GuidanceDoc] = []
    next_known: dict[str, dict[str, t.Any]] = dict(known)

    for doc in current:
        key = _doc_key(doc)
        digest = _sha256(doc.pdf_bytes)
        prev = known.get(key)
        if prev is None:
            new_docs.append(doc)
        elif prev.get("sha256") != digest:
            changed_docs.append(doc)
        next_known[key] = {
            "title": doc.title,
            "center_code": doc.center_code,
            "sha256": digest,
            "pdf_url": doc.pdf_url,
            "detail_url": doc.detail_url,
            "seen_epoch": int(time.time()),
        }

    state["documents"] = next_known
    state["last_sync_epoch"] = int(time.time())
    save_sync_state(state_path, state)

    return {
        "new_docs": new_docs,
        "changed_docs": changed_docs,
        "fetched_count": len(current),
        "state_path": state_path,
    }
