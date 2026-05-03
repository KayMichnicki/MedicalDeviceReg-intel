"""
OpenFDA JSON API helpers for medical device recalls (device/recall)
and adverse events / MAUDE (device/event).

Docs: https://open.fda.gov/apis/device/

Optional API key: set OPENFDA_API_KEY for higher rate limits.
"""

from __future__ import annotations

import os
import typing as t

import requests

OPENFDA_BASE = "https://api.fda.gov"
DEFAULT_TIMEOUT = 45


class OpenFDAError(Exception):
    pass


def _get_json(path: str, *, params: dict[str, t.Any]) -> dict:
    api_key = (os.environ.get("OPENFDA_API_KEY") or "").strip()
    if api_key:
        params = {**params, "api_key": api_key}
    url = f"{OPENFDA_BASE}{path}"
    resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    if resp.status_code == 404:
        return {"results": [], "meta": {}}
    if not resp.ok:
        raise OpenFDAError(f"OpenFDA HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def fetch_device_recalls(
    *,
    limit: int = 10,
    skip: int = 0,
    search: str | None = None,
) -> list[dict]:
    """
    Fetch device recall enforcement records.

    `search` uses OpenFDA search syntax, e.g.
    `classification:\"Class I\"` or `product_description:\"catheter\"`.
    """
    params: dict[str, t.Any] = {"limit": min(max(limit, 1), 100), "skip": max(skip, 0)}
    if search and search.strip():
        params["search"] = search.strip()
    data = _get_json("/device/recall.json", params=params)
    return list(data.get("results") or [])


def fetch_device_events(
    *,
    limit: int = 10,
    skip: int = 0,
    search: str | None = None,
) -> list[dict]:
    """Fetch MAUDE device adverse event records."""
    params: dict[str, t.Any] = {"limit": min(max(limit, 1), 100), "skip": max(skip, 0)}
    if search and search.strip():
        params["search"] = search.strip()
    data = _get_json("/device/event.json", params=params)
    return list(data.get("results") or [])


def recall_record_to_narrative(rec: dict) -> str:
    """Flatten a recall JSON object into searchable narrative text."""
    parts: list[str] = []
    if rec.get("recall_number"):
        parts.append(f"Recall number: {rec['recall_number']}")
    if rec.get("classification"):
        parts.append(f"Classification: {rec['classification']}")
    if rec.get("product_description"):
        parts.append(f"Product: {rec['product_description']}")
    if rec.get("reason_for_recall"):
        parts.append(f"Reason for recall: {rec['reason_for_recall']}")
    if rec.get("recalling_firm"):
        parts.append(f"Recalling firm: {rec['recalling_firm']}")
    if rec.get("code_information"):
        ci = rec["code_information"]
        if isinstance(ci, list):
            parts.append("Code information: " + "; ".join(str(x) for x in ci[:5]))
        else:
            parts.append(f"Code information: {ci}")
    if rec.get("distribution_pattern"):
        parts.append(f"Distribution: {rec['distribution_pattern']}")
    return "\n".join(parts)


def device_event_to_narrative(rec: dict) -> str:
    """Flatten a MAUDE device/event record into narrative text."""
    parts: list[str] = []
    if rec.get("report_number"):
        parts.append(f"Report number: {rec['report_number']}")
    md = rec.get("mdr_report_key")
    if md:
        parts.append(f"MDR report key: {md}")

    dev = rec.get("device") or []
    if isinstance(dev, list) and dev:
        d0 = dev[0] if isinstance(dev[0], dict) else {}
        brand = d0.get("brand_name") or d0.get("generic_name")
        model = d0.get("model_number")
        gn = d0.get("generic_name")
        if brand:
            parts.append(f"Device brand/name: {brand}")
        if gn and gn != brand:
            parts.append(f"Generic name: {gn}")
        if model:
            parts.append(f"Model: {model}")

    pt = rec.get("patient") or []
    if isinstance(pt, list) and pt:
        seq = []
        for p in pt[:3]:
            if not isinstance(p, dict):
                continue
            if p.get("patient_sequence_number"):
                seq.append(f"sequence {p['patient_sequence_number']}")
            prob = p.get("patient_problem")
            if isinstance(prob, list) and prob:
                seq.append("patient problems: " + ", ".join(str(x) for x in prob[:6]))
            elif prob:
                seq.append(f"patient problem: {prob}")
        if seq:
            parts.append("Patient: " + "; ".join(seq))

    if rec.get("event_type"):
        parts.append(f"Event type: {rec['event_type']}")

    mdtxt = rec.get("manufacturer_contact_zip_ext") or ""

    report = rec.get("report_source_code") or rec.get("source_type") or ""

    mi = rec.get("mdr_text") or []
    if isinstance(mi, list):
        for block in mi[:8]:
            if isinstance(block, dict):
                txt = block.get("text")
                if txt:
                    parts.append(f"MDR narrative: {txt}")

    return "\n".join(parts) + (f"\nReport meta: {report} {mdtxt}".strip() if report or mdtxt else "")


def recalls_as_training_snippets(records: list[dict]) -> list[dict]:
    """Normalize records for downstream indexing or evaluation (not neural training)."""
    out = []
    for r in records:
        out.append(
            {
                "kind": "device_recall",
                "id": r.get("recall_number") or r.get("openfda", {}).get("recall_number"),
                "text": recall_record_to_narrative(r),
            }
        )
    return out


def events_as_training_snippets(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        rid = r.get("report_number") or r.get("mdr_report_key")
        out.append({"kind": "device_adverse_event", "id": rid, "text": device_event_to_narrative(r)})
    return out
