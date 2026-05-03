#!/usr/bin/env python3
"""
Generate optional corpus data (FDA guidance PDFs), sanity-check OpenFDA,
and verify the local embedding index builds and retrieves chunks.

Usage (from repo root FDASearchWeb or from fda-search-app):

  cd fda-search-app
  export EMBEDDING_BACKEND=local   # no OpenAI key needed for index smoke test
  python scripts/demo_generate_and_verify.py

  # Also pull a few fresh CDRH guidance PDFs into fda_docs/:
  python scripts/demo_generate_and_verify.py --fetch-guidance --fetch-max 8

Requires: pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FDA_DOCS = ROOT / "fda_docs"


def main() -> int:
    parser = argparse.ArgumentParser(description="FDA Search Web demo data + verification")
    parser.add_argument(
        "--fetch-guidance",
        action="store_true",
        help="Fetch recent CDRH guidance PDFs via fda_fetcher and save under fda_docs/",
    )
    parser.add_argument(
        "--fetch-max",
        type=int,
        default=8,
        help="Max PDFs when --fetch-guidance (default 8)",
    )
    parser.add_argument(
        "--skip-openfda",
        action="store_true",
        help="Skip OpenFDA recall ping",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    FDA_DOCS.mkdir(parents=True, exist_ok=True)

    print(f"[*] Working dir: {ROOT}")
    print(f"[*] FDA docs folder: {FDA_DOCS}")

    if args.fetch_guidance:
        print("\n[*] Fetching CDRH guidance PDFs (may take 1–3 minutes)…")
        from fda_fetcher import fetch_guidance_pdfs, persist_guidance_pdfs

        docs = fetch_guidance_pdfs(
            centers=("CDRH",),
            query=None,
            max_docs=max(1, min(args.fetch_max, 40)),
            sleep_seconds=0.9,
        )
        if not docs:
            print("[!] No PDFs fetched (site HTML may have changed or filters excluded all).")
        else:
            saved = persist_guidance_pdfs(docs, folder=str(FDA_DOCS))
            print(f"[+] Fetched {len(docs)} document(s); saved {len(saved)} PDF path(s).")

    if not args.skip_openfda:
        print("\n[*] OpenFDA device recall ping (limit=1)…")
        try:
            from openfda_client import fetch_device_recalls

            rows = fetch_device_recalls(limit=1)
            print(f"[+] OpenFDA OK — sample recall rows: {len(rows)}")
            if rows:
                rn = rows[0].get("recall_number") or rows[0].get("product_description", "")[:60]
                print(f"    Example id/snippet: {rn}")
        except Exception as exc:
            print(f"[!] OpenFDA check failed: {exc}")

    print("\n[*] Loading PDFs from disk…")
    cap_env = os.environ.get("FDA_INDEX_MAX_DOCS", "").strip()
    if cap_env:
        print(f"    FDA_INDEX_MAX_DOCS={cap_env} (only that many PDFs, sorted by name)")
    from loader import load_documents

    records = load_documents(str(FDA_DOCS))
    pdfs = [r for r in records if r.get("name", "").lower().endswith(".pdf")]
    print(f"[+] Loaded {len(pdfs)} PDF-backed document record(s).")
    if not pdfs:
        print("[!] No PDFs in fda_docs/. Add files or run with --fetch-guidance.")
        return 1

    backend = (os.environ.get("EMBEDDING_BACKEND") or "openai").strip().lower()
    print(f"\n[*] Embedding backend: {backend or 'openai'}")
    if backend not in ("local", "huggingface", "hf"):
        print(
            "    Tip: set EMBEDDING_BACKEND=local for a free offline embedding smoke test "
            "(downloads MiniLM once)."
        )

    print("[*] Building FAISS index (first HF run downloads the model)…")
    from vector_store import build_vector_store

    vs = build_vector_store(records)

    queries = [
        "design controls risk management medical device",
        "complaint handling medical device reporting",
        "software validation SaMD",
    ]
    print("\n[*] Sample retrievals (top 3 chunks per query):\n")
    for q in queries:
        hits = vs.similarity_search(q, k=3)
        print(f"— Query: {q}")
        for i, h in enumerate(hits, 1):
            src = (h.metadata or {}).get("source", "?")
            prev = (h.page_content or "")[:220].replace("\n", " ")
            print(f"   {i}. [{src}] {prev}…")
        print()

    print("[+] Demo verification finished successfully.")
    print("\nNext steps:")
    print("  • Streamlit (needs OPENAI_API_KEY for LLM tabs): streamlit run app.py")
    print("  • Portfolio doc: docs/DEMO_WORKSHOWCASE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
