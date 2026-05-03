import json
import os
import re
import textwrap
import typing as t

from rapidfuzz import fuzz
from langchain_openai import ChatOpenAI

VALID_BRANCHES = frozenset({"drug", "device", "biologic"})


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def _parse_classification_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    m = re.search(r"\{[\s\S]*\}", raw)
    blob = m.group(0) if m else raw
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    branches = data.get("branches") or []
    if not isinstance(branches, list):
        branches = []
    branches = [b for b in branches if b in VALID_BRANCHES]
    primary = data.get("primary")
    if primary not in VALID_BRANCHES:
        primary = branches[0] if branches else None
    rationale = data.get("rationale") if isinstance(data.get("rationale"), str) else ""
    return {"branches": branches, "primary": primary, "rationale": rationale.strip()}


def classify_sop_branches(sop_text: str, llm: ChatOpenAI, max_chars: int = 14000) -> dict:
    sample = (sop_text or "")[:max_chars]
    prompt = textwrap.dedent(
        f"""
        You are an FDA regulatory analyst. Read the SOP or internal procedure and decide which FDA product jurisdictions apply.

        Respond with ONLY valid JSON (no markdown fences) in this exact shape:
        {{"branches": ["drug"|"device"|"biologic", ...], "primary": "drug"|"device"|"biologic"|null, "rationale": "one short sentence"}}

        Rules:
        - "drug" = pharmaceutical drug regulation (CDER): manufacturing, labeling, stability, clinical supplies, GMP for drugs, etc.
        - "device" = medical device / IVD / radiological health (CDRH): design controls, 510(k), PMA support, QMS, software as device when applicable.
        - "biologic" = biologics (CBER): vaccines, blood, gene/cellular therapy, tissue, similar biological products.
        - Use multiple entries in "branches" when the SOP clearly spans more than one (e.g., combination product, shared QA for drug+device).
        - If uncertain, set "primary" to the single most likely branch and still list only that branch.

        Document:
        {sample}
        """
    ).strip()
    resp = llm.invoke(prompt)
    text_out = resp.content.strip() if hasattr(resp, "content") else str(resp)
    parsed = _parse_classification_json(text_out)
    if not parsed.get("branches"):
        parsed = {"branches": ["drug"], "primary": "drug", "rationale": "Fallback: no parse; defaulting to drug."}
    if parsed.get("primary") is None and parsed.get("branches"):
        parsed["primary"] = parsed["branches"][0]
    return parsed


def summarize_incident_for_retrieval(
    incident_text: str,
    llm: ChatOpenAI,
    *,
    incident_kind: str = "device_incident",
    max_chars: int = 12000,
) -> str:
    """Compress recall / MAUDE narrative into a retrieval-oriented paragraph."""
    sample = (incident_text or "")[:max_chars]
    kind_hint = (
        "FDA medical device recall or field correction narrative"
        if "recall" in incident_kind.lower()
        else "FDA MAUDE medical device adverse event narrative"
    )
    prompt = textwrap.dedent(
        f"""
        Compress this {kind_hint} into one paragraph (<=140 words) optimized for semantic search
        against FDA device guidance (CDRH): design controls, risk management ISO 14971,
        labeling, complaints/Maude reporting, recalls, CAPA, software as medical device,
        cybersecurity, sterility, biocompatibility, human factors, postmarket surveillance.

        Incident text:
        {sample}
        """
    ).strip()
    resp = llm.invoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)


def summarize_sop_for_retrieval(sop_text: str, llm: ChatOpenAI, max_chars: int = 12000) -> str:
    sample = (sop_text or "")[:max_chars]
    prompt = textwrap.dedent(
        f"""
        Compress this SOP into one paragraph (<=120 words) optimized for semantic search against FDA guidance titles and text.
        Include: product/process type, regulated activities, keywords (GMP, validation, clinical, labeling, risk, software, sterility, etc.).

        SOP:
        {sample}
        """
    ).strip()
    resp = llm.invoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)


def _branch_label(branch: str) -> str:
    return {"drug": "Drugs (CDER)", "device": "Devices (CDRH)", "biologic": "Biologics (CBER)"}.get(
        branch, branch
    )


def build_regulatory_answer(
    *,
    sop_summary: str,
    branches: list[str],
    context_blocks: list[tuple[str, str, str]],
    llm: ChatOpenAI,
    user_summary_label: str = "SOP summary",
) -> str:
    lines = []
    for source, branch, excerpt in context_blocks:
        lines.append(f"Source [{branch}]: {source}\nExcerpt: {excerpt}")
    context = "\n\n".join(lines)
    branch_pretty = ", ".join(_branch_label(b) for b in branches)
    prompt = textwrap.dedent(
        f"""
        You are an FDA regulatory assistant. The user's document was classified under: {branch_pretty}.

        Using ONLY the FDA context excerpts below, list the most relevant regulations/guidances to consider.
        For each bullet: guidance or topic name, why it applies, and the source filename in parentheses.
        If the context is thin, say what is missing and suggest broader search keywords—do not invent citations.

        FDA context:
        {context}

        {user_summary_label}:
        {sop_summary}
        """
    ).strip()
    resp = llm.invoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)


def _rerank_docs(query: str, docs: t.Sequence, top_k: int) -> list:
    scored = []
    q = (query or "").lower().strip()
    for idx, d in enumerate(docs or []):
        meta = getattr(d, "metadata", None) or {}
        source = str(meta.get("source", "") or "")
        text = (getattr(d, "page_content", None) or "")[:2000]
        lexical = fuzz.token_set_ratio(q, f"{source} {text}".lower()) / 100.0
        pos_prior = max(0.0, 1.0 - (idx / 40.0))
        score = 0.72 * lexical + 0.28 * pos_prior
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def retrieve_branch_filtered(vector_store, query: str, branches: list[str], k: int = 8, pool: int = 48):
    allowed = set(branches) | {"general"}
    raw = vector_store.similarity_search(query, k=pool)
    filtered = [d for d in raw if (d.metadata or {}).get("regulatory_branch") in allowed]
    if len(filtered) < max(4, k // 2):
        seen = {id(x) for x in filtered}
        for d in raw:
            if id(d) not in seen:
                filtered.append(d)
                seen.add(id(d))
            if len(filtered) >= pool:
                break
    return _rerank_docs(query, filtered, top_k=k)


def match_sop_to_regulations(
    *,
    sop_text: str,
    vector_store,
    llm: ChatOpenAI,
    branches_override: t.Optional[list[str]] = None,
    top_k: int = 8,
) -> dict:
    if branches_override:
        picked = [b for b in branches_override if b in VALID_BRANCHES]
        if picked:
            classification = {
                "branches": picked,
                "primary": picked[0],
                "rationale": "User-selected FDA jurisdictions.",
            }
        else:
            classification = classify_sop_branches(sop_text, llm)
    else:
        classification = classify_sop_branches(sop_text, llm)

    branches = classification.get("branches") or ["drug"]
    summary = summarize_sop_for_retrieval(sop_text, llm)
    retrieval_query = f"{summary}\n\nJurisdictions: {', '.join(branches)}"
    docs = retrieve_branch_filtered(vector_store, retrieval_query, branches, k=top_k)
    context_blocks = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "Unknown")
        branch = meta.get("regulatory_branch", "general")
        excerpt = (d.page_content or "")[:1400]
        context_blocks.append((str(src), str(branch), excerpt))

    answer = build_regulatory_answer(
        sop_summary=summary,
        branches=branches,
        context_blocks=context_blocks,
        llm=llm,
        user_summary_label="SOP summary",
    )

    sources_out = []
    for src, branch, excerpt in context_blocks:
        sources_out.append(
            {
                "filename": src,
                "regulatory_branch": branch,
                "branch_label": _branch_label(branch) if branch in VALID_BRANCHES else branch,
                "excerpt": excerpt,
            }
        )

    return {
        "classification": classification,
        "sop_summary": summary,
        "retrieval_query": retrieval_query,
        "answer_markdown": answer,
        "sources": sources_out,
    }


def match_incident_to_regulations(
    *,
    incident_text: str,
    vector_store,
    llm: ChatOpenAI,
    incident_kind: str = "device_incident",
    top_k: int = 8,
) -> dict:
    """
    Match recall / MAUDE-style narratives to CDRH-oriented guidance chunks only.

    Does not call jurisdictional classifier; branch is fixed to device for focused retrieval.
    """
    classification = {
        "branches": ["device"],
        "primary": "device",
        "rationale": "Medical device incident workflow; retrieval restricted to device (CDRH) jurisdiction.",
    }
    summary = summarize_incident_for_retrieval(
        incident_text, llm, incident_kind=incident_kind
    )
    retrieval_query = f"{summary}\n\nJurisdictions: device\nIncident kind: {incident_kind}"
    docs = retrieve_branch_filtered(vector_store, retrieval_query, ["device"], k=top_k)
    context_blocks = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "Unknown")
        branch = meta.get("regulatory_branch", "general")
        excerpt = (d.page_content or "")[:1400]
        context_blocks.append((str(src), str(branch), excerpt))

    answer = build_regulatory_answer(
        sop_summary=summary,
        branches=["device"],
        context_blocks=context_blocks,
        llm=llm,
        user_summary_label="Incident narrative summary",
    )

    sources_out = []
    for src, branch, excerpt in context_blocks:
        sources_out.append(
            {
                "filename": src,
                "regulatory_branch": branch,
                "branch_label": _branch_label(branch) if branch in VALID_BRANCHES else branch,
                "excerpt": excerpt,
            }
        )

    return {
        "classification": classification,
        "incident_summary": summary,
        "retrieval_query": retrieval_query,
        "answer_markdown": answer,
        "sources": sources_out,
        "incident_kind": incident_kind,
    }


def openai_configured() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    return False
