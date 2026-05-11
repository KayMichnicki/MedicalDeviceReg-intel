import os
import time
import textwrap
import streamlit as st
from rapidfuzz import fuzz
from streamlit.errors import StreamlitSecretNotFoundError
from loader import (
    load_documents,
    extract_text_from_pdf_bytes,
    extract_text_from_docx_bytes,
    extract_text_from_txt_bytes,
    metadata_from_center_code,
)
from vector_store import build_vector_store
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from fda_fetcher import (
    fetch_guidance_pdfs,
    persist_guidance_pdfs,
    check_for_guidance_updates,
    load_sync_state,
)
from openfda_client import (
    OpenFDAError,
    device_event_to_narrative,
    fetch_device_events,
    fetch_device_recalls,
    recall_record_to_narrative,
)
from regulatory_core import match_incident_to_regulations, retrieve_branch_filtered


st.set_page_config(page_title="FDA Guidance Search", layout="wide")

MODE_TO_BRANCH = {
    "Medical devices (CDRH)": "device",
    "Drugs (CDER)": "drug",
    "Biologics (CBER)": "biologic",
    "All (no filter)": None,
}


def _mode_default_centers(mode_label: str) -> list[str]:
    if mode_label == "Drugs (CDER)":
        return ["CDER (Drugs)"]
    if mode_label == "Biologics (CBER)":
        return ["CBER (Biologics)"]
    if mode_label == "All (no filter)":
        return ["CDRH (Devices)", "CDER (Drugs)", "CBER (Biologics)"]
    return ["CDRH (Devices)"]


mode_label = st.sidebar.radio(
    "Mode",
    options=list(MODE_TO_BRANCH.keys()),
    index=0,
    help="Controls retrieval filtering for search/protocol/draft. Device incidents are available in device mode only.",
)
ACTIVE_BRANCH = MODE_TO_BRANCH[mode_label]

title_map = {
    "device": "FDA Guidance Search — Medical Devices",
    "drug": "FDA Guidance Search — Drugs",
    "biologic": "FDA Guidance Search — Biologics",
    None: "FDA Guidance Search — All",
}
st.title(f"🔎 {title_map.get(ACTIVE_BRANCH, 'FDA Guidance Search')}")

logo_cols = st.columns(3)
with logo_cols[0]:
    st.image("static/logo_device.svg", width=54, caption="Devices")
with logo_cols[1]:
    st.image("static/logo_drug.svg", width=54, caption="Drugs")
with logo_cols[2]:
    st.image("static/logo_biologic.svg", width=54, caption="Biologics")

st.caption(
    "Corpus: FDA guidance PDFs in `fda_docs/`. "
    "Use the mode selector to focus retrieval; Live updates fetches new PDFs by center."
)


def _ensure_openai_key():
    key = None
    # Prefer Streamlit secrets if available
    try:
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            key = st.secrets["openai"]["api_key"]
    except StreamlitSecretNotFoundError:
        # It's fine if secrets.toml doesn't exist; we can fall back to env var.
        pass
    # Fallback to environment variable
    if key is None:
        key = os.environ.get("OPENAI_API_KEY")
    if key:
        os.environ["OPENAI_API_KEY"] = key
    else:
        st.warning("OpenAI API key not found. Set in st.secrets['openai']['api_key'] or env OPENAI_API_KEY.")
    return bool(os.environ.get("OPENAI_API_KEY"))


def _index_cap_cache_key() -> str:
    """Invalidate vector cache when OpenAI index size cap changes."""
    return (os.environ.get("FDA_INDEX_MAX_DOCS") or "").strip()


@st.cache_resource(show_spinner=False)
def get_vector_store(index_cap_key: str):
    documents = load_documents("fda_docs")
    return build_vector_store(documents)


@st.cache_resource(show_spinner=False)
def get_retriever(index_cap_key: str):
    return get_vector_store(index_cap_key).as_retriever(search_kwargs={"k": 20})


@st.cache_resource(show_spinner=False)
def get_llm():
    # Use a cost-effective, capable model
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@st.cache_resource(show_spinner=False)
def get_qa_chain(index_cap_key: str):
    # Return source documents for citation display
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=get_retriever(index_cap_key),
        return_source_documents=True,
    )


def render_sources(sources):
    if not sources:
        return
    with st.expander("Show sources"):
        for i, doc in enumerate(sources, start=1):
            source_name = doc.metadata.get("source") or doc.metadata.get("name") or f"Doc {i}"
            snippet = doc.page_content[:500].replace("\n", " ") + ("…" if len(doc.page_content) > 500 else "")
            st.markdown(f"**{i}. {source_name}**")
            st.caption(snippet)


def _rerank_docs(query: str, docs, top_k: int = 6):
    """
    Hybrid rerank: combine vector retrieval with lexical similarity.
    """
    scored = []
    q = (query or "").lower().strip()
    for idx, d in enumerate(docs or []):
        source = (d.metadata.get("source", "") if getattr(d, "metadata", None) else "") or ""
        text = (d.page_content or "")[:2000]
        lexical = fuzz.token_set_ratio(q, f"{source} {text}".lower()) / 100.0
        # Early vector results tend to be stronger, so keep a small positional prior.
        pos_prior = max(0.0, 1.0 - (idx / 25.0))
        score = 0.75 * lexical + 0.25 * pos_prior
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def _retrieve_top_docs(query: str, *, top_k: int = 6):
    if not openai_available:
        st.warning("Set `OPENAI_API_KEY` to enable FDA guidance retrieval.")
        return []
    if ACTIVE_BRANCH is None:
        raw_docs = retriever.get_relevant_documents(query)
        return _rerank_docs(query, raw_docs, top_k=top_k)
    try:
        vs = get_vector_store(_index_cap_cache_key())
    except Exception as exc:
        st.error(f"Vector store error: {exc}")
        return []
    # Pull from FAISS, then metadata-filter by branch (plus 'general'), then rerank.
    return retrieve_branch_filtered(vs, query, [ACTIVE_BRANCH], k=top_k, pool=48)


def _answer_from_retrieved_docs(*, user_query: str, docs) -> str:
    ctx_lines = []
    for d in (docs or [])[:8]:
        src = (d.metadata or {}).get("source", "Unknown")
        excerpt = (d.page_content or "")[:1400]
        ctx_lines.append(f"Source: {src}\nExcerpt: {excerpt}")
    context = "\n\n".join(ctx_lines)
    prompt = textwrap.dedent(
        f"""
        You are an FDA regulatory assistant.
        Using ONLY the context excerpts below, answer the user's question.
        If the context is insufficient, say what is missing and suggest search terms—do not invent citations.

        User question:
        {user_query}

        Context excerpts:
        {context}
        """
    ).strip()
    resp = get_llm().invoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)


def _rebuild_library_index():
    if not openai_available:
        st.warning("Set `OPENAI_API_KEY` to enable index building.")
        return
    try:
        get_vector_store.clear()
        get_retriever.clear()
        get_qa_chain.clear()
    except Exception:
        pass
    _ = get_qa_chain(_index_cap_cache_key())


openai_available = _ensure_openai_key()
INDEX_CAP_KEY = _index_cap_cache_key()
qa = get_qa_chain(INDEX_CAP_KEY) if openai_available else None
retriever = get_retriever(INDEX_CAP_KEY) if openai_available else None

_cap_raw = os.environ.get("FDA_INDEX_MAX_DOCS", "").strip()
if _cap_raw:
    try:
        _cap_n = int(_cap_raw)
        if _cap_n > 0:
            st.sidebar.info(
                f"**Cost-saving index:** using only the first **{_cap_n}** PDF(s) "
                "in `fda_docs/` (`FDA_INDEX_MAX_DOCS`). Remove the env var to index all PDFs."
            )
    except ValueError:
        st.sidebar.warning("`FDA_INDEX_MAX_DOCS` is not a positive integer; ignoring.")

search_tab, check_tab, incident_tab, live_tab, draft_tab = st.tabs(
    [
        "Search guidance",
        "Protocol checker",
        "Device incidents → guidance",
        "Live updates",
        "Draft assistant",
    ]
)

with search_tab:
    st.caption("Search runs only when you click **Search** (saves API tokens on reruns while you edit).")
    with st.form("guidance_search_form", clear_on_submit=False):
        query = st.text_input(
            "Ask a question about FDA guidance documents",
            key="search_query",
            placeholder="e.g., What does FDA expect for design validation under QSR?",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        q = (query or "").strip()
        if not q:
            st.warning("Enter a question, then click **Search**.")
        elif not openai_available:
            st.warning("Set `OPENAI_API_KEY` to enable guidance search.")
        else:
            with st.spinner("Searching FDA documents…"):
                if ACTIVE_BRANCH is None:
                    result = qa.invoke({"query": q})
                    st.success("Answer:")
                    st.write(result.get("result") or result)
                    render_sources(result.get("source_documents"))
                else:
                    docs = _retrieve_top_docs(q, top_k=8)
                    answer = _answer_from_retrieved_docs(user_query=q, docs=docs)
                    st.success("Answer:")
                    st.write(answer)
                    render_sources(docs)


def _extract_uploaded_files_text(uploaded_files) -> str:
    texts = []
    for f in uploaded_files or []:
        name = f.name.lower()
        data = f.read()
        if name.endswith(".pdf"):
            texts.append(extract_text_from_pdf_bytes(data))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx_bytes(data))
        elif name.endswith(".txt"):
            texts.append(extract_text_from_txt_bytes(data))
        else:
            st.warning(f"Unsupported file type for {f.name}. Upload PDF, DOCX, or TXT.")
    return "\n\n".join(t for t in texts if t)


def _summarize_protocol(text: str) -> str:
    text_sample = text[:15000]
    prompt = (
        "Summarize the following clinical/protocol document in <=250 words. "
        "Highlight: product type, indication, endpoints, study design, key procedures, populations, and regulated areas.\n\n"
        f"Document:\n{text_sample}"
    )
    resp = get_llm().invoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)


with check_tab:
    st.markdown("Upload your protocol(s) and get applicable FDA guidance and a citation-ready checklist.")
    uploaded = st.file_uploader(
        "Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    if uploaded:
        if not openai_available:
            st.warning("Set `OPENAI_API_KEY` to enable protocol matching and checklist generation.")
            st.stop()
        protocol_text = _extract_uploaded_files_text(uploaded)
        if not protocol_text:
            st.warning("No readable text found in uploaded files.")
        else:
            with st.spinner("Analyzing protocol and matching to FDA guidance…"):
                summary = _summarize_protocol(protocol_text)
                st.subheader("Protocol summary")
                st.write(summary)

                query_for_guidance = (
                    "Identify the most relevant FDA guidance documents and CFR parts to consider given this protocol. "
                    "Return titles and short rationales.\n\nProtocol summary:\n"
                    + summary
                )
                docs = _retrieve_top_docs(query_for_guidance, top_k=6)

                # Compose a concise context from retrieved documents
                ctx_lines = []
                for d in docs[:6]:
                    source = d.metadata.get("source", "Unknown")
                    excerpt = d.page_content[:1200]
                    ctx_lines.append(f"Source: {source}\nExcerpt: {excerpt}")
                context = "\n\n".join(ctx_lines)

                checklist_prompt = textwrap.dedent(
                    f"""
                    You are an FDA submissions assistant. Using ONLY the context, produce:
                    1) A checklist of applicable FDA guidance documents/regulations to cite
                    2) Specific sections to reference where possible
                    3) Gaps or missing elements in the protocol to add

                    Output as markdown with bullet points and include source filenames in parentheses.

                    Context:
                    {context}

                    Protocol summary:
                    {summary}
                    """
                ).strip()

                checklist = get_llm().invoke(checklist_prompt)

                st.subheader("Guidance checklist and recommendations")
                st.write(checklist.content if hasattr(checklist, "content") else str(checklist))

                st.subheader("Top retrieved sources")
                render_sources(docs)


with incident_tab:
    if ACTIVE_BRANCH not in (None, "device"):
        st.info("Device incidents are available in **Medical devices (CDRH)** mode only.")
        st.stop()
    st.markdown(
        "Pull **device recalls** or **MAUDE adverse events** from OpenFDA, or paste your own narrative, "
        "then map the incident to **CDRH guidance chunks** already in your library index."
    )
    if not openai_available:
        st.warning("Set `OPENAI_API_KEY` to summarize incidents and retrieve guidance.")
    source_kind = st.radio(
        "Incident source",
        options=[
            "OpenFDA device recall",
            "OpenFDA MAUDE event",
            "Paste incident text",
        ],
        horizontal=True,
    )
    col_ia, col_ib = st.columns(2)
    with col_ia:
        of_limit = st.number_input("OpenFDA limit", min_value=1, max_value=50, value=5, step=1)
    with col_ib:
        of_search = st.text_input(
            "OpenFDA search filter (optional)",
            placeholder=r'e.g. product_description:"catheter" OR reason_for_recall:software',
            help="Uses OpenFDA `search=` syntax. Leave empty for recent records.",
        )

    narrative_key = "incident_narrative_text"
    if narrative_key not in st.session_state:
        st.session_state[narrative_key] = ""

    if source_kind.startswith("OpenFDA"):
        fetch_open = st.button("Fetch from OpenFDA")
        if fetch_open:
            try:
                with st.spinner("Calling OpenFDA…"):
                    if source_kind.endswith("recall"):
                        st.session_state["_last_events"] = None
                        recs = fetch_device_recalls(
                            limit=int(of_limit),
                            search=of_search.strip() or None,
                        )
                        if not recs:
                            st.session_state["_last_recalls"] = None
                            st.info("No recalls returned. Try a different filter or raise the limit.")
                        else:
                            st.session_state["_last_recalls"] = recs
                            st.success(f"Fetched {len(recs)} recall record(s).")
                    else:
                        st.session_state["_last_recalls"] = None
                        recs = fetch_device_events(
                            limit=int(of_limit),
                            search=of_search.strip() or None,
                        )
                        if not recs:
                            st.session_state["_last_events"] = None
                            st.info("No MAUDE events returned. Try a different filter.")
                        else:
                            st.session_state["_last_events"] = recs
                            st.success(f"Fetched {len(recs)} event record(s).")
            except OpenFDAError as exc:
                st.error(str(exc))

    if source_kind == "OpenFDA device recall" and st.session_state.get("_last_recalls"):
        recs = st.session_state["_last_recalls"]
        labels = [
            f"{i}: {r.get('recall_number', '')} — {(r.get('product_description') or '')[:80]}"
            for i, r in enumerate(recs)
        ]
        pick = st.selectbox("Select recall", options=range(len(recs)), format_func=lambda i: labels[i])
        st.session_state[narrative_key] = recall_record_to_narrative(recs[pick])

    if source_kind == "OpenFDA MAUDE event" and st.session_state.get("_last_events"):
        evs = st.session_state["_last_events"]
        ev_labels = [
            f"{i}: {e.get('report_number', e.get('mdr_report_key', ''))}"
            for i, e in enumerate(evs)
        ]
        pick_e = st.selectbox(
            "Select MAUDE report", options=range(len(evs)), format_func=lambda i: ev_labels[i]
        )
        st.session_state[narrative_key] = device_event_to_narrative(evs[pick_e])

    if source_kind == "Paste incident text":
        st.text_area(
            "Incident / recall narrative",
            height=220,
            placeholder="Paste recall text, complaint summary, or MAUDE-style narrative…",
            key=narrative_key,
        )

    if source_kind != "Paste incident text" and st.session_state.get(narrative_key):
        with st.expander("Preview narrative sent to retrieval"):
            st.text(st.session_state[narrative_key][:8000])

    run_match = st.button("Match incident to CDRH guidance")
    if run_match:
        text_in = (st.session_state.get(narrative_key) or "").strip()
        if not text_in:
            st.warning("Fetch a record or paste incident text first.")
        elif not openai_available:
            st.stop()
        else:
            if source_kind == "OpenFDA device recall":
                ik = "device_recall"
            elif source_kind == "OpenFDA MAUDE event":
                ik = "maude_adverse_event"
            else:
                ik = "pasted_incident"
            with st.spinner("Summarizing incident and retrieving device guidance chunks…"):
                try:
                    vs = get_vector_store(_index_cap_cache_key())
                except Exception as exc:
                    st.error(f"Vector store error: {exc}")
                    st.stop()
                result = match_incident_to_regulations(
                    incident_text=text_in,
                    vector_store=vs,
                    llm=get_llm(),
                    incident_kind=ik,
                    top_k=8,
                )
            st.subheader("LLM incident summary (retrieval query seed)")
            st.write(result.get("incident_summary"))
            st.subheader("Suggested guidances / regulatory topics")
            st.markdown(result.get("answer_markdown") or "")
            st.subheader("Retrieved sources")
            for s in result.get("sources") or []:
                with st.expander(f"{s.get('branch_label')}: {s.get('filename')}"):
                    st.text((s.get("excerpt") or "")[:2500])


with live_tab:
    st.markdown("Fetch the latest FDA guidance PDFs filtered by center (e.g., Drugs, Devices), then query them.")
    col1, col2, col3 = st.columns(3)
    with col1:
        centers_select = st.multiselect(
            "Centers",
            options=["CDER (Drugs)", "CDRH (Devices)", "CBER (Biologics)"],
            default=_mode_default_centers(mode_label),
            help="Defaults follow the selected mode; adjust as needed.",
        )
    with col2:
        search_term = st.text_input("Optional search term", placeholder="e.g., clinical trial endpoints")
    with col3:
        max_docs = st.number_input(
            "Max docs (fetch)",
            min_value=5,
            max_value=100,
            value=5,
            step=5,
            help="How many FDA guidance listings to crawl this run (not the embedding index cap).",
        )

    centers_codes = []
    for item in centers_select:
        if item.startswith("CDER"):
            centers_codes.append("CDER")
        elif item.startswith("CDRH"):
            centers_codes.append("CDRH")
        elif item.startswith("CBER"):
            centers_codes.append("CBER")

    do_fetch = st.button("Fetch latest")
    if do_fetch:
        with st.spinner("Fetching and parsing FDA guidance PDFs…"):
            fetched = fetch_guidance_pdfs(centers=centers_codes or ("CDER", "CDRH", "CBER"), query=search_term or None, max_docs=int(max_docs))
        if not fetched:
            st.info("No documents found. Try broadening filters.")
        else:
            st.success(f"Fetched {len(fetched)} documents.")
            for d in fetched[:10]:
                st.markdown(f"- **{d.center_code}**: {d.title}")

            if not openai_available:
                st.warning("Set `OPENAI_API_KEY` to ask questions about the freshly fetched documents.")
            else:
                # Build an ad-hoc vector store over fetched docs only
                docs_for_vs = []
                for d in fetched:
                    meta = metadata_from_center_code(d.center_code)
                    safe_name = f"{d.center_code}_{d.title}.pdf".replace("/", "-")
                    docs_for_vs.append(
                        {"content": d.text, "name": safe_name, **meta}
                    )
                live_vs = build_vector_store(docs_for_vs)
                live_retriever = live_vs.as_retriever(search_kwargs={"k": 5})
                live_qa = RetrievalQA.from_chain_type(
                    llm=get_llm(),
                    retriever=live_retriever,
                    return_source_documents=True,
                )

                live_query = st.text_input(
                    "Ask about the freshly fetched documents", key="live_query"
                )
                if live_query:
                    with st.spinner("Answering from freshly fetched set…"):
                        result = live_qa.invoke({"query": live_query})
                        st.write(result.get("result") or result)
                        render_sources(result.get("source_documents"))

            st.divider()
            st.caption("Optionally save fetched PDFs to your local library and refresh the global index.")
            save = st.checkbox("Save fetched PDFs into fda_docs", value=False)
            if save:
                saved = persist_guidance_pdfs(fetched, folder="fda_docs")
                st.success(f"Saved {len(saved)} files into fda_docs/")
                if st.button("Rebuild library index"):
                    _rebuild_library_index()
                    st.success("Library index rebuilt.")

    st.divider()
    st.subheader("Update monitor")
    st.caption("Detect newly published or changed guidance PDFs since your last sync.")

    interval_label = st.selectbox(
        "Auto-check interval",
        options=["Off", "Daily", "Weekly", "Monthly"],
        index=1,
        key="auto_check_interval",
    )

    interval_seconds = {
        "Off": 0,
        "Daily": 24 * 60 * 60,
        "Weekly": 7 * 24 * 60 * 60,
        "Monthly": 30 * 24 * 60 * 60,
    }[interval_label]

    auto_save_updates = st.checkbox("Auto-save new/updated PDFs to fda_docs", value=True)
    auto_rebuild_after_save = st.checkbox(
        "Auto-rebuild index after auto-save",
        value=True,
        help="Keeps search and draft assistant in sync with newly saved documents.",
    )

    # App-open auto-check if the interval has elapsed.
    if interval_seconds > 0 and not st.session_state.get("auto_sync_ran_this_session", False):
        sync_state = load_sync_state(".fda_sync_state.json")
        last_sync = sync_state.get("last_sync_epoch")
        due = (last_sync is None) or ((int(time.time()) - int(last_sync)) >= interval_seconds)
        if due:
            with st.spinner("Auto-checking FDA updates based on selected interval…"):
                update_result_auto = check_for_guidance_updates(
                    state_path=".fda_sync_state.json",
                    centers=centers_codes or ("CDER", "CDRH", "CBER"),
                    query=search_term or None,
                    max_docs=int(max_docs),
                )
            new_auto = update_result_auto["new_docs"]
            changed_auto = update_result_auto["changed_docs"]
            st.info(
                f"Auto-check complete. New: {len(new_auto)} | Updated: {len(changed_auto)}"
            )
            if auto_save_updates and (new_auto or changed_auto):
                saved = persist_guidance_pdfs(new_auto + changed_auto, folder="fda_docs")
                st.success(f"Auto-saved {len(saved)} files into fda_docs/")
                if auto_rebuild_after_save and saved:
                    _rebuild_library_index()
                    st.success("Auto-rebuilt library index after auto-save.")
        st.session_state["auto_sync_ran_this_session"] = True

    check_updates = st.button("Check for updates")
    if check_updates:
        with st.spinner("Checking latest guidance against local sync state…"):
            update_result = check_for_guidance_updates(
                state_path=".fda_sync_state.json",
                centers=centers_codes or ("CDER", "CDRH", "CBER"),
                query=search_term or None,
                max_docs=int(max_docs),
            )
        new_docs = update_result["new_docs"]
        changed_docs = update_result["changed_docs"]
        fetched_count = update_result["fetched_count"]

        st.success(
            f"Checked {fetched_count} docs. New: {len(new_docs)} | Updated: {len(changed_docs)}"
        )

        if new_docs:
            st.markdown("**New documents detected**")
            for d in new_docs[:15]:
                st.markdown(f"- **{d.center_code}**: {d.title}")
        if changed_docs:
            st.markdown("**Changed documents detected**")
            for d in changed_docs[:15]:
                st.markdown(f"- **{d.center_code}**: {d.title}")

        if not new_docs and not changed_docs:
            st.info("No new or changed documents were detected.")

        to_save = new_docs + changed_docs
        if to_save:
            save_updates = st.checkbox(
                "Save new/updated PDFs into fda_docs",
                value=True,
                key="save_updates_checkbox",
            )
            rebuild_after_update_save = st.checkbox(
                "Rebuild index automatically after saving updates",
                value=True,
                key="rebuild_after_update_save",
            )
            if save_updates:
                saved = persist_guidance_pdfs(to_save, folder="fda_docs")
                st.success(f"Saved {len(saved)} files into fda_docs/")
                if rebuild_after_update_save and saved:
                    _rebuild_library_index()
                    st.success("Library index rebuilt automatically.")
                if st.button("Rebuild library index from updates", key="rebuild_after_update"):
                    _rebuild_library_index()
                    st.success("Library index rebuilt.")


def _guided_regulatory_query(
    product_class: str,
    intended_use: str,
    objective: str,
    procedure_details: str,
    area_focus: str,
) -> str:
    prompt = textwrap.dedent(
        f"""
        Normalize this project into a compact FDA-retrieval query.
        Include: product class, intended use, procedure objective, key risk/safety terms,
        likely regulatory pathways, and likely guidance keywords.
        Return only one paragraph query text.

        Product class: {product_class}
        Intended use: {intended_use}
        Objective: {objective}
        Area focus: {area_focus}
        Procedure details:
        {procedure_details}
        """
    ).strip()
    resp = get_llm().invoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)


def _generate_protocol_draft(
    user_context: str,
    retrieved_context: str,
) -> str:
    prompt = textwrap.dedent(
        f"""
        You are an FDA regulatory and protocol-writing assistant.
        Create a first-draft protocol outline using the user's goals and the FDA context.
        Keep output concise and practical.

        Include sections:
        1) Scope and intended use
        2) Regulatory classification assumptions
        3) Applicable guidance/regulations to cite
        4) Procedure workflow draft
        5) Data collection and documentation requirements
        6) Risk controls and quality checks
        7) Open questions for user

        User context:
        {user_context}

        FDA context:
        {retrieved_context}
        """
    ).strip()
    resp = get_llm().invoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)


with draft_tab:
    st.markdown(
        "Answer a few questions to match your procedure to relevant FDA guidance and generate a draft."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        product_class = st.selectbox(
            "Is this primarily for a medical device, drug, biologic, or combination product?",
            options=["Medical device", "Drug", "Biologic", "Combination product", "Not sure"],
        )
        area_focus = st.text_input(
            "Regulatory area focus (optional)",
            placeholder="e.g., sterility assurance, software validation, clinical endpoints",
        )
    with col_b:
        intended_use = st.text_input(
            "What is the product/procedure intended for?",
            placeholder="e.g., diagnose skin lesions, support oncology treatment decisions",
        )
        objective = st.text_input(
            "What do you want this procedure to accomplish?",
            placeholder="e.g., establish repeatable sample preparation and analysis steps",
        )

    procedure_details = st.text_area(
        "Describe the lab procedure details",
        placeholder="Include materials, steps, controls, outputs, and any safety considerations.",
        height=180,
    )

    run_assistant = st.button("Match guidance and generate draft")
    if run_assistant:
        if not openai_available:
            st.warning("Set `OPENAI_API_KEY` to enable draft generation.")
            st.stop()
        required = [product_class, intended_use.strip(), objective.strip(), procedure_details.strip()]
        if any(not r for r in required):
            st.warning("Please fill in product class, intended use, objective, and procedure details.")
        else:
            with st.spinner("Building regulatory query and retrieving applicable guidance…"):
                retrieval_query = _guided_regulatory_query(
                    product_class=product_class,
                    intended_use=intended_use,
                    objective=objective,
                    procedure_details=procedure_details,
                    area_focus=area_focus,
                )
                docs = _retrieve_top_docs(retrieval_query, top_k=6)

            st.subheader("Generated retrieval query")
            st.write(retrieval_query)

            context_blocks = []
            for d in docs[:6]:
                source = d.metadata.get("source", "Unknown")
                excerpt = d.page_content[:1000]
                context_blocks.append(f"Source: {source}\nExcerpt: {excerpt}")
            merged_context = "\n\n".join(context_blocks)
            user_context = (
                f"Product class: {product_class}\n"
                f"Intended use: {intended_use}\n"
                f"Objective: {objective}\n"
                f"Area focus: {area_focus}\n"
                f"Procedure details:\n{procedure_details}"
            )

            with st.spinner("Generating draft protocol guidance…"):
                draft = _generate_protocol_draft(user_context=user_context, retrieved_context=merged_context)

            st.subheader("Draft protocol and regulatory plan")
            st.write(draft)
            st.subheader("Top matched sources")
            render_sources(docs)
