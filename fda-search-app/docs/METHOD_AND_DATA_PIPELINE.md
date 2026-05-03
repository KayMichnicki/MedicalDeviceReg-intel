# FDA Regulation Matcher: Data Pipeline, Retrieval Method, and Update Semantics

This note supports a longer paper or thesis chapter on the FDA Search Web application. It separates **what is fetched**, **what is indexed**, **how matching works mathematically**, and **how the system stays current** when FDA publishes new material.

## 1. Scope: medical devices (CDRH)

The current product emphasis is **medical devices** under CDRH. Operationally:

- **Guidance corpus**: PDFs stored under `fda_docs/`, preferably prefixed with `CDRH_` so chunk metadata carries `regulatory_branch=device`.
- **Incident narratives**: Pulled from OpenFDA endpoints `device/recall` and `device/event` (MAUDE), or pasted manually.
- **Retrieval**: For incidents, jurisdiction is **fixed to device** so ranking is not diluted by drug/biologic chunks.

Drugs (CDER) and biologics (CBER) remain supported where filenames encode center codes and where the generic SOP matcher still classifies jurisdiction.

## 2. Acquiring FDA data (two channels)

### 2.1 Guidance documents (HTML → PDF)

The module `fda_fetcher.py` requests FDA’s public guidance search HTML, follows detail pages, downloads linked PDFs, and extracts text (PyMuPDF). Centers are inferred from page text (e.g., “Center for Devices and Radiological Health” → `CDRH`).

This is **batch scraping**, not the official OpenFDA drug/device approval APIs; it is appropriate for **narrative regulatory text** (guidance) that powers semantic search.

### 2.2 Recalls and adverse events (JSON API)

`openfda_client.py` calls the **OpenFDA REST API**:

- `GET https://api.fda.gov/device/recall.json`
- `GET https://api.fda.gov/device/event.json`

Optional query filters use OpenFDA’s `search` syntax (fielded Boolean queries). An optional `OPENFDA_API_KEY` raises rate limits but is not required for low-volume use.

Raw JSON records are **flattened to plain-language narratives** (`recall_record_to_narrative`, `device_event_to_narrative`) so the same retrieval stack used for SOPs can consume recalls and MAUDE reports without a separate schema-specific encoder.

## 3. “Training data” vs what this application actually does

Colloquially people say “train on FDA data.” This stack uses **retrieval-augmented generation (RAG)**, not supervised fine-tuning of the matcher:

| Phase | What happens |
| --- | --- |
| **Corpus build** | PDFs → text → chunked passages (RecursiveCharacterTextSplitter, ~1000 chars, overlap 200). |
| **Embedding index** | Each chunk mapped to a vector with `OpenAIEmbeddings` (default) or `HuggingFaceEmbeddings` if `EMBEDDING_BACKEND=local`. Vectors stored in an in-memory **FAISS** index with metadata (`source`, `regulatory_branch`). |
| **Query time** | User/incident text → LLM-written **compact retrieval query** → nearest-neighbor search → **hybrid rerank** → LLM generates an answer **conditioned only on retrieved excerpts**. |

So the FDA PDFs are **training data only in the loose sense** of *building the searchable knowledge base*. The LLM weights are not updated by gradient descent on FDA PDFs in this codebase.

If you later add **supervised learning** (e.g., learning-to-rank from labeled recall–guidance pairs), that would be true “training”; the paper should distinguish this clearly from RAG indexing.

## 4. Staying current when FDA publishes new guidance

The flow is intentionally transparent:

1. **Live fetch** (`fetch_guidance_pdfs`) retrieves recent listings and optionally persists PDFs into `fda_docs/`.
2. **Change detection** (`check_for_guidance_updates`) stores each PDF URL’s SHA-256 in `.fda_sync_state.json` and surfaces **new** or **changed** blobs.
3. **Index rebuild** clears cached LangChain/FAISS objects (`get_vector_store.clear()` in Streamlit, or `POST /api/reload-index` in FastAPI) and re-embeds all PDFs in `fda_docs/`.

**Implication for the paper**: new FDA data appears in search **after** it is downloaded and the index is rebuilt. There is no automatic background job in the core library; Streamlit can run a scheduled check on app open. For production, you would document a cron job or CI step that pulls, verifies checksums, and rebuilds.

OpenFDA recalls/events are **always live** at query time when the user clicks fetch; only the **guidance side** is snapshot-based (local corpus).

## 5. Method: why embeddings + hybrid reranking + LLM

### 5.1 Dense retrieval

Let each chunk \(d_i\) have an embedding \(\mathbf{e}_i = f_\theta(\text{chunk}_i)\) and query embedding \(\mathbf{q} = f_\theta(\text{query})\), with \(f_\theta\) a fixed sentence embedding model (OpenAI API or Sentence-Transformers). FAISS returns top candidates under **inner product** (or cosine, depending on normalization). For L2-normalized vectors, cosine similarity satisfies

\[
\cos(\mathbf{q}, \mathbf{e}_i) = \mathbf{q}^\top \mathbf{e}_i.
\]

Dense retrieval captures **paraphrase and topical overlap** (“design control” vs “design history file”) better than keyword-only search.

### 5.2 Branch filter (jurisdiction prior)

Before reranking, device incidents use **metadata filtering**: only chunks whose `regulatory_branch` is in \(\{\texttt{device}, \texttt{general}\}\) are preferred. This acts as a **hard or soft prior** over document types—analogous to restricting a search engine to a site section—reducing false positives from other centers.

Implementation detail: if too few chunks pass the filter, the code **backfills** from the raw similarity list so recall does not collapse when the corpus is mis-tagged.

### 5.3 Lexical reranking (RapidFuzz)

Pure embedding retrieval can miss **exact regulatory tokens** (e.g., specific guidance acronyms, “21 CFR Part 820”). The code therefore forms a second score using **token-set ratio** fuzzy match between the query string and `source filename + chunk text` (prefix truncated for cost). For document at rank index \(i\) (0-based) in the pooled list, a heuristic score is:

\[
s_i = \alpha \cdot \text{RapidFuzz\_ratio}(q, \text{text}_i) + (1-\alpha)\cdot \left(1 - \frac{i}{N}\right),
\]

with \(\alpha\) near **0.72–0.75** and \(N\) a cap (~40). The position term is a weak **prior favoring stronger embedding ranks**.

This is an inexpensive approximation to **linear fusion** or **cross-encoder reranking**; a cross-encoder would replace RapidFuzz with a learned relevance model \(\mathrm{score}(q, d_i)\) at higher latency/cost.

### 5.4 LLM stages

Three distinct roles:

1. **Summarization for retrieval**: collapse long SOP/incident text into a short paragraph biased toward regulatory vocabulary (improves embedding alignment).
2. **Classification** (SOP path): JSON structured branch labels—implemented as prompted decoding, not a separate classifier network.
3. **Grounded answer**: generate bullets citing **only** retrieved excerpts; prompts forbid hallucinating citations.

Model choice (**GPT-4o-mini**, temperature 0) trades **cost/latency** against reasoning depth; deterministic decoding aids reproducibility for demos and papers.

### 5.5 Why not fine-tune an end-to-end matcher?

Reasons aligned with regulatory tooling:

- **Traceability**: answers cite filenames and excerpts auditors can open.
- **Updateability**: new guidance only requires **re-embedding**, not retraining.
- **Data scarcity**: labeled recall→guidance pairs are expensive to curate; RAG works with **zero-shot** pairing given a good corpus.

Fine-tuning or adapters remain attractive **after** you accumulate labeled pairs (see future work).

## 6. Evaluation suggestions (for “good enough” before GitHub)

- **Manual**: experts score top-\(k\) chunks for \(N\) incidents (precision@\(k\), MRR).
- **Semi-automatic**: if you collect URLs or filenames deemed relevant for each incident, compute overlap with retrieved sets.
- **Ablation**: disable lexical rerank or branch filter and measure degradation—supports claims in the paper.

## 7. Limitations

- Scraped guidance depends on FDA site HTML stability.
- OCR-scanned PDFs may produce poor text extraction.
- LLM summaries can **drop rare terms**; keeping an optional “raw incident + summary” concatenation for retrieval is a possible mitigation.
- MAUDE narratives vary widely in structure; flattening loses hierarchy.

## 8. Future work

1. **Learned reranker**: train on (query, chunk, relevance) triples from curator labels or clicks.
2. **Structured CFR graph**: combine dense retrieval over guidance with symbolic links (e.g., 21 CFR 820 sections) for deterministic citations.
3. **Temporal versioning**: index `(document_id, effective_date)` and retrieve latest superseded-aware answers.
4. **Calibration**: score thresholds for “no sufficient guidance context” to reduce overconfident prose.
5. **Multimodal**: diagrams in guidance via captioning models.
6. **Privacy**: separate on-prem embeddings for proprietary SOP text vs public FDA corpus.

---

*This document is meant to accompany the codebase under `fda-search-app/` and can be cited as technical documentation or expanded into a formal publication.*
