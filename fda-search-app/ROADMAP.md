# Roadmap: Multi-Regulator Comparison & Gap Analysis

**Goal:** Evolve this app from FDA-only search into a showcase that demonstrates regulatory + technical skills: understand FDA → add EU/Japan → compare authorities → perform gap analysis → expand to other markets.

---

## Phase 1: Solidify FDA foundation (current app)

- [ ] **1.1** Ensure FDA guidance search and protocol checker work end-to-end locally and in deployment (Render/Docker).
- [ ] **1.2** Add a small “About / Vision” section in the app and README: position as “first step toward multi-regulator comparison and gap analysis.”
- [ ] **1.3** In the protocol checker output, add a note: *“This analysis is FDA-centric. For EU/Japan submissions, run gap analysis once those authorities are available.”*
- [ ] **1.4** Add a **Compare** tab (placeholder): e.g. “Compare FDA vs EMA vs PMDA (coming next)” so the direction is visible to hiring managers.

**Outcome:** Current app is polished and clearly positioned for what comes next.

---

## Phase 2: Add EU (EMA/EC) and Japan (PMDA/MHLW) sources

- [ ] **2.1** Research and document official guidance sources:
  - EU: EMA guidelines, EC regulations/directives (e.g. EUR-Lex, EMA website).
  - Japan: PMDA guidelines, MHLW notifications (English where available).
- [ ] **2.2** Implement fetchers/loaders for at least one EU source and one Japan source (similar pattern to `fda_fetcher.py`):
  - e.g. `ema_fetcher.py`, `pmda_fetcher.py` (or unified `regulatory_fetcher.py` with authority-specific adapters).
- [ ] **2.3** Define a common document format (e.g. `RegulatoryDoc`: title, authority, source_url, text, maybe topic/category) so all authorities feed the same pipeline.
- [ ] **2.4** Add folders or labels: `ema_docs/`, `pmda_docs/` (or single `guidance_library/` with authority in metadata).
- [ ] **2.5** In the app: either separate “libraries” per authority or one unified index with authority filter (e.g. “Search: FDA only / EMA only / PMDA only / All”).

**Outcome:** Users can search and retrieve guidance from FDA, EU, and Japan in one place.

---

## Phase 3: Compare view (same topic across FDA / EMA / PMDA)

- [ ] **3.1** Define “topics” or “themes” (e.g. clinical trial design, CMC, safety, labeling). Optionally tag guidance by topic (manual list or LLM-assisted).
- [ ] **3.2** Build a **Compare** feature: user picks a topic (or enters a question); system retrieves relevant chunks from each authority and shows side-by-side (e.g. three columns: FDA | EMA | PMDA) with citations.
- [ ] **3.3** Use one vector store per authority (or one with strong authority filter) so comparison is explicit and cited.
- [ ] **3.4** Add a simple “Comparison” tab in the app: topic selector + “Compare” button → display side-by-side excerpts and links to source docs.

**Outcome:** Hiring managers see “compare how regulators differ” in action.

---

## Phase 4: Gap analysis (“What do we need for EU/Japan given our FDA approach?”)

- [ ] **4.1** Define gap analysis workflow:
  - Input: FDA-centric protocol/summary (or existing protocol checker output).
  - Output: structured list of “gaps” per authority (EMA, PMDA): what’s required or different, with citations to their guidance.
- [ ] **4.2** Implement gap analysis logic:
  - Use FDA-oriented summary to query EU and Japan vector stores; LLM summarizes “additional or different requirements” and lists specific guidance/sections.
- [ ] **4.3** Add **Gap analysis** tab: upload protocol (or reuse protocol summary) → select target authorities (e.g. EU, Japan) → show checklist of gaps with source links.
- [ ] **4.4** Format output for submission use: e.g. “For EMA: cite these guidelines; for PMDA: these; here are the gaps to address.”

**Outcome:** The app delivers the core value: “fulfill requirements of other regulators” with traceability.

---

## Phase 5: Polish for hiring managers and portfolio

- [ ] **5.1** README: clear title (“Multi-Regulator Guidance Comparison & Gap Analysis”), 2–3 sentence value prop, and this roadmap (or link to ROADMAP.md).
- [ ] **5.2** Resume/LinkedIn one-liner: “Built a multi-regulator guidance comparison and gap analysis tool (FDA, EMA, PMDA) to support global submissions, with traceability to source guidance.”
- [ ] **5.3** Optional: short “About” or “How it works” in the app (1–2 paragraphs) for visitors who open the demo.
- [ ] **5.4** Optional: add 1–2 example “topic” comparisons or gap reports in the repo (e.g. in `/docs` or README) so recruiters can see output quality without running the app.

**Outcome:** The project clearly communicates your unique RA + technical skills.

---

## Phase 6: Expand to other markets (later)

- [ ] **6.1** China (NMPA): identify English guidance sources; add fetcher and compare/gap support.
- [ ] **6.2** Middle East: identify key authorities (e.g. GCC, SFDA) and sources; add same pattern.
- [ ] **6.3** Keep architecture authority-agnostic (fetchers + common doc format + per-authority or filtered vector stores) so new regions are “add a fetcher + config.”

**Outcome:** Shows you think in terms of global expansion and scalable design.

---

## Summary checklist (high level)

| Step | Focus |
|------|--------|
| 1 | Solidify FDA app + positioning + “Compare” placeholder |
| 2 | Add EU and Japan guidance sources and search |
| 3 | Build compare view (same topic across FDA / EMA / PMDA) |
| 4 | Build gap analysis (FDA-based input → gaps for EU/Japan) |
| 5 | Polish README, copy, and portfolio narrative |
| 6 | (Later) Add China, Middle East, other markets |

You can treat Phases 1–4 as the core path to a strong showcase; 5 makes it visible to hiring managers; 6 shows scalability.
