# MedicalDeviceReg-intel
**AI-based regulatory intelligence for medical devices**

**AI-powered global regulatory intelligence for medical devices**


**Overview**:
MedReg Intel uses natural language processing and machine learning to compare medical device regulations across global markets, including FDA, EU MDR/IVDR, and other regions.

**Goal**:
To identify similarities and differences between regulatory frameworks and support global regulatory strategy.

**Scope**:
FDA 21 CFR Part 820 / QMSR
EU MDR and IVDR
Future: NMPA (China), TGA (Australia)

**Approach**:
Text preprocessing and structuring
Embedding-based similarity analysis
Cross-framework comparison

**Disclaimer**:
For research and educational purposes only. Not regulatory advice.

---

## Code in this repository

The runnable **FDA guidance search and device incident matcher** lives in [`fda-search-app/`](fda-search-app/):

- **Streamlit UI**: `streamlit run fda-search-app/app.py` (see [`fda-search-app/README.md`](fda-search-app/README.md) for env vars and deployment).
- **FastAPI** (SOP / incident API): `uvicorn web_server:app --app-dir fda-search-app`.
- **Methods & data pipeline** (RAG, OpenFDA, updates): [`fda-search-app/docs/METHOD_AND_DATA_PIPELINE.md`](fda-search-app/docs/METHOD_AND_DATA_PIPELINE.md).

Guidance PDFs for indexing are under `fda-search-app/fda_docs/` (or fetch via the app’s Live updates tab).
