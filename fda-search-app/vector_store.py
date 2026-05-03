import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def build_embeddings():
    backend = (os.environ.get("EMBEDDING_BACKEND") or "openai").strip().lower()
    if backend in ("local", "huggingface", "hf"):
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model = os.environ.get(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        return HuggingFaceEmbeddings(model_name=model)
    return OpenAIEmbeddings()


def build_vector_store(documents: list[dict]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []

    for doc in documents:
        meta: dict = {"source": doc["name"]}
        for key in ("center_code", "regulatory_branch"):
            if key in doc and doc[key] is not None:
                meta[key] = doc[key]
        chunks = splitter.create_documents([doc["content"]], metadatas=[meta])
        all_chunks.extend(chunks)

    embeddings = build_embeddings()
    db = FAISS.from_documents(all_chunks, embeddings)
    return db
