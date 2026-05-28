

import os
import datetime
from typing import List
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from backend.ingestion.chunking import chunk_documents
from backend.vector_store.chroma_store import get_chroma_store


def ingest_pdf(file_path: str) -> List[str]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    print(f"Ingesting PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")

    chunks = chunk_documents(pages)

    for chunk in chunks:
        chunk.metadata.update({
            "source_type": "pdf",
            "file_name": os.path.basename(file_path),
            "ingested_at": datetime.datetime.utcnow().isoformat(),
        })

    store = get_chroma_store()
    ids = store.add_documents(chunks)
    print(f"  ✅ Stored {len(ids)} chunks in ChromaDB")
    return ids


def ingest_url(url: str) -> List[str]:
    
    print(f"Ingesting URL: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    chunks = chunk_documents(docs)

    for chunk in chunks:
        chunk.metadata.update({"source_type": "web", "url": url})

    ids = get_chroma_store().add_documents(chunks)
    print(f"  ✅ Stored {len(ids)} chunks from {url}")
    return ids


def ingest_text(text: str, metadata: dict = None) -> List[str]:
    """Ingest raw text directly into ChromaDB."""
    doc = Document(page_content=text, metadata=metadata or {"source_type": "raw"})
    chunks = chunk_documents([doc])
    return get_chroma_store().add_documents(chunks)