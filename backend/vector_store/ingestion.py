import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.vector_store.chroma_store import get_chroma_store


def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def ingest_pdf(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = create_text_splitter()
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source_type"] = "pdf"
        chunk.metadata["file_name"] = os.path.basename(file_path)
    return get_chroma_store().add_documents(chunks)


def ingest_url(url: str) -> List[str]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = create_text_splitter()
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["source_type"] = "web"
        chunk.metadata["url"] = url
    return get_chroma_store().add_documents(chunks)


def ingest_text(text: str, metadata: dict = None) -> List[str]:
    doc = Document(page_content=text, metadata=metadata or {"source_type": "raw_text"})
    splitter = create_text_splitter()
    chunks = splitter.split_documents([doc])
    return get_chroma_store().add_documents(chunks)