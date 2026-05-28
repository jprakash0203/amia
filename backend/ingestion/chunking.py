from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def create_splitter(chunk_size: int = 1000, overlap: int = 200):
    return RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap = overlap,
        separators=["\n\n", "\n", ". ", " ", ""]

    )

def chunk_documents(docs: List[Document], **kwargs) -> List[Document]:
    splitter = create_splitter(**kwargs)
    chunks = splitter.split_documents(docs)
    print(f"  Chunked {len(docs)} documents → {len(chunks)} chunks")
    return chunks