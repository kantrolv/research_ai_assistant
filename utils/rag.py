"""
utils/rag.py — RAG pipeline: Chunking, Embedding, FAISS Vector Store, Retrieval

Implements the Retrieval-Augmented Generation pipeline:
    1. Chunk scraped web content into smaller pieces
    2. Embed chunks using all-MiniLM-L6-v2 (runs locally, free)
    3. Store embeddings in FAISS (in-memory, no disk persistence)
    4. Retrieve top-k relevant chunks via similarity search

The query vector stays in RAM only — it is NOT stored in the vector store.

Functions:
    - get_embeddings()                      → Load embedding model
    - chunk_documents(search_results)       → Split content into chunks with metadata
    - build_vectorstore(chunks)             → Create FAISS index from chunks
    - retrieve_relevant_chunks(vs, query)   → Similarity search, return top-k
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K_RETRIEVAL


# Cache the embedding model to avoid reloading on every call
_embeddings_cache = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the all-MiniLM-L6-v2 embedding model.
    Cached after first load to avoid redundant downloads.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_cache


def chunk_documents(search_results: list[dict]) -> list[Document]:
    """
    Take scraped search results and split content into smaller chunks.
    Each chunk preserves metadata (title, url) for proper citation.

    Args:
        search_results: List of dicts with keys: title, url, snippet, content

    Returns:
        List of LangChain Document objects (chunked, with metadata)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    for result in search_results:
        content = result.get("content", "")
        if not content or len(content.strip()) < 20:
            continue  # Skip empty or too-short content

        doc = Document(
            page_content=content,
            metadata={
                "title": result.get("title", "Unknown"),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
            },
        )
        documents.append(doc)

    if not documents:
        return []

    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Create an in-memory FAISS vector store from document chunks.

    Args:
        chunks: List of LangChain Document objects to embed and index

    Returns:
        FAISS vector store ready for similarity search
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_relevant_chunks(
    vectorstore: FAISS, query: str, k: int = TOP_K_RETRIEVAL
) -> list[Document]:
    """
    Embed the query (in RAM only — NOT stored in the DB)
    and find the top-k most similar chunks via similarity search.

    Args:
        vectorstore: FAISS vector store to search
        query: The search query to find relevant chunks for
        k: Number of top results to return

    Returns:
        List of the k most relevant Document chunks
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
