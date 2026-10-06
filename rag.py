"""Core RAG logic using Ollama (local embeddings + LLM) and ChromaDB."""
import io
import uuid
from typing import Iterator

import chromadb
import ollama
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"   # ollama pull nomic-embed-text
LLM_MODEL = "llama3.2"             # ollama pull llama3.2
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

_db = chromadb.PersistentClient(path="./chroma_store")
_col = _db.get_or_create_collection("campus_docs", metadata={"hnsw:space": "cosine"})

SYSTEM_PROMPT = (
    "You are the Campus Knowledge OS assistant for students and staff. "
    "Answer ONLY from the numbered context passages. Cite them like [1] or [2]. "
    "If the answer is not in the context, say you could not find it in the campus "
    "documents and suggest which office to contact. Keep answers short and clear."
)


def _embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 40]


def ingest(filename: str, data: bytes, category: str) -> int:
    chunks = chunk_text(extract_text(filename, data))
    if not chunks:
        return 0
    _col.delete(where={"source": filename})  # re-uploading replaces the old version
    _col.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=chunks,
        embeddings=_embed(chunks),
        metadatas=[{"source": filename, "category": category, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def list_sources() -> list[dict]:
    summary: dict[str, dict] = {}
    for m in _col.get(include=["metadatas"])["metadatas"]:
        row = summary.setdefault(m["source"], {"source": m["source"], "category": m["category"], "chunks": 0})
        row["chunks"] += 1
    return sorted(summary.values(), key=lambda r: r["source"])


def delete_source(source: str) -> None:
    _col.delete(where={"source": source})


def retrieve(question: str, category: str | None = None) -> list[dict]:
    if _col.count() == 0:
        return []
    res = _col.query(
        query_embeddings=_embed([question]),
        n_results=TOP_K,
        where={"category": category} if category else None,
    )
    return [
        {"n": i + 1, "source": m["source"], "category": m["category"], "text": d}
        for i, (d, m) in enumerate(zip(res["documents"][0], res["metadatas"][0]))
    ]


def stream_answer(question: str, passages: list[dict]) -> Iterator[str]:
    context = "\n\n".join(f"[{p['n']}] ({p['source']}) {p['text']}" for p in passages)
    stream = ollama.chat(
        model=LLM_MODEL,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    for chunk in stream:
        yield chunk["message"]["content"]
