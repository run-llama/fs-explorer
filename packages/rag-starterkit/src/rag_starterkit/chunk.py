from chonkie import SentenceChunker, Chunk
from typing import TypedDict
from fastembed import SparseEmbedding


class ChunkWithMetadata(TypedDict):
    chunk: Chunk
    file_path: str
    embedding: list[float]
    sparse_embedding: SparseEmbedding | None


class Chunker:
    def __init__(self) -> None:
        self._chunker = SentenceChunker(
            chunk_overlap=200,  # allow 10% chunk size overlap
            chunk_size=2048,
        )

    def chunk_texts(self, contents: dict[str, str]) -> list[ChunkWithMetadata]:
        texts = list(contents.values())
        files = list(contents.keys())
        batch_chunks = self._chunker.chunk_batch(texts=texts)
        chunks_w_meta: list[ChunkWithMetadata] = []
        for i, batch_chunk in enumerate(batch_chunks):
            for chunk in batch_chunk:
                chunks_w_meta.append(
                    ChunkWithMetadata(
                        chunk=chunk,
                        file_path=files[i],
                        embedding=[],
                        sparse_embedding=None,
                    )
                )
        return chunks_w_meta
