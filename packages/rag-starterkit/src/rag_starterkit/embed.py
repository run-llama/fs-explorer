from openai import AsyncOpenAI
from fastembed import SparseTextEmbedding, SparseEmbedding

from .chunk import ChunkWithMetadata

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_FASTEMBED_MODEL = "Qdrant/bm25"


class Embedder:
    def __init__(
        self,
        api_key: str,
        openai_model: str | None = None,
        fastembed_model: str | None = None,
    ):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = openai_model or DEFAULT_EMBEDDING_MODEL
        self._sparse_embedder = SparseTextEmbedding(
            model_name=(fastembed_model or DEFAULT_FASTEMBED_MODEL),
            cache_dir="tmp/fastembed",
        )

    async def embed_chunks(
        self, chunks: list[ChunkWithMetadata]
    ) -> list[ChunkWithMetadata]:
        texts = [chunk["chunk"].text for chunk in chunks]
        embeddings = await self._client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=768,
        )
        for i, embedding in enumerate(embeddings.data):
            chunks[i]["embedding"] = embedding.embedding
        return chunks

    def sparse_embed_chunks(
        self, chunks: list[ChunkWithMetadata]
    ) -> list[ChunkWithMetadata]:
        texts = [chunk["chunk"].text for chunk in chunks]
        embeddings = list(self._sparse_embedder.embed(texts))
        for i, embedding in enumerate(embeddings):
            chunks[i]["sparse_embedding"] = embedding
        return chunks

    async def embed_query(self, query: str) -> list[float]:
        embeddings = await self._client.embeddings.create(
            input=query,
            model=self.model,
            dimensions=768,
        )
        return embeddings.data[0].embedding

    def sparse_embed_query(self, query: str) -> SparseEmbedding:
        embeddings = list(self._sparse_embedder.query_embed(query=query))
        return embeddings[0]
