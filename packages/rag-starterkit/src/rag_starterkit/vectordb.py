from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    SparseIndexParams,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)
from statistics import mean
from typing import TypedDict, Literal, cast


from .chunk import ChunkWithMetadata
from .embed import Embedder


class SearchResult(TypedDict):
    id: int
    content: str
    file_path: str
    score: float
    type: Literal["sparse", "dense"]


class SimpleReranker:
    def __init__(self) -> None:
        pass

    def _dedupe(
        self, dense_results: list[SearchResult], sparse_results: list[SearchResult]
    ) -> list[SearchResult]:
        dense_results_ranked = dense_results.copy()
        sparse_results_ranked = sparse_results.copy()
        dense_results_ranked.sort(key=lambda x: x["content"], reverse=False)
        sparse_results_ranked.sort(key=lambda x: x["content"], reverse=False)
        for i, r in enumerate(dense_results_ranked):
            r["score"] = i + 1
        for i, r in enumerate(sparse_results_ranked):
            r["score"] = i + 1
        for result in sparse_results_ranked:
            for i, r in enumerate(dense_results_ranked):
                if r["content"] == result["content"]:
                    r["score"] = mean([r["score"], result["score"]])
                    dense_results_ranked[i] = r
                    break
            else:
                dense_results_ranked.append(result)
        return dense_results_ranked

    def rerank(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        limit: int = 1,
    ) -> list[SearchResult]:
        results = self._dedupe(dense_results, sparse_results)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


class VectorDB:
    def __init__(
        self, qdrant_client: AsyncQdrantClient, collection_name: str, embedder: Embedder
    ) -> None:
        self._client = qdrant_client
        self.collection_name = collection_name
        self.embedder = embedder
        self._reranker = SimpleReranker()

    async def configure_collection(self) -> None:
        if await self._client.collection_exists(self.collection_name):
            return None
        else:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense-text": VectorParams(size=768, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse-text": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )

    async def upload(self, data: list[ChunkWithMetadata]) -> None:
        sparse_embeddings: list[dict[str, SparseVector]] = []
        dense_embeddings: list[dict[str, list[float]]] = []
        payloads: list[dict[str, str]] = []
        for d in data:
            assert d["sparse_embedding"] is not None
            sparse_embedding = {
                "sparse-text": SparseVector(
                    indices=d["sparse_embedding"].indices.tolist(),
                    values=d["sparse_embedding"].values.tolist(),
                )
            }
            dense_embedding = {"dense-text": d["embedding"]}
            payload = {"content": d["chunk"].text, "file_path": d["file_path"]}
            sparse_embeddings.append(sparse_embedding)
            dense_embeddings.append(dense_embedding)
            payloads.append(payload)
        self._client.upload_collection(
            self.collection_name,
            vectors=dense_embeddings,
            payload=payloads,
            ids=range(len(dense_embeddings)),
        )
        self._client.upload_collection(
            self.collection_name,
            vectors=sparse_embeddings,
            payload=payloads,
            ids=range(
                len(dense_embeddings), len(dense_embeddings) + len(sparse_embeddings)
            ),
        )

    async def search(
        self, query: str, file_path: str | None = None, limit: int = 1
    ) -> list[SearchResult]:
        dense_embedding = await self.embedder.embed_query(query)
        sparse_embedding = self.embedder.sparse_embed_query(query)
        if file_path:
            filt = Filter(
                must=FieldCondition(key="file_path", match=MatchValue(value=file_path))
            )
        else:
            filt = None
        result_dense = await self._client.query_points(
            collection_name=self.collection_name,
            query=dense_embedding,
            using="dense-text",
            query_filter=filt,
        )
        result_sparse = await self._client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist(),
            ),
            using="sparse-text",
            query_filter=filt,
        )
        dense_results: list[SearchResult] = []
        sparse_results: list[SearchResult] = []
        for point in result_dense.points:
            if point.payload is not None:
                result = SearchResult(
                    id=cast(int, point.id),
                    content=point.payload.get("content", ""),
                    file_path=point.payload.get("file_path", ""),
                    score=point.score,
                    type="dense",
                )
                dense_results.append(result)
        for point in result_sparse.points:
            if point.payload is not None:
                result = SearchResult(
                    id=cast(int, point.id),
                    content=point.payload.get("content", ""),
                    file_path=point.payload.get("file_path", ""),
                    score=point.score,
                    type="sparse",
                )
                sparse_results.append(result)
        return self._reranker.rerank(dense_results, sparse_results, limit)
