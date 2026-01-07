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
    def __init__(self, k: int = 60) -> None:
        """
        Args:
            k: Constant for RRF formula. Higher values reduce the impact of top-ranked items. Default of 60 is commonly used in literature.
        """
        self.k = k

    def _reciprocal_rank_fusion(
        self, dense_results: list[SearchResult], sparse_results: list[SearchResult]
    ) -> dict[str, float]:
        rrf_scores: dict[str, float] = {}
        for rank, result in enumerate(dense_results, start=1):
            content = result["content"]
            rrf_scores[content] = rrf_scores.get(content, 0.0) + 1 / (self.k + rank)
        for rank, result in enumerate(sparse_results, start=1):
            content = result["content"]
            rrf_scores[content] = rrf_scores.get(content, 0.0) + 1 / (self.k + rank)

        return rrf_scores

    def _dedupe_and_merge(
        self, dense_results: list[SearchResult], sparse_results: list[SearchResult]
    ) -> dict[str, SearchResult]:
        results_map: dict[str, SearchResult] = {}

        for result in dense_results:
            if result["content"] not in results_map:
                results_map[result["content"]] = result

        for result in sparse_results:
            if result["content"] not in results_map:
                results_map[result["content"]] = result

        return results_map

    def rerank(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        limit: int = 1,
    ) -> list[SearchResult]:
        rrf_scores = self._reciprocal_rank_fusion(dense_results, sparse_results)
        results_map = self._dedupe_and_merge(dense_results, sparse_results)
        reranked_results: list[SearchResult] = []
        for content, result in results_map.items():
            result_copy = result.copy()
            result_copy["score"] = rrf_scores[content]
            reranked_results.append(result_copy)
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results[:limit]


class VectorDB:
    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        collection_name: str,
        embedder: Embedder,
        rrf_constant: int = 60,
    ) -> None:
        self._client = qdrant_client
        self.collection_name = collection_name
        self.embedder = embedder
        self._reranker = SimpleReranker(k=rrf_constant)

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

    async def check_if_loaded(self) -> bool:
        if not await self._client.collection_exists(self.collection_name):
            return False
        collection = await self._client.get_collection(self.collection_name)
        return collection.points_count is not None and collection.points_count > 0

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
