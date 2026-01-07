import os
import inspect

from qdrant_client import AsyncQdrantClient
from typing import Any, cast
from .parse import parse_directory, contents_from_cache
from .chunk import Chunker
from .embed import Embedder
from .vectordb import VectorDB
from .llm_filter import LLMFilter


class Pipeline:
    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_collection_name: str,
        rrf_constant: int = 60,
        parsing_kwargs: dict[str, Any] | None = None,
        cache_directory: str | None = None,
        openai_api_key: str | None = None,
        openai_emebdding_model: str | None = None,
        fastembed_model: str | None = None,
        openai_llm_model: str | None = None,
    ):
        if cache_directory is None and parsing_kwargs is None:
            raise ValueError(
                "At least one between parsing_kwargs and cache_directory has to be provided"
            )
        self.parsing_strategy = (
            parse_directory if cache_directory is None else contents_from_cache
        )
        self.parsing_kwargs = (
            parsing_kwargs
            if cache_directory is None
            else {"cache_directory": cache_directory}
        )
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY must be set within the environment if openai_api_key is not provided as argument"
            )
        self.chunker = Chunker()
        self.embedder = Embedder(
            api_key=openai_api_key,
            openai_model=openai_emebdding_model,
            fastembed_model=fastembed_model,
        )
        self.vector_db = VectorDB(
            qdrant_client=qdrant_client,
            collection_name=qdrant_collection_name,
            embedder=self.embedder,
            rrf_constant=rrf_constant,
        )
        self.filter_llm = LLMFilter(api_key=openai_api_key, model=openai_llm_model)
        self.file_paths: list[str] = []
        self.is_ready = False

    async def prepare(self) -> None:
        if not self.is_ready and not (await self.vector_db.check_if_loaded()):
            if inspect.iscoroutinefunction(self.parsing_strategy):
                assert self.parsing_kwargs is not None, "parsing_kwargs cannot be null"
                contents = await self.parsing_strategy(**self.parsing_kwargs)
            else:
                contents = self.parsing_strategy(**self.parsing_kwargs)  # type: ignore
            contents = cast(dict[str, str], contents)
            self.file_paths = [key for key in contents]
            chunks = self.chunker.chunk_texts(contents)
            chunks = await self.embedder.embed_chunks(chunks)
            chunks = self.embedder.sparse_embed_chunks(chunks)
            await self.vector_db.configure_collection()
            await self.vector_db.upload(chunks)
        self.is_ready = True

    async def run(self, query: str, limit: int = 1) -> tuple[str | None, str | None]:
        if not self.is_ready:
            raise ValueError("Pipeline has not been prepared before running")
        filter_file = await self.filter_llm.generate_filter(query, self.file_paths)
        file_path = (
            filter_file.file_path
            if filter_file is not None and filter_file.confidence > 50
            else None
        )
        results = await self.vector_db.search(query, file_path=file_path, limit=limit)
        context = "\n\n".join([result["content"] for result in results])
        response = await self.filter_llm.generate_response(query, context)
        return response.response if response is not None else None, file_path
