import os
import asyncio
import logging

from typing import cast
from diskcache import Cache
from pathlib import Path
from llama_cloud_services.parse.utils import ResultType
from llama_cloud_services.parse.types import JobResult
from llama_cloud_services import LlamaParse

CACHING_DIR = Path("tmp/cache")


class ParsedFileCache:
    def __init__(self) -> None:
        self._cache = Cache(directory=str(CACHING_DIR))
        self._is_warmed_up = CACHING_DIR.is_dir()

    def warmup(self) -> None:
        if not self._is_warmed_up:
            os.makedirs(CACHING_DIR, exist_ok=True)
            self._is_warmed_up = True
        return None

    @property
    def is_empty(self) -> bool:
        return len(list(self._cache.iterkeys())) == 0

    def add_file(self, file_path: str, content: str) -> None:
        resolved_path = str(Path(file_path).resolve())
        self._cache.add(resolved_path, content)

    def get_file(self, file_path: str) -> str | None:
        resolved_path = str(Path(file_path).resolve())
        return cast(str | None, self._cache.get(resolved_path))

    def close(self) -> None:
        self._cache.close()


CACHE = ParsedFileCache()


async def parse_and_cache(directory: str, recursive: bool, to_skip: list[str]) -> None:
    logging.basicConfig(
        filename="fs-explorer.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    CACHE.warmup()
    dir_path = Path(directory)
    to_skip_resolved = [str((dir_path / path).resolve()) for path in to_skip]
    if not recursive:
        files = []
        fls = os.listdir(dir_path)
        for fl in fls:
            resolved = str(Path(dir_path / fl).resolve())
            if resolved not in to_skip_resolved:
                files.append(resolved)
    else:
        files = []
        for root, dirs, fls in os.walk(dir_path):
            dirs[:] = [
                str((Path(root) / d).resolve())
                for d in dirs
                if d not in to_skip_resolved
            ]  # type: ignore[invalid-assignment]
            fls[:] = [
                str((Path(root) / f).resolve())
                for f in fls
                if f not in to_skip_resolved
            ]  # type: ignore[invalid-assignment]
            for fl in fls:
                files.append(str((Path(root) / fl).resolve()))
    semaphore = asyncio.Semaphore(5)
    parser = LlamaParse(
        api_key=cast(str, os.getenv("LLAMA_CLOUD_API_KEY")),
        result_type=ResultType.TXT,
        fast_mode=True,
    )

    async def parse_job(file_path: str) -> None:
        async with semaphore:
            result = cast(JobResult, await parser.aparse(file_path=file_path))
            if result.error is None:
                text = await result.aget_text()
                CACHE.add_file(file_path, text)
            else:
                logging.info(
                    f"Could not parse file {file_path} because of {result.error}"
                )

    await asyncio.gather(*(parse_job(file) for file in files))
