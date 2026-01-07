import logging
import asyncio
import os

from pathlib import Path
from typing import cast
from diskcache import Cache
from llama_cloud_services.parse.utils import ResultType
from llama_cloud_services.parse.types import JobResult
from llama_cloud_services import LlamaParse


async def parse_directory(
    directory: str, recursive: bool, to_skip: list[str]
) -> dict[str, str]:
    logging.basicConfig(
        filename="rag-starterkit.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
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
            ]
            fls[:] = [
                str((Path(root) / f).resolve())
                for f in fls
                if f not in to_skip_resolved
            ]
            for fl in fls:
                files.append(str((Path(root) / fl).resolve()))
    semaphore = asyncio.Semaphore(5)
    parser = LlamaParse(
        api_key=cast(str, os.getenv("LLAMA_CLOUD_API_KEY")),
        result_type=ResultType.TXT,
        fast_mode=True,
    )

    async def parse_job(file_path: str) -> tuple[str, str] | None:
        async with semaphore:
            result = cast(JobResult, await parser.aparse(file_path=file_path))
            if result.error is None:
                text = await result.aget_text()
                return file_path, text
            else:
                logging.info(
                    f"Could not parse file {file_path} because of {result.error}"
                )

    files_contents = await asyncio.gather(*(parse_job(file) for file in files))
    data: dict[str, str] = {}
    for el in files_contents:
        if el is not None:
            data[el[0]] = el[1]
    return data


def contents_from_cache(cache_directory: str = "tmp/cache") -> dict[str, str]:
    logging.basicConfig(
        filename="rag-starterkit.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cache = Cache(directory=cache_directory)
    data: dict[str, str] = {}
    cache_keys = list(cache.iterkeys())
    for key in cache_keys:
        if isinstance(key, str):
            data_key = key
        elif isinstance(key, bytes):
            data_key = str(key, encoding="utf-8")
        else:
            raise ValueError(
                "Cache keys shold be either strings or byte-encoded strings"
            )
        value = cache.get(data_key)
        if value is not None:
            data[data_key] = cast(str, value)
        else:
            logging.info(f"Skipping file {data_key} as it stored a null content")
    return data
