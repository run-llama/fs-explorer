from diskcache import Cache
from pathlib import Path

BASE_DIRECTORY = "."
CACHE_DIRECTORY = "tmp/cache"
TEXTS_DIRECTORY = "texts"


def cache_texts(
    base_dir: str | None = None,
    cache_directory: str | None = None,
    texts_directory: str | None = None,
) -> None:
    cache = Cache(directory=cache_directory or CACHE_DIRECTORY)
    files_dir = Path((base_dir or BASE_DIRECTORY)) / (
        texts_directory or TEXTS_DIRECTORY
    )
    for root, _, files in files_dir.walk():
        for file in files:
            path = root / file
            text = path.read_text()
            cache.add(str(path.resolve()), text)
    return None
