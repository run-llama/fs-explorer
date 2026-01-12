from typer import Typer, Option
from typing import Annotated
from .utils import cache_texts


app = Typer()


@app.command()
def main(
    directory: Annotated[
        str | None,
        Option(
            "-d",
            "--directory",
            help="Base directory where the arXiv papers and metadata are stored. Defaults to the current working directory.",
        ),
    ] = None,
    cache_directory: Annotated[
        str | None,
        Option(
            "-c",
            "--cache-dir",
            help="Sub-directory where to persistently cache arXiv paper texts. Defaults to `tmp/cache`.",
        ),
    ] = None,
    texts_directory: Annotated[
        str | None,
        Option(
            "-t",
            "--texts-dir",
            help="Sub-directory where the texts from the arXiv papers summaries are stored. Defaults to `texts`.",
        ),
    ] = None,
) -> None:
    cache_texts(directory, cache_directory, texts_directory)
