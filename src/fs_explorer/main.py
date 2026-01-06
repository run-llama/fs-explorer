import json
import asyncio

from typer import Typer, Option
from typing import Annotated
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console

from .workflow import (
    workflow,
    InputEvent,
    ToolCallEvent,
    GoDeeperEvent,
    AskHumanEvent,
    HumanAnswerEvent,
)
from .caching import parse_and_cache, CACHE

app = Typer()


async def run_workflow(task: str):
    console = Console()
    handler = workflow.run(start_event=InputEvent(task=task))
    with console.status(status="Working on your request...") as status:
        async for event in handler.stream_events():
            if isinstance(event, ToolCallEvent):
                status.update("Tool calling...")
                content = f"Calling tool `{event.tool_name}` with input:\n\n```\n{json.dumps(event.tool_input, indent=2)}\n```\n\nThe tool call is motivated by: {event.reason}"
                panel = Panel(
                    Markdown(content),
                    title_align="left",
                    title="Tool Call",
                    border_style="bold yellow",
                )
                console.print(panel)
                status.update("Working on the next move...")
            elif isinstance(event, GoDeeperEvent):
                status.update("Going deeper into the filesystem...")
                content = f"Going to directory: `{event.directory}` because of: {event.reason}"
                panel = Panel(
                    Markdown(content),
                    title_align="left",
                    title="Moving within the file system",
                    border_style="bold magenta",
                )
                console.print(panel)
                status.update("Working on the next move...")
            elif isinstance(event, AskHumanEvent):
                status.stop()
                console.print()
                answer = console.input(
                    f"[bold cyan]Human response required[/]\n[bold]Question:[/]\n{event.question}\n[bold]Reason for asking[/]\n{event.reason}\n[bold cyan]Your answer:[/] "
                )
                while answer.strip() == "":
                    console.print("[bold red]You need to provide an answer[/]\n")
                    answer = console.input(
                        f"[bold cyan]Human response required[/]\n[bold]Question:[/]\n{event.question}\n[bold]Reason for asking[/]\n{event.reason}\n[bold cyan]Your answer:[/] "
                    )
                handler.ctx.send_event(HumanAnswerEvent(response=answer.strip()))
                console.print()
                status.start()
                status.update("Working on your request...")
        result = await handler
        status.update("Gathering the final result...")
        await asyncio.sleep(0.1)
        content = result.final_result
        panel = Panel(
            Markdown(content),
            title_align="left",
            title="Final result",
            border_style="bold green",
        )
        console.print(panel)
        status.stop()
    return None


@app.command(
    name="run",
    help="Run the exploration with a specific task",
)
def main(
    task: Annotated[
        str,
        Option(
            "--task",
            "-t",
            help="Task that the FsExplorer Agent has to perform while exploring the current directory.",
        ),
    ],
) -> None:
    asyncio.run(run_workflow(task))

@app.command(
    name="load-cache",
    help="Parse all the files in a directory at once (also recursively) and add them to a persistent cache for faster retrieval at agent runtime",
)
def load_cache(
    directory: Annotated[
        str,
        Option(
            "--directory",
            "-d",
            help="Directory containing the files to parse and load to cache. Defaults to current working directory.",
        ),
    ] = ".",
    recursive: Annotated[
        bool,
        Option(
            "--recursive/--no-recursive",
            "-r",
            help="Find files recursively within the target directory",
            is_flag=True
        )
    ] = False,
    to_skip: Annotated[ 
        list[str],
        Option(
            "--skip",
            "-s",
            help="Skip one or more directories or files within the target directory. The path should be relative to the target directory (e.g. `testfile.txt` and not `data/testfile.txt` if `data` is the target directory). Can be used multiple times. Defaults to an empty list. Used only if `--recursive` is set."
        )
    ] = [],
) -> None:
    asyncio.run(parse_and_cache(directory, recursive, to_skip))

@app.command(
    name="get-cached",
    help="Get the content of a cached file, if it exists",
)
def get_cached(
    file: Annotated[ 
        str,
        Option(
            "--file",
            "-f",
            help="The cached file whose content should be retrieved",
        )
    ],
    max_chars: Annotated[ 
        int,
        Option(
            "--max",
            "-m",
            help="Max charachters to display. Defaults to 10.000"
        )
    ] = 10000,
) -> None:
    content = CACHE.get_file(file)
    console = Console()
    if content is not None:
        content = content[:max_chars]+"\n\nCONTINUES..." if len(content) > max_chars else content
        markdown = Markdown(content)
        panel = Panel(
            markdown,
            title_align="left",
            title=f"Content for {file}",
            border_style="bold",
        )
        console.print(panel)
    else:
        console.print(f"[bold yellow]No cached content for {file}[/]")