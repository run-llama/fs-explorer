import time

from typing import TypedDict
from fs_explorer.workflow import (
    workflow,
    InputEvent,
    ToolCallEvent,
    ExplorationEndEvent,
)
from qdrant_client import AsyncQdrantClient
from rag_starterkit.pipeline import Pipeline
from ._templating import Template

FS_EXPLORER_PROMPT = Template(
    "Search the answer to the following question: '{{question}}' by using one of the PDF files available in the current directory. In your final response, you must report the answer to the question. In this task, you MUST NOT ask for any human assistance and you MUST ONLY use tool calling."
)


class RunResult(TypedDict):
    time_taken: float
    tool_calls: list[str] | None
    error: str | None
    final_answer: str | None
    file_path: str | list[str] | None


async def run_workflow(question: str) -> RunResult:
    start_event = InputEvent(task=FS_EXPLORER_PROMPT.render({"question": question}))
    tool_calls = []
    file_names: list[str] = []
    start_time = time.time()
    handler = workflow.run(start_event=start_event)
    async for event in handler.stream_events():
        if isinstance(event, ToolCallEvent):
            tool_calls.append(event.tool_name)
            if event.tool_name == "parse_file":
                file_name = event.tool_input.get("file_path")
                if file_name is not None:
                    file_names.append(file_name)
    result = await handler
    end_time = time.time()
    assert isinstance(result, ExplorationEndEvent)
    return RunResult(
        time_taken=(end_time - start_time),
        tool_calls=tool_calls,
        error=result.error,
        final_answer=result.final_result,
        file_path=file_names,
    )


PIPELINE = Pipeline(
    qdrant_client=AsyncQdrantClient(location="http://localhost:6333"),
    qdrant_collection_name="rag-benchmark",
    cache_directory="tmp/cache",
)


async def run_pipeline(question: str) -> RunResult:
    await PIPELINE.prepare()
    start_time = time.time()
    try:
        result, file_path = await PIPELINE.run(question)
        error = None
    except Exception as e:
        file_path = None
        result = None
        error = str(e)
    end_time = time.time()
    return RunResult(
        time_taken=(end_time - start_time),
        tool_calls=None,
        error=error,
        final_answer=result,
        file_path=file_path,
    )
