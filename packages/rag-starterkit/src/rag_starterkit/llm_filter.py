from openai import AsyncOpenAI
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from typing import Any
from pydantic import BaseModel, Field

DEFAULT_OPENAI_MODEL = "gpt-4.1"
SYSTEM_PROMPT = """
Your task is to individuate, among the files that the user provides, the one that is most likely to have the answer to the user's query. Provide the file, and the confidence you have in your response (as an integer between 0 and 100).
"""


class FileFilter(BaseModel):
    file_path: str = Field(
        description="File path to filter for when searching an answer for the query"
    )
    confidence: int = Field(
        description="Confidence in your choice for the file_path field. Must be between 0 and 100",
        ge=0,
        le=100,
    )


class LLMFilter:
    def __init__(self, api_key: str, model: str | None = None):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model or DEFAULT_OPENAI_MODEL

    def _build_user_message(
        self, query: str, file_paths: list[str]
    ) -> EasyInputMessageParam:
        fls = "\n- ".join(file_paths)
        content = f"Find, among these files:\n\n- {fls}\n\nThe one that would be the most likely to contain the answer to this query: '{query}'"
        return EasyInputMessageParam(role="user", content=content, type="message")

    async def generate_filter(
        self, query: str, file_paths: list[str]
    ) -> FileFilter | None:
        messages: list[Any] = [
            EasyInputMessageParam(content=SYSTEM_PROMPT, role="system", type="message")
        ]
        messages.append(self._build_user_message(query, file_paths))
        response = await self._client.responses.parse(
            text_format=FileFilter,
            input=messages,
            model=self.model,
        )
        return response.output_parsed
