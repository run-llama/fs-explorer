import os
import json
import asyncio

from openai import AsyncOpenAI
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, Field
from typing import TypedDict, cast
from .run import run_pipeline, run_workflow
from ._templating import Template


class EvalTask(TypedDict):
    question: str
    answer: str
    file: str


class LLMEvaluation(TypedDict):
    relevance: int
    correctness: int
    reason: str


class BestTime(TypedDict):
    fs_explorer: float
    rag: float


class FilePath(TypedDict):
    fs_explorer: list[str] | str | None
    rag: list[str] | str | None


class HasError(TypedDict):
    fs_explorer: bool
    rag: bool


class LLMEvaluations(TypedDict):
    fs_explorer: LLMEvaluation | None
    rag: LLMEvaluation | None


class Answers(TypedDict):
    fs_explorer: str | None
    rag: str | None


class EvalResult(TypedDict):
    task: EvalTask
    llm_evaluations: LLMEvaluations
    answers: Answers
    time_taken: BestTime
    used_files: FilePath
    tool_calls: list[str]
    has_error: HasError


class Evaluation(BaseModel):
    relevance: int = Field(
        description="Evaluation of the response, based on its relevance compared with the ground truth. Ranges between 0 and 10",
        ge=0,
        le=10,
    )
    correctness: int = Field(
        description="Evaluation of the response, based on its correctness compared with the ground truth. Ranges between 0 and 10",
        ge=0,
        le=10,
    )
    reason: str = Field(description="Brief explanation of the evaluation")

    def to_llm_evaluation(self) -> LLMEvaluation:
        return LLMEvaluation(**self.model_dump())


LLM_AS_A_JUDGE_PROMPT = Template(
    "The following question: '{{question}}' has this ground truth answer: '{{ground_truth}}'. Please evaluate this answer: '{{answer}}' grading its correctness and relevance between 0 and 10, and providing a brief explanation of the evaluation."
)
LLM_AS_A_JUDGE_MODEL = "gpt-5.2"


async def llm_as_a_judge(
    question: str, ground_truth: str, produced_answer: str
) -> Evaluation | None:
    content = LLM_AS_A_JUDGE_PROMPT.render(
        {"question": question, "ground_truth": ground_truth, "answer": produced_answer}
    )
    message = EasyInputMessageParam(content=content, role="user")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.responses.parse(
        text_format=Evaluation,
        input=[message],
        reasoning=Reasoning(effort="none"),
        model=LLM_AS_A_JUDGE_MODEL,
    )
    return response.output_parsed


def get_evaluation_dataset(dataset_file: str) -> list[EvalTask]:
    with open(dataset_file, "r") as f:
        data = json.load(f)
    assert isinstance(data, list)
    eval_tasks: list[EvalTask] = []
    for d in data:
        assert isinstance(d, dict)
        assert "question" in d
        assert "answer" in d
        assert "file" in d
        eval_tasks.append(cast(EvalTask, d))
    return eval_tasks


async def run_evaluation(dataset_file: str, results_file: str = "results.json") -> None:
    tasks = get_evaluation_dataset(dataset_file)
    results: list[EvalResult] = []
    try:
        for i, task in enumerate(tasks):
            print(f"Starting task {i + 1} of {len(tasks)}")
            wf_result = await run_workflow(question=task["question"])
            pipeline_result = await run_pipeline(question=task["question"])
            best_time = BestTime(
                fs_explorer=wf_result["time_taken"], rag=pipeline_result["time_taken"]
            )
            file_check = FilePath(
                fs_explorer=wf_result["file_path"], rag=pipeline_result["file_path"]
            )
            tool_calls = wf_result["tool_calls"] or []
            has_error = HasError(fs_explorer=True, rag=True)
            if wf_result["error"] is None:
                has_error["fs_explorer"] = False
                if wf_result["final_answer"] is not None:
                    wf_evaluation = await llm_as_a_judge(
                        question=task["question"],
                        ground_truth=task["answer"],
                        produced_answer=wf_result["final_answer"],
                    )
                    if wf_evaluation is not None:
                        wf_evaluation = wf_evaluation.to_llm_evaluation()
                else:
                    wf_evaluation = None
            else:
                wf_evaluation = None
            if pipeline_result["error"] is None:
                has_error["rag"] = False
                if pipeline_result["final_answer"] is not None:
                    pipeline_evaluation = await llm_as_a_judge(
                        question=task["question"],
                        ground_truth=task["answer"],
                        produced_answer=pipeline_result["final_answer"],
                    )
                    if pipeline_evaluation is not None:
                        pipeline_evaluation = pipeline_evaluation.to_llm_evaluation()
                else:
                    pipeline_evaluation = None
            else:
                pipeline_evaluation = None
            llm_evaluations = LLMEvaluations(
                fs_explorer=wf_evaluation, rag=pipeline_evaluation
            )
            eval_result = EvalResult(
                task=task,
                tool_calls=tool_calls,
                llm_evaluations=llm_evaluations,
                used_files=file_check,
                has_error=has_error,
                time_taken=best_time,
                answers=Answers(
                    fs_explorer=wf_result["final_answer"],
                    rag=pipeline_result["final_answer"],
                ),
            )
            results.append(eval_result)
            print(
                f"Finished task {i + 1} of {len(tasks)}, sleeping 1 sec to avoid rate limiting issues..."
            )
            await asyncio.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
