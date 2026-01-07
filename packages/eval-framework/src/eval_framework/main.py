import asyncio

from typing import Annotated
from typer import Typer, Option
from .evaluate import run_evaluation
from .stats import get_eval_stats

app_eval = Typer()
app_stats = Typer()


@app_eval.command()
def run_evaluations(
    dataset_file: Annotated[
        str,
        Option(
            "-df",
            "--dataset-file",
            help="JSON file containing the questions and answers dataset",
        ),
    ],
    results_file: Annotated[
        str,
        Option(
            "-o",
            "--output",
            help="JSON file where to save the output of the evaluation.",
        ),
    ] = "results.json",
) -> None:
    asyncio.run(run_evaluation(dataset_file=dataset_file, results_file=results_file))


@app_stats.command()
def get_stats(
    results_file: Annotated[
        str,
        Option(
            "-rf",
            "--results-file",
            help="JSON file where the evaluation output has been saved.",
        ),
    ] = "results.json",
    stats_file: Annotated[
        str,
        Option(
            "-j",
            "--output-json",
            help="JSON file where the extracted statistics will be saved",
        ),
    ] = "statistics.json",
    report_file: Annotated[
        str,
        Option(
            "-r",
            "--output-report",
            help="Markdown file where a human-readable report on the extracted statistics will be saved",
        ),
    ] = "report.md",
) -> None:
    get_eval_stats(
        results_file=results_file,
        result_json_file=stats_file,
        result_md_file=report_file,
    )
