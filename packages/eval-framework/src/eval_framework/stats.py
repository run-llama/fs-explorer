import json

from statistics import mean
from typing import cast, TypedDict, Literal
from .evaluate import EvalResult, BestTime, LLMEvaluations

FrameworkType = Literal["rag", "fs-explorer"]


class TimeAverage(TypedDict):
    fs_explorer: float
    rag: float
    best: FrameworkType


class LLMAverage(TypedDict):
    correctness: float
    relevance: float


class LLMStats(TypedDict):
    fs_explorer: LLMAverage
    rag: LLMAverage
    best_correctness: FrameworkType
    best_relevance: FrameworkType


class EvalStats(TypedDict):
    time_stats: TimeAverage
    llm_stats: LLMStats


def get_results(results_file: str) -> list[EvalResult]:
    with open(results_file, "r") as f:
        data = json.load(f)
    assert isinstance(data, list)
    results: list[EvalResult] = []
    for d in data:
        assert isinstance(d, dict)
        assert "task" in d
        assert "llm_evaluations" in d
        assert "answers" in d
        assert "time_taken" in d
        assert "used_files" in d
        assert "tool_calls" in d
        assert "has_error" in d
        results.append(cast(EvalResult, d))
    return results


def get_time_average(time_stats: list[BestTime]) -> TimeAverage:
    fs_expl = []
    rag = []
    for time in time_stats:
        fs_expl.append(time["fs_explorer"])
        rag.append(time["rag"])
    fs_expl_mean = mean(fs_expl)
    rag_mean = mean(rag)
    best = "rag" if fs_expl_mean > rag_mean else "fs-explorer"
    return TimeAverage(fs_explorer=fs_expl_mean, rag=rag_mean, best=best)


def get_llm_stats(llm_stats: list[LLMEvaluations]) -> LLMStats:
    fs_expl_corr = []
    rag_corr = []
    fs_expl_rel = []
    rag_rel = []
    for stat in llm_stats:
        if stat["fs_explorer"] is not None:
            fs_expl_corr.append(stat["fs_explorer"]["correctness"])
            fs_expl_rel.append(stat["fs_explorer"]["relevance"])
        if stat["rag"] is not None:
            rag_corr.append(stat["rag"]["correctness"])
            rag_rel.append(stat["rag"]["relevance"])
    rag_corr_avg = mean(rag_corr)
    rag_rel_avg = mean(rag_rel)
    fs_expl_corr_avg = mean(fs_expl_corr)
    fs_expl_rel_avg = mean(fs_expl_rel)
    best_corr = "fs-explorer" if fs_expl_corr_avg > rag_corr_avg else "rag"
    best_rel = "fs-explorer" if fs_expl_rel_avg > rag_rel_avg else "rag"
    return LLMStats(
        fs_explorer=LLMAverage(correctness=fs_expl_corr_avg, relevance=fs_expl_rel_avg),
        rag=LLMAverage(correctness=rag_corr_avg, relevance=rag_rel_avg),
        best_correctness=best_corr,
        best_relevance=best_rel,
    )


def create_markdown_report(eval_stats: EvalStats, num_tasks: int) -> str:
    """Generate a markdown report from evaluation statistics."""
    time_stats = eval_stats["time_stats"]
    llm_stats = eval_stats["llm_stats"]

    # Helper function to format framework names
    def format_framework(name: str) -> str:
        return (
            "Agentic File Search with FileSystem Tools (fs-explorer)"
            if name == "fs-explorer"
            else "Traditional RAG"
        )

    # Helper function to add winner emoji
    def add_winner(framework: str, best: str) -> str:
        return (
            f"**{format_framework(framework)}**"
            if framework == best
            else format_framework(framework)
        )

    md = f"""# Evaluation Results Report

## Summary

Total tasks evaluated: **{num_tasks}**

---

## Time Performance

Average execution time across all tasks:

| Framework | Average Time (seconds) | Status |
|-----------|------------------------|--------|
| {add_winner("fs-explorer", time_stats["best"])} | {time_stats["fs_explorer"]:.2f}s | {"**Faster**" if time_stats["best"] == "fs-explorer" else ""} |
| {add_winner("rag", time_stats["best"])} | {time_stats["rag"]:.2f}s | {"**Faster**" if time_stats["best"] == "rag" else ""} |

**Winner:** {format_framework(time_stats["best"])} ({abs(time_stats["fs_explorer"] - time_stats["rag"]):.2f}s faster)

---

## LLM Evaluation Metrics

### Correctness Scores

| Framework | Average Score | Status |
|-----------|---------------|--------|
| {add_winner("fs-explorer", llm_stats["best_correctness"])} | {llm_stats["fs_explorer"]["correctness"]:.2f} | {"**Higher**" if llm_stats["best_correctness"] == "fs-explorer" else ""} |
| {add_winner("rag", llm_stats["best_correctness"])} | {llm_stats["rag"]["correctness"]:.2f} | {"**Higher**" if llm_stats["best_correctness"] == "rag" else ""} |

**Winner:** {format_framework(llm_stats["best_correctness"])} (+{abs(llm_stats["fs_explorer"]["correctness"] - llm_stats["rag"]["correctness"]):.2f} points)

### Relevance Scores

| Framework | Average Score | Status |
|-----------|---------------|--------|
| {add_winner("fs-explorer", llm_stats["best_relevance"])} | {llm_stats["fs_explorer"]["relevance"]:.2f} | {"**Higher**" if llm_stats["best_relevance"] == "fs-explorer" else ""} |
| {add_winner("rag", llm_stats["best_relevance"])} | {llm_stats["rag"]["relevance"]:.2f} | {"**Higher**" if llm_stats["best_relevance"] == "rag" else ""} |

**Winner:** {format_framework(llm_stats["best_relevance"])} (+{abs(llm_stats["fs_explorer"]["relevance"] - llm_stats["rag"]["relevance"]):.2f} points)

---

## Overall Comparison

| Metric | FS-Explorer | RAG | Winner |
|--------|-------------|-----|--------|
| **Speed** | {time_stats["fs_explorer"]:.2f}s | {time_stats["rag"]:.2f}s | {format_framework(time_stats["best"])} |
| **Correctness** | {llm_stats["fs_explorer"]["correctness"]:.2f} | {llm_stats["rag"]["correctness"]:.2f} | {format_framework(llm_stats["best_correctness"])} |
| **Relevance** | {llm_stats["fs_explorer"]["relevance"]:.2f} | {llm_stats["rag"]["relevance"]:.2f} | {format_framework(llm_stats["best_relevance"])} |

---

## Key Takeaways

- **Fastest Framework:** {format_framework(time_stats["best"])} is {abs(time_stats["fs_explorer"] - time_stats["rag"]):.2f}s faster on average
- **Most Correct:** {format_framework(llm_stats["best_correctness"])} produces more correct answers
- **Most Relevant:** {format_framework(llm_stats["best_relevance"])} produces more relevant answers
"""

    # Add overall winner summary
    fs_wins = sum(
        [
            time_stats["best"] == "fs-explorer",
            llm_stats["best_correctness"] == "fs-explorer",
            llm_stats["best_relevance"] == "fs-explorer",
        ]
    )
    rag_wins = 3 - fs_wins

    if fs_wins > rag_wins:
        overall_winner = "FS-Explorer"
    elif rag_wins > fs_wins:
        overall_winner = "RAG"
    else:
        overall_winner = "Tie"

    md += (
        f"\n**Overall Winner:** {overall_winner} ({max(fs_wins, rag_wins)}/3 metrics)\n"
    )

    return md


def get_eval_stats(
    results_file: str, result_json_file: str, result_md_file: str
) -> None:
    results = get_results(results_file)
    times = [result["time_taken"] for result in results]
    llm_evals = [result["llm_evaluations"] for result in results]
    time_stats = get_time_average(times)
    llm_stats = get_llm_stats(llm_evals)
    eval_stats = EvalStats(
        time_stats=time_stats,
        llm_stats=llm_stats,
    )

    with open(result_json_file, "w") as f:
        json.dump(eval_stats, f, indent=2)

    markdown_report = create_markdown_report(eval_stats, len(results))
    with open(result_md_file, "w") as f:
        f.write(markdown_report)
