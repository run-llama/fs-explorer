# Evaluation Framework

Framework to evaluate the performance of the fs-explorer agent (agentic file search with filesystem tools) agains the performance of traditional RAG. 

## Run the evaluation

> This assumes that you have `fs-explorer` and `rag-starterkit` installed

Start local Qdrant Server (from the `packages/eval-framework` folder):

```bash
docker compose up -d
```

Move to the benchmark folder with the data:

```bash
cd ../../data/benchmark
```

Pre-parse all PDF files with LlamaParse for the benchmark to be faster:

```bash
explore load-cache --directory . --skip tmp --skip questions_and_answers.json --skip fs-explorer.log --skip rag-starterkit.log
```

Run evaluation (might need to break down the questions in [`questions_and_answers.json`](../../data/benchmark/questions_and_answers.json) into sub-groups because of rate-limiting issues):

```bash
run-eval -df questions_and_answers.json
```

This will produce a [`results.json`](../../data/benchmark/results.json) file (containing all the details on the evaluation tasks and results), that you can use to produce a [`statistics.json`](../../data/benchmark/statistics.json) file (containing summary statistics from the evaluation results) and a [`report.md`](../../data/benchmark/report.md) file (containing a human-readable report on the evaluation statistics). In order to get statistics, run:

```bash
get-stats
```