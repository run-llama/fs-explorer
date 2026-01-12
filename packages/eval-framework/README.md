# Evaluation Framework

Framework to evaluate the performance of the fs-explorer agent (agentic file search with filesystem tools) agains the performance of traditional RAG. 

## Run the evaluation (small scale)

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

## Run the evaluation (at scale)

The evaluation can be brought to higher scales by using 100 or 1000 text-based abstracts from the most recent AI-related papers on arXiv.

In order to get the data:

- Go to the benchmark directory:

```bash
cd benchmarks
```

- Run the scripts to collect 100 and 1000 arXiv papers:

```bash
bash scripts/download_arxiv_100.sh
bash scripts/download_arxiv_1000.sh
```

- The previous step will create a `texts/` directory under both `arxiv-100-papers` and `arxiv-1000-papers`, as well as a `metadata.jsonl` file. Before running the evaluation on the existing `question_and_answers.json` file, we suggest you check for the existence of the files mentioned in the Q&A pairs, and, if they don't exist, you should create a new set of Q&A.

- Cache the papers in both the directories:

```bash
# install the cache-arxiv package first, under the packages/ directory
cd arxiv-100-papers
cache-arxiv
cd ../arxiv-1000-papers
cache-arxiv
cd ..
```

- Now run the evaluation on `arxiv-100-papers`:

```bash
cd arxiv-100-papers
run-eval -df question_and_answers.json --advanced
get-stats
cd ..
```

- Once the evaluation is done and you collected the statistics, remove the `rag-benchmark-advanced` collection from Qdrant (we will be re-using it for the next experiment):

```bash
curl  -X DELETE \
  'http://localhost:6333/collections/rag-benchmark-advanced'
```

- Then, head to `arxiv-1000-papers`, and run the evaluation (you might need to split the tasks into sub-tasks because of rate limiting):

```bash
cd arxiv-1000-papers
run-eval -df question_and_answers.json --advanced
get-stats
```