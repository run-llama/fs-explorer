# fs-explorer

CLI agent that helps you explore a directory and its content (it can also read PDF/PPTX/DOCX/XLSX files!).

## Installation and Usage

Clone this repository:

```bash
git clone https://github.com/run-llama/fs-explorer
cd fs-explorer
```

Install locally:

```bash
uv pip install .
```

Export a Google API key (must have access to EAP models):

```bash
export GOOGLE_API_KEY="..."
```

Run:

```bash
explore run --task "Within the data/ directory, can you help e find the PDF file that contains an order or a complaint, and, once you found them, ask me which one I would like you to summarize"
```

## Evaluation against traditional RAG

In the [`packages`](./packages/) directory we have two packages:

- [**rag-starterkit**](./packages/rag-starterkit/): a traditional RAG application implementing hybrid search. (More in the [dedicated README](./packages/rag-starterkit/README.md)).
- [**eval-framework**](./packages/eval-framework/): a simple evaluation framework that produces LLM-as-a-judge-based evals along with collecting time-, tool usage- and file-search-based statistics. Find out how to run the evaluation and collect statistics in the [dedicated README](./packages/eval-framework/README.md).
