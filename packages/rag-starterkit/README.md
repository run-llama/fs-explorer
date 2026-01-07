# rag-starterkit

RAG application built around a starter kit to benchmark in performance against the fs-explorer agent.

## Stack

- [LlamaParse](https://developers.llamaindex.ai/python/cloud/llamaparse/) for advanced, OCR-driven text parsing and extraction
- [Chonkie](https://chonkie.ai) for sentence-based chunking
- OpenAI for dense embeddings
- [FastEmbed](https://github.com/qdrant/fastembed) for sparse embeddings
- [Qdrant](https://qdrant.tech) for vector storage and search

## Flow

### Data Ingestion

```mermaid
flowchart TD
    A(PDF files)
    B[Pre-parse with LlamaParse and cache]
    C(Load texts from Cache)
    D[Chunk wth Chonkie]
    H(Chunks)
    E[Embed with OpenAI - dense]
    I(Dense Embeddings)
    F[Embed with FastEmbed - sparse]
    J(Sparse Embeddings)
    G[Upload to Qdrant]
    A --> B
    B --> C
    C --> D
    D --> H
    H --> E
    H --> F
    E --> I
    F --> J 
    I --> G
    J --> G
```

### Retrieval Augmented Generation

```mermaid
flowchart TD
    A(Query)
    M[Determine which file to use]
    N(File filter for Qdrant search)
    B[Embed with OpenAI - Dense]
    C(Dense Query Embedding)
    D[Embed with Fastembed - Sparse]
    E(Sparse Query Embedding)
    F[Search Qdrant]
    G(Dense Search Results)
    H(Sparse Search Results)
    I[Reranking with Reciprocal Rank Fusion]
    J(Most relevant context - hybrid)
    K[OpenAI LLM]
    L(Generated Response)
    A --> M
    M --> N
    N --> F
    A --> B
    B --> C 
    A --> D
    D --> E
    E --> F
    C --> F
    F --> G
    F --> H 
    H --> I
    G --> I
    I --> J
    J --> K
    K --> L
```