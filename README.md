# Hybrid RAG & GraphRAG Question Answering

This project explores and compares two Retrieval-Augmented Generation approaches for question answering over *12 Angry Men*: a traditional **Hybrid RAG** pipeline based on document retrieval and a **GraphRAG** pipeline based on structured knowledge-graph evidence.

The goal is to investigate how different retrieval strategies can provide relevant evidence to a large language model and improve grounded question answering.

## Approaches

### 1. Hybrid RAG

The Hybrid RAG pipeline retrieves relevant passages from the screenplay using two complementary retrieval methods:

- **BM25** for lexical keyword-based retrieval
- **FAISS** with Sentence Transformer embeddings for semantic retrieval

The two ranked result lists are combined using **weighted Reciprocal Rank Fusion (RRF)**. The highest-ranked passages are then provided as context to **Qwen3-4B-Instruct-2507** for answer generation.

The pipeline follows:

```text
Question
   ↓
BM25 Retrieval ─────┐
                    ├── Weighted RRF → Retrieved Context → LLM → Answer
FAISS Retrieval ────┘
```

### 2. GraphRAG

The GraphRAG pipeline operates over a **pre-built knowledge graph** representing entities and relationships from the screenplay.

Relevant graph elements are identified for each question and their neighborhoods are explored to retrieve relational evidence. The resulting subgraph information is converted into structured context and supplied to the language model for answer generation.

The experiments investigate the effect of factors such as:

- graph traversal depth
- community-level context
- retrieved evidence size
- missing graph evidence
- graph inspection and targeted corrections

The pipeline follows:

```text
Question
   ↓
Graph Retrieval
   ↓
Neighborhood Traversal
   ↓
Structured Graph Evidence
   ↓
LLM
   ↓
Answer
```

> **Note:** The knowledge graph was provided as the starting point for the project. The GraphRAG work focuses on graph inspection and correction, retrieval, traversal, evidence construction, experimentation, and evaluation rather than constructing the graph from scratch.

## Evaluation

The Hybrid RAG system was evaluated against the same language model without retrieved screenplay context.

| System | Correct Answers |
| --- | ---: |
| Baseline LLM | 1 / 10 |
| Hybrid RAG | **8 / 10** |

Analysis of the two unsuccessful RAG cases indicated that the relevant evidence was not included in the retrieved context, suggesting that the primary limitation occurred during retrieval rather than answer generation.

The GraphRAG notebook further explores how graph structure, traversal depth, community information, evidence availability, and graph quality affect retrieval and downstream question answering.
## Repository Structure

```text
rag-graphrag-question-answering/
├── README.md
├── hybrid_rag.ipynb
├── graph_rag.ipynb
├── requirements.txt
└── .gitignore
```

## Technologies

**Retrieval & NLP**
- Sentence Transformers
- BM25
- FAISS
- Reciprocal Rank Fusion

**Graph-Based Retrieval**
- NetworkX
- Knowledge graphs
- Graph traversal
- Community-based context

**Language Models**
- Qwen3-4B-Instruct-2507

**Core**
- Python
- PyTorch
- NumPy
- pandas
- scikit-learn

## Notebooks

### `hybrid_rag.ipynb`

Implements the document-based RAG pipeline, including:

- screenplay ingestion and chunking
- BM25 lexical retrieval
- Sentence Transformer embeddings
- FAISS semantic retrieval
- weighted Reciprocal Rank Fusion
- LLM-based answer generation
- baseline comparison and error analysis

### `graph_rag.ipynb`

Explores graph-based retrieval using a provided knowledge graph, including:

- graph inspection and targeted correction
- relevant graph-element retrieval
- neighborhood traversal
- structured evidence construction
- community-level context
- GraphRAG question answering
- ablation and robustness experiments

## Data & Reproducibility

The project uses screenplay-related data and intermediate resources required by the original experiments. Some external data, prepared graph resources, and model artifacts are not included in this repository.

The notebooks retain the outputs of the original experiments so that the retrieval process, generated answers, evaluations, and experimental analyses can be inspected without rerunning the complete pipeline.

The repository is intended primarily as a documented presentation of the experimental workflow and results rather than as a fully self-contained reproduction package.

## Project Context

This project was developed as part of the **Artificial Intelligence** coursework at the **National Technical University of Athens (NTUA)**.

The repository has been reorganized and documented for portfolio presentation while preserving the original experimental implementations and results.
