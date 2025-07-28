Retrieval-Augmented Generation (RAG) Pipeline using Open-Source Models

This repository implements a **fully local Retrieval-Augmented Generation (RAG) pipeline** for answering textbook-aligned or document-based queries in an interpretable and efficient manner. It leverages **LangChain**, **Chroma**, **Qwen Embeddings**, and **LLaMA 3** for document retrieval and answer generation.

---

 Features

- Open-source LLMs only — No proprietary APIs
- Hybrid retrieval using dense (Qwen3 Embeddings + ChromaDB) and sparse (BM25) methods
- Column-aware PDF parsing (two-column layout)
- Contextual answering using Meta-LLaMA-3-8B-Instruct
- Efficient memory usage via **4-bit quantization** and GPU caching
- Page number extraction and result traceability
  
---

 System Requirements

- Python 3.10+
- GPU with 40GB VRAM (e.g., A100 40GB)
- At least 100 GB disk space for model caching
- Hugging Face Token for model download
