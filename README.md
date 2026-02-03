# ECCN Classification Service (RAG-based)

## Overview

This project implements a **production-style ECCN classification service** that maps product descriptions to the correct Export Control Classification Number (ECCN) using **semantic retrieval + LLM-based decision making**.

Instead of relying on brittle rule-based matching or direct classification, the system uses a **Retrieval-Augmented Generation (RAG)** pipeline to ensure accuracy, safety, and explainability in a regulatory context.

---

## High-Level Flow (One-Line)

**Product description → semantic retrieval (FAISS) → candidate ECCNs → LLM decision constrained to retrieved context → final ECCN**

---

## Architecture & Tech Stack

* **Embeddings:** `BAAI/bge-base-en-v1.5` (semantic retrieval over regulatory text)
* **Vector Store:** FAISS (fast local similarity search)
* **LLM:** Gemini 2.5 Flash (decision layer, hallucination-constrained)
* **API:** FastAPI (REST interface)
* **Evaluation:** Recall@K-focused synthetic evaluation
* **Language:** Python

---

## Why This Approach?

ECCN definitions are:

* Highly granular
* Semantically overlapping
* Risk-sensitive (hallucinations are unacceptable)

A pure classifier or a free-form LLM is unsafe.
This system **first narrows the search space** and then **forces the LLM to choose only from valid regulatory definitions**.

---

## Key Design Decisions

### 1. Leaf-Only ECCN Indexing

Only final (leaf) ECCN nodes are indexed.

* Removes semantic noise
* Prevents invalid parent-level predictions

### 2. Retrieval First, Reasoning Second

The LLM **never sees the full dataset**.
It only decides among top-K retrieved ECCNs.

### 3. Real-Time Prompt Constraints (No Fine-Tuning)

Instead of model fine-tuning:

* Strict prompt rules enforce safe outputs
* ECCN hallucination is structurally prevented
* Easier to audit and iterate

---

## Challenges & How They Were Solved

### Semantic Overlap Between ECCNs

**Solution:** Dense embeddings + FAISS to retrieve only close siblings.

### Parent / Non-Terminal ECCNs Polluting Results

**Solution:** Indexed only leaf nodes.

### Embeddings Underperforming on Regulatory Text

**Solution:** Switched to BGE embeddings optimized for semantic retrieval.

### LLM Hallucination Risk

**Solution:** Retrieval-bounded prompts + strict output format.

### No Labeled Product Dataset

**Solution:** Generated a synthetic evaluation set via LLM-based paraphrasing.

---

## Evaluation Summary

| Metric       | Baseline       | Final     |
| ------------ | -------------- | --------- |
| Recall@5     | ~0.65          | **0.865** |
| Exact Match  | Low (expected) | Low       |
| Parent Match | Low            | Improved  |

> Recall@K was prioritized over exact match accuracy, as retrieval quality is the critical safety metric in RAG-based regulatory systems.

---

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone <repo-url>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Building the Index

The FAISS index is **generated locally** and not committed to GitHub.

```bash
python app.py
```

This:

* Loads ECCN data
* Builds embeddings
* Creates FAISS index

---

## Starting the API

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API Docs:

```
http://localhost:8000/docs
```

---

## Example API Request

**POST** `/classify`

```json
{
  "product_text": "Weapon-mounted holographic optical sight"
}
```

**Response**

```json
{
  "predicted_ecn": "0A504.B",
  "reason": "The description directly matches holographic weapon sighting systems.",
  "retrieved_candidates": ["0A504.B", "0A504.C", "0A504.D"]
}
```

---

## How to Reuse This in Your Own Project

To adapt this system:

1. Replace the ECCN dataset with your domain taxonomy
2. Embed only **final decision nodes**
3. Tune retrieval (`top_k`, embeddings)
4. Constrain the LLM to retrieved candidates only
5. Evaluate using Recall@K, not just accuracy

This pattern works well for:

* Regulatory classification
* Product taxonomy mapping
* Legal or compliance search
* Controlled decision systems

---

## Trade-Offs & Future Improvements

* Hierarchy-aware scoring (parent-child consistency)
* Confidence calibration
* Active learning on low-confidence predictions
* Hybrid rule + semantic filtering

---

## Final Note

This project focuses on **correct system design over shortcut accuracy**, emphasizing explainability, safety, and reproducibility — all critical for real-world compliance systems.

---
