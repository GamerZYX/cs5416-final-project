# CS 5416 Final Project — Team Work Breakdown (CPU-Only Version)

This document describes the team structure and concrete deliverables for each group member
in our 4-person CPU-only implementation of the **Distributed ML Inference Pipeline**.

Our CPU-only architecture uses:
- **Node 0:** Client-facing gateway + embedder + orchestration  
- **Node 1:** FAISS + document retrieval + reranker  
- **Node 2:** LLM generation + sentiment analysis + toxicity detection  
All nodes communicate via JSON over HTTP (Flask), with batching implemented at every stage.

---

# 👥 Team Roles & Responsibilities

Below is the clear division of work for each member of our team.

---

## 👤 A — Node 0 & System Orchestration

### **Deliverables**
- `run.sh`, `install.sh`, and `main.py` to launch the correct node based on environment variables.
- Full implementation of the **Node 0 Flask server**:
  - `/request` endpoint (compatible with the official client)
  - Parse `{request_id, query}`
  - Run local **embedding** (batched)
  - Forward batch requests to Node 1 (FAISS pipeline)
  - Forward Node 1 results to Node 2 (LLM pipeline)
  - Aggregate final responses and return to client.
- Request state tracking + end-to-end latency logging.
- Robustness handling (timeouts, retries, malformed responses).
- **Report Contribution:**  
  System design overview, orchestration strategy, and Node0 pipeline details.

---

## 👤 B — Node 1: FAISS + Document Retrieval + Reranking

### **Deliverables**
- Load the **13GB FAISS index** and documents using CPU.  
  Ensures only Node 1 loads the large index to avoid exceeding the 16GB RAM limit.
- Implement the **Node 1 Flask server**:
  - Endpoint: `/rag_batch`
  - Input: `[{ request_id, query, embedding }]`
  - Output: `[{ request_id, context }]`
- Full **RAG pipeline implementation**:
  - Batched FAISS search
  - Document loading (with optional caching)
  - Batched reranker model
- RAG optimizations:
  - Tune `top_k`
  - Batch size experiments
  - Memory control and latency measurement
- **Report Contribution:**  
  FAISS pipeline description, document loading design, and Node1 profiling results.

---

## 👤 C — Node 2: LLM Generation + Sentiment + Toxicity (CPU-Only)

### **Deliverables**
- Load all Node 2 models on CPU:
  - LLM (max 128 output tokens)
  - Sentiment classifier
  - Toxicity / sensitivity classifier
- Implement **Node 2 Flask server**:
  - Endpoint: `/generate_and_analyze_batch`
  - Input: `[{ request_id, context }]`
  - Output:
    ```json
    {
      "request_id": "...",
      "generated_response": "...",
      "sentiment": "...",
      "is_toxic": "false"
    }
    ```
- Full generation pipeline:
  - Batched prompt construction
  - Batched LLM inference (CPU)
  - Batched sentiment analysis
  - Batched toxicity detection
- Performance optimization:
  - Batch size tuning (1, 2, 4, 8)
  - CPU latency measurements
  - Memory usage control (<16GB)
- **Report Contribution:**  
  LLM design, CPU-only performance analysis, Node2 bottleneck analysis.

---

## 👤 D — Batching Framework + Profiling + Final Report

### **Deliverables**
- Implementation of a **universal `BatchingWorker`** used by Node0, Node1, and Node2:
  - Accepts `max_batch_size` and `max_wait_ms`
  - Background processing thread
  - Orders results correctly by request_id
- Integration of batching into all stages:
  - Node0: embedding  
  - Node1: FAISS + rerank  
  - Node2: LLM + sentiment + toxicity  
- Complete **profiling & experiments**:
  - End-to-end latency (Node 0)
  - FAISS latency vs batch size (Node1)
  - LLM latency vs batch size (Node2)
  - Overall throughput
  - Memory estimates
- Produce tables & plots for:
  - latency vs batch size
  - throughput vs batch size
  - monolithic vs distributed
  - CPU utilization
- Final report (primary writer):
  - System design  
  - Batching strategy  
  - Profiling results  
  - Optimization choices  
  - Bottleneck & trade-off analysis  
  - Final conclusions
- Organize final submission folder and perform final 3-node test.

---

# ✔️ Final Notes
This division ensures:
- Workload is balanced  
- No role conflicts  
- Clear ownership  
- Efficient integration  
- Smooth Preliminary & Final submissions  

Each member has their own module, but nodes communicate cleanly through JSON over HTTP, making collaboration straightforward.

Good luck team! 🚀
