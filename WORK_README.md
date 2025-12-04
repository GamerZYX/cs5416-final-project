# CS 5416 Final Project - Distributed ML Inference Pipeline

## 📋 Overview

This is a distributed 3-node ML inference pipeline for customer support queries. The system processes queries through multiple stages: embedding, FAISS retrieval, document reranking, LLM generation, sentiment analysis, and toxicity detection.

### System Architecture

- **Node 0:** Client-facing gateway + embedding + orchestration
- **Node 1:** FAISS retrieval + document fetching + reranking
- **Node 2:** LLM generation + sentiment analysis + toxicity detection

All nodes communicate via JSON over HTTP (Flask), with opportunistic batching implemented at every stage.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 - 3.11
- ~16GB RAM per node
- Network connectivity between nodes

### Installation

1. **Clone the repository** (if not already done)

2. **Run installation script:**
   ```bash
   bash install.sh
   ```
   
   This will:
   - Install all Python dependencies from `requirements.txt`
   - Download required ML models (may take several minutes)

3. **Create test data** (if not already present):
   ```bash
   python3 create_test_docs.py
   ```
   
   This creates:
   - `documents/documents.db` - SQLite database with 1M documents
   - `faiss_index.bin` - FAISS index for vector search

---

## 🏃 Running the System

### For TA Evaluation (3 Separate Machines)

The TAs will run your `run.sh` on 3 separate machines with the following environment variables:

```bash
# On Machine 1 (Node 0)
TOTAL_NODES=3 \
NODE_NUMBER=0 \
NODE_0_IP=<ip0>:<port0> \
NODE_1_IP=<ip1>:<port1> \
NODE_2_IP=<ip2>:<port2> \
FAISS_INDEX_PATH=<path> \
DOCUMENTS_DIR=<path> \
bash run.sh

# On Machine 2 (Node 1)
TOTAL_NODES=3 \
NODE_NUMBER=1 \
NODE_0_IP=<ip0>:<port0> \
NODE_1_IP=<ip1>:<port1> \
NODE_2_IP=<ip2>:<port2> \
FAISS_INDEX_PATH=<path> \
DOCUMENTS_DIR=<path> \
bash run.sh

# On Machine 3 (Node 2)
TOTAL_NODES=3 \
NODE_NUMBER=2 \
NODE_0_IP=<ip0>:<port0> \
NODE_1_IP=<ip1>:<port1> \
NODE_2_IP=<ip2>:<port2> \
FAISS_INDEX_PATH=<path> \
DOCUMENTS_DIR=<path> \
bash run.sh
```

### For Local Testing (Single Machine, 3 Ports)

You can test the system locally using different ports:

**Terminal 1 - Node 0:**
```bash
source venv/bin/activate

TOTAL_NODES=3 \
NODE_NUMBER=0 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8001 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
python3 pipeline.py
```

**Terminal 2 - Node 1:**
```bash
source venv/bin/activate

TOTAL_NODES=3 \
NODE_NUMBER=1 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8001 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
python3 pipeline.py
```

**Terminal 3 - Node 2:**
```bash
source venv/bin/activate

TOTAL_NODES=3 \
NODE_NUMBER=2 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8001 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
python3 pipeline.py
```

**Terminal 4 - Client:**
```bash
source venv/bin/activate

NODE_0_IP=localhost:8000 python3 client.py
```

---

## 📡 API Endpoints

### Node 0 (Client Interface)

- **POST `/query`** - Main client endpoint
  - Input: `{"request_id": "string", "query": "string"}`
  - Output: `{"request_id": "string", "generated_response": "string", "sentiment": "string", "is_toxic": "string"}`

- **GET `/health`** - Health check
  - Output: `{"status": "healthy", "node": 0, "total_nodes": 3}`

### Node 1 (FAISS Retrieval)

- **POST `/retrieval_batch`** - FAISS search endpoint (internal)
- **GET `/health`** - Health check

### Node 2 (LLM + Analysis)

- **POST `/llm_rag_analysis_batch`** - LLM generation endpoint (internal)
- **GET `/health`** - Health check

---

## 📁 File Structure

```
cs5416-final-project/
├── install.sh              # Installation script
├── run.sh                  # Entry point for each node
├── pipeline.py             # Main pipeline code (handles all 3 nodes)
├── client.py               # Test client script
├── create_test_docs.py     # Generate test documents and FAISS index
├── download_model.py       # Pre-download ML models
├── requirements.txt        # Python dependencies
├── README.md              # Project requirements and spec
├── WORK_README.md         # This file - usage guide
└── documents/              # Document database directory
    └── documents.db        # SQLite database (generated)
```

---

## ⚙️ Configuration

The system reads configuration from environment variables:

- `TOTAL_NODES` - Total number of nodes (should be 3)
- `NODE_NUMBER` - Current node number (0, 1, or 2)
- `NODE_0_IP` - IP:port for Node 0
- `NODE_1_IP` - IP:port for Node 1
- `NODE_2_IP` - IP:port for Node 2
- `FAISS_INDEX_PATH` - Path to FAISS index file
- `DOCUMENTS_DIR` - Directory containing documents database

Batching parameters are configured in `pipeline.py`:
- `MAX_BATCH_SIZE` - Maximum batch size (default: 4)
- `MAX_WAIT_TIME_MS` - Maximum wait time for batching (default: 5ms)

---

## 🔍 Verification

### Check Node Health

```bash
# Node 0
curl http://localhost:8000/health

# Node 1
curl http://localhost:8001/health

# Node 2
curl http://localhost:8002/health
```

Expected response:
```json
{"status": "healthy", "node": 0, "total_nodes": 3}
```

### Test Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test_001", "query": "How do I return a product?"}'
```

---

## 🐛 Troubleshooting

### Port Already in Use

If a port is already in use, change the port numbers in the environment variables:
```bash
NODE_0_IP=localhost:8003  # Instead of 8000
NODE_1_IP=localhost:8004  # Instead of 8001
NODE_2_IP=localhost:8005  # Instead of 8002
```

### Models Not Found

If models are missing, run:
```bash
python3 download_model.py
```

Models are cached in `~/.cache/huggingface/hub/` and shared across all Python environments.

### Connection Errors

- Ensure all 3 nodes are running before starting the client
- Check that IP addresses and ports are correct
- Verify network connectivity between nodes (for distributed setup)

---

## 📊 System Flow

1. **Client** sends query to **Node 0** (`/query`)
2. **Node 0** generates embedding (batched)
3. **Node 0** forwards to **Node 1** (`/retrieval_batch`)
4. **Node 1** performs FAISS search and returns doc IDs
5. **Node 0** forwards to **Node 2** (`/llm_rag_analysis_batch`)
6. **Node 2** fetches documents, reranks, generates LLM response, analyzes sentiment and toxicity
7. **Node 2** returns final result to **Node 0**
8. **Node 0** returns result to **Client**

---

## 📝 Notes

- Models are loaded on-demand and unloaded after use to manage memory
- Batching is opportunistic - requests are batched when possible to improve throughput
- The system is designed to run on CPU-only (no GPU required)
- Each node should stay within 16GB RAM limit

---

## 🎯 For TAs

To evaluate this submission:

1. Run `bash install.sh` on each node (or ensure dependencies are installed)
2. Ensure `faiss_index.bin` and `documents/documents.db` exist (or run `create_test_docs.py`)
3. Run `bash run.sh` on each of the 3 nodes with appropriate environment variables
4. Use the provided `client.py` or your own client to send requests to Node 0

The system will automatically:
- Load appropriate models based on `NODE_NUMBER`
- Start Flask servers on the correct ports
- Handle inter-node communication
- Process requests with batching
