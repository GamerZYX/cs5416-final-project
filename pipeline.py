import os
import json
import time
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Any
from queue import Queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
import faiss
import requests
from flask import Flask, request, jsonify

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer


# =============================
#  ENV & CONFIG
# =============================

TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 3))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "localhost:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8002")

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents")

CONFIG = {
    "faiss_dim": 768,
    "retrieval_k": 10,       # must retrieve 10 docs
    "truncate_length": 512,  # must use this truncate length
    "max_tokens": 128,       # must use this max token limit
    "MAX_BATCH_SIZE": 4,
    "MAX_WAIT_MS": 5,
}

app = Flask(__name__)


# =============================
#  BATCHING SERVICE
# =============================

@dataclass
class BatchItem:
    request_id: str
    data: Dict[str, Any]
    future: Future


class BatchingService:
    def __init__(self, process_fn, max_batch_size: int, max_wait_ms: int):
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.queue: "Queue[BatchItem]" = Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print(f"[Node {NODE_NUMBER}] BatchingService started with max_batch_size={max_batch_size}")

    def submit(self, request_id: str, data: Dict[str, Any]) -> Future:
        f = Future()
        self.queue.put(BatchItem(request_id, data, f))
        return f

    def _worker(self):
        current_batch: List[BatchItem] = []
        timer_start = None

        while True:
            try:
                item = self.queue.get(timeout=0.001)
                current_batch.append(item)
                if timer_start is None:
                    timer_start = time.time()
            except Exception:
                pass

            elapsed = time.time() - timer_start if timer_start else 0.0

            if (
                current_batch
                and (
                    len(current_batch) >= self.max_batch_size
                    or elapsed >= self.max_wait
                )
            ):
                batch_ids = [b.request_id for b in current_batch]
                batch_data = [b.data for b in current_batch]
                try:
                    results = self.process_fn(batch_ids, batch_data)
                    for b, r in zip(current_batch, results):
                        b.future.set_result(r)
                except Exception as e:
                    for b in current_batch:
                        b.future.set_exception(e)

                current_batch = []
                timer_start = None

            if not current_batch:
                time.sleep(0.001)


# =============================
#  PIPELINE CLASS
# =============================

class Pipeline:
    """
    Three-node pipeline:

    Node 0: embedding + doc fetch + rerank + sentiment + toxicity + orchestration
    Node 1: FAISS only
    Node 2: LLM only
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Node {NODE_NUMBER}] using device {self.device}")

        self.doc_db_path = os.path.join(DOCUMENTS_DIR, "documents.db")
        self.executor = ThreadPoolExecutor(max_workers=CONFIG["MAX_BATCH_SIZE"])

        # Node 0: embedder + reranker + sentiment + safety
        if NODE_NUMBER == 0:
            print("[Node 0] Loading embedding model...")
            self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5").to(self.device)

            print("[Node 0] Loading reranker...")
            self.reranker_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-base"
            ).to(self.device)
            self.reranker_model.eval()

            print("[Node 0] Loading sentiment classifier...")
            self.sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if self.device.type == "cuda" else -1,
            )

            print("[Node 0] Loading safety classifier...")
            self.safety_pipeline = hf_pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if self.device.type == "cuda" else -1,
            )

        # Node 1: FAISS only
        elif NODE_NUMBER == 1:
            print("[Node 1] Loading FAISS index...")
            if not os.path.exists(FAISS_INDEX_PATH):
                raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)

        # Node 2: LLM only
        elif NODE_NUMBER == 2:
            print("[Node 2] Loading LLM...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                torch_dtype=torch.float16,
            ).to(self.device)
            self.llm_model.eval()

    # ---------- Node 0: embedding batch fn ----------
    def embed_batch(self, batch_ids: List[str], batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        queries = [d["query"] for d in batch_data]
        embeddings = self.embedding_model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        out: List[Dict[str, Any]] = []
        for q, e in zip(queries, embeddings):
            out.append(
                {
                    "query": q,
                    "embedding": e.tolist(),  # JSON serializable
                }
            )
        return out

    # ---------- Node 1: FAISS batch fn ----------
    def faiss_batch(self, batch_ids: List[str], batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Input: [{'embedding': list, 'query': str (optional)}]
        Output: [{'doc_ids': [int, ...]}]
        """
        emb_list = [np.array(d["embedding"], dtype="float32") for d in batch_data]
        q_emb = np.stack(emb_list, axis=0)  # [B, D]
        _, idx = self.faiss_index.search(q_emb, CONFIG["retrieval_k"])  # [B, K]

        results: List[Dict[str, Any]] = []
        for row in idx:
            results.append(
                {
                    "doc_ids": row.tolist(),
                }
            )
        return results

    # ---------- Node 2: LLM batch fn ----------
    def llm_batch(self, batch_ids: List[str], batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Input: [{'prompt': str}]
        Output: [{'generated_response': str}]
        """
        prompts = [d["prompt"] for d in batch_data]
        inputs = self.llm_tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            gen_ids = self.llm_model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_tokens"],
                temperature=0.01,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        gen_ids = gen_ids[:, input_len:]  # strip prompt
        texts = self.llm_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        results: List[Dict[str, Any]] = []
        for t in texts:
            results.append({"generated_response": t})
        return results

    # ---------- Node 0: Helper functions (non-batched per request) ----------

    def fetch_documents(self, doc_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch docs from sqlite; doc_ids may contain duplicates."""
        if not doc_ids:
            return []

        unique_ids = sorted(set(doc_ids))
        placeholders = ",".join("?" * len(unique_ids))
        conn = sqlite3.connect(self.doc_db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT doc_id, title, content, category FROM documents WHERE doc_id IN ({placeholders})",
            unique_ids,
        )
        doc_map: Dict[int, Dict[str, Any]] = {}
        for doc_id, title, content, category in cursor.fetchall():
            doc_map[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "content": content,
                "category": category,
            }
        conn.close()

        # preserve order (with duplicates) from doc_ids
        ordered_docs: List[Dict[str, Any]] = []
        for d in doc_ids:
            if d in doc_map:
                ordered_docs.append(doc_map[d])
        return ordered_docs

    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[float]:
        if not docs:
            return []

        pairs = [[query, d["content"]] for d in docs]
        tok = self.reranker_tok(
            pairs,
            padding=True,
            truncation=True,
            max_length=CONFIG["truncate_length"],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.reranker_model(**tok).logits.view(-1)
        return logits.float().tolist()

    def analyze_sentiment(self, text: str) -> str:
        truncated = text[:CONFIG["truncate_length"]]
        res = self.sentiment_pipeline([truncated])[0]
        label = res["label"]
        mapping = {
            "1 star": "very negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "very positive",
        }
        return mapping.get(label, "neutral")

    def analyze_safety(self, text: str) -> str:
        truncated = text[:CONFIG["truncate_length"]]
        res = self.safety_pipeline([truncated])[0]
        is_toxic = res["score"] > 0.5
        return "true" if is_toxic else "false"


pipeline = Pipeline()

# Node-specific batchers
embed_batcher: BatchingService = None  # type: ignore
faiss_batcher: BatchingService = None  # type: ignore
llm_batcher: BatchingService = None  # type: ignore


# =============================
#  INTER-NODE COMM
# =============================

def get_node_base_url(node_id: int) -> str:
    if node_id == 0:
        return f"http://{NODE_0_IP}"
    elif node_id == 1:
        return f"http://{NODE_1_IP}"
    elif node_id == 2:
        return f"http://{NODE_2_IP}"
    else:
        raise ValueError("Invalid node id")


def forward_batch(node_id: int, endpoint: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    items: [{'request_id': ..., 'data': {...}}, ...]
    """
    url = f"{get_node_base_url(node_id)}/{endpoint}"
    payload = {"batch": items}
    start = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    dt = time.time() - start
    print(f"[Node {NODE_NUMBER}] → Node {node_id} /{endpoint}, batch={len(items)}, took={dt:.3f}s")

    resp.raise_for_status()
    body = resp.json()
    return body["results"]


# =============================
#  ROUTES
# =============================

@app.route("/query", methods=["POST"])
def handle_query():
    """
    Client-facing endpoint (Node 0 only).
    """
    if NODE_NUMBER != 0:
        return jsonify({"error": "Only Node 0 handles /query"}), 400

    try:
        data = request.json or {}
        request_id = data.get("request_id")
        query = data.get("query")

        if not request_id:
            return jsonify({"error": "Missing request_id"}), 400
        if not query:
            return jsonify({"error": "Missing query"}), 400

        t0 = time.time()

        # 1) Node0: embed (through batcher)
        future_emb = embed_batcher.submit(request_id, {"query": query})
        emb_res = future_emb.result(timeout=300)   # {'query': ..., 'embedding': [...]}

        # 2) Node1: FAISS (through /faiss_batch)
        faiss_results = forward_batch(
            1,
            "faiss_batch",
            [{"request_id": request_id, "data": emb_res}],
        )
        faiss_out = faiss_results[0]  # {'request_id', 'doc_ids': [...]}
        doc_ids = faiss_out["doc_ids"]

        # 3) Node0: doc fetch + rerank + build context
        docs = pipeline.fetch_documents(doc_ids)
        scores = pipeline.rerank(query, docs) if docs else []

        # sort docs by rerank score desc
        doc_score_pairs = list(zip(docs, scores)) if scores else [(d, 0.0) for d in docs]
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        top_k = 3
        top_docs = doc_score_pairs[:top_k]
        context_parts = [f"- {d['title']}: {d['content'][:200]}" for d, _ in top_docs]
        context = "\n".join(context_parts)

        # 4) Build prompt and send to Node2 for LLM only
        prompt = (
            "You are a helpful customer support assistant.\n"
            "When given Context and Question, reply strictly in the form: 'Answer: <final answer>'.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

        llm_results = forward_batch(
            2,
            "llm_batch",
            [{"request_id": request_id, "data": {"prompt": prompt}}],
        )
        llm_out = llm_results[0]  # {'request_id', 'generated_response': ...}
        generated = llm_out["generated_response"]

        # 5) Node0: sentiment + toxicity
        sentiment = pipeline.analyze_sentiment(generated)
        is_toxic = pipeline.analyze_safety(generated)

        processing_time = time.time() - t0

        return jsonify(
            {
                "request_id": request_id,
                "generated_response": generated,
                "sentiment": sentiment,
                "is_toxic": is_toxic,
                "processing_time": f"{processing_time:.2f}s",
            }
        ), 200

    except Exception as e:
        print(f"[Node 0] Error in /query: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/faiss_batch", methods=["POST"])
def handle_faiss_batch():
    """
    Node1-only endpoint for FAISS retrieval.
    """
    if NODE_NUMBER != 1:
        return jsonify({"error": "FAISS runs only on Node 1"}), 403

    try:
        body = request.json or {}
        batch = body.get("batch", [])
        futures: List[Future] = []

        for item in batch:
            rid = item["request_id"]
            data = item["data"]
            # embedding is list[float]; our batch fn will convert to np internally
            futures.append(faiss_batcher.submit(rid, data))

        results_with_id: List[Dict[str, Any]] = []
        for fut, item in zip(futures, batch):
            res = fut.result(timeout=300)
            out = {"request_id": item["request_id"], **res}
            results_with_id.append(out)

        return jsonify({"results": results_with_id}), 200

    except Exception as e:
        print(f"[Node 1] Error in /faiss_batch: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/llm_batch", methods=["POST"])
def handle_llm_batch():
    """
    Node2-only endpoint for LLM generation.
    """
    if NODE_NUMBER != 2:
        return jsonify({"error": "LLM runs only on Node 2"}), 403

    try:
        body = request.json or {}
        batch = body.get("batch", [])
        futures: List[Future] = []

        for item in batch:
            rid = item["request_id"]
            data = item["data"]  # {'prompt': str}
            futures.append(llm_batcher.submit(rid, data))

        results_with_id: List[Dict[str, Any]] = []
        for fut, item in zip(futures, batch):
            res = fut.result(timeout=300)
            out = {"request_id": item["request_id"], **res}
            results_with_id.append(out)

        return jsonify({"results": results_with_id}), 200

    except Exception as e:
        print(f"[Node 2] Error in /llm_batch: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}
    ), 200


# =============================
#  MAIN
# =============================

def main():
    global embed_batcher, faiss_batcher, llm_batcher

    if NODE_NUMBER == 0:
        embed_batcher = BatchingService(
            pipeline.embed_batch,
            CONFIG["MAX_BATCH_SIZE"],
            CONFIG["MAX_WAIT_MS"],
        )
    elif NODE_NUMBER == 1:
        faiss_batcher = BatchingService(
            pipeline.faiss_batch,
            CONFIG["MAX_BATCH_SIZE"],
            CONFIG["MAX_WAIT_MS"],
        )
    elif NODE_NUMBER == 2:
        llm_batcher = BatchingService(
            pipeline.llm_batch,
            CONFIG["MAX_BATCH_SIZE"],
            CONFIG["MAX_WAIT_MS"],
        )
    else:
        print(f"[Node {NODE_NUMBER}] WARNING: no role assigned.")

    # decide host/port
    if NODE_NUMBER == 0:
        ip_port = NODE_0_IP
    elif NODE_NUMBER == 1:
        ip_port = NODE_1_IP
    else:
        ip_port = NODE_2_IP

    host, port_str = ip_port.split(":")
    port = int(port_str)

    print(f"[Node {NODE_NUMBER}] Starting Flask at {host}:{port}")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
