import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue
import threading

import requests
from concurrent.futures import Future, ThreadPoolExecutor
# https://docs.python.org/3/library/concurrent.futures.html#future-objects

import logging
import time
from abc import ABC, abstractmethod
import itertools

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8001')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8002')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768, #You must use this dimension
    'max_tokens': 128, #You must use this max token limit
    'retrieval_k': 10, #You must retrieve this many documents from the FAISS index
    'truncate_length': 512, # You must use this truncate length
    ############################
}

app = Flask(__name__)



@dataclass
class BatchItem:
    """Request and a Future object for returning the result."""
    request_id: str
    data: Dict[str, Any]
    future: Future 

@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float

@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    processing_time: float 

# Batching Service

class BatchingService:
    """
    Manages the input queue and runs the opportunistic batching worker.
    """
    
    def __init__(self, process_batch_func, max_size, max_wait_ms, name):
        self.input_queue = Queue()
        self.process_batch_func = process_batch_func
        self.max_size = max_size
        self.max_wait = max_wait_ms / 1000.0 # Convert ms to seconds
        self.name = name
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print(f"Batching Service {self.name} started for Node {NODE_NUMBER}", )

    def submit(self, request_id: str, data: Dict[str, Any]) -> Future:
        """
        Submit a request 
        return concurrent.Future
        """
        f = Future()
        item = BatchItem(request_id=request_id, data=data, future=f)
        self.input_queue.put(item)
        return f

    def _worker(self):
        """
        opportunistic batching logic
        """
        current_batch = []
        timer_start = None

        while True:
            try:
                item = self.input_queue.get(timeout=0.001)
                current_batch.append(item)
                if timer_start is None:
                    timer_start = time.perf_counter()
            except Exception: 
                pass
            
            time_elapsed = time.perf_counter() - timer_start if timer_start else 0.0
            
            if (
                len(current_batch) >= self.max_size or # reached max size 
                (current_batch and time_elapsed >= self.max_wait) # waited long enough
            ):
                
                # TODO: DEL ME
                report = f"[{NODE_NUMBER}, {self.name}]\n"
                if (len(current_batch) >= self.max_size): 
                    report += f"\tBatch triggered: Queue Full after {time_elapsed}\n"
                else: 
                    report += f"\tBatch triggered: Timed out with batch size {len(current_batch)}\n"
                
                batch_ids = [item.request_id for item in current_batch]
                batch_data = [item.data for item in current_batch]
                
                report += f"\tProcessing batch of size {len(current_batch)}\n"
                
                try:
                    # process_batch_func should return a list of results 
                    # matching the order of the inputs
                    start = time.perf_counter()
                    results = self.process_batch_func(batch_ids, batch_data)
                    end = time.perf_counter()
                    report += f"\tProcessed in {end - start} seconds\n"
                    print(report, flush=True)
                    # 4. Set results/exceptions on the Futures
                    for item, result in zip(current_batch, results):
                        item.future.set_result(result)

                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    for item in current_batch:
                        item.future.set_exception(e)
                
                # 5. Reset for the next batch
                current_batch = []
                timer_start = None
            
            # If the queue was empty and no batch was processed, sleep briefly
            if not current_batch:
                time.sleep(0.001)

class Base_Service(ABC): 
    def __init__(self, max_size, max_wait_ms, name): 
        self.batch_service = BatchingService(self._batch_job, max_size, max_wait_ms, name)
    def submit(self, request_id, data:Dict[str, Any]) -> Future: 
        return self.batch_service.submit(request_id, data)
    @abstractmethod
    def _batch_job(self, batch_ids: list[str], batch_data: List[Dict]) -> List[Dict]: 
        pass

class Embedder(Base_Service):
    def __init__(self, max_size, max_wait_ms, device):
            self.device = device 
            self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
            self.embedding_model = SentenceTransformer(self.embedding_model_name).to(self.device)
            print(f"Loaded Embedder: {self.embedding_model_name}.")

            super().__init__(max_size, max_wait_ms, "Embedder")
    
    def _batch_job(self, batch_id: list[str], batch_data: List[Dict]) -> List[Dict]: 
        queries = [data['query'] for data in batch_data]
        embeddings = self.embedding_model.encode(
            queries, 
            normalize_embeddings=True, 
        )
        results = []
        for embedding in embeddings: 
            results.append({
                "embedding": embedding.tolist()
            })
        return results

class FAISS(Base_Service): 
    def __init__(self, max_size, max_wait_ms): 
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError("FAISS index not found on Node 1.")
        self.faiss_index = faiss.read_index(CONFIG['faiss_index_path'])#, faiss.IO_FLAG_MMAP)
        print(f"Loaded FAISS Index.")
        super().__init__(max_size, max_wait_ms, "FAISS")
    def _batch_job(self, batch_ids: List[str], batch_data: List[Dict]) -> List[Dict]:
        """
        Main Node 1 code (FAISS index lookup only)
        Input: [{'embedding': list, 'query': str}]
        Output: [{'query': str, 'embedding': list, 'doc_ids': list}]
        """
        
        query_embeddings = np.array([data['embedding'] for data in batch_data], dtype='float32')
        _, indices = self.faiss_index.search(query_embeddings, CONFIG['retrieval_k'])
        doc_id_batches = [row.tolist() for row in indices]
        
        results = []
        for idx, doc_ids in enumerate(doc_id_batches):
            results.append({
                'doc_ids': doc_ids
            })            
        return results

class DocFetch(Base_Service): 
    def __init__(self, max_size, max_wait_ms): 
        self.db_path = f"{CONFIG['documents_path']}/documents.db"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False) # all threads read only

        super().__init__(max_size, max_wait_ms, "DB Fetch")
    
    def _batch_job(self, batch_ids: list[str], batch_data: List[Dict]) -> List[Dict]:
        doc_ids = set()
        for data in batch_data:
            doc_ids.update(data.get('doc_ids', []))
        
        
        cursor = self.conn.cursor()
        doc_map = {}
        doc_ids_str = ','.join('?' * len(doc_ids))
        
        # Single batched SQL query
        cursor.execute(
            f'SELECT doc_id, title, content, category FROM documents WHERE doc_id IN ({doc_ids_str})',
            tuple(doc_ids)
        )
        
        for result in cursor.fetchall():
            doc_id, title, content, category = result
            doc_map[doc_id] = {
                'doc_id': doc_id,
                'title': title,
                'content': content,
                'category': category
            }
        results = []
        for data in batch_data:
            results.append({
                doc_id: doc_map[doc_id] for doc_id in data.get('doc_ids', [])
            })
        return results

class Rerank(Base_Service): 
    def __init__(self, max_size, max_wait_ms, device): 
        self.device = device
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device).eval()
        print("Loaded Reranker.")

        super().__init__(max_size, max_wait_ms, "Reranker")
    def _batch_job(self, batch_ids: List[str], batch_data: List[Dict]) -> List[Dict]:
        query_doc_pairs = []
        pair_counts = []
        for data in batch_data: 
            per_req_pair = []
            for doc_id, doc in data['doc_ids'].items(): 
                per_req_pair.append([data['query'], doc['content']])
            pair_counts.append(len(per_req_pair))
            query_doc_pairs.extend(per_req_pair)
        
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                query_doc_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=CONFIG['truncate_length']
            ).to(self.device) # type: ignore
            # Reranker scores are logits; we return them as float list
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float().tolist() # type: ignore

        context_list = []
        score_idx = 0
        res = []
        for i, data in enumerate(batch_data):
            count = pair_counts[i]
            
            current_scores = scores[score_idx : score_idx + count]
            
            res.append({"scores": current_scores})
            
            score_idx += count
        # docs will still have to be sorted
        return res

class LLM(Base_Service): 
    def __init__(self, max_size, max_wait_ms, device): 
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.device = device

        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16, 
        ).to(self.device)
        self.llm_model.eval()
        print("Loaded LLM.")

        super().__init__(max_size, max_wait_ms, "LLM")
    def _batch_job(self, batch_ids: List[str], batch_data: List[Dict]) -> List[Dict]:
        llm_input_texts = []
        for data in batch_data:
            query = data['query']
            context = data['context']
            messages = [
                {"role": "system",
                "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            llm_input_texts.append(text)
            
        model_inputs = self.llm_tokenizer(llm_input_texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        
        input_length = model_inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, input_length:]
        generated_responses = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [{"generated_response": gr} for gr in generated_responses]

class Sentiment(Base_Service): 
    def __init__(self, max_size, max_wait_ms, device): 
        self.device = device

        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.sentiment_classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
        )
        print("Loaded Sentiment Classifier.")

        super().__init__(max_size, max_wait_ms, "Sentiment")
    def _batch_job(self, batch_ids: List[str], batch_data: List[Dict]) -> List[Dict]:
        truncated_texts = [data['generated_response'][:CONFIG['truncate_length']] for data in batch_data]
        raw_results = self.sentiment_classifier(truncated_texts) 
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        return [{"sentiment": sentiment_map.get(res['label'], 'neutral')} for res in raw_results]

class Safety(Base_Service): 
    def __init__(self, max_size, max_wait_ms, device): 
        self.device = device
        self.safety_model_name = 'unitary/toxic-bert'

        self.safety_classifier = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device
        )
        print("Loaded Safety Classifier.")
        super().__init__(max_size, max_wait_ms, "Safety")
    
    def _batch_job(self, batch_ids: List[str], batch_data: List[Dict]) -> List[Dict]:
        """Filter response for safety for a batch of responses."""
        truncated_texts = [data['generated_response'][:CONFIG['truncate_length']] for data in batch_data]
        raw_results = self.safety_classifier(truncated_texts) 
        
        results = []
        for res in raw_results:
            is_toxic = res['score'] > 0.5
            results.append({"is_toxic": "true" if is_toxic else "false"})
            
        return results

def get_node_ip(node_id):
    """Helper to get IP of a specific node."""
    if node_id == 0: 
        return f"http://{NODE_0_IP}"
    if node_id == 1: 
        return f"http://{NODE_1_IP}"
    if node_id == 2: 
        return f"http://{NODE_2_IP}"
    raise ValueError("Invalid node ID")


def forward_request(node_id: int, endpoint: str, data: Dict) -> Dict:
    """Handles inter-node communication."""
    url = f"{get_node_ip(node_id)}/{endpoint}"
    

    try:
        start_time = time.time()
        response = requests.post(url, json=data, timeout=300)
        end_time = time.time()
        print(f"Node {NODE_NUMBER} -> Node {node_id} ({endpoint}): took {end_time - start_time:.4f}s")
        
        response.raise_for_status() 
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Forwarding error to Node {node_id} on {endpoint}: {e}")
        # Re-raise to be caught by the batching worker and set as Future exception
        raise


@app.route('/query', methods=['POST'])
def handle_query():
    """Client-facing endpoint (Node 0 only)."""    
    try:
        if NODE_NUMBER == 0: 
            worker_id = get_next_worker_url()

            if worker_id == 1: 
                return forward_request(1, "/query", request.json), 200


        data = request.json # request_id, query
        request_id = data.get('request_id')
        query = data.get('query')

        future_embed = embedder_service.submit(request_id, {'query': query})
        embedded_result = future_embed.result(timeout=300)

        # --- 2. FORWARD to Node 2 for FAISS RETRIEVAL ---
        # Node 2 expects a batch, we send a batch of size 1
        node2_result = forward_request(
            node_id=2,
            endpoint='retrieval_batch',
            data={'request_id': request_id, 'embedding': embedded_result['embedding']}
        ) # doc id: [int]
        # embedder_service, docfetch_service, rerank_service, llm_service, sentiment_service, safety_service, faiss_service
        future_doc_fetch = docfetch_service.submit(request_id, node2_result) 
        doc_fetch_results = future_doc_fetch.result(timeout=300)

        future_rerank = rerank_service.submit(request_id, {"query": query, "doc_ids": doc_fetch_results})
        rerank_results = future_rerank.result(timeout=300)

        zipped = list(zip(doc_fetch_results.values(), rerank_results['scores']))
        doc_scores = sorted(zipped, key=lambda x: x[1], reverse=True)
        context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc, _ in doc_scores[:3]])
        
        future_llm = llm_service.submit(request_id, {"query": query, "context": context})
        llm_results = future_llm.result(timeout=300)

        future_sentiment = sentiment_service.submit(request_id, llm_results)
        future_safety = safety_service.submit(request_id, llm_results)

        sentiment_result = future_sentiment.result(timeout=300)
        safety_result = future_safety.result(timeout=100)

        
        return jsonify({
            'request_id': request_id,
            'generated_response': llm_results['generated_response'],
            'sentiment': sentiment_result['sentiment'],
            'is_toxic': safety_result['is_toxic'],
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/retrieval_batch', methods=['POST'])
def handle_retrieval_batch():
    """FAISS Retrieval (index search only) request endpoint (Node 2 only)."""
    try:
        data = request.json # type: ignore
        
        # Must convert list back to numpy array for FAISS processing
        f = faiss_service.submit(data['request_id'], data)
            
        result = f.result(timeout=300)
        # print('faiss_result', result)
            
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# SERVER 
pipeline_stages = None
batching_service = None

if TOTAL_NODES >  2: 
    PIPELINE_WORKER_URLS = [0, 1]
else: 
    PIPELINE_WORKER_URLS = [0]
FAISS_NODE_URL = f"http://{NODE_2_IP}/retrieval_batch"

# Round-Robin state for pipeline workers
worker_cycler = itertools.cycle(PIPELINE_WORKER_URLS)
selection_lock = threading.Lock() 

def get_next_worker_url():
    """Gets the next node URL for the main pipeline processing."""
    with selection_lock:
        return next(worker_cycler)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


def main():
    global embedder_service, docfetch_service, rerank_service, llm_service, sentiment_service, safety_service, faiss_service    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Node {NODE_NUMBER} initializing on device: {device}")
    
    if NODE_NUMBER in [0, 1]:
        # NOTE: MAX_BATCH_SIZE and MAX_WAIT_TIME_MS must be defined in CONFIG
        
        embedder_service = Embedder(8, (150-30)//2, device)
        docfetch_service = DocFetch(8, (172-131)//2)
        rerank_service = Rerank(16, 60, device)
        llm_service = LLM(4, 500, device)
        sentiment_service = Sentiment(8, (600-170)//2, device)
        safety_service = Safety(4, (160-80)//2, device)
        
        print(f"Node {NODE_NUMBER} loaded full pipeline services.")
        
    # Node 2 loads only FAISS
    elif NODE_NUMBER == 2:
        MAX_SIZE = CONFIG.get('MAX_BATCH_SIZE', 128) # FAISS can handle larger batches
        MAX_WAIT = CONFIG.get('MAX_WAIT_TIME_MS', 10)
        
        faiss_service = FAISS(MAX_SIZE, MAX_WAIT)
        print(f"Node {NODE_NUMBER} loaded FAISS service only.")
    
    # Determine hostname and port
    if NODE_NUMBER == 0: ip_port = NODE_0_IP
    elif NODE_NUMBER == 1: ip_port = NODE_1_IP
    elif NODE_NUMBER == 2: ip_port = NODE_2_IP
    else: ip_port = 'localhost:8000'

    hostname, port_str = ip_port.split(':') if ':' in ip_port else ('localhost', '8000')
    port = int(port_str)
    
    print(f"\nStarting Flask server on {hostname}:{port}")
    # Use threaded=False if you want gunicorn for production, but Flask's built-in 
    # threaded server is fine for this project.
    app.run(host=hostname, port=port)


if __name__ == "__main__":
    main()