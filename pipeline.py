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
from functools import wraps

def time_function(func):
    """
    A decorator that logs the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)        
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Function '{func.__name__}' executed in {duration:.4f} seconds.")        
        return result
    return wrapper

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8000')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8000')
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
    'MAX_BATCH_SIZE': 4, # this will be determined by memory
    'MAX_WAIT_TIME_MS': 5 # lower number --> lower latency but lower throughput
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
    
    def __init__(self, process_batch_func, max_size, max_wait_ms):
        self.input_queue = Queue()
        self.process_batch_func = process_batch_func
        self.max_size = max_size
        self.max_wait = max_wait_ms / 1000.0 # Convert ms to seconds
        
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print(f"Batching Service started for Node {NODE_NUMBER}")

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
                    timer_start = time.time()
            except Exception: 
                pass
            
            time_elapsed = time.time() - timer_start if timer_start else 0.0
            
            if (
                len(current_batch) >= self.max_size or # reached max size 
                (current_batch and time_elapsed >= self.max_wait) # waited long enough
            ):
                if (len(current_batch) >= self.max_size): 
                    print("\t Batch triggered: Queue Full after", time_elapsed)
                else: 
                    print("\t Batch triggered: Timed out with batch size", len(current_batch))
                
                batch_ids = [item.request_id for item in current_batch]
                batch_data = [item.data for item in current_batch]
                
                print(f"[{NODE_NUMBER}]Processing batch of size {len(current_batch)}")
                
                try:
                    # process_batch_func should return a list of results 
                    # matching the order of the inputs
                    results = self.process_batch_func(batch_ids, batch_data)
                    
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


class PipelineStages:
    """
    Cut up Monolith
    """
    def __init__(self, device):
        self.device = device

        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'


        self.embedding_model = None
        self.faiss_index = None # Node 1
        self.reranker_model = None # Node 2
        self.reranker_tokenizer = None # Node 2
        self.llm_model = None # Node 2
        self.llm_tokenizer = None # Node 2
        self.sentiment_classifier = None # Node 2
        self.safety_classifier = None # Node 2
        
        # Parallel executor for Node 2 analysis
        self.executor = ThreadPoolExecutor(max_workers=CONFIG['MAX_BATCH_SIZE']) 

    @time_function
    def load_models_for_node(self, node_id):
        """
        Load models for the current node.
        """
        print(f"Node {node_id}: Load Models")
        
        # Node 0: Embedder
        if node_id == 0:
            self.embedding_model = SentenceTransformer(self.embedding_model_name).to(self.device)
            print(f"Loaded Embedder: {self.embedding_model_name}.")
        
        # Node 1: FAISS Index
        if node_id == 1:
            if not os.path.exists(CONFIG['faiss_index_path']):
                raise FileNotFoundError("FAISS index not found on Node 1.")
            self.faiss_index = faiss.read_index(CONFIG['faiss_index_path'])
            print(f"Loaded FAISS Index.")

        # Node 2:  Reranker, LLM, Sentiment, Toxicity
        if node_id == 2:
            # reranker
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device).eval()
            print("Loaded Reranker.")

            # LLM
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16, 
            ).to(self.device)
            self.llm_model.eval()
            print("Loaded LLM.")

            # Sentiment
            self.sentiment_classifier = hf_pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                device=self.device
            )
            print("Loaded Sentiment Classifier.")

            # Safety
            self.safety_classifier = hf_pipeline(
                "text-classification",
                model=self.safety_model_name,
                device=self.device
            )
            print("Loaded Safety Classifier.")
            
            

        
        print(f"Node {node_id}: Model loading complete.")

    # NODE 0: EMBEDDING
    @time_function
    def _embed_batch(self, queries): 
        return self.embedding_model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    @time_function
    def process_embedding_batch(self, batch_ids: List[str], batch_data: List[Dict]) -> List[Dict]:
        """Input: [{'query': str}] -> Output: [{'embedding': np.ndarray, 'query': str}]"""
        queries = [data['query'] for data in batch_data]
        embeddings = self._embed_batch(queries)
        
        results = []
        for query, embedding in zip(queries, embeddings):
            # Convert numpy array to list for JSON serialization in transit
            results.append({
                'query': query, 
                'embedding': embedding.tolist() 
            })
        return results

    # NODE 1 : FAISS Retrieval (No Doc Fetch)
    @time_function
    def process_retrieval_batch(self, batch_ids: List[str], batch_data: List[Dict]) -> List[Dict]:
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
                'query': batch_data[idx]['query'],
                'embedding': batch_data[idx]['embedding'], # can probably cut embeddings and query
                'doc_ids': doc_ids
            })
            
        return results

    
    # NODE 2 UTILITIES (Doc Fetch, Rerank)
    @time_function
    def _fetch_documents_batch(self, unique_doc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Fetch all unique documents requested by the batch in a single query.
        Input: list of unique doc_ids
        Output: {doc_id: {'doc_id': int, 'title': str, 'content': str, 'category': str}}
        """
        if not unique_doc_ids:
            return {}
            
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path, check_same_thread=False) # all threads read only
        cursor = conn.cursor()
        
        doc_map = {}
        doc_ids_str = ','.join('?' * len(unique_doc_ids))
        
        # Single batched SQL query
        cursor.execute(
            f'SELECT doc_id, title, content, category FROM documents WHERE doc_id IN ({doc_ids_str})',
            unique_doc_ids
        )
        
        for result in cursor.fetchall():
            doc_id, title, content, category = result
            doc_map[doc_id] = {
                'doc_id': doc_id,
                'title': title,
                'content': content,
                'category': category
            }
        conn.close()
        return doc_map
    @time_function
    def _rerank_documents_batch(self, query_document_pairs: List[List[str]]) -> List[float]:
        """
        Rerank retrieved documents using a single batch inference call.
        Input: [[query_1, doc_1], [query_1, doc_2], [query_2, doc_1], ...]
        Output: [score_1, score_2, score_3, ...]
        """
        if not query_document_pairs:
            return []
        
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                query_document_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=CONFIG['truncate_length']
            ).to(self.device) # type: ignore
            # Reranker scores are logits; we return them as float list
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float().tolist() # type: ignore
        
        return scores        
    @time_function
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Analyze sentiment for a batch of responses."""
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.sentiment_classifier(truncated_texts) # type: ignore
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        return [sentiment_map.get(res['label'], 'FAILURE') for res in raw_results]
    @time_function
    def _filter_safety_batch(self, texts: List[str]) -> List[str]:
        """Filter response for safety for a batch of responses."""
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.safety_classifier(truncated_texts) 
        
        results = []
        for res in raw_results:
            is_toxic = res['score'] > 0.5
            results.append("true" if is_toxic else "false")
            
        return results

    @time_function
    def process_llm_rag_analysis_batch(self, batch_ids: List[str], batch_data: List[Dict]) -> List[Dict]:
        """
        Main Node 2 logic: RAG (Unified Fetch/Batch Rerank), LLM, and Analysis, optimized for CPU throughput.
        Input: [{'query': str, 'embedding': list, 'doc_ids': list}]
        Output: final response fields
        """
        
        # --- 1. Collect Unique IDs (Set Operation) ---
        all_requested_doc_ids = set()
        for data in batch_data:
            all_requested_doc_ids.update(data['doc_ids'])

        # --- 2. Single Batch Document Fetching ---
        # Convert set to list for the SQL query and perform the single fetch.
        unique_doc_ids_list = list(all_requested_doc_ids)
        # master_doc_map: {doc_id: full_document_content}
        master_doc_map = self._fetch_documents_batch(unique_doc_ids_list)
        
        # --- 3. Reranking Preparation (Map Content Back) ---
        
        query_document_pairs = []
        pair_counts = []
        
        for data in batch_data:
            query = data['query']
            request_pairs = []
            
            # Use master_doc_map to look up content for this request's specific doc_ids
            for doc_id in data['doc_ids']:
                doc = master_doc_map.get(doc_id)
                if doc: # Only add if the document was successfully fetched
                    request_pairs.append([query, doc['content']])
            
            query_document_pairs.extend(request_pairs)
            pair_counts.append(len(request_pairs)) # Count of pairs for this request

        # --- 4. Reranking (Model Batching) ---
        
        # Single batch inference call for reranking across ALL requests
        rerank_scores = self._rerank_documents_batch(query_document_pairs)
        
        # --- 5. Context Construction ---
        
        context_list = []
        score_idx = 0
        
        for i, data in enumerate(batch_data):
            count = pair_counts[i]
            
            # Reconstruct the documents and scores for the current request
            # Note: We need to pull the original document objects *again* to get titles/metadata, 
            # using the original doc_ids and the master_doc_map.
            
            # Re-associate scores with documents based on the original doc_ids list
            current_scores = rerank_scores[score_idx : score_idx + count]
            
            # This assumes the order of documents in query_document_pairs for this request matches 
            # the order of valid documents in data['doc_ids'].
            doc_scores = []
            valid_docs = [master_doc_map[doc_id] for doc_id in data['doc_ids'] if doc_id in master_doc_map]
            
            for doc, score in zip(valid_docs, current_scores):
                doc_scores.append((doc, score))

            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Build context from top 3 reranked documents
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc, _ in doc_scores[:3]])
            context_list.append(context)
            
            score_idx += count


        # --- 6. LLM Generation (Model Batching) ---
        
        # ... (LLM Generation logic remains the same, using context_list) ...
        llm_input_texts = []
        for query, context in zip([d['query'] for d in batch_data], context_list):
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
        
        
        # --- 7. Analysis (Model Batching) ---
        
        sentiment_results = self._analyze_sentiment_batch(generated_responses)
        safety_results = self._filter_safety_batch(generated_responses)
        
        # --- 8. Final Results Consolidation ---
        
        final_results = []
        for generated_response, sentiment, is_toxic in zip(generated_responses, sentiment_results, safety_results):
            final_results.append({
                'generated_response': generated_response,
                'sentiment': sentiment,
                'is_toxic': is_toxic
            })
            
        return final_results

# SERVER 

app = Flask(__name__)
pipeline_stages = None
batching_service = None

def get_node_ip(node_id):
    """Helper to get IP of a specific node."""
    if node_id == 0: 
        return f"http://{NODE_0_IP}"
    if node_id == 1: 
        return f"http://{NODE_1_IP}"
    if node_id == 2: 
        return f"http://{NODE_2_IP}"
    raise ValueError("Invalid node ID")

def forward_request_batch(node_id: int, endpoint: str, batch_data: List[Dict]) -> List[Dict]:
    """Handles inter-node communication."""
    url = f"{get_node_ip(node_id)}/{endpoint}"
    
    payload = [{'request_id': data['id'], 'data': data['data']} for data in batch_data]

    try:
        start_time = time.time()
        response = requests.post(url, json={'batch': payload}, timeout=300)
        end_time = time.time()
        print(f"Node {NODE_NUMBER} -> Node {node_id} ({endpoint}): {len(batch_data)} items, took {end_time - start_time:.4f}s")
        
        response.raise_for_status() 
        return response.json()['results']
        
    except requests.exceptions.RequestException as e:
        print(f"Forwarding error to Node {node_id} on {endpoint}: {e}")
        # Re-raise to be caught by the batching worker and set as Future exception
        raise

# NODE 0: Entrypoint & Orchestration

@app.route('/query', methods=['POST'])
def handle_query():
    """Client-facing endpoint (Node 0 only)."""
    if NODE_NUMBER != 0:
        return jsonify({'error': 'Only Node 0 handles client queries'}), 400
    
    try:
        data = request.json
        request_id = data.get('request_id')
        query = data.get('query')
        
        if not request_id:
            return jsonify({'error': 'Missing request_id'}), 400
        if not query: 
            return jsonify({'error': "Missing query"}), 400
        start_time = time.time()
        
        # 1. Submit to Node 0's local embedding batcher
        future_embed = batching_service.submit(
            request_id,
            {'query': query}
        )
        
        # Block until embedding is ready
        embedded_result = future_embed.result(timeout=300)
        
        # 2. Forward to Node 1 for FAISS retrieval (returns doc IDs)
        node1_result = forward_request_batch(
            node_id=1,
            endpoint='retrieval_batch',
            batch_data=[{'id': request_id, 'data': embedded_result}]
        )[0] 

        # 3. Forward to Node 2 for RAG (Fetch/Rerank) + LLM/Analysis
        node2_result = forward_request_batch(
            node_id=2,
            endpoint='llm_rag_analysis_batch',
            batch_data=[{'id': request_id, 'data': node1_result}]
        )[0]

        # 4. Final Response
        processing_time = time.time() - start_time
        
        return jsonify({
            'request_id': request_id,
            'generated_response': node2_result['generated_response'],
            'sentiment': node2_result['sentiment'],
            'is_toxic': node2_result['is_toxic'],
            'processing_time': f"{processing_time:.2f}s"
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NODE 1: FAISS Indexing
@app.route('/retrieval_batch', methods=['POST'])
def handle_retrieval_batch():
    """FAISS Retrieval (index search only) request endpoint (Node 1 only)."""
    if NODE_NUMBER != 1:
        return jsonify({'error': 'FAISS retrieval runs only on Node 1'}), 403

    # glue code
    def process_retrieval_batch_wrapper(batch_ids, batch_data):
        return pipeline_stages.process_retrieval_batch(batch_ids, batch_data)

    try:
        data = request.json.get('batch', [])
        
        
        futures = []
        for item in data:
            # Need to convert list back to numpy array for the internal stage
            item['data']['embedding'] = np.array(item['data']['embedding'])
            futures.append(batching_service.submit(item['request_id'], item['data']))
            
        results_with_id = []
        for f, item in zip(futures, data):
            result = f.result(timeout=300)
            # Re-package with request_id for the next hop
            results_with_id.append({'request_id': item['request_id'], **result})
        
        return jsonify({'results': results_with_id}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Node 2
@app.route('/llm_rag_analysis_batch', methods=['POST'])
def handle_llm_rag_analysis_batch():
    """LLM + RAG + Analysis request endpoint (Node 2 only)."""
    if NODE_NUMBER != 2:
        return jsonify({'error': 'LLM/RAG/Analysis runs only on Node 2'}), 403

    def process_llm_rag_analysis_batch_wrapper(batch_ids, batch_data):
        return pipeline_stages.process_llm_rag_analysis_batch(batch_ids, batch_data)

    try:
        data = request.json.get('batch', [])
        
        # Submit to local batcher
        futures = []
        for item in data:
            futures.append(batching_service.submit(item['request_id'], item['data']))
            
        results_with_id = []
        for f, item in zip(futures, data):
            result = f.result(timeout=300)
            # Final result is already in the expected format, just add ID
            results_with_id.append({'request_id': item['request_id'], **result})
        
        return jsonify({'results': results_with_id}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


def main():
    global pipeline_stages, batching_service
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Node {NODE_NUMBER} initializing on device: {device}")
    
    pipeline_stages = PipelineStages(device)
    pipeline_stages.load_models_for_node(NODE_NUMBER)
    
    # Define the appropriate processing function for the local batcher
    # Node 1 now runs the FAISS lookup only
    # Node 2 now runs RAG (Fetch/Rerank) + LLM + Analysis
    processing_map = {
        0: pipeline_stages.process_embedding_batch,
        1: pipeline_stages.process_retrieval_batch,
        2: pipeline_stages.process_llm_rag_analysis_batch
    }
    
    if NODE_NUMBER in processing_map:
        batching_service = BatchingService(
            process_batch_func=processing_map[NODE_NUMBER],
            max_size=CONFIG['MAX_BATCH_SIZE'],
            max_wait_ms=CONFIG['MAX_WAIT_TIME_MS']
        )
    else:
        print("Warning: Node number not configured for any service.")

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
    app.run(host=hostname, port=port, threaded=True)


if __name__ == "__main__":
    main()