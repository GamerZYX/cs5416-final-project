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
import random
import csv
import sys
from pathlib import Path


# add parent to pathdir
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from memtracker import MemoryTracker
# Read environment variables
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', '../faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', '../documents/')

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


class test_DOC_FETCH: 
    NAME = "DOC_FETCH"
    
    def __init__(self, tracker, NROWS = 100000): 
        self.tracker = tracker
        self.N_ROWS = 100000 # this might need to change for large docs  
        
        db_path = f"{CONFIG['documents_path']}/documents.db"
        self.tracker.set_event(f"{self.NAME}_LOAD_START")
        self.conn = sqlite3.connect(db_path, check_same_thread=False) # all threads read only
        self.tracker.set_event(f"{self.NAME}_LOAD_END")
        print(f"Loaded DB.")

    def run_test(self, batch_size, n_iters=10):
        """
        Main Node 1 code (FAISS index lookup only)
        Input: [{'embedding': list, 'query': str}]
        Output: [{'query': str, 'embedding': list, 'doc_ids': list}]
        """

        times = []
        unique_doc_ids = random.sample(range(0, self.N_ROWS), batch_size)

            
        self.tracker.set_event(f"BATCH_SIZE_START: {batch_size}") # Mark the start of the batch test
        cursor = self.conn.cursor()
        doc_ids_str = ','.join('?' * len(unique_doc_ids))
        
        # Single batched SQL query
        for i in range(n_iters):
            doc_map = {}
            start = time.perf_counter()
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

            end = time.perf_counter()
            # if i ==0: 
            #     with open("./test_data/test_docs.json", 'w') as f: 
            #         json.dump(doc_map, f)
            times.append((end - start))
        self.tracker.set_event(f"BATCH_SIZE_END: {batch_size}") # Mark the end of the batch test

        
        average_time = np.mean(times)
        std_dev = np.std(times)
        return average_time, std_dev


if __name__=="__main__": 
    # 1. Initialize and start the Memory Tracker
    memory_tracker = MemoryTracker(interval=0.1, output_file=f"./test_results/{test_DOC_FETCH.NAME}_mem.csv")
    memory_tracker.start()
    
    # 2. Initialize the test runner, which triggers model loading events
    try:
        tester = test_DOC_FETCH(memory_tracker)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Stop and log memory data even if model loading fails
        memory_tracker.stop()
        memory_tracker.join()
        memory_tracker.write_csv()
        exit(1)


    # 3. Run the tests, triggering batch events
    res = []
    # Test batch sizes 1, 2, 4, 8, 16
    for i in (1, 2, 4, 8, 16): 
        avg, std = tester.run_test(i*CONFIG['retrieval_k'])
        res.append((i, avg, std))
    
    # 4. Write embedding timing results
    with open(f"./test_results/{tester.NAME}_timing.csv", "w", newline='') as f: 
        writer = csv.writer(f)
        writer.writerow(["Batch_Size", "Avg_Time_s", "Std_Dev_s"])
        writer.writerows(res)
    print("Embedder timing results written to embedder_timing.csv")
    
    # 5. Stop the Memory Tracker and write its log
    memory_tracker.set_event("PROGRAM_END")
    memory_tracker.stop()
    memory_tracker.join() # Wait for the thread to finish writing data
    memory_tracker.write_csv()
     

