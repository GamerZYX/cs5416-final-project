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


class test_embeddings: 
    TEST_QUERIES = [
        "How do I return a defective product?",
        "What is your refund policy?",
        "My order hasn't arrived yet, tracking number is ABC123",
        "How do I update my billing information?",
        "Is there a warranty on electronic items?",
        "Can I change my shipping address after placing an order?",
        "What payment methods do you accept?",
        "How long does shipping typically take?"
    ]
    def __init__(self, tracker): 
        self.tracker = tracker
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        
        self.tracker.set_event(f"MODEL_LOAD_START: {self.embedding_model_name}")
        print(f'Loading Embedder: ')
        # The actual model loading happens here
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        self.tracker.set_event(f"MODEL_LOAD_END: {self.embedding_model_name}")
        print(f"Loaded Embedder: {self.embedding_model_name}.")

    def run_test(self, batch_size, n_iters=10, r_sample=True):
        queries = []
        if r_sample: 
            queries = [random.choice(test_embeddings.TEST_QUERIES) for _ in range(batch_size)]
        else: 
            queries = [test_embeddings.TEST_QUERIES[i % len(test_embeddings.TEST_QUERIES)] for i in range(batch_size)]
        
        times = []
        
        self.tracker.set_event(f"BATCH_SIZE_START: {batch_size}") # Mark the start of the batch test
        
        for _ in range(n_iters):
            start = time.perf_counter()
            embeddings = self.embedding_model.encode(
                queries,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            end = time.perf_counter()
            times.append((end - start))
            
        self.tracker.set_event(f"BATCH_SIZE_END: {batch_size}") # Mark the end of the batch test

        average_time = np.mean(times)
        std_dev = np.std(times)
        return average_time, std_dev

if __name__=="__main__": 
    # 1. Initialize and start the Memory Tracker
    memory_tracker = MemoryTracker(interval=0.1, output_file="./test_results/embedding_mem.csv")
    memory_tracker.start()
    
    # 2. Initialize the test runner, which triggers model loading events
    try:
        te = test_embeddings(memory_tracker)
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
        avg, std = te.run_test(i)
        res.append((i, avg, std))
    
    # 4. Write embedding timing results
    with open("./test_results/embedder_timing.csv", "w", newline='') as f: 
        writer = csv.writer(f)
        writer.writerow(["Batch_Size", "Avg_Time_s", "Std_Dev_s"])
        writer.writerows(res)
    print("Embedder timing results written to embedder_timing.csv")
    
    # 5. Stop the Memory Tracker and write its log
    memory_tracker.set_event("PROGRAM_END")
    memory_tracker.stop()
    memory_tracker.join() # Wait for the thread to finish writing data
    memory_tracker.write_csv()