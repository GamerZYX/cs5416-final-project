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


class test_TOXIC: 
    NAME = "TOXIC"
    def __init__(self, tracker): 
        self.tracker = tracker
        
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

        with open("./test_data/asdf.json", 'r') as f: 
            self.generated_responses = json.load(f)

        self.tracker.set_event(f"{self.NAME}_LOAD_START")
        self.sentiment_classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            # device=self.device
        )
        self.tracker.set_event(f"{self.NAME}_LOAD_END")
        print("Loaded Sentiment Classifier.")



    def run_test(self, batch_size, n_iters=10):

        random_choices = np.random.choice(a=len(self.generated_responses), size=batch_size, replace=True)
        texts = [self.generated_responses[i] for i in random_choices]
        times = []

        self.tracker.set_event(f"BATCH_SIZE_START: {batch_size}") # Mark the start of the batch test
        for _ in range(n_iters): 
            start = time.perf_counter()

            truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
            raw_results = self.sentiment_classifier(truncated_texts)
            sentiment_map = {
                '1 star': 'very negative',
                '2 stars': 'negative',
                '3 stars': 'neutral',
                '4 stars': 'positive',
                '5 stars': 'very positive'
            }
            _ = [sentiment_map.get(res['label'], 'neutral') for res in raw_results]
        
            end = time.perf_counter()
            times.append((end - start))
        self.tracker.set_event(f"BATCH_SIZE_END: {batch_size}") # Mark the end of the batch test

        
        average_time = np.mean(times)
        std_dev = np.std(times)
        return average_time, std_dev


if __name__=="__main__": 
    # 1. Initialize and start the Memory Tracker
    memory_tracker = MemoryTracker(interval=0.1, output_file=f"./test_results/{test_TOXIC.NAME}_mem.csv")
    memory_tracker.start()
    
    # 2. Initialize the test runner, which triggers model loading events
    try:
        tester = test_TOXIC(memory_tracker)
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
        avg, std = tester.run_test(i)
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
     

