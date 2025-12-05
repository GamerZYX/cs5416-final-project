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


class test_llm: 
    NAME = "LLM"

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
        self.TOP_N = 3


        with open("./test_data/test_docs.json", 'r') as f: 
            self.docs = json.load(f)
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

        self.tracker.set_event(f"{self.NAME}_LOAD_START")    

        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16, 
        )
        self.llm_model.eval()
        print("Loaded LLM.")

        self.tracker.set_event(f"{self.NAME}_LOAD_END")
        print(f"Loaded {self.NAME}.")

    def run_test(self, batch_size, n_iters=10):
        times = []

        context_list = []
        queries = random.choices(self.TEST_QUERIES, k=batch_size)
        for _ in queries: 
            context_list.append([random.choices(list(self.docs.values()), k=self.TOP_N)])
        for _ in range(batch_size): 
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc, _ in doc_scores[:self.TOP_N]])
            context_list.append(context)


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

        query_document_pairs = []
        for _ in range(batch_size): 
            query = random.choice(test_llm.TEST_QUERIES)
            docs = random.choices(list(self.docs.values()), k=CONFIG["retrieval_k"])
            query_document_pairs.extend([query, doc['content']] for doc in docs)

            
        self.tracker.set_event(f"BATCH_SIZE_START: {batch_size}") # Mark the start of the batch test
        
        for i in range(n_iters):
            start = time.perf_counter()

            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    query_document_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                )
                scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float().tolist() # type: ignore


            end = time.perf_counter()
            times.append((end - start))
        self.tracker.set_event(f"BATCH_SIZE_END: {batch_size}") # Mark the end of the batch test

        
        average_time = np.mean(times)
        std_dev = np.std(times)
        return average_time, std_dev


if __name__=="__main__": 
    # 1. Initialize and start the Memory Tracker
    memory_tracker = MemoryTracker(interval=0.1, output_file=f"./test_results/{test_llm.NAME}_mem.csv")
    memory_tracker.start()
    
    # 2. Initialize the test runner, which triggers model loading events
    try:
        tester = test_llm(memory_tracker)
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
    print("results written to csv")
    
    # 5. Stop the Memory Tracker and write its log
    memory_tracker.set_event("PROGRAM_END")
    memory_tracker.stop()
    memory_tracker.join() # Wait for the thread to finish writing data
    memory_tracker.write_csv()
     

