#!/usr/bin/env python3
"""
Client script with timing data collection.
Sends requests and records time-series data to CSV files similar to FAISS_mem.csv format.
Records: Time_Relative_s, Request_ID, Event_Tag, Elapsed_Time_s
"""

import os
import time
import requests
import json
import threading
import csv
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from collections import deque

# Read NODE_0_IP from environment variable
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
SERVER_URL = f"http://{NODE_0_IP}/query"

# Test queries
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

# Shared data structures
results = {}
results_lock = threading.Lock()
requests_sent = []
requests_lock = threading.Lock()

# Timing data collection
timing_data: deque = deque()
timing_lock = threading.Lock()
program_start_time = None


def record_timing_event(relative_time: float, request_id: str, event_tag: str, elapsed_time: Optional[float] = None):
    """Record a timing event to the data structure"""
    with timing_lock:
        timing_data.append({
            'Time_Relative_s': relative_time,
            'Request_ID': request_id,
            'Event_Tag': event_tag,
            'Elapsed_Time_s': elapsed_time if elapsed_time is not None else ''
        })


def send_request_async(request_id: str, query: str, send_time: float):
    """Send a single request to the server asynchronously"""
    global program_start_time
    
    try:
        # Record send event
        relative_time = time.perf_counter() - program_start_time
        record_timing_event(relative_time, request_id, 'SEND_REQUEST')
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending request {request_id}")
        print(f"Query: {query}")
        
        payload = {
            'request_id': request_id,
            'query': query
        }
        
        # Record send time (high precision)
        send_timestamp = time.perf_counter()
        start_time = time.time()
        response = requests.post(SERVER_URL, json=payload, timeout=300)
        receive_timestamp = time.perf_counter()
        elapsed_time = time.time() - start_time
        
        # Record receive event
        relative_time = time.perf_counter() - program_start_time
        record_timing_event(relative_time, request_id, 'RECEIVE_RESPONSE', elapsed_time)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Response received for {request_id} in {elapsed_time:.2f}s")
            print(f"  Generated Response: {result.get('generated_response', '')[:100]}...")
            print(f"  Sentiment: {result.get('sentiment')}")
            print(f"  Is Toxic: {result.get('is_toxic')}")
            
            # Record success event
            relative_time = time.perf_counter() - program_start_time
            record_timing_event(relative_time, request_id, 'SUCCESS', elapsed_time)
            
            with results_lock:
                results[request_id] = {
                    'result': result,
                    'elapsed_time': elapsed_time,
                    'send_time': send_time,
                    'send_timestamp': send_timestamp,
                    'receive_timestamp': receive_timestamp,
                    'success': True
                }
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for {request_id}: HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            
            # Record error event
            relative_time = time.perf_counter() - program_start_time
            record_timing_event(relative_time, request_id, f'ERROR_HTTP_{response.status_code}', elapsed_time)
            
            with results_lock:
                results[request_id] = {
                    'error': f"HTTP {response.status_code}",
                    'elapsed_time': elapsed_time,
                    'send_time': send_time,
                    'send_timestamp': send_timestamp if 'send_timestamp' in locals() else None,
                    'receive_timestamp': receive_timestamp if 'receive_timestamp' in locals() else None,
                    'success': False
                }
            
    except requests.exceptions.Timeout:
        relative_time = time.perf_counter() - program_start_time
        record_timing_event(relative_time, request_id, 'TIMEOUT')
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} timed out after 300s")
        with results_lock:
            results[request_id] = {
                'error': 'Timeout',
                'send_time': send_time,
                'success': False
            }
    except requests.exceptions.ConnectionError:
        relative_time = time.perf_counter() - program_start_time
        record_timing_event(relative_time, request_id, 'CONNECTION_ERROR')
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Failed to connect to server for {request_id}")
        with results_lock:
            results[request_id] = {
                'error': 'Connection error',
                'send_time': send_time,
                'success': False
            }
    except Exception as e:
        relative_time = time.perf_counter() - program_start_time
        record_timing_event(relative_time, request_id, f'ERROR_{str(e)[:20]}')
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for {request_id}: {str(e)}")
        with results_lock:
            results[request_id] = {
                'error': str(e),
                'send_time': send_time,
                'success': False
            }


def write_timing_csv(output_file: str = 'client_timing.csv'):
    """Write timing data to CSV file"""
    with timing_lock:
        if not timing_data:
            print("No timing data to write")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs('client_test_results', exist_ok=True)
        output_path = os.path.join('client_test_results', output_file)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Time_Relative_s', 'Request_ID', 'Event_Tag', 'Elapsed_Time_s'])
            writer.writeheader()
            writer.writerows(timing_data)
        
        print(f"\nTiming data written to {output_path}")
        print(f"Total events recorded: {len(timing_data)}")


def write_summary_csv(output_file: str = 'client_summary.csv'):
    """Write summary statistics to CSV file (similar to embedder_timing.csv format)"""
    with results_lock:
        successful_results = [r for r in results.values() if r.get('success', False) and 'elapsed_time' in r]
        
        if not successful_results:
            print("No successful requests to summarize")
            return
        
        elapsed_times = [r['elapsed_time'] for r in successful_results]
        
        import numpy as np
        avg_time = np.mean(elapsed_times)
        std_dev = np.std(elapsed_times)
        min_time = np.min(elapsed_times)
        max_time = np.max(elapsed_times)
        
        # Create output directory if it doesn't exist
        os.makedirs('client_test_results', exist_ok=True)
        output_path = os.path.join('client_test_results', output_file)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Total_Requests', 'Successful_Requests', 'Avg_Time_s', 'Std_Dev_s', 'Min_Time_s', 'Max_Time_s'])
            writer.writerow([
                len(results),
                len(successful_results),
                avg_time,
                std_dev,
                min_time,
                max_time
            ])
        
        print(f"Summary statistics written to {output_path}")


def write_request_timing_csv(output_file: str = 'client_request_timing.csv'):
    """Write per-request timing data (Request_ID, Elapsed_Time_s)"""
    with results_lock:
        successful_results = [r for r in results.values() if r.get('success', False) and 'elapsed_time' in r]
        
        if not successful_results:
            print("No successful requests to write")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs('client_test_results', exist_ok=True)
        output_path = os.path.join('client_test_results', output_file)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Request_ID', 'Elapsed_Time_s'])
            
            # Get request IDs in order
            with requests_lock:
                for req_info in requests_sent:
                    req_id = req_info['request_id']
                    if req_id in results and results[req_id].get('success', False):
                        elapsed = results[req_id].get('elapsed_time', '')
                        writer.writerow([req_id, elapsed])
        
        print(f"Request timing data written to {output_path}")


def main():
    """
    Main function: sends requests every 10 seconds for 1 minute
    Requests are sent at fixed intervals regardless of response time
    """
    global program_start_time
    program_start_time = time.perf_counter()
    
    # Record program start
    record_timing_event(0.0, 'PROGRAM', 'CLIENT_START')
    
    print("="*70)
    print("ML INFERENCE PIPELINE CLIENT (WITH TIMING)")
    print("="*70)
    print(f"Server URL: {SERVER_URL}")
    print(f"Sending 6 requests")
    print("="*70)
    
    # Check if server is healthy
    try:
        health_response = requests.get(f"http://{NODE_0_IP}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"Server is healthy: {health_response.json()}")
            relative_time = time.perf_counter() - program_start_time
            record_timing_event(relative_time, 'HEALTH_CHECK', 'SERVER_HEALTHY')
        else:
            print(f"Server health check returned status {health_response.status_code}")
            relative_time = time.perf_counter() - program_start_time
            record_timing_event(relative_time, 'HEALTH_CHECK', 'SERVER_UNHEALTHY')
    except:
        print(f"Could not reach server health endpoint")
        relative_time = time.perf_counter() - program_start_time
        record_timing_event(relative_time, 'HEALTH_CHECK', 'CONNECTION_FAILED')
    
    start_time = time.time()
    threads = []
    
    # Send 6 requests at 10-second intervals
    for i in range(6):
        # Calculate when this request should be sent
        target_send_time = start_time + (i * 10)
        
        # Wait until the target send time
        current_time = time.time()
        if current_time < target_send_time:
            wait_time = target_send_time - current_time
            if i > 0:
                print(f"\nWaiting {wait_time:.2f}s before next request...")
            time.sleep(wait_time)
        
        # Send request in a separate thread
        request_id = f"req_{int(time.time())}_{i}"
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        
        with requests_lock:
            requests_sent.append({
                'request_id': request_id,
                'query': query,
                'send_time': time.time()
            })
        
        thread = threading.Thread(
            target=send_request_async,
            args=(request_id, query, time.time())
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete (with a reasonable timeout)
    print(f"\n\nWaiting for all responses (up to 5 minutes)...")
    for thread in threads:
        thread.join(timeout=320)  # 5 min 20 sec to allow for some buffer
    
    # Record program end
    relative_time = time.perf_counter() - program_start_time
    record_timing_event(relative_time, 'PROGRAM', 'CLIENT_END')
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total requests sent: 6")
    
    with results_lock:
        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"Successful responses: {successful}")
        print(f"Failed requests: {6 - successful}")
    
    print(f"Total elapsed time: {total_time:.2f}s")
    
    with results_lock:
        if results:
            print("\nResults:")
            with requests_lock:
                for i, req_info in enumerate(requests_sent, 1):
                    req_id = req_info['request_id']
                    if req_id in results:
                        res_info = results[req_id]
                        print(f"\n{i}. Request ID: {req_id}")
                        print(f"   Query: {req_info['query'][:60]}...")
                        
                        if res_info.get('success'):
                            result = res_info['result']
                            print(f"   Success (took {res_info['elapsed_time']:.2f}s)")
                            print(f"   Sentiment: {result.get('sentiment')}")
                            print(f"   Is Toxic: {result.get('is_toxic')}")
                            print(f"   Response: {result.get('generated_response', '')[:80]}...")
                        else:
                            print(f"   Failed: {res_info.get('error', 'Unknown error')}")
                    else:
                        print(f"\n{i}. Request ID: {req_id}")
                        print(f"   ⏳ Still pending or not received")
    
    # Write CSV files
    print("\n" + "="*70)
    write_timing_csv('client_timing.csv')
    write_summary_csv('client_summary.csv')
    write_request_timing_csv('client_request_timing.csv')
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

