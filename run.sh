#!/bin/bash

# Run script for ML Inference Pipeline
# This script will be executed on each node

echo "Starting pipeline on Node $NODE_NUMBER..."
# python3 pipeline.py

TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8000')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8000')

# frank, morgan, charles
TOTAL_NODES=3 \
NODE_NUMBER=2 \
NODE_0_IP=132.236.91.184:8000 \
NODE_1_IP=132.236.91.180:8000 \
NODE_2_IP=132.236.91.187:8000 \
python3 pipeline.py


#test if server is on
# after done, let someone else know to ping you 
NODE_0_IP=132.236.91.179:8000 python3 asdf.py

# ping command
curl http://132.236.91.187:8000/health

TOTAL_NODES=3 NODE_NUMBER=1 NODE_0_IP=132.236.91.182:8000 NODE_1_IP=132.236.91.188:8000 NODE_2_IP=132.236.91.187:8000 python3 asdf.py

NODE_0_IP=132.236.91.184:8000 python3 client.py