#!/bin/bash

echo "Running install.sh..."

pip install -r requirements.txt
# python3 create_test_docs.py
python3 download_model.py > /dev/null || echo "non fatal error downloading model"

echo "Installation complete!"