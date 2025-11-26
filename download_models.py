#!/usr/bin/env python3
"""
Pre-download all models to avoid timeout during first request
"""

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Downloading all required models...")
print("="*60)

models = [
    ("Embedding model", "BAAI/bge-base-en-v1.5", "SentenceTransformer"),
    ("Reranker model", "BAAI/bge-reranker-base", "AutoModelForSequenceClassification"),
    ("LLM model", "Qwen/Qwen2.5-0.5B-Instruct", "AutoModelForCausalLM"),
    ("Sentiment model", "nlptown/bert-base-multilingual-uncased-sentiment", "AutoModelForSequenceClassification"),
    ("Safety model", "unitary/toxic-bert", "AutoModelForSequenceClassification"),
]

for name, model_name, model_type in models:
    print(f"\n[{name}] Downloading {model_name}...")
    try:
        if model_type == "SentenceTransformer":
            model = SentenceTransformer(model_name)
            print(f"  ✓ {name} downloaded successfully")
        elif model_type == "AutoModelForCausalLM":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"  ✓ {name} downloaded successfully")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print(f"  ✓ {name} downloaded successfully")
        del model
        if 'tokenizer' in locals():
            del tokenizer
    except Exception as e:
        print(f"  ✗ Error downloading {name}: {e}")

print("\n" + "="*60)
print("Model download complete!")
print("="*60)

