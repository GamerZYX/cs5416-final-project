#!/usr/bin/env python3
"""
Pre-download Hugging Face models to cache them locally.
Matches the models used in the monolithic pipeline.
"""
import sys
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import torch

# Model names (matching the pipeline configuration)
MODELS = {
    'embedding': 'BAAI/bge-base-en-v1.5',  # 0.1B param - SentenceTransformer
    'reranker': 'BAAI/bge-reranker-base',  # 0.3B params - AutoModelForSequenceClassification
    'llm': 'Qwen/Qwen2.5-0.5B-Instruct',  # 0.5B params - AutoModelForCausalLM
    'sentiment': 'nlptown/bert-base-multilingual-uncased-sentiment',  # 0.2B - pipeline
    'safety': 'unitary/toxic-bert',  # 0.1B - pipeline
}

def download_embedding_model(model_name):
    """Download embedding model using SentenceTransformer"""
    print(f"\n{'='*60}")
    print(f"Downloading Embedding Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        print("  → Downloading via SentenceTransformer...")
        model = SentenceTransformer(model_name)
        print(f"  ✓ Successfully downloaded {model_name}")
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
        del model
        return True
    except Exception as e:
        print(f"  ✗ Error downloading {model_name}: {e}")
        return False

def download_reranker_model(model_name):
    """Download reranker model"""
    print(f"\n{'='*60}")
    print(f"Downloading Reranker Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        print("  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("  → Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        print(f"  ✓ Successfully downloaded {model_name}")
        del model, tokenizer
        return True
    except Exception as e:
        print(f"  ✗ Error downloading {model_name}: {e}")
        return False

def download_llm_model(model_name):
    """Download LLM model"""
    print(f"\n{'='*60}")
    print(f"Downloading LLM Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        print("  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("  → Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        
        print(f"  ✓ Successfully downloaded {model_name}")
        del model, tokenizer
        return True
    except Exception as e:
        print(f"  ✗ Error downloading {model_name}: {e}")
        return False

def download_pipeline_model(model_name, task):
    """Download models used via HuggingFace pipeline"""
    print(f"\n{'='*60}")
    print(f"Downloading Pipeline Model: {model_name}")
    print(f"Task: {task}")
    print(f"{'='*60}")
    
    try:
        print("  → Downloading via pipeline...")
        pipe = hf_pipeline(task, model=model_name)
        
        print(f"  ✓ Successfully downloaded {model_name}")
        del pipe
        return True
    except Exception as e:
        print(f"  ✗ Error downloading {model_name}: {e}")
        return False

def main():
    print("Hugging Face Model Downloader for Monolithic Pipeline")
    print("=" * 60)
    print(f"Models to download: {len(MODELS)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    successful = 0
    failed = 0
    
    # Download embedding model (SentenceTransformer)
    if download_embedding_model(MODELS['embedding']):
        successful += 1
    else:
        failed += 1
    
    # Download reranker model
    if download_reranker_model(MODELS['reranker']):
        successful += 1
    else:
        failed += 1
    
    # Download LLM model
    if download_llm_model(MODELS['llm']):
        successful += 1
    else:
        failed += 1
    
    # Download sentiment analysis model (via pipeline)
    if download_pipeline_model(MODELS['sentiment'], "sentiment-analysis"):
        successful += 1
    else:
        failed += 1
    
    # Download safety/toxicity model (via pipeline)
    if download_pipeline_model(MODELS['safety'], "text-classification"):
        successful += 1
    else:
        failed += 1
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {len(MODELS)}")
    
    if successful == len(MODELS):
        print("\nAll models downloaded successfully!")
        print("Your pipeline is ready to run without downloads.")
    else:
        print(f"\n{failed} model(s) failed to download.")
        print("Please check the errors above and try again.")

    print("\nModels are cached in: ~/.cache/huggingface/")
    sys.exit(failed)

if __name__ == "__main__":
    main()