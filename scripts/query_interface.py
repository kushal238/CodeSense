# scripts/query_interface.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import faiss
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

MODEL_NAME = "microsoft/codebert-base"
INDEX_FILE = "data/indexes/faiss_index.bin"
METADATA_FILE = "data/metadata/faiss_metadata.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_embedding(model, tokenizer, text):
    """
    Generate a normalized embedding for the input text using mean pooling.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    # Normalize to unit vector
    emb = emb / np.linalg.norm(emb)
    return emb.astype('float32').reshape(1, -1)

def semantic_search(query, top_k=5):
    """
    Perform semantic search for the given query and return top_k results.
    """
    # Check if index and metadata files exist
    if not os.path.exists(INDEX_FILE):
        print(f"FAISS index file {INDEX_FILE} not found.")
        return []
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file {METADATA_FILE} not found.")
        return []

    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)

    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # Generate embedding for the query
    query_emb = get_embedding(model, tokenizer, query)

    # Perform search using inner product (cosine similarity)
    D, I = index.search(query_emb, top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(metadata):
            item = metadata[idx].copy()
            item["similarity"] = float(dist)  # Inner product as similarity score
            results.append(item)
    return results

def main():
    print("🔍 **CodeSense: Semantic Search for Developer Documentation**")
    query = input("Enter your query: ").strip()
    if not query:
        print("Please enter a valid query.")
        return

    results = semantic_search(query, top_k=5)
    if not results:
        print("No results found or index/metadata missing.")
        return

    for i, r in enumerate(results, 1):
        print("-----------")
        print(f"Result {i}:")
        print(f"Repo: {r['repo']}")
        print(f"Filepath: {r['filepath']}")
        print(f"URL: {r['url']}")
        print(f"Function Name: {r['func_name']}")
        print(f"Function Code:\n{r['whole_func_string']}")
        print(f"Similarity Score: {r['similarity']:.4f}")

if __name__ == "__main__":
    main()