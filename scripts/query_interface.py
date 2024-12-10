import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "krlvi/sentence-t5-base-nlpl-code_search_net"
INDEX_FILE = "data/indexes/faiss_index.bin"
METADATA_FILE = "data/metadata/faiss_metadata.json"

def semantic_search(query, top_k=5):
    # Load FAISS index and metadata
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    # Load model
    model = SentenceTransformer(MODEL_NAME)

    # Encode query
    query_emb = model.encode([query], convert_to_tensor=True)
    query_emb = query_emb.cpu().numpy().astype('float32')

    # Search
    D, I = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        item = metadata[idx]
        item["distance"] = float(dist)
        results.append(item)
    return results

if __name__ == "__main__":
    query = input("Enter your query: ")
    results = semantic_search(query, top_k=5)
    for r in results:
        print("-----------")
        print(f"Repo: {r['repo']}")
        print(f"Filepath: {r['filepath']}")
        print(f"URL: {r['url']}")
        print(f"Function: {r['func_name']}")
        print(f"Distance: {r['distance']}")
