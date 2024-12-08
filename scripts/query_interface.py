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
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()
    return emb

def semantic_search(query, top_k=5):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    query_emb = get_embedding(model, tokenizer, query).astype('float32').reshape(1,-1)
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
        print(f"Function string: {r['whole_func_string']}")
        print(f"Distance: {r['distance']}")
#