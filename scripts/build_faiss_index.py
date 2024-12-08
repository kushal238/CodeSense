# scripts/build_faiss_index.py

import json
import numpy as np
import faiss
import os

EMB_FILE = "data/embeddings/python_embeddings.json"
INDEX_FILE = "data/indexes/faiss_index.bin"
METADATA_FILE = "data/metadata/faiss_metadata.json"

def main():
    os.makedirs('data/indexes', exist_ok=True)
    os.makedirs('data/metadata', exist_ok=True)

    with open(EMB_FILE, 'r') as f:
        emb_data = json.load(f)

    vectors = np.array([d["embedding"] for d in emb_data], dtype='float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    meta = [{"id": d["id"], "filepath": d["filepath"], "repo": d["repo"], "url": d["url"], "func_name": d["func_name"]} for d in emb_data]

    with open(METADATA_FILE, 'w') as f:
        json.dump(meta, f)

    print(f"FAISS index saved to {INDEX_FILE}")
    print(f"Metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    main()
