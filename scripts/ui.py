#scripts/ui.py
import streamlit as st
import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "krlvi/sentence-t5-base-nlpl-code_search_net"
INDEX_FILE = "data/indexes/faiss_index.bin"
METADATA_FILE = "data/metadata/faiss_metadata.json"

@st.cache_resource
def load_model_index():
    """
    Loads and returns the SentenceTransformer model, FAISS index, and metadata.
    """
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    return model, index, metadata

def semantic_search(model, index, metadata, query, top_k=5):
    """
    Perform semantic search using the model, index, and metadata.

    Args:
        model: The SentenceTransformer model.
        index: The FAISS index.
        metadata: Metadata for the indexed items.
        query: The input query string.
        top_k: Number of results to return.

    Returns:
        A list of top-k results with metadata, distance, and score.
    """
    query_emb = model.encode([query], convert_to_tensor=True)
    query_emb = query_emb.cpu().numpy().astype('float32')
    D, I = index.search(query_emb, top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        item = metadata[idx]
        # Add distance as before
        item["distance"] = float(dist)
        # Convert the squared distance to a similarity score
        item["score"] = 1 / (1 + float(dist))
        results.append(item)

    return results

st.title("CodeSense: Semantic Code Search (FAISS + SentenceTransformers)")

query = st.text_input("Enter your query:", "How to load a YAML file?")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    model, index, metadata = load_model_index()
    results = semantic_search(model, index, metadata, query, top_k=top_k)
    for r in results:
        st.markdown("---")
        st.write(f"**Repo:** {r['repo']}")
        st.write(f"**File:** {r['filepath']}")
        st.write(f"**URL:** {r['url']}")
        st.write(f"**Function:** {r['func_name']}")
        st.write(f"**Score:** {r['score']}")