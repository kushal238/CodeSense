# scripts/ui.py

import streamlit as st
import json
import faiss
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

MODEL_NAME = "microsoft/codebert-base"
INDEX_FILE = "data/indexes/faiss_index.bin"
METADATA_FILE = "data/metadata/faiss_metadata.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model_index():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    return model, tokenizer, index, metadata

def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()
    return emb

def semantic_search(query, top_k=5):
    model, tokenizer, index, metadata = load_model_index()
    query_emb = get_embedding(model, tokenizer, query).astype('float32').reshape(1,-1)
    D, I = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        item = metadata[idx]
        item["distance"] = float(dist)
        results.append(item)
    return results

st.title("CodeSense: Semantic Code Search")
query = st.text_input("Enter your query:", "How to create a virtual environment in Python?")
top_k = st.slider("Number of results", 1, 10, 5)
if st.button("Search"):
    results = semantic_search(query, top_k=top_k)
    for r in results:
        st.markdown("---")
        st.write(f"**Repo:** {r['repo']}")
        st.write(f"**File:** {r['filepath']}")
        st.write(f"**URL:** {r['url']}")
        st.write(f"**Function:** {r['func_name']}")
        st.write(f"**Distance:** {r['distance']}")
