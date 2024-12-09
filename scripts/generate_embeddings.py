# scripts/generate_embeddings.py

import json
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import faiss  # Ensure FAISS is installed

MODEL_NAME = "microsoft/codebert-base"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
MAX_LENGTH = 256  # Adjust based on your data

def get_embeddings(model, tokenizer, texts):
    """
    Generate normalized embeddings for a batch of texts using mean pooling.
    """
    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings.astype('float32')

def main():
    # Load the cleaned dataset
    dataset_path = 'data/cleaned_train_sample.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset file {dataset_path} not found.")
        return
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} samples.")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # Prepare texts by combining documentation and code
    texts = (df['func_documentation_string'].fillna('') + ' ' + df['func_code_string'].fillna('')).tolist()

    # Generate embeddings in batches
    embeddings = []
    metadata = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_embeddings = get_embeddings(model, tokenizer, batch_texts)
        embeddings.append(batch_embeddings)
        # Collect metadata for this batch
        batch_metadata = df.iloc[i:i+BATCH_SIZE][['repository_name', 'func_path_in_repository', 'func_code_url', 'func_name', 'whole_func_string']].to_dict(orient='records')
        metadata.extend(batch_metadata)
    
    # Stack all embeddings into a single numpy array
    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Normalize embeddings to unit vectors (already done in get_embeddings)
    # Build FAISS index using Inner Product (which simulates cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    print("Building FAISS index...")
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # Save FAISS index
    index_dir = 'data/indexes'
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    # Save metadata
    metadata_dir = 'data/metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, "faiss_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()
