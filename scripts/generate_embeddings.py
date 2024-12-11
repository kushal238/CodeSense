# scripts/generate_embeddings.py
import json
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "krlvi/sentence-t5-base-nlpl-code_search_net"

def main():
    # Load the cleaned dataset
    df = pd.read_csv('data/cleaned_train_sample.csv') # change directory

    # Load the SentenceTransformer model
    model = SentenceTransformer(MODEL_NAME)

    # Encode documentation + code together
    texts = (df['func_documentation_string'].fillna("") + " " + df['func_code_string'].fillna("")).tolist()

    print("Generating embeddings...")
    code_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    code_embeddings = code_embeddings.cpu().numpy() # implement or not based on your device


    """
    Creating a list of dictionaries, with each dictionary containing metadata for the function
    and the function's semantic embedding. this helps us associate metadata with embeddings and
    enables easy storage of embeddings in a JSON file for later use.
    """
    embeddings = []
    for i, row in df.iterrows():
        embeddings.append({
            "id": i,
            "repo": row['repository_name'],
            "filepath": row['func_path_in_repository'],
            "url": row['func_code_url'],
            "func_name": row['func_name'],
            "embedding": code_embeddings[i].tolist()
        })
    
    os.makedirs('data/embeddings', exist_ok=True)
    with open("data/embeddings/python_embeddings.json", "w") as f:
        json.dump(embeddings, f)

    print("Saved embeddings to data/embeddings/python_embeddings.json")

if __name__ == "__main__":
    main()