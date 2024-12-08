# scripts/generate_embeddings.py

import json
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

MODEL_NAME = "microsoft/codebert-base"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()
    return emb

def main():
    # Load the cleaned dataset
    df = pd.read_csv('data/cleaned_train_sample.csv')

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    embeddings = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Combine documentation and code
        text = (row['func_documentation_string'] or "") + " " + (row['func_code_string'] or "")
        emb = get_embedding(model, tokenizer, text)
        embeddings.append({
    "id": i,
    "repo": row['repository_name'],                  # replaced 'repo' with 'repository_name'
    "filepath": row['func_path_in_repository'],       # replaced 'path' with 'func_path_in_repository'
    "url": row['func_code_url'],                      # replaced 'url' with 'func_code_url'
    "func_name": row['func_name'],                    # this column matches directly
    "embedding": emb.tolist()
})


    os.makedirs('data/embeddings', exist_ok=True)
    with open("data/embeddings/python_embeddings.json", "w") as f:
        json.dump(embeddings, f)

    print("Saved embeddings to data/embeddings/python_embeddings.json")

if __name__ == "__main__":
    main()
