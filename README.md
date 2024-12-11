# CodeSense

"CodeSense" is a semantic search engine designed to enhance the discoverability of code documentation by leveraging natural language processing (NLP) and advanced search techniques. Unlike traditional keyword-based search systems, CodeSense utilizes word embeddings and semantic search algorithms to understand the intent behind a developer’s query, providing more relevant and contextually appropriate documentation and code snippets. This tool aims to streamline the coding process by making it easier for developers to find the right information quickly, even when they don’t know the exact terms to search for.

## Steps to Run

**1. Clean the Dataset**

Clean the raw dataset to remove duplicates, retain Python files, and preprocess for semantic search.

python data_cleaning.py

Input: Raw dataset file path (e.g., data/raw_train_sample.csv).

Output: Cleaned dataset saved at data/cleaned_train_sample.csv.

**2. Generate Embeddings**

Create embeddings for the cleaned dataset using the pre-trained SentenceTransformer model.

python embed.py

Input: data/cleaned_train_sample.csv.

Output: Embeddings saved at data/embeddings/python_embeddings.json.

**3. Build FAISS Index**

Build a FAISS index from the generated embeddings for fast retrieval during search.

python build_faiss_index.py

Input: data/embeddings/python_embeddings.json.

Output:

FAISS index at data/indexes/faiss_index.bin.

Metadata at data/metadata/faiss_metadata.json.

**4. Run the Query Interface**

Run the user interface to perform semantic code search.

python ui.py

Input: User-provided query via terminal or UI.

Output: Top-k similar code snippets with metadata (e.g., file path, repo URL).

