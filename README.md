# CodeSense

"CodeSense" is a semantic search engine designed to enhance the discoverability of code documentation by leveraging natural language processing (NLP) and advanced search techniques. Unlike traditional keyword-based search systems, CodeSense utilizes word embeddings and semantic search algorithms to understand the intent behind a developer’s query, providing more relevant and contextually appropriate documentation and code snippets. This tool aims to streamline the coding process by making it easier for developers to find the right information quickly, even when they don’t know the exact terms to search for.

## Requirements
1. Python 3.7+
2. Install dependencies:
   ```pip install -r requirements.txt```

## Steps to Run

**1. Clean the Dataset**

Clean the raw dataset to remove duplicates, retain Python files, and preprocess for semantic search.

```python data_cleaning.py```

Input: Raw dataset

Output: Cleaned dataset

**2. Generate Embeddings**

Create embeddings for the cleaned dataset using the pre-trained SentenceTransformer model.

```python embed.py```

Input: Cleaned dataset

Output: Embeddings 

**3. Build FAISS Index**

Build a FAISS index from the generated embeddings for fast retrieval during search.

```python build_faiss_index.py```

Input: Embeddings

Output:

1. FAISS index
2. Metadata

**4. Run the Query Interface**

Run the user interface to perform semantic code search.

```python ui.py```

Input: User-provided query via terminal or UI.

Output: Top-k similar code snippets with metadata

