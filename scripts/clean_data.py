# scripts/clean_data.py
from datasets import load_dataset
import pandas as pd
import os

def main():
    os.makedirs("data", exist_ok=True)

    # Load the Python subset of CodeSearchNet
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True) 
    # trust_remote_code=True might fetch an error, if it does, pass only the first two params
    
    # Convert each split to a pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    validation_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    # Remove duplicates based on 'func_code_string'
    train_df.drop_duplicates(subset=['func_code_string'], inplace=True)

    # Remove rows with missing documentation
    train_df.dropna(subset=['func_documentation_string'], inplace=True)
    
    # Ensure Python only
    non_python = train_df[train_df['language'] != 'python']
    train_df = train_df[train_df['language'] == 'python']

    # Sample a subset for development
    sample_size = 50000
    train_sample = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(train_sample)} examples for development.")

    # Save the cleaned training sample to a CSV file
    train_sample.to_csv('data/cleaned_train_sample.csv', index=False) # change directory path here
    print("Saved cleaned and sampled dataset to data/cleaned_train_sample.csv")

if __name__ == "__main__":
    main()