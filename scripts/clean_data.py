# scripts/clean_data.py

from datasets import load_dataset
import pandas as pd

def main():
    # Load the Python subset of CodeSearchNet
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
    
    # Convert each split to a pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    validation_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    # Remove duplicates based on 'func_code_string'
    initial_count = len(train_df)
    train_df.drop_duplicates(subset=['func_code_string'], inplace=True)
    final_count = len(train_df)
    print(f"Removed {initial_count - final_count} duplicate code snippets.")

    # Remove rows with missing documentation
    initial_count = len(train_df)
    train_df.dropna(subset=['func_documentation_string'], inplace=True)
    final_count = len(train_df)
    print(f"Removed {initial_count - final_count} samples with missing documentation.")

    # Ensure Python only
    non_python = train_df[train_df['language'] != 'python']
    print(f"Number of non-Python samples: {len(non_python)}")
    train_df = train_df[train_df['language'] == 'python']

    # Sample a subset for development
    sample_size = 10000
    train_sample = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(train_sample)} examples for development.")

    # Save the cleaned training sample to a CSV file
    train_sample.to_csv('data/cleaned_train_sample.csv', index=False)
    print("Saved cleaned and sampled dataset to data/cleaned_train_sample.csv")

if __name__ == "__main__":
    main()
