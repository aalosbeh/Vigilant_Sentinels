import os
import pandas as pd

def save_dataframe(df, path):
    """Save a DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_dataframe(path):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)