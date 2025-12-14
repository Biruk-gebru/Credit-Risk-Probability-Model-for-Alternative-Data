import pandas as pd
import os

def load_data(filepath):
    """
    Loads data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def check_missing_values(df):
    """
    Checks for missing values in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to check.

    Returns:
        pd.Series: Count of missing values per column.
    """
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print("Missing values found:")
        print(missing[missing > 0])
    return missing
