"""Data loader module for the Data Cleaning & Visualization project."""

from __future__ import annotations

import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError as error:
        raise FileNotFoundError(f"CSV file not found: {file_path}") from error
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"CSV file is empty: {file_path}") from error
    except Exception as error:
        raise RuntimeError(f"Unable to load CSV data: {error}") from error


def display_dataset_info(df: pd.DataFrame) -> None:
    """Print basic dataset information for a DataFrame."""
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
