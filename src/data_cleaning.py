"""Data cleaning utilities for the Data Cleaning & Visualization project."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase, strip whitespace, and replace spaces with underscores."""
    df_copy = df.copy()
    df_copy.columns = [str(col).strip().lower().replace(" ", "_") for col in df_copy.columns]
    return df_copy


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a report of missing values by column."""
    missing_report = df.isna().sum().to_frame(name="missing_count")
    missing_report["missing_ratio"] = missing_report["missing_count"] / len(df)
    return missing_report.sort_values(by="missing_count", ascending=False)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values based on column type."""
    df_copy = df.copy()
    for column in df_copy.columns:
        if df_copy[column].dtype == "object":
            df_copy[column] = df_copy[column].fillna("Unknown")
        elif pd.api.types.is_numeric_dtype(df_copy[column]):
            median_value = df_copy[column].median()
            df_copy[column] = df_copy[column].fillna(median_value)
        elif pd.api.types.is_datetime64_any_dtype(df_copy[column]):
            df_copy[column] = df_copy[column].fillna(df_copy[column].mode().iloc[0] if not df_copy[column].mode().empty else pd.Timestamp("today"))
        else:
            df_copy[column] = df_copy[column].fillna("Unknown")
    return df_copy


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from a DataFrame."""
    return df.drop_duplicates().reset_index(drop=True)


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Detect outliers using the IQR method and return a mask of outlier rows."""
    outlier_mask = pd.DataFrame(index=df.index)
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask[column] = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outlier_mask.any(axis=1)


def cap_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cap numeric outliers at the IQR bounds for selected columns."""
    df_copy = df.copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(df_copy[column]):
            q1 = df_copy[column].quantile(0.25)
            q3 = df_copy[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_copy[column] = np.where(df_copy[column] < lower_bound, lower_bound, df_copy[column])
            df_copy[column] = np.where(df_copy[column] > upper_bound, upper_bound, df_copy[column])
    return df_copy


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types for analysis."""
    df_copy = df.copy()
    if "order_date" in df_copy.columns:
        df_copy["order_date"] = pd.to_datetime(df_copy["order_date"], errors="coerce")
    numeric_columns = [col for col in df_copy.columns if col not in ["order_date", "region", "category", "sub-category", "product_name", "order_id"]]
    for column in numeric_columns:
        df_copy[column] = pd.to_numeric(df_copy[column], errors="coerce")
    return df_copy


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Execute a series of cleaning steps and return the cleaned DataFrame."""
    df_clean = standardize_columns(df)
    df_clean = convert_data_types(df_clean)
    df_clean = handle_missing_values(df_clean)
    df_clean = remove_duplicates(df_clean)
    numeric_columns = [col for col in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[col]) and col not in ["quantity"]]
    df_clean = cap_outliers(df_clean, numeric_columns)
    return df_clean
