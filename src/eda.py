"""Exploratory Data Analysis utilities for the Data Cleaning & Visualization project."""

from __future__ import annotations

import pandas as pd


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for numerical columns."""
    return df.describe(include="all").transpose()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    return numeric_df.corr()


def unique_values_report(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, pd.Series]:
    """Return unique value counts for selected columns."""
    columns = columns or [col for col in df.columns if df[col].dtype == object]
    return {column: df[column].value_counts() for column in columns}


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and ratios."""
    return df.isna().sum().to_frame(name="count").assign(ratio=lambda x: x["count"] / len(df))


def distribution_analysis(df: pd.DataFrame, numeric_columns: list[str] | None = None) -> pd.DataFrame:
    """Return distribution statistics for numeric columns."""
    numeric_columns = numeric_columns or df.select_dtypes(include=["number"]).columns.tolist()
    distribution = df[numeric_columns].agg(["mean", "median", "std", "min", "max"]).transpose()
    return distribution
