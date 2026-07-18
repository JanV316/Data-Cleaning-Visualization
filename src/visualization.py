"""Visualization utilities for the Data Cleaning & Visualization project."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_directory(path: str) -> None:
    """Ensure the destination directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_bar_chart(df, x_column: str, y_column: str, output_path: str) -> None:
    """Generate and save a bar chart."""
    _ensure_directory(os.path.dirname(output_path))
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x_column, y=y_column, ci=None)
    plt.title(f"{y_column.title()} by {x_column.title()}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_line_chart(df, x_column: str, y_column: str, output_path: str) -> None:
    """Generate and save a line chart."""
    _ensure_directory(os.path.dirname(output_path))
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_column, y=y_column, marker="o")
    plt.title(f"{y_column.title()} Trend by {x_column.title()}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_histogram(df, column: str, output_path: str) -> None:
    """Generate and save a histogram."""
    _ensure_directory(os.path.dirname(output_path))
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=True, bins=15)
    plt.title(f"Distribution of {column.title()}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_boxplot(df, column: str, output_path: str) -> None:
    """Generate and save a box plot."""
    _ensure_directory(os.path.dirname(output_path))
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column].dropna())
    plt.title(f"Boxplot of {column.title()}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlation_heatmap(corr_matrix, output_path: str) -> None:
    """Generate and save a correlation heatmap."""
    _ensure_directory(os.path.dirname(output_path))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pie_chart(df, column: str, output_path: str) -> None:
    """Generate and save a pie chart for value counts."""
    _ensure_directory(os.path.dirname(output_path))
    counts = df[column].value_counts()
    plt.figure(figsize=(8, 8))
    counts.plot.pie(autopct="%.1f%%", startangle=140)
    plt.ylabel("")
    plt.title(f"Share of {column.title()}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scatter(df, x_column: str, y_column: str, output_path: str) -> None:
    """Generate and save a scatter plot."""
    _ensure_directory(os.path.dirname(output_path))
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.title(f"{y_column.title()} vs {x_column.title()}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
