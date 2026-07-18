"""Main pipeline entrypoint for Data Cleaning & Visualization."""

from __future__ import annotations

import os

from data_loader import display_dataset_info, load_csv
from data_cleaning import clean_data, detect_missing_values
from eda import correlation_matrix, distribution_analysis, missing_value_report, summary_statistics, unique_values_report
from insights import generate_insights, save_insights
from visualization import (
    plot_bar_chart,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_histogram,
    plot_line_chart,
    plot_pie_chart,
    plot_scatter,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_dataset.csv")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
SUMMARY_REPORT_PATH = os.path.join(OUTPUT_DIR, "summary_report.txt")


def main() -> None:
    """Load raw data, clean it, analyze it, visualize it, and save outputs."""
    try:
        df_raw = load_csv(RAW_DATA_PATH)
        display_dataset_info(df_raw)

        df_cleaned = clean_data(df_raw)
        df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

        print("\n=== Cleaning Report ===")
        print(detect_missing_values(df_cleaned).to_string())

        print("\n=== Exploratory Data Analysis ===")
        print(summary_statistics(df_cleaned).to_string())
        print("\nMissing values report:")
        print(missing_value_report(df_cleaned).to_string())

        corr_matrix = correlation_matrix(df_cleaned)
        dist_report = distribution_analysis(df_cleaned)
        unique_report = unique_values_report(df_cleaned)

        print("\nCorrelation matrix:")
        print(corr_matrix.to_string())

        insights = generate_insights(df_cleaned)
        save_insights(insights, SUMMARY_REPORT_PATH)

        df_category = df_cleaned.groupby("category")["sales"].sum().reset_index()
        plot_bar_chart(df_category, "category", "sales", os.path.join(VISUALIZATIONS_DIR, "sales_by_category.png"))

        if "order_date" in df_cleaned.columns:
            df_time = df_cleaned.set_index("order_date").resample("ME")["sales"].sum().reset_index()
            plot_line_chart(df_time, "order_date", "sales", os.path.join(VISUALIZATIONS_DIR, "monthly_sales_trend.png"))

        plot_histogram(df_cleaned, "profit", os.path.join(VISUALIZATIONS_DIR, "profit_distribution.png"))
        plot_boxplot(df_cleaned, "sales", os.path.join(VISUALIZATIONS_DIR, "sales_boxplot.png"))
        plot_correlation_heatmap(corr_matrix, os.path.join(VISUALIZATIONS_DIR, "correlation_heatmap.png"))
        plot_pie_chart(df_cleaned, "region", os.path.join(VISUALIZATIONS_DIR, "region_share.png"))
        plot_scatter(df_cleaned, "sales", "profit", os.path.join(VISUALIZATIONS_DIR, "sales_vs_profit.png"))

        print("\n=== Outputs Saved ===")
        print(f"Cleaned dataset: {CLEANED_DATA_PATH}")
        print(f"Summary report: {SUMMARY_REPORT_PATH}")
        print(f"Visualizations folder: {VISUALIZATIONS_DIR}")
        print("\n=== Key Insights ===")
        for insight in insights:
            print(f"- {insight}")

    except Exception as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
