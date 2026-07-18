"""Business insights generator for the Data Cleaning & Visualization project."""

from __future__ import annotations

import pandas as pd


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Generate business insights from the cleaned dataset."""
    insights: list[str] = []

    if "sales" in df.columns:
        sales_by_category = df.groupby("category")["sales"].sum().sort_values(ascending=False)
        insights.append(f"Highest sales category: {sales_by_category.index[0]} with ${sales_by_category.iloc[0]:,.2f} in sales.")
        insights.append(f"Lowest sales category: {sales_by_category.index[-1]} with ${sales_by_category.iloc[-1]:,.2f} in sales.")

    if "region" in df.columns and "sales" in df.columns:
        sales_by_region = df.groupby("region")["sales"].sum().sort_values(ascending=False)
        insights.append(f"Best-performing region: {sales_by_region.index[0]} with ${sales_by_region.iloc[0]:,.2f} in total sales.")

    if "order_date" in df.columns and "sales" in df.columns:
        monthly_sales = df.set_index("order_date").resample("ME")["sales"].sum()
        if not monthly_sales.empty:
            insights.append(f"Sales trend shows a monthly peak of ${monthly_sales.max():,.2f} in {monthly_sales.idxmax().strftime('%B %Y')}.")
            insights.append(f"Sales grew by ${monthly_sales.diff().mean():,.2f} on average per month during the observed period.")

    if "profit" in df.columns and "sales" in df.columns:
        profit_rate = (df["profit"].sum() / df["sales"].sum()) if df["sales"].sum() != 0 else 0
        insights.append(f"Overall profit margin is {profit_rate:.1%}, which indicates how much profit is earned for every dollar of sales.")

    if "product_name" in df.columns:
        top_products = df.groupby("product_name")["sales"].sum().sort_values(ascending=False).head(3)
        for product, amount in top_products.items():
            insights.append(f"Top product: {product} generated ${amount:,.2f} in sales.")

    if "discount" in df.columns and "profit" in df.columns:
        discount_effect = df.groupby("discount")["profit"].mean().sort_index()
        insights.append("Discounts affect profitability: average profit by discount level shows how margins shift with promotions.")
        insights.append(f"Average profit at the most common discount level ({discount_effect.index[0]:.2%}) is ${discount_effect.iloc[0]:.2f}.")

    if "category" in df.columns and "profit" in df.columns:
        profit_by_category = df.groupby("category")["profit"].sum().sort_values(ascending=False)
        insights.append(f"Most profitable category: {profit_by_category.index[0]} with ${profit_by_category.iloc[0]:,.2f} in profit.")

    if "quantity" in df.columns:
        highest_quantity = df.loc[df["quantity"].idxmax()]
        insights.append(f"Single largest order by quantity: {int(highest_quantity['quantity'])} units of {highest_quantity['product_name']}.")

    if "order_date" in df.columns:
        anomalies = df[df["order_date"].dt.month.isin([1, 12])]
        insights.append(f"There are {len(anomalies)} orders in peak season months (January and December).")

    insights.append("The cleaned dataset is ready for further modeling and deeper customer segmentation analysis.")
    return insights


def save_insights(insights: list[str], output_file: str) -> None:
    """Save generated insights to a text file."""
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Business Insights\n")
        file.write("================\n\n")
        for insight in insights:
            file.write(f"- {insight}\n")
