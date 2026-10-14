"""Load the AdventureWorks star schema and flatten it into one clean sales table.

Source: Microsoft "AdventureWorks Sales" sample workbook (MIT licence),
https://github.com/microsoft/powerbi-desktop-samples

Cleaning steps (documented so a client can audit them):
1. Join the fact table (Sales_data) to Date, Product, Customer, Reseller,
   Sales Territory and Sales Order dimensions.
2. Replace the "[Not Applicable]" placeholder rows (key -1) with proper nulls:
   Internet orders have no reseller, Reseller orders have no consumer.
3. Parse integer date keys (20170701) into real dates; ship dates that are
   missing stay null instead of becoming 1970-01-01.
4. Compute profit = sales - total product cost, and keep money as float64.
5. Flag the last month if it is incomplete (the sample stops on 2020-06-15),
   so trends and forecasts are not distorted by a half month.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = ROOT / "data" / "sales.parquet"

COLUMNS = [
    "order_line", "order_id", "channel", "order_date", "ship_date", "due_date",
    "quantity", "unit_price", "discount_pct", "sales", "cost", "profit",
    "product", "sku", "model", "color", "subcategory", "category",
    "customer_id", "customer", "customer_city", "customer_country",
    "reseller", "business_type", "region", "country", "territory_group",
]


def _date_from_key(series: pd.Series) -> pd.Series:
    as_text = pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")
    return pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")


def build_dataset(xlsx_path: str | Path) -> pd.DataFrame:
    """Read the raw workbook and return the flat, typed sales table."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    fact = sheets["Sales_data"]
    orders = sheets["Sales Order_data"].rename(
        columns={"SalesOrderLineKey": "order_line", "Sales Order": "order_id", "Channel": "channel"}
    )[["order_line", "order_id", "channel"]]
    product = sheets["Product_data"].rename(
        columns={"Product": "product", "SKU": "sku", "Model": "model", "Color": "color",
                 "Subcategory": "subcategory", "Category": "category"}
    )[["ProductKey", "product", "sku", "model", "color", "subcategory", "category"]]
    customer = sheets["Customer_data"].rename(
        columns={"Customer ID": "customer_id", "Customer": "customer", "City": "customer_city",
                 "Country-Region": "customer_country"}
    )[["CustomerKey", "customer_id", "customer", "customer_city", "customer_country"]]
    reseller = sheets["Reseller_data"].rename(
        columns={"Reseller": "reseller", "Business Type": "business_type"}
    )[["ResellerKey", "reseller", "business_type"]]
    territory = sheets["Sales Territory_data"].rename(
        columns={"Region": "region", "Country": "country", "Group": "territory_group"}
    )[["SalesTerritoryKey", "region", "country", "territory_group"]]

    df = (
        fact.rename(columns={"SalesOrderLineKey": "order_line"})
        .merge(orders, on="order_line", how="left", validate="one_to_one")
        .merge(product, on="ProductKey", how="left", validate="many_to_one")
        .merge(customer, on="CustomerKey", how="left", validate="many_to_one")
        .merge(reseller, on="ResellerKey", how="left", validate="many_to_one")
        .merge(territory, on="SalesTerritoryKey", how="left", validate="many_to_one")
    )
    df = df.rename(columns={
        "Order Quantity": "quantity", "Unit Price": "unit_price", "Unit Price Discount Pct": "discount_pct",
        "Sales Amount": "sales", "Total Product Cost": "cost",
    })
    df["order_date"] = _date_from_key(df["OrderDateKey"])
    df["ship_date"] = _date_from_key(df["ShipDateKey"])
    df["due_date"] = _date_from_key(df["DueDateKey"])
    df["profit"] = df["sales"] - df["cost"]

    placeholder = "[Not Applicable]"
    for col in ["customer_id", "customer", "customer_city", "customer_country", "reseller", "business_type"]:
        df[col] = df[col].replace(placeholder, np.nan)
    df.loc[df["channel"] == "Reseller", ["customer_id", "customer", "customer_city", "customer_country"]] = np.nan
    df.loc[df["channel"] == "Internet", ["reseller", "business_type"]] = np.nan

    df = df[COLUMNS].sort_values(["order_date", "order_line"]).reset_index(drop=True)
    for col in ["channel", "category", "subcategory", "color", "region", "country", "territory_group", "business_type"]:
        df[col] = df[col].astype("category")
    for col in ["quantity"]:
        df[col] = df[col].astype("int32")
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the data breaks basic business rules."""
    problems = []
    if df["order_line"].duplicated().any():
        problems.append("duplicate order lines")
    if df["order_date"].isna().any():
        problems.append("missing order dates")
    if (df["sales"] < 0).any() or (df["quantity"] <= 0).any():
        problems.append("negative sales or non-positive quantities")
    if df["product"].isna().any() or df["country"].isna().any():
        problems.append("orphan product/territory keys")
    if not np.allclose(df["profit"], df["sales"] - df["cost"]):
        problems.append("profit does not equal sales - cost")
    if problems:
        raise ValueError("Data validation failed: " + ", ".join(problems))


def save_parquet(df: pd.DataFrame, path: str | Path = DEFAULT_PARQUET) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return path


def load_sales(path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate(df)
    return df


def last_complete_month(df: pd.DataFrame) -> pd.Period:
    """The data stops mid-month; analyses that bucket by month should end here."""
    last_day = df["order_date"].max()
    month_end = last_day + pd.offsets.MonthEnd(0)
    period = last_day.to_period("M")
    return period if last_day.normalize() == month_end.normalize() else period - 1


def fiscal_year(dates: pd.Series) -> pd.Series:
    """AdventureWorks fiscal years start in July: 2017-07-01 belongs to FY2018."""
    return ("FY" + (dates.dt.year + (dates.dt.month >= 7).astype(int)).astype(str)).astype("category")
