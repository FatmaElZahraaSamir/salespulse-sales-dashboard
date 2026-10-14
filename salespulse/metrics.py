"""KPIs, breakdowns and filters used by the dashboard, the report and the tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import fiscal_year, last_complete_month


@dataclass
class Filters:
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    channels: list[str] | None = None
    groups: list[str] | None = None
    countries: list[str] | None = None
    categories: list[str] | None = None


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if f.start is not None:
        mask &= df["order_date"] >= pd.Timestamp(f.start)
    if f.end is not None:
        mask &= df["order_date"] <= pd.Timestamp(f.end)
    for column, values in [("channel", f.channels), ("territory_group", f.groups),
                           ("country", f.countries), ("category", f.categories)]:
        if values:
            mask &= df[column].isin(values)
    return df[mask]


def kpis(df: pd.DataFrame) -> dict[str, float]:
    revenue = float(df["sales"].sum())
    profit = float(df["profit"].sum())
    orders = int(df["order_id"].nunique())
    customers = int(df["customer_id"].nunique() + df["reseller"].nunique())
    return {
        "revenue": revenue,
        "profit": profit,
        "margin_pct": profit / revenue * 100 if revenue else 0.0,
        "orders": orders,
        "units": int(df["quantity"].sum()),
        "aov": revenue / orders if orders else 0.0,
        "customers": customers,
    }


def same_period_last_year(full: pd.DataFrame, current: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """The same filters shifted back 12 months (for YoY deltas)."""
    start = pd.Timestamp(f.start) if f.start is not None else current["order_date"].min()
    end = pd.Timestamp(f.end) if f.end is not None else current["order_date"].max()
    shifted = Filters(start - pd.DateOffset(years=1), end - pd.DateOffset(years=1),
                      f.channels, f.groups, f.countries, f.categories)
    return apply_filters(full, shifted)


def kpi_deltas(current: dict, previous: dict) -> dict[str, float | None]:
    out = {}
    for key, value in current.items():
        before = previous.get(key)
        if key == "margin_pct":
            out[key] = value - before if before is not None else None  # percentage points
        else:
            out[key] = (value - before) / before * 100 if before else None
    return out


def monthly(df: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    months = df["order_date"].dt.to_period("M")
    out = df.groupby(months).agg(revenue=("sales", "sum"), profit=("profit", "sum"),
                                 orders=("order_id", "nunique"))
    if complete_only and not df.empty:
        out = out[out.index <= last_complete_month(df)]
    out["margin_pct"] = np.where(out["revenue"] > 0, out["profit"] / out["revenue"] * 100, np.nan)
    out.index = out.index.to_timestamp()
    return out.rename_axis("month").reset_index()


def breakdown(df: pd.DataFrame, dim: str, top: int | None = None) -> pd.DataFrame:
    out = (
        df.groupby(dim, observed=True)
        .agg(revenue=("sales", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"),
             units=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
    )
    total = out["revenue"].sum()
    out["margin_pct"] = np.where(out["revenue"] > 0, out["profit"] / out["revenue"] * 100, np.nan)
    out["share_pct"] = out["revenue"] / total * 100 if total else 0.0
    out = out.reset_index()
    return out.head(top) if top else out


def pareto(df: pd.DataFrame, level: str = "product") -> tuple[pd.DataFrame, dict]:
    """Cumulative revenue share by item; how many items make 80 % of revenue."""
    items = df.groupby(level)["sales"].sum().sort_values(ascending=False)
    if items.empty:
        return pd.DataFrame(columns=[level, "revenue", "cum_share_pct", "rank_pct"]), {}
    cum = items.cumsum() / items.sum() * 100
    table = pd.DataFrame({level: items.index, "revenue": items.values, "cum_share_pct": cum.values})
    table["rank_pct"] = np.arange(1, len(table) + 1) / len(table) * 100
    n80 = int((cum < 80).sum() + 1)
    stats = {"items": len(table), "items_for_80": n80, "pct_items_for_80": n80 / len(table) * 100}
    return table, stats


def fiscal_channel_table(df: pd.DataFrame) -> pd.DataFrame:
    data = df.assign(fiscal_year=fiscal_year(df["order_date"]))
    return (
        data.groupby(["fiscal_year", "channel"], observed=True)["sales"].sum()
        .unstack("channel").fillna(0.0).reset_index()
    )
