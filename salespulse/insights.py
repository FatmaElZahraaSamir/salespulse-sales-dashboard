"""Plain-English executive summary computed from the data (no hard-coded numbers)."""

from __future__ import annotations

import pandas as pd

from .data import fiscal_year
from .metrics import breakdown, kpis, monthly, pareto


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def generate(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No sales match the current filters."]
    out: list[str] = []
    k = kpis(df)
    out.append(f"**{money(k['revenue'])}** revenue from **{k['orders']:,}** orders "
               f"at a **{k['margin_pct']:.1f}%** gross margin (average order {money(k['aov'])}).")

    fy = df.assign(fy=fiscal_year(df["order_date"])).groupby("fy", observed=True)["sales"].sum()
    if len(fy) >= 2:
        # compare the last two *full* fiscal years when the latest one is partial
        last, prev = fy.index[-1], fy.index[-2]
        growth = (fy[last] - fy[prev]) / fy[prev] * 100
        fy_end = pd.Timestamp(year=int(str(last)[2:]), month=6, day=30)
        partial = " (latest year not complete in the data)" if df["order_date"].max() < fy_end else ""
        out.append(f"Revenue went from {money(fy[prev])} in {prev} to {money(fy[last])} in {last} "
                   f"({growth:+.0f}%){partial}.")

    by_channel = breakdown(df, "channel")
    if len(by_channel) > 1:
        rows = by_channel.set_index("channel")
        hi = rows["margin_pct"].idxmax()
        lo = rows["margin_pct"].idxmin()
        low_margin = rows.loc[lo, "margin_pct"]
        how = (f"is sold at a loss ({low_margin:.1f}% margin)" if low_margin < 0
               else f"earns only a {low_margin:.1f}% margin")
        out.append(f"**{lo}** brings {rows.loc[lo, 'share_pct']:.0f}% of revenue but {how}, "
                   f"versus {rows.loc[hi, 'margin_pct']:.1f}% for {hi} — "
                   f"pricing and discounts in {lo} are the biggest profit lever.")

    cat = breakdown(df, "category")
    top_cat = cat.iloc[0]
    out.append(f"**{top_cat['category']}** is {top_cat['share_pct']:.0f}% of revenue; "
               f"the best margin is in **{cat.loc[cat['margin_pct'].idxmax(), 'category']}** "
               f"({cat['margin_pct'].max():.0f}%).")

    country = breakdown(df, "country")
    out.append(f"Top market: **{country.iloc[0]['country']}** ({country.iloc[0]['share_pct']:.0f}% of revenue), "
               f"then {country.iloc[1]['country']} ({country.iloc[1]['share_pct']:.0f}%)." if len(country) > 1
               else f"All revenue comes from {country.iloc[0]['country']}.")

    _, stats = pareto(df, "product")
    if stats:
        out.append(f"**{stats['items_for_80']} of {stats['items']} products** "
                   f"({stats['pct_items_for_80']:.0f}%) generate 80% of revenue — protect their stock levels.")

    m = monthly(df)
    if len(m) >= 3:
        best = m.loc[m["revenue"].idxmax()]
        out.append(f"Best month: **{best['month']:%B %Y}** with {money(best['revenue'])}.")
    return out
