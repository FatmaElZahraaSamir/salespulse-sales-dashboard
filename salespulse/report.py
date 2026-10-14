"""Client-ready Excel report with native (editable) Excel charts."""

from __future__ import annotations

import io
import re
from datetime import datetime

import pandas as pd

from . import insights
from .forecast import forecast, monthly_series
from .metrics import breakdown, kpis, monthly
from .segmentation import customer_table, rfm_segments, segment_summary

BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#0b0b0b"


def build_report(df: pd.DataFrame, filters_text: str = "All data") -> bytes:
    """Return the .xlsx file as bytes (ready for a download button or to save to disk)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        wb = writer.book
        title = wb.add_format({"bold": True, "font_size": 18, "font_color": INK})
        sub = wb.add_format({"font_color": "#52514e", "italic": True})
        head = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#0F4C5C", "border": 0})
        money = wb.add_format({"num_format": "$#,##0"})
        pct = wb.add_format({"num_format": "0.0\"%\""})
        count = wb.add_format({"num_format": "#,##0"})
        text = wb.add_format({"text_wrap": True, "valign": "top"})

        # ------------------------------------------------------------ Summary
        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws
        ws.write("A1", "SalesPulse — Sales Performance Report", title)
        ws.write("A2", f"Filters: {filters_text}   ·   Generated {datetime.now():%Y-%m-%d %H:%M}", sub)
        k = kpis(df)
        rows = [("Revenue", k["revenue"], money), ("Gross profit", k["profit"], money),
                ("Gross margin", k["margin_pct"], pct), ("Orders", k["orders"], count),
                ("Units sold", k["units"], count), ("Average order value", k["aov"], money),
                ("Customers & resellers", k["customers"], count)]
        ws.write_row("A4", ["KPI", "Value"], head)
        for i, (name, value, fmt) in enumerate(rows, start=4):
            ws.write(i, 0, name)
            ws.write(i, 1, value, fmt)
        ws.write(18, 0, "Key insights", wb.add_format({"bold": True, "font_size": 13}))
        for i, line in enumerate(insights.generate(df), start=19):
            ws.merge_range(i, 0, i, 9, re.sub(r"\*\*(.*?)\*\*", r"\1", line), text)
            ws.set_row(i, 32)
        ws.set_column("A:A", 26)
        ws.set_column("B:B", 18)
        ws.set_landscape()
        ws.fit_to_pages(1, 1)

        # ------------------------------------------------------------ Monthly
        m = monthly(df)
        m_out = m.assign(month=m["month"].dt.strftime("%Y-%m"))[["month", "revenue", "profit", "margin_pct", "orders"]]
        _table(writer, m_out, "Monthly", head, {"revenue": money, "profit": money, "margin_pct": pct})
        chart = wb.add_chart({"type": "line"})
        n = len(m_out)
        for col, color, name in [(1, BLUE, "Revenue"), (2, ORANGE, "Profit")]:
            chart.add_series({"name": name, "categories": ["Monthly", 1, 0, n, 0],
                              "values": ["Monthly", 1, col, n, col], "line": {"color": color, "width": 2}})
        chart.set_title({"name": "Monthly revenue and profit"})
        chart.set_y_axis({"num_format": "$#,##0,,\"M\"", "major_gridlines": {"visible": True, "line": {"color": "#e1e0d9"}}})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 860, "height": 360})
        writer.sheets["Monthly"].insert_chart("G2", chart)

        # same trend on the one-page summary, next to the KPIs
        summary_chart = wb.add_chart({"type": "line"})
        for col, color, name in [(1, BLUE, "Revenue"), (2, ORANGE, "Profit")]:
            summary_chart.add_series({"name": name, "categories": ["Monthly", 1, 0, n, 0],
                                      "values": ["Monthly", 1, col, n, col], "line": {"color": color, "width": 2}})
        summary_chart.set_title({"name": "Monthly revenue and profit", "name_font": {"size": 11}})
        summary_chart.set_y_axis({"num_format": "$#,##0,,\"M\"", "major_gridlines": {"visible": True, "line": {"color": "#e1e0d9"}}})
        summary_chart.set_legend({"position": "bottom"})
        summary_chart.set_size({"width": 560, "height": 250})
        writer.sheets["Summary"].insert_chart("D4", summary_chart)

        # ------------------------------------------- Category / country breakdowns
        for sheet, dim in [("Categories", "category"), ("Countries", "country")]:
            b = breakdown(df, dim)[[dim, "revenue", "profit", "margin_pct", "share_pct", "orders", "units"]]
            _table(writer, b, sheet, head, {"revenue": money, "profit": money, "margin_pct": pct, "share_pct": pct})
            bar = wb.add_chart({"type": "bar"})
            bar.add_series({"name": "Revenue", "categories": [sheet, 1, 0, len(b), 0],
                            "values": [sheet, 1, 1, len(b), 1], "fill": {"color": BLUE}, "gap": 60})
            bar.set_title({"name": f"Revenue by {dim}"})
            bar.set_y_axis({"reverse": True})
            bar.set_x_axis({"num_format": "$#,##0,,\"M\""})
            bar.set_legend({"none": True})
            bar.set_size({"width": 620, "height": 300 + 18 * len(b)})
            writer.sheets[sheet].insert_chart("I2", bar)

        top = breakdown(df, "product", top=25)[["product", "revenue", "profit", "margin_pct", "units"]]
        _table(writer, top, "Top products", head, {"revenue": money, "profit": money, "margin_pct": pct})

        # ------------------------------------------------------ Customer segments
        customers = customer_table(df)
        if not customers.empty and len(customers) >= 5:
            seg = segment_summary(rfm_segments(customers))
            _table(writer, seg, "Customer segments", head,
                   {"revenue": money, "avg_spend": money, "revenue_share_%": pct, "customers_%": pct},
                   widths={"recommended_action": 60})

        # ---------------------------------------------------------------- Forecast
        series = monthly_series(df)
        if len(series) >= 18:
            fc = forecast(series, horizon=6)
            out = fc.table.assign(month=fc.table["month"].dt.strftime("%Y-%m"))
            _table(writer, out, "Forecast", head, {"forecast": money, "lower": money, "upper": money})
            ws = writer.sheets["Forecast"]
            ws.write(len(out) + 2, 0, f"Model: {fc.model} (lowest backtest error)")
            fc.backtest.to_excel(writer, sheet_name="Forecast", startrow=len(out) + 4, index=False)
    return buffer.getvalue()


def _table(writer, frame: pd.DataFrame, sheet: str, head, formats: dict, widths: dict | None = None) -> None:
    frame.to_excel(writer, sheet_name=sheet, index=False)
    ws = writer.sheets[sheet]
    widths = widths or {}
    for i, col in enumerate(frame.columns):
        ws.write(0, i, col, head)
        width = widths.get(col, min(45, max(12, int(frame[col].astype("string").str.len().max() or 10) + 2)))
        ws.set_column(i, i, width, formats.get(col))
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(frame), len(frame.columns) - 1)
    ws.set_landscape()
    ws.fit_to_pages(1, 0)  # all columns on one page width when printed
