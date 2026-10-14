import io

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from salespulse import insights
from salespulse.forecast import backtest, forecast, mape, monthly_series, seasonal_naive
from salespulse.metrics import (
    Filters, apply_filters, breakdown, kpi_deltas, kpis, monthly, pareto, same_period_last_year,
)
from salespulse.report import build_report
from salespulse.segmentation import (
    cohort_retention, customer_table, kmeans_clusters, rfm_segments, segment_summary,
)


# ------------------------------------------------------------------- metrics
def test_kpis_on_hand_made_orders(lines):
    df = lines([
        {"order_id": "SO1", "sales": 100.0, "cost": 60.0, "quantity": 1},
        {"order_id": "SO1", "sales": 50.0, "cost": 20.0, "quantity": 2},
        {"order_id": "SO2", "sales": 250.0, "cost": 200.0, "quantity": 1, "customer_id": "C2"},
    ])
    k = kpis(df)
    assert k["revenue"] == 400 and k["profit"] == 120
    assert k["margin_pct"] == pytest.approx(30.0)
    assert k["orders"] == 2 and k["units"] == 4
    assert k["aov"] == pytest.approx(200.0)
    assert k["customers"] == 2


def test_filters_and_year_over_year(sales):
    f = Filters(pd.Timestamp("2019-07-01"), pd.Timestamp("2020-06-30"), ["Internet"], None, None, ["Bikes"])
    current = apply_filters(sales, f)
    assert set(current["channel"].astype(str)) == {"Internet"}
    assert set(current["category"].astype(str)) == {"Bikes"}
    previous = same_period_last_year(sales, current, f)
    assert previous["order_date"].max() < pd.Timestamp("2019-07-01")
    deltas = kpi_deltas(kpis(current), kpis(previous))
    assert deltas["revenue"] > 0  # online bike sales grew in FY2020


def test_monthly_excludes_partial_last_month(sales):
    m = monthly(sales)
    assert m["month"].max() == pd.Timestamp("2020-05-01")
    assert m["revenue"].sum() < sales["sales"].sum()


def test_breakdown_shares_sum_to_100(sales):
    b = breakdown(sales, "country")
    assert b["share_pct"].sum() == pytest.approx(100.0)
    assert b.iloc[0]["country"] == "United States"


def test_pareto_80_percent(sales):
    table, stats = pareto(sales, "product")
    assert table["cum_share_pct"].iloc[-1] == pytest.approx(100.0)
    assert 0 < stats["items_for_80"] < stats["items"]
    assert table.loc[stats["items_for_80"] - 1, "cum_share_pct"] >= 80


# --------------------------------------------------------------- segmentation
def test_rfm_puts_recent_frequent_big_spenders_on_top(lines):
    rows = []
    # 25 customers: customer i orders i times, spends more, most recently
    for i in range(1, 26):
        for j in range(i % 5 + 1):
            rows.append({"customer_id": f"C{i}", "order_id": f"SO{i}-{j}", "sales": 100.0 * i, "cost": 10.0,
                         "order_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 10 + j)})
    rfm = rfm_segments(customer_table(lines(rows)))
    assert set(rfm[["R", "F", "M"]].stack().unique()) <= {1, 2, 3, 4, 5}
    best = rfm.loc[rfm["monetary"].idxmax()]
    assert best["R"] == 5 and best["M"] == 5
    summary = segment_summary(rfm)
    assert summary["customers"].sum() == 25
    assert summary["revenue_share_%"].sum() == pytest.approx(100, abs=0.5)
    assert summary["recommended_action"].notna().all()


def test_kmeans_on_real_customers(sales):
    rfm = rfm_segments(customer_table(sales))
    labelled, profile, k = kmeans_clusters(rfm, k_range=range(3, 5))
    assert k in (3, 4)
    assert profile["customers"].sum() == len(rfm) == labelled["cluster"].notna().sum()
    assert 0 < profile.attrs["silhouette"] <= 1
    assert profile["cluster"].is_unique


def test_cohort_retention_shape_and_unknown_future(sales):
    ret = cohort_retention(sales, max_months=6)
    assert (ret[0] == 100).all()                      # everyone is active in their first month
    assert np.isnan(ret.loc["2020-05", 3])            # Aug 2020 has not happened yet
    assert ret.loc["2019-07", 1] > 0                  # repeat purchases exist after mid-2019


# ------------------------------------------------------------------ forecast
def synthetic_series(n=36, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-07-01", periods=n, freq="MS")
    trend = np.linspace(100, 200, n)
    season = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 12)
    return pd.Series(trend * season * rng.normal(1, 0.03, n), index=idx)


def test_models_beat_naive_on_trending_seasonal_data():
    series = synthetic_series()
    bt = backtest(series, horizon=6)
    errors = bt.set_index("model")["MAPE %"]
    assert errors.min() < errors["Seasonal naive (baseline)"]


def test_forecast_output_shape_and_interval():
    fc = forecast(synthetic_series(), horizon=6)
    t = fc.table
    assert len(t) == 6 and t["month"].iloc[0] == pd.Timestamp("2020-07-01")
    assert (t["lower"] <= t["forecast"]).all() and (t["forecast"] <= t["upper"]).all()
    assert fc.model in fc.backtest["model"].tolist()


def test_seasonal_naive_repeats_last_year():
    s = synthetic_series()
    assert np.allclose(seasonal_naive(s, 3), s.iloc[-12:-9].to_numpy())


def test_mape():
    assert mape(np.array([100, 200]), np.array([110, 180])) == pytest.approx(10.0)
    assert mape(np.array([0, 100]), np.array([5, 100])) == pytest.approx(0.0)  # zero actuals ignored


def test_real_forecast_beats_baseline(sales):
    fc = forecast(monthly_series(sales), horizon=6)
    assert fc.improvement_vs_baseline is not None and fc.improvement_vs_baseline > 0


# ------------------------------------------------------------- report/insights
def test_insights_quote_real_numbers(sales):
    text = " ".join(insights.generate(sales))
    assert "$109.8M" in text and "Reseller" in text and "United States" in text


def test_excel_report_has_all_sheets_and_charts(sales):
    data = build_report(sales[sales["order_date"] >= "2018-07-01"], "test")
    wb = load_workbook(io.BytesIO(data))
    assert wb.sheetnames == ["Summary", "Monthly", "Categories", "Countries", "Top products",
                             "Customer segments", "Forecast"]
    assert wb["Summary"]["A1"].value.startswith("SalesPulse")
    assert len(wb["Monthly"]._charts) == 1
