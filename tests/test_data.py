import numpy as np
import pandas as pd
import pytest

from salespulse.data import build_dataset, fiscal_year, last_complete_month, validate


def test_build_dataset_joins_and_cleans(tiny_workbook):
    df = build_dataset(tiny_workbook)
    assert len(df) == 4
    internet = df[df["channel"] == "Internet"]
    reseller = df[df["channel"] == "Reseller"]
    # placeholders become real nulls, per channel
    assert internet["reseller"].isna().all() and internet["customer"].eq("Ann Lee").all()
    assert reseller["customer_id"].isna().all() and reseller["reseller"].eq("Bike World").all()
    # dates parsed; a missing ship date stays missing (not 1970-01-01)
    assert df["order_date"].min() == pd.Timestamp("2019-07-02")
    assert df["ship_date"].isna().sum() == 1
    # profit = sales - cost
    assert np.allclose(df["profit"], df["sales"] - df["cost"])
    assert set(df["category"].astype(str)) == {"Bikes", "Accessories"}


def test_validate_rejects_bad_data(tiny_workbook):
    df = build_dataset(tiny_workbook)
    broken = df.copy()
    broken.loc[0, "sales"] = -5
    with pytest.raises(ValueError, match="negative sales"):
        validate(broken)
    dupes = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate order lines"):
        validate(dupes)


def test_shipped_dataset_matches_source(sales):
    """The parquet in the repo is the full AdventureWorks sample, not a subset."""
    assert len(sales) == 121_253
    assert sales["order_date"].min() == pd.Timestamp("2017-07-01")
    assert sales["order_date"].max() == pd.Timestamp("2020-06-15")
    assert sales["sales"].sum() == pytest.approx(109_809_274, rel=1e-4)
    assert set(sales["channel"].astype(str)) == {"Internet", "Reseller"}
    assert sales["order_line"].is_unique


def test_last_complete_month_skips_partial_month(sales):
    assert str(last_complete_month(sales)) == "2020-05"  # data stops on 15 June


def test_fiscal_year_starts_in_july():
    dates = pd.Series(pd.to_datetime(["2019-06-30", "2019-07-01", "2020-06-15"]))
    assert list(fiscal_year(dates).astype(str)) == ["FY2019", "FY2020", "FY2020"]
