"""Customer analytics for the Internet channel (individual consumers).

* RFM scoring with business-friendly segments and a recommended action each
* K-Means clustering on log-scaled RFM (k picked by silhouette score)
* Monthly cohort retention matrix
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_MAP = {  # (R score, F score) patterns -> segment, the widely used RFM grid
    r"[1-2][1-2]": "Hibernating",
    r"[1-2][3-4]": "At Risk",
    r"[1-2]5": "Can't Lose Them",
    r"3[1-2]": "About to Sleep",
    r"33": "Need Attention",
    r"[3-4][4-5]": "Loyal Customers",
    r"41": "Promising",
    r"51": "New Customers",
    r"[4-5][2-3]": "Potential Loyalists",
    r"5[4-5]": "Champions",
}

ACTIONS = {
    "Champions": "Reward them: early access, referral programme, premium bundles.",
    "Loyal Customers": "Upsell higher-value bikes and accessories; ask for reviews.",
    "Potential Loyalists": "Membership / loyalty offer to turn them into regulars.",
    "New Customers": "Onboarding emails, accessories that match their first purchase.",
    "Promising": "Time-limited offer to secure the second purchase.",
    "Need Attention": "Personalised recommendations before they drift away.",
    "About to Sleep": "Re-engage with popular products and a small discount.",
    "At Risk": "Win-back campaign: 'we miss you' + strong incentive.",
    "Can't Lose Them": "Personal outreach from sales; find out why they stopped.",
    "Hibernating": "Low-cost reactivation only; exclude from expensive campaigns.",
}


def customer_table(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per Internet customer with recency (days), frequency (orders) and monetary (sales)."""
    internet = df[(df["channel"] == "Internet") & df["customer_id"].notna()]
    if internet.empty:
        return pd.DataFrame(columns=["customer_id", "customer", "country", "recency", "frequency", "monetary"])
    as_of = as_of or internet["order_date"].max() + pd.Timedelta(days=1)
    grouped = internet.groupby("customer_id").agg(
        customer=("customer", "first"),
        country=("customer_country", "first"),
        last_order=("order_date", "max"),
        first_order=("order_date", "min"),
        frequency=("order_id", "nunique"),
        monetary=("sales", "sum"),
    )
    grouped["recency"] = (as_of - grouped["last_order"]).dt.days
    return grouped.reset_index()


def _score(values: pd.Series, ascending: bool) -> pd.Series:
    """Quintile score 1-5. Ranking first makes ties (e.g. many 1-order customers) splittable."""
    ranks = values.rank(method="first", ascending=ascending)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm_segments(customers: pd.DataFrame) -> pd.DataFrame:
    out = customers.copy()
    if out.empty:
        return out.assign(R=[], F=[], M=[], segment=[])
    out["R"] = _score(out["recency"], ascending=False)    # recent buyers get 5
    out["F"] = _score(out["frequency"], ascending=True)
    out["M"] = _score(out["monetary"], ascending=True)
    key = out["R"].astype(str) + out["F"].astype(str)
    out["segment"] = key.replace(SEGMENT_MAP, regex=True)
    return out


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    total_sales = rfm["monetary"].sum()
    summary = (
        rfm.groupby("segment")
        .agg(customers=("customer_id", "count"), revenue=("monetary", "sum"),
             avg_recency_days=("recency", "mean"), avg_orders=("frequency", "mean"),
             avg_spend=("monetary", "mean"))
        .reset_index()
    )
    summary["revenue_share_%"] = (summary["revenue"] / total_sales * 100).round(1)
    summary["customers_%"] = (summary["customers"] / len(rfm) * 100).round(1)
    summary["recommended_action"] = summary["segment"].map(ACTIONS)
    return summary.sort_values("revenue", ascending=False).round(
        {"avg_recency_days": 0, "avg_orders": 2, "avg_spend": 0, "revenue": 0}
    ).reset_index(drop=True)


def kmeans_clusters(rfm: pd.DataFrame, k_range=range(3, 7), random_state: int = 42,
                    sample: int = 6000) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Cluster customers on log-scaled, standardised RFM; pick k by silhouette score.

    Returns (customers with cluster label, cluster profile, chosen k).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    features = np.log1p(rfm[["recency", "frequency", "monetary"]].to_numpy(dtype=float))
    x = StandardScaler().fit_transform(features)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)

    best_k, best_score, best_model = None, -1.0, None
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit(x)
        score = silhouette_score(x[idx], model.labels_[idx])
        if score > best_score:
            best_k, best_score, best_model = k, score, model

    labelled = rfm.copy()
    labelled["cluster"] = best_model.labels_
    profile = (
        labelled.groupby("cluster")
        .agg(customers=("customer_id", "count"), avg_recency_days=("recency", "mean"),
             avg_orders=("frequency", "mean"), avg_spend=("monetary", "mean"), revenue=("monetary", "sum"))
        .sort_values("avg_spend", ascending=False)
    )
    rename = _describe_clusters(profile)
    labelled["cluster_name"] = labelled["cluster"].map(rename)
    profile.index = profile.index.map(rename)
    profile = profile.reset_index(names="cluster").round({"avg_recency_days": 0, "avg_orders": 2,
                                                          "avg_spend": 0, "revenue": 0})
    profile.attrs["silhouette"] = round(float(best_score), 3)
    return labelled, profile, int(best_k)


def _describe_clusters(profile: pd.DataFrame) -> dict:
    """Readable names such as 'Recent repeat buyers · high spend' from cluster averages."""
    spend_rank = profile["avg_spend"].rank(pct=True)
    recency_median = profile["avg_recency_days"].median()
    names = {}
    for cluster, row in profile.iterrows():
        recency = "Recent" if row["avg_recency_days"] <= recency_median else "Lapsed"
        frequency = "repeat buyers" if row["avg_orders"] >= 1.5 else "one-time buyers"
        spend = "high spend" if spend_rank[cluster] > 2 / 3 else ("mid spend" if spend_rank[cluster] > 1 / 3 else "low spend")
        name = f"{recency} {frequency} · {spend}"
        while name in names.values():  # keep names unique
            name += "*"
        names[cluster] = name
    return names


def cohort_retention(df: pd.DataFrame, max_months: int = 12) -> pd.DataFrame:
    """Share of each first-purchase cohort that orders again N months later (Internet)."""
    internet = df[(df["channel"] == "Internet") & df["customer_id"].notna()][["customer_id", "order_date"]].copy()
    if internet.empty:
        return pd.DataFrame()
    internet["month"] = internet["order_date"].dt.to_period("M")
    first = internet.groupby("customer_id")["month"].min().rename("cohort")
    internet = internet.join(first, on="customer_id")
    internet["age"] = (internet["month"] - internet["cohort"]).apply(lambda d: d.n)
    active = internet.groupby(["cohort", "age"])["customer_id"].nunique().unstack(fill_value=0)
    sizes = active[0]
    retention = active.div(sizes, axis=0).iloc[:, : max_months + 1] * 100
    # months that have not happened yet are unknown, not 0 %
    last = internet["month"].max()
    for cohort in retention.index:
        future = [age for age in retention.columns if (cohort + age) > last]
        retention.loc[cohort, future] = np.nan
    retention.insert(0, "customers", sizes)
    retention.index = retention.index.astype(str)
    return retention.round(1)
