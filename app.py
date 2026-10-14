"""SalesPulse - sales analytics dashboard with forecasting and customer segmentation.

Start with:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from salespulse import insights
from salespulse.analyst import EXAMPLES, Analyst, AnalystAnswer, chart_spec, run_example
from salespulse.data import DEFAULT_PARQUET, fiscal_year, load_sales
from salespulse.forecast import forecast, monthly_series
from salespulse.metrics import (
    Filters, apply_filters, breakdown, fiscal_channel_table, kpi_deltas, kpis, monthly, pareto,
    same_period_last_year,
)
from salespulse.llm import PROVIDERS, LLMClient, LLMError
from salespulse.report import build_report
from salespulse.segmentation import (
    cohort_retention, customer_table, kmeans_clusters, rfm_segments, segment_summary,
)
from salespulse.sql import SalesDB

st.set_page_config(page_title="SalesPulse · Sales Analytics", page_icon="📈", layout="wide")

# Validated colour-blind-safe categorical order; one hue for single-series charts.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_SCALE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
LOSS = "#e34948"
UPWORK_URL = "https://www.upwork.com/freelancers/~01f79195ceec502c59"
ENGINES = {
    "Example questions (no API key)": None,
    "Google Gemini (free tier)": "gemini",
    "OpenAI": "openai",
    "Groq (free tier)": "groq",
    "Ollama (runs on your computer)": "ollama",
}


def style(fig, height: int = 340, legend: bool = False):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=44, b=8),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
        title_font=dict(size=15), colorway=PALETTE, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        bargap=0.25, bargroupgap=0.06, barcornerradius=4, hoverlabel=dict(bgcolor="white", font_size=12), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
    )
    axis = dict(gridcolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED), title_font=dict(size=13, color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis, zeroline=False)
    return fig


def show(fig, **kwargs):
    st.plotly_chart(fig, theme=None, width="stretch", **kwargs)


def money(x: float) -> str:
    return insights.money(x)


@st.cache_data(show_spinner="Loading 121,253 sales lines…")
def data() -> pd.DataFrame:
    return load_sales(DEFAULT_PARQUET)


@st.cache_data(show_spinner="Segmenting customers…")
def customer_models(frame: pd.DataFrame):
    customers = customer_table(frame)
    if len(customers) < 50:
        return None
    rfm = rfm_segments(customers)
    labelled, profile, k = kmeans_clusters(rfm)
    return rfm, segment_summary(rfm), labelled, profile, k


@st.cache_data(show_spinner="Backtesting forecast models…")
def run_forecast(frame: pd.DataFrame, horizon: int):
    series = monthly_series(frame)
    if len(series) < 18:
        return None
    return forecast(series, horizon=horizon)


@st.cache_data(show_spinner="Building the Excel report…")
def report_bytes(frame: pd.DataFrame, label: str) -> bytes:
    return build_report(frame, label)


@st.cache_resource(show_spinner="Preparing the SQL database…")
def sales_db() -> SalesDB:
    return SalesDB(load_sales(DEFAULT_PARQUET))


def secret(name: str | None) -> str | None:
    if not name:
        return None
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file
        value = None
    return value or os.environ.get(name)


df_all = data()
min_date, max_date = df_all["order_date"].min().date(), df_all["order_date"].max().date()
fiscal_years = sorted(fiscal_year(df_all["order_date"]).unique(), reverse=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📈 SalesPulse")
    st.caption("Sales analytics on Microsoft's AdventureWorks sample: 121K order lines, "
               "Jul 2017 – Jun 2020, 2 channels, 6 countries.")
    period = st.radio("Period", [*fiscal_years, "All time", "Custom"], index=0, horizontal=True,
                      help="Fiscal years run July → June.")
    if period == "All time":
        start, end = min_date, max_date
    elif period == "Custom":
        picked = st.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
        start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (min_date, max_date))
    else:
        year = int(period[2:])
        start, end = max(pd.Timestamp(year - 1, 7, 1).date(), min_date), min(pd.Timestamp(year, 6, 30).date(), max_date)
    channels = st.multiselect("Channel", sorted(df_all["channel"].cat.categories), placeholder="All channels")
    groups = st.multiselect("Region", sorted(df_all["territory_group"].cat.categories.drop("Corporate HQ", errors="ignore")),
                            placeholder="All regions")
    categories = st.multiselect("Product category", sorted(df_all["category"].cat.categories), placeholder="All categories")
    st.divider()
    st.markdown("**AI engine** · for *Ask your data*")
    engine = st.selectbox("AI engine", list(ENGINES), index=1 if secret("GEMINI_API_KEY") else 0,
                          label_visibility="collapsed")
    provider = ENGINES[engine]
    llm, llm_error = None, None
    if provider:
        cfg = PROVIDERS[provider]
        stored_key = secret(cfg["env"])
        typed_key = ""
        if provider != "ollama":
            typed_key = st.text_input("API key", type="password",
                                      placeholder="Using the key from the app settings" if stored_key
                                      else "Paste your API key", help="Used only for this session and never saved.")
            if not (typed_key or stored_key):
                st.caption(f"Get a free key: [{cfg['key_url'].split('/')[2]}]({cfg['key_url']})")
        else:
            st.caption("Needs [Ollama](https://ollama.com/download) running on this computer.")
        model = st.text_input("Model", cfg["model"])
        try:
            llm = LLMClient(provider, api_key=typed_key.strip() or stored_key, model=model.strip() or None)
        except LLMError as exc:
            llm_error = str(exc)
    st.divider()
    st.caption("Data: [AdventureWorks Sales sample](https://github.com/microsoft/powerbi-desktop-samples) "
               "by Microsoft (MIT licence).")
    st.caption(f"Built by **Fatma El Zahraa**. Want this on your own sales data? [Let's talk on Upwork]({UPWORK_URL}).")

filters = Filters(pd.Timestamp(start), pd.Timestamp(end), channels or None, groups or None, None, categories or None)
df = apply_filters(df_all, filters)
label = (f"{period if period not in ('Custom',) else f'{start} → {end}'}"
         f"{' · ' + ', '.join(channels) if channels else ''}{' · ' + ', '.join(groups) if groups else ''}"
         f"{' · ' + ', '.join(categories) if categories else ''}")

st.header("Sales Performance Dashboard")
st.caption(f"Showing **{label}** · {len(df):,} order lines")
if df.empty:
    st.warning("No sales match these filters.")
    st.stop()

overview, ask_tab, products_tab, customers_tab, forecast_tab, report_tab = st.tabs(
    ["📊 Overview", "✨ Ask your data", "📦 Products", "👥 Customers", "🔮 Forecast", "📥 Report"]
)

# ------------------------------------------------------------------ overview
with overview:
    k = kpis(df)
    deltas = {}
    if period not in ("All time",):
        prev = same_period_last_year(df_all, df, filters)
        if not prev.empty:
            deltas = kpi_deltas(k, kpis(prev))

    def delta(key, pp=False):
        v = deltas.get(key)
        if v is None:
            return None
        return f"{v:+.1f} pp vs last year" if pp else f"{v:+.0f}% vs last year"

    c = st.columns(5)
    c[0].metric("Revenue", money(k["revenue"]), delta("revenue"))
    c[1].metric("Gross profit", money(k["profit"]), delta("profit"))
    c[2].metric("Gross margin", f"{k['margin_pct']:.1f}%", delta("margin_pct", pp=True))
    c[3].metric("Orders", f"{k['orders']:,}", delta("orders"))
    c[4].metric("Avg order value", money(k["aov"]), delta("aov"))

    with st.container(border=True):
        st.markdown("**Executive summary**")
        # escape "$" so Streamlit does not treat dollar amounts as LaTeX
        st.markdown("\n".join(f"- {line}".replace("$", "\\$") for line in insights.generate(df)))

    m = monthly(df)
    fig = go.Figure()
    fig.add_scatter(x=m["month"], y=m["revenue"], name="Revenue", mode="lines+markers",
                    line=dict(color=PALETTE[0], width=2), marker=dict(size=6))
    fig.add_scatter(x=m["month"], y=m["profit"], name="Gross profit", mode="lines+markers",
                    line=dict(color=PALETTE[1], width=2), marker=dict(size=6))
    fig.update_layout(title="Monthly revenue and gross profit (complete months)", yaxis_title="USD",
                      hovermode="x unified")
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    show(style(fig, 360, legend=True))

    left, right = st.columns(2)
    fy = fiscal_channel_table(df).melt(id_vars="fiscal_year", var_name="channel", value_name="revenue")
    fig = px.bar(fy, x="fiscal_year", y="revenue", color="channel", barmode="group",
                 title="Revenue by fiscal year and channel", labels={"fiscal_year": "", "revenue": "Revenue"},
                 color_discrete_sequence=PALETTE)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    with left:
        show(style(fig, legend=True))

    countries = breakdown(df, "country").sort_values("revenue")
    fig = px.bar(countries, x="revenue", y="country", orientation="h", title="Revenue by country",
                 labels={"revenue": "Revenue", "country": ""}, hover_data={"margin_pct": ":.1f", "share_pct": ":.1f"})
    fig.update_traces(marker_color=PALETTE[0])
    fig.update_xaxes(tickprefix="$", tickformat="~s")
    with right:
        show(style(fig))

    left, right = st.columns(2)
    cats = breakdown(df, "category").sort_values("revenue")
    fig = px.bar(cats, x="revenue", y="category", orientation="h", title="Revenue by product category",
                 labels={"revenue": "Revenue", "category": ""}, hover_data={"share_pct": ":.1f"})
    fig.update_traces(marker_color=PALETTE[0])
    fig.update_xaxes(tickprefix="$", tickformat="~s")
    with left:
        show(style(fig, 300))
    ch = breakdown(df, "channel")
    fig = px.bar(ch, x="channel", y="margin_pct", title="Gross margin by channel",
                 labels={"channel": "", "margin_pct": "Gross margin %"}, text=ch["margin_pct"].map("{:.1f}%".format))
    fig.update_traces(marker_color=PALETTE[0], textposition="outside", cliponaxis=False)
    with right:
        show(style(fig, 300))

# ------------------------------------------------------------- ask your data
def pretty(name: str) -> str:
    """margin_pct -> Margin %, new_customers -> New customers."""
    text = str(name).replace("_pct", " %").replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def answer_chart(answer: AnalystAnswer) -> None:
    final = answer.final
    frame = final.result.frame
    spec = chart_spec(frame, final.sql)
    if spec is None:
        return
    data_ = frame.copy()
    if spec.kind == "barh":
        data_ = data_.sort_values(spec.x)
        colors = [LOSS if v < 0 else PALETTE[0] for v in data_[spec.x]]
        fig = px.bar(data_, x=spec.x, y=spec.y, orientation="h", title=final.title,
                     labels={spec.y: "", spec.x: pretty(spec.x)})
        fig.update_traces(marker_color=colors)
        axis = fig.update_xaxes
        height = max(260, 34 * len(data_) + 90)
    else:
        labels = {spec.x: "", spec.y: pretty(spec.y), **({spec.color: ""} if spec.color else {})}
        if spec.kind == "bar":
            prefix = "FY" if "fiscal" in str(spec.x).lower() else ""
            data_[spec.x] = prefix + data_[spec.x].astype(str)
            fig = px.bar(data_, x=spec.x, y=spec.y, color=spec.color, title=final.title, labels=labels,
                         color_discrete_sequence=PALETTE, barmode="group")
        else:
            fig = px.line(data_, x=spec.x, y=spec.y, color=spec.color, markers=True, title=final.title,
                          labels=labels, color_discrete_sequence=PALETTE)
            fig.update_traces(line=dict(width=2), marker=dict(size=7))
        if spec.color is None:
            fig.update_traces(marker_color=PALETTE[0], selector=dict(type="bar"))
        axis = fig.update_yaxes
        height = 340
    if spec.money:
        axis(tickformat="$~s")
    elif spec.percent:
        axis(ticksuffix="%")
    show(style(fig, height, legend=spec.color is not None))


def answer_table(answer: AnalystAnswer, key: str) -> None:
    final = answer.final
    frame = final.result.frame
    config = {}
    for column in frame.columns:
        name, lower, values = str(column), str(column).lower(), frame[column]
        label = pretty(name)
        if pd.api.types.is_datetime64_any_dtype(values):
            monthly_ = bool((values.dropna().dt.day == 1).all())
            config[name] = st.column_config.DateColumn(label, format="MMM YYYY" if monthly_ else "YYYY-MM-DD")
        elif pd.api.types.is_float_dtype(values):
            if any(w in lower for w in ("pct", "percent", "margin", "share", "rate", "growth")):
                config[name] = st.column_config.NumberColumn(label, format="%.1f%%")
            elif any(w in lower for w in ("revenue", "sales", "profit", "cost", "price", "value", "spend",
                                          "aov", "amount", "loss")):
                config[name] = st.column_config.NumberColumn(label, format="dollar")
            else:
                config[name] = st.column_config.NumberColumn(label, format="localized")
        elif pd.api.types.is_integer_dtype(values):
            year_like = "year" in lower and values.between(1900, 2100).all()
            config[name] = st.column_config.NumberColumn(label, format="%d" if year_like else "localized")
        else:
            config[name] = st.column_config.TextColumn(label)
    st.dataframe(frame, hide_index=True, width="stretch", column_config=config,
                 height=min(35 * (len(frame) + 1) + 3, 420))
    note = f"{len(frame):,} rows" + (" (first rows only)" if final.result.truncated else "")
    left, right = st.columns([3, 1], vertical_alignment="center")
    left.caption(note)
    right.download_button("Download CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                          f"{final.title[:40].strip().replace(' ', '_').lower() or 'result'}.csv", "text/csv",
                          key=f"csv_{key}", width="stretch", icon=":material/download:")


def answer_card(answer: AnalystAnswer, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{answer.question}**")
        st.markdown(answer.text.replace("$", "\\$"))
        with st.container(horizontal=True):
            if answer.mode == "llm":
                st.badge(answer.engine, icon=":material/auto_awesome:", color="violet")
            else:
                st.badge("Example · hand-written SQL", icon=":material/code:", color="gray")
            queries = len(answer.steps)
            st.badge(f"{queries} quer{'y' if queries == 1 else 'ies'} · {answer.seconds:.1f}s",
                     icon=":material/database:", color="gray")
            check = answer.check
            if check.matched and check.ok:
                count = len(check.matched)
                st.badge(f"{count} number{'s' if count > 1 else ''} checked against the data",
                         icon=":material/verified:", color="green")
            elif check.unmatched:
                st.badge(f"Not in the query results: {', '.join(check.unmatched)}", icon=":material/warning:",
                         color="orange")
        if answer.final:
            answer_chart(answer)
            answer_table(answer, key)
        with st.expander(f"SQL and agent steps ({len(answer.steps)})"):
            for i, step in enumerate(answer.steps, 1):
                outcome = (f"{step.rows:,} rows · {step.seconds * 1000:.0f} ms" if step.ok
                           else f"❌ {step.error}")
                st.markdown(f"**{i}. {step.title}** · {outcome}")
                st.code(step.sql, language="sql")
            if answer.mode == "llm":
                st.caption(f"{answer.llm_calls} model calls. The model only writes SQL and the final sentences; "
                           "the numbers come from the database.")


with ask_tab:
    state = st.session_state
    state.setdefault("ai_answers", [])
    db = sales_db()
    top_left, top_right = st.columns([3, 2], vertical_alignment="bottom")
    with top_left:
        st.markdown("#### Ask your data")
    with top_right:
        with st.container(horizontal=True, horizontal_alignment="right"):
            if llm:
                st.badge(f"{llm.label} · {llm.model}", icon=":material/auto_awesome:", color="violet")
            else:
                st.badge("Demo mode · example questions", icon=":material/code:", color="gray")
    st.caption(f"Ask a question in plain English. The AI writes the SQL, a read-only database runs it on all "
               f"{db.row_count:,} order lines, and every number in the answer is checked against the result. "
               "The sidebar filters don't apply here.")
    if llm_error:
        st.warning(f"{llm_error} Until then, the example questions run with hand-written SQL.")
    asked = st.chat_input("Ask about revenue, profit, products, customers or resellers…" if llm else
                          "Choose an AI engine in the sidebar to ask your own questions", disabled=llm is None,
                          key="ai_question")
    st.markdown("**Try an example**")
    with st.container(horizontal=True):
        for i, example in enumerate(EXAMPLES):
            if st.button(example.question, key=f"example_{i}", width="content"):
                asked = example.question
    if asked and not llm:  # demo mode: the example's hand-written SQL, no AI
        example = next(e for e in EXAMPLES if e.question == asked)
        state.ai_answers.insert(0, run_example(db, example))
    elif asked:
        with st.status("Thinking…", expanded=True) as status:
            def progress(step):
                status.write(f"{'✅' if step.ok else '⚠️'} {step.title}"
                             + (f" · {step.rows:,} rows" if step.ok else f" · {step.error[:120]}"))
            try:
                history = [(a.question, a.for_history()) for a in reversed(state.ai_answers[:2]) if a.mode == "llm"]
                result = Analyst(llm, db).ask(asked, history, on_step=progress)
            except LLMError as exc:
                status.update(label="The AI engine returned an error", state="error")
                st.error(str(exc))
            else:
                state.ai_answers.insert(0, result)
                status.update(label=f"Answered in {result.seconds:.1f}s", state="complete", expanded=False)
    if state.ai_answers:
        for i, answer in enumerate(state.ai_answers):
            answer_card(answer, f"{len(state.ai_answers) - i}")
        if st.button("Clear answers", icon=":material/delete_sweep:"):
            state.ai_answers = []
            st.rerun()
    else:
        st.info("Pick an example question to see how it works: the answer, the chart, the table and the SQL behind it.",
                icon=":material/lightbulb:")

# ------------------------------------------------------------------ products
with products_tab:
    left, right = st.columns([3, 2])
    top = breakdown(df, "product", top=15).sort_values("revenue")
    fig = px.bar(top, x="revenue", y="product", orientation="h", title="Top 15 products by revenue",
                 labels={"revenue": "Revenue", "product": ""}, hover_data={"margin_pct": ":.1f", "units": True})
    fig.update_traces(marker_color=PALETTE[0])
    fig.update_xaxes(tickprefix="$", tickformat="~s")
    with left:
        show(style(fig, 520))
    table, stats = pareto(df, "product")
    fig = go.Figure()
    fig.add_scatter(x=table["rank_pct"], y=table["cum_share_pct"], mode="lines", line=dict(color=PALETTE[0], width=2),
                    name="Cumulative revenue share", hovertemplate="%{x:.0f}% of products → %{y:.0f}% of revenue<extra></extra>")
    fig.add_hline(y=80, line=dict(color=MUTED, dash="dot", width=1))
    fig.add_vline(x=stats["pct_items_for_80"], line=dict(color=MUTED, dash="dot", width=1))
    fig.add_annotation(x=stats["pct_items_for_80"], y=80, text=f"{stats['pct_items_for_80']:.0f}% of products = 80% of revenue",
                       showarrow=False, xanchor="left", yanchor="top", xshift=6, yshift=-6, font=dict(color=INK, size=12))
    fig.update_layout(title="Pareto: revenue concentration", xaxis_title="% of products (ranked)",
                      yaxis_title="% of revenue")
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%", range=[0, 102])
    with right:
        show(style(fig, 520))

    st.markdown("**Subcategory performance**")
    sub = breakdown(df, "subcategory")[["subcategory", "revenue", "profit", "margin_pct", "share_pct", "units"]]
    st.dataframe(sub, hide_index=True, width="stretch", column_config={
        "subcategory": st.column_config.TextColumn("Subcategory"),
        "revenue": st.column_config.NumberColumn("Revenue", format="$%,.0f"),
        "profit": st.column_config.NumberColumn("Gross profit", format="$%,.0f"),
        "margin_pct": st.column_config.NumberColumn("Margin", format="%.1f%%"),
        "share_pct": st.column_config.ProgressColumn("Share of revenue", format="%.1f%%", min_value=0,
                                                     max_value=float(max(sub["share_pct"].max(), 1))),
        "units": st.column_config.NumberColumn("Units", format="%,d"),
    })

# ----------------------------------------------------------------- customers
with customers_tab:
    models = customer_models(df)
    if models is None:
        st.info("Customer analytics use the **Internet** channel (individual customers). "
                "Include it in the channel filter to see segments.")
    else:
        rfm, seg, labelled, profile, k_best = models
        st.markdown(f"**{len(rfm):,} online customers** scored on Recency, Frequency and Monetary value (RFM).")
        left, right = st.columns(2)
        chart = seg.sort_values("revenue")
        fig = px.bar(chart, x="revenue_share_%", y="segment", orientation="h", title="Share of online revenue by segment",
                     labels={"revenue_share_%": "% of revenue", "segment": ""},
                     hover_data={"customers": ":,", "avg_spend": ":,.0f"})
        fig.update_traces(marker_color=PALETTE[0])
        fig.update_xaxes(ticksuffix="%")
        with left:
            show(style(fig, 400))
        chart = seg.sort_values("customers")
        fig = px.bar(chart, x="customers", y="segment", orientation="h", title="Customers per segment",
                     labels={"customers": "Customers", "segment": ""}, hover_data={"customers_%": ":.1f"})
        fig.update_traces(marker_color=PALETTE[0])
        with right:
            show(style(fig, 400))
        st.markdown("**What to do with each segment**")
        st.dataframe(seg[["segment", "customers", "revenue_share_%", "avg_spend", "recommended_action"]],
                     hide_index=True, width="stretch", height=390, column_config={
                         "segment": st.column_config.TextColumn("Segment"),
                         "customers": st.column_config.NumberColumn("Customers", format="%,d"),
                         "revenue_share_%": st.column_config.NumberColumn("Revenue", format="%.1f%%"),
                         "avg_spend": st.column_config.NumberColumn("Avg spend", format="$%,.0f"),
                         "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
                     })

        st.markdown(f"**Machine-learning view:** K-Means on log-scaled RFM picked **k = {k_best}** clusters "
                    f"(silhouette {profile.attrs.get('silhouette', 0):.2f}).")
        st.dataframe(profile, hide_index=True, width="stretch", column_config={
            "cluster": st.column_config.TextColumn("Cluster"),
            "customers": st.column_config.NumberColumn("Customers", format="%,d"),
            "avg_recency_days": st.column_config.NumberColumn("Days since last order", format="%d"),
            "avg_orders": st.column_config.NumberColumn("Orders per customer", format="%.2f"),
            "avg_spend": st.column_config.NumberColumn("Avg spend", format="$%,.0f"),
            "revenue": st.column_config.NumberColumn("Revenue", format="$%,.0f"),
        })

        retention = cohort_retention(df)
        if not retention.empty:
            heat = retention.drop(columns=["customers", 0], errors="ignore")
            fig = go.Figure(go.Heatmap(
                z=heat.to_numpy(), x=[f"M+{c}" for c in heat.columns], y=heat.index,
                colorscale=[[i / (len(BLUE_SCALE) - 1), c] for i, c in enumerate(BLUE_SCALE)],
                hovertemplate="Cohort %{y}, %{x}: %{z:.1f}% ordered again<extra></extra>",
                colorbar=dict(title="% active", ticksuffix="%"), xgap=2, ygap=2,
            ))
            fig.update_layout(title="Cohort retention: % of each first-purchase month that orders again",
                              yaxis=dict(autorange="reversed", title="First purchase month"))
            show(style(fig, 640))
            st.caption("Repeat purchases only appear from mid-2019, when accessories and clothing were added to the "
                       "online store — a cross-selling opportunity the bike-only catalogue did not have.")

# ------------------------------------------------------------------ forecast
with forecast_tab:
    horizon = st.slider("Months to forecast", 3, 12, 6)
    # A forecast needs the full history: keep the channel/region/category filters but not the start date.
    history_df = apply_filters(df_all, Filters(None, pd.Timestamp(end), filters.channels, filters.groups,
                                               None, filters.categories))
    st.caption(f"Trained on all history up to {pd.Timestamp(end):%d %b %Y} for the selected channel, region and category.")
    fc = run_forecast(history_df, horizon)
    if fc is None:
        st.info("Not enough history for a forecast with these filters (need at least 18 complete months).")
    else:
        hist = fc.history
        fig = go.Figure()
        fig.add_scatter(x=hist.index, y=hist.values, name="Actual", mode="lines+markers",
                        line=dict(color=PALETTE[0], width=2), marker=dict(size=6))
        t = fc.table
        fig.add_scatter(x=list(t["month"]) + list(t["month"][::-1]), y=list(t["upper"]) + list(t["lower"][::-1]),
                        fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), mode="lines",
                        name="80% range", hoverinfo="skip")
        fig.add_scatter(x=[hist.index[-1], *t["month"]], y=[hist.iloc[-1], *t["forecast"]], name="Forecast",
                        mode="lines+markers", line=dict(color=PALETTE[0], width=2, dash="dash"), marker=dict(size=6))
        fig.update_layout(title=f"Monthly revenue forecast · model: {fc.model}", hovermode="x unified")
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        show(style(fig, 400, legend=True))

        c1, c2, c3 = st.columns(3)
        c1.metric("Chosen model", fc.model)
        best_mape = float(fc.backtest.iloc[0]["MAPE %"])
        c2.metric("Backtest error (MAPE)", f"{best_mape:.1f}%")
        imp = fc.improvement_vs_baseline
        c3.metric("Better than baseline", f"{imp:.0f}%" if imp else "baseline is best",
                  help="Error reduction versus 'same month last year'.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Backtest** — 3 rolling cut-offs, each forecasting the following months")
            st.dataframe(fc.backtest, hide_index=True, width="stretch", column_config={
                "model": st.column_config.TextColumn("Model"),
                "MAPE %": st.column_config.NumberColumn("MAPE", format="%.1f%%"),
                "MAE": st.column_config.NumberColumn("MAE", format="$%,.0f"),
            })
        with right:
            st.markdown("**Forecast**")
            show_t = t.assign(month=t["month"].dt.strftime("%b %Y"))
            st.dataframe(show_t, hide_index=True, width="stretch", column_config={
                "month": st.column_config.TextColumn("Month"),
                "forecast": st.column_config.NumberColumn("Forecast", format="$%,.0f"),
                "lower": st.column_config.NumberColumn("Low", format="$%,.0f"),
                "upper": st.column_config.NumberColumn("High", format="$%,.0f"),
            })
        st.caption("Tip: forecast each channel separately (sidebar → Channel). The smooth online channel suits "
                   "Holt-Winters; lumpy reseller orders are best predicted by last year's pattern.")

# -------------------------------------------------------------------- report
with report_tab:
    st.markdown("Download everything for the current filters — ready to send to a manager.")
    st.markdown("- **Excel report**: KPI summary + insights, monthly trend, categories, countries, top products, "
                "customer segments and forecast, with native Excel charts\n"
                "- **CSV**: the filtered order lines")
    a, b = st.columns(2)
    a.download_button("⬇️ Excel report (.xlsx)", report_bytes(df, label), "salespulse_report.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                      width="stretch")
    b.download_button("⬇️ Filtered data (.csv)", df.to_csv(index=False).encode("utf-8-sig"), "sales_filtered.csv",
                      "text/csv", width="stretch")
