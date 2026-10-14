"""The AI analyst: the locked SQL database, the tool loop, the number check, charts, CLI and app."""

from __future__ import annotations

import json
import re
import time

import pandas as pd
import pytest

from salespulse.__main__ import main as cli
from salespulse.analyst import (
    EXAMPLES, SYSTEM_PROMPT, TOOLS, Analyst, ask, chart_spec, check_numbers, result_text, run_example,
)
from salespulse.data import fiscal_year
from salespulse.llm import LLMClient, LLMError, ScriptedLLM
from salespulse.metrics import Filters, apply_filters, breakdown
from salespulse.sql import QueryError, QueryResult, SalesDB

FY2020_MARGIN_SQL = EXAMPLES[0].sql


def sql_call(sql: str, title: str = "Result") -> dict:
    return {"tool_calls": [{"name": "run_sql", "arguments": {"sql": sql, "title": title}}]}


def result(frame: pd.DataFrame) -> QueryResult:
    return QueryResult("SELECT 1", frame, False, 0.0)


# ------------------------------------------------------------------ database
@pytest.mark.parametrize("sql", [
    "DELETE FROM sales", "DROP TABLE sales", "INSERT INTO sales SELECT * FROM sales LIMIT 1",
    "UPDATE sales SET sales = 0", "COPY sales TO 'out.csv'", "ATTACH 'other.db'", "SET threads = 1",
    "INSTALL httpfs", "CREATE TABLE x AS SELECT 1", "SELECT 1; DROP TABLE sales", "",
])
def test_database_accepts_only_one_select(db, sql):
    with pytest.raises(QueryError):
        db.run(sql)
    assert db.run("SELECT count(*) AS n FROM sales").frame["n"][0] == 121_253


@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM glob('*')", "SELECT * FROM 'data/sales.parquet'",
    "SELECT getenv('GEMINI_API_KEY')", "SELECT * FROM private_frame",
])
def test_database_cannot_read_files_secrets_or_python_objects(db, sql):
    private_frame = pd.DataFrame({"secret": ["do not leak"]})  # DuckDB can read local DataFrames by default
    with pytest.raises(QueryError):
        db.run(sql)
    assert len(private_frame) == 1


def test_locked_settings_stay_locked(db):
    assert db.run("SELECT current_setting('enable_external_access') AS v").frame["v"][0] in (False, "false")
    with pytest.raises(QueryError):
        db.run("SET enable_external_access = true")


def test_slow_queries_are_stopped(sales):
    slow = SalesDB(sales, timeout=0.5)
    started = time.perf_counter()
    with pytest.raises(QueryError, match="stopped"):
        slow.run("SELECT count(*) FROM sales a, sales b, sales c")
    assert time.perf_counter() - started < 5


def test_row_limit_types_and_cleanup(db):
    big = db.run("SELECT * FROM sales")
    assert big.truncated and big.rows == db.max_rows
    assert pd.api.types.is_datetime64_any_dtype(big.frame["order_date"])
    small = db.run("```sql\nSELECT CAST(1.25 AS DECIMAL(10, 2)) AS x, 2 AS x;;\n```")
    assert list(small.frame.columns) == ["x", "x_2"] and small.frame["x"][0] == 1.25 and not small.truncated


def test_schema_describes_every_column(db):
    schema = db.schema()
    for name, _ in db.columns:
        assert f"- {name} " in schema
    assert "121,253 rows" in schema and "fiscal_year 2020 = 2019-07-01 to 2020-06-30" in schema
    assert "{schema}" in SYSTEM_PROMPT and TOOLS[0]["function"]["name"] == "run_sql"


def test_sql_agrees_with_the_dashboard_code(sales, db):
    fy = sales.assign(fy=fiscal_year(sales["order_date"])).groupby("fy", observed=True)["sales"].sum()
    by_sql = db.run("SELECT fiscal_year, SUM(sales) AS revenue FROM sales GROUP BY 1 ORDER BY 1").frame
    assert list(by_sql["revenue"].round(2)) == list(fy.round(2))

    fy2020 = apply_filters(sales, Filters(pd.Timestamp("2019-07-01"), pd.Timestamp("2020-06-30")))
    expected = breakdown(fy2020, "country").set_index("country")["margin_pct"].round(1)
    got = db.run(FY2020_MARGIN_SQL).frame.set_index("country")["margin_pct"]
    assert got.to_dict() == pytest.approx(expected.to_dict())


# ------------------------------------------------------------------ examples
@pytest.mark.parametrize("example", EXAMPLES, ids=[e.title for e in EXAMPLES])
def test_every_example_answer_is_backed_by_its_result(db, example):
    answer = run_example(db, example)
    assert answer.mode == "example" and answer.final.ok and answer.final.rows >= 2
    assert answer.check.ok and len(answer.check.matched) >= 2, answer.text
    assert chart_spec(answer.final.result.frame, answer.final.sql) is not None


def test_example_answers_quote_the_real_numbers(db):
    margin = run_example(db, EXAMPLES[0])
    assert margin.text.startswith("Australia had the highest gross margin in FY2020: 27.1%")
    losses = run_example(db, EXAMPLES[1])
    assert "lost $1.2M on Touring Bikes (-11.8% margin)" in losses.text


def test_without_a_model_only_examples_are_answered(db):
    assert ask(EXAMPLES[2].question, db).mode == "example"
    with pytest.raises(LLMError, match="API key"):
        ask("Which customer spent the most?", db)


# --------------------------------------------------------------- number check
def test_number_check_allows_rounding_and_flags_the_rest():
    frame = pd.DataFrame({"product": ["Road-250 Black, 44", "Touring-1000"],
                          "revenue": [5_812_345.67, 412_000.0], "margin_pct": [27.1, -11.8],
                          "profit": [-1_237_594.98, 9_000.0], "orders": [2030, 15]})
    text = ("Road-250 Black, 44 made $5.8M (27.1% margin) from 2,030 orders, but Touring lost −$1.2M (an 11.8% loss). "
            "That is 35% more than in 2019 and the top 3 products. Revenue was $412,000 in June 15 terms, Q2 FY2020.")
    check = check_numbers(text, [result(frame)])
    assert check.matched == ["$5.8M", "27.1%", "2,030", "−$1.2M", "11.8%", "$412,000"]
    assert check.unmatched == ["35%"] and not check.ok


def test_number_check_ignores_numbers_from_the_question():
    check = check_numbers("Here are the top 25 products.", [result(pd.DataFrame({"x": [1.0]}))],
                          "Show the top 25 products")
    assert check.matched == [] and check.unmatched == []


# ------------------------------------------------------------------- charts
def test_chart_spec_picks_line_bar_or_nothing():
    months = pd.DataFrame({"month": pd.date_range("2019-01-01", periods=12, freq="MS"), "revenue": range(12)})
    assert chart_spec(months).kind == "line"
    years = pd.DataFrame({"fiscal_year": [2018, 2019, 2020], "new_customers": [1, 2, 3]})
    assert chart_spec(years).kind == "bar"
    countries = pd.DataFrame({"country": ["A", "B", "C"], "revenue": [3.0, 2, 1], "margin_pct": [1.0, 5, 3]})
    spec = chart_spec(countries, "SELECT ... ORDER BY margin_pct DESC")
    assert (spec.kind, spec.x, spec.y, spec.percent) == ("barh", "margin_pct", "country", True)
    assert chart_spec(countries).x == "revenue" and chart_spec(countries).money
    assert chart_spec(countries.head(1)) is None
    assert chart_spec(pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})) is None
    by_channel = pd.DataFrame({"month": pd.to_datetime(["2020-01-01"] * 2 + ["2020-02-01"] * 2),
                               "channel": ["Internet", "Reseller"] * 2, "revenue": [1.0, 2, 3, 4]})
    assert chart_spec(by_channel).color == "channel"


# ------------------------------------------------------------------- the agent
def test_agent_fixes_its_own_sql_error_then_answers(db):
    llm = ScriptedLLM([
        sql_call("SELECT country, SUM(revenue) FROM sales GROUP BY country", "First try"),
        sql_call(FY2020_MARGIN_SQL, "Gross margin by country, FY2020"),
        "Australia had the highest gross margin in fiscal 2020: 27.1% on $5.7M of revenue.",
    ])
    seen = []
    answer = Analyst(llm, db).ask("Which country had the best margin in FY2020?", on_step=seen.append)
    assert [s.ok for s in answer.steps] == [False, True] and seen == answer.steps
    assert "revenue" in answer.steps[0].error
    assert answer.final.title == "Gross margin by country, FY2020" and answer.final.rows == 6
    assert answer.llm_calls == 3 and answer.check.ok and answer.check.matched == ["27.1%", "$5.7M"]
    tool_messages = [m for m in llm.calls[-1] if m["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("ERROR:") and "Fix the query" in tool_messages[0]["content"]
    assert tool_messages[1]["content"].startswith("6 rows\ncountry | revenue | profit | margin_pct\nAustralia |")


def test_agent_survives_bad_tool_calls(db):
    llm = ScriptedLLM([
        {"tool_calls": [{"name": "delete_everything", "arguments": {}},
                        {"name": "run_sql", "arguments": "{not json"},
                        {"name": "run_sql", "arguments": {"sql": "DROP TABLE sales", "title": "Oops"}}]},
        "I couldn't run that.",
    ])
    answer = Analyst(llm, db).ask("Delete the data")
    feedback = [m["content"] for m in llm.calls[-1] if m["role"] == "tool"]
    assert "unknown tool" in feedback[0] and "JSON object" in feedback[1] and "Only read-only SELECT" in feedback[2]
    assert len(answer.steps) == 1 and not answer.steps[0].ok and answer.final is None
    assert db.run("SELECT count(*) AS n FROM sales").frame["n"][0] == 121_253


def test_invented_numbers_are_flagged(db):
    llm = ScriptedLLM([sql_call(FY2020_MARGIN_SQL), "Australia leads with 27.1%, 4.5 points ahead of last year."])
    answer = Analyst(llm, db).ask("Best margin?")
    assert answer.check.matched == ["27.1%"] and answer.check.unmatched == ["4.5"]


def test_agent_stops_after_max_steps(db):
    llm = ScriptedLLM(default=sql_call("SELECT 1 AS one"))
    answer = Analyst(llm, db, max_steps=3).ask("Loop forever")
    assert answer.llm_calls == 3 and len(answer.steps) == 3 and "couldn't finish" in answer.text


def test_follow_up_questions_get_the_previous_answer(db):
    llm = ScriptedLLM(["Same for 2019: Australia again."])
    Analyst(llm, db).ask("And in 2019?", history=[("Best margin in 2020?", "Australia, 27.1%.")])
    roles = [(m["role"], m["content"]) for m in llm.calls[0]]
    assert roles[1:] == [("user", "Best margin in 2020?"), ("assistant", "Australia, 27.1%."),
                         ("user", "And in 2019?")]


def test_result_text_is_compact_and_capped(db):
    text = result_text(db.run("SELECT order_line, order_date, sales FROM sales ORDER BY order_line LIMIT 80"))
    lines = text.splitlines()
    assert lines[0] == "80 rows, showing the first 50" and len(lines) == 52
    assert re.fullmatch(r"\d+ \| \d{4}-\d{2}-\d{2} \| \d+(\.\d{1,2})?", lines[2]), lines[2]
    assert result_text(db.run("SELECT * FROM sales WHERE country = 'Mars'")).startswith("The query returned no rows")


def test_full_loop_over_http_with_function_calling(db, fake_api):
    def call(call_id, sql):
        return {"id": call_id, "type": "function",
                "function": {"name": "run_sql", "arguments": json.dumps({"sql": sql, "title": "Margin"})}}

    fake_api.script = [
        {"role": "assistant", "content": None, "tool_calls": [call("c1", FY2020_MARGIN_SQL)],
         "extra_content": {"google": {"thought_signature": "sig-1"}}},
        {"role": "assistant", "content": "Australia: 27.1% gross margin in FY2020."},
    ]
    llm = LLMClient("gemini", api_key="test-key", base_url=fake_api.base_url, backoff=0.01)
    answer = Analyst(llm, db).ask("Which country had the best margin in FY2020?")
    assert answer.text == "Australia: 27.1% gross margin in FY2020." and answer.check.ok
    first, second = (r["body"] for r in fake_api.requests)
    assert first["tools"][0]["function"]["name"] == "run_sql" and first["tool_choice"] == "auto"
    assert "Table `sales`" in first["messages"][0]["content"]
    echoed, tool = second["messages"][-2], second["messages"][-1]
    assert echoed["extra_content"]["google"]["thought_signature"] == "sig-1"
    assert tool["tool_call_id"] == "c1" and tool["content"].startswith("6 rows")


def test_http_errors_become_friendly_messages(db, fake_api):
    fake_api.status_queue = [(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})]
    llm = LLMClient("gemini", api_key="bad", base_url=fake_api.base_url, backoff=0.01)
    with pytest.raises(LLMError, match="API key was rejected"):
        Analyst(llm, db).ask("Revenue?")


# --------------------------------------------------------------------- CLI & app
def test_cli_runs_the_examples(capsys):
    assert cli(["examples"]) == 0
    out = capsys.readouterr().out
    assert out.count("\nQ: ") == len(EXAMPLES) and "Numbers checked" in out and "SELECT country" in out


def test_app_answers_an_example_in_demo_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not app.exception
    app.button(key="example_1").click().run()
    assert not app.exception
    texts = " ".join(m.value for m in app.markdown)
    assert "resellers lost $1.2M on Touring Bikes" in texts.replace("\\$", "$")
