"""A read-only SQL database over the sales table, safe to hand to an AI.

DuckDB runs in memory on a copy of the data. Before an AI-written query runs:
* only one SELECT statement is accepted (no INSERT, UPDATE, COPY, ATTACH, SET, INSTALL ...);
* file and network access are switched off, Python objects can't be read, and these
  settings are locked so a query can't switch them back on;
* every query has a time limit and a row limit.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
import time
from dataclasses import dataclass
from decimal import Decimal

import duckdb
import pandas as pd

TABLE = "sales"

# What each column means, written for the model (and for people reading the prompt).
COLUMN_NOTES = {
    "order_line": "unique id of the order line",
    "order_id": "the order; one order has several lines, so count orders with COUNT(DISTINCT order_id)",
    "channel": "'Internet' (online store, individual customers) or 'Reseller' (bike shops and warehouses)",
    "order_date": "date of the order",
    "ship_date": "ship date (may be NULL)",
    "due_date": "due date",
    "fiscal_year": "fiscal year, July to June: fiscal_year 2020 = 2019-07-01 to 2020-06-30",
    "quantity": "units on the line",
    "unit_price": "price per unit in USD",
    "discount_pct": "always 0 in this data (no line discounts recorded)",
    "sales": "revenue of the line in USD",
    "cost": "product cost of the line in USD",
    "profit": "sales - cost in USD; gross margin % = 100 * SUM(profit) / SUM(sales)",
    "product": "product name with colour and size, e.g. 'Mountain-200 Black, 38'",
    "sku": "product code",
    "model": "product model without colour and size, e.g. 'Mountain-200'",
    "color": "product colour (may be NULL)",
    "subcategory": "e.g. 'Road Bikes', 'Mountain Bikes', 'Helmets', 'Jerseys' (35 values)",
    "category": "'Bikes', 'Accessories', 'Clothing' or 'Components'",
    "customer_id": "online customer id; NULL for Reseller lines",
    "customer": "online customer name; NULL for Reseller lines",
    "customer_city": "online customer city; NULL for Reseller lines",
    "customer_country": "online customer country; NULL for Reseller lines",
    "reseller": "reseller name; NULL for Internet lines",
    "business_type": "'Warehouse', 'Value Added Reseller' or 'Specialty Bike Shop'; NULL for Internet lines",
    "region": "sales region: a US region ('Southwest', 'Northwest', 'Southeast', 'Northeast', 'Central') "
              "or the country name outside the US",
    "country": "sales territory: 'United States', 'Canada', 'Australia', 'United Kingdom', 'France' or 'Germany'",
    "territory_group": "'North America', 'Europe' or 'Pacific'",
}

_SELECT_SQL = f"""
CREATE TABLE {TABLE} AS SELECT
    order_line, order_id, channel,
    CAST(order_date AS DATE) AS order_date, CAST(ship_date AS DATE) AS ship_date, CAST(due_date AS DATE) AS due_date,
    CAST(year(order_date) + CASE WHEN month(order_date) >= 7 THEN 1 ELSE 0 END AS INTEGER) AS fiscal_year,
    quantity, unit_price, discount_pct, sales, cost, profit,
    product, sku, model, color, subcategory, category,
    customer_id, customer, customer_city, customer_country, reseller, business_type,
    region, country, territory_group
FROM _frame
"""


class QueryError(ValueError):
    """A refused or failed query. The message is short and safe to show, and to send back to the model."""


@dataclass
class QueryResult:
    sql: str
    frame: pd.DataFrame
    truncated: bool  # True when the query returned more rows than the limit
    seconds: float

    @property
    def rows(self) -> int:
        return len(self.frame)


class SalesDB:
    """The sales table in an in-memory DuckDB, locked down for AI-written SQL."""

    def __init__(self, df: pd.DataFrame, timeout: float = 10.0, max_rows: int = 500):
        self.timeout = timeout
        self.max_rows = max_rows
        self.con = duckdb.connect(":memory:", config={"threads": 2, "memory_limit": "1GB"})
        frame = df.copy()
        for column in frame.select_dtypes("category").columns:
            frame[column] = frame[column].astype("string")
        self.con.register("_frame", frame)
        self.con.execute(_SELECT_SQL)
        self.con.unregister("_frame")
        for setting in ("SET enable_external_access = false", "SET python_enable_replacements = false",
                        "SET lock_configuration = true"):
            self.con.execute(setting)
        self.columns: list[tuple[str, str]] = [(row[0], row[1]) for row in self.con.execute(f"DESCRIBE {TABLE}").fetchall()]
        self.row_count, first, last = self.con.execute(
            f"SELECT count(*), min(order_date), max(order_date) FROM {TABLE}").fetchone()
        self.first_date: dt.date = first
        self.last_date: dt.date = last

    def run(self, sql: str, max_rows: int | None = None) -> QueryResult:
        limit = max_rows or self.max_rows
        sql = clean_sql(sql)
        check_sql(sql)
        cursor = self.con.cursor()
        timer = threading.Timer(self.timeout, cursor.interrupt)
        started = time.perf_counter()
        timer.start()
        try:
            cursor.execute(sql)
            names = [d[0] for d in cursor.description]
            rows = cursor.fetchmany(limit + 1)
        except duckdb.InterruptException as exc:
            raise QueryError(f"The query ran for more than {self.timeout:.0f} seconds and was stopped. "
                             "Simplify it (for example, don't join the table to itself).") from exc
        except duckdb.Error as exc:
            raise QueryError(short_error(exc)) from exc
        finally:
            timer.cancel()
            cursor.close()
        frame = to_frame(rows[:limit], names)
        return QueryResult(sql, frame, len(rows) > limit, time.perf_counter() - started)

    def schema(self) -> str:
        """The table description given to the model."""
        lines = [f"Table `{TABLE}`: one row per order line, {self.row_count:,} rows, orders from "
                 f"{self.first_date:%Y-%m-%d} to {self.last_date:%Y-%m-%d} (the last month is incomplete).",
                 "Columns:"]
        for name, kind in self.columns:
            lines.append(f"- {name} {kind}: {COLUMN_NOTES.get(name, '')}".rstrip(": "))
        return "\n".join(lines)


def clean_sql(sql: str) -> str:
    """Remove markdown fences and trailing semicolons that models like to add."""
    sql = re.sub(r"^\s*```(?:sql)?\s*|\s*```\s*$", "", sql or "", flags=re.I).strip()
    return re.sub(r"[;\s]+$", "", sql)


def check_sql(sql: str) -> None:
    if not sql:
        raise QueryError("The query is empty.")
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as exc:
        raise QueryError(short_error(exc)) from exc
    if len(statements) != 1:
        raise QueryError("Send exactly one SQL statement per query.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise QueryError("Only read-only SELECT queries are allowed.")


def short_error(exc: Exception) -> str:
    """DuckDB's message without the long caret diagram, kept short."""
    text = str(exc).split("\n\nLINE")[0].strip()
    return re.sub(r"\s+", " ", text)[:400]


def to_frame(rows: list[tuple], names: list[str]) -> pd.DataFrame:
    seen: dict[str, int] = {}
    unique = []
    for name in names:  # SELECT a, a would give two columns with the same name
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    frame = pd.DataFrame.from_records(rows, columns=unique) if rows else pd.DataFrame(columns=unique)
    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        first = values.iloc[0]
        if isinstance(first, Decimal):
            frame[column] = pd.to_numeric(frame[column].map(lambda v: None if v is None else float(v)))
        elif isinstance(first, (dt.date, dt.datetime)):
            frame[column] = pd.to_datetime(frame[column])
    return frame
