import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from salespulse.data import load_sales
from salespulse.sql import SalesDB


@pytest.fixture(scope="session")
def sales() -> pd.DataFrame:
    """The real, shipped dataset (121,253 order lines)."""
    return load_sales()


@pytest.fixture(scope="session")
def db(sales) -> SalesDB:
    """The locked-down SQL database the AI analyst queries."""
    return SalesDB(sales)


def make_lines(rows: list[dict]) -> pd.DataFrame:
    """Small hand-made sales table with the same columns as the real one."""
    base = {
        "order_line": 0, "order_id": "SO1", "channel": "Internet", "order_date": pd.Timestamp("2020-01-15"),
        "ship_date": pd.Timestamp("2020-01-20"), "due_date": pd.Timestamp("2020-01-27"), "quantity": 1,
        "unit_price": 100.0, "discount_pct": 0, "sales": 100.0, "cost": 60.0, "profit": 40.0,
        "product": "Bike A", "sku": "A", "model": "A", "color": "Red", "subcategory": "Road Bikes",
        "category": "Bikes", "customer_id": "C1", "customer": "Ann", "customer_city": "Paris",
        "customer_country": "France", "reseller": np.nan, "business_type": np.nan, "region": "France",
        "country": "France", "territory_group": "Europe",
    }
    out = []
    for i, row in enumerate(rows):
        line = {**base, "order_line": i + 1, **row}
        line["profit"] = line["sales"] - line["cost"]
        out.append(line)
    df = pd.DataFrame(out)
    for col in ["channel", "category", "subcategory", "color", "region", "country", "territory_group", "business_type"]:
        df[col] = df[col].astype("category")
    return df


@pytest.fixture
def lines():
    return make_lines


@pytest.fixture
def tiny_workbook(tmp_path) -> Path:
    """A 4-line workbook with the exact AdventureWorks sheet layout, including placeholder rows."""
    path = tmp_path / "aw.xlsx"
    na = "[Not Applicable]"
    sheets = {
        "Sales_data": pd.DataFrame({
            "SalesOrderLineKey": [1, 2, 3, 4], "ResellerKey": [-1, -1, 7, 7], "CustomerKey": [11, 11, -1, -1],
            "ProductKey": [1, 2, 1, 2], "OrderDateKey": [20190702, 20190910, 20190702, 20200105],
            "DueDateKey": [20190712, 20190920, 20190712, 20200115],
            "ShipDateKey": [20190709, None, 20190709, 20200112], "SalesTerritoryKey": [1, 1, 2, 2],
            "Order Quantity": [1, 2, 3, 1], "Unit Price": [500.0, 20.0, 450.0, 18.0],
            "Extended Amount": [500.0, 40.0, 1350.0, 18.0], "Unit Price Discount Pct": [0, 0, 0, 0],
            "Product Standard Cost": [300.0, 8.0, 300.0, 8.0], "Total Product Cost": [300.0, 16.0, 900.0, 8.0],
            "Sales Amount": [500.0, 40.0, 1350.0, 18.0],
        }),
        "Sales Order_data": pd.DataFrame({"Channel": ["Internet", "Internet", "Reseller", "Reseller"],
                                          "SalesOrderLineKey": [1, 2, 3, 4], "Sales Order": ["SO1", "SO2", "SO3", "SO4"],
                                          "Sales Order Line": ["SO1 - 1", "SO2 - 1", "SO3 - 1", "SO4 - 1"]}),
        "Product_data": pd.DataFrame({"ProductKey": [1, 2], "SKU": ["BK-1", "HL-1"], "Product": ["Road-150", "Helmet"],
                                      "Standard Cost": [300.0, 8.0], "Color": ["Red", None], "List Price": [500.0, 20.0],
                                      "Model": ["Road-150", "Sport-100"], "Subcategory": ["Road Bikes", "Helmets"],
                                      "Category": ["Bikes", "Accessories"]}),
        "Customer_data": pd.DataFrame({"CustomerKey": [-1, 11], "Customer ID": [na, "AW11"], "Customer": [na, "Ann Lee"],
                                       "City": [na, "Paris"], "State-Province": [na, "IDF"],
                                       "Country-Region": [na, "France"], "Postal Code": [na, "75001"]}),
        "Reseller_data": pd.DataFrame({"ResellerKey": [-1, 7], "Reseller ID": [na, "R7"],
                                       "Business Type": [na, "Warehouse"], "Reseller": [na, "Bike World"],
                                       "City": [na, "Berlin"], "State-Province": [na, "BE"],
                                       "Country-Region": [na, "Germany"], "Postal Code": [na, "10115"]}),
        "Sales Territory_data": pd.DataFrame({"SalesTerritoryKey": [1, 2], "Region": ["France", "Germany"],
                                              "Country": ["France", "Germany"], "Group": ["Europe", "Europe"]}),
    }
    with pd.ExcelWriter(path) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return path


class FakeOpenAI:
    """A tiny OpenAI-compatible server. `script` holds the assistant messages to return, in order."""

    def __init__(self):
        self.requests: list[dict] = []
        self.script: list[dict] = []
        self.status_queue: list[tuple[int, dict]] = []

    def handle(self, path: str, body: dict) -> tuple[int, dict]:
        self.requests.append({"path": path, "body": body})
        if self.status_queue:
            return self.status_queue.pop(0)
        message = self.script.pop(0) if self.script else {"role": "assistant", "content": "done"}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return 200, {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body.get("model"),
                     "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


@pytest.fixture()
def fake_api():
    state = FakeOpenAI()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            status, payload = state.handle(self.path, json.loads(self.rfile.read(length) or b"{}"))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()
    server.server_close()
