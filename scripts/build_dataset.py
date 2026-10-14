"""Rebuild data/sales.parquet from Microsoft's AdventureWorks Sales workbook.

    python scripts/build_dataset.py                 # downloads the workbook (~14 MB)
    python scripts/build_dataset.py --source path/to/AdventureWorks Sales.xlsx
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from salespulse.data import DEFAULT_PARQUET, build_dataset, save_parquet  # noqa: E402

SOURCE_URL = ("https://raw.githubusercontent.com/microsoft/powerbi-desktop-samples/main/"
              "AdventureWorks%20Sales%20Sample/AdventureWorks%20Sales.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="local .xlsx (default: download from GitHub)")
    parser.add_argument("--out", default=str(DEFAULT_PARQUET))
    args = parser.parse_args()

    source = args.source
    if not source:
        source = str(Path(tempfile.gettempdir()) / "AdventureWorks Sales.xlsx")
        print(f"Downloading {SOURCE_URL} ...")
        urllib.request.urlretrieve(SOURCE_URL, source)

    started = time.time()
    df = build_dataset(source)
    path = save_parquet(df, args.out)
    print(f"{len(df):,} order lines, {df['order_date'].min():%Y-%m-%d} to {df['order_date'].max():%Y-%m-%d}")
    print(f"revenue ${df['sales'].sum():,.0f} · profit ${df['profit'].sum():,.0f}")
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
