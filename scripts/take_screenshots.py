"""Capture README screenshots of the running dashboard with headless Chromium.

    streamlit run app.py &
    python scripts/take_screenshots.py --url http://localhost:8501
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
WIDTH = 1440


async def shoot(page, name: str, settle_ms: int = 1500) -> None:
    await page.wait_for_timeout(settle_ms)
    bottom = await page.evaluate(
        "() => { const el = document.querySelector('[data-testid=\"stMainBlockContainer\"]');"
        " return el ? el.getBoundingClientRect().bottom + window.scrollY : document.body.scrollHeight; }"
    )
    await page.set_viewport_size({"width": WIDTH, "height": int(min(max(bottom + 24, 700), 3600))})
    await page.wait_for_timeout(1500)
    await page.screenshot(path=OUT / name)


async def main(url: str, executable: str | None) -> None:
    from playwright.async_api import async_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**({"executable_path": executable} if executable else {}))
        page = await browser.new_page(viewport={"width": WIDTH, "height": 1600}, device_scale_factor=2)
        await page.goto(url, wait_until="networkidle")
        await page.get_by_text("Sales Performance Dashboard").wait_for(timeout=90_000)
        await page.mouse.move(WIDTH - 10, 10)
        await shoot(page, "overview.png", 4000)
        await page.get_by_role("tab", name="✨ Ask your data").click()
        await page.get_by_text("Try an example").wait_for()
        await page.get_by_role("button", name="Where did resellers lose money on bikes in fiscal year 2020?").click()
        await page.get_by_text("In FY2020, resellers lost").wait_for(timeout=30_000)
        await page.get_by_text("SQL and agent steps").first.click()
        await page.mouse.move(WIDTH - 10, 10)
        await shoot(page, "ask-your-data.png", 2500)
        await page.set_viewport_size({"width": WIDTH, "height": 1600})
        for tab, name, wait in [("📦 Products", "products.png", 2500), ("👥 Customers", "customers.png", 9000),
                                ("🔮 Forecast", "forecast.png", 7000)]:
            await page.get_by_role("tab", name=tab).click()
            await page.mouse.move(WIDTH - 10, 10)
            await shoot(page, name, wait)
        await browser.close()
    print(f"saved to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--executable", help="path to a Chromium binary (optional)")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.executable))
