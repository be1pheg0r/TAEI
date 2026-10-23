"""Additional data-integrity and real interaction acceptance checks."""

import io
import json
import os
import zipfile
from pathlib import Path

import requests
from playwright.sync_api import expect, sync_playwright

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8091").rstrip("/")


def test_complete_descriptions_and_trace_ingestion():
    payload = requests.get(BASE + "/api/example", timeout=30).json()
    assert payload["summary"]["status"] == "finished"
    assert payload["runs"][0]["primary_status"] == "rejected"
    assert all(r.get("description") for r in payload["records"])
    assert all(f.get("description") and f.get("source") for f in payload["fields"])
    sample = io.BytesIO()
    # Explicitly synthetic parser fixture; never a demonstration of GPU measurement.
    with zipfile.ZipFile(sample, "w") as archive:
        archive.writestr(
            "trace/steps.jsonl", json.dumps({"step_id": 0, "duration_ms": 2.5}) + "\n"
        )
        archive.writestr(
            "trace/requests.csv", "request_id,input_tokens,output_tokens\nr0,10,3\n"
        )
    response = requests.post(
        BASE + "/api/analyze",
        files={"file": ("parser-fixture.zip", sample.getvalue(), "application/zip")},
        timeout=30,
    )
    assert response.ok, response.text
    data = response.json()
    assert len(data["tables"]["steps"]) == 1
    assert len(data["tables"]["requests"]) == 1
    assert data["tables"]["steps"][0]["duration_ms"] == 2.5
    assert data["tables"]["requests"][0]["input_tokens"] == 10


def test_real_chart_interactions_exports_and_responsive_layout(tmp_path):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE, wait_until="networkidle")
        page.get_by_role("button", name="Графики", exact=False).first.click()
        page.wait_for_selector("[data-chart-id]")
        audit = page.evaluate("window.__chartAudit()")
        assert len({c["id"] for c in audit}) == len(audit)
        available = [c for c in audit if c["available"] and c["points"]]
        assert len(available) >= 30, audit
        options = page.evaluate("""() =>
            [...document.querySelectorAll('[_echarts_instance_]')].map(el => {
            const chart = window.echarts.getInstanceByDom(el);
            const option = chart.getOption();
            return {
                series: option.series,
                tooltip: option.tooltip,
                size: [el.clientWidth, el.clientHeight]
            };
        })""")
        assert len(options) >= 30
        assert all(
            o["series"] and o["tooltip"] and min(o["size"]) > 100 for o in options
        )
        assert all(o["tooltip"][0].get("renderMode") == "richText" for o in options)
        panel = page.locator("[data-chart-id]").first
        chart = panel.locator("[_echarts_instance_]")
        box = chart.bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.6)
        for extension in ("CSV", "PNG"):
            with page.expect_download() as download:
                panel.get_by_role("button", name=extension, exact=True).click()
            file = download.value
            file.save_as(tmp_path / file.suggested_filename)
            assert Path(file.path()).stat().st_size > 10
        toggle = panel.get_by_title("Таблица данных")
        if toggle.count():
            toggle.click()
            expect(panel.locator(".data-view")).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(500)
        overflow = page.evaluate(
            "document.documentElement.scrollWidth > innerWidth + 2"
        )
        assert not overflow, "whole page overflows horizontally on mobile"
        page.get_by_role("button", name="Все значения", exact=True).click()
        page.locator("tbody tr").first.click()
        expect(page.locator("tr.detail pre")).to_be_visible()
        assert not errors, errors
        browser.close()
