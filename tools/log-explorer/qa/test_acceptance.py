"""Black-box acceptance tests for the log explorer; no backend/frontend imports."""

import math
import os
import re
import zipfile
from pathlib import Path

import requests
from playwright.sync_api import expect, sync_playwright

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8091").rstrip("/")
ARCHIVE = Path(os.getenv("ARCHIVE", "/root/.hermes/attachments/logs-2.zip"))
EVIDENCE = Path(os.getenv("EVIDENCE_DIR", Path(__file__).parent / "evidence"))
TABS = ["Обзор", "Графики", "Журнал", "Все значения", "Файлы"]


def archive_facts():
    with zipfile.ZipFile(ARCHIVE) as z:
        names = z.namelist()
        text = z.read("run.log").decode("utf-8", "replace")
    counts = {}
    for key in (
        "requests",
        "steps",
        "step_requests",
        "cache_events",
        "scheduler_events",
        "token_events",
        "telemetry",
    ):
        m = re.search(r"'" + re.escape(key) + r"': (\d+)", text)
        if m:
            counts[key] = int(m.group(1))
    return {
        "names": names,
        "hydra_size": zinfo_size(ARCHIVE, "hydra.log"),
        "experiment": "nano_vllm_dataset_rtx_a4000_tp2",
        "run_id": "R15",
        "primary_status": "rejected",
        "accepted_primary_count": 0,
        "execution_completed": True,
        "qc_reasons": [
            "overhead calibration is pending or unverified",
            "runs.json does not declare primary_status accepted",
        ],
        "counts": counts,
    }


def zinfo_size(path, name):
    with zipfile.ZipFile(path) as z:
        return z.getinfo(name).file_size


def api_json(path):
    response = requests.get(BASE_URL + path, timeout=20)
    assert response.ok, (
        f"GET {path}: HTTP {response.status_code}: {response.text[:500]}"
    )
    return response.json()


def payload_from(response):
    # Tolerate standard response wrappers without hiding schema errors.
    for key in ("data", "result", "analysis"):
        if isinstance(response, dict) and isinstance(response.get(key), dict):
            response = response[key]
    return response


def assert_real_payload(payload):
    assert isinstance(payload, dict)
    assert isinstance(payload.get("archive"), dict)
    assert isinstance(payload.get("metrics"), dict)
    assert isinstance(payload.get("runs"), list)
    assert isinstance(payload.get("records"), list)
    assert isinstance(payload.get("fields"), list)
    assert isinstance(payload.get("tables"), dict)
    assert isinstance(payload.get("summary"), dict)
    for k in (
        "steps",
        "requests",
        "step_requests",
        "cache_events",
        "scheduler_events",
        "token_events",
        "telemetry",
    ):
        assert isinstance(payload["tables"].get(k), list), f"missing table {k}"
    assert payload["summary"].get("experiment") == archive_facts()["experiment"]
    assert payload["summary"].get("accepted_count") == 0
    assert payload["runs"][0]["primary_status"] == "rejected"
    assert payload["runs"][0]["real_generate_completed"] is True
    assert payload["runs"][0]["qc"]["counts"] == archive_facts()["counts"]
    assert_finite(payload)


def assert_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite numeric value: {value}"
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite(item)


def test_api_health_example_and_upload_contract():
    health = api_json("/api/health")
    assert isinstance(health, dict)
    example = payload_from(api_json("/api/example"))
    assert_real_payload(example)
    with ARCHIVE.open("rb") as f:
        response = requests.post(
            BASE_URL + "/api/analyze",
            files={"file": (ARCHIVE.name, f, "application/zip")},
            timeout=120,
        )
    assert response.ok, (
        f"POST /api/analyze: HTTP {response.status_code}: {response.text[:500]}"
    )
    assert_real_payload(payload_from(response.json()))


def find_tab(page, label):
    candidates = page.get_by_role("tab", name=re.compile(re.escape(label), re.I))
    if candidates.count():
        return candidates.first
    candidates = page.get_by_role("button", name=re.compile(re.escape(label), re.I))
    if candidates.count():
        return candidates.first
    return page.get_by_text(re.compile(re.escape(label), re.I)).first


def click_tab(page, label):
    target = find_tab(page, label)
    expect(target).to_be_visible(timeout=12000)
    target.click()
    page.wait_for_timeout(450)


def charts_with_data(page):
    return page.evaluate("""() => {
      const els=[...document.querySelectorAll('[_echarts_instance_]')];
      return els.map(el => {
        const chart = window.echarts?.getInstanceByDom(el);
        const option = chart?.getOption?.();
        return {
          rect: el.getBoundingClientRect().toJSON(),
          option: !!option,
          series: option?.series || [],
          html: el.innerHTML
        };
      });
    }""")


def test_ui_real_example_navigation_search_upload_and_screenshots():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    archive_facts()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for viewport, tag in [
            ({"width": 1440, "height": 1000}, "desktop"),
            ({"width": 390, "height": 844}, "mobile"),
        ]:
            context = browser.new_context(viewport=viewport)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
            expect(
                page.get_by_text(
                    re.compile("nano_vllm_dataset_rtx_a4000_tp2", re.I)
                ).first
            ).to_be_visible(timeout=30000)
            for label in TABS:
                click_tab(page, label)
                page.screenshot(
                    path=str(EVIDENCE / f"{tag}-{label.replace(' ', '-')}.png"),
                    full_page=True,
                )
            click_tab(page, "Графики")
            page.wait_for_timeout(1000)
            charts = charts_with_data(page)
            assert len(charts) >= 30, (
                f"Expected >=30 ECharts DOM instances, found {len(charts)}"
            )
            for idx, chart in enumerate(charts):
                assert chart["option"], f"chart {idx} has no ECharts option"
                assert chart["series"], f"chart {idx} has no series data"
                assert_finite(chart["series"])
            click_tab(page, "Все значения")
            field_inputs = page.locator("input").all()
            assert field_inputs, "no search/input control found on values view"
            # Use real record/data keys as search probes, avoiding selector assumptions.
            field_paths = page.locator("body").inner_text()
            searchable = page.get_by_placeholder(
                re.compile("поиск|search|фильтр|filter", re.I)
            ).first
            if searchable.count():
                key = next(
                    (
                        k
                        for k in (
                            "tensor_parallel_size",
                            "primary_status",
                            "experiment",
                            "run_id",
                        )
                        if k in field_paths
                    ),
                    "experiment",
                )
                searchable.fill(key)
                expect(page.get_by_text(re.compile(key, re.I)).first).to_be_visible(
                    timeout=5000
                )
            click_tab(page, "Журнал")
            search = page.get_by_placeholder(
                re.compile("поиск|search|фильтр|filter", re.I)
            ).first
            if search.count():
                search.fill("R15")
                expect(page.get_by_text(re.compile("R15", re.I)).first).to_be_visible(
                    timeout=5000
                )
            click_tab(page, "Файлы")
            expect(
                page.get_by_text(re.compile("hydra\\.log", re.I)).first
            ).to_be_visible(timeout=8000)
            file_text = page.locator("body").inner_text()
            assert "0 B" in file_text or re.search(
                r"hydra\.log.{0,80}(0\s*(B|bytes|байт))", file_text, re.I | re.S
            ), "hydra.log is not shown as empty (0 bytes)"
            upload = page.locator('input[type="file"]')
            assert upload.count(), "file upload input absent"
            upload.first.set_input_files(str(ARCHIVE))
            expect(
                page.get_by_text(
                    re.compile("nano_vllm_dataset_rtx_a4000_tp2", re.I)
                ).first
            ).to_be_visible(timeout=30000)
            # Invalid archive must report an error while preserving a usable page.
            bad = EVIDENCE / "invalid-upload.zip"
            bad.write_bytes(b"not a zip")
            upload.first.set_input_files(str(bad))
            page.wait_for_timeout(800)
            assert page.locator("body").inner_text().strip(), (
                "page became blank after invalid ZIP"
            )
            assert page.get_by_text(
                re.compile("error|invalid|ошиб|не удалось|zip", re.I)
            ).count(), "invalid ZIP produced no visible recoverable error"
            assert not errors, f"uncaught page errors: {errors}"
            context.close()
        browser.close()
