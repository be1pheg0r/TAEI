# Log explorer black-box acceptance suite

Run after starting the service at `BASE_URL` (default `http://127.0.0.1:8091`):

```bash
cd /root/TAEI/tools/log-explorer/qa
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
BASE_URL=http://127.0.0.1:8091 pytest -v
```

Override `ARCHIVE=/path/to/logs.zip` and `EVIDENCE_DIR=/path/to/output` if needed. Screenshots are saved under `qa/evidence/`. Tests are independent of backend/frontend source code. The primary suite uses the actual provided archive; `test_integrity.py` additionally sends a clearly synthetic, two-row parser fixture to verify JSONL/CSV ingestion. This fixture is not measurement evidence or demonstration data. API upload uses multipart field `file`.

## Fixture facts (read directly from `logs-2.zip`)

- ZIP contains `hydra.log` (0 bytes) and `run.log` (4,329 bytes); 51 log lines.
- Experiment: `nano_vllm_dataset_rtx_a4000_tp2`; run `R15`.
- One actual generation completed, but primary status is `rejected`; accepted primary count is 0. QC cites pending/unverified overhead calibration and absent `primary_status accepted` in `runs.json`.
- Logged table counts: 32 requests, 2,048 steps, 16,384 step_requests, 16,448 cache_events, 2,112 scheduler_events, 16,384 token_events, 44 telemetry.
- Per-step GPU duration and per-generated-token GPU timestamps are explicitly unsupported in the fixture; acceptance should not interpret missing data as zero.

The test checks actual example and analyzed payloads, 30+ instantiated ECharts with nonempty series and no NaN, navigation/screenshots at desktop/mobile sizes, search, empty `hydra.log`, actual archive upload, and visible recovery after invalid ZIP. DOM labels/inputs are discovered with loose roles/text/placeholder matching rather than app-specific CSS selectors. If the implementation uses a different visible error wording or non-placeholder search label, loosen that assertion only after checking captured UI evidence.

`test_integrity.py` additionally requires the final experiment status, descriptions on every record and field, actual trace table extraction and numeric CSV values. It exercises actual CSV and PNG downloads, chart data view, chart audit IDs, ECharts options, and absence of whole-page horizontal overflow on mobile. Non-finite values are checked by numeric type rather than substring matching (`nano-vllm` is not a NaN).

Run the full suite against the public TLS endpoint before reporting acceptance:

```bash
BASE_URL=https://logs.87.120.84.200.sslip.io pytest -v --tb=short
```

Do not change an assertion merely to make an incomplete implementation pass. If selectors change, inspect the rendered UI and preserve the underlying acceptance condition.
