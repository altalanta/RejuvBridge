# PR: Refactor imports, optional extras, logging, tests, and CI

Branch name: `refactor/lazy-imports-optional-extras-logging-ci`

## Summary
- Make CLI and heavy modules import-light via lazy imports.
- Move heavyweight dependencies to optional extras (api, data, ui, mlops, imaging, training).
- Add structured logging helper and replace prints in core paths.
- Add schema and optional API tests; keep CI lean with a Python matrix.

## Motivation
- Reduce friction for users/tests who only need core functionality.
- Improve install time and reliability by installing only what’s needed.
- Strengthen platform hygiene (logging, tests, CI) without overreach.

## Changes (by file)

- `src/rejuvbridge/cli.py`
  - Lazy imports inside commands: `ingest`, `prepare`, `train`, `api`, `ui`, `demo`.
- `src/rejuvbridge/data/pipeline.py`
  - Validate inputs; add logging; import `pyarrow` inside functions; clearer error messages with install hint.
- `src/rejuvbridge/training/runner.py`
  - Lazy import `mlflow`; add logging; no-MLflow fallback path still writes `TRAINING_PLANNED.txt`.
- `src/rejuvbridge/utils/log.py`
  - New logging utility (`get_logger`) honoring `REJUV_LOG_LEVEL`.
- `tests/test_imports.py`
  - Treat API import as optional; keep core import guarantees.
- `tests/test_schema.py`
  - New: verify Parquet schemas for manifest/tiles; skips if `pyarrow` missing.
- `tests/test_api_health.py`
  - New: FastAPI health check via `TestClient`; skipped without `api` extras.
- `.github/workflows/ci.yml`
  - Python 3.9/3.10/3.11 matrix, pip cache, installs `.[dev,data]` only.
- `pyproject.toml`
  - Base deps trimmed to light core; new extras:
    - `data`: pyarrow
    - `api`: fastapi, uvicorn, pydantic
    - `mlops`: mlflow
    - `imaging`: torch, torchvision, einops
    - `training`: deepspeed, ray[default], tensorboard, scikit-learn
    - `ui`: gradio

## Install changes
- Core development: `pip install -e ".[dev]"`
- With data pipeline (Parquet): `pip install -e ".[dev,data]"`
- Add features as needed: `.[api]`, `.[ui]`, `.[mlops]`, `.[imaging]`, `.[training]`

## Backward compatibility
- CLI command names and behavior unchanged.
- Users relying on implicit installation of FastAPI/MLflow/pyarrow/torch must now include the relevant extras.
- Logging default level INFO; override with `REJUV_LOG_LEVEL`.

## Testing & validation
- Unit tests pass on lean install (`.[dev,data]`).
- Optional tests skipped when extras are not present.
- CI runs ruff + pytest across Python 3.9–3.11 with pip caching.

## Review checklist
- [ ] Confirm CLI help works without extras: `rejuvbridge --help`
- [ ] Run demo flow: `rejuvbridge demo` (writes sample tile + Parquet)
- [ ] Run tests locally: `pip install -e ".[dev,data]" && pytest -q`
- [ ] (Optional) API test: `pip install -e ".[api]" && pytest -q tests/test_api_health.py`
- [ ] Verify logging messages appear at INFO and honor `REJUV_LOG_LEVEL=DEBUG`

## Notes & follow-ups
- Consider schema validation via `pandera`/pydantic for manifests.
- Add a tiny training loop to log a scalar metric for CI asserts.
- README: add architecture diagram and a version command.

