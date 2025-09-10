import pytest


def test_api_health_optional():
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    try:
        from rejuvbridge.deploy.app import app
        from fastapi.testclient import TestClient
    except Exception as e:
        pytest.skip(f"FastAPI test client not available: {e}")

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"

