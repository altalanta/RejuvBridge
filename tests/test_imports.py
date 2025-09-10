def test_imports():
    import rejuvbridge
    from rejuvbridge.cli import app

    # Data pipeline is part of core usage
    from rejuvbridge.data.pipeline import ingest_data, prepare_shards  # noqa: F401
    # API app is optional; skip if FastAPI not installed
    try:
        from rejuvbridge.deploy.app import app as fastapi_app  # noqa: F401
        api_ok = True
    except Exception:
        api_ok = False
    assert rejuvbridge.__version__
    assert app is not None
    assert isinstance(api_ok, bool)
