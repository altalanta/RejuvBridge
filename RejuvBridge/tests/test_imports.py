def test_imports():
    import rejuvbridge
    from rejuvbridge.cli import app
    from rejuvbridge.data.pipeline import ingest_data, prepare_shards
    from rejuvbridge.deploy.app import app as fastapi_app
    assert rejuvbridge.__version__
    assert app is not None and fastapi_app is not None

