import mlflow


def register_model(run_id: str, artifact_path: str, name: str) -> str:
    artifact_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(artifact_uri=artifact_uri, name=name)
    print(f"Registered model {name} v{result.version}")
    return result.version

