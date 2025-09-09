from pathlib import Path
from omegaconf import OmegaConf
import mlflow


def train_from_config(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    out = Path(cfg.get("outputs", "outputs"))
    out.mkdir(parents=True, exist_ok=True)

    # Stubs: log config to MLflow and print planned strategy
    mlflow.set_tracking_uri(cfg.get("mlflow_uri", "file:mlruns"))
    mlflow.set_experiment(cfg.get("experiment", "rejuvbridge-demo"))
    with mlflow.start_run(run_name=cfg.get("run_name", "demo")):
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
        strategy = cfg.get("distributed", {}).get("strategy", "fsdp")
        print(f"[train] Strategy={strategy} AMP={cfg.get('amp', True)} ckpt={cfg.get('checkpointing', True)}")
        # Here you would: build datasets (WebDataset), models, and launch FSDP/DeepSpeed.
        # This stub only demonstrates config-driven training orchestration.
        (out / "TRAINING_PLANNED.txt").write_text(f"Planned training with strategy {strategy}\n")
        mlflow.log_artifact(out / "TRAINING_PLANNED.txt")

