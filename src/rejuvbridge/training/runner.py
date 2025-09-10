from pathlib import Path

from omegaconf import OmegaConf

from ..utils.log import get_logger

logger = get_logger(__name__)


def train_from_config(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    out = Path(cfg.get("outputs", "outputs"))
    out.mkdir(parents=True, exist_ok=True)

    # Stubs: log config to MLflow and print planned strategy
    try:
        import mlflow  # lazy import; optional dependency
    except Exception:
        mlflow = None

    if mlflow is not None:
        mlflow.set_tracking_uri(cfg.get("mlflow_uri", "file:mlruns"))
        mlflow.set_experiment(cfg.get("experiment", "rejuvbridge-demo"))
        with mlflow.start_run(run_name=cfg.get("run_name", "demo")):
            mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
            strategy = cfg.get("distributed", {}).get("strategy", "fsdp")
            logger.info(
                "[train] Strategy=%s AMP=%s ckpt=%s",
                strategy,
                cfg.get("amp", True),
                cfg.get("checkpointing", True),
            )
            (out / "TRAINING_PLANNED.txt").write_text(
                f"Planned training with strategy {strategy}\n"
            )
            mlflow.log_artifact(out / "TRAINING_PLANNED.txt")
    else:
        strategy = cfg.get("distributed", {}).get("strategy", "fsdp")
        logger.info(
            "[train] (no-mlflow) Strategy=%s AMP=%s ckpt=%s",
            strategy,
            cfg.get("amp", True),
            cfg.get("checkpointing", True),
        )
        (out / "TRAINING_PLANNED.txt").write_text(f"Planned training with strategy {strategy}\n")
