RejuvBridge — Multimodal Imaging↔Omics ML (Scalable)

Short
- Open, production-grade repo to ingest EM/LM pathology tiles, align with spatial/scRNA-seq, and train distributed vision+omics models for cell-state mapping and protein-structure signal discovery. Ships with reproducible data pipelines, exascale-ready training, model registry, and an interpretable web UI for wet-lab teams.

Highlights
- Imaging models: 2D/3D U-Nets, ViT backbones; 3D CNN for Cryo-EM classification.
- Cross-modal mapping: contrastive/CCA-inspired encoders for histology↔omics alignment; zero-shot transfer.
- Data pipeline: Ingest → QC → tiling → WebDataset/Zarr sharding; Parquet metadata and data contracts.
- Distributed training: PyTorch + FSDP/DeepSpeed (ZeRO-3), NCCL multi-node; AMP + checkpointing; Ray Train/K8s ready.
- MLOps: MLflow registry + model cards; Hydra configs; CI/CD (lint, unit/integration, schema checks).
- Interpretability: Grad-CAM/IG; feature probes vs known markers; uncertainty quantification.
- Deployment: FastAPI service + Gradio UI (tile viewer, overlays, cross-plots); batch & streaming.

Install (dev)
- pip: `pip install -e ".[dev]"`
- pipx CLI: `pipx install .` then `rejuvbridge --help`

Quickstart
1) Validate environment: `rejuvbridge env`
2) Ingest sample data (stubs): `rejuvbridge ingest --input ./samples --out ./data/raw`
3) Tile and shard: `rejuvbridge prepare --raw ./data/raw --out ./data/shards`
4) Train (single-node demo): `rejuvbridge train --config conf/train_demo.yaml`
5) Launch UI: `rejuvbridge ui --host 0.0.0.0 --port 7860`

Demo (one command)
- `rejuvbridge demo` creates a tiny example tile, then ingests and prepares shards under `./data/`.

Data Sources (public, suggested)
- CAMELYON16, Human Protein Atlas images, EMPIAR Cryo-EM maps, a spatial transcriptomics/CITE-seq dataset.

Repo Structure
- `src/rejuvbridge/` — package modules (data, imaging, crossmodal, training, mlops, interpretability, deploy, cli, utils)
- `conf/` — Hydra-style YAML configs
- `api/` — FastAPI service
- `ui/` — Gradio app
- `tests/` — unit tests
- `.github/workflows/ci.yml` — lint + tests

License
- MIT (see LICENSE)
