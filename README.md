# RejuvBridge-lite

Synthetic cross-modal alignment demo that pairs toy histology tiles with gene expression vectors and trains a CLIP-style model for image↔omics retrieval.

> **Capabilities**: deterministic data generation, ResNet18 + MLP encoders, CLIP InfoNCE training, retrieval@K with bootstrap CIs, UMAP and Grad-CAM visual diagnostics.
>
>  **Not in scope**: clinical claims, whole-slide imaging pipelines, patient data, spatial transcriptomics, or deployment guidance.

## Quickstart

```bash
make env      # create virtualenv and install deps (CPU only)
make data     # generate synthetic tiles + omics
make train    # fit ResNet18 / MLP using InfoNCE (≤5 epochs)
make eval     # retrieval metrics + UMAP + Grad-CAM
make report   # assemble markdown report summarising outputs
```

All targets run on a laptop CPU in under 10 minutes. Outputs are reproducible thanks to fixed seeds.

## Project layout

```
rejuvbridge/
├── configs/            # Hydra configs for data/train/paths
├── src/rejuvbridge/    # package with data, models, train/eval/infer utilities
├── data/               # (generated) synthetic tiles + omics parquet
├── checkpoints/        # CLIP checkpoint + metrics (post-train)
├── reports/            # retrieval.json, umap.png, gradcam.png, README.md
└── tests/              # pytest smoke tests (shapes, determinism, retrieval)
```

## CI and testing

Run the test suite locally or rely on the included GitHub Actions workflow:

```bash
pytest -q
```

The tests cover embedding shapes/L2 normalisation, deterministic training with fixed seeds, and retrieval quality beating random baselines.

## Artifacts

Pre-generated artifacts are checked into `reports/` so reviewers can inspect outputs without running the pipeline (if `umap-learn` is unavailable at runtime the pipeline automatically falls back to PCA for the joint embedding figure):

- `reports/retrieval.json` – Retrieval@K metrics with bootstrap CIs.
- `reports/umap.png` – UMAP projection of joint embeddings (images + omics).
- `reports/gradcam.png` – Grad-CAM overlays highlighting salient tile regions.
- `reports/README.md` – Short model card created by `make report`.

## License

MIT — see [LICENSE](LICENSE).

