.PHONY: env data train eval report clean

PYTHON ?= python
VENV ?= .venv
BIN := $(VENV)/bin

env:
	$(PYTHON) -m venv $(VENV)
	. $(BIN)/activate && pip install --upgrade pip setuptools wheel
	. $(BIN)/activate && pip install -e .
	. $(BIN)/activate && pip install torch torchvision numpy pandas scikit-learn matplotlib umap-learn hydra-core pyarrow pytest

DATA_DIR := data
TILES_DIR := $(DATA_DIR)/synth_tiles
OMICS_FILE := $(DATA_DIR)/omics.parquet

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

data: $(DATA_DIR)
	. $(BIN)/activate && python -m rejuvbridge.data.synth_tiles --out $(TILES_DIR) --n 800 --img_size 64 --seed 42
	. $(BIN)/activate && python -m rejuvbridge.data.synth_omics --tiles $(TILES_DIR) --out $(OMICS_FILE) --n_genes 200 --seed 42

train:
	. $(BIN)/activate && python -m rejuvbridge.train data=synthetic paths=default train=cpu_fast

eval:
	. $(BIN)/activate && python -m rejuvbridge.eval --tiles $(TILES_DIR) --omics $(OMICS_FILE) --ckpt checkpoints/clip.ckpt --out reports

report:
	. $(BIN)/activate && python - <<'PY'
from pathlib import Path
import json
reports_dir = Path('reports')
metrics_path = reports_dir / 'retrieval.json'
if not metrics_path.exists():
    raise SystemExit('Run `make eval` before `make report`.')
metrics = json.loads(metrics_path.read_text())
report = f"""# RejuvBridge-lite Report

**Retrieval**: R@1={metrics['R@1']:.3f} (95% CI {metrics['R@1_ci'][0]:.3f}-{metrics['R@1_ci'][1]:.3f}), R@5={metrics['R@5']:.3f}

![UMAP](umap.png)

![GradCAM](gradcam.png)

## Model card (lite)
- Intended use: synthetic demo of image–omics alignment.
- Not in scope: clinical claims, real WSI pipelines, spatial transcriptomics.
- Runtime: CPU < 10 min; deterministic seeds.
"""
reports_dir.mkdir(exist_ok=True)
(reports_dir / 'README.md').write_text(report)
PY

clean:
	rm -rf $(VENV) data reports checkpoints
	mkdir -p reports checkpoints
	touch reports/.gitkeep checkpoints/.gitkeep
