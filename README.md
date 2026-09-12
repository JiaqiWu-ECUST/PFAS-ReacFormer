# PFAS-ReacFormer

Official code repository for the study:

> **Multimodal deep learning decodes PFAS transformation in water under data scarcity**

**PFAS-ReacFormer** is a reaction-informed multimodal deep-learning framework for modeling the transformation behavior of per- and polyfluoroalkyl substances (PFAS) in water under data-scarce conditions. The framework combines reaction-center-guided molecular pretraining with condition-aware transfer learning for two downstream tasks:

- PFAS degradation kinetic prediction
- PFAS transformation product generation

## Overview

PFAS transformation datasets are typically small and heterogeneous with respect to molecular structures and experimental conditions. PFAS-ReacFormer addresses this challenge by first learning transferable chemical representations from general reaction data and then adapting the pretrained representation to PFAS-specific downstream tasks.

The computational workflow consists of three components:

1. **Dual Reaction Graph Modeling (DRGM)** for reaction-center-guided pretraining.
2. **PFAS degradation kinetic prediction** using transfer learning and multimodal molecular representations.
3. **PFAS transformation product generation** using molecular structure, physicochemical descriptors, and experimental reaction conditions.

The downstream models can initialize their molecular encoder from the DRGM-pretrained checkpoint.

## Repository structure

```text
PFAS-ReacFormer/
├── DRGM/
│   ├── Config/
│   │   └── pretrain.yaml
│   ├── data.py
│   ├── layers.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── Data/
│   ├── k_prediction/
│   │   ├── k.csv
│   │   ├── k_dataset.pt
│   │   └── splits.pt
│   ├── product_generation/
│   │   ├── product_data_split.json
│   │   ├── product_dataset.csv
│   │   └── product_generation_dataset.pt
│   └── reaction_center/
│       ├── data.csv
│       └── pretrain_data_split.json
│
├── PFAS-ReacFormer-k prediction/
│   ├── Config/
│   │   └── k.yaml
│   ├── data.py
│   ├── layers.py
│   ├── model.py
│   ├── train_compound_split.py
│   ├── train_random_split.py
│   └── train_single.py
│
├── PFAS-ReacFormer-product generation/
│   ├── Config/
│   │   └── pretrain.yaml
│   ├── data.py
│   ├── layers.py
│   ├── model.py
│   └── train.py
│
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/JiaqiWu-ECUST/PFAS-ReacFormer.git
cd PFAS-ReacFormer
```

We recommend using a dedicated Python environment. The code relies on the PyTorch/PyTorch Geometric ecosystem and RDKit. Major Python dependencies used by the repository include:

```text
torch
torch-geometric
rdkit
numpy
pandas
scikit-learn
PyYAML
addict
tensorboard
```

Please install PyTorch and PyTorch Geometric versions compatible with your CUDA environment.

## Data

The repository is organized around three datasets.

### Reaction-center data

`Data/reaction_center/` contains the source table and predefined split information used for reaction-center-guided pretraining:

```text
Zenodo DOI:
pretrain_data_split.json
```

The processed PyTorch data object used by `DRGM/train.py` should be specified through `data.data_path` in `DRGM/Config/pretrain.yaml`.

### PFAS kinetic data

`Data/k_prediction/` contains:

```text
k.csv
k_dataset.pt
splits.pt
```

`k_dataset.pt` is the processed dataset used by the kinetic prediction models, while `splits.pt` contains the predefined split information used by the corresponding evaluation workflow.

### PFAS transformation-product data

`Data/product_generation/` contains:

```text
product_dataset.csv
product_generation_dataset.pt
product_data_split.json
```

The JSON file contains the predefined train/validation/test split used for transformation-product modeling.

### Large files

Large processed data files may be archived separately on Zenodo because of GitHub file-size limitations.

**Zenodo DOI:** `TO BE ADDED`

After downloading the archived data, place the files in the corresponding `Data/` subdirectory and update the paths in the YAML configuration files accordingly.

## Configuration paths

The YAML files currently expose explicit data, output, and pretrained-checkpoint paths. **These paths must be adapted to the local environment before running the code.**

For example, the repository data can be referenced using paths such as:

```yaml
# DRGM/Config/pretrain.yaml
data:
  data_path: Data/reaction_center/<processed_reaction_center_file>.pt
  split_file: Data/reaction_center/pretrain_data_split.json
```

```yaml
# PFAS-ReacFormer-k prediction/Config/k.yaml
data:
  data_path: Data/k_prediction/k_dataset.pt

cv:
  split_path: Data/k_prediction/splits.pt
```

For downstream transfer learning, set the appropriate pretrained checkpoint path:

```yaml
model:
  pretrained_model_path: /path/to/DRGM/checkpoint.pt
```

or, for product generation:

```yaml
model:
  pretrained_path: /path/to/DRGM/checkpoint.pt
```

Output directories (`save_dir`) should likewise be changed to a writable local directory.

## 1. DRGM pretraining

The DRGM module performs reaction-center-guided molecular pretraining.

Configuration:

```text
DRGM/Config/pretrain.yaml
```

Training entry point:

```bash
python DRGM/train.py --config DRGM/Config/pretrain.yaml
```

Before training, ensure that `data.data_path`, `data.split_file`, and `model.save_dir` in the YAML file point to valid local paths.

The resulting pretrained checkpoint can subsequently be supplied to the PFAS downstream models.

## 2. PFAS degradation kinetic prediction

The kinetic module predicts PFAS degradation kinetics and provides random-split and compound-level evaluation workflows.

Configuration:

```text
PFAS-ReacFormer-k prediction/Config/k.yaml
```

Because the directory name contains a space, quote the paths when running from the repository root.

### Single/fixed-fold run

```bash
python "PFAS-ReacFormer-k prediction/train_single.py" \
  --config "PFAS-ReacFormer-k prediction/Config/k.yaml"
```

To use the DRGM-pretrained encoder, set `model.pretrained_model_path` in `k.yaml` to the corresponding checkpoint. Leaving this field empty trains without loading pretrained weights.

The kinetic workflows report regression metrics including R², RMSE, MSE, and MAE and save model/prediction outputs to the configured output directory.

## 3. PFAS transformation product generation

The product-generation module performs condition-aware generation and ranking of PFAS transformation products.

Configuration:

```text
PFAS-ReacFormer-product generation/Config/pretrain.yaml
```

Training:

```bash
python "PFAS-ReacFormer-product generation/train.py" \
  --config "PFAS-ReacFormer-product generation/Config/pretrain.yaml"
```

Before running, update:

- `data.data_path`
- `data.split_file`
- `model.save_dir`
- `model.pretrained_path`

to match the local data and DRGM checkpoint locations.

Two-stage training strategy

Product generation is trained in two stages.

Stage 1 — frozen-encoder training.
The molecular encoder is initialized with the DRGM-pretrained weights and kept frozen.

Stage 2 — encoder fine-tuning.
The best Stage 1 checkpoint (best_top5.pt) is used to initialize Stage 2.

The model uses beam-search decoding during evaluation. The corresponding parameters, including beam size, maximum sequence length, length penalty, temperature, and candidate limits, are defined under `eval` in the YAML configuration.

During training, validation performance is used for checkpoint selection. The training workflow saves a `best_top5.pt` checkpoint and, when available, loads this checkpoint for final testing.

## Model transfer

The intended transfer-learning workflow is:

```text

DRGM reaction-center-guided pretraining
        │
        ▼
Pretrained molecular encoder
        │
        ├──────────────► PFAS degradation kinetic prediction
        │
        └──────────────► PFAS transformation product generation
```

The kinetic and product-generation modules therefore use the DRGM checkpoint as an upstream initialization rather than transferring a downstream task head between the two PFAS tasks.

