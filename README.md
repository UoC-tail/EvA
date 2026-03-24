# EVA: Evolutionary Attacks on Graphs

Official implementation of **"EVA: Evolutionary Attacks on Graphs"**, accepted at **ICLR 2026**.

**Authors:** Sadegh Akhondzadeh\*, Soroush H. Zargarbashi\*, Jimin Cao, Aleksandar Bojchevski

## Overview

EVA is a black-box adversarial attack framework for graph neural networks (GNNs) based on evolutionary algorithms. It perturbs graph structure (edges) to degrade GNN performance without requiring access to model gradients. The key variant, **EVA Accelerated**, uses batched population evaluation and efficient genetic operators to scale to large graphs.


### Installation with `uv`

```bash
# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric and extensions
uv pip install torch_geometric
uv pip install torch_sparse torch_scatter torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Install dependencies
uv pip install seaborn ml-collections==0.1.1 tinydb cvxpy evotorch sacred \
    ogb wandb gmpy2 statsmodels sympy numba "setuptools<81"
uv pip install "sparse_smoothing @ git+https://github.com/abojchevski/sparse_smoothing.git"

# Install local packages
cd GNNSetup && uv pip install -e . && cd ..
cd Eva && uv pip install -e . && cd ..
```

### Configuration

Edit `conf/general-config.yaml` to set paths for your environment:

```yaml
dataset_root: ./data/dataset
splits_root:  ./data/splits
models_root:  ./data/models
results_root: ./data/logs
reports_root: ./data/logs
```

## Usage

All experiments use the [Sacred](https://github.com/IDSIA/sacred) framework for configuration management.

### 1. Create Data Splits

```bash
python3 bin/make_splits.py with \
    dataset_name=cora_ml n_runs=5 \
    training_split_type=non-stratified \
    validation_split_type=non-stratified \
    test_split_type=non-stratified
```

### 2. Train a GNN Model

```bash
python3 bin/vanilla_train.py with \
    dataset_name=cora_ml model_name=GCN n_runs=5 \
    inductive=True wandb_flag=False device="cuda:0"
```

### 3. Run EVA Attack

```bash
python3 bin/make_attack.py with \
    dataset_name=cora_ml model_name=GCN \
    attack_name=EvAttackAccelerated \
    n_runs=5 inductive=True epsilon=0.1 \
    wandb_flag=False device="cuda:0"
```

### Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `dataset_name` | Dataset: `cora_ml`, `citeseer`, `pubmed`, `amazon_computers`, `amazon_photo` | `cora_ml` |
| `model_name` | GNN model: `GCN`, `GAT`, `GPRGNN`, `APPNP`, `ChebNetII`, etc. | `GCN` |
| `attack_name` | Attack method (see below) | `EvAttackAccelerated` |
| `inductive` | Inductive (`True`) or transductive (`False`) setting | `True` |
| `epsilon` | Perturbation budget as fraction of feasible edges | `0.1` |
| `n_runs` | Number of random splits to evaluate | `5` |
| `overwrite_n_edges` | Override attack budget with a fixed edge count | `None` |
| `device` | GPU device | `cuda:0` |
| `wandb_flag` | Enable Weights & Biases logging | `True` |

### Attacking Robust Models

```bash
python3 bin/make_attack.py with \
    dataset_name=cora_ml model_name=GCN \
    attack_name=EvAttackAccelerated epsilon=0.1 \
    robust_model_flag=True self_training=True robust_training=True \
    inductive=True device="cuda:0"
```
## Citation

```bibtex
@article{akhondzadeh2025eva,
  title={EvA: Evolutionary Attacks on Graphs},
  author={Akhondzadeh, Mohammad Sadegh and Zargarbashi, Soroush H and Cao, Jimin and Bojchevski, Aleksandar},
  journal={arXiv preprint arXiv:2507.08212},
  year={2025}
}
```

## License

GNU License
