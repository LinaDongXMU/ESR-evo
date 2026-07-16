# EnzymeCAGE with Evolutionary Augmentation

## Overview

This implementation augments EnzymeCAGE with full-sequence and catalytic-pocket representations extracted by MSA Transformer. Evolutionary features are projected to the ESM-C embedding dimension and fused through a learned gate before reaction-conditioned scoring.

## Variants

| Variant | Full-sequence MSA | Pocket MSA | Training config | Checkpoint directory |
| --- | --- | --- | --- | --- |
| Full-sequence | Yes | No | `config/train/pretrain/seed_42_msa_mean_only.yaml` | `checkpoints/pretrain/seed_42_msa_mean_only` |
| Pocket | No | Yes | `config/train/pretrain/seed_42_msa_node_only.yaml` | `checkpoints/pretrain/seed_42_msa_node_only` |
| Combined | Yes | Yes | `config/train/pretrain/seed_42.yaml` | `checkpoints/pretrain/seed_42` |

Each variant has matching Enzyme-405 and Orphan-335 inference configurations under `config/infer/`.

## Environment

```bash
conda create -n enzymecage-evo python=3.10
conda activate enzymecage-evo
bash setup_env.sh
```

The PyTorch build in `setup_env.sh` targets CUDA 12.1 and may be adjusted for the local CUDA installation.

## Data

Place the training, benchmark, structure, reaction, ESM-C, and MSA files under `dataset/` using the paths declared in the YAML configurations. The evolutionary inputs are:

```text
dataset/RHEA/2025-02-05/feature/protein/MSA_Transformer/
|-- protein_level/seq2feature.pkl
`-- pocket_node_feature/msa_node_feature.pt
```

`seq2feature.pkl` maps protein sequences to 768-dimensional full-sequence features. `msa_node_feature.pt` maps UniProt identifiers to aligned 768-dimensional pocket-residue features.

## Feature Preparation

Generate the shared MSA features from the repository root. Place one protein in each `fasta1/<UniprotID>.fasta` file and configure `UNIREF_DB`, `THREADS`, `BATCH`, and `USE_GPU` at the top of `seq2a3m.sh`.

```bash
cd ..

bash seq2a3m.sh

python run_msa_feature.py \
  --input-csv EnzymeCAGE-evo/dataset/RHEA/2025-02-05/all_enzymes.csv \
  --msa-dir msa1 \
  --output-dir EnzymeCAGE-evo/dataset/RHEA/2025-02-05/feature/protein/MSA_Transformer \
  --pocket-info EnzymeCAGE-evo/dataset/RHEA/2025-02-05/pocket_info.csv \
  --device cuda
```

The complete CLI and output specification are documented in the root `README.md`.

## Training

```bash
# Full-sequence MSA
python train.py --config config/train/pretrain/seed_42_msa_mean_only.yaml

# Pocket MSA
python train.py --config config/train/pretrain/seed_42_msa_node_only.yaml

# Combined MSA
python train.py --config config/train/pretrain/seed_42.yaml
```

## Evaluation

Run inference with the configuration matching the trained variant, then evaluate the generated prediction table.

```bash
# Example: combined MSA on Enzyme-405
python infer.py --config config/infer/Enzyme-405.yaml
python evaluate.py \
  --result_path checkpoints/pretrain/seed_42/Enzyme-405_best_model.csv

# Example: combined MSA on Orphan-335
python infer.py --config config/infer/Orphan-335.yaml
python evaluate.py \
  --result_path checkpoints/pretrain/seed_42/Orphan-335_retrievel_cands_best_model.csv \
  --pos_pair_db_path dataset/RHEA/2025-02-05/rhea_rxn2uids.csv
```

For the single-view variants, use the `_msa_mean_only` or `_msa_node_only` inference config and its corresponding checkpoint directory. Evaluation reports Top-1/3/5/10 success rate, Top-10 DCG, and enrichment factors.

## License

See `LICENSE`.
