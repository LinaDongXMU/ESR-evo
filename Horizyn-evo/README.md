# Horizyn with Evolutionary Augmentation

## Overview

This implementation augments Horizyn protein representations with full-sequence and catalytic-pocket features extracted by MSA Transformer. Evolutionary views are projected to the Horizyn latent space and combined with ProtT5 representations through learned gates.

## Variants

| Variant | Evolutionary input | Build script | Training config | Checkpoint directory |
| --- | --- | --- | --- | --- |
| Full-sequence | `data1/seq2feature.pkl` | `scripts/build_t5_msa_embeddings.py` | `configs/data1_t5_msa_projected_gate.yaml` | `checkpoints/data1_t5_msa_projected_gate` |
| Pocket | `data1/msa_node_feature.pt` | `scripts/build_t5_pocket_msa_embeddings.py` | `configs/data1_t5_pocket_msa_projected_gate.yaml` | `checkpoints/data1_t5_pocket_msa_projected_gate` |
| Combined | Both inputs | `scripts/build_t5_full_pocket_msa_embeddings.py` | `configs/data1_t5_full_pocket_msa_projected_gate.yaml` | `checkpoints/data1_t5_full_pocket_msa_projected_gate` |

## Environment

Horizyn requires Python 3.10 and the dependencies declared in `pyproject.toml`.

```bash
uv sync
uv pip install "transformers<5" sentencepiece
```

The additional packages are required only when missing ProtT5 embeddings must be generated locally.

## Data

Place the following inputs under `data1/`:

```text
train.csv
valid.csv
Enzyme-405.csv
Orphan-335_retrievel_cands.csv
rhea_rxn2uids.csv
seq2feature.pkl
msa_node_feature.pt
```

Prepare Horizyn pair tables and a complete ProtT5 embedding file:

```bash
python scripts/prepare_data1.py \
  --data-dir data1 \
  --output-dir data/data1_horizyn \
  --protein-embeds-source data/sota/prots_t5.h5

python scripts/embed_data1_missing_proteins.py \
  --data-dir data1 \
  --source-h5 data/sota/prots_t5.h5 \
  --output-h5 data/data1_horizyn/prots_t5.h5 \
  --model-name Rostlab/prot_t5_xl_uniref50 \
  --device cuda \
  --batch-size 1 \
  --long-sequence-strategy chunk \
  --overwrite
```

## Feature Preparation

First generate `seq2feature.pkl` and `msa_node_feature.pt` with the root-level `seq2a3m.sh` and `run_msa_feature.py` workflow. Place or link the resulting files at:

```text
data1/seq2feature.pkl
data1/msa_node_feature.pt
```

Build the augmented protein HDF5 file for each evolutionary view:

```bash
# Full-sequence MSA
python scripts/build_t5_msa_embeddings.py \
  --data-dir data1 \
  --source-h5 data/data1_horizyn/prots_t5.h5 \
  --seq2feature data1/seq2feature.pkl \
  --output-h5 data/data1_horizyn/prots_t5_msa.h5 \
  --overwrite

# Pocket MSA
python scripts/build_t5_pocket_msa_embeddings.py \
  --data-dir data1 \
  --source-h5 data/data1_horizyn/prots_t5.h5 \
  --pocket-node-feature data1/msa_node_feature.pt \
  --output-h5 data/data1_horizyn/prots_t5_pocket_msa.h5 \
  --pool mean \
  --overwrite

# Combined MSA
python scripts/build_t5_full_pocket_msa_embeddings.py \
  --data-dir data1 \
  --source-h5 data/data1_horizyn/prots_t5.h5 \
  --seq2feature data1/seq2feature.pkl \
  --pocket-node-feature data1/msa_node_feature.pt \
  --output-h5 data/data1_horizyn/prots_t5_full_pocket_msa.h5 \
  --pocket-pool mean \
  --overwrite
```

The combined representation has the layout `[ProtT5 (1024) | full MSA (768) | mask | pocket MSA (768) | mask]`.

## Training

```bash
# Full-sequence MSA
python train.py --config configs/data1_t5_msa_projected_gate.yaml

# Pocket MSA
python train.py --config configs/data1_t5_pocket_msa_projected_gate.yaml

# Combined MSA
python train.py --config configs/data1_t5_full_pocket_msa_projected_gate.yaml
```

## Evaluation

Score both benchmarks with the checkpoint and configuration of the selected variant. The scorer writes predictions and reports Top-1/3/5/10 success rates.

```bash
# Example: combined MSA on Enzyme-405
python scripts/score_data1_candidates.py \
  --checkpoint checkpoints/data1_t5_full_pocket_msa_projected_gate/last.ckpt \
  --config configs/data1_t5_full_pocket_msa_projected_gate.yaml \
  --input data1/Enzyme-405.csv \
  --output outputs/data1_t5_full_pocket_msa_projected_gate/Enzyme-405_pred.csv \
  --device cuda

# Example: combined MSA on Orphan-335
python scripts/score_data1_candidates.py \
  --checkpoint checkpoints/data1_t5_full_pocket_msa_projected_gate/last.ckpt \
  --config configs/data1_t5_full_pocket_msa_projected_gate.yaml \
  --input data1/Orphan-335_retrievel_cands.csv \
  --output outputs/data1_t5_full_pocket_msa_projected_gate/Orphan-335_pred.csv \
  --device cuda \
  --pos-pair-db-path data1/rhea_rxn2uids.csv \
  --fill-labels-from-pos-pair-db
```

Use the paths in the variant table for the full-sequence and pocket experiments.

## License

See `LICENSE`.
