# ESR-evo Data and Pretrained Checkpoints

## Description

This Zenodo record provides the evolutionary features, pretrained checkpoints, and Horizyn-formatted benchmark data required to reproduce the ESR-evo experiments. ESR-evo augments reaction-conditioned enzyme recommendation models with full-sequence and catalytic-pocket representations derived from multiple sequence alignments (MSAs).

The record contains only artifacts introduced or reformatted for ESR-evo. Unchanged backbone data, non-evolutionary features, and other resources distributed with EnzymeCAGE and Horizyn are not duplicated. They should be obtained from the original repositories and data releases following the instructions in the ESR-evo GitHub repository.

## Archive Inventory

### `evo-feature.tar.gz`

This archive contains the two MSA Transformer feature files shared by the EnzymeCAGE and Horizyn experiments:

```text
MSA_Transformer/
|-- protein_level/seq2feature.pkl
`-- pocket_node_feature/msa_node_feature.pt
```

- `seq2feature.pkl` maps full protein sequences to 768-dimensional mean-pooled MSA Transformer representations.
- `msa_node_feature.pt` maps protein identifiers to residue-level, 768-dimensional MSA Transformer representations for catalytic-pocket residues.

For EnzymeCAGE, place the extracted `MSA_Transformer` directory at:

```text
EnzymeCAGE-evo/dataset/RHEA/2025-02-05/feature/protein/MSA_Transformer/
```

For Horizyn, place or link the two files at:

```text
Horizyn-evo/data1/seq2feature.pkl
Horizyn-evo/data1/msa_node_feature.pt
```

### `checkpoint_EnzymeCage-evo.tar.gz`

This archive contains the three pretrained EnzymeCAGE checkpoints:

| Directory | Evolutionary input |
| --- | --- |
| `seed_42_msa_mean_only` | Full-sequence MSA features |
| `seed_42_msa_node_only` | Pocket-level MSA features |
| `seed_42` | Full-sequence and pocket-level MSA features |

Extract the checkpoint directories under:

```text
EnzymeCAGE-evo/checkpoints/pretrain/
```

### `checkpoint_Horizyn-evo.tar.gz`

This archive contains the three pretrained Horizyn checkpoints:

| Directory | Evolutionary input |
| --- | --- |
| `data1_t5_msa_projected_gate` | Full-sequence MSA features |
| `data1_t5_pocket_msa_projected_gate` | Pocket-level MSA features |
| `data1_t5_full_pocket_msa_projected_gate` | Full-sequence and pocket-level MSA features |

Extract the checkpoint directories under:

```text
Horizyn-evo/checkpoints/
```

### `Horizyn-evo_data1.tar.gz`

This archive contains the five CSV files used to prepare, train, and evaluate the Horizyn ESR-evo variants:

```text
data1/
|-- train.csv
|-- valid.csv
|-- Enzyme-405.csv
|-- Orphan-335_retrievel_cands.csv
`-- rhea_rxn2uids.csv
```

- `train.csv` and `valid.csv` define the training and validation data.
- `Enzyme-405.csv` contains the Enzyme-405 benchmark candidates.
- `Orphan-335_retrievel_cands.csv` contains the Orphan-335 candidate rankings to be scored.
- `rhea_rxn2uids.csv` provides the positive reaction-enzyme mapping used for Orphan-335 evaluation.

Extract the `data1` directory under:

```text
Horizyn-evo/data1/
```

## External Data and Software

The remaining data and features required by the two backbones are available from the original EnzymeCAGE and Horizyn repositories and their associated data distributions. This includes the original reaction features, protein language-model features, structure-derived features, and backbone-specific training resources.

UniRef90, MMseqs2, MSA Transformer, ESM-C, and ProtT5 model weights are not redistributed in this record. Their versions and usage are documented in the ESR-evo repository. The repository also provides the scripts used to generate A3M files and MSA Transformer features.

## Citation and License

When using this record, please cite the accompanying ESR-evo article and the original EnzymeCAGE and Horizyn publications. Data and pretrained models derived from the original backbones remain subject to their respective licenses and terms of use.
