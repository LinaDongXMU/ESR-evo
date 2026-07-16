#!/usr/bin/env python3
"""
Score data1 candidate CSVs with a trained Horizyn checkpoint.

The output CSV keeps the original columns, adds a prediction column named
"pred" by default, and reports the benchmark ranking metrics unless
--skip-metrics is specified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = None
load_config = None
BaseDataset = None
MergeDataset = None
DRFPFingerprintDataset = None
RDKitPlusFingerprintDataset = None
EmbedDataset = None
ConcatTensorTransform = None
HorizynLitModule = None


def load_runtime_dependencies() -> None:
    global torch
    global load_config
    global BaseDataset
    global MergeDataset
    global DRFPFingerprintDataset
    global RDKitPlusFingerprintDataset
    global EmbedDataset
    global ConcatTensorTransform
    global HorizynLitModule

    import torch as torch_module
    from horizyn.config import load_config as load_config_fn
    from horizyn.datasets.base import BaseDataset as BaseDatasetCls
    from horizyn.datasets.collection import MergeDataset as MergeDatasetCls
    from horizyn.datasets.fingerprints import (
        DRFPFingerprintDataset as DRFPFingerprintDatasetCls,
        RDKitPlusFingerprintDataset as RDKitPlusFingerprintDatasetCls,
    )
    from horizyn.datasets.hdf5 import EmbedDataset as EmbedDatasetCls
    from horizyn.datasets.transform import ConcatTensorTransform as ConcatTensorTransformCls
    from horizyn.lightning_module import HorizynLitModule as HorizynLitModuleCls

    torch = torch_module
    load_config = load_config_fn
    BaseDataset = BaseDatasetCls
    MergeDataset = MergeDatasetCls
    DRFPFingerprintDataset = DRFPFingerprintDatasetCls
    RDKitPlusFingerprintDataset = RDKitPlusFingerprintDatasetCls
    EmbedDataset = EmbedDatasetCls
    ConcatTensorTransform = ConcatTensorTransformCls
    HorizynLitModule = HorizynLitModuleCls


def set_large_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def first_nonempty(row: dict[str, str], columns: Iterable[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_reaction_smiles(row: dict[str, str]) -> str:
    return first_nonempty(row, ("CANO_RXN_SMILES", "reaction", "SMILES"))


def get_protein_id(row: dict[str, str]) -> str:
    return first_nonempty(row, ("enzyme", "UniprotID"))


def get_reaction_column(fieldnames: list[str]) -> str:
    if "CANO_RXN_SMILES" in fieldnames:
        return "CANO_RXN_SMILES"
    if "reaction" in fieldnames:
        return "reaction"
    return "SMILES"


def get_enzyme_column(fieldnames: list[str]) -> str:
    if "enzyme" in fieldnames:
        return "enzyme"
    return "UniprotID"


def parse_label(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).strip() or 0.0)
    except ValueError:
        return 0.0


def csv_has_positive_labels(path: Path) -> bool:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "Label" not in (reader.fieldnames or []):
            return False
        for row in reader:
            if parse_label(row.get("Label")) > 0:
                return True
    return False


def load_true_reaction_enzyme_mapping(path: Path) -> dict[str, set[str]]:
    true_mapping: dict[str, set[str]] = defaultdict(set)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rxn_col = get_reaction_column(fieldnames)
        enz_col = get_enzyme_column(fieldnames)
        has_label = "Label" in fieldnames

        for row in reader:
            if has_label and parse_label(row.get("Label")) <= 0:
                continue
            rxn = row.get(rxn_col, "")
            enzyme = row.get(enz_col, "")
            if rxn and enzyme:
                true_mapping[rxn].add(enzyme)

    return true_mapping


def reaction_key_for(smiles: str, suffix: str) -> str:
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]
    return f"rxn_{digest}_{suffix}"


def reverse_reaction_smiles(smiles: str) -> str | None:
    parts = smiles.split(">>")
    if len(parts) != 2:
        return None
    return f"{parts[1]}>>{parts[0]}"


def collect_candidates(input_path: Path) -> tuple[list[str], set[str], set[str], int]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        reactions: set[str] = set()
        proteins: set[str] = set()
        rows = 0
        for row in reader:
            rows += 1
            reaction_smiles = get_reaction_smiles(row)
            protein_id = get_protein_id(row)
            if reaction_smiles:
                reactions.add(reaction_smiles)
            if protein_id:
                proteins.add(protein_id)
    return fieldnames, reactions, proteins, rows


def build_reaction_dataset(
    reaction_smiles: set[str], score_mode: str
) -> tuple[BaseDataset[str], dict[str, list[str]]]:
    keys: list[str] = []
    data: list[dict[str, str]] = []
    reaction_to_keys: dict[str, list[str]] = {}

    for smiles in sorted(reaction_smiles):
        forward_key = reaction_key_for(smiles, "f")
        keys.append(forward_key)
        data.append({"reaction_smiles": smiles})
        scoring_keys = [forward_key]

        if score_mode in {"max", "mean"}:
            reversed_smiles = reverse_reaction_smiles(smiles)
            if reversed_smiles is not None:
                reverse_key = reaction_key_for(smiles, "r")
                keys.append(reverse_key)
                data.append({"reaction_smiles": reversed_smiles})
                scoring_keys.append(reverse_key)

        reaction_to_keys[smiles] = scoring_keys

    return BaseDataset(keys=keys, array_data=data), reaction_to_keys


def create_fingerprint_dataset(reactions: BaseDataset[str], config):
    rdkit_fp = RDKitPlusFingerprintDataset(
        reaction_dataset=reactions,
        vec_dim=config.data.get("rdkit_fp_dim", 1024),
        mol_fp_type="morgan",
        rxn_fp_type="struct",
        use_chirality=True,
        standardize=config.data.get("standardize_reactions", True),
        standardize_hypervalent=config.data.get("standardize_hypervalent", True),
        standardize_remove_hs=config.data.get("standardize_remove_hs", True),
        standardize_kekulize=config.data.get("standardize_kekulize", False),
        standardize_uncharge=config.data.get("standardize_uncharge", True),
        standardize_metals=config.data.get("standardize_metals", True),
        standardize_error_policy=config.data.get("standardize_error_policy", "raise"),
        fingerprint_error_policy=config.data.get("fingerprint_error_policy", "raise"),
    )
    drfp_fp = DRFPFingerprintDataset(
        reaction_dataset=reactions,
        vec_dim=config.data.get("drfp_dim", 1024),
        radius=3,
        rings=True,
        standardize=config.data.get("standardize_reactions", True),
        standardize_hypervalent=config.data.get("standardize_hypervalent", True),
        standardize_remove_hs=config.data.get("standardize_remove_hs", True),
        standardize_kekulize=config.data.get("standardize_kekulize", False),
        standardize_uncharge=config.data.get("standardize_uncharge", True),
        standardize_metals=config.data.get("standardize_metals", True),
        standardize_error_policy=config.data.get("standardize_error_policy", "raise"),
        fingerprint_error_policy=config.data.get("fingerprint_error_policy", "raise"),
    )
    merged = MergeDataset(datasets={"rdkit": rdkit_fp, "drfp": drfp_fp}, add_prefix=False)
    merged.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))
    return merged


def encode_reactions(
    model: HorizynLitModule,
    fingerprint_dataset,
    reaction_keys: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    encoded: dict[str, torch.Tensor] = {}
    model.eval()

    with torch.no_grad():
        for start in range(0, len(reaction_keys), batch_size):
            batch_keys = reaction_keys[start : start + batch_size]
            vectors = torch.stack([fingerprint_dataset[key] for key in batch_keys]).to(device)
            embeds = model.model.query_encoder(vectors).detach().cpu()
            for key, embedding in zip(batch_keys, embeds):
                encoded[key] = embedding
            print(f"  Encoded reactions: {min(start + batch_size, len(reaction_keys)):,}/{len(reaction_keys):,}")

    return encoded


def encode_proteins(
    model: HorizynLitModule,
    embed_dataset: EmbedDataset,
    protein_ids: set[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    missing = sorted(protein_ids - set(embed_dataset.keys))
    if missing:
        raise KeyError(
            f"{len(missing)} candidate proteins are missing from {embed_dataset.file_path}. "
            f"First missing IDs: {missing[:10]}"
        )

    encoded: dict[str, torch.Tensor] = {}
    sorted_ids = sorted(protein_ids)
    model.eval()

    with torch.no_grad():
        for start in range(0, len(sorted_ids), batch_size):
            batch_ids = sorted_ids[start : start + batch_size]
            vectors = torch.stack([embed_dataset[protein_id] for protein_id in batch_ids]).to(device)
            embeds = model.model.target_encoder(vectors).detach().cpu()
            for protein_id, embedding in zip(batch_ids, embeds):
                encoded[protein_id] = embedding
            print(f"  Encoded proteins: {min(start + batch_size, len(sorted_ids)):,}/{len(sorted_ids):,}")

    return encoded


def combine_scores(scores: list[float], score_mode: str) -> float:
    if score_mode == "mean":
        return sum(scores) / len(scores)
    if score_mode == "max":
        return max(scores)
    return scores[0]


def score_rows(
    input_path: Path,
    output_path: Path,
    fieldnames: list[str],
    reaction_to_keys: dict[str, list[str]],
    reaction_embeds: dict[str, torch.Tensor],
    protein_embeds: dict[str, torch.Tensor],
    score_mode: str,
    pred_col: str,
    label_mapping: dict[str, set[str]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = list(fieldnames)
    if label_mapping is not None and "Label" not in output_fields:
        output_fields.append("Label")
    if pred_col not in output_fields:
        output_fields.append(pred_col)

    with input_path.open("r", newline="", encoding="utf-8") as input_handle, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()

        for row in reader:
            reaction_smiles = get_reaction_smiles(row)
            protein_id = get_protein_id(row)
            scoring_keys = reaction_to_keys.get(reaction_smiles, [])
            if not scoring_keys or protein_id not in protein_embeds:
                raise KeyError(
                    f"Cannot score row with reaction={reaction_smiles!r}, protein={protein_id!r}"
                )

            target_embedding = protein_embeds[protein_id]
            scores = [
                torch.dot(reaction_embeds[reaction_key], target_embedding).item()
                for reaction_key in scoring_keys
            ]
            if label_mapping is not None:
                row["Label"] = "1" if protein_id in label_mapping.get(reaction_smiles, set()) else "0"
            row[pred_col] = f"{combine_scores(scores, score_mode):.10g}"
            writer.writerow(row)


def compute_dcg(relevances: list[float], k: int = 10) -> float:
    total = 0.0
    for idx, relevance in enumerate(relevances[:k]):
        total += (2.0**relevance - 1.0) / math.log2(idx + 2.0)
    return total


def enrichment_factor(rows: list[dict[str, str]], pred_col: str, top_percent: float) -> float:
    sorted_rows = sorted(rows, key=lambda row: float(row[pred_col]), reverse=True)
    topk = max(int(top_percent * len(sorted_rows)), 5)
    top_rows = sorted_rows[:topk]
    total_active = sum(parse_label(row.get("Label")) for row in sorted_rows)
    if total_active == 0:
        return 0.0
    active_top = sum(parse_label(row.get("Label")) for row in top_rows)
    random_active_top = total_active * top_percent
    return active_top / random_active_top


def evaluate_prediction_csv(path: Path, pred_col: str) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rxn_col = get_reaction_column(fieldnames)
        enz_col = get_enzyme_column(fieldnames)

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        true_enzymes: dict[str, set[str]] = defaultdict(set)
        seen_by_rxn: dict[str, set[str]] = defaultdict(set)

        for row in reader:
            rxn = row[rxn_col]
            enzyme = row[enz_col]
            if enzyme in seen_by_rxn[rxn]:
                continue
            seen_by_rxn[rxn].add(enzyme)
            grouped[rxn].append(row)
            if parse_label(row.get("Label")) > 0:
                true_enzymes[rxn].add(enzyme)

    test_rxns = list(grouped.keys())
    dcg_values = []
    ef1_values = []
    ef2_values = []
    best_ranks = []

    for rxn, rows in grouped.items():
        ranked = sorted(rows, key=lambda row: float(row[pred_col]), reverse=True)
        dcg_values.append(compute_dcg([parse_label(row.get("Label")) for row in ranked], k=10))
        ef1_values.append(enrichment_factor(rows, pred_col=pred_col, top_percent=0.01))
        ef2_values.append(enrichment_factor(rows, pred_col=pred_col, top_percent=0.02))

        ranked_enzymes = [row[get_enzyme_column(list(row.keys()))] for row in ranked]
        hits = [
            ranked_enzymes.index(enzyme) + 1
            for enzyme in true_enzymes.get(rxn, set())
            if enzyme in ranked_enzymes
        ]
        best_ranks.append(min(hits) if hits else -1)

    denom = max(1, len(test_rxns))
    results = {
        "top10_dcg": sum(dcg_values) / denom,
        "top1_ef": sum(ef1_values) / denom,
        "top2_ef": sum(ef2_values) / denom,
        "num_reactions": float(len(test_rxns)),
    }
    for topk in (1, 3, 5, 10):
        successes = [rank for rank in best_ranks if 0 < rank <= topk]
        results[f"top{topk}_sr"] = len(successes) / denom

    return results


def print_metrics(results: dict[str, float]) -> None:
    print("\n########### Evaluation Results ###########")
    print(f"Top-10 DCG: {results['top10_dcg']:.4f}")
    print(f"Top-1% EF : {results['top1_ef']:.4f}")
    print(f"Top-2% EF : {results['top2_ef']:.4f}")
    print(f"Top-1  SR : {results['top1_sr'] * 100:.2f}%")
    print(f"Top-3  SR : {results['top3_sr'] * 100:.2f}%")
    print(f"Top-5  SR : {results['top5_sr'] * 100:.2f}%")
    print(f"Top-10 SR : {results['top10_sr'] * 100:.2f}%")
    print("###########################################\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score data1 candidate CSVs with Horizyn.")
    parser.add_argument("--checkpoint", required=True, help="Trained Horizyn checkpoint.")
    parser.add_argument(
        "--config",
        default="configs/data1_t5_full_pocket_msa_projected_gate.yaml",
        help="Horizyn evolutionary-augmentation config path.",
    )
    parser.add_argument("--input", required=True, help="Input candidate CSV.")
    parser.add_argument("--output", required=True, help="Output scored CSV.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for encoding. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Reaction encoding batch size.")
    parser.add_argument(
        "--protein-batch-size", type=int, default=2048, help="Protein encoding batch size."
    )
    parser.add_argument(
        "--score-mode",
        choices=["forward", "max", "mean"],
        default="max",
        help="How to score bidirectional reactions. 'max' matches Horizyn's reversible setup.",
    )
    parser.add_argument("--pred-col", default="pred", help="Prediction column name.")
    parser.add_argument(
        "--pos-pair-db-path",
        default=None,
        help=(
            "Optional true reaction-enzyme mapping CSV. For Orphan-335 this should be "
            "data1/rhea_rxn2uids.csv so Label and metrics can be computed."
        ),
    )
    parser.add_argument(
        "--fill-labels-from-pos-pair-db",
        action="store_true",
        help="Overwrite/add Label in the scored output using --pos-pair-db-path.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only write predictions; do not print data1-style metrics.",
    )
    return parser.parse_args()


def main() -> None:
    set_large_csv_field_limit()
    args = parse_args()
    load_runtime_dependencies()

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    label_mapping = None
    pos_pair_db_path = Path(args.pos_pair_db_path) if args.pos_pair_db_path else None
    if pos_pair_db_path is not None:
        if not pos_pair_db_path.exists():
            raise FileNotFoundError(f"Positive pair DB not found: {pos_pair_db_path}")
        should_fill_labels = args.fill_labels_from_pos_pair_db or not csv_has_positive_labels(input_path)
        if should_fill_labels:
            print(f"Loading true reaction-enzyme mapping from {pos_pair_db_path}")
            label_mapping = load_true_reaction_enzyme_mapping(pos_pair_db_path)
            true_pair_count = sum(len(v) for v in label_mapping.values())
            print(
                f"  Loaded {true_pair_count:,} true pairs across "
                f"{len(label_mapping):,} reactions"
            )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    print(f"Loading checkpoint from {checkpoint_path}")
    model = HorizynLitModule.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device)
    model.eval()

    print(f"Collecting candidates from {input_path}")
    fieldnames, reactions, proteins, row_count = collect_candidates(input_path)
    print(f"  Rows: {row_count:,}")
    print(f"  Unique reactions: {len(reactions):,}")
    print(f"  Unique proteins: {len(proteins):,}")

    print("Building reaction fingerprints")
    reaction_dataset, reaction_to_keys = build_reaction_dataset(reactions, args.score_mode)
    fingerprint_dataset = create_fingerprint_dataset(reaction_dataset, config)

    print("Encoding reactions")
    reaction_embeds = encode_reactions(
        model,
        fingerprint_dataset,
        reaction_dataset.keys,
        device=device,
        batch_size=args.batch_size,
    )

    print(f"Loading protein embeddings from {config.data.protein_embeds_path}")
    protein_dataset = EmbedDataset(config.data.protein_embeds_path, in_memory=True)

    try:
        print("Encoding candidate proteins")
        protein_embeds = encode_proteins(
            model,
            protein_dataset,
            proteins,
            device=device,
            batch_size=args.protein_batch_size,
        )
    except KeyError as exc:
        missing_path = output_path.with_name(f"{output_path.stem}.missing_embeddings.txt")
        missing = sorted(proteins - set(protein_dataset.keys))
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise KeyError(f"{exc}\nMissing ID list written to: {missing_path}") from exc

    print(f"Writing scored candidates to {output_path}")
    score_rows(
        input_path=input_path,
        output_path=output_path,
        fieldnames=fieldnames,
        reaction_to_keys=reaction_to_keys,
        reaction_embeds=reaction_embeds,
        protein_embeds=protein_embeds,
        score_mode=args.score_mode,
        pred_col=args.pred_col,
        label_mapping=label_mapping,
    )

    if not args.skip_metrics:
        results = evaluate_prediction_csv(output_path, pred_col=args.pred_col)
        print_metrics(results)


if __name__ == "__main__":
    main()
