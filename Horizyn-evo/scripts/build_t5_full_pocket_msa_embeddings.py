#!/usr/bin/env python3
"""
Build Horizyn protein embeddings from ProtT5, full-sequence MSA, and pocket MSA.

The output HDF5 keeps Horizyn's simple embedding format:

    ids:     protein IDs
    vectors: [ProtT5 | full MSA | full mask | pocket MSA | pocket mask]

Full-sequence MSA features are read from data1/seq2feature.pkl, whose keys are
protein sequences. Pocket MSA features are read from the torch file produced by
the root-level run_msa_feature.py script:

    protein_id -> pocket_node_feature

Pocket node features are pooled to one protein vector with --pocket-pool.
Missing full or pocket features get zero vectors with mask=0, allowing the
dual-MSA target encoder to use whichever MSA view is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "2")

DATA1_FILES = (
    "train.csv",
    "valid.csv",
    "Enzyme-405.csv",
    "Orphan-335_retrievel_cands.csv",
)


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


def get_protein_id(row: dict[str, str]) -> str:
    return first_nonempty(row, ("enzyme", "UniprotID"))


def scan_data1_sequences(data_dir: Path) -> tuple[dict[str, str], set[str], set[str]]:
    protein_to_sequence: dict[str, str] = {}
    requested_ids: set[str] = set()
    conflicts: set[str] = set()

    for file_name in DATA1_FILES:
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing data1 file: {path}")

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                protein_id = get_protein_id(row)
                sequence = first_nonempty(row, ("sequence",))
                if not protein_id:
                    continue
                requested_ids.add(protein_id)
                if not sequence:
                    continue

                existing = protein_to_sequence.get(protein_id)
                if existing is None:
                    protein_to_sequence[protein_id] = sequence
                elif existing != sequence:
                    conflicts.add(protein_id)

    return protein_to_sequence, requested_ids, conflicts


def load_seq2feature(path: Path) -> dict[str, np.ndarray]:
    # Some pickles created with newer NumPy refer to numpy._core. This shim lets
    # older NumPy installations read them.
    try:
        import numpy.core as np_core

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    except Exception:
        pass

    with path.open("rb") as handle:
        seq2feature = pickle.load(handle)

    if not isinstance(seq2feature, dict):
        raise ValueError(f"Expected dict in {path}, got {type(seq2feature).__name__}")
    if not seq2feature:
        raise ValueError(f"No full-sequence MSA features found in {path}")

    return {
        str(sequence): np.asarray(feature, dtype=np.float32)
        for sequence, feature in seq2feature.items()
    }


def decode_h5_ids(raw_ids) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_ids
    ]


def to_numpy_feature(value: object) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pool_node_feature(feature: np.ndarray, pool: str) -> np.ndarray | None:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 1:
        return feature
    if feature.ndim != 2 or feature.shape[0] == 0:
        return None
    if pool == "max":
        return feature.max(axis=0)
    return feature.mean(axis=0)


def load_pocket_msa_features(path: Path, pool: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    import torch

    try:
        raw_features = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw_features = torch.load(path, map_location="cpu")

    if not isinstance(raw_features, dict):
        raise ValueError(f"Expected dict in {path}, got {type(raw_features).__name__}")

    features: dict[str, np.ndarray] = {}
    skipped_empty_or_bad_shape = 0
    skipped_non_numeric = 0

    for raw_key, raw_value in raw_features.items():
        protein_id = str(raw_key).strip()
        if not protein_id:
            skipped_empty_or_bad_shape += 1
            continue

        try:
            pooled = pool_node_feature(to_numpy_feature(raw_value), pool=pool)
        except (TypeError, ValueError):
            skipped_non_numeric += 1
            continue

        if pooled is None or pooled.size == 0:
            skipped_empty_or_bad_shape += 1
            continue

        features[protein_id] = pooled.astype(np.float32, copy=False)

    if not features:
        raise ValueError(f"No usable pocket MSA features found in {path}")

    dims = {int(feature.shape[0]) for feature in features.values()}
    if len(dims) != 1:
        raise ValueError(f"Pocket MSA feature dimensions are inconsistent: {sorted(dims)}")

    summary = {
        "raw_entries": len(raw_features),
        "usable_entries": len(features),
        "skipped_empty_or_bad_shape": skipped_empty_or_bad_shape,
        "skipped_non_numeric": skipped_non_numeric,
        "msa_dim": next(iter(dims)),
    }
    return features, summary


def write_full_pocket_h5(
    source_h5_path: Path,
    output_h5_path: Path,
    protein_to_sequence: dict[str, str],
    requested_ids: set[str],
    seq2feature: dict[str, np.ndarray],
    pocket_features: dict[str, np.ndarray],
    overwrite: bool,
    batch_size: int,
    seq2feature_path: Path,
    pocket_feature_path: Path,
    pocket_pool: str,
) -> dict[str, int | str]:
    if output_h5_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_h5_path}. Pass --overwrite to replace it.")

    full_msa_dim = int(next(iter(seq2feature.values())).shape[0])
    pocket_msa_dim = int(next(iter(pocket_features.values())).shape[0])
    if full_msa_dim != pocket_msa_dim:
        raise ValueError(
            "This encoder expects full and pocket MSA features to have the same dim. "
            f"Got full={full_msa_dim}, pocket={pocket_msa_dim}."
        )

    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_h5_path.with_suffix(output_h5_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    with h5py.File(source_h5_path, "r") as source:
        source_ids = decode_h5_ids(source["ids"][:])
        source_id_set = set(source_ids)
        missing_t5_ids = sorted(requested_ids - source_id_set)
        if missing_t5_ids:
            raise KeyError(
                f"{len(missing_t5_ids)} data1 proteins are missing ProtT5 embeddings. "
                f"First missing IDs: {missing_t5_ids[:10]}"
            )

        selected = [
            (idx, protein_id)
            for idx, protein_id in enumerate(source_ids)
            if protein_id in requested_ids
        ]
        t5_dim = int(source["vectors"].shape[1])
        output_dim = t5_dim + full_msa_dim + 1 + pocket_msa_dim + 1
        string_dtype = h5py.string_dtype(encoding="utf-8")

        full_present = 0
        full_missing = 0
        pocket_present = 0
        pocket_missing = 0
        both_present = 0
        either_present = 0
        missing_sequence = 0

        full_start = t5_dim
        full_mask_idx = full_start + full_msa_dim
        pocket_start = full_mask_idx + 1
        pocket_mask_idx = pocket_start + pocket_msa_dim

        with h5py.File(temp_path, "w") as output:
            output.create_dataset("ids", shape=(len(selected),), dtype=string_dtype)
            output.create_dataset(
                "vectors",
                shape=(len(selected), output_dim),
                dtype=np.float32,
                chunks=(min(1024, max(1, len(selected))), output_dim),
            )

            row = 0
            for start in range(0, len(selected), batch_size):
                batch = selected[start : start + batch_size]
                source_indices = [idx for idx, _ in batch]
                protein_ids = [protein_id for _, protein_id in batch]
                t5_vectors = source["vectors"][source_indices].astype(np.float32, copy=False)

                fused_vectors = np.zeros((len(batch), output_dim), dtype=np.float32)
                fused_vectors[:, :t5_dim] = t5_vectors

                for local_idx, protein_id in enumerate(protein_ids):
                    has_full = False
                    has_pocket = False

                    sequence = protein_to_sequence.get(protein_id)
                    if not sequence:
                        missing_sequence += 1
                    else:
                        full_feature = seq2feature.get(sequence)
                        if full_feature is not None:
                            if full_feature.shape != (full_msa_dim,):
                                raise ValueError(
                                    f"Full MSA feature for {protein_id} has shape "
                                    f"{full_feature.shape}, expected ({full_msa_dim},)"
                                )
                            fused_vectors[local_idx, full_start:full_mask_idx] = full_feature
                            fused_vectors[local_idx, full_mask_idx] = 1.0
                            has_full = True

                    pocket_feature = pocket_features.get(protein_id)
                    if pocket_feature is not None:
                        if pocket_feature.shape != (pocket_msa_dim,):
                            raise ValueError(
                                f"Pocket MSA feature for {protein_id} has shape "
                                f"{pocket_feature.shape}, expected ({pocket_msa_dim},)"
                            )
                        fused_vectors[local_idx, pocket_start:pocket_mask_idx] = pocket_feature
                        fused_vectors[local_idx, pocket_mask_idx] = 1.0
                        has_pocket = True

                    full_present += int(has_full)
                    full_missing += int(not has_full)
                    pocket_present += int(has_pocket)
                    pocket_missing += int(not has_pocket)
                    both_present += int(has_full and has_pocket)
                    either_present += int(has_full or has_pocket)

                end = row + len(batch)
                output["ids"][row:end] = protein_ids
                output["vectors"][row:end] = fused_vectors
                row = end
                print(f"  Wrote proteins: {row:,}/{len(selected):,}")

            output.attrs["source_h5"] = str(source_h5_path)
            output.attrs["seq2feature_path"] = str(seq2feature_path)
            output.attrs["pocket_msa_feature_path"] = str(pocket_feature_path)
            output.attrs["pocket_pool"] = pocket_pool
            output.attrs["layout"] = (
                f"ProtT5 {t5_dim} | full MSA {full_msa_dim} | full mask 1 | "
                f"pocket MSA {pocket_msa_dim} | pocket mask 1"
            )
            output.attrs["t5_dim"] = t5_dim
            output.attrs["full_msa_dim"] = full_msa_dim
            output.attrs["pocket_msa_dim"] = pocket_msa_dim

    temp_path.replace(output_h5_path)

    return {
        "output_h5": str(output_h5_path),
        "requested_proteins": len(requested_ids),
        "written_proteins": len(selected),
        "t5_dim": t5_dim,
        "full_msa_dim": full_msa_dim,
        "pocket_msa_dim": pocket_msa_dim,
        "output_dim": output_dim,
        "full_msa_present": full_present,
        "full_msa_missing": full_missing,
        "pocket_msa_present": pocket_present,
        "pocket_msa_missing": pocket_missing,
        "both_msa_present": both_present,
        "either_msa_present": either_present,
        "missing_sequence": missing_sequence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ProtT5+full-MSA+pocket-MSA HDF5 for Horizyn."
    )
    parser.add_argument("--data-dir", default="data1", help="Directory containing data1 CSV files.")
    parser.add_argument(
        "--source-h5",
        default="data/data1_horizyn/prots_t5.h5",
        help="Completed ProtT5 HDF5 for data1 proteins.",
    )
    parser.add_argument(
        "--seq2feature",
        default="data1/seq2feature.pkl",
        help="Pickle mapping sequence string to full-sequence 768-dim MSA feature.",
    )
    parser.add_argument(
        "--pocket-node-feature",
        default="data1/msa_node_feature.pt",
        help="Torch file mapping protein IDs to pocket residue MSA features.",
    )
    parser.add_argument(
        "--output-h5",
        default="data/data1_horizyn/prots_t5_full_pocket_msa.h5",
        help="Output HDF5 with [ProtT5 | full MSA | full mask | pocket MSA | pocket mask].",
    )
    parser.add_argument("--pocket-pool", choices=["mean", "max"], default="mean")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    set_large_csv_field_limit()
    args = parse_args()

    data_dir = Path(args.data_dir)
    source_h5 = Path(args.source_h5)
    seq2feature_path = Path(args.seq2feature)
    pocket_node_feature_path = Path(args.pocket_node_feature)
    output_h5 = Path(args.output_h5)

    if not source_h5.exists():
        raise FileNotFoundError(f"Source ProtT5 HDF5 not found: {source_h5}")
    if not seq2feature_path.exists():
        raise FileNotFoundError(f"Full-sequence MSA feature pickle not found: {seq2feature_path}")
    if not pocket_node_feature_path.exists():
        raise FileNotFoundError(f"Pocket MSA feature file not found: {pocket_node_feature_path}")

    print(f"Scanning data1 sequences in {data_dir}")
    protein_to_sequence, requested_ids, conflicts = scan_data1_sequences(data_dir)
    print(f"  Requested proteins: {len(requested_ids):,}")
    print(f"  Proteins with sequence: {len(protein_to_sequence):,}")
    print(f"  Sequence conflicts: {len(conflicts):,}")

    print(f"Loading full-sequence MSA features from {seq2feature_path}")
    seq2feature = load_seq2feature(seq2feature_path)
    print(f"  Loaded full-sequence MSA features for {len(seq2feature):,} sequences")

    print(f"Loading pocket MSA node features from {pocket_node_feature_path}")
    pocket_features, pocket_feature_summary = load_pocket_msa_features(
        pocket_node_feature_path,
        args.pocket_pool,
    )
    print(f"  Loaded usable pocket MSA features: {len(pocket_features):,}")
    print(f"  Pocket MSA dim: {pocket_feature_summary['msa_dim']}")

    full_present_ids = {
        protein_id
        for protein_id, sequence in protein_to_sequence.items()
        if sequence in seq2feature
    }
    pocket_present_ids = requested_ids & set(pocket_features)
    full_missing_ids = sorted(requested_ids - full_present_ids)
    pocket_missing_ids = sorted(requested_ids - pocket_present_ids)
    print(f"  Requested proteins with full MSA: {len(full_present_ids & requested_ids):,}")
    print(f"  Requested proteins with pocket MSA: {len(pocket_present_ids):,}")
    print(f"  Requested proteins with both MSA views: {len(full_present_ids & pocket_present_ids):,}")

    print(f"Writing ProtT5+full+pocket MSA HDF5 to {output_h5}")
    summary = write_full_pocket_h5(
        source_h5_path=source_h5,
        output_h5_path=output_h5,
        protein_to_sequence=protein_to_sequence,
        requested_ids=requested_ids,
        seq2feature=seq2feature,
        pocket_features=pocket_features,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        seq2feature_path=seq2feature_path,
        pocket_feature_path=pocket_node_feature_path,
        pocket_pool=args.pocket_pool,
    )
    summary["sequence_conflicts"] = len(conflicts)
    summary["pocket_feature_summary"] = pocket_feature_summary
    summary["requested_missing_full_msa"] = len(full_missing_ids)
    summary["requested_missing_pocket_msa"] = len(pocket_missing_ids)

    summary_path = output_h5.with_suffix(output_h5.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if full_missing_ids:
        missing_full_path = output_h5.with_suffix(output_h5.suffix + ".missing_full_msa.txt")
        missing_full_path.write_text("\n".join(full_missing_ids) + "\n", encoding="utf-8")
    if pocket_missing_ids:
        missing_pocket_path = output_h5.with_suffix(output_h5.suffix + ".missing_pocket_msa.txt")
        missing_pocket_path.write_text("\n".join(pocket_missing_ids) + "\n", encoding="utf-8")

    print("\nProtT5+full+pocket MSA HDF5 ready.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
