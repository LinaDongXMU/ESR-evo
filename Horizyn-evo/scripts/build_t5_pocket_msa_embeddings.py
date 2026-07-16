#!/usr/bin/env python3
"""
Build Horizyn protein embeddings from ProtT5 plus pocket-only MSA features.

The output HDF5 keeps the same format used by the projected-gate MSA configs:

    ids:     protein IDs
    vectors: [ProtT5 1024 | pocket MSA 768 | MSA mask 1]

Pocket MSA features are read from the torch file produced by the root-level
run_msa_feature.py script. That file is expected to contain a dictionary:

    protein_id -> pocket_node_feature

where each value is either a [num_pocket_residues, msa_dim] array/tensor or an
already pooled [msa_dim] vector. Node-level features are pooled to one protein
vector with --pool. Proteins without a pocket MSA feature get a zero MSA vector
and mask=0, so the gated target encoder falls back to ProtT5.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


def scan_data1_proteins(data_dir: Path) -> set[str]:
    requested_ids: set[str] = set()

    for file_name in DATA1_FILES:
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing data1 file: {path}")

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                protein_id = get_protein_id(row)
                if protein_id:
                    requested_ids.add(protein_id)

    return requested_ids


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


def write_t5_pocket_msa_h5(
    source_h5_path: Path,
    output_h5_path: Path,
    requested_ids: set[str],
    pocket_features: dict[str, np.ndarray],
    overwrite: bool,
    batch_size: int,
    feature_path: Path,
    pool: str,
) -> dict[str, int | str]:
    if output_h5_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_h5_path}. Pass --overwrite to replace it.")

    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_h5_path.with_suffix(output_h5_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    msa_dim = int(next(iter(pocket_features.values())).shape[0])

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
        output_dim = t5_dim + msa_dim + 1
        string_dtype = h5py.string_dtype(encoding="utf-8")

        msa_present = 0
        msa_missing = 0

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
                    pocket_feature = pocket_features.get(protein_id)
                    if pocket_feature is None:
                        msa_missing += 1
                        continue
                    if pocket_feature.shape != (msa_dim,):
                        raise ValueError(
                            f"Pocket MSA feature for {protein_id} has shape "
                            f"{pocket_feature.shape}, expected ({msa_dim},)"
                        )

                    fused_vectors[local_idx, t5_dim : t5_dim + msa_dim] = pocket_feature
                    fused_vectors[local_idx, t5_dim + msa_dim] = 1.0
                    msa_present += 1

                end = row + len(batch)
                output["ids"][row:end] = protein_ids
                output["vectors"][row:end] = fused_vectors
                row = end
                print(f"  Wrote proteins: {row:,}/{len(selected):,}")

            output.attrs["source_h5"] = str(source_h5_path)
            output.attrs["pocket_msa_feature_path"] = str(feature_path)
            output.attrs["pocket_pool"] = pool
            output.attrs["layout"] = f"ProtT5 {t5_dim} | pocket MSA {msa_dim} | mask 1"
            output.attrs["t5_dim"] = t5_dim
            output.attrs["msa_dim"] = msa_dim

    temp_path.replace(output_h5_path)

    return {
        "output_h5": str(output_h5_path),
        "requested_proteins": len(requested_ids),
        "written_proteins": len(selected),
        "t5_dim": t5_dim,
        "msa_dim": msa_dim,
        "output_dim": output_dim,
        "msa_present": msa_present,
        "msa_missing": msa_missing,
        "pocket_feature_entries": len(pocket_features),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ProtT5+pocket-MSA HDF5 for Horizyn.")
    parser.add_argument("--data-dir", default="data1", help="Directory containing data1 CSV files.")
    parser.add_argument(
        "--source-h5",
        default="data/data1_horizyn/prots_t5.h5",
        help="Completed ProtT5 HDF5 for data1 proteins.",
    )
    parser.add_argument(
        "--pocket-node-feature",
        default="data1/msa_node_feature.pt",
        help="Torch file mapping protein IDs to pocket residue MSA features.",
    )
    parser.add_argument(
        "--output-h5",
        default="data/data1_horizyn/prots_t5_pocket_msa.h5",
        help="Output HDF5 with [ProtT5 | pocket MSA | mask] vectors.",
    )
    parser.add_argument("--pool", choices=["mean", "max"], default="mean")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    set_large_csv_field_limit()
    args = parse_args()

    data_dir = Path(args.data_dir)
    source_h5 = Path(args.source_h5)
    pocket_node_feature_path = Path(args.pocket_node_feature)
    output_h5 = Path(args.output_h5)

    if not source_h5.exists():
        raise FileNotFoundError(f"Source ProtT5 HDF5 not found: {source_h5}")
    if not pocket_node_feature_path.exists():
        raise FileNotFoundError(f"Pocket MSA feature file not found: {pocket_node_feature_path}")

    print(f"Scanning data1 proteins in {data_dir}")
    requested_ids = scan_data1_proteins(data_dir)
    print(f"  Requested proteins: {len(requested_ids):,}")

    print(f"Loading pocket MSA node features from {pocket_node_feature_path}")
    pocket_features, feature_summary = load_pocket_msa_features(pocket_node_feature_path, args.pool)
    print(f"  Loaded usable pocket features: {len(pocket_features):,}")
    print(f"  Pocket MSA dim: {feature_summary['msa_dim']}")
    missing_pocket_feature_ids = sorted(requested_ids - set(pocket_features))
    print(f"  Requested proteins with pocket features: {len(requested_ids) - len(missing_pocket_feature_ids):,}")
    print(f"  Requested proteins missing pocket features: {len(missing_pocket_feature_ids):,}")

    print(f"Writing ProtT5+pocket-MSA HDF5 to {output_h5}")
    summary = write_t5_pocket_msa_h5(
        source_h5_path=source_h5,
        output_h5_path=output_h5,
        requested_ids=requested_ids,
        pocket_features=pocket_features,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        feature_path=pocket_node_feature_path,
        pool=args.pool,
    )
    summary["pocket_feature_summary"] = feature_summary
    summary["requested_with_pocket_feature"] = len(requested_ids) - len(missing_pocket_feature_ids)
    summary["requested_missing_pocket_feature"] = len(missing_pocket_feature_ids)

    summary_path = output_h5.with_suffix(output_h5.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if missing_pocket_feature_ids:
        missing_path = output_h5.with_suffix(output_h5.suffix + ".missing_pocket_msa.txt")
        missing_path.write_text("\n".join(missing_pocket_feature_ids) + "\n", encoding="utf-8")

    print("\nProtT5+pocket-MSA HDF5 ready.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
