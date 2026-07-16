#!/usr/bin/env python3
"""
Build Horizyn protein embeddings that contain ProtT5 + MSA features.

The output HDF5 keeps Horizyn's simple embedding format:

    ids:     protein IDs
    vectors: [ProtT5 1024 | MSA 768 | MSA mask 1]

MSA features are read from data1/seq2feature.pkl, whose keys are full protein
sequences and whose values are protein-level MSA Transformer mean features.
Proteins without an MSA feature get a zero MSA vector and mask=0, allowing the
gated target encoder to fall back to ProtT5.
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
        raise ValueError(f"No MSA features found in {path}")

    return seq2feature


def decode_h5_ids(raw_ids) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_ids
    ]


def write_t5_msa_h5(
    source_h5_path: Path,
    output_h5_path: Path,
    protein_to_sequence: dict[str, str],
    requested_ids: set[str],
    seq2feature: dict[str, np.ndarray],
    overwrite: bool,
    batch_size: int,
) -> dict[str, int | str]:
    if output_h5_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_h5_path}. Pass --overwrite to replace it.")

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
        t5_dim = source["vectors"].shape[1]

        first_feature = next(iter(seq2feature.values()))
        msa_dim = int(first_feature.shape[0])
        output_dim = t5_dim + msa_dim + 1
        string_dtype = h5py.string_dtype(encoding="utf-8")

        msa_present = 0
        msa_missing = 0
        missing_sequence = 0

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
                    sequence = protein_to_sequence.get(protein_id)
                    if not sequence:
                        missing_sequence += 1
                        msa_missing += 1
                        continue

                    msa_feature = seq2feature.get(sequence)
                    if msa_feature is None:
                        msa_missing += 1
                        continue

                    fused_vectors[local_idx, t5_dim : t5_dim + msa_dim] = msa_feature
                    fused_vectors[local_idx, t5_dim + msa_dim] = 1.0
                    msa_present += 1

                end = row + len(batch)
                output["ids"][row:end] = protein_ids
                output["vectors"][row:end] = fused_vectors
                row = end
                print(f"  Wrote proteins: {row:,}/{len(selected):,}")

            output.attrs["source_h5"] = str(source_h5_path)
            output.attrs["seq2feature_path"] = "sequence-keyed MSA features"
            output.attrs["layout"] = f"ProtT5 {t5_dim} | MSA {msa_dim} | mask 1"
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
        "missing_sequence": missing_sequence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ProtT5+MSA HDF5 for Horizyn.")
    parser.add_argument("--data-dir", default="data1", help="Directory containing data1 CSV files.")
    parser.add_argument(
        "--source-h5",
        default="data/data1_horizyn/prots_t5.h5",
        help="Completed ProtT5 HDF5 for data1 proteins.",
    )
    parser.add_argument(
        "--seq2feature",
        default="data1/seq2feature.pkl",
        help="Pickle mapping sequence string to 768-dim MSA feature.",
    )
    parser.add_argument(
        "--output-h5",
        default="data/data1_horizyn/prots_t5_msa.h5",
        help="Output HDF5 with [ProtT5 | MSA | mask] vectors.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    set_large_csv_field_limit()
    args = parse_args()

    data_dir = Path(args.data_dir)
    source_h5 = Path(args.source_h5)
    seq2feature_path = Path(args.seq2feature)
    output_h5 = Path(args.output_h5)

    if not source_h5.exists():
        raise FileNotFoundError(f"Source ProtT5 HDF5 not found: {source_h5}")
    if not seq2feature_path.exists():
        raise FileNotFoundError(f"MSA feature pickle not found: {seq2feature_path}")

    print(f"Scanning data1 sequences in {data_dir}")
    protein_to_sequence, requested_ids, conflicts = scan_data1_sequences(data_dir)
    print(f"  Requested proteins: {len(requested_ids):,}")
    print(f"  Proteins with sequence: {len(protein_to_sequence):,}")
    print(f"  Sequence conflicts: {len(conflicts):,}")

    print(f"Loading MSA sequence features from {seq2feature_path}")
    seq2feature = load_seq2feature(seq2feature_path)
    print(f"  Loaded MSA features for {len(seq2feature):,} sequences")

    print(f"Writing ProtT5+MSA HDF5 to {output_h5}")
    summary = write_t5_msa_h5(
        source_h5_path=source_h5,
        output_h5_path=output_h5,
        protein_to_sequence=protein_to_sequence,
        requested_ids=requested_ids,
        seq2feature=seq2feature,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )
    summary["sequence_conflicts"] = len(conflicts)

    summary_path = output_h5.with_suffix(output_h5.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nProtT5+MSA HDF5 ready.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
