#!/usr/bin/env python3
"""
Prepare data1 CSV files for Horizyn training and evaluation.

The data1 files are candidate-ranking tables. Horizyn training only needs
positive reaction-protein pairs, so this script converts rows with Label > 0
into the train/validation CSV format expected by Horizyn:

    train_pairs.csv: pr_id,reaction_id,protein_id
    train_rxns.csv:  rs_id,reaction_id,reaction_smiles

It also writes protein FASTA/CSV files and can subset an existing Horizyn
protein embedding HDF5 file to the protein IDs needed by data1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable


FIT_FILES = ("train.csv", "valid.csv")
TEST_FILES = ("Enzyme-405.csv", "Orphan-335_retrievel_cands.csv")


def set_large_csv_field_limit() -> None:
    """Raise csv field limit for long reaction/template/sequence columns."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_label(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def is_positive(row: dict[str, str]) -> bool:
    return parse_label(row.get("Label")) > 0.0


def first_nonempty(row: dict[str, str], columns: Iterable[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_reaction_smiles(row: dict[str, str]) -> str:
    return first_nonempty(row, ("CANO_RXN_SMILES", "reaction", "SMILES"))


def get_protein_id(row: dict[str, str]) -> str:
    # Candidate files may use either the benchmark or backbone identifier column.
    return first_nonempty(row, ("enzyme", "UniprotID"))


def reaction_id_for(smiles: str) -> str:
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]
    return f"rxn_{digest}"


def wrap_fasta(sequence: str, width: int = 80) -> Iterable[str]:
    for start in range(0, len(sequence), width):
        yield sequence[start : start + width]


def read_h5_ids(path: Path) -> set[str]:
    os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "2")
    import h5py

    with h5py.File(path, "r") as handle:
        ids = handle["ids"][:]

    decoded = []
    for value in ids:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return set(decoded)


def write_h5_subset(source_path: Path, output_path: Path, requested_ids: set[str]) -> set[str]:
    """Write source embeddings whose IDs are requested. Returns written IDs."""
    os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "2")
    import h5py

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source:
        raw_ids = source["ids"][:]
        source_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_ids
        ]
        selected = [
            (idx, protein_id)
            for idx, protein_id in enumerate(source_ids)
            if protein_id in requested_ids
        ]

        vector_shape = source["vectors"].shape
        vector_dtype = source["vectors"].dtype
        string_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(output_path, "w") as output:
            output.create_dataset("ids", shape=(len(selected),), dtype=string_dtype)
            vector_kwargs = {}
            if selected:
                vector_kwargs["chunks"] = (min(1024, len(selected)), vector_shape[1])
            output.create_dataset(
                "vectors",
                shape=(len(selected), vector_shape[1]),
                dtype=vector_dtype,
                **vector_kwargs,
            )

            batch_size = 2048
            for start in range(0, len(selected), batch_size):
                batch = selected[start : start + batch_size]
                source_indices = [idx for idx, _ in batch]
                protein_ids = [protein_id for _, protein_id in batch]
                end = start + len(batch)

                output["ids"][start:end] = protein_ids
                output["vectors"][start:end] = source["vectors"][source_indices]

    return {protein_id for _, protein_id in selected}


class SplitWriter:
    def __init__(
        self,
        name: str,
        pairs_path: Path,
        reactions_path: Path,
        available_embeddings: set[str] | None,
        drop_missing_embeddings: bool,
    ) -> None:
        self.name = name
        self.pairs_path = pairs_path
        self.reactions_path = reactions_path
        self.available_embeddings = available_embeddings
        self.drop_missing_embeddings = drop_missing_embeddings

        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        reactions_path.parent.mkdir(parents=True, exist_ok=True)

        self._pairs_fh = pairs_path.open("w", newline="", encoding="utf-8")
        self._rxns_fh = reactions_path.open("w", newline="", encoding="utf-8")
        self._pairs_writer = csv.DictWriter(
            self._pairs_fh, fieldnames=["pr_id", "reaction_id", "protein_id"]
        )
        self._rxns_writer = csv.DictWriter(
            self._rxns_fh, fieldnames=["rs_id", "reaction_id", "reaction_smiles"]
        )
        self._pairs_writer.writeheader()
        self._rxns_writer.writeheader()

        self._seen_reactions: dict[str, str] = {}
        self._seen_pairs: set[tuple[str, str]] = set()
        self._pair_count = 0
        self._reaction_count = 0

        self.rows_seen = 0
        self.positive_rows = 0
        self.duplicate_positive_pairs = 0
        self.skipped_missing_required_columns = 0
        self.skipped_missing_embeddings = 0
        self.positive_proteins: set[str] = set()

    def add_row(self, row: dict[str, str]) -> None:
        self.rows_seen += 1
        if not is_positive(row):
            return

        self.positive_rows += 1
        reaction_smiles = get_reaction_smiles(row)
        protein_id = get_protein_id(row)
        if not reaction_smiles or not protein_id:
            self.skipped_missing_required_columns += 1
            return

        self.positive_proteins.add(protein_id)
        if (
            self.drop_missing_embeddings
            and self.available_embeddings is not None
            and protein_id not in self.available_embeddings
        ):
            self.skipped_missing_embeddings += 1
            return

        reaction_id = self._seen_reactions.get(reaction_smiles)
        if reaction_id is None:
            reaction_id = reaction_id_for(reaction_smiles)
            self._seen_reactions[reaction_smiles] = reaction_id
            self._rxns_writer.writerow(
                {
                    "rs_id": self._reaction_count,
                    "reaction_id": reaction_id,
                    "reaction_smiles": reaction_smiles,
                }
            )
            self._reaction_count += 1

        pair = (reaction_id, protein_id)
        if pair in self._seen_pairs:
            self.duplicate_positive_pairs += 1
            return

        self._seen_pairs.add(pair)
        self._pairs_writer.writerow(
            {
                "pr_id": self._pair_count,
                "reaction_id": reaction_id,
                "protein_id": protein_id,
            }
        )
        self._pair_count += 1

    def close(self) -> None:
        self._pairs_fh.close()
        self._rxns_fh.close()

    def summary(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "rows_seen": self.rows_seen,
            "positive_rows": self.positive_rows,
            "unique_positive_pairs_written": self._pair_count,
            "unique_reactions_written": self._reaction_count,
            "duplicate_positive_pairs": self.duplicate_positive_pairs,
            "skipped_missing_required_columns": self.skipped_missing_required_columns,
            "skipped_missing_embeddings": self.skipped_missing_embeddings,
        }


def record_protein(
    row: dict[str, str],
    requested_proteins: set[str],
    sequences: dict[str, str],
    sequence_conflicts: set[str],
) -> None:
    protein_id = get_protein_id(row)
    if not protein_id:
        return

    requested_proteins.add(protein_id)
    sequence = first_nonempty(row, ("sequence",))
    if not sequence:
        return

    existing = sequences.get(protein_id)
    if existing is None:
        sequences[protein_id] = sequence
    elif existing != sequence:
        sequence_conflicts.add(protein_id)


def scan_csv(
    path: Path,
    requested_proteins: set[str],
    sequences: dict[str, str],
    sequence_conflicts: set[str],
    split_writer: SplitWriter | None = None,
    extra_split_writer: SplitWriter | None = None,
) -> int:
    row_count = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            record_protein(row, requested_proteins, sequences, sequence_conflicts)
            if split_writer is not None:
                split_writer.add_row(row)
            if extra_split_writer is not None:
                extra_split_writer.add_row(row)
    return row_count


def write_protein_files(
    output_dir: Path,
    requested_proteins: set[str],
    sequences: dict[str, str],
    missing_embeddings: set[str],
) -> None:
    sequences_csv = output_dir / "protein_sequences.csv"
    all_fasta = output_dir / "prots.fasta"
    missing_csv = output_dir / "missing_proteins.csv"
    missing_fasta = output_dir / "missing_prots.fasta"

    with sequences_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "sequence"])
        writer.writeheader()
        for protein_id in sorted(requested_proteins):
            writer.writerow({"protein_id": protein_id, "sequence": sequences.get(protein_id, "")})

    with all_fasta.open("w", encoding="utf-8") as handle:
        for protein_id in sorted(requested_proteins):
            sequence = sequences.get(protein_id, "")
            if not sequence:
                continue
            handle.write(f">{protein_id}\n")
            for line in wrap_fasta(sequence):
                handle.write(f"{line}\n")

    with missing_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "has_sequence", "sequence"])
        writer.writeheader()
        for protein_id in sorted(missing_embeddings):
            sequence = sequences.get(protein_id, "")
            writer.writerow(
                {
                    "protein_id": protein_id,
                    "has_sequence": bool(sequence),
                    "sequence": sequence,
                }
            )

    with missing_fasta.open("w", encoding="utf-8") as handle:
        for protein_id in sorted(missing_embeddings):
            sequence = sequences.get(protein_id, "")
            if not sequence:
                continue
            handle.write(f">{protein_id}\n")
            for line in wrap_fasta(sequence):
                handle.write(f"{line}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert data1 candidate-ranking CSVs into Horizyn training files."
    )
    parser.add_argument("--data-dir", default="data1", help="Directory containing data1 CSVs.")
    parser.add_argument(
        "--output-dir",
        default="data/data1_horizyn",
        help="Directory to write converted Horizyn files.",
    )
    parser.add_argument(
        "--protein-embeds-source",
        default="data/sota/prots_t5.h5",
        help="Existing HDF5 protein embeddings to subset. Use '' to skip.",
    )
    parser.add_argument(
        "--protein-embeds-output",
        default=None,
        help="Output HDF5 path. Defaults to <output-dir>/prots_t5.h5.",
    )
    parser.add_argument(
        "--merge-valid-into-train",
        action="store_true",
        help="Also add valid.csv positives to the training CSVs.",
    )
    parser.add_argument(
        "--drop-missing-embeddings",
        action="store_true",
        help="Drop positive train/valid pairs whose protein ID is absent from the source HDF5.",
    )
    return parser.parse_args()


def main() -> None:
    set_large_csv_field_limit()
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_embeddings: set[str] | None = None
    source_h5 = Path(args.protein_embeds_source) if args.protein_embeds_source else None
    if source_h5 is not None:
        if not source_h5.exists():
            raise FileNotFoundError(f"Protein embedding source not found: {source_h5}")
        print(f"Reading embedding IDs from {source_h5} ...")
        available_embeddings = read_h5_ids(source_h5)
        print(f"  Found {len(available_embeddings):,} source embeddings")

    train_writer = SplitWriter(
        name="train",
        pairs_path=output_dir / "train_pairs.csv",
        reactions_path=output_dir / "train_rxns.csv",
        available_embeddings=available_embeddings,
        drop_missing_embeddings=args.drop_missing_embeddings,
    )
    valid_writer = SplitWriter(
        name="valid",
        pairs_path=output_dir / "valid_pairs.csv",
        reactions_path=output_dir / "valid_rxns.csv",
        available_embeddings=available_embeddings,
        drop_missing_embeddings=args.drop_missing_embeddings,
    )

    requested_proteins: set[str] = set()
    sequences: dict[str, str] = {}
    sequence_conflicts: set[str] = set()
    input_rows: dict[str, int] = {}

    try:
        train_path = data_dir / "train.csv"
        valid_path = data_dir / "valid.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing required file: {train_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"Missing required file: {valid_path}")

        print(f"Converting {train_path} ...")
        input_rows["train.csv"] = scan_csv(
            train_path,
            requested_proteins,
            sequences,
            sequence_conflicts,
            split_writer=train_writer,
        )

        print(f"Converting {valid_path} ...")
        input_rows["valid.csv"] = scan_csv(
            valid_path,
            requested_proteins,
            sequences,
            sequence_conflicts,
            split_writer=valid_writer,
            extra_split_writer=train_writer if args.merge_valid_into_train else None,
        )

        for file_name in TEST_FILES:
            test_path = data_dir / file_name
            if not test_path.exists():
                raise FileNotFoundError(f"Missing required file: {test_path}")
            print(f"Scanning proteins in {test_path} ...")
            input_rows[file_name] = scan_csv(
                test_path,
                requested_proteins,
                sequences,
                sequence_conflicts,
            )
    finally:
        train_writer.close()
        valid_writer.close()

    written_embedding_ids: set[str] = set()
    if source_h5 is not None:
        h5_output = Path(args.protein_embeds_output) if args.protein_embeds_output else output_dir / "prots_t5.h5"
        if source_h5.resolve() == h5_output.resolve():
            print(f"Using existing HDF5 embedding file without rewriting: {h5_output}")
            written_embedding_ids = available_embeddings & requested_proteins if available_embeddings else set()
        else:
            print(f"Writing HDF5 embedding subset to {h5_output} ...")
            written_embedding_ids = write_h5_subset(source_h5, h5_output, requested_proteins)
            print(f"  Wrote {len(written_embedding_ids):,} embeddings")

    missing_embeddings = (
        requested_proteins - written_embedding_ids if source_h5 is not None else requested_proteins
    )
    write_protein_files(output_dir, requested_proteins, sequences, missing_embeddings)

    train_missing_positive = (
        train_writer.positive_proteins - written_embedding_ids if source_h5 is not None else set()
    )
    valid_missing_positive = (
        valid_writer.positive_proteins - written_embedding_ids if source_h5 is not None else set()
    )

    summary = {
        "input_rows": input_rows,
        "merge_valid_into_train": args.merge_valid_into_train,
        "drop_missing_embeddings": args.drop_missing_embeddings,
        "splits": {
            "train": train_writer.summary(),
            "valid": valid_writer.summary(),
        },
        "proteins": {
            "requested": len(requested_proteins),
            "with_sequence": len(sequences),
            "sequence_conflicts": len(sequence_conflicts),
            "embeddings_written": len(written_embedding_ids),
            "missing_embeddings": len(missing_embeddings),
            "train_positive_proteins_missing_embeddings": len(train_missing_positive),
            "valid_positive_proteins_missing_embeddings": len(valid_missing_positive),
        },
        "files": {
            "train_pairs": str(output_dir / "train_pairs.csv"),
            "train_reactions": str(output_dir / "train_rxns.csv"),
            "valid_pairs": str(output_dir / "valid_pairs.csv"),
            "valid_reactions": str(output_dir / "valid_rxns.csv"),
            "protein_sequences": str(output_dir / "protein_sequences.csv"),
            "protein_fasta": str(output_dir / "prots.fasta"),
            "missing_proteins": str(output_dir / "missing_proteins.csv"),
            "missing_fasta": str(output_dir / "missing_prots.fasta"),
            "protein_embeddings": str(output_dir / "prots_t5.h5"),
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nPrepared data1 for Horizyn.")
    print(json.dumps(summary, indent=2))
    if missing_embeddings:
        print(
            "\nWARNING: Some data1 proteins do not have embeddings in the source HDF5. "
            "See missing_proteins.csv and missing_prots.fasta. For a fair final test, "
            "provide embeddings for these proteins or rerun with a complete HDF5."
        )
    if train_missing_positive or valid_missing_positive:
        print(
            "\nWARNING: Some positive train/valid pairs reference missing embeddings. "
            "Training will fail unless you provide those embeddings or rerun this script "
            "with --drop-missing-embeddings."
        )


if __name__ == "__main__":
    main()
