#!/usr/bin/env python3
"""
Create a data1 protein embedding HDF5 with missing proteins filled by ProtT5.

This script scans all data1 CSV files, compares the protein IDs against an
existing Horizyn protein embedding HDF5 file, encodes missing proteins from the
CSV "sequence" column with ProtT5, and writes a new HDF5 containing embeddings
for every data1 protein it can cover.

Output format matches Horizyn's EmbedDataset:
    ids:     string identifiers
    vectors: float embeddings with shape [N, 1024]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "2")


DATA1_FILES = (
    "train.csv",
    "valid.csv",
    "Enzyme-405.csv",
    "Orphan-335_retrievel_cands.csv",
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")


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
    # Orphan candidates have "enzyme"; other files have "UniprotID".
    return first_nonempty(row, ("enzyme", "UniprotID"))


def clean_sequence(sequence: str) -> str:
    """Clean protein sequence for ProtT5 tokenization."""
    sequence = re.sub(r"\s+", "", sequence).upper()
    sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    return "".join(aa if aa in VALID_AA else "X" for aa in sequence)


def scan_data1_sequences(data_dir: Path, files: Iterable[str]) -> tuple[dict[str, str], dict]:
    sequences: dict[str, str] = {}
    requested_ids: set[str] = set()
    conflicts: set[str] = set()
    rows_by_file: dict[str, int] = {}
    missing_sequence_ids: set[str] = set()

    for file_name in files:
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing data1 file: {path}")

        rows = 0
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows += 1
                protein_id = get_protein_id(row)
                if not protein_id:
                    continue

                requested_ids.add(protein_id)
                raw_sequence = first_nonempty(row, ("sequence",))
                if not raw_sequence:
                    missing_sequence_ids.add(protein_id)
                    continue

                sequence = clean_sequence(raw_sequence)
                if not sequence:
                    missing_sequence_ids.add(protein_id)
                    continue

                existing = sequences.get(protein_id)
                if existing is None:
                    sequences[protein_id] = sequence
                elif existing != sequence:
                    conflicts.add(protein_id)
                    # Keep the first sequence so the run is deterministic.

        rows_by_file[file_name] = rows

    summary = {
        "rows_by_file": rows_by_file,
        "requested_proteins": len(requested_ids),
        "proteins_with_sequence": len(sequences),
        "proteins_missing_sequence": len(requested_ids - set(sequences)),
        "sequence_conflicts": len(conflicts),
        "requested_ids": requested_ids,
        "missing_sequence_ids": requested_ids - set(sequences),
        "conflict_ids": conflicts,
    }
    return sequences, summary


def read_h5_index(path: Path) -> tuple[list[str], dict[str, int], tuple[int, int], str]:
    import h5py

    with h5py.File(path, "r") as handle:
        raw_ids = handle["ids"][:]
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_ids
        ]
        shape = tuple(handle["vectors"].shape)
        dtype = str(handle["vectors"].dtype)

    return ids, {protein_id: idx for idx, protein_id in enumerate(ids)}, shape, dtype


def chunk_sequence(sequence: str, max_residues: int, strategy: str) -> list[str]:
    if len(sequence) <= max_residues:
        return [sequence]
    if strategy == "error":
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds --max-residues {max_residues}"
        )
    if strategy == "truncate":
        return [sequence[:max_residues]]
    return [sequence[start : start + max_residues] for start in range(0, len(sequence), max_residues)]


def load_prott5(
    model_name: str,
    device_name: str,
    cache_dir: str | None,
    local_files_only: bool,
    use_half: bool,
    use_safetensors: bool,
    load_retries: int,
    retry_sleep: int,
):
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" and use_half else torch.float32

    last_error = None
    attempts = max(1, load_retries)
    for attempt in range(1, attempts + 1):
        try:
            tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            model = T5EncoderModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_safetensors=use_safetensors,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            print(
                f"WARNING: failed to load/download ProtT5 model "
                f"(attempt {attempt}/{attempts}): {exc}"
            )
            print(f"Sleeping {retry_sleep} seconds before retrying...")
            time.sleep(retry_sleep)
    else:
        raise RuntimeError("Failed to load ProtT5 model") from last_error

    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_missing_sequences(
    sequences: dict[str, str],
    missing_ids: list[str],
    model_name: str,
    device_name: str,
    cache_dir: str | None,
    local_files_only: bool,
    batch_size: int,
    max_residues: int,
    long_sequence_strategy: str,
    use_half: bool,
    use_safetensors: bool,
    load_retries: int,
    retry_sleep: int,
) -> dict[str, object]:
    import torch

    if not missing_ids:
        return {}

    print(f"Loading ProtT5 model: {model_name}")
    tokenizer, model, device = load_prott5(
        model_name=model_name,
        device_name=device_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_half=use_half,
        use_safetensors=use_safetensors,
        load_retries=load_retries,
        retry_sleep=retry_sleep,
    )

    chunks: list[tuple[str, str]] = []
    for protein_id in missing_ids:
        for chunk in chunk_sequence(sequences[protein_id], max_residues, long_sequence_strategy):
            chunks.append((protein_id, chunk))

    chunks.sort(key=lambda item: len(item[1]))
    print(f"Encoding {len(missing_ids):,} proteins as {len(chunks):,} sequence chunks")

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            protein_ids = [protein_id for protein_id, _ in batch]
            chunk_sequences = [chunk for _, chunk in batch]
            tokenized_sequences = [" ".join(chunk) for chunk in chunk_sequences]

            tokens = tokenizer(
                tokenized_sequences,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs = model(**tokens).last_hidden_state

            for idx, (protein_id, chunk) in enumerate(zip(protein_ids, chunk_sequences)):
                # ProtT5 tokenization maps each spaced residue to one token, followed by EOS.
                residue_count = len(chunk)
                residue_embeddings = outputs[idx, :residue_count, :].detach().float().cpu()
                if protein_id not in sums:
                    sums[protein_id] = residue_embeddings.sum(dim=0)
                    counts[protein_id] = residue_count
                else:
                    sums[protein_id] += residue_embeddings.sum(dim=0)
                    counts[protein_id] += residue_count

            done = min(start + batch_size, len(chunks))
            print(f"  Encoded chunks: {done:,}/{len(chunks):,}")

    embeddings = {
        protein_id: (sums[protein_id] / counts[protein_id]).numpy()
        for protein_id in missing_ids
    }
    return embeddings


def write_merged_h5(
    source_path: Path,
    output_path: Path,
    requested_ids: set[str],
    source_ids: list[str],
    source_index: dict[str, int],
    new_embeddings: dict[str, object],
    overwrite: bool,
    metadata: dict,
) -> None:
    import h5py
    import numpy as np

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    copied_ids = [protein_id for protein_id in source_ids if protein_id in requested_ids]
    new_ids = sorted(new_embeddings)
    all_ids = copied_ids + new_ids

    vector_dim = None
    vector_dtype = None
    with h5py.File(source_path, "r") as source:
        vector_dim = source["vectors"].shape[1]
        vector_dtype = source["vectors"].dtype

    if new_embeddings:
        first_new = next(iter(new_embeddings.values()))
        if first_new.shape[0] != vector_dim:
            raise ValueError(
                f"New embedding dim {first_new.shape[0]} does not match source dim {vector_dim}"
            )

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temp_path, "w") as output:
        output.create_dataset("ids", shape=(len(all_ids),), dtype=string_dtype)

        vector_kwargs = {}
        if all_ids:
            vector_kwargs["chunks"] = (min(1024, len(all_ids)), vector_dim)
        output.create_dataset(
            "vectors",
            shape=(len(all_ids), vector_dim),
            dtype=vector_dtype,
            **vector_kwargs,
        )
        output["ids"][:] = all_ids

        row = 0
        with h5py.File(source_path, "r") as source:
            batch_size = 2048
            for start in range(0, len(copied_ids), batch_size):
                batch_ids = copied_ids[start : start + batch_size]
                source_rows = [source_index[protein_id] for protein_id in batch_ids]
                end = row + len(batch_ids)
                output["vectors"][row:end] = source["vectors"][source_rows]
                row = end
                print(f"  Copied source embeddings: {row:,}/{len(copied_ids):,}")

        for start in range(0, len(new_ids), 2048):
            batch_ids = new_ids[start : start + 2048]
            vectors = np.stack([new_embeddings[protein_id] for protein_id in batch_ids]).astype(
                vector_dtype, copy=False
            )
            end = row + len(batch_ids)
            output["vectors"][row:end] = vectors
            row = end
            print(f"  Wrote new embeddings: {start + len(batch_ids):,}/{len(new_ids):,}")

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                output.attrs[key] = value
            else:
                output.attrs[key] = json.dumps(value, sort_keys=True)

    temp_path.replace(output_path)


def write_reports(output_path: Path, report: dict) -> None:
    report_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    json_report = {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in report.items()
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(json_report, handle, indent=2)

    missing_sequence = report.get("missing_sequence_ids", set())
    if missing_sequence:
        missing_path = output_path.with_suffix(output_path.suffix + ".missing_sequences.txt")
        missing_path.write_text("\n".join(sorted(missing_sequence)) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill data1 proteins missing from an existing Horizyn ProtT5 HDF5."
    )
    parser.add_argument("--data-dir", default="data1", help="Directory containing data1 CSV files.")
    parser.add_argument(
        "--source-h5",
        default="data/sota/prots_t5.h5",
        help="Existing Horizyn protein embedding HDF5.",
    )
    parser.add_argument(
        "--output-h5",
        default="data/data1_horizyn/prots_t5.h5",
        help="Output HDF5 containing all data1 proteins.",
    )
    parser.add_argument(
        "--model-name",
        default="Rostlab/prot_t5_xl_uniref50",
        help="Hugging Face model used to encode missing proteins.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Encoding device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="ProtT5 chunk batch size.")
    parser.add_argument(
        "--max-residues",
        type=int,
        default=1022,
        help="Maximum residues per ProtT5 chunk before splitting/truncating.",
    )
    parser.add_argument(
        "--long-sequence-strategy",
        choices=["chunk", "truncate", "error"],
        default="chunk",
        help="How to handle proteins longer than --max-residues.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for the ProtT5 model.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download the model; only use local Hugging Face cache.",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable float16 model weights on CUDA.",
    )
    parser.add_argument(
        "--use-safetensors",
        action="store_true",
        help=(
            "Prefer safetensors weights. Disabled by default because some "
            "transformers versions start a huge auto-conversion download for ProtT5."
        ),
    )
    parser.add_argument(
        "--load-retries",
        type=int,
        default=5,
        help="How many times to retry loading/downloading the ProtT5 model.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=int,
        default=60,
        help="Seconds to wait between model download/load retries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-h5 if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    set_large_csv_field_limit()
    args = parse_args()

    data_dir = Path(args.data_dir)
    source_h5 = Path(args.source_h5)
    output_h5 = Path(args.output_h5)

    if not source_h5.exists():
        raise FileNotFoundError(f"Source HDF5 not found: {source_h5}")

    print(f"Scanning data1 CSVs in {data_dir}")
    sequences, scan_summary = scan_data1_sequences(data_dir, DATA1_FILES)
    requested_ids = scan_summary["requested_ids"]
    print(f"  Requested proteins: {len(requested_ids):,}")
    print(f"  Proteins with sequences: {len(sequences):,}")

    print(f"Reading source embedding index from {source_h5}")
    source_ids, source_index, source_shape, source_dtype = read_h5_index(source_h5)
    source_id_set = set(source_ids)
    present_ids = requested_ids & source_id_set
    missing_ids = sorted(requested_ids - source_id_set)
    encodable_missing_ids = [protein_id for protein_id in missing_ids if protein_id in sequences]
    unencodable_missing_ids = set(missing_ids) - set(encodable_missing_ids)

    print(f"  Source embeddings: {len(source_ids):,} vectors, shape={source_shape}, dtype={source_dtype}")
    print(f"  Present data1 proteins in source: {len(present_ids):,}")
    print(f"  Missing data1 proteins: {len(missing_ids):,}")
    print(f"  Missing proteins with sequence: {len(encodable_missing_ids):,}")

    if unencodable_missing_ids:
        print(
            "WARNING: Some missing proteins have no sequence and cannot be encoded. "
            "They will be listed in the summary report."
        )

    device_name = args.device
    if device_name == "auto":
        import torch

        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    new_embeddings = encode_missing_sequences(
        sequences=sequences,
        missing_ids=encodable_missing_ids,
        model_name=args.model_name,
        device_name=device_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
        max_residues=args.max_residues,
        long_sequence_strategy=args.long_sequence_strategy,
        use_half=not args.no_half,
        use_safetensors=args.use_safetensors,
        load_retries=args.load_retries,
        retry_sleep=args.retry_sleep,
    )

    metadata = {
        "source_h5": str(source_h5),
        "model_name": args.model_name,
        "pooling": "mean over residue encoder states; long proteins are residue-weighted over chunks",
        "max_residues": args.max_residues,
        "long_sequence_strategy": args.long_sequence_strategy,
        "device": device_name,
        "copied_from_source": len(present_ids),
        "encoded_missing": len(new_embeddings),
    }

    print(f"Writing merged data1 HDF5 to {output_h5}")
    write_merged_h5(
        source_path=source_h5,
        output_path=output_h5,
        requested_ids=requested_ids,
        source_ids=source_ids,
        source_index=source_index,
        new_embeddings=new_embeddings,
        overwrite=args.overwrite,
        metadata=metadata,
    )

    report = {
        "output_h5": str(output_h5),
        "source_h5": str(source_h5),
        "requested_proteins": len(requested_ids),
        "copied_from_source": len(present_ids),
        "missing_in_source": len(missing_ids),
        "encoded_missing": len(new_embeddings),
        "unencodable_missing": len(unencodable_missing_ids),
        "missing_sequence_ids": unencodable_missing_ids | scan_summary["missing_sequence_ids"],
        "sequence_conflict_ids": scan_summary["conflict_ids"],
        "rows_by_file": scan_summary["rows_by_file"],
        "model_name": args.model_name,
        "max_residues": args.max_residues,
        "long_sequence_strategy": args.long_sequence_strategy,
    }
    write_reports(output_h5, report)

    print("\nProtein embedding completion finished.")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, set)}, indent=2))
    if unencodable_missing_ids:
        raise SystemExit(
            "Some requested proteins are still missing because no sequence was available. "
            "See the .missing_sequences.txt report."
        )


if __name__ == "__main__":
    main()
