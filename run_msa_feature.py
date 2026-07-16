#!/usr/bin/env python3
"""Extract full-sequence and pocket MSA Transformer representations."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_ID_COLUMN = "UniprotID"
DEFAULT_SEQUENCE_COLUMN = "sequence"
MSA_DIMENSION = 768
torch = None


def require_torch():
    """Import PyTorch only when feature extraction is requested."""
    global torch
    if torch is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for MSA feature extraction") from exc
        torch = torch_module
    return torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode per-protein A3M alignments with MSA Transformer and write "
            "full-sequence and optional pocket-level features."
        )
    )
    parser.add_argument(
        "--input-csv",
        "--data_path",
        dest="input_csv",
        required=True,
        help="CSV containing protein identifiers and amino-acid sequences.",
    )
    parser.add_argument(
        "--msa-dir",
        "--msa_dir",
        dest="msa_dir",
        required=True,
        help="Directory containing one <protein_id>.a3m file per protein.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for MSA Transformer features.",
    )
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN)
    parser.add_argument("--sequence-column", default=DEFAULT_SEQUENCE_COLUMN)
    parser.add_argument(
        "--pocket-info",
        default=None,
        help=(
            "Optional CSV containing protein identifiers and one-based pocket "
            "residue indices."
        ),
    )
    parser.add_argument(
        "--pocket-id-column",
        default=None,
        help="Pocket CSV identifier column; defaults to --id-column.",
    )
    parser.add_argument("--pocket-residues-column", default="pocket_residues")
    parser.add_argument("--max-msa-sequences", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-msa-sequences", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute residue features instead of reusing existing NPZ files.",
    )
    args = parser.parse_args()

    if args.max_msa_sequences < 1:
        parser.error("--max-msa-sequences must be positive")
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.min_msa_sequences < 1:
        parser.error("--min-msa-sequences must be positive")
    return args


def validate_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def validate_identifier(identifier: str) -> str:
    identifier = identifier.strip()
    if not identifier:
        raise ValueError("Protein identifiers must not be empty")
    if identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
        raise ValueError(
            f"Protein identifier {identifier!r} cannot be used as an A3M filename"
        )
    return identifier


def load_proteins(
    input_csv: Path,
    id_column: str,
    sequence_column: str,
) -> list[tuple[str, str]]:
    validate_file(input_csv, "Input CSV")
    frame = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    missing = {id_column, sequence_column} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    proteins: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for raw_id, raw_sequence in frame[[id_column, sequence_column]].itertuples(
        index=False, name=None
    ):
        protein_id = validate_identifier(str(raw_id))
        sequence = re.sub(r"\s+", "", str(raw_sequence)).upper()
        if not sequence:
            raise ValueError(f"Sequence is empty for protein {protein_id!r}")
        previous = seen.get(protein_id)
        if previous is not None and previous != sequence:
            raise ValueError(f"Conflicting sequences found for protein {protein_id!r}")
        if previous is None:
            seen[protein_id] = sequence
            proteins.append((protein_id, sequence))
    return proteins


def read_a3m(path: Path) -> list[str]:
    """Read aligned sequences from an A3M file."""
    sequences: list[str] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def remove_a3m_insertions(sequence: str) -> str:
    """Remove lowercase insertion residues and insertion-gap markers."""
    return "".join(character for character in sequence if not character.islower() and character != ".")


def prepare_alignment(
    raw_sequences: Iterable[str],
    max_depth: int,
    max_length: int,
) -> list[str]:
    cleaned = [remove_a3m_insertions(sequence) for sequence in raw_sequences]
    cleaned = [sequence for sequence in cleaned if sequence]
    if not cleaned:
        return []

    query_length = len(cleaned[0])
    aligned = [sequence for sequence in cleaned if len(sequence) == query_length]
    return [sequence[:max_length] for sequence in aligned[:max_depth]]


def resolve_device(requested: str) -> torch.device:
    torch_module = require_torch()
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return torch_module.device(requested)


def load_msa_transformer(device: torch.device):
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "MSA Transformer requires fair-esm. Install it with: pip install fair-esm"
        ) from exc

    if not hasattr(esm, "pretrained") or not hasattr(
        esm.pretrained, "esm_msa1b_t12_100M_UR50S"
    ):
        raise RuntimeError(
            "The imported 'esm' package does not provide MSA Transformer. "
            "Use a separate environment with fair-esm installed."
        )

    print(f"Loading MSA Transformer on {device}...")
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model = model.eval().to(device)
    return model, alphabet


def encode_alignment(
    protein_id: str,
    alignment: list[str],
    model,
    alphabet,
    device: torch.device,
) -> np.ndarray:
    batch_converter = alphabet.get_batch_converter()
    msa_input = [(f"{protein_id}_{index}", sequence) for index, sequence in enumerate(alignment)]
    _, _, batch_tokens = batch_converter([msa_input])
    batch_tokens = batch_tokens.to(device)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        representations = model(
            batch_tokens,
            repr_layers=[12],
            return_contacts=False,
        )["representations"][12]

    query_tokens = batch_tokens[0, 0]
    token_count = int((query_tokens != alphabet.padding_idx).sum().item())
    start = 1 if alphabet.prepend_bos else 0
    end = token_count - (1 if alphabet.append_eos else 0)
    node_feature = representations[0, 0, start:end].float().cpu().numpy()
    if node_feature.ndim != 2 or node_feature.shape[1] != MSA_DIMENSION:
        raise ValueError(
            f"Unexpected MSA representation shape for {protein_id}: {node_feature.shape}"
        )
    return node_feature.astype(np.float32, copy=False)


def write_pickle_atomic(data: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(output_path)


def extract_sequence_features(
    proteins: list[tuple[str, str]],
    msa_dir: Path,
    node_dir: Path,
    mean_output: Path,
    max_depth: int,
    max_length: int,
    min_depth: int,
    device: torch.device,
    overwrite: bool,
) -> dict[str, object]:
    if not msa_dir.is_dir():
        raise FileNotFoundError(f"MSA directory not found: {msa_dir}")
    node_dir.mkdir(parents=True, exist_ok=True)

    sequence_to_feature: dict[str, np.ndarray] = {}
    if mean_output.is_file() and not overwrite:
        with mean_output.open("rb") as handle:
            existing = pickle.load(handle)
        if not isinstance(existing, dict):
            raise TypeError(f"Expected a dictionary in {mean_output}")
        sequence_to_feature.update(existing)

    counters = {
        "proteins": len(proteins),
        "encoded": 0,
        "reused": 0,
        "missing_a3m": 0,
        "shallow_or_invalid_a3m": 0,
        "failed": 0,
    }
    missing_ids: list[str] = []
    failed_ids: list[str] = []
    model = alphabet = None

    for protein_id, sequence in tqdm(proteins, desc="Encoding MSAs"):
        node_path = node_dir / f"{protein_id}.npz"
        try:
            if node_path.is_file() and not overwrite:
                node_feature = np.load(node_path)["node_feature"].astype(np.float32, copy=False)
                if node_feature.ndim != 2 or node_feature.shape[1] != MSA_DIMENSION:
                    raise ValueError(
                        f"Unexpected cached representation shape: {node_feature.shape}"
                    )
                counters["reused"] += 1
            else:
                msa_path = msa_dir / f"{protein_id}.a3m"
                if not msa_path.is_file():
                    counters["missing_a3m"] += 1
                    missing_ids.append(protein_id)
                    continue
                alignment = prepare_alignment(
                    read_a3m(msa_path),
                    max_depth=max_depth,
                    max_length=max_length,
                )
                if len(alignment) < min_depth:
                    counters["shallow_or_invalid_a3m"] += 1
                    failed_ids.append(protein_id)
                    continue
                if model is None:
                    model, alphabet = load_msa_transformer(device)
                node_feature = encode_alignment(
                    protein_id,
                    alignment,
                    model,
                    alphabet,
                    device,
                )
                np.savez_compressed(node_path, node_feature=node_feature)
                counters["encoded"] += 1

            if node_feature.shape[0] == 0:
                raise ValueError("Residue representation is empty")
            sequence_to_feature[sequence] = node_feature.mean(axis=0).astype(np.float32)
        except Exception as exc:
            counters["failed"] += 1
            failed_ids.append(protein_id)
            print(f"[ERROR] {protein_id}: {exc}")

    write_pickle_atomic(sequence_to_feature, mean_output)
    counters["full_sequence_features"] = len(sequence_to_feature)
    counters["missing_ids"] = missing_ids
    counters["failed_ids"] = sorted(set(failed_ids))
    return counters


def parse_pocket_indices(value: str) -> list[int]:
    return [int(token) - 1 for token in re.findall(r"\d+", value)]


def extract_pocket_features(
    pocket_info: Path,
    node_dir: Path,
    output_path: Path,
    id_column: str,
    residues_column: str,
) -> dict[str, int]:
    torch_module = require_torch()
    validate_file(pocket_info, "Pocket information CSV")
    frame = pd.read_csv(pocket_info, dtype=str, keep_default_na=False)
    missing = {id_column, residues_column} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Pocket CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    features: dict[str, object] = {}
    skipped = 0
    for raw_id, raw_indices in tqdm(
        frame[[id_column, residues_column]].itertuples(index=False, name=None),
        total=len(frame),
        desc="Extracting pocket features",
    ):
        protein_id = validate_identifier(str(raw_id))
        node_path = node_dir / f"{protein_id}.npz"
        indices = parse_pocket_indices(str(raw_indices))
        if not node_path.is_file() or not indices:
            skipped += 1
            continue
        node_feature = np.load(node_path)["node_feature"].astype(np.float32, copy=False)
        valid_indices = [index for index in indices if 0 <= index < node_feature.shape[0]]
        if not valid_indices:
            skipped += 1
            continue
        features[protein_id] = torch_module.from_numpy(
            node_feature[valid_indices].copy()
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(features, output_path)
    return {"pocket_features": len(features), "pocket_entries_skipped": skipped}


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    msa_dir = Path(args.msa_dir)
    output_dir = Path(args.output_dir)
    node_dir = output_dir / "node_level"
    mean_output = output_dir / "protein_level" / "seq2feature.pkl"
    pocket_output = output_dir / "pocket_node_feature" / "msa_node_feature.pt"

    proteins = load_proteins(input_csv, args.id_column, args.sequence_column)
    device = resolve_device(args.device)
    summary = extract_sequence_features(
        proteins=proteins,
        msa_dir=msa_dir,
        node_dir=node_dir,
        mean_output=mean_output,
        max_depth=args.max_msa_sequences,
        max_length=args.max_length,
        min_depth=args.min_msa_sequences,
        device=device,
        overwrite=args.overwrite,
    )

    if args.pocket_info:
        summary.update(
            extract_pocket_features(
                pocket_info=Path(args.pocket_info),
                node_dir=node_dir,
                output_path=pocket_output,
                id_column=args.pocket_id_column or args.id_column,
                residues_column=args.pocket_residues_column,
            )
        )

    summary.update(
        {
            "input_csv": str(input_csv),
            "msa_dir": str(msa_dir),
            "device": str(device),
            "mean_output": str(mean_output),
            "pocket_output": str(pocket_output) if args.pocket_info else None,
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
