#!/usr/bin/env bash
set -euo pipefail

# Configuration. For parallel jobs, assign distinct fastaX/msaX/workX directories.
FASTA_DIR="fasta1"
OUT_DIR="msa1"
WORK_ROOT="work1"

UNIREF_DB="uniref90_pad"

THREADS=6
MAX_SEQS=1000

# Maximum number of successfully written A3M files in this job.
BATCH=2000

USE_GPU=1

# Preserve CUDA_VISIBLE_DEVICES when it is supplied by the job scheduler.
: "${CUDA_VISIBLE_DEVICES:=1}"

mkdir -p "$OUT_DIR" "$WORK_ROOT"

shopt -s nullglob
fasta_files=("$FASTA_DIR"/*.fasta)
if [[ "${#fasta_files[@]}" -eq 0 ]]; then
  echo "ERROR: No FASTA files found in $FASTA_DIR" >&2
  exit 1
fi

cleanup_one() {
  local workdir="$1"
  [[ "$workdir" == "$WORK_ROOT/"* ]] && rm -rf "$workdir"
}

done_count=0

for fasta in "${fasta_files[@]}"; do
  id=$(basename "$fasta" .fasta)
  out_a3m="$OUT_DIR/${id}.a3m"

  # Skip an existing non-empty output without counting it toward this batch.
  if [[ -s "$out_a3m" ]]; then
    continue
  fi

  echo "===== Processing $id ====="

  workdir="$WORK_ROOT/$id"
  mkdir -p "$workdir"
  trap 'echo "[ERROR] failed on '"$id"'"; cleanup_one "'"$workdir"'"; exit 1' ERR

  QUERY_DB="$workdir/query"
  RESULT_DB="$workdir/result"
  MSA_DB="$workdir/msa"
  TMP_DIR="$workdir/tmp"
  UNPACK_DIR="$workdir/unpack"

  mkdir -p "$TMP_DIR" "$UNPACK_DIR"

  # 1) Create the single-query MMseqs2 database on CPU.
  mmseqs createdb "$fasta" "$QUERY_DB"

  # 2) Search UniRef90 on GPU when enabled; otherwise use CPU.
  if [[ "$USE_GPU" -eq 1 ]]; then
    mmseqs search \
      "$QUERY_DB" \
      "$UNIREF_DB" \
      "$RESULT_DB" \
      "$TMP_DIR" \
      --alignment-mode 3 \
      --max-seqs "$MAX_SEQS" \
      --gpu 1 \
      --gpu-server 0 \
      --threads "$THREADS"
  else
    mmseqs search \
      "$QUERY_DB" \
      "$UNIREF_DB" \
      "$RESULT_DB" \
      "$TMP_DIR" \
      --alignment-mode 3 \
      --max-seqs "$MAX_SEQS" \
      --threads "$THREADS"
  fi

  # 3) Convert the search result to an A3M database on CPU.
  mmseqs result2msa \
    "$QUERY_DB" \
    "$UNIREF_DB" \
    "$RESULT_DB" \
    "$MSA_DB" \
    --msa-format-mode 5 \
    --threads "$THREADS"

  # 4) Unpack the A3M database into an isolated temporary directory.
  mmseqs unpackdb \
    "$MSA_DB" \
    "$UNPACK_DIR" \
    --unpack-suffix a3m \
    --unpack-name-mode 1 \
    --threads "$THREADS"

  # 5) Locate the unpacked file and rename it to <FASTA filename stem>.a3m.
  a3m_file=""
  if compgen -G "$UNPACK_DIR/*a3m" > /dev/null; then
    a3m_file=$(ls -1 "$UNPACK_DIR"/*a3m | head -n 1)
  fi

  if [[ -z "$a3m_file" || ! -s "$a3m_file" ]]; then
    echo "WARNING: No a3m produced for $id"
  else
    mv "$a3m_file" "$out_a3m"
    echo "Wrote: $out_a3m"
    done_count=$((done_count+1))
  fi

  # 6) Remove temporary files for this query.
  cleanup_one "$workdir"
  trap - ERR

  echo "===== Done $id ====="
  echo

  # Stop when this job reaches its successful-output limit.
  if [[ "$done_count" -ge "$BATCH" ]]; then
    echo "[BATCH] reached $BATCH outputs, exit."
    break
  fi
done
