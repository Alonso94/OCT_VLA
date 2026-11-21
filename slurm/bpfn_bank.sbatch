#!/usr/bin/env bash
# BimanualPFN: pre-generate the training episode bank once (CPU work; the A40 is only
# the partition's admission ticket). Shards are deterministic per unit and written
# atomically, so a rerun after a timeout builds only the missing ones.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
set -euo pipefail
: "${OCTVLA_REPO:?}"; : "${OCTVLA_POLICY_PYTHON:?}"; : "${BPFN_BANK:?bank directory}"
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun "$OCTVLA_POLICY_PYTHON" -u "$OCTVLA_REPO/scripts/bimanual_pfn.py" bank \
  --out "$BPFN_BANK" --name "${BPFN_BANK_NAME:-v1}" --shards "${BPFN_BANK_SHARDS:-32}" \
  --pool "${BPFN_BANK_POOL:-24}" --conflict "${BPFN_BANK_CONFLICT:-24}" \
  --handover "${BPFN_BANK_HANDOVER:-48}" --workers "${SLURM_CPUS_PER_TASK:-16}"
