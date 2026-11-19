#!/usr/bin/env bash
# run.sh — thread-capped launcher (user args override the default)
set -euo pipefail
THREADS="${THREADS:-4}"
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" \
       TOKENIZERS_PARALLELISM=false
exec python -m subgen --threads "$THREADS" "$@"