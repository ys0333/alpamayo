#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python3"
elif [[ -x "/home/jys/a1_5_venv/bin/python3" ]]; then
  PYTHON_BIN="/home/jys/a1_5_venv/bin/python3"
else
  PYTHON_BIN="$ROOT_DIR/a1_5_venv/bin/python3"
fi
INFER_SCRIPT="$ROOT_DIR/src/alpamayo1_5/test_inference.py"

NUMS="${NUMS:-20}"
CLIP_ID="${CLIP_ID:-030c760c-ae38-49aa-9ad8-f5650a545d26}"
T0_US="${T0_US:-5100000}"
T0_STEP_US="${T0_STEP_US:-100000}"
SELECT_PREFIX="${SELECT_PREFIX:-}"

LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-35}"
HEAD_START="${HEAD_START:-0}"
HEAD_END="${HEAD_END:-15}"
DIM_START="${DIM_START:-0}"
DIM_END="${DIM_END:-127}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/codex_history/dim_ablation_sweep_${TIMESTAMP}"
RAW_DIR="$OUT_DIR/raw"
SUMMARY_CSV="$OUT_DIR/summary.csv"

mkdir -p "$RAW_DIR"

cat > "$SUMMARY_CSV" <<'EOF'
layer,head,dim,select_prefix,mean_minade_m,mean_e2e_ms,mean_diffusion_ms,mean_ratio_pct,status
EOF

echo "Writing raw logs to: $RAW_DIR"
echo "Writing summary to:  $SUMMARY_CSV"
echo "WARNING: full dim sweep is extremely large. Narrow ranges with LAYER_START/END, HEAD_START/END, DIM_START/END when iterating."

for layer in $(seq "$LAYER_START" "$LAYER_END"); do
  for head in $(seq "$HEAD_START" "$HEAD_END"); do
    for dim in $(seq "$DIM_START" "$DIM_END"); do
      raw_log="$RAW_DIR/layer_${layer}_head_${head}_dim_${dim}.log"
      cmd=(
        "$PYTHON_BIN" "$INFER_SCRIPT"
        "--clip-id" "$CLIP_ID"
        "--t0-us" "$T0_US"
        "--t0-step-us" "$T0_STEP_US"
        "--nums" "$NUMS"
        "--ablate-dim" "${layer}:${head}:${dim}"
      )
      if [[ -n "$SELECT_PREFIX" ]]; then
        cmd+=("--select-prefix" "$SELECT_PREFIX")
      fi

      echo
      echo "=== Running dim ablation layer=$layer head=$head dim=$dim ==="
      {
        printf 'COMMAND:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
      } | tee "$raw_log"

      if "${cmd[@]}" | tee -a "$raw_log"; then
        summary_line="$(grep 'Summary over ' "$raw_log" | tail -n 1 || true)"
        if [[ -n "$summary_line" ]]; then
          mean_minade="$(printf '%s\n' "$summary_line" | sed -n 's/.*mean minADE=\([0-9.]*\)m.*/\1/p')"
          mean_e2e="$(printf '%s\n' "$summary_line" | sed -n 's/.*mean e2e latency=\([0-9.]*\) ms.*/\1/p')"
          mean_diffusion="$(printf '%s\n' "$summary_line" | sed -n 's/.*mean diffusion latency=\([0-9.]*\) ms.*/\1/p')"
          mean_ratio="$(printf '%s\n' "$summary_line" | sed -n 's/.*mean diffusion \/ e2e ratio=\([0-9.]*\)%.*/\1/p')"
          printf '%s,%s,%s,"%s",%s,%s,%s,%s,%s\n' \
            "$layer" "$head" "$dim" "$SELECT_PREFIX" "${mean_minade:-}" "${mean_e2e:-}" \
            "${mean_diffusion:-}" "${mean_ratio:-}" "ok" >> "$SUMMARY_CSV"
        else
          printf '%s,%s,%s,"%s",,,,,%s\n' \
            "$layer" "$head" "$dim" "$SELECT_PREFIX" "missing_summary" >> "$SUMMARY_CSV"
        fi
      else
        printf '%s,%s,%s,"%s",,,,,%s\n' \
          "$layer" "$head" "$dim" "$SELECT_PREFIX" "failed" >> "$SUMMARY_CSV"
      fi
    done
  done
done

echo
echo "Dim ablation sweep complete."
echo "Summary:"
cat "$SUMMARY_CSV"
