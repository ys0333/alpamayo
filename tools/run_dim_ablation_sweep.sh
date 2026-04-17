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
SELECT_PREFIX="${SELECT_PREFIX:-0,1,4,5}"
TOP_HEADS="${TOP_HEADS:-8}"
HEAD_SUMMARY_PATH="${HEAD_SUMMARY_PATH:-}"
DIM_START="${DIM_START:-0}"
DIM_END="${DIM_END:-127}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/codex_history/dim_ablation_sweep_${TIMESTAMP}"
RAW_DIR="$OUT_DIR/raw"
SUMMARY_CSV="$OUT_DIR/summary.csv"

mkdir -p "$RAW_DIR"

cat > "$SUMMARY_CSV" <<'EOF'
case_name,layer,head,dim,select_prefix,mean_minade_m,delta_vs_baseline_m,mean_e2e_ms,mean_diffusion_ms,mean_ratio_pct,status
EOF

find_latest_head_summary() {
  ls -dt "$ROOT_DIR"/codex_history/head_ablation_sweep_* 2>/dev/null | head -n 1
}

if [[ -z "$HEAD_SUMMARY_PATH" ]]; then
  latest_head_dir="$(find_latest_head_summary || true)"
  if [[ -z "$latest_head_dir" ]]; then
    echo "No head ablation summary found. Run tools/run_head_ablation_sweep.sh first or set HEAD_SUMMARY_PATH." >&2
    exit 1
  fi
  HEAD_SUMMARY_PATH="$latest_head_dir/summary.csv"
fi

if [[ ! -f "$HEAD_SUMMARY_PATH" ]]; then
  echo "HEAD_SUMMARY_PATH does not exist: $HEAD_SUMMARY_PATH" >&2
  exit 1
fi

run_case() {
  local ablate_dim_spec="$1"
  local raw_log="$2"

  local -a cmd=(
    "$PYTHON_BIN" "$INFER_SCRIPT"
    "--clip-id" "$CLIP_ID"
    "--t0-us" "$T0_US"
    "--t0-step-us" "$T0_STEP_US"
    "--nums" "$NUMS"
  )
  if [[ -n "$SELECT_PREFIX" ]]; then
    cmd+=("--select-prefix" "$SELECT_PREFIX")
  fi
  if [[ -n "$ablate_dim_spec" ]]; then
    cmd+=("--ablate-dim" "$ablate_dim_spec")
  fi

  {
    printf 'COMMAND:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } | tee "$raw_log"

  if "${cmd[@]}" | tee -a "$raw_log"; then
    grep 'Summary over ' "$raw_log" | tail -n 1 || true
  else
    return 1
  fi
}

parse_metric() {
  local summary_line="$1"
  local sed_expr="$2"
  printf '%s\n' "$summary_line" | sed -n "$sed_expr"
}

echo "Writing raw logs to: $RAW_DIR"
echo "Writing summary to:  $SUMMARY_CSV"
echo "Using head summary:  $HEAD_SUMMARY_PATH"
echo "Design: ablate all 128 dims for the top ${TOP_HEADS} most sensitive heads from the head sweep."

baseline_log="$RAW_DIR/baseline.log"
baseline_summary="$(run_case "" "$baseline_log")"
if [[ -z "$baseline_summary" ]]; then
  echo "Failed to obtain baseline summary." >&2
  exit 1
fi

baseline_minade="$(parse_metric "$baseline_summary" 's/.*mean minADE=\([0-9.]*\)m.*/\1/p')"
baseline_e2e="$(parse_metric "$baseline_summary" 's/.*mean e2e latency=\([0-9.]*\) ms.*/\1/p')"
baseline_diffusion="$(parse_metric "$baseline_summary" 's/.*mean diffusion latency=\([0-9.]*\) ms.*/\1/p')"
baseline_ratio="$(parse_metric "$baseline_summary" 's/.*mean diffusion \/ e2e ratio=\([0-9.]*\)%.*/\1/p')"

printf '%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s\n' \
  "baseline" "" "" "" "$SELECT_PREFIX" "$baseline_minade" "0.0000" \
  "$baseline_e2e" "$baseline_diffusion" "$baseline_ratio" "ok" >> "$SUMMARY_CSV"

mapfile -t TOP_PAIRS < <(
  tail -n +2 "$HEAD_SUMMARY_PATH" \
    | awk -F',' '$1=="ablate_head" && $10=="ok" { print $2 "," $3 "," $6 }' \
    | sort -t',' -k3,3nr \
    | head -n "$TOP_HEADS"
)

if [[ "${#TOP_PAIRS[@]}" -eq 0 ]]; then
  echo "No valid ablate_head rows found in $HEAD_SUMMARY_PATH" >&2
  exit 1
fi

for triple in "${TOP_PAIRS[@]}"; do
  layer="${triple%%,*}"
  rest="${triple#*,}"
  head="${rest%%,*}"
  for dim in $(seq "$DIM_START" "$DIM_END"); do
    raw_log="$RAW_DIR/layer_${layer}_head_${head}_dim_${dim}.log"
    summary_line="$(run_case "${layer}:${head}:${dim}" "$raw_log" || true)"
    if [[ -n "$summary_line" ]]; then
      mean_minade="$(parse_metric "$summary_line" 's/.*mean minADE=\([0-9.]*\)m.*/\1/p')"
      mean_e2e="$(parse_metric "$summary_line" 's/.*mean e2e latency=\([0-9.]*\) ms.*/\1/p')"
      mean_diffusion="$(parse_metric "$summary_line" 's/.*mean diffusion latency=\([0-9.]*\) ms.*/\1/p')"
      mean_ratio="$(parse_metric "$summary_line" 's/.*mean diffusion \/ e2e ratio=\([0-9.]*\)%.*/\1/p')"
      delta_vs_baseline="$(awk -v a="$mean_minade" -v b="$baseline_minade" 'BEGIN { printf "%.4f", a - b }')"
      printf '%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s\n' \
        "ablate_dim" "$layer" "$head" "$dim" "$SELECT_PREFIX" "$mean_minade" "$delta_vs_baseline" \
        "$mean_e2e" "$mean_diffusion" "$mean_ratio" "ok" >> "$SUMMARY_CSV"
    else
      printf '%s,%s,%s,%s,"%s",,,,,,%s\n' \
        "ablate_dim" "$layer" "$head" "$dim" "$SELECT_PREFIX" "failed" >> "$SUMMARY_CSV"
    fi
  done
done

echo
echo "Dim ablation sweep complete."
echo "Summary:"
cat "$SUMMARY_CSV"
