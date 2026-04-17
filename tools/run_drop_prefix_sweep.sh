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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/codex_history/drop_prefix_sweep_${TIMESTAMP}"
RAW_DIR="$OUT_DIR/raw"
SUMMARY_CSV="$OUT_DIR/summary.csv"

mkdir -p "$RAW_DIR"

cat > "$SUMMARY_CSV" <<'EOF'
case_name,drop_prefix,mean_minade_m,delta_vs_baseline_m,mean_e2e_ms,mean_diffusion_ms,mean_ratio_pct,status
EOF

declare -a CASE_NAMES=(
  "baseline_full_prefix"
  "drop_0_5"
  "drop_6_11"
  "drop_12_17"
  "drop_18_23"
  "drop_24_29"
  "drop_30_35"
  "drop_0_11"
  "drop_12_23"
  "drop_24_35"
)

declare -a CASE_VALUES=(
  ""
  "0,1,2,3,4,5"
  "6,7,8,9,10,11"
  "12,13,14,15,16,17"
  "18,19,20,21,22,23"
  "24,25,26,27,28,29"
  "30,31,32,33,34,35"
  "0,1,2,3,4,5,6,7,8,9,10,11"
  "12,13,14,15,16,17,18,19,20,21,22,23"
  "24,25,26,27,28,29,30,31,32,33,34,35"
)

run_case() {
  local case_name="$1"
  local drop_prefix="$2"
  local raw_log="$3"

  local -a cmd=(
    "$PYTHON_BIN" "$INFER_SCRIPT"
    "--clip-id" "$CLIP_ID"
    "--t0-us" "$T0_US"
    "--t0-step-us" "$T0_STEP_US"
    "--nums" "$NUMS"
  )
  if [[ -n "$drop_prefix" ]]; then
    cmd+=("--drop-prefix" "$drop_prefix")
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
echo "Design: full-prefix baseline vs drop-prefix group ablations."

baseline_log="$RAW_DIR/baseline.log"
baseline_summary="$(run_case "baseline" "" "$baseline_log")"
if [[ -z "$baseline_summary" ]]; then
  echo "Failed to obtain baseline summary." >&2
  exit 1
fi

baseline_minade="$(parse_metric "$baseline_summary" 's/.*mean minADE=\([0-9.]*\)m.*/\1/p')"
baseline_e2e="$(parse_metric "$baseline_summary" 's/.*mean e2e latency=\([0-9.]*\) ms.*/\1/p')"
baseline_diffusion="$(parse_metric "$baseline_summary" 's/.*mean diffusion latency=\([0-9.]*\) ms.*/\1/p')"
baseline_ratio="$(parse_metric "$baseline_summary" 's/.*mean diffusion \/ e2e ratio=\([0-9.]*\)%.*/\1/p')"

printf '%s,"%s",%s,%s,%s,%s,%s,%s\n' \
  "baseline_full_prefix" "" "$baseline_minade" "0.0000" "$baseline_e2e" \
  "$baseline_diffusion" "$baseline_ratio" "ok" >> "$SUMMARY_CSV"

for idx in "${!CASE_NAMES[@]}"; do
  case_name="${CASE_NAMES[$idx]}"
  drop_prefix="${CASE_VALUES[$idx]}"
  [[ "$case_name" == "baseline_full_prefix" ]] && continue
  raw_log="$RAW_DIR/${case_name}.log"
  summary_line="$(run_case "$case_name" "$drop_prefix" "$raw_log" || true)"
  if [[ -n "$summary_line" ]]; then
    mean_minade="$(parse_metric "$summary_line" 's/.*mean minADE=\([0-9.]*\)m.*/\1/p')"
    mean_e2e="$(parse_metric "$summary_line" 's/.*mean e2e latency=\([0-9.]*\) ms.*/\1/p')"
    mean_diffusion="$(parse_metric "$summary_line" 's/.*mean diffusion latency=\([0-9.]*\) ms.*/\1/p')"
    mean_ratio="$(parse_metric "$summary_line" 's/.*mean diffusion \/ e2e ratio=\([0-9.]*\)%.*/\1/p')"
    delta_vs_baseline="$(awk -v a="$mean_minade" -v b="$baseline_minade" 'BEGIN { printf "%.4f", a - b }')"
    printf '%s,"%s",%s,%s,%s,%s,%s,%s\n' \
      "$case_name" "$drop_prefix" "$mean_minade" "$delta_vs_baseline" "$mean_e2e" \
      "$mean_diffusion" "$mean_ratio" "ok" >> "$SUMMARY_CSV"
  else
    printf '%s,"%s",,,,,,%s\n' "$case_name" "$drop_prefix" "failed" >> "$SUMMARY_CSV"
  fi
done

echo
echo "Drop-prefix sweep complete."
echo "Summary:"
cat "$SUMMARY_CSV"
