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
OUT_DIR="$ROOT_DIR/codex_history/sensitivity_followups_${TIMESTAMP}"
RAW_DIR="$OUT_DIR/raw"
SUMMARY_CSV="$OUT_DIR/summary.csv"
mkdir -p "$RAW_DIR"

cat > "$SUMMARY_CSV" <<'EOF'
stage,case_name,mode,spec,mean_minade_m,delta_vs_baseline_m,mean_e2e_ms,mean_diffusion_ms,mean_ratio_pct,status
EOF

run_case() {
  local stage="$1"
  local case_name="$2"
  local mode="$3"
  local spec="$4"
  local raw_log="$5"

  local -a cmd=(
    "$PYTHON_BIN" "$INFER_SCRIPT"
    "--clip-id" "$CLIP_ID"
    "--t0-us" "$T0_US"
    "--t0-step-us" "$T0_STEP_US"
    "--nums" "$NUMS"
  )
  if [[ "$mode" == "drop-prefix" && -n "$spec" ]]; then
    cmd+=("--drop-prefix" "$spec")
  elif [[ "$mode" == "ablate-head" && -n "$spec" ]]; then
    cmd+=("--ablate-head" "$spec")
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

append_summary() {
  local stage="$1"
  local case_name="$2"
  local mode="$3"
  local spec="$4"
  local summary_line="$5"
  local baseline_minade="$6"

  local mean_minade mean_e2e mean_diffusion mean_ratio delta_vs_baseline
  mean_minade="$(parse_metric "$summary_line" 's/.*mean minADE=\([0-9.]*\)m.*/\1/p')"
  mean_e2e="$(parse_metric "$summary_line" 's/.*mean e2e latency=\([0-9.]*\) ms.*/\1/p')"
  mean_diffusion="$(parse_metric "$summary_line" 's/.*mean diffusion latency=\([0-9.]*\) ms.*/\1/p')"
  mean_ratio="$(parse_metric "$summary_line" 's/.*mean diffusion \/ e2e ratio=\([0-9.]*\)%.*/\1/p')"
  delta_vs_baseline="$(awk -v a="$mean_minade" -v b="$baseline_minade" 'BEGIN { printf "%.4f", a - b }')"
  printf '%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s\n' \
    "$stage" "$case_name" "$mode" "$spec" "$mean_minade" "$delta_vs_baseline" \
    "$mean_e2e" "$mean_diffusion" "$mean_ratio" "ok" >> "$SUMMARY_CSV"
}

echo "Writing raw logs to: $RAW_DIR"
echo "Writing summary to:  $SUMMARY_CSV"

# Baseline once for all follow-up comparisons.
baseline_log="$RAW_DIR/baseline.log"
baseline_summary="$(run_case "baseline" "baseline_full_prefix" "baseline" "" "$baseline_log")"
if [[ -z "$baseline_summary" ]]; then
  echo "Failed to obtain baseline summary." >&2
  exit 1
fi
baseline_minade="$(parse_metric "$baseline_summary" 's/.*mean minADE=\([0-9.]*\)m.*/\1/p')"
append_summary "baseline" "baseline_full_prefix" "baseline" "" "$baseline_summary" "$baseline_minade"

# 1) Re-check top insensitive/sensitive single-block drops from prior sweep.
declare -A DROP_RECHECKS=(
  [insensitive_34]="34"
  [insensitive_27]="27"
  [insensitive_23]="23"
  [insensitive_26]="26"
  [insensitive_18]="18"
  [sensitive_15]="15"
  [sensitive_13]="13"
  [sensitive_12]="12"
  [sensitive_24]="24"
)

for case_name in "${!DROP_RECHECKS[@]}"; do
  spec="${DROP_RECHECKS[$case_name]}"
  raw_log="$RAW_DIR/recheck_${case_name}.log"
  summary_line="$(run_case "recheck" "$case_name" "drop-prefix" "$spec" "$raw_log" || true)"
  if [[ -n "$summary_line" ]]; then
    append_summary "recheck" "$case_name" "drop-prefix" "$spec" "$summary_line" "$baseline_minade"
  else
    printf '%s,%s,%s,"%s",,,,,,%s\n' "recheck" "$case_name" "drop-prefix" "$spec" "failed" >> "$SUMMARY_CSV"
  fi
done

# 2) Refine around least-sensitive group 18-23.
declare -A GROUP_REFINES=(
  [group_18_20]="18,19,20"
  [group_21_23]="21,22,23"
  [group_18_19_22_23]="18,19,22,23"
  [group_18_21_23]="18,21,23"
  [group_17_22]="17,18,19,20,21,22"
  [group_19_24]="19,20,21,22,23,24"
)

for case_name in "${!GROUP_REFINES[@]}"; do
  spec="${GROUP_REFINES[$case_name]}"
  raw_log="$RAW_DIR/refine_${case_name}.log"
  summary_line="$(run_case "group_refine" "$case_name" "drop-prefix" "$spec" "$raw_log" || true)"
  if [[ -n "$summary_line" ]]; then
    append_summary "group_refine" "$case_name" "drop-prefix" "$spec" "$summary_line" "$baseline_minade"
  else
    printf '%s,%s,%s,"%s",,,,,,%s\n' "group_refine" "$case_name" "drop-prefix" "$spec" "failed" >> "$SUMMARY_CSV"
  fi
done

# 3) Focused head ablation inside sensitive blocks (full-prefix context).
for layer in 12 13 15; do
  for head in $(seq 0 15); do
    case_name="head_l${layer}_h${head}"
    spec="${layer}:${head}"
    raw_log="$RAW_DIR/${case_name}.log"
    summary_line="$(run_case "head_focus" "$case_name" "ablate-head" "$spec" "$raw_log" || true)"
    if [[ -n "$summary_line" ]]; then
      append_summary "head_focus" "$case_name" "ablate-head" "$spec" "$summary_line" "$baseline_minade"
    else
      printf '%s,%s,%s,"%s",,,,,,%s\n' "head_focus" "$case_name" "ablate-head" "$spec" "failed" >> "$SUMMARY_CSV"
    fi
  done
done

echo
echo "Follow-up sensitivity experiments complete."
echo "Summary:"
cat "$SUMMARY_CSV"
