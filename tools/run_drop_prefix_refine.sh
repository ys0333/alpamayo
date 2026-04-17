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
TOP_SINGLES="${TOP_SINGLES:-4}"
TOP_GROUPS="${TOP_GROUPS:-2}"
SWEEP_SUMMARY_PATH="${SWEEP_SUMMARY_PATH:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/codex_history/drop_prefix_refine_${TIMESTAMP}"
RAW_DIR="$OUT_DIR/raw"
SUMMARY_CSV="$OUT_DIR/summary.csv"

mkdir -p "$RAW_DIR"

cat > "$SUMMARY_CSV" <<'EOF'
case_name,source_case,drop_prefix,mean_minade_m,delta_vs_baseline_m,mean_e2e_ms,mean_diffusion_ms,mean_ratio_pct,status
EOF

find_latest_sweep_summary() {
  local latest_dir
  latest_dir="$(ls -dt "$ROOT_DIR"/codex_history/drop_prefix_sweep_* 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest_dir" && -f "$latest_dir/summary.csv" ]]; then
    printf '%s\n' "$latest_dir/summary.csv"
  fi
}

if [[ -z "$SWEEP_SUMMARY_PATH" ]]; then
  SWEEP_SUMMARY_PATH="$(find_latest_sweep_summary)"
fi

if [[ -z "$SWEEP_SUMMARY_PATH" || ! -f "$SWEEP_SUMMARY_PATH" ]]; then
  echo "No drop-prefix sweep summary found. Run tools/run_drop_prefix_sweep.sh first or set SWEEP_SUMMARY_PATH." >&2
  exit 1
fi

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
echo "Using sweep summary: $SWEEP_SUMMARY_PATH"
echo "Design: refine around the least-sensitive single blocks and groups from the first-stage drop-prefix sweep."

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

printf '%s,%s,"%s",%s,%s,%s,%s,%s,%s\n' \
  "baseline_full_prefix" "baseline" "" "$baseline_minade" "0.0000" "$baseline_e2e" \
  "$baseline_diffusion" "$baseline_ratio" "ok" >> "$SUMMARY_CSV"

mapfile -t BEST_CASES < <(
  python3 - "$SWEEP_SUMMARY_PATH" "$TOP_SINGLES" "$TOP_GROUPS" <<'PY'
import csv, sys
path = sys.argv[1]
top_singles = int(sys.argv[2])
top_groups = int(sys.argv[3])
rows = []
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] != "ok" or row["case_type"] == "baseline":
            continue
        rows.append(row)
singles = sorted(
    [r for r in rows if r["case_type"] == "single"],
    key=lambda r: float(r["delta_vs_baseline_m"])
)[:top_singles]
groups = sorted(
    [r for r in rows if r["case_type"] == "group"],
    key=lambda r: float(r["delta_vs_baseline_m"])
)[:top_groups]
for row in singles + groups:
    print(f'{row["case_name"]}\t{row["drop_prefix"]}')
PY
)

if [[ "${#BEST_CASES[@]}" -eq 0 ]]; then
  echo "No valid candidates found in $SWEEP_SUMMARY_PATH" >&2
  exit 1
fi

for entry in "${BEST_CASES[@]}"; do
  source_case="${entry%%$'\t'*}"
  drop_prefix="${entry#*$'\t'}"
  raw_log="$RAW_DIR/${source_case}.log"
  summary_line="$(run_case "$source_case" "$drop_prefix" "$raw_log" || true)"
  if [[ -n "$summary_line" ]]; then
    mean_minade="$(parse_metric "$summary_line" 's/.*mean minADE=\([0-9.]*\)m.*/\1/p')"
    mean_e2e="$(parse_metric "$summary_line" 's/.*mean e2e latency=\([0-9.]*\) ms.*/\1/p')"
    mean_diffusion="$(parse_metric "$summary_line" 's/.*mean diffusion latency=\([0-9.]*\) ms.*/\1/p')"
    mean_ratio="$(parse_metric "$summary_line" 's/.*mean diffusion \/ e2e ratio=\([0-9.]*\)%.*/\1/p')"
    delta_vs_baseline="$(awk -v a="$mean_minade" -v b="$baseline_minade" 'BEGIN { printf "%.4f", a - b }')"
    printf '%s,%s,"%s",%s,%s,%s,%s,%s,%s\n' \
      "${source_case}_rerun" "$source_case" "$drop_prefix" "$mean_minade" "$delta_vs_baseline" "$mean_e2e" \
      "$mean_diffusion" "$mean_ratio" "ok" >> "$SUMMARY_CSV"
  else
    printf '%s,%s,"%s",,,,,,%s\n' \
      "${source_case}_rerun" "$source_case" "$drop_prefix" "failed" >> "$SUMMARY_CSV"
  fi
done

echo
echo "Drop-prefix refinement complete."
echo "Summary:"
cat "$SUMMARY_CSV"
