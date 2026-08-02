#!/usr/bin/env bash
# Local smoke: run ``assess build`` inside slm-thesis-eval with no network,
# then assert cefr_tsar_status=ok and kvl_v2_status=ok on every scores.csv row.
#
# Prerequisites: image built via ./scripts/clusteruy/build_eval_image.sh
# See docs/clusteruy.md — eval image / local smoke section.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-srobaina99/slm-thesis-eval:latest}"
FIXTURE_ID="20260101_000000_phase2_weights_smoke"
FIXTURE_SRC="$SCRIPT_DIR/fixtures/$FIXTURE_ID"
FIXTURE_DST="$REPO_ROOT/results/runs/$FIXTURE_ID"
PYTHON_HOST="${REPO_ROOT}/venv/bin/python"

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_HOST" ]]; then
  echo "ERROR: host venv python missing at $PYTHON_HOST" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "ERROR: image $IMAGE_TAG not found. Run ./scripts/clusteruy/build_eval_image.sh first." >&2
  exit 1
fi

if [[ ! -f "$FIXTURE_SRC/full.csv" || ! -f "$FIXTURE_SRC/manifest.json" ]]; then
  echo "ERROR: committed fixture missing under $FIXTURE_SRC" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/results/runs"
rm -rf "$FIXTURE_DST"
cp -R "$FIXTURE_SRC" "$FIXTURE_DST"
echo "Staged fixture → $FIXTURE_DST"

echo "Running in-container assess (network=none, TRANSFORMERS_OFFLINE=1)…"
set +e
ASSESS_LOG="$(
  docker run --rm \
    --platform=linux/amd64 \
    --network=none \
    -v "$REPO_ROOT":/workspace \
    -e PYTHONPATH=/workspace/src \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e HF_HOME=/opt/hf_home \
    -e NLTK_DATA=/usr/share/nltk_data \
    "$IMAGE_TAG" \
    bash -c 'cd /workspace && python -m slm_experiments assess build \
      --source-run-ids '"$FIXTURE_ID"' --no-plot'
)"
ASSESS_RC=$?
set -e
printf '%s\n' "$ASSESS_LOG"

if [[ "$ASSESS_RC" -ne 0 ]]; then
  echo "ERROR: assess build exited $ASSESS_RC" >&2
  exit "$ASSESS_RC"
fi

RUN_ID="$(
  printf '%s\n' "$ASSESS_LOG" \
    | sed -n 's/^Assessment bundle complete: //p' \
    | tail -n 1
)"
if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: could not parse assessment run id from assess output" >&2
  exit 1
fi

SCORES="$REPO_ROOT/results/runs/$RUN_ID/scores.csv"
if [[ ! -f "$SCORES" ]]; then
  echo "ERROR: scores.csv not found at $SCORES" >&2
  exit 1
fi

echo "Asserting scorer statuses in $SCORES …"
EXPECTED_ROWS=2
"$PYTHON_HOST" - "$SCORES" "$EXPECTED_ROWS" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
expected_rows = int(sys.argv[2])
df = pd.read_csv(path)
if df.empty:
    raise SystemExit(f"scores.csv is empty: {path}")
if len(df) != expected_rows:
    raise SystemExit(
        f"FAIL: expected {expected_rows} scores.csv rows (fixture size), got {len(df)}"
    )
for col in ("cefr_tsar_status", "kvl_v2_status"):
    if col not in df.columns:
        raise SystemExit(f"missing column {col} in {path}")
    bad = df.loc[df[col].astype(str) != "ok"]
    if not bad.empty:
        print(bad[["item_id", col]].to_string(index=False), file=sys.stderr)
        # Include error columns when present for debugging.
        err_col = col.replace("_status", "_error")
        if err_col in bad.columns:
            print(bad[["item_id", err_col]].to_string(index=False), file=sys.stderr)
        raise SystemExit(f"FAIL: {col} is not ok for {len(bad)}/{len(df)} rows")
print(
    f"OK: {len(df)}/{expected_rows} rows with "
    "cefr_tsar_status=ok and kvl_v2_status=ok"
)
print(f"assessment_run_id={path.rsplit('/', 2)[-2]}")
PY

echo "smoke_assess_local: SUCCESS (run_id=$RUN_ID)"
