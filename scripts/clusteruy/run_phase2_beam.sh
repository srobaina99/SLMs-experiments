#!/bin/bash
# ============================================================
# DEPRECATED — do not submit.
#
# phase2 beam (best-of-N) is void at temperature=0.0 and hard-fails
# in the CLI. Use KVL beam or guided decoding instead:
#
#   sbatch scripts/clusteruy/run_phase2_kvl_beam.sh
#   # or: python -m slm_experiments phase2 guided --prompts all --no-plot
#
# See docs/experiment-setup-recommendations.md item 3 and docs/clusteruy.md.
# ============================================================

echo "ERROR: scripts/clusteruy/run_phase2_beam.sh is deprecated." >&2
echo "Use scripts/clusteruy/run_phase2_kvl_beam.sh or phase2 guided." >&2
exit 1
