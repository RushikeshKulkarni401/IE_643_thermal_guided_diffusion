#!/usr/bin/env bash
# 5-10 min end-to-end check on Kaggle (GPU) of EVERY torch code path before the long run:
#   model build + adapter hooks, calibration, step-0 eval, MSRS + night training steps, EMA, checkpoint save/load,
#   tiled full-frame sampling, all metrics, grids and report.
set -e
cd "$(dirname "$0")/.."
python -u slim_ckpt_v13.py          # one-time: 9 GB last.ckpt -> ~2.5 GB local copy (fast loads)
python tests/test_v13_numpy.py
python -m trg13.paths
python train_v13.py --name smoke --smoke
RUNS=${RUNS:-/kaggle/working/runs}
python infer_v13.py --ckpts $RUNS/smoke/best.pt --labels smoke --out /kaggle/working/infer_smoke \
       --n_msrs 2 --n_llvip 2 --n_msrs_night 1 --k_bins 1 --steps 4 --grid_rows 2
echo "SMOKE TEST PASSED"
