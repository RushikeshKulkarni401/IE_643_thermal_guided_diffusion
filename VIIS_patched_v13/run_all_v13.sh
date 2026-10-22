#!/usr/bin/env bash
# Full pipeline for a Kaggle 2xT4 session (or "Save & Run All" commit):
#   GPU0: v13_full (thermal-consistency losses on)    GPU1: v13_no_tc (ablation, identical except the thermal terms)
#   then: inference + evaluation of VIIS zero-shot, v13_no_tc and v13_full on MSRS / LLVIP / light-low-dark samples
#   then: alignment audit.  Extra args are passed to both training runs, e.g.  bash run_all_v13.sh --steps 2000
set -e
cd "$(dirname "$0")"
python -u slim_ckpt_v13.py          # one-time: 9 GB last.ckpt -> ~2.5 GB local copy (fast loads)
RUNS=${RUNS:-/kaggle/working/runs}; mkdir -p $RUNS
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
CUDA_VISIBLE_DEVICES=0 nohup python -u train_v13.py --name v13_full "$@" > $RUNS/v13_full.log 2>&1 &
P1=$!
if [ "$N_GPU" -ge 2 ]; then
  sleep 90                                    # stagger: both processes read the 9 GB VIIS checkpoint
  CUDA_VISIBLE_DEVICES=1 nohup python -u train_v13.py --name v13_no_tc --thermal_loss 0 "$@" > $RUNS/v13_no_tc.log 2>&1 &
  P2=$!
fi
wait $P1; [ -n "$P2" ] && wait $P2
[ "$N_GPU" -lt 2 ] && python -u train_v13.py --name v13_no_tc --thermal_loss 0 "$@" > $RUNS/v13_no_tc.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u infer_v13.py \
   --ckpts $RUNS/v13_no_tc/best.pt,$RUNS/v13_full/best.pt --labels v13_no_tc,v13 \
   --out /kaggle/working/infer_v13 2>&1 | tee /kaggle/working/infer_v13.log
python -u align_audit_v13.py --out /kaggle/working/align_audit | tail -30
echo "ALL DONE: runs in $RUNS, evaluation in /kaggle/working/infer_v13, alignment audit in /kaggle/working/align_audit"
