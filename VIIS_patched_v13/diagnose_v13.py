"""Diagnostic sweeps. Every arm starts from pretrained VIIS (step 0 = VIIS) and evaluates every 125 steps.

    python diagnose_v13.py            # sweep 2 (default, about 1.6 h on 2xT4)
    python diagnose_v13.py --sweep 1  # the first sweep (night branch / L_in / image losses / lr)

Sweep 1 result (21 Oct): ALL four arms fell from 21.8 dB to 12-15 dB by step 125, including pure eps loss at 3x lower
lr -> the losses were not the cause. Two things all four arms shared: (i) 2 of every 3 MSRS samples were forced to
t < 300 (biased timestep distribution, high-noise steps that set global tone barely trained), (ii) the new adapter.
Sweep 2 separates them:
  GPU0: z_sanity  lr = 0 for 125 steps          -> must stay at exactly step-0 values (else: pipeline bug)
        e_uniform_notpa   uniform t, NO adapter, MSRS only
        g_full_notpa      uniform t, NO adapter, full objective incl. night branch (fixed L_in)
  GPU1: f_uniform_tpa     uniform t, adapter,    MSRS only
        h_full_tpa        uniform t, adapter,    full objective incl. night branch (fixed L_in)  = LLVIP candidate
"""
import os, sys, json, subprocess, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--sweep", type=int, default=2); A = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
R = os.environ.get("RUNS", "/kaggle/working/runs")
COMMON = ["--steps", "500", "--eval_every", "125", "--patience", "99", "--guard_hf", "99", "--guard_fid", "99"]
if A.sweep == 1:
    QUEUES = [[("d_msrs_only", ["--lam_night", "0"]), ("d_eps_only", ["--lam_night", "0", "--w_l1", "0", "--w_ssim", "0", "--w_lp", "0", "--lam_tc", "0"])],
              [("d_no_in", ["--w_in", "0"]), ("d_lowlr", ["--lam_night", "0", "--lr", "3e-6", "--lr_tpa", "1e-5"])]]
    COMMON += ["--t_uniform", "0"]
else:
    QUEUES = [[("z_sanity", ["--lr", "0", "--lr_tpa", "0", "--steps", "125", "--lam_night", "0"]),
               ("e_uniform_notpa", ["--no_tpa", "--lam_night", "0"]),
               ("g_full_notpa", ["--no_tpa"])],
              [("f_uniform_tpa", ["--lam_night", "0"]),
               ("h_full_tpa", [])]]
os.makedirs(R, exist_ok=True); T0 = time.time()

def run_queue(gpu, queue, logf):
    lines = []
    for name, extra in queue:
        cmd = [sys.executable, "-u", "train_v13.py", "--name", name] + COMMON + extra
        lines.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                      stdout=open(f"{R}/{name}.log", "w"), stderr=subprocess.STDOUT))
        while lines[-1].poll() is None:
            time.sleep(60)
    return lines

import threading
threads = [threading.Thread(target=run_queue, args=(g, q, None)) for g, q in enumerate(QUEUES)]
for th in threads: th.start(); time.sleep(20)
seen = {}
while any(th.is_alive() for th in threads):
    time.sleep(240)
    for q in QUEUES:
        for name, _ in q:
            try: L = open(f"{R}/{name}.log", errors="ignore").read().splitlines()
            except FileNotFoundError: continue
            for l in L[seen.get(name, 0):]:
                if l.startswith("EVAL") or "Traceback" in l or "Error" in l or "WARNING" in l:
                    print(f"[{(time.time() - T0) / 3600:.2f} h] {name}: {l[:330]}", flush=True)
            seen[name] = len(L)
print("\n=== SUMMARY (MSRS val = real GT, step 0 = pretrained VIIS; n_* = held-out real night LLVIP/MSRS, no GT) ===")
print(f"{'arm':16s} {'step':>4s} {'psnr':>6s} {'ssim':>5s} {'lpips':>5s} {'p_lig':>5s} {'p_low':>5s} {'p_drk':>5s} | "
      f"{'n_tsc':>5s} {'n_obj':>5s} {'n_fid':>5s} {'hf_rat':>6s}")
for q in QUEUES:
    for name, _ in q:
        try:
            for l in open(f"{R}/{name}/eval.jsonl"):
                e = json.loads(l)
                print(f"{name:16s} {e['step']:4d} {e['psnr']:6.2f} {e['ssim']:5.3f} {e.get('lpips', float('nan')):5.3f} "
                      f"{e['psnr_light']:5.1f} {e['psnr_low']:5.1f} {e['psnr_dark']:5.1f} | {e['n_tsc']:5.3f} {e['n_obj']:5.2f} "
                      f"{e['n_fid']:5.3f} {e.get('hf_ratio', 1):6.2f}")
        except FileNotFoundError:
            print(name, "no eval.jsonl (see its log)")
print(f"done in {(time.time() - T0) / 3600:.2f} h")
