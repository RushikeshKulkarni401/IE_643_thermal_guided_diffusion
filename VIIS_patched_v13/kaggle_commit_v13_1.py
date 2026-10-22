# ==== v13.1: full pipeline for "Save Version -> Save & Run All (Commit)"  (paste as the ONLY notebook cell) ====
# GPU0: v13_full (all terms)   GPU1: v13_no_tc (identical, thermal-consistency terms off = ablation)
# then: inference + evaluation (VIIS zero-shot vs both) on MSRS / LLVIP / light-low-dark, then alignment audit.
import os, sys, glob, shutil, subprocess, time, re
T0 = time.time()
BUDGET_H = 8.5      # stop training at this many hours -> evaluation still fits in Kaggle's 12 h
STEPS = 2500
VIIS_SRC = "/kaggle/input/datasets/elixir4o4/viis-master/media/Harddisk/zc/Image_Generation/latent-diffusion-main/logs/2024-06-02T12-01-20_res_condition/checkpoints/last.ckpt"
def el(): return f"[{(time.time() - T0) / 3600:5.2f} h]"
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-3000:], flush=True)

# 1. dependencies + code
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "omegaconf", "kornia", "lpips", "piq", "ultralytics"])
try: import pytorch_lightning
except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning"])
src = [os.path.dirname(p) for k in range(1, 6) for p in glob.glob("/kaggle/input/" + "*/" * k + "train_v13.py")]
REPO = "/kaggle/working/VIIS"; shutil.rmtree(REPO, ignore_errors=True); shutil.copytree(src[0], REPO); os.chdir(REPO)
print(el(), "code from", src[0], flush=True)
assert "w_traj" in open(f"{REPO}/train_v13.py").read(), "old v13 code attached -- update the dataset input to v13.1"

# 2. slim checkpoint on local disk (reuse an attached one if present)
slim_in = [p for k in range(1, 6) for p in glob.glob("/kaggle/input/" + "*/" * k + "viis_slim.ckpt")]
if slim_in:
    shutil.copy(slim_in[0], "/kaggle/working/viis_slim.ckpt")
else:
    sh(f'dd if="{VIIS_SRC}" of=/kaggle/working/last.ckpt bs=64M 2>&1 | tail -n 1')
    sh("python slim_ckpt_v13.py --src /kaggle/working/last.ckpt && rm -f /kaggle/working/last.ckpt")
sh("python -m trg13.paths"); print(el(), "checkpoint ready", flush=True)

# 3. train both runs in the background, report every 5 minutes
R = "/kaggle/working/runs"; os.makedirs(R, exist_ok=True)
procs = {}
for gpu, name, extra in [(0, "v13_full", []), (1, "v13_no_tc", ["--thermal_loss", "0"])]:
    procs[name] = subprocess.Popen([sys.executable, "-u", "train_v13.py", "--name", name, "--steps", str(STEPS)] + extra,
                                   cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                   stdout=open(f"{R}/{name}.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(30)
seen = {n: 0 for n in procs}
while any(p.poll() is None for p in procs.values()):
    time.sleep(300)
    for n, p in procs.items():
        try: lines = open(f"{R}/{n}.log", errors="ignore").read().splitlines()
        except FileNotFoundError: continue
        new = [l for l in lines[seen[n]:] if re.search(r"EVAL|NEW BEST|STOP|DONE|Traceback|Error|WARNING|loaded in|calibrated", l)]
        seen[n] = len(lines)
        last = [l for l in lines if re.match(r'\[\s*\d+s\] \{"step"', l)]
        print(el(), f"{n:10s}", "running" if p.poll() is None else f"exited {p.poll()}", "|", last[-1][:150] if last else "", flush=True)
        for l in new: print("      ", l[:420], flush=True)
    if (time.time() - T0) / 3600 > BUDGET_H:
        print(el(), "time budget reached -> stopping training (best.pt / latest.pt are kept)", flush=True)
        for p in procs.values():
            if p.poll() is None: p.terminate()
        time.sleep(30)
for n in procs:
    print(el(), n, "tail:"); sh(f"tail -n 4 {R}/{n}.log | cut -c1-300")

# 4. evaluation: VIIS zero-shot vs v13_no_tc vs v13
def ck(n):
    for f in ("best.pt", "final.pt", "latest.pt"):
        if os.path.exists(f"{R}/{n}/{f}"): return f"{R}/{n}/{f}"
pairs = [(n, ck(n)) for n in ("v13_no_tc", "v13_full") if ck(n)]
lab = {"v13_no_tc": "v13_no_tc", "v13_full": "v13"}
print(el(), "inference with", pairs, flush=True)
subprocess.run(f"python -u infer_v13.py --ckpts {','.join(c for _, c in pairs)} --labels {','.join(lab[n] for n, _ in pairs)} "
               f"--out /kaggle/working/infer_v13 > /kaggle/working/infer.log 2>&1", shell=True, cwd=REPO)
sh("tail -n 80 /kaggle/working/infer.log")
sh("python align_audit_v13.py --out /kaggle/working/align_audit 2>&1 | tail -n 25")
# keep best.pt (the model); drop resumable optimizer states to keep the saved output small
for f in glob.glob(f"{R}/*/latest.pt"): os.remove(f)
print(el(), "ALL DONE -> Output tab: runs/ (best.pt = model), infer_v13/ (report.md, grids), align_audit/", flush=True)
