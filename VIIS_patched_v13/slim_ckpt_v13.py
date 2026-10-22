"""One-time conversion of the 9 GB VIIS last.ckpt into a ~2.5 GB local file with only what v13 needs.

    python slim_ckpt_v13.py            # -> /kaggle/working/viis_slim.ckpt  (picked up automatically afterwards)

Why: last.ckpt also holds the Lightning optimizer state, and on Kaggle /kaggle/input is network storage. Memory-mapping it
and reading the model weights page by page can take 20+ minutes per process (training x2, inference, smoke test).
This script reads the file ONCE sequentially, writes the EMA weights into the model weights (what VIIS samples with
and what v13 starts from), drops the optimizer state and the EMA copy, and saves to local disk.
"""
import argparse, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from trg13 import paths

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="")
ap.add_argument("--dst", default=paths.SLIM_DEFAULT)
a = ap.parse_args()
src = a.src or os.environ.get("VIIS_CKPT_ORIG") or paths.find_ckpt()
assert src and os.path.exists(src), f"VIIS last.ckpt not found ({src}); pass --src"
if os.path.exists(a.dst):
    print("already exists:", a.dst); sys.exit(0)
t0 = time.time()
print(f"reading {src} ({os.path.getsize(src) / 1e9:.1f} GB) sequentially ...", flush=True)
obj = torch.load(src, map_location="cpu", mmap=False, weights_only=False)
sd = obj.get("state_dict", obj)
print(f"  read in {time.time() - t0:.0f}s, {len(sd)} tensors", flush=True)
n_ema = 0
for k in list(sd.keys()):
    if k.startswith("model.") and not k.startswith("model_ema."):
        e = "model_ema." + k[len("model."):].replace(".", "")
        if e in sd:
            sd[k] = sd[e]; n_ema += 1
slim = {k: v for k, v in sd.items() if not k.startswith("model_ema.")}
n_model = sum(1 for k in slim if k.startswith("model."))
print(f"  EMA weights copied into {n_ema} / {n_model} model tensors", flush=True)
assert n_ema > 0.9 * n_model, "EMA keys did not match the model keys; refusing to write a non-EMA checkpoint"
tmp = a.dst + ".tmp"
torch.save({"state_dict": slim, "v13_slim": True}, tmp)
os.replace(tmp, a.dst)
print(f"wrote {a.dst} ({os.path.getsize(a.dst) / 1e9:.2f} GB) in {time.time() - t0:.0f}s total", flush=True)
