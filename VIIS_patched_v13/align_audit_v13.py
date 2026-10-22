"""Deliverable 1 evidence: RGB-thermal registration audit (CPU only, ~1-2 min).

    python align_audit_v13.py --out /kaggle/working/align_audit --n 12

For MSRS-test and LLVIP-test pairs: (1) residual misregistration of the shipped pairs (ECC warp of the original pair),
(2) recovery of KNOWN synthetic misalignments (shift 2/4/8 px + rotation) -> corner error before / after,
(3) edge-overlap of strong thermal edges with RGB edges before / after. Writes audit.csv, audit.json, overlay figure.
"""
import argparse, os, sys, json, csv
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from trg13 import paths, data as DT, align as AL

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/kaggle/working/align_audit"); ap.add_argument("--n", type=int, default=12)
ap.add_argument("--motion", default="euclidean", choices=["translation", "euclidean", "affine"])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
P = paths.resolve()
pairs = []
for d in DT.eval_real(P["MSRS"], P["LLVIP"], n_llvip=a.n, n_msrs_night=0)[:a.n]:
    pairs.append((d["src"], d["name"], d["rgb"], d["ir"]))
if P["MSRS"]:
    names = DT.msrs_names(P["MSRS"], "test", "D")
    for n in names[::max(1, len(names) // a.n)][:a.n]:
        pairs.append(("msrs_day", n, DT.load_rgb(f"{P['MSRS']}/test/vi/{n}"), DT.load_ir(f"{P['MSRS']}/test/ir/{n}")))
rows = []
for src, name, rgb, ir in pairs:
    s = 640 / max(rgb.shape[:2])                       # audit at <= 640 px (LLVIP frames are 1280x1024)
    if s < 1:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    for r in AL.audit([(rgb, ir)], shifts=(0, 2, 4, 8), motion=a.motion, rs=np.random.RandomState(len(rows))):
        r.update(src=src, name=name); rows.append(r)
with open(f"{a.out}/audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
summ = {}
for src in sorted({r["src"] for r in rows}):
    rr = [r for r in rows if r["src"] == src]
    summ[src] = dict(pairs=len({r["name"] for r in rr}),
                     residual_px_of_shipped_pairs=float(np.median([r["residual_W0"] for r in rr if r["shift"] == 0])),
                     **{f"shift{s}_before_px": float(np.nanmean([r["before"] for r in rr if r["shift"] == s])) for s in (2, 4, 8)},
                     **{f"shift{s}_after_px": float(np.nanmean([r["after"] for r in rr if r["shift"] == s])) for s in (2, 4, 8)},
                     success_rate=float(np.mean([r["ok"] for r in rr])),
                     edge_overlap_misaligned8=float(np.mean([r["overlap_before"] for r in rr if r["shift"] == 8])),
                     edge_overlap_realigned8=float(np.mean([r["overlap_after"] for r in rr if r["shift"] == 8])))
json.dump(summ, open(f"{a.out}/audit.json", "w"), indent=1)
print(json.dumps(summ, indent=1))

# overlay figure: thermal edges (red) on lifted RGB, misaligned by 8 px vs re-aligned
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = pairs[:4]
fig, ax = plt.subplots(len(show), 2, figsize=(8, 3.2 * len(show)), squeeze=False)
for k, (src, name, rgb, ir) in enumerate(show):
    h, w = ir.shape[:2]
    M = np.float32([[1, 0, 8], [0, 1, 8]])
    pert = cv2.warpAffine(ir, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    al, _, ok, _ = AL.align_pair(rgb, pert, motion=a.motion)
    base = np.power(rgb / 255.0, 0.45)
    for j, (t, ttl) in enumerate([(pert, "thermal edges, misaligned 8 px"), (al, f"after ECC ({'ok' if ok else 'failed'})")]):
        e = AL.grad_map(t); m = e > np.percentile(e, 93)
        o = base.copy(); o[m] = [1, 0.1, 0.1]
        ax[k, j].imshow(np.clip(o, 0, 1)); ax[k, j].axis("off"); ax[k, j].set_title(f"{src} {name}: {ttl}", fontsize=7)
plt.tight_layout(); plt.savefig(f"{a.out}/overlay.png", dpi=80)
print("wrote", a.out)
