"""v13 inference + evaluation on MSRS, LLVIP and light / low-light / dark samples.

    python infer_v13.py --ckpts /kaggle/working/runs/v13_full/best.pt,/kaggle/working/runs/v13_no_tc/best.pt \
                        --labels v13,v13_no_tc --out /kaggle/working/infer_v13

Sets (all from OFFICIAL TEST splits, never used in training or model selection):
  msrs_syn   MSRS test day images, deterministic light / low / dark degradation -> FULL-REFERENCE metrics vs the real day image
  real       LLVIP test + MSRS test night frames at full frame (tiled latent sampling) -> thermal / object / fidelity metrics,
             LLVIP person boxes -> box contrast + YOLO person recall
  bins       real test inputs binned by input brightness into light / low / dark (k per bin) -> grids + metrics
  user       --samples_dir with rgb/ + ir/ (same file names); optional ECC alignment (--align 1)
Methods: input, CLAHE (classical), VIIS zero-shot (pretrained weights = v13 at step 0), and every --ckpts entry.
Outputs: <out>/<set>/<method>/<name>.png, grids, metrics.csv, summary.json, report.md (+ paired tests vs VIIS zero-shot).
"""
import argparse, os, sys, json, time, csv
import numpy as np, cv2, torch

cv2.setNumThreads(2)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
from trg13 import paths, data as DT, metrics as ME, sampler as SM, model as MD, align as AL

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", default="", help="comma-separated v13 checkpoints (best.pt)")
ap.add_argument("--labels", default="", help="comma-separated names for --ckpts")
ap.add_argument("--out", default="/kaggle/working/infer_v13")
ap.add_argument("--zeroshot", type=int, default=1, help="include pretrained VIIS zero-shot as a baseline")
ap.add_argument("--sets", default="msrs_syn,real,bins,user")
ap.add_argument("--n_msrs", type=int, default=30); ap.add_argument("--n_llvip", type=int, default=24)
ap.add_argument("--n_msrs_night", type=int, default=12); ap.add_argument("--k_bins", type=int, default=4)
ap.add_argument("--samples_dir", default=""); ap.add_argument("--align", type=int, default=1)
ap.add_argument("--long_side", type=int, default=512, help="full frames are resized to this long side (multiples of 32)")
ap.add_argument("--steps", type=int, default=25); ap.add_argument("--eta", type=float, default=0.0)
ap.add_argument("--guid", type=float, default=1.0, help="thermal guidance scale (1 = off); trained via thermal dropout")
ap.add_argument("--seed", type=int, default=123); ap.add_argument("--tile_bs", type=int, default=4)
ap.add_argument("--yolo", type=int, default=1); ap.add_argument("--grid_rows", type=int, default=6)
a = ap.parse_args()

P = paths.resolve(); dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
def log(s): print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)
sets = [s.strip() for s in a.sets.split(",") if s.strip()]
ckpts = [c for c in a.ckpts.split(",") if c]
labels = [l for l in a.labels.split(",") if l] or [os.path.basename(os.path.dirname(c)) or f"m{i}" for i, c in enumerate(ckpts)]
assert len(labels) == len(ckpts), "--labels must match --ckpts"
u8 = lambda x01: (np.clip(x01, 0, 1) * 255 + 0.5).astype(np.uint8)
m11_to01 = lambda z: (z + 1) / 2

# ------------------------------------------------------------------------------------------------ data
items = {}
log("loading evaluation data (Kaggle input storage can be slow; this may take a few minutes) ...")
if "msrs_syn" in sets and P["MSRS"]:
    items["msrs_syn"] = DT.eval_msrs_synthetic(P["MSRS"], n=a.n_msrs)
    for d in items["msrs_syn"]:
        d["key"] = f"{d['sev_name']}_{os.path.splitext(d['name'])[0]}"
def prep_real(lst, tag):
    out = []
    for d in lst:
        rgb, ir = d["rgb"], d["ir"]
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        info = {}
        if tag == "user" and a.align:
            ir, W, ok, cc = AL.align_pair(rgb, ir)
            info = dict(aligned=bool(ok), ecc=cc)
        r, i = SM.prep_frame(rgb, ir, a.long_side)
        sy, sx = r.shape[0] / rgb.shape[0], r.shape[1] / rgb.shape[1]
        boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in d.get("boxes", [])]
        out.append(dict(d, rgb=r, ir=i, boxes=boxes, key=f"{d.get('src', tag)}_{os.path.splitext(d['name'])[0]}", info=info,
                        bin=d.get("bin") or DT.lum_bin(u8(m11_to01(r)))))
    return out
if "real" in sets:
    items["real"] = prep_real(DT.eval_real(P["MSRS"], P["LLVIP"], a.n_llvip, a.n_msrs_night), "real")
log("msrs_syn / real loaded" if items else "no msrs_syn / real items")
if "bins" in sets:
    for b, lst in DT.sample_bins(P["MSRS"], P["LLVIP"], k=a.k_bins).items():
        for d in prep_real(lst, "bins"):
            d["bin"] = b; items.setdefault("bins", []).append(d)
if "user" in sets and a.samples_dir:
    items["user"] = prep_real(DT.user_samples(a.samples_dir), "user")
log("items: " + ", ".join(f"{k}={len(v)}" for k, v in items.items()))
assert items, "no evaluation data found (check MSRS / LLVIP paths)"

# ------------------------------------------------------------------------------------------------ classical baselines
def clahe(rgb01):
    x = np.power(np.clip(rgb01, 0, 1), 0.6)
    lab = cv2.cvtColor(u8(x), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

OUTS = {s: {} for s in items}            # set -> method -> key -> uint8 image
def store(set_, method, key, img01):
    OUTS[set_].setdefault(method, {})[key] = u8(img01)
    d = f"{a.out}/{set_}/{method}"; os.makedirs(d, exist_ok=True)
    cv2.imwrite(f"{d}/{key}.png", cv2.cvtColor(u8(img01), cv2.COLOR_RGB2BGR))

for s, lst in items.items():
    for d in lst:
        store(s, "input", d["key"], m11_to01(d["rgb"]))
        store(s, "clahe", d["key"], clahe(m11_to01(d["rgb"])))
        if s == "msrs_syn":
            store(s, "gt", d["key"], m11_to01(d["gt"]))

# ------------------------------------------------------------------------------------------------ diffusion methods
def run_all(model, method):
    with MD.ema_scope(model):
        for s, lst in items.items():
            t1 = time.time()
            for d in lst:
                o = SM.enhance(model, d["rgb"], d["ir"], steps=a.steps, eta=a.eta, seed=a.seed, guid=a.guid, tile_bs=a.tile_bs)
                store(s, method, d["key"], o)
            log(f"{method:14s} {s:9s} {len(lst)} images  {time.time() - t1:5.0f}s")

model, cur_mode = None, None
def get_model(mode, use_tpa):
    global model, cur_mode
    if model is None or cur_mode != (mode, use_tpa):
        model = None; torch.cuda.empty_cache()
        model = MD.build(HERE, P["CKPT"], device=dev, mode=mode, use_tpa=use_tpa); model.eval()
        cur_mode = (mode, use_tpa)
    return model

methods = ["input", "clahe"]
if a.zeroshot:
    m = get_model("attn_out", True)       # zero-initialised adapter + no checkpoint == pretrained VIIS
    run_all(m, "viis_zs"); methods.append("viis_zs")
for ck, lb in zip(ckpts, labels):
    meta, step, _ = MD.ckpt_meta(ck)
    m = get_model(meta.get("mode", "attn_out"), meta.get("use_tpa", True))
    MD.load_ckpt(m, ck); log(f"loaded {ck} (step {step}, {meta})")
    run_all(m, lb); methods.append(lb)
del model; torch.cuda.empty_cache()

# ------------------------------------------------------------------------------------------------ YOLO (optional)
yolo = None
if a.yolo and "real" in items and any(d.get("boxes") for d in items["real"]):
    try:
        from ultralytics import YOLO
        yolo = YOLO(f"{HERE}/yolov8n.pt"); log("YOLO person detector ready")
    except Exception as e:
        log(f"YOLO unavailable ({e}); pip install ultralytics to add person recall")

def iou(p, q):
    ix, iy = max(0, min(p[2], q[2]) - max(p[0], q[0])), max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = ix * iy; return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)

def yolo_recall(img_u8, boxes):
    if yolo is None or not boxes:
        return float("nan")
    r = yolo.predict(np.ascontiguousarray(img_u8[..., ::-1]), classes=[0], conf=0.25, verbose=False)[0]
    det = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
    return float(np.mean([any(iou(g, q) >= 0.5 for q in det) for g in boxes]))

lp_fn = None
try:
    import lpips
    _alex = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    def lp_fn(x01, y01):
        with torch.no_grad():
            t = lambda z: torch.from_numpy(np.ascontiguousarray(z)).permute(2, 0, 1)[None].float().to(dev) * 2 - 1
            return float(_alex(t(x01), t(y01)))
except Exception as e:
    log(f"LPIPS unavailable ({e})")
brisque = None
try:
    import piq
    def brisque(x01):
        with torch.no_grad():
            return float(piq.brisque(torch.from_numpy(np.ascontiguousarray(x01)).permute(2, 0, 1)[None].float().clamp(0, 1), data_range=1.0))
except Exception:
    pass

# ------------------------------------------------------------------------------------------------ metrics
rows = []
for s, lst in items.items():
    for d in lst:
        in01, th01 = m11_to01(d["rgb"]), m11_to01(d["ir"])[..., 0]
        for meth in methods + (["gt"] if s == "msrs_syn" else []):
            o = OUTS[s][meth][d["key"]].astype(np.float32) / 255.0
            r = dict(set=s, method=meth, key=d["key"], src=d.get("src", "msrs"), sev=d.get("sev_name", d.get("bin", "")))
            if s == "msrs_syn":
                r.update(ME.full_ref_all(o, m11_to01(d["gt"]), in01, th01, lp_fn))
            r.update({("nr_" + k): v for k, v in ME.no_ref_all(o, in01, th01, d.get("boxes")).items()})
            if s == "real" and d.get("boxes"):
                r["yolo_recall"] = yolo_recall(OUTS[s][meth][d["key"]], d["boxes"])
            if brisque is not None and s != "msrs_syn":
                try: r["brisque"] = brisque(o)
                except Exception: r["brisque"] = float("nan")
            rows.append(r)
keys = sorted({k for r in rows for k in r})
with open(f"{a.out}/metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

HIGHER = {"psnr", "ssim", "psnr_hot", "nr_tsc", "nr_tes", "nr_obj", "nr_box", "nr_obj_L", "nr_box_L", "nr_in_fid", "yolo_recall", "nr_entropy"}
LOWER = {"lpips", "tsc_gap", "nr_hf", "brisque"}
def summarise(group_fn):
    out = {}
    for r in rows:
        g = group_fn(r)
        if g is None: continue
        out.setdefault(g, []).append(r)
    res = {}
    for g, rr in out.items():
        num = [k for k in keys if any(isinstance(x.get(k), (float, int)) and not isinstance(x.get(k), bool) for x in rr)]
        res["|".join(g)] = {k: float(np.nanmean([x.get(k, np.nan) for x in rr])) for k in num}
        res["|".join(g)]["n"] = len(rr)
    return res
summary = dict(by_set=summarise(lambda r: (r["set"], r["method"])),
               by_severity=summarise(lambda r: (r["set"], r["sev"], r["method"]) if r["set"] in ("msrs_syn", "bins") else None))

# paired comparison vs VIIS zero-shot (per image, Wilcoxon signed-rank)
paired = {}
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
if "viis_zs" in methods:
    for s in items:
        base = {r["key"]: r for r in rows if r["set"] == s and r["method"] == "viis_zs"}
        for lb in labels:
            cur = {r["key"]: r for r in rows if r["set"] == s and r["method"] == lb}
            for k in sorted(HIGHER | LOWER):
                pairs = [(cur[x][k], base[x][k]) for x in cur if x in base and np.isfinite(cur[x].get(k, np.nan)) and np.isfinite(base[x].get(k, np.nan))]
                if len(pairs) < 5: continue
                c, b = np.array(pairs).T
                better = (c > b) if k in HIGHER else (c < b)
                p = float(wilcoxon(c, b).pvalue) if wilcoxon is not None and np.any(c != b) else float("nan")
                paired[f"{s}|{lb}|{k}"] = dict(n=len(c), mean_model=float(c.mean()), mean_viis=float(b.mean()),
                                               win_rate=float(better.mean()), p=p)
summary["paired_vs_viis_zeroshot"] = paired
json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ grids
def grid(set_, keys_, cols, path, row_label=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys_ = keys_[:a.grid_rows]
    if not keys_: return
    th = {d["key"]: m11_to01(d["ir"]) for d in items[set_]}
    fig, ax = plt.subplots(len(keys_), len(cols) + 1, figsize=(2.3 * (len(cols) + 1), 2.1 * len(keys_)), squeeze=False)
    for r, k in enumerate(keys_):
        ims = [OUTS[set_][cols[0]][k], th[k]] + [OUTS[set_][c][k] for c in cols[1:]]
        names = [cols[0], "thermal"] + cols[1:]
        for c, (im, nm) in enumerate(zip(ims, names)):
            ax[r, c].imshow(im); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(nm, fontsize=8)
        if row_label: ax[r, 0].set_ylabel(row_label(k), fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)

cols = ["input", "clahe"] + (["viis_zs"] if "viis_zs" in methods else []) + labels
if "msrs_syn" in items:
    for sev in ("light", "low", "dark"):
        ks = [d["key"] for d in items["msrs_syn"] if d["sev_name"] == sev]
        grid("msrs_syn", ks[::max(1, len(ks) // a.grid_rows)], cols + ["gt"], f"{a.out}/grid_msrs_{sev}.png")
if "real" in items:
    for src in ("llvip", "msrs_night"):
        ks = [d["key"] for d in items["real"] if d["src"] == src]
        grid("real", ks[::max(1, len(ks) // a.grid_rows)], cols, f"{a.out}/grid_{src}.png")
if "bins" in items:
    for b in ("light", "low", "dark"):
        grid("bins", [d["key"] for d in items["bins"] if d["bin"] == b], cols, f"{a.out}/grid_bin_{b}.png")
if "user" in items:
    grid("user", [d["key"] for d in items["user"]], cols, f"{a.out}/grid_user.png")

# ------------------------------------------------------------------------------------------------ report
def table(set_, ks, title):
    lines = [f"\n### {title}\n", "| method | " + " | ".join(ks) + " |", "|---" * (len(ks) + 1) + "|"]
    for m in methods + (["gt"] if set_ == "msrs_syn" else []):
        v = summary["by_set"].get(f"{set_}|{m}")
        if v: lines.append(f"| {m} | " + " | ".join(f"{v.get(k, float('nan')):.4g}" for k in ks) + " |")
    return "\n".join(lines)
rep = [f"# v13 evaluation\n\nsteps={a.steps} eta={a.eta} guid={a.guid} long_side={a.long_side}\n",
       "Higher is better: psnr, ssim, psnr_hot, tsc (thermal structural consistency), tes (thermal edge score), obj/box (object "
       "separation, Lab), in_fid (input fidelity), yolo_recall. Lower is better: lpips, tsc_gap (|tsc - tsc of the real day "
       "image|), hf (texture in regions neither modality supports), brisque. For tsc / obj, MORE is not automatically better: "
       "compare with the gt row (natural level)."]
if "msrs_syn" in items:
    rep.append(table("msrs_syn", ["psnr", "ssim", "lpips", "psnr_hot", "tsc", "tsc_gap", "obj_ratio", "obj_ab"], "MSRS test, synthetic degradation (full reference; obj_ab = colour-only object separation, compare with the gt row)"))
    for sev in ("light", "low", "dark"):
        ln = [f"\n**{sev}**: " + ", ".join(f"{m} {summary['by_severity'].get(f'msrs_syn|{sev}|{m}', {}).get('psnr', float('nan')):.2f} dB / "
                                           f"{summary['by_severity'].get(f'msrs_syn|{sev}|{m}', {}).get('ssim', float('nan')):.3f}" for m in methods)]
        rep += ln
if "real" in items:
    rep.append(table("real", ["nr_tsc", "nr_tes", "nr_obj_L", "nr_box_L", "nr_obj_ab", "yolo_recall", "nr_in_fid", "nr_hf", "brisque"], "Real night (LLVIP test + MSRS test night), no reference (obj_L/box_L = brightness separation; obj_ab = colour separation, compare with the input row: much higher = invented colour)"))
if "bins" in items:
    rep.append(table("bins", ["nr_tsc", "nr_obj_L", "nr_obj_ab", "nr_in_fid", "nr_hf", "brisque"], "Real light / low-light / dark samples"))
if paired:
    rep.append("\n### Paired per-image comparison vs VIIS zero-shot (win rate, Wilcoxon p)\n")
    rep += [f"- {k}: model {v['mean_model']:.4g} vs VIIS {v['mean_viis']:.4g}, win {100 * v['win_rate']:.0f}% (n={v['n']}, p={v['p']:.3g})"
            for k, v in paired.items()]
open(f"{a.out}/report.md", "w").write("\n".join(rep) + "\n")
print("\n".join(rep))
log(f"DONE -> {a.out}  (grids: grid_*.png, metrics.csv, summary.json, report.md)")
