"""TRG-Diffusion v13: thermal-guided fine-tuning of pretrained VIIS, WITHOUT any zero-shot/pseudo targets.

    python train_v13.py --name v13_full                      # full model
    python train_v13.py --name v13_no_tc --thermal_loss 0    # ablation: identical, minus the thermal-consistency terms
    python train_v13.py --name smoke --smoke                 # 3-5 min end-to-end check of every code path

Supervision (see V13_DESIGN.md):
  MSRS day image (real GT)  <-  physically degraded night/smoke/glare input + thermal  : eps-MSE + L1 + SSIM + LPIPS
                                                                                         + thermal-consistency (GT level)
  real night frames (LLVIP train + MSRS night, NO GT)                                   : input-structure fidelity
                                                                                         + thermal-consistency (calibrated)
Model selection: MSRS validation (held-out train day images, fixed degradations) = real GT, plus night guards.
"""
import argparse, os, sys, json, time, math, random, shutil
import numpy as np, torch, torch.nn.functional as F, cv2
from torch.utils.data import DataLoader

cv2.setNumThreads(0)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
# optimisation
ap.add_argument("--steps", type=int, default=2500)
ap.add_argument("--bs", type=int, default=3, help="MSRS batch")
ap.add_argument("--n_img", type=int, default=2, help="MSRS samples per step that get image-space losses (low t)")
ap.add_argument("--lr", type=float, default=1e-5, help="pretrained VIIS params (attention / output blocks)")
ap.add_argument("--lr_tpa", type=float, default=3e-5, help="new zero-initialised thermal adapter")
ap.add_argument("--warmup", type=int, default=200, help="Adam's first steps are sign-like (every weight moves ~lr): warm up")
ap.add_argument("--min_lr_frac", type=float, default=0.1)
ap.add_argument("--mode", default="attn_out", choices=["attn2", "attn", "attn_out"])
ap.add_argument("--no_tpa", action="store_true", help="disable the thermal prior adapter (architecture ablation)")
ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--clip", type=float, default=1.0)
# supervised MSRS terms
ap.add_argument("--t_img", type=int, default=300)
ap.add_argument("--t_uniform", type=int, default=1, help="1: timesteps uniform for ALL samples (as VIIS was trained); image losses only on samples that land below t_img. 0: old v13 behaviour (2 of 3 samples forced below t_img)")
ap.add_argument("--hot_w", type=float, default=3.0)
ap.add_argument("--w_l1", type=float, default=0.5)
ap.add_argument("--w_ssim", type=float, default=0.5)
ap.add_argument("--w_lp", type=float, default=0.2)
ap.add_argument("--th_drop", type=float, default=0.1, help="thermal dropout prob (enables trained thermal guidance)")
# thermal consistency / night terms
ap.add_argument("--thermal_loss", type=int, default=1, help="0 = ablation without L_struct / L_pres")
ap.add_argument("--lam_tc", type=float, default=0.3, help="weight of thermal terms on MSRS (GT-level targets)")
ap.add_argument("--lam_night", type=float, default=0.2, help="weight of the real-night self-supervised block")
ap.add_argument("--w_in", type=float, default=1.0); ap.add_argument("--w_struct", type=float, default=1.0)
ap.add_argument("--w_pres", type=float, default=1.0); ap.add_argument("--pres_cap", type=float, default=0.35)
ap.add_argument("--night_every", type=int, default=2); ap.add_argument("--night_bs", type=int, default=2)
ap.add_argument("--night_t", default="150,400", help="t range where the night x0 estimate is taken")
ap.add_argument("--night_sigma_cap", type=float, default=0.06, help="noise-estimate cap for the night q map (0.02 in the first run let sensor noise count as texture, so L_in taught the model to copy noise)")
ap.add_argument("--ramp", type=int, default=300, help="steps to ramp the thermal/night terms in")
ap.add_argument("--w_traj", type=float, default=1.0, help="MSRS sampling-trajectory loss vs real GT (fixes exposure drift)")
ap.add_argument("--traj_every", type=int, default=1)
# validation
ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--eval_steps", type=int, default=25)
ap.add_argument("--n_val", type=int, default=12); ap.add_argument("--min_steps", type=int, default=750)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--guard_hf", type=float, default=3.0, help="flag if unsupported HF energy on night val > x step-0 (VIIS is over-smooth, so 1.5-2.2x is normal detail)")
ap.add_argument("--select", default="night", choices=["night", "msrs"], help="best.pt criterion: night = LLVIP-first composite with an MSRS tolerance")
ap.add_argument("--msrs_tol", type=float, default=2.0, help="night selection: checkpoint is eligible only if MSRS val PSNR >= step-0 PSNR - tol")
ap.add_argument("--guard_fid", type=float, default=0.08, help="flag if input fidelity drops by more than this vs step-0")
ap.add_argument("--workers", type=int, default=2); ap.add_argument("--llvip_limit", type=int, default=6000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    # plumbing test only: keeps the REAL warm-up, so 4 steps barely move the weights and outputs should stay ~ step 0
    a.steps, a.eval_every, a.n_val, a.eval_steps, a.night_every, a.min_steps, a.ramp = 4, 2, 3, 4, 1, 0, 1
    a.workers, a.llvip_limit = 0, 50

from trg13 import paths, data as DT, maps as MP, losses as LS, metrics as ME, sampler as SM
from trg13 import model as MD

P = paths.resolve()
for k in ("MSRS", "CKPT"):
    assert P[k] and os.path.exists(P[k]), f"{k} not found ({P[k]}); set env var {k if k != 'CKPT' else 'VIIS_CKPT'}"
OUT = f"{P['RUNS']}/{a.name}"; os.makedirs(OUT, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
def log(s): print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)
log(f"paths: {P}")

# ------------------------------------------------------------------------------------------------ model
model = MD.build(HERE, P["CKPT"], device=dev, mode=a.mode, ema_decay=a.ema, use_tpa=not a.no_tpa)
params = [p for p in model.parameters() if p.requires_grad]
p_tpa = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("model.tpa.")]
id_tpa = {id(p) for p in p_tpa}
p_pre = [p for p in params if id(p) not in id_tpa]
opt = torch.optim.AdamW([dict(params=p_pre, scale=1.0), dict(params=p_tpa, scale=(a.lr_tpa / a.lr) if a.lr > 0 else 0.0)] if p_tpa else
                        [dict(params=p_pre, scale=1.0)], lr=a.lr, weight_decay=0.0)
lp = None
if a.w_lp > 0:
    try:
        import lpips
        lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        for p in lp.parameters():
            p.requires_grad = False
    except Exception as e:
        log(f"LPIPS unavailable ({e}); continuing without the LPIPS term (pip install lpips, internet on)")
lp_eval = None
try:
    import lpips as _lp
    _alex = _lp.LPIPS(net="alex", verbose=False).to(dev).eval()
    def lp_eval(x01, y01):
        with torch.no_grad():
            t = lambda z: torch.from_numpy(z).permute(2, 0, 1)[None].float().to(dev) * 2 - 1
            return float(_alex(t(x01), t(y01)))
except Exception:
    pass

# ------------------------------------------------------------------------------------------------ data
tr_names = DT.msrs_names(P["MSRS"], "train", "D"); va_names, _ = DT.msrs_test_split(P["MSRS"])
if a.smoke:
    tr_names = tr_names[:16]
def _wi(_):
    cv2.setNumThreads(0); torch.set_num_threads(1); np.random.seed((torch.initial_seed() + _) % 2 ** 32)
ms_dl = DataLoader(DT.MSRSPairs(P["MSRS"], "train", names=tr_names), batch_size=a.bs, shuffle=True,
                   num_workers=a.workers, drop_last=True, worker_init_fn=_wi, persistent_workers=a.workers > 0)
night_tr = DT.night_items(P["MSRS"], P["LLVIP"], "train", llvip_limit=a.llvip_limit)
use_night = a.lam_night > 0 and len(night_tr) >= a.night_bs
nt_dl = DataLoader(DT.NightPool(night_tr), batch_size=a.night_bs, shuffle=True, num_workers=a.workers, drop_last=True,
                   worker_init_fn=_wi, persistent_workers=a.workers > 0) if use_night else None
sev_cycle = ["light", "low", "dark"]
val_ms = []
for j, n in enumerate(va_names[:a.n_val]):
    d = DT.MSRSPairs(P["MSRS"], "test", names=[n], train=False, severity=sev_cycle[j % 3], seed_base=40000 + j, att_p=0.5)[0]
    val_ms.append(d)
val_nt = [DT.NightPool(DT.night_items(P["MSRS"], P["LLVIP"], "val"), train=False, crop=(0.6, 0.6))[i]
          for i in range(min(a.n_val, len(DT.night_items(P["MSRS"], P["LLVIP"], "val"))))]
log(f"MSRS train {len(tr_names)} / val {len(val_ms)} | night pool {len(night_tr)} (use={use_night}) / val {len(val_nt)}")

# ------------------------------------------------------------------------------------------------ calibration
STATE = f"{OUT}/state.json"
state = json.load(open(STATE)) if os.path.exists(STATE) else {}
if "calib" not in state:
    gts, irs = [], []
    cal_ds = DT.MSRSPairs(P["MSRS"], "train", names=tr_names[::max(1, len(tr_names) // 64)][:64], train=False, seed_base=60000)
    for i in range(0, len(cal_ds), 8):
        b = [cal_ds[j] for j in range(i, min(i + 8, len(cal_ds)))]
        gts.append(torch.stack([torch.from_numpy(x["gt"]) for x in b]).permute(0, 3, 1, 2).to(dev))
        irs.append(torch.stack([torch.from_numpy(x["ir"]) for x in b]).permute(0, 3, 1, 2).to(dev))
    state["calib"] = LS.calibrate(gts, irs)
    if not all(np.isfinite(v) for v in state["calib"].values()):
        state["calib"] = dict(tau_struct=0.35, tau_pres=0.08)
    del gts, irs
calib = state["calib"]
log(f"calibrated natural thermal-consistency levels on MSRS GT: {calib}")
json.dump(dict(vars(a), calib=calib, paths=P), open(f"{OUT}/config.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ resume
start = 0
if os.path.exists(f"{OUT}/latest.pt") and not a.smoke:
    start = MD.load_ckpt(model, f"{OUT}/latest.pt", opt)["step"]; log(f"resumed at step {start}")

# ------------------------------------------------------------------------------------------------ validation
def preview(rows, titles, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n, m = len(rows), len(titles)
    fig, ax = plt.subplots(n, m, figsize=(2 * m, 2 * n), squeeze=False)
    for r in range(n):
        for c in range(m):
            im = rows[r][c]
            if isinstance(im, str):
                ax[r, c].text(0.5, 0.5, im, ha="center", va="center", fontsize=7)
            elif im is not None:
                ax[r, c].imshow(np.clip(im, 0, 1))
            ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(titles[c], fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=70); plt.close(fig)


@torch.no_grad()
def evaluate(step):
    model.eval()
    o_ms, o_nt = [], []
    with MD.ema_scope(model):
        for d in val_ms:
            o_ms.append(SM.enhance(model, d["rgb"], d["ir"], steps=a.eval_steps, seed=123))
        for d in val_nt:
            o_nt.append(SM.enhance(model, d["rgb"], d["ir"], steps=a.eval_steps, seed=123))
    model.train()
    u = lambda z: (z + 1) / 2
    fr = [ME.full_ref_all(o, u(d["gt"]), u(d["rgb"]), u(d["ir"])[..., 0], lp_eval) for o, d in zip(o_ms, val_ms)]
    nr = [ME.no_ref_all(o, u(d["rgb"]), u(d["ir"])[..., 0]) for o, d in zip(o_nt, val_nt)]
    nm = lambda rows, k: float(np.nanmean([r.get(k, np.nan) for r in rows])) if rows else float("nan")
    res = dict(step=step, psnr=nm(fr, "psnr"), ssim=nm(fr, "ssim"), lpips=nm(fr, "lpips"), psnr_hot=nm(fr, "psnr_hot"),
               tsc_gap=nm(fr, "tsc_gap"), obj_ratio=nm(fr, "obj_ratio"),
               n_tsc=nm(nr, "tsc"), n_tes=nm(nr, "tes"), n_obj=nm(nr, "obj"), n_fid=nm(nr, "in_fid"), n_hf=nm(nr, "hf"))
    for sev in range(3):
        sel = [r for r, j in zip(fr, range(len(fr))) if j % 3 == sev]
        res[f"psnr_{sev_cycle[sev]}"] = nm(sel, "psnr")
    lpv = res["lpips"] if np.isfinite(res["lpips"]) else 0.0
    tg = res["tsc_gap"] if np.isfinite(res["tsc_gap"]) else 0.0
    msrs_score = res["ssim"] + res["psnr"] / 50 - lpv - 0.5 * tg
    night_score = (res["n_fid"] if np.isfinite(res["n_fid"]) else 0) + (res["n_tsc"] if np.isfinite(res["n_tsc"]) else 0) \
        + 0.02 * (res["n_obj"] if np.isfinite(res["n_obj"]) else 0)
    res["msrs_score"], res["night_score"] = msrs_score, night_score
    if a.select == "night":
        res["score"] = night_score + 0.5 * msrs_score
        p0 = state.get("step0", {}).get("psnr", res["psnr"]) if step > 0 else res["psnr"]
        res["eligible"] = bool(res["psnr"] >= p0 - a.msrs_tol)
    else:
        res["score"], res["eligible"] = msrs_score, True
    ref = {k: res[k] for k in ("n_hf", "n_fid")} if step == 0 else state.get("ref", {})
    flag = False
    if np.isfinite(res["n_hf"]) and np.isfinite(ref.get("n_hf", np.nan)) and ref["n_hf"] > 0:
        res["hf_ratio"] = res["n_hf"] / ref["n_hf"]; flag |= res["hf_ratio"] > a.guard_hf
    if np.isfinite(res["n_fid"]) and np.isfinite(ref.get("n_fid", np.nan)):
        res["fid_drop"] = ref["n_fid"] - res["n_fid"]; flag |= res["fid_drop"] > a.guard_fid
    res["flag"] = bool(flag)
    prev0 = PREV0.setdefault("ms", o_ms) if step == 0 else PREV0.get("ms", [None] * len(o_ms))
    prev0n = PREV0.setdefault("nt", o_nt) if step == 0 else PREV0.get("nt", [None] * len(o_nt))
    rows = [(u(d["rgb"]), u(d["ir"]), p0, o, u(d["gt"])) for d, o, p0 in zip(val_ms, o_ms, prev0)]
    rows += [(u(d["rgb"]), u(d["ir"]), p0, o, "real night: no GT") for d, o, p0 in zip(val_nt, o_nt, prev0n)]
    preview(rows, ["input", "thermal", "VIIS (step 0, visual only)", f"v13 @{step}", "GT (MSRS rows only)"], f"{OUT}/prev_{step:05d}.png")
    print("EVAL " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    with open(f"{OUT}/eval.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    return res

PREV0 = {}
if start == 0:
    log("step-0 evaluation (= pretrained VIIS: the adapter is zero-initialised)")
    r0 = evaluate(0); state["ref"] = {k: r0[k] for k in ("n_hf", "n_fid")}; state["step0"] = r0
    # step 0 (= pretrained VIIS) is the baseline every later checkpoint must BEAT to become best.pt
    MD.save_ckpt(model, f"{OUT}/best.pt", 0, extra=dict(calib=calib, eval=r0, note="step 0 = pretrained VIIS"))
    state["best"] = dict(score=r0["score"], step=0)
    json.dump(state, open(STATE, "w"), indent=1)

# ------------------------------------------------------------------------------------------------ training
def lr_at(step):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    pr = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * pr)))

def inf(dl):
    while True:
        for b in dl:
            yield b

ms_it = inf(ms_dl); nt_it = inf(nt_dl) if use_night else None
t_lo, t_hi = (int(x) for x in a.night_t.split(","))
bchw = lambda z: z.to(dev).permute(0, 3, 1, 2).float()
best = state.get("best", dict(score=-1e9, step=None)); bad = state.get("bad", 0); flags = 0
logf = open(f"{OUT}/log.jsonl", "a")
model.train()
step = start
for step in range(start, a.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step) * g.get("scale", 1.0)
    ramp = min(1.0, (step + 1) / max(1, a.ramp))
    opt.zero_grad(set_to_none=True)
    rec = dict(step=step, lr=lr_at(step))

    # ---------------- MSRS: real GT ----------------
    d = next(ms_it)
    gt, rgb, ir = d["gt"].to(dev), d["rgb"].to(dev), d["ir"].to(dev)
    B = gt.shape[0]
    drop = torch.rand(B, device=dev) < a.th_drop
    ir_in = torch.where(drop.view(-1, 1, 1, 1), torch.full_like(ir, -1.0), ir)
    with torch.no_grad():
        x_l, c1_l, c2_l, c1_img, c2_img = model.get_input((gt, rgb, ir_in, None), None, return_original_cond=True)
        ir_img, gt_img = bchw(ir), bchw(gt)
        maps = MP.adapter_maps(c1_img, ir_img, drop)
        s_img = MP.thermal_saliency(ir_img)
        s_lat = F.interpolate(s_img, size=x_l.shape[-2:], mode="area")
    t = torch.randint(0, model.num_timesteps, (B,), device=dev).long()
    if not a.t_uniform:
        n0 = min(a.n_img, B); t[:n0] = torch.randint(0, a.t_img, (n0,), device=dev)
    noise = torch.randn_like(x_l); x_t = model.q_sample(x_l, t, noise)
    with MD.tpa_maps(model, maps):
        eps = model.apply_model(x_t, t, c1_l, c2_l)
    w = 1 + a.hot_w * s_lat
    l_eps = ((w * (eps - noise) ** 2).sum([1, 2, 3]) / (w.sum([1, 2, 3]) * eps.shape[1])).mean()
    loss = l_eps
    zero = torch.zeros((), device=dev)
    il = dict(l1=zero, ssim=zero, lpips=zero); l_st = l_pr = zero
    sel = (t < a.t_img).nonzero().flatten()[:a.n_img]
    use_img = len(sel) > 0 and (a.w_l1 > 0 or a.w_ssim > 0 or a.w_lp > 0 or (a.thermal_loss and a.lam_tc > 0))
    if use_img:
        img = model.differentiable_decode_first_stage(model.predict_start_from_noise(x_t[sel], t[sel], eps[sel]))
        g_s, i_s, s_s = gt_img[sel], ir_img[sel], s_img[sel]
        il = LS.image_losses(img, g_s, 1 + a.hot_w * s_s, lp)
        loss = loss + a.w_l1 * il["l1"] + a.w_ssim * il["ssim"] + a.w_lp * il["lpips"]
        if a.thermal_loss and a.lam_tc > 0:
            keep = (~drop[sel]).float().view(-1, 1, 1, 1)
            with torch.no_grad():
                tau_s = LS.thermal_corr(g_s, i_s)
                C_gt, _ = LS.object_contrast_map(g_s, s_s)
            l_st = LS.thermal_struct_loss(img, i_s, LS.struct_weight(i_s) * keep, tau_s)
            l_pr = LS.presence_loss(img, s_s, keep.expand_as(s_s), C_gt.clamp(max=a.pres_cap))
            loss = loss + ramp * a.lam_tc * (a.w_struct * l_st + a.w_pres * l_pr)
    else:
        img = None
    loss.backward()
    f_ = lambda v: float(v.detach())
    rec.update(eps=f_(l_eps), l1=f_(il["l1"]), ssim=f_(il["ssim"]), lpips=f_(il["lpips"]), st=f_(l_st), pres=f_(l_pr),
               n_img=int(len(sel)) if use_img else 0)
    del eps, img, x_t, loss

    # ---------------- MSRS sampling-trajectory loss (real GT) ----------------
    # The q_sample loss above only ever shows the network noisy versions of the TRUE latent. At inference it sees its own
    # DDIM trajectory starting from pure noise, which at high t carries no trace of the GT brightness (the LDM linear
    # schedule never reaches zero SNR). Without this term every MSRS-only arm drifted darker (sweep 2: mean luminance
    # 0.29 -> 0.21) while its training loss improved. Here: deterministic DDIM walk from pure noise, conditioned on the
    # degraded input, down to a random t_stop (no grad), one step with grad, decode, compare with the REAL day image.
    l_tr = torch.zeros((), device=dev)
    if a.w_traj > 0 and step % a.traj_every == 0:
        k = int(torch.randint(0, B, (1,)))
        t_stop = int(np.random.randint(0, model.num_timesteps))
        n_walk = max(1, min(4, int(round((model.num_timesteps - 1 - t_stop) / 200)) + 1))
        with torch.no_grad():
            m_k = maps[k:k + 1]
        x0k, _ = SM.ddim_x0_train(model, c1_l[k:k + 1], c2_l[k:k + 1], m_k, t_stop, n_steps=n_walk)
        img_k = model.differentiable_decode_first_stage(x0k)
        g_k = gt_img[k:k + 1]
        lo = lambda z: F.avg_pool2d(z, 8)                                     # global tone / colour at 1/8 resolution
        l_tr = (lo(img_k) - lo(g_k)).abs().mean()
        if t_stop < a.t_img:                                                  # fine detail only where it is predictable
            il_k = LS.image_losses(img_k, g_k, 1 + a.hot_w * s_img[k:k + 1], lp)
            l_tr = l_tr + 0.5 * (il_k["l1"] + 0.5 * il_k["ssim"] + 0.2 * il_k["lpips"])
        # x0 = (x_t - sqrt(1-ab) eps) / sqrt(ab): at high t an eps error is amplified 1/sqrt(ab) (~15x at t=999), which gave
        # gradient norms >100 in the v13.1 smoke test and would swamp every other term after clipping. Weight by
        # min(1, sqrt(SNR)) so the gradient reaching the eps prediction is O(1) at every t_stop.
        ab = float(model.alphas_cumprod[t_stop])
        w_t = min(1.0, (ab / (1.0 - ab)) ** 0.5)
        (ramp * a.w_traj * w_t * l_tr).backward()
        rec["traj_w"] = round(w_t, 4)
        del img_k, x0k
    rec["traj"] = float(l_tr.detach())

    # ---------------- real night: no GT, bounded self-supervision ----------------
    if use_night and step % a.night_every == 0:
        dn = next(nt_it)
        rgb_n, ir_n = bchw(dn["rgb"]), bchw(dn["ir"])
        with torch.no_grad():
            c1n, c2n = model.get_learned_conditioning(rgb_n), model.get_learned_conditioning(ir_n)
            maps_n = MP.adapter_maps(rgb_n, ir_n)
            R = MP.routing(rgb_n, ir_n, sigma_cap=a.night_sigma_cap)
        t_stop = int(np.random.randint(t_lo, t_hi))
        x0n, _ = SM.ddim_x0_train(model, c1n, c2n, maps_n, t_stop)
        img_n = model.differentiable_decode_first_stage(x0n)
        l_in = LS.input_struct_loss(img_n, rgb_n, R["w_vis"])
        loss_n = a.w_in * l_in
        n_st = n_pr = torch.zeros((), device=dev)
        if a.thermal_loss:
            n_st = LS.thermal_struct_loss(img_n, ir_n, LS.struct_weight(ir_n) * (1 - R["q"]), calib["tau_struct"])
            n_pr = LS.presence_loss(img_n, R["s"], 1 - R["q"], calib["tau_pres"])
            loss_n = loss_n + a.w_struct * n_st + a.w_pres * n_pr
        (ramp * a.lam_night * loss_n).backward()
        rec.update(n_in=f_(l_in), n_st=f_(n_st), n_pres=f_(n_pr), cov_vis=float(R["w_vis"].mean()),
                   cov_th=float(R["w_th"].mean()))
        del img_n, x0n, loss_n

    gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
    opt.step(); model.model_ema(model.model)
    rec["gnorm"] = float(gn)
    logf.write(json.dumps(rec) + "\n"); logf.flush()
    if step % 25 == 0 or a.smoke:
        mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}) + f" mem {mem:.1f}GB")

    done = step + 1
    if done % a.eval_every == 0 or done == a.steps:
        res = evaluate(done)
        MD.save_ckpt(model, f"{OUT}/latest.pt", done, opt, extra=dict(calib=calib))
        improved = (not res["flag"]) and res.get("eligible", True) and res["score"] > best["score"] + 1e-4
        if improved:
            best = dict(score=res["score"], step=done)
            MD.save_ckpt(model, f"{OUT}/best.pt", done, extra=dict(calib=calib, eval=res))
            log(f"NEW BEST {best}")
        elif done >= a.min_steps:
            bad += 1
        flags = flags + 1 if res["flag"] else 0
        state.update(best=best, bad=bad); json.dump(state, open(STATE, "w"), indent=1)
        if flags >= 2:
            log("STOP: hallucination guard tripped twice in a row (night HF energy / input fidelity)"); break
        if bad >= a.patience:
            log(f"STOP: no validation improvement for {a.patience} evaluations (converged)"); break

MD.save_ckpt(model, f"{OUT}/final.pt", step + 1, extra=dict(calib=calib))
if best["step"] is None:
    shutil.copy(f"{OUT}/final.pt", f"{OUT}/best.pt"); log("no step-0 baseline recorded (resumed run); best.pt = final.pt")
elif best["step"] == 0:
    log("WARNING: no fine-tuned checkpoint beat pretrained VIIS on MSRS validation; best.pt = step 0 (VIIS). "
        "Training did not help -- see eval.jsonl")
open(f"{OUT}/done.txt", "w").write(json.dumps(best))
log(f"DONE  best={best}  outputs in {OUT}")
