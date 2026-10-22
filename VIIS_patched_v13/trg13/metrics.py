"""Evaluation metrics (numpy/OpenCV). Images: HxWx3 float in [0,1]; thermal: HxW float in [0,1].

Full reference (MSRS synthetic, real GT)   : PSNR, SSIM, LPIPS(optional), hot-region PSNR, thermal-consistency gap to GT.
Thermal consistency (all sets)             : TSC  = mean |local NCC(output luma, thermal)| on salient, structured thermal
                                             pixels (bounded in [0,1], NOT increased by adding texture);
                                             TES  = chance-corrected recall of strong thermal edges in the output.
Object preservation                        : OBJ  = Lab colour distance between hot objects and their surrounding ring;
                                             BOX  = same for annotated LLVIP person boxes; YOLO person recall (infer).
Hallucination / fidelity guards (no GT)    : IN_FID = local NCC of output with the (denoised, lifted) input where the
                                             input has real texture;  HF = high-frequency energy in "both modalities
                                             poor" regions (texture that neither input supports).
"""
import numpy as np
import cv2
from skimage.metrics import structural_similarity as _ssim

from .degrade import thermal_saliency_np


def luma(x01):
    return (0.299 * x01[..., 0] + 0.587 * x01[..., 1] + 0.114 * x01[..., 2]).astype(np.float32)


def psnr(a, b, mask=None):
    e = (a.astype(np.float64) - b) ** 2
    e = e.mean() if mask is None else (e * mask[..., None]).sum() / (mask.sum() * 3 + 1e-8)
    return float(10 * np.log10(1.0 / max(e, 1e-10)))


def ssim(a, b):
    return float(_ssim(a.astype(np.float32), b.astype(np.float32), channel_axis=2, data_range=1.0))


def _box(x, k):
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)


def lncc(a, b, k=15):
    ma, mb = _box(a, k), _box(b, k)
    va = np.clip(_box(a * a, k) - ma * ma, 0, None)
    vb = np.clip(_box(b * b, k) - mb * mb, 0, None)
    cov = _box(a * b, k) - ma * mb
    return cov / np.sqrt(va * vb + 1e-6), np.sqrt(va), np.sqrt(vb)


def thermal_weight(th01):
    """Pixels where thermal is informative: salient (hot object, dilated) AND locally structured."""
    s = thermal_saliency_np(th01)
    tb = cv2.GaussianBlur(th01, (0, 0), 1.5)
    _, _, sd = lncc(tb, tb)
    c = np.clip((sd - 0.02) / 0.04, 0, 1)
    return s * c


def tsc(out01, th01, w=None):
    """Thermal structural consistency in [0,1]."""
    w = thermal_weight(th01) if w is None else w
    if w.sum() < 20:
        return float("nan")
    y = cv2.GaussianBlur(luma(out01), (0, 0), 1.5)
    t = cv2.GaussianBlur(th01.astype(np.float32), (0, 0), 1.5)
    r, _, _ = lncc(y, t)
    return float((np.abs(r) * w).sum() / w.sum())


def thermal_edge_score(out01, th01, top=0.05, tol=1):
    y = cv2.GaussianBlur((luma(out01) * 255).astype(np.float32), (0, 0), 1.0)
    t = cv2.GaussianBlur((th01 * 255).astype(np.float32), (0, 0), 1.0)
    g = lambda z: cv2.magnitude(cv2.Sobel(z, cv2.CV_32F, 1, 0), cv2.Sobel(z, cv2.CV_32F, 0, 1))
    mt, my = g(t), g(y)
    tm = mt > np.percentile(mt, 100 * (1 - top))
    ym = my > np.percentile(my, 100 * (1 - top))
    ym = cv2.dilate(ym.astype(np.uint8), np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
    if tm.sum() == 0:
        return float("nan")
    rec, chance = (tm & ym).sum() / tm.sum(), ym.mean()
    return float(np.clip((rec - chance) / (1 - chance + 1e-8), -1, 1))


def lab(x01):
    return cv2.cvtColor(np.clip(x01, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)   # L 0..100, a/b ~ -128..127


def object_contrast(out01, th01, ring=15):
    s = thermal_saliency_np(th01)
    obj = s > 0.5
    if obj.sum() < 30:
        return float("nan")
    rg = (cv2.dilate(obj.astype(np.uint8), np.ones((ring, ring), np.uint8)) > 0) & (s < 0.1)
    if rg.sum() < 30:
        return float("nan")
    L = lab(out01)
    return float(np.linalg.norm(L[obj].mean(0) - L[rg].mean(0)))


def box_contrast(out01, boxes, margin=8):
    if not boxes:
        return float("nan")
    L = lab(out01); H, W = L.shape[:2]; vals = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        X1, Y1, X2, Y2 = max(0, x1 - margin), max(0, y1 - margin), min(W, x2 + margin), min(H, y2 + margin)
        m = np.ones((Y2 - Y1, X2 - X1), bool); m[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
        if m.sum() < 10:
            continue
        vals.append(np.linalg.norm(L[y1:y2, x1:x2].reshape(-1, 3).mean(0) - L[Y1:Y2, X1:X2][m].mean(0)))
    return float(np.mean(vals)) if vals else float("nan")


def visible_quality_np(rgb01, lum_lo=0.25, lum_hi=0.45, cnr_lo=1.5, cnr_hi=4.0, tex_floor=0.5, q_lo=0.15, q_hi=0.75):
    """numpy mirror of trg13.maps.visible_quality (exposure x texture-above-noise)."""
    sm = lambda x, lo, hi: (lambda t: t * t * (3 - 2 * t))(np.clip((x - lo) / (hi - lo), 0, 1))
    V = rgb01.max(2).astype(np.float32)
    Vl = _box(V, 9)
    q_exp = sm(Vl, lum_lo, lum_hi) * (1 - sm(Vl, 0.93, 0.99))
    Y = luma(rgb01); Yd = _box(Y, 5)
    sigma = float(np.clip(np.percentile(np.abs(Y - Yd), 25) / 0.3186, 1e-3, 0.02))
    kt = 17
    std = np.sqrt(np.clip(_box(Yd * Yd, kt) - _box(Yd, kt) ** 2, 0, None))
    tex = sm(std / sigma, cnr_lo, cnr_hi)
    return sm(q_exp * (tex_floor + (1 - tex_floor) * tex), q_lo, q_hi).astype(np.float32)


def input_fidelity(out01, in01):
    """Local NCC between output luma and the denoised, gamma-lifted input luma, where the input carries real texture."""
    q = visible_quality_np(np.power(np.clip(in01, 0, 1), 0.45))
    if q.sum() < 50:
        return float("nan")
    # log-luminance: NCC is invariant to gain AND gamma there (log x^g = g log x), so brightening is not penalised
    yi = np.log(cv2.GaussianBlur(luma(in01), (0, 0), 1.5) + 0.01)
    yo = np.log(cv2.GaussianBlur(luma(out01), (0, 0), 1.5) + 0.01)
    r, _, _ = lncc(yo, yi, 11)
    return float((r * q).sum() / q.sum())


def hf_energy(out01, in01, th01):
    """Mean |Y - blur(Y)| in regions where neither input is informative (dark, textureless RGB and non-salient thermal)."""
    q = visible_quality_np(np.power(np.clip(in01, 0, 1), 0.45))
    s = thermal_saliency_np(th01)
    w = (1 - q) * (1 - s)
    y = luma(out01)
    hf = np.abs(y - cv2.blur(y, (3, 3)))
    return float((hf * w).sum() / (w.sum() + 1e-6))


def colorfulness(x01):
    r, g, b = [x01[..., i] * 255 for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def entropy(x01):
    g = (np.clip(luma(x01), 0, 1) * 255).astype(np.uint8)
    h = np.bincount(g.ravel(), minlength=256).astype(np.float64) / g.size
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def no_ref_all(out01, in01, th01, boxes=None):
    return dict(tsc=tsc(out01, th01), tes=thermal_edge_score(out01, th01), obj=object_contrast(out01, th01),
                box=box_contrast(out01, boxes or []), in_fid=input_fidelity(out01, in01), hf=hf_energy(out01, in01, th01),
                color=colorfulness(out01), entropy=entropy(out01), mean_l=float(luma(out01).mean()))


def full_ref_all(out01, gt01, in01, th01, lpips_fn=None):
    s = thermal_saliency_np(th01)
    r = dict(psnr=psnr(out01, gt01), ssim=ssim(out01, gt01), psnr_hot=psnr(out01, gt01, s) if s.sum() > 30 else float("nan"))
    w = thermal_weight(th01)
    t_out, t_gt = tsc(out01, th01, w), tsc(gt01, th01, w)
    r.update(tsc=t_out, tsc_gt=t_gt, tsc_gap=abs(t_out - t_gt) if np.isfinite(t_out) and np.isfinite(t_gt) else float("nan"))
    o, og = object_contrast(out01, th01), object_contrast(gt01, th01)
    r.update(obj=o, obj_gt=og, obj_ratio=o / og if np.isfinite(o) and np.isfinite(og) and og > 1e-3 else float("nan"))
    if lpips_fn is not None:
        r["lpips"] = float(lpips_fn(out01, gt01))
    return r
