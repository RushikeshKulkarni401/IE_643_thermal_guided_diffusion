"""Physically-motivated night / smoke / glare degradation of a clean day image (numpy + OpenCV only).

Why this replaces the VIIS/v3-v12 degradation (contrast x U(0.1,1) -> gamma U(3,10) -> Poisson+Gaussian in sRGB):
  * real night frames (LLVIP, MSRS-N) are NOT uniformly dark: street-lamp pools, warm sodium / LED colour casts, glare,
    JPEG blocking and sensor noise that is signal-dependent in LINEAR light. A model trained only on a global gamma curve
    sees a domain gap on LLVIP, which is the main reason VIIS zero-shot looks hazy/washed out there.
  * the clean day image stays the ONLY training target, so closing this gap is a legitimate way to beat VIIS zero-shot
    on real night images without ever using a model output as a target.

Everything happens in linear light:  lin = srgb^2.2  ->  illumination map * colour cast  ->  fog  ->  glare
  ->  hot-object attenuation (object obscured in RGB, visible only in thermal)  ->  Poisson-Gaussian sensor noise
  ->  camera tone curve  ->  8-bit quantisation  ->  optional blur / JPEG.

Severity presets (exposure scale k of the median illumination):
  light : k in [0.35, 1.0]   mild noise, rarely a cast    -> teaches the model to leave good RGB (almost) alone
  low   : k in [0.06, 0.30]  dusk / lit street
  dark  : k in [0.006, 0.06] night, smoke, glare
"""
import math
import numpy as np
import cv2

SEVERITIES = {"light": (0.35, 1.0), "low": (0.06, 0.30), "dark": (0.006, 0.06)}
DEFAULT_MIX = (("light", 0.15), ("low", 0.35), ("dark", 0.50))


def _loguniform(rs, lo, hi):
    return float(math.exp(rs.uniform(math.log(lo), math.log(hi))))


def thermal_saliency_np(ir_gray01, thr=0.08, abs_thr=0.40, sharp=40.0, k=31):
    """numpy mirror of trg13.maps.thermal_saliency (hot relative to surroundings AND absolutely warm). HxW in [0,1]."""
    t = ir_gray01.astype(np.float32)
    bg = cv2.blur(t, (k, k), borderType=cv2.BORDER_REFLECT)
    s = 1 / (1 + np.exp(-(t - bg - thr) * sharp)) * 1 / (1 + np.exp(-(t - abs_thr) * 20))
    s = cv2.dilate(s, np.ones((9, 9), np.uint8))
    return np.clip(cv2.blur(s, (5, 5)), 0, 1).astype(np.float32)


def _illumination(h, w, rs, k, sev):
    """Spatially varying illumination with median ~k: a smooth gradient plus 0-4 lamp pools."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= h; xx /= w
    g = 1.0 + rs.uniform(-0.5, 0.5) * (xx - 0.5) + rs.uniform(-0.5, 0.5) * (yy - 0.5)
    n_lamps = 0 if sev == "light" else rs.choice([0, 1, 1, 2, 3, 4])
    for _ in range(n_lamps):
        cx, cy = rs.uniform(-0.1, 1.1), rs.uniform(-0.2, 0.9)
        r = rs.uniform(0.10, 0.45)
        amp = rs.uniform(1.0, 6.0)
        g = g + amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    g = np.clip(g, 0.05, None)
    g = g / np.median(g) * k
    return g.astype(np.float32)


def _cast(rs, sev):
    if sev == "light" or rs.rand() < 0.35:                       # near-neutral white balance error
        return np.array([rs.uniform(0.92, 1.08), 1.0, rs.uniform(0.92, 1.08)], np.float32)
    kind = rs.choice(["sodium", "warm_led", "cool_led"], p=[0.5, 0.35, 0.15])
    if kind == "sodium":
        return np.array([rs.uniform(1.15, 1.55), rs.uniform(0.85, 1.05), rs.uniform(0.30, 0.65)], np.float32)
    if kind == "warm_led":
        return np.array([rs.uniform(1.05, 1.30), 1.0, rs.uniform(0.60, 0.90)], np.float32)
    return np.array([rs.uniform(0.80, 0.95), 1.0, rs.uniform(1.05, 1.25)], np.float32)


def degrade(rgb_u8, ir_u8=None, rs=None, severity=None, mix=DEFAULT_MIX, p_fog=0.15, p_glare=0.15, p_att=0.35,
            att_keep=(0.0, 0.3), p_jpeg=0.5, p_blur=0.2, return_info=False):
    """rgb_u8: HxWx3 uint8 clean day image.  ir_u8: HxW uint8 thermal (needed only for hot-object attenuation).
    rs: np.random.RandomState (pass a seeded one for deterministic validation).  Returns HxWx3 uint8 (and info dict)."""
    rs = rs if rs is not None else np.random
    if severity is None:
        names, probs = zip(*mix)
        severity = names[rs.choice(len(names), p=np.array(probs) / sum(probs))]
    lo, hi = SEVERITIES[severity]
    k = _loguniform(rs, lo, hi)
    h, w = rgb_u8.shape[:2]
    lin = (rgb_u8.astype(np.float32) / 255.0) ** 2.2
    lin = lin * _illumination(h, w, rs, k, severity)[..., None] * _cast(rs, severity)[None, None]
    info = dict(severity=severity, k=k, fog=False, glare=False, att=False)

    if severity != "light" and rs.rand() < p_fog:                 # Koschmieder haze / smoke, airlight ~ illumination
        tr = math.exp(-rs.uniform(0.3, 1.2))
        air = k * rs.uniform(0.5, 2.0) * np.array([1.0, rs.uniform(0.9, 1.0), rs.uniform(0.75, 1.0)], np.float32)
        lin = lin * tr + air[None, None] * (1 - tr)
        info["fog"] = True
    if severity != "light" and rs.rand() < p_glare:               # headlights / lamps in view: saturated core + bloom
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for _ in range(rs.randint(1, 3)):
            cx, cy, r = rs.uniform(0.1, 0.9) * w, rs.uniform(0.1, 0.8) * h, rs.uniform(0.02, 0.08) * max(h, w)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            core = np.exp(-d2 / (2 * r * r)) * rs.uniform(2.0, 8.0)
            bloom = np.exp(-d2 / (2 * (4 * r) ** 2)) * rs.uniform(0.05, 0.3)
            lin = lin + (core + bloom)[..., None] * np.array([1.0, rs.uniform(0.85, 1.0), rs.uniform(0.6, 1.0)], np.float32)
        info["glare"] = True
    if ir_u8 is not None and severity != "light" and rs.rand() < p_att:   # object hidden in RGB, present in thermal
        m = thermal_saliency_np(ir_u8.astype(np.float32) / 255.0)
        if m.mean() > 0.002:
            keep = rs.uniform(*att_keep)
            lin = lin * (1 - m[..., None] * (1 - keep))
            info["att"] = True

    # Poisson-Gaussian sensor noise in linear light; darker exposure -> higher analogue gain -> relatively more noise
    gain = float(np.clip(0.25 / max(k, 1e-3), 1.0, 60.0))
    shot = rs.uniform(2e-5, 1.5e-4) * gain
    read = rs.uniform(2e-4, 1.5e-3) * gain ** 0.5
    if severity == "light":
        shot, read = shot * 0.3, read * 0.3
    lin = np.clip(lin, 0, None)
    noisy = lin + rs.randn(*lin.shape).astype(np.float32) * np.sqrt(lin * shot + read ** 2)
    if severity != "light" and rs.rand() < 0.15:                  # mild row / banding noise of cheap sensors
        noisy = noisy + rs.randn(h, 1, 1).astype(np.float32) * read * 0.15
    noisy = np.clip(noisy, 0, 1)
    out = noisy ** (1.0 / rs.uniform(1.8, 2.4))
    out = (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if rs.rand() < p_blur:
        s = rs.uniform(0.4, 1.1)
        out = cv2.GaussianBlur(out, (0, 0), s)
    if rs.rand() < p_jpeg:
        q = int(rs.randint(35, 91))
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    out = np.ascontiguousarray(out)
    return (out, info) if return_info else out


def misregister(ir_u8, rs, max_shift=3.0, max_rot=1.5, max_scale=0.02):
    """Small random affine of the thermal image (training robustness to imperfect registration). Returns (img, 2x3 M)."""
    h, w = ir_u8.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rs.uniform(-max_rot, max_rot), 1 + rs.uniform(-max_scale, max_scale))
    M[0, 2] += rs.uniform(-max_shift, max_shift) * w / 256
    M[1, 2] += rs.uniform(-max_shift, max_shift) * h / 256
    return cv2.warpAffine(ir_u8, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), M.astype(np.float32)


def jitter_thermal(ir_u8, rs):
    t = ir_u8.astype(np.float32) / 255.0
    t = np.clip(t * rs.uniform(0.9, 1.1) + rs.uniform(-0.04, 0.04), 0, 1) ** rs.uniform(0.9, 1.1)
    t = t + rs.randn(*t.shape).astype(np.float32) * rs.uniform(0, 0.01)
    return (np.clip(t, 0, 1) * 255 + 0.5).astype(np.uint8)
