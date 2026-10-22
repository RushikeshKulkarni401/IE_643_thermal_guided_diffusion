"""Deliverable 1 (spatial alignment): coarse-to-fine ECC registration of thermal to RGB on gradient maps.

RGB and thermal differ in appearance (a dark coat can be hot, a bright sign cold), so intensities are not comparable;
their GRADIENT maps share object boundaries. ECC (enhanced correlation coefficient) is invariant to gain/offset and is
maximised on blurred gradient magnitudes, from coarse to fine. MSRS and LLVIP ship pre-registered, so in training the
module is used (a) to measure residual misregistration and (b) at inference for user pairs (--align). Training adds
small random misregistrations (degrade.misregister) so the network itself tolerates a few pixels of error.
"""
import numpy as np
import cv2


def grad_map(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = img.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    # lift dark RGB so its edges are measurable (gain/offset do not matter to ECC, gamma does)
    g = np.power(np.clip(g, 0, 1), 0.5)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return (m / (np.percentile(m, 99.5) + 1e-6)).clip(0, 1).astype(np.float32)


def align_pair(rgb_u8, ir_u8, motion="euclidean", scales=(4, 2, 1), max_iter=100, eps=1e-5, max_corner_px=None, min_gain=0.005):
    """Estimate W with rgb(x) ~ ir(W x) and return (ir_aligned_u8, W 2x3, ok, cc).
    Default motion is euclidean (shift + small rotation): cross-modal gradient ECC with a full affine is under-constrained
    and drifts in scale (measured on an MSRS pair: affine moved the corners 17 px, euclidean 5.6 px, a self-pair 0.02 px).
    ok=False (identity returned) when ECC fails, when the warp is implausible (corner motion > max_corner_px) or when it
    does not raise the correlation of the gradient maps by at least min_gain over the identity."""
    gv, gi = grad_map(rgb_u8), grad_map(ir_u8)
    h, w = ir_u8.shape[:2]
    if gv.shape != gi.shape:
        gv = cv2.resize(gv, (w, h), interpolation=cv2.INTER_AREA)
    mode = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN, "affine": cv2.MOTION_AFFINE}[motion]
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    cc = float("nan")
    try:
        for li, s in enumerate(scales):
            a = cv2.resize(gv, (w // s, h // s), interpolation=cv2.INTER_AREA) if s > 1 else gv
            b = cv2.resize(gi, (w // s, h // s), interpolation=cv2.INTER_AREA) if s > 1 else gi
            if li > 0:
                warp[:, 2] *= scales[li - 1] / s
            cc, warp = cv2.findTransformECC(a, b, warp, mode, crit, None, 5)
    except cv2.error:
        return ir_u8, np.eye(2, 3, dtype=np.float32), False, float("nan")
    lim = max_corner_px if max_corner_px is not None else 0.08 * max(h, w)
    try:
        cc0 = float(cv2.computeECC(gv, gi))
    except cv2.error:
        cc0 = -1.0
    if corner_error(warp, np.eye(2, 3, dtype=np.float32), w, h) > lim or cc < cc0 + min_gain:
        return ir_u8, np.eye(2, 3, dtype=np.float32), False, float(cc)
    out = cv2.warpAffine(ir_u8, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out, warp, True, float(cc)


def corner_error(Ma, Mb, w, h):
    """Mean displacement (px) of the 4 image corners between two 2x3 affines."""
    c = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float32).T
    return float(np.mean(np.linalg.norm((np.asarray(Ma) - np.asarray(Mb)) @ c, axis=0)))


def edge_overlap(rgb_u8, ir_u8, top=0.08):
    """Fraction of the strongest thermal edges that coincide (+-1 px) with strong RGB edges. Alignment quality proxy."""
    gv, gi = grad_map(rgb_u8), grad_map(ir_u8)
    mv = gv > np.percentile(gv, 100 * (1 - top))
    mi = gi > np.percentile(gi, 100 * (1 - top))
    mv = cv2.dilate(mv.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float((mi & mv).sum() / max(1, mi.sum()))


def _compose(M, W):
    """2x3 affines -> 2x3 of (M o W) as 3x3 product M3 @ W3."""
    a = np.vstack([np.asarray(M, np.float64), [0, 0, 1]]); b = np.vstack([np.asarray(W, np.float64), [0, 0, 1]])
    return (a @ b)[:2].astype(np.float32)


def audit(pairs, shifts=(0, 2, 4, 8), rs=None, motion="euclidean"):
    """Synthetic misregistration audit. For each (rgb, ir) pair: W0 = alignment of the original pair (datasets are already
    registered, so W0 ~ identity up to their residual error); then perturb the thermal by a KNOWN warp M (shift s px,
    rotation 0.3*s deg), re-align and compare with the ideal M o W0.  pert(x) = ir(M^-1 x), aligned(x) = pert(W x)
    = ir(M^-1 W x), and the ideal is M^-1 W = W0, i.e. W = M W0.  Rows report corner error before / after (px)."""
    rs = rs or np.random.RandomState(0)
    rows = []
    for i, (rgb, ir) in enumerate(pairs):
        h, w = ir.shape[:2]
        _, W0, ok0, _ = align_pair(rgb, ir, motion=motion, min_gain=-1.0)
        if not ok0:
            W0 = np.eye(2, 3, dtype=np.float32)
        for s in shifts:
            ang = rs.uniform(-1, 1) * 0.3 * s if motion != "translation" else 0.0
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0).astype(np.float32)
            M[0, 2] += s * rs.choice([-1, 1]); M[1, 2] += s * rs.choice([-1, 1])
            pert = cv2.warpAffine(ir, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            al, W, ok, cc = align_pair(rgb, pert, motion=motion, min_gain=-1.0)
            ideal = _compose(M, W0)
            rows.append(dict(i=i, shift=s, before=corner_error(ideal, W0, w, h),
                             after=corner_error(W, ideal, w, h) if ok else float("nan"), ok=ok, cc=cc,
                             residual_W0=corner_error(W0, np.eye(2, 3, dtype=np.float32), w, h),
                             overlap_before=edge_overlap(rgb, pert), overlap_after=edge_overlap(rgb, al)))
    return rows
