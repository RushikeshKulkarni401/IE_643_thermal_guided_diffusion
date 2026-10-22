"""v13 losses (torch). Images Bx3xHxW in [-1,1] unless stated. Every term is BOUNDED and has a FIXED POINT, so none of
them can be lowered indefinitely by adding texture (the v3 failure) and none uses a model output as a target (the v10-v12
problem).

Supervised (MSRS, real day ground truth):
  eps-MSE (hot-weighted) in the VQ latent, L1 + (1-SSIM) + LPIPS on the decoded x0 estimate.
Thermal consistency (deliverable 4) -- both terms are hinges against a NATURAL level, never "more is better":
  L_struct  = sum_w relu(tau - |LNCC(Y_out, T)|) / sum_w      LNCC = local normalised cross-correlation (window ~15 px)
              * |LNCC| <= 1 and is invariant to gain/offset: amplifying contrast does not help;
              * random texture DEcorrelates and RAISES the loss (opposite of v3's edge reward);
              * tau is the level real day images reach: per pixel from the GT on MSRS (fixed point = GT), a calibrated
                scalar on real night frames, so copying the thermal image into the RGB is not rewarded beyond that.
  L_pres    = sum_M relu(tau_c - ||Lab_in(obj) - Lab_ring(obj)||) / sum_M      (object preservation, deliverable 3)
              sign-agnostic colour/brightness separation of each hot object from its surrounding ring, capped at the
              natural level (GT on MSRS, calibrated on night). Fine texture does not change regional means -> not gameable.
Self-supervised data fidelity on real night frames (no GT):
  L_in      = sum_q (1 - LNCC(log Y_out, log Y_in)) / sum_q   structure of the output must follow the (denoised) RGB input
              wherever the input carries real texture (q). log makes it invariant to brightening (gain and gamma).
"""
import torch
import torch.nn.functional as F

from .maps import luminance, blur, gauss, odd, smoothstep, thermal_saliency, thermal_structure_conf


def wmean(d, w, frac=1e-3):
    """per-image weighted mean, averaged over the batch; images with ~no weight shrink to 0 instead of exploding."""
    den = w.sum([1, 2, 3]) + frac * w.shape[-2] * w.shape[-1]
    return ((d * w).sum([1, 2, 3]) / den).mean()


def lncc(a, b, k, eps=1e-5):
    ma, mb = blur(a, k), blur(b, k)
    va = (blur(a * a, k) - ma * ma).clamp(min=0)
    vb = (blur(b * b, k) - mb * mb).clamp(min=0)
    cov = blur(a * b, k) - ma * mb
    return cov / torch.sqrt(va * vb + eps * eps + eps * (va + vb))


def ssim(a01, b01, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    mu_a, mu_b = gauss(a01, sigma), gauss(b01, sigma)
    saa = gauss(a01 * a01, sigma) - mu_a ** 2
    sbb = gauss(b01 * b01, sigma) - mu_b ** 2
    sab = gauss(a01 * b01, sigma) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return m.mean([1, 2, 3])


def rgb_to_lab01(img):
    """[-1,1] RGB -> Lab scaled by 1/100 (L in [0,1], a/b roughly [-1,1])."""
    x = ((img.float().clamp(-1, 1) + 1) / 2).clamp(1e-5, 1)
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = lambda t: torch.where(t > 0.008856, t.clamp(min=1e-6) ** (1 / 3), 7.787 * t + 16 / 116)
    fx, fy, fz = f(X), f(Y), f(Z)
    return torch.stack([(116 * fy - 16) / 100, 5 * (fx - fy), 2 * (fy - fz)], 1)


# ------------------------------------------------------------------------------------------------ thermal consistency
def thermal_corr(img, ir, sigma=1.5):
    """|LNCC| between output luminance and thermal luminance, Bx1xHxW in [0,1]."""
    y, t = gauss(luminance(img), sigma), gauss(luminance(ir), sigma)
    return lncc(y, t, 15).abs()


def struct_weight(ir):
    return (thermal_saliency(ir) * thermal_structure_conf(ir)).detach()


def thermal_struct_loss(img, ir, w, tau):
    """tau: tensor broadcastable to Bx1xHxW (GT level on MSRS) or float (calibrated level on night frames)."""
    r = thermal_corr(img, ir)
    if torch.is_tensor(tau):
        tau = tau.detach()
    return wmean(F.relu(tau - r), w)


PRES_LUM_ONLY = True   # v13.2: set from train_v13.py --pres_lum


def object_contrast_map(img, s, k_ctx=None, ring=None, lum_only=None):
    """Local separation between the hot object (soft mask s) and its surrounding ring, Bx1xHxW (Lab/100 units).
    lum_only (default PRES_LUM_ONLY): brightness separation only. With colour included (v13.1) the cheapest way for the
    model to separate a dark-clothed pedestrian from sodium-yellow ground was to paint the pedestrian BLUE (opposite of
    yellow on the b* axis): 3.9% strongly-blue pixels on LLVIP vs 1.0% without the term. Thermal evidence says WHERE an
    object is, not what colour it has, so the presence term now acts on L* only."""
    k_ctx = k_ctx or 31
    ring = ring or 15
    lab = rgb_to_lab01(img)
    if (PRES_LUM_ONLY if lum_only is None else lum_only):
        lab = lab[:, :1]
    M = s
    R = (F.max_pool2d(s, ring, 1, ring // 2) - s).clamp(0, 1) * (s < 0.1).float()
    bm, br = blur(M, k_ctx), blur(R, k_ctx)
    inside = blur(lab * M, k_ctx) / (bm + 1e-4)
    outside = blur(lab * R, k_ctx) / (br + 1e-4)
    C = torch.sqrt(((inside - outside) ** 2).sum(1, keepdim=True) + 1e-8)
    valid = smoothstep(br, 0.02, 0.10) * smoothstep(bm, 0.05, 0.20)
    return C, valid.detach()


def presence_loss(img, s, w, tau):
    C, valid = object_contrast_map(img, s)
    if torch.is_tensor(tau):
        tau = tau.detach()
    return wmean(F.relu(tau - C), (w * s * valid).detach())


# ------------------------------------------------------------------------------------------------ input fidelity
def input_struct_loss(img, rgb_in, q):
    """Structure (not noise) of the input: heavier blur and half resolution, so pixel-level sensor noise and JPEG blocks
    in a dark input cannot be a target (the first full run learned to copy them: n_in 0.35 -> 0.06, mottled output)."""
    yo = torch.log(F.avg_pool2d(gauss(luminance(img), 2.5), 2) + 0.01)
    yi = torch.log(F.avg_pool2d(gauss(luminance(rgb_in), 2.5), 2) + 0.01).detach()
    r = lncc(yo, yi, 9)
    return wmean(1 - r, F.avg_pool2d(q.detach(), 2))


# ------------------------------------------------------------------------------------------------ supervised image terms
def image_losses(img, gt, w_hot, lp=None):
    """img/gt [-1,1]; w_hot Bx1xHxW (1 + hot_w*s). Returns dict(l1, ssim, lpips)."""
    l1 = ((w_hot * (img - gt).abs()).sum([1, 2, 3]) / (w_hot.sum([1, 2, 3]) * 3)).mean()
    a01, b01 = (img.clamp(-1, 1) + 1) / 2, (gt + 1) / 2
    l_ssim = (1 - ssim(a01, b01)).mean()
    l_lp = lp(img.clamp(-1, 1), gt).mean() if lp is not None else img.new_zeros(())
    return dict(l1=l1, ssim=l_ssim, lpips=l_lp)


@torch.no_grad()
def calibrate(gt_batches, ir_batches, q_struct=0.5, q_pres=0.4):
    """Natural thermal-consistency levels measured on REAL day images (MSRS train GT): the weighted median |LNCC| and the
    40th-percentile object/ring separation. Used as tau on night frames, where no GT exists."""
    rs, ws, cs, wc = [], [], [], []
    for gt, ir in zip(gt_batches, ir_batches):
        w = struct_weight(ir); r = thermal_corr(gt, ir)
        s = thermal_saliency(ir); C, valid = object_contrast_map(gt, s)
        rs.append(r.flatten()); ws.append(w.flatten()); cs.append(C.flatten()); wc.append((s * valid).flatten())
    def wq(v, w, q):
        v, w = torch.cat(v), torch.cat(w)
        m = w > 0.05
        if m.sum() < 10:
            return float("nan")
        v, w = v[m], w[m]
        o = torch.argsort(v); cw = torch.cumsum(w[o], 0)
        idx = int(torch.searchsorted(cw, (q * cw[-1]).reshape(1))[0].clamp(max=len(v) - 1))
        return float(v[o][idx])
    return dict(tau_struct=wq(rs, ws, q_struct), tau_pres=wq(cs, wc, q_pres))
