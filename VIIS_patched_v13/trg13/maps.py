"""Per-pixel modality maps (torch, no parameters, computed under no_grad). Images: Bx3xHxW in [-1,1].

  s  thermal saliency      : hot relative to surroundings AND absolutely warm, dilated + feathered (object + its border)
  q  visible quality       : exposure x texture-above-noise of the RGB input  (port of the user's v11/v12 'which modality
                             to trust per pixel' idea; in v13 it routes the consistency losses and is an adapter input,
                             it never selects a pseudo-target)
  routing weights          : w_vis = q,  w_th = (1-q) s,  w_fb = (1-q)(1-s)   (sum to 1)
  adapter maps (5 ch)      : thermal luma, thermal edges, s, q, gamma-lifted RGB luma
All kernel sizes are FIXED IN PIXELS (set at the 256-px training scale). Inference resizes frames so objects have the
training pixel scale and tiles the latent, so the maps mean the same thing in training and at inference.
"""
import torch
import torch.nn.functional as F

_KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)


def luminance(img):
    x = (img.float().clamp(-1, 1) + 1) / 2
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def odd(k):
    k = int(round(k))
    return max(3, k + 1 - (k % 2))


def blur(x, k):
    k = odd(k)
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, 1)


def gauss(x, sigma):
    if sigma <= 0:
        return x
    r = max(1, int(round(3 * sigma)))
    g = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma * sigma))
    g = g / g.sum()
    C = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(x, g.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def sobel(y):
    kx = _KX.to(y.device, y.dtype)
    yp = F.pad(y, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(yp, kx), F.conv2d(yp, kx.transpose(2, 3))
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def smoothstep(x, lo, hi):
    t = ((x - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
    return t * t * (3 - 2 * t)


@torch.no_grad()
def thermal_saliency(ir, thr=0.08, abs_thr=0.40, sharp=40.0):
    T = luminance(ir)
    bg = blur(T, 31)
    s = torch.sigmoid((T - bg - thr) * sharp) * torch.sigmoid((T - abs_thr) * 20)
    s = F.max_pool2d(s, 9, 1, 4)
    return blur(s, 5).clamp(0, 1)


@torch.no_grad()
def visible_quality(rgb, lum_lo=0.25, lum_hi=0.45, cnr_lo=1.5, cnr_hi=4.0, sigma_cap=0.02, tex_floor=0.5, q_lo=0.15, q_hi=0.75):
    x = (rgb.float().clamp(-1, 1) + 1) / 2
    x = x.clamp(1e-6, 1) ** 0.45      # judge what is recoverable after a display lift (same as the metric); thresholds are
    #                                   in lifted units: 0.25 / 0.45 lifted ~ 0.046 / 0.17 linear-ish sRGB
    V = x.max(1, keepdim=True)[0]
    Vl = blur(V, 8)
    q_exp = smoothstep(Vl, lum_lo, lum_hi) * (1 - smoothstep(Vl, 0.93, 0.99))
    Y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    Yd = blur(Y, 5)
    res = (Y - Yd).abs().flatten(1)
    sigma = (torch.quantile(res, 0.25, dim=1) / 0.3186).view(-1, 1, 1, 1).clamp(1e-3, sigma_cap)
    kt = 16
    std = (blur(Yd * Yd, kt) - blur(Yd, kt) ** 2).clamp(min=0).sqrt()
    tex = smoothstep(std / sigma, cnr_lo, cnr_hi)
    return smoothstep(q_exp * (tex_floor + (1 - tex_floor) * tex), q_lo, q_hi)


@torch.no_grad()
def thermal_structure_conf(ir, sigma=1.5):
    """1 where the thermal image has local structure (so a correlation with it is meaningful)."""
    T = gauss(luminance(ir), sigma)
    k = 15
    sd = (blur(T * T, k) - blur(T, k) ** 2).clamp(min=0).sqrt()
    return smoothstep(sd, 0.02, 0.06)


@torch.no_grad()
def routing(rgb, ir, sigma_cap=0.02):
    q, s = visible_quality(rgb, sigma_cap=sigma_cap), thermal_saliency(ir)
    return dict(q=q, s=s, w_vis=q, w_th=(1 - q) * s, w_fb=(1 - q) * (1 - s))


@torch.no_grad()
def adapter_maps(rgb, ir, drop=None):
    """-> Bx5xHxW in roughly [-1,1]. drop: optional B bool tensor = thermal dropped (maps from a black thermal)."""
    T = luminance(ir)
    e = sobel(T)
    e = e / (torch.quantile(e.flatten(1), 0.99, dim=1).view(-1, 1, 1, 1) + 1e-4)
    s = thermal_saliency(ir)
    q = visible_quality(rgb)
    Yl = luminance(rgb).clamp(1e-6, 1) ** 0.45
    m = torch.cat([T, e.clamp(0, 1), s, q, Yl], 1) * 2 - 1
    if drop is not None and bool(drop.any()):
        z = torch.cat([torch.zeros_like(T), torch.zeros_like(T), torch.zeros_like(T), q, Yl], 1) * 2 - 1
        m = torch.where(drop.view(-1, 1, 1, 1), z, m)
    return m
