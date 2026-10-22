"""DDIM sampling for v13: 256-px crops directly, full frames by latent tiling (MultiDiffusion-style: at every step the eps
of overlapping 64x64 latent tiles is averaged with feathered weights, so there are no seams and every tile sees exactly the
input statistics it was trained on). Optional classifier-free thermal guidance (trained via thermal dropout)."""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from .maps import adapter_maps
from .model import tpa_maps


def to_bchw(hwc_m11, device):
    return torch.from_numpy(np.ascontiguousarray(hwc_m11)).permute(2, 0, 1)[None].float().to(device)


def prep_frame(rgb_u8, ir_u8, long_side=512, mult=32, min_side=256):
    """Resize a full frame so objects keep roughly the training pixel scale; sides become multiples of 32."""
    h, w = rgb_u8.shape[:2]
    s = long_side / max(h, w)
    nh, nw = max(min_side, int(round(h * s / mult)) * mult), max(min_side, int(round(w * s / mult)) * mult)
    r = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_AREA)
    i = cv2.resize(ir_u8, (nw, nh), interpolation=cv2.INTER_AREA)
    if i.ndim == 2:
        i = np.repeat(i[..., None], 3, 2)
    return (r.astype(np.float32) / 127.5 - 1), (i.astype(np.float32) / 127.5 - 1)


def _positions(n, tile, stride):
    if n <= tile:
        return [0]
    p = list(range(0, n - tile + 1, stride))
    if p[-1] != n - tile:
        p.append(n - tile)
    return p


def _feather(tile, overlap, device):
    r = torch.ones(tile, device=device)
    if overlap > 0:
        ramp = torch.linspace(0.05, 1.0, overlap, device=device)
        r[:overlap] = ramp; r[-overlap:] = ramp.flip(0)
    return (r[:, None] * r[None, :])[None, None]


class Conditioning:
    """Everything the UNet needs for one image, at full resolution."""

    def __init__(self, model, rgb_bchw, ir_bchw, guid=1.0):
        with torch.no_grad():
            self.c1 = model.get_learned_conditioning(rgb_bchw)
            self.c2 = model.get_learned_conditioning(ir_bchw)
            self.maps = adapter_maps(rgb_bchw, ir_bchw)
            self.guid = guid
            if guid != 1.0:
                black = torch.full_like(ir_bchw, -1.0)
                self.c2n = model.get_learned_conditioning(black)
                self.mapsn = adapter_maps(rgb_bchw, ir_bchw, drop=torch.ones(rgb_bchw.shape[0], dtype=torch.bool, device=rgb_bchw.device))


@torch.no_grad()
def _eps_tile(model, x, t, c1, c2, maps):
    with tpa_maps(model, maps):
        return model.apply_model(x, t, c1, c2)


@torch.no_grad()
def eps_full(model, x, tv, cond, tile=64, stride=48, tile_bs=4):
    B, C, h, w = x.shape
    dev = x.device
    ys, xs = _positions(h, tile, stride), _positions(w, tile, stride)
    if len(ys) == 1 and len(xs) == 1:
        t = torch.full((B,), int(tv), device=dev, dtype=torch.long)
        e = _eps_tile(model, x, t, cond.c1, cond.c2, cond.maps)
        if cond.guid != 1.0:
            e0 = _eps_tile(model, x, t, cond.c1, cond.c2n, cond.mapsn)
            e = e0 + cond.guid * (e - e0)
        return e
    f = 4                                                           # VQ-f4: latent px -> image px
    wgt = _feather(tile, tile - stride, dev)
    acc, norm = torch.zeros_like(x), torch.zeros_like(x[:, :1])
    boxes = [(y, xx) for y in ys for xx in xs]
    for k in range(0, len(boxes), tile_bs):
        bb = boxes[k:k + tile_bs]
        cut = lambda z, s=1: torch.cat([z[:, :, y * s:(y + tile) * s, xx * s:(xx + tile) * s] for y, xx in bb], 0)
        xt, c1, c2, mp = cut(x), cut(cond.c1), cut(cond.c2), cut(cond.maps, f)
        t = torch.full((xt.shape[0],), int(tv), device=dev, dtype=torch.long)
        e = _eps_tile(model, xt, t, c1, c2, mp)
        if cond.guid != 1.0:
            e0 = _eps_tile(model, xt, t, c1, cut(cond.c2n), cut(cond.mapsn, f))
            e = e0 + cond.guid * (e - e0)
        for j, (y, xx) in enumerate(bb):
            acc[:, :, y:y + tile, xx:xx + tile] += e[j * B:(j + 1) * B] * wgt
            norm[:, :, y:y + tile, xx:xx + tile] += wgt
    return acc / norm.clamp(min=1e-6)


@torch.no_grad()
def ddim(model, cond, steps=25, eta=0.0, seed=123, tile=64, stride=48, tile_bs=4):
    dev = cond.c1.device
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(cond.c1.shape, generator=g).to(dev)
    ts = sorted(set(np.linspace(model.num_timesteps - 1, 0, steps).round().astype(int).tolist()), reverse=True)
    for i, tv in enumerate(ts):
        eps = eps_full(model, x, tv, cond, tile, stride, tile_bs)
        t = torch.full((x.shape[0],), int(tv), device=dev, dtype=torch.long)
        x0 = model.predict_start_from_noise(x, t, eps)
        if i == len(ts) - 1:
            return x0
        a_t, a = model.alphas_cumprod[int(tv)], model.alphas_cumprod[int(ts[i + 1])]
        sigma = eta * ((1 - a) / (1 - a_t)).sqrt() * (1 - a_t / a).sqrt()
        noise = torch.randn(x.shape, generator=g).to(dev) if eta > 0 else 0.0
        x = a.sqrt() * x0 + (1 - a - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * noise


@torch.no_grad()
def enhance(model, rgb_m11, ir_m11, steps=25, eta=0.0, seed=123, guid=1.0, tile_bs=4):
    """rgb_m11 / ir_m11: HxWx3 float [-1,1] (H, W multiples of 32, >= 256). Returns HxWx3 float [0,1] (numpy)."""
    dev = next(model.parameters()).device
    r, i = to_bchw(rgb_m11, dev), to_bchw(ir_m11, dev)
    cond = Conditioning(model, r, i, guid)
    x0 = ddim(model, cond, steps=steps, eta=eta, seed=seed, tile_bs=tile_bs)
    img = model.decode_first_stage(x0)
    return ((img[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def ddim_x0_train(model, c1, c2, maps, t_stop, n_steps=4, seed=None):
    """Training helper for real-night frames (no GT, so no clean latent to noise): a short deterministic DDIM walk from
    pure noise down to t_stop WITHOUT grad, then one step WITH grad that predicts x0. Returns (x0_with_grad, t)."""
    dev = c1.device
    x = torch.randn(c1.shape, device=dev)
    ts = sorted(set(np.linspace(model.num_timesteps - 1, t_stop, max(1, n_steps)).round().astype(int).tolist()), reverse=True)
    with tpa_maps(model, maps):
        for i, tv in enumerate(ts):
            t = torch.full((x.shape[0],), int(tv), device=dev, dtype=torch.long)
            last = i == len(ts) - 1
            with torch.set_grad_enabled(last):
                eps = model.apply_model(x, t, c1, c2)
                x0 = model.predict_start_from_noise(x, t, eps)
            if last:
                return x0, t
            a = model.alphas_cumprod[int(ts[i + 1])]
            x = (a.sqrt() * x0 + (1 - a).sqrt() * eps).detach()
