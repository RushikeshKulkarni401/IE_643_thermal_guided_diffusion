"""VIIS model build + v13 Thermal Prior Adapter (TPA).

Deliverable 2 ("encode thermal information into the diffusion latent space") is met at three levels:
  (a) VIIS already concatenates the VQ latent of the thermal image to x_t (UNet input = [x_t, z_rgb, z_ir], 9 ch) and
      cross-attends to multi-scale thermal features (encoder_i -> DualModalSpraseAttn 'attn2');
  (b) v13 adds the TPA: a small conv pyramid over explicit modality maps (thermal luma, thermal edges, thermal saliency s,
      visible-quality q, lifted RGB luma) whose outputs are added to every UNet input block through ZERO-INITIALISED 1x1
      convs (ControlNet-style). At step 0 the network is exactly pretrained VIIS; training can only add thermal evidence
      where it lowers the GT-supervised loss.  q tells the network where the RGB can be trusted (the user's per-pixel
      decision idea, used as an input feature instead of a hand-set rule);
  (c) classifier-free thermal dropout (thermal -> black, TPA thermal maps -> 0, prob th_drop) so thermal guidance
      eps = eps(no-T) + g * (eps(T) - eps(no-T)) is a trained, well-defined operation at inference.
"""
import contextlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma


class ThermalPriorAdapter(nn.Module):
    def __init__(self, in_ch=5, block_specs=(), width=(96, 128, 192, 256)):
        """block_specs: list of (block_index, channels, downsample_factor_vs_latent) for each UNet input block."""
        super().__init__()
        c0 = width[0]
        self.stem = nn.Sequential(                       # image res -> latent res (VQ-f4: /4)
            nn.Conv2d(in_ch, c0 // 2, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(c0 // 2, c0, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(c0, c0, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(c0, c0, 3, 1, 1), nn.SiLU())
        self.levels = sorted({f for _, _, f in block_specs}) or [1]
        self.down = nn.ModuleDict()
        prev, ch_of = c0, {1: c0}
        for i, f in enumerate(self.levels):
            if f == 1:
                continue
            c = width[min(i, len(width) - 1)]
            self.down[str(f)] = nn.Sequential(nn.Conv2d(prev, c, 3, 2, 1), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1), nn.SiLU())
            prev = c; ch_of[f] = c
        self.zero = nn.ModuleDict()
        self.spec = {}
        for bi, ch, f in block_specs:
            conv = nn.Conv2d(ch_of[f], ch, 1)
            nn.init.zeros_(conv.weight); nn.init.zeros_(conv.bias)
            self.zero[str(bi)] = conv
            self.spec[int(bi)] = f

    def forward(self, maps):
        h = self.stem(maps)
        feats = {1: h}
        for f in self.levels:
            if f == 1:
                continue
            h = self.down[str(f)](h); feats[f] = h
        return {bi: self.zero[str(bi)](feats[f]) for bi, f in self.spec.items()}


class _TPAState:
    def __init__(self):
        self.maps = None       # Bx5xHxW (image resolution) for the NEXT apply_model call(s)
        self.res = None


def _install_hooks(model):
    dm = model.model.diffusion_model
    st = model.model._tpa_state

    def make_hook(bi):
        def hook(mod, inp, out):
            if st.res is None or bi not in st.res:
                return out
            r = st.res[bi]
            if r.shape[-2:] != out.shape[-2:]:
                r = F.interpolate(r, size=out.shape[-2:], mode="bilinear", align_corners=False)
            return out + r
        return hook

    for bi, blk in enumerate(dm.input_blocks):
        if bi in model.model.tpa.spec:
            blk.register_forward_hook(make_hook(bi))

    orig_forward = model.model.forward

    def forward(x, t, c_concat=None, c_1=None, c_2=None):
        if st.maps is not None:
            m = st.maps
            if m.shape[0] != x.shape[0]:
                m = m.repeat(x.shape[0] // m.shape[0], 1, 1, 1)
            st.res = model.model.tpa(m.to(x.device, x.dtype))
        try:
            return orig_forward(x, t, c_concat=c_concat, c_1=c_1, c_2=c_2)
        finally:
            st.res = None
    model.model.forward = forward


@contextlib.contextmanager
def tpa_maps(model, maps):
    """Set the adapter input for every apply_model call inside the block."""
    st = getattr(model.model, "_tpa_state", None)
    if st is None:
        yield; return
    old = st.maps; st.maps = maps
    try:
        yield
    finally:
        st.maps = old


@torch.no_grad()
def _probe_blocks(model, device):
    """Run one dummy forward to record (channels, resolution) after each UNet input block."""
    dm = model.model.diffusion_model
    rec, hs = {}, []
    for bi, blk in enumerate(dm.input_blocks):
        hs.append(blk.register_forward_hook(lambda m, i, o, bi=bi: rec.__setitem__(bi, tuple(o.shape[1:]))))
    try:
        x = torch.randn(1, 3, 64, 64, device=device)
        model.apply_model(x, torch.tensor([500], device=device), torch.randn_like(x), torch.randn_like(x))
    finally:
        for h in hs:
            h.remove()
    lat = 64
    return [(bi, c, lat // h) for bi, (c, h, w) in sorted(rec.items())]


def set_trainable(model, mode="attn_out", last_k_out=3):
    """attn2      : thermal cross-attention only
       attn       : both cross-attentions (RGB attn1 + thermal attn2)
       attn_out   : attn + UNet 'out' conv + last k output blocks (global tone/detail can move)   [default]
       + always   : the TPA adapter (new)"""
    outs = [int(n.split("output_blocks.")[1].split(".")[0]) for n, _ in model.named_parameters() if "output_blocks." in n]
    n_out = max(outs) + 1
    for n, p in model.named_parameters():
        ok = n.startswith("model.tpa.")
        if n.startswith("model.") and ".attn2." in n:
            ok = True
        if mode in ("attn", "attn_out") and n.startswith("model.") and ".attn1." in n:
            ok = True
        if mode == "attn_out" and (n.startswith("model.diffusion_model.out.") or (".output_blocks." in n and
                                   int(n.split("output_blocks.")[1].split(".")[0]) >= n_out - last_k_out)):
            ok = True
        p.requires_grad = ok
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tpa = sum(p.numel() for n, p in model.named_parameters() if n.startswith("model.tpa."))
    print(f"[v13] trainable mode={mode}: {tr / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"(adapter {tpa / 1e6:.2f}M)", flush=True)


def build(repo, ckpt_path, device="cuda", mode="attn_out", ema_decay=0.999, use_tpa=True, yaml=None):
    cfg = OmegaConf.load(yaml or f"{repo}/informationSynthesis/Image2Image.yaml")
    model = instantiate_from_config(cfg.model)
    if ckpt_path:
        import time
        t0 = time.time()
        local = not ckpt_path.startswith("/kaggle/input")
        print(f"[v13] loading VIIS weights from {ckpt_path} ({os.path.getsize(ckpt_path) / 1e9:.1f} GB, "
              f"{'mmap' if local else 'sequential read'}) ...", flush=True)
        # mmap on Kaggle's network-mounted /kaggle/input faults pages in one by one (20+ min for the 9 GB file):
        # read sequentially there; local files (the slim checkpoint) are memory-mapped.
        sd = torch.load(ckpt_path, map_location="cpu", mmap=local, weights_only=False)
        sd = sd.get("state_dict", sd)
        has_ema = any(k.startswith("model_ema.") for k in sd)
        r = model.load_state_dict(sd, strict=False)
        miss = [k for k in r.missing_keys if not k.startswith("model_ema.")]
        print(f"[v13] VIIS weights loaded in {time.time() - t0:.0f}s: missing {len(miss)} unexpected {len(r.unexpected_keys)} "
              f"({'EMA copy present' if has_ema else 'slim: EMA already folded in'})", flush=True)
        assert len(miss) == 0, f"checkpoint is missing model weights: {miss[:5]}"
        del sd
        if has_ema:
            model.model_ema.copy_to(model.model)    # start from the EMA weights VIIS itself samples with
    model = model.to(device)
    model.model._tpa_state = _TPAState()
    if use_tpa:
        specs = _probe_blocks(model, device)
        model.model.tpa = ThermalPriorAdapter(5, specs).to(device)
        _install_hooks(model)
    set_trainable(model, mode)
    model.model_ema = LitEma(model.model, decay=ema_decay).to(device)   # EMA over trainable params only
    model.v13_meta = dict(mode=mode, use_tpa=use_tpa, ema_decay=ema_decay)
    return model


def save_ckpt(model, path, step, opt=None, extra=None):
    obj = dict(step=step, meta=getattr(model, "v13_meta", {}), extra=extra or {},
               ema=model.model_ema.state_dict(),
               trainable={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad})
    if opt is not None:
        obj["opt"] = opt.state_dict()
    torch.save(obj, path)


def load_ckpt(model, path, opt=None):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    r = model.load_state_dict(obj["trainable"], strict=False)
    if r.unexpected_keys:
        print("[v13] unexpected keys in checkpoint:", r.unexpected_keys[:5])
    model.model_ema.load_state_dict(obj["ema"])
    if opt is not None and "opt" in obj:
        opt.load_state_dict(obj["opt"])
    return obj


def ckpt_meta(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("meta", {}), obj.get("step"), obj.get("extra", {})


@contextlib.contextmanager
def ema_scope(model, enabled=True):
    if not enabled:
        yield; return
    params = [p for p in model.model.parameters() if p.requires_grad]    # EMA only tracks these; avoids a 2 GB clone
    model.model_ema.store(params)
    model.model_ema.copy_to(model.model)
    try:
        yield
    finally:
        model.model_ema.restore(params)
