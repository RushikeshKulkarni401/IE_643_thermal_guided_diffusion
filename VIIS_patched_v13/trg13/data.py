"""Datasets for v13. All arrays returned to the model are HxWx3 float32 in [-1, 1] (VIIS convention); thermal is the
single IR channel replicated to 3 channels (as VIIS was trained).

Leak rules enforced here:
  * MSRS train day images are split into train / val by a fixed rule (every 10th name -> val); val never trains.
  * Real-night pool (no ground truth): LLVIP train frames minus one whole held-out sequence (by 2-digit prefix) that is
    used for night validation, plus MSRS train night frames minus every 10th (validation).
  * Final evaluation uses only the official MSRS test and LLVIP test splits.
  * Nothing in this file (or anywhere in v13) produces a training target from a model output.
"""
import os, glob, xml.etree.ElementTree as ET
import numpy as np, cv2
from PIL import Image
try:
    from torch.utils.data import Dataset
except ImportError:            # lets the numpy-only tests run without torch
    Dataset = object

from .degrade import degrade, misregister, jitter_thermal

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def load_ir(p):
    return np.array(Image.open(p).convert("L"))


def to_m11(u8, size=None):
    """uint8 HxW(x3) -> float32 HxWx3 in [-1,1]; optional square resize."""
    if size is not None and (u8.shape[0] != size or u8.shape[1] != size):
        u8 = cv2.resize(u8, (size, size), interpolation=cv2.INTER_AREA)
    if u8.ndim == 2:
        u8 = np.repeat(u8[..., None], 3, axis=2)
    return (u8.astype(np.float32) / 127.5 - 1.0)


def mean_lum(rgb_u8):
    x = rgb_u8.astype(np.float32) / 255.0
    return float((0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).mean())


def lum_bin(rgb_u8, t_dark=0.08, t_light=0.25):
    m = mean_lum(rgb_u8)
    return "dark" if m < t_dark else ("light" if m >= t_light else "low")


# ------------------------------------------------------------------------------------------------------------- MSRS
def msrs_names(root, split, kind="D"):
    d = f"{root}/{split}/vi"
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT) and os.path.splitext(f)[0].endswith(kind))


def msrs_train_val(root):
    """Legacy (v13.0) split -- NOT a fair validation set: VIIS itself was trained on MSRS/train day images, so step-0 VIIS
    scores are inflated on it by memorisation. Kept only for reference."""
    names = msrs_names(root, "train", "D")
    val = names[::10]
    vs = set(val)
    return [n for n in names if n not in vs], val


def msrs_test_split(root):
    """v13.1: validation = every 5th MSRS TEST day image (unseen by VIIS and by our training); final evaluation uses the
    remaining test images. Returns (val_names, eval_names) -- disjoint."""
    names = msrs_names(root, "test", "D")
    val = names[::5]
    vs = set(val)
    return val, [n for n in names if n not in vs]


class MSRSPairs(Dataset):
    """Clean MSRS day image -> synthetic degraded input. The clean day image is the target (the only real GT we have).
    train=True : random crop (crop_lo..1 of short side) + flip + random degradation + thermal augmentation.
    train=False: centre crop, degradation seeded by index (severity can be forced) -> identical every evaluation."""

    def __init__(self, root, split="train", names=None, size=256, train=True, severity=None, crop_lo=0.6,
                 p_misreg=0.3, seed_base=1000, att_p=None):
        self.root, self.split, self.size, self.train = root, split, size, train
        self.names = list(names) if names is not None else msrs_names(root, split, "D")
        self.severity, self.crop_lo, self.p_misreg, self.seed_base, self.att_p = severity, crop_lo, p_misreg, seed_base, att_p

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        gt = load_rgb(f"{self.root}/{self.split}/vi/{n}")
        ir = load_ir(f"{self.root}/{self.split}/ir/{n}")
        if ir.shape != gt.shape[:2]:
            ir = cv2.resize(ir, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        rs = np.random.RandomState() if self.train else np.random.RandomState(self.seed_base + i)
        h, w = gt.shape[:2]
        c = int(min(h, w) * (rs.uniform(self.crop_lo, 1.0) if self.train else 1.0))
        y0, x0 = (rs.randint(0, h - c + 1), rs.randint(0, w - c + 1)) if self.train else ((h - c) // 2, (w - c) // 2)
        gt, ir = gt[y0:y0 + c, x0:x0 + c], ir[y0:y0 + c, x0:x0 + c]
        if self.train and rs.rand() < 0.5:
            gt, ir = gt[:, ::-1], ir[:, ::-1]
        gt = cv2.resize(np.ascontiguousarray(gt), (self.size, self.size), interpolation=cv2.INTER_AREA)
        ir = cv2.resize(np.ascontiguousarray(ir), (self.size, self.size), interpolation=cv2.INTER_AREA)
        kw = {} if self.att_p is None else dict(p_att=self.att_p)
        dark, info = degrade(gt, ir, rs=rs, severity=self.severity, return_info=True, **kw)
        if self.train:
            if rs.rand() < self.p_misreg:
                ir, _ = misregister(ir, rs)
            ir = jitter_thermal(ir, rs)
        sev = {"light": 0, "low": 1, "dark": 2}[info["severity"]]
        return dict(gt=to_m11(gt), rgb=to_m11(dark), ir=to_m11(ir), name=n, sev=sev, att=int(info["att"]))


# ------------------------------------------------------------------------------------------------------------- LLVIP
def llvip_names(root, split):
    return sorted(f for f in os.listdir(f"{root}/visible/{split}") if f.lower().endswith(IMG_EXT))


def llvip_train_val(root):
    """Hold out one whole sequence (2-digit filename prefix) for night validation: no near-duplicate frames leak."""
    names = llvip_names(root, "train")
    prefixes = sorted({n[:2] for n in names})
    vp = prefixes[-1]
    return [n for n in names if n[:2] != vp], [n for n in names if n[:2] == vp], vp


def llvip_boxes(root, name):
    stem = os.path.splitext(name)[0]
    for p in (f"{root}/Annotations/{stem}.xml", f"{root}/annotations/{stem}.xml"):
        if os.path.exists(p):
            out = []
            try:
                for obj in ET.parse(p).getroot().iter("object"):
                    bb = obj.find("bndbox")
                    out.append([float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
            except Exception:
                pass
            return out
    return []


class NightPool(Dataset):
    """Real night pairs WITHOUT ground truth (LLVIP train + MSRS train night). Used only by the bounded self-supervised
    consistency terms (input-structure fidelity, thermal structure, object presence). gt is NOT provided."""

    def __init__(self, items, size=256, train=True, crop=(0.4, 0.75)):
        self.items, self.size, self.train, self.crop = items, size, train, crop   # items: list of (rgb_path, ir_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pr, pi = self.items[i]
        rgb, ir = load_rgb(pr), load_ir(pi)
        if ir.shape != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        rs = np.random.RandomState() if self.train else np.random.RandomState(5000 + i)
        f = rs.uniform(*self.crop) if self.train else float(np.mean(self.crop))
        c = int(min(h, w) * f)
        y0, x0 = (rs.randint(0, h - c + 1), rs.randint(0, w - c + 1)) if self.train else ((h - c) // 2, (w - c) // 2)
        rgb, ir = rgb[y0:y0 + c, x0:x0 + c], ir[y0:y0 + c, x0:x0 + c]
        if self.train and rs.rand() < 0.5:
            rgb, ir = rgb[:, ::-1], ir[:, ::-1]
        return dict(rgb=to_m11(np.ascontiguousarray(rgb), self.size), ir=to_m11(np.ascontiguousarray(ir), self.size),
                    name=os.path.basename(pr))


def night_items(msrs_root, llvip_root, which="train", llvip_limit=None):
    """-> list of (rgb_path, ir_path). which: 'train' or 'val' (held-out sequence / every-10th MSRS night)."""
    items = []
    if llvip_root and os.path.isdir(f"{llvip_root}/visible/train"):
        tr, va, _ = llvip_train_val(llvip_root)
        names = tr if which == "train" else va
        if which == "train" and llvip_limit and llvip_limit < len(names):
            names = names[::max(1, len(names) // llvip_limit)][:llvip_limit]
        if which == "val":
            names = names[::max(1, len(names) // 8)][:8]
        items += [(f"{llvip_root}/visible/train/{n}", f"{llvip_root}/infrared/train/{n}") for n in names]
    if msrs_root and os.path.isdir(f"{msrs_root}/train/vi"):
        nn_ = msrs_names(msrs_root, "train", "N")
        val = nn_[::10]
        names = [n for n in nn_ if n not in set(val)] if which == "train" else val[:4]
        items += [(f"{msrs_root}/train/vi/{n}", f"{msrs_root}/train/ir/{n}") for n in names]
    return items


# ------------------------------------------------------------------------------------------------------------- eval
def eval_msrs_synthetic(root, n=30, size=256, severities=("light", "low", "dark")):
    """Official MSRS TEST day images (minus the validation fifth), deterministic degradation at each severity."""
    names = msrs_test_split(root)[1]
    names = names[::max(1, len(names) // max(1, n))][:n]
    out = []
    for sev in severities:
        ds = MSRSPairs(root, "test", names=names, size=size, train=False, severity=sev,
                       seed_base={"light": 70000, "low": 80000, "dark": 90000}[sev], att_p=0.5)
        for i in range(len(ds)):
            d = ds[i]; d["sev_name"] = sev; out.append(d)
    return out


def eval_real(msrs_root, llvip_root, n_llvip=24, n_msrs_night=12):
    """Real night frames at native resolution (uint8), with LLVIP person boxes. No ground truth."""
    out = []
    if llvip_root and os.path.isdir(f"{llvip_root}/visible/test"):
        names = llvip_names(llvip_root, "test")
        for nme in names[::max(1, len(names) // max(1, n_llvip))][:n_llvip]:
            out.append(dict(src="llvip", name=nme, rgb=load_rgb(f"{llvip_root}/visible/test/{nme}"),
                            ir=load_ir(f"{llvip_root}/infrared/test/{nme}"), boxes=llvip_boxes(llvip_root, nme)))
    if msrs_root and os.path.isdir(f"{msrs_root}/test/vi"):
        names = msrs_names(msrs_root, "test", "N")
        for nme in names[::max(1, len(names) // max(1, n_msrs_night))][:n_msrs_night]:
            out.append(dict(src="msrs_night", name=nme, rgb=load_rgb(f"{msrs_root}/test/vi/{nme}"),
                            ir=load_ir(f"{msrs_root}/test/ir/{nme}"), boxes=[]))
    return out


def sample_bins(msrs_root, llvip_root, k=4, scan=120):
    """Real (not synthetic) test inputs binned by input brightness into light / low / dark; k per bin."""
    pool = []
    if msrs_root and os.path.isdir(f"{msrs_root}/test/vi"):
        for nme in sorted(os.listdir(f"{msrs_root}/test/vi")):
            if nme.lower().endswith(IMG_EXT):
                pool.append(("msrs", f"{msrs_root}/test/vi/{nme}", f"{msrs_root}/test/ir/{nme}", None))
    if llvip_root and os.path.isdir(f"{llvip_root}/visible/test"):
        for nme in llvip_names(llvip_root, "test"):
            pool.append(("llvip", f"{llvip_root}/visible/test/{nme}", f"{llvip_root}/infrared/test/{nme}", nme))
    if not pool:
        return {}
    pool = pool[::max(1, len(pool) // scan)]
    bins = {"light": [], "low": [], "dark": []}
    for src, pr, pi, nme in pool:
        im = Image.open(pr); im.draft("RGB", (160, 128)); im = im.convert("RGB"); im.thumbnail((96, 96))   # draft: fast JPEG decode
        bins[lum_bin(np.array(im))].append((src, pr, pi, nme))
    out = {}
    for b, lst in bins.items():
        sel = lst[::max(1, len(lst) // k)][:k]
        out[b] = [dict(src=s, name=os.path.basename(pr), rgb=load_rgb(pr), ir=load_ir(pi),
                       boxes=llvip_boxes(llvip_root, nm) if nm else []) for s, pr, pi, nm in sel]
    return out


def user_samples(folder):
    """folder/rgb/*.png|jpg + folder/ir/<same name>  (or folder/visible + folder/infrared)."""
    for a, b in (("rgb", "ir"), ("visible", "infrared"), ("vi", "ir")):
        if os.path.isdir(f"{folder}/{a}") and os.path.isdir(f"{folder}/{b}"):
            out = []
            for nme in sorted(os.listdir(f"{folder}/{a}")):
                if not nme.lower().endswith(IMG_EXT):
                    continue
                stem = os.path.splitext(nme)[0]
                cand = [p for p in glob.glob(f"{folder}/{b}/{stem}.*") if p.lower().endswith(IMG_EXT)]
                if cand:
                    rgb = load_rgb(f"{folder}/{a}/{nme}")
                    out.append(dict(src="user", name=nme, rgb=rgb, ir=load_ir(cand[0]), boxes=[], bin=lum_bin(rgb)))
            return out
    return []
