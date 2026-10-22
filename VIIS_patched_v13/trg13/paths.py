"""Dataset / checkpoint discovery. Override anything with env vars MSRS, LLVIP, VIIS_CKPT, RUNS."""
import os

_SKIP = {"vi", "ir", "visible", "infrared", "annotations", "label", "labels", "__pycache__", ".git"}
DEFAULTS = dict(
    MSRS="/kaggle/input/datasets/elixir4o4/msrs-main/MSRS-main",
    LLVIP="/kaggle/input/datasets/afradhossain/llvip-dataset/LLVIP",
    RUNS="/kaggle/working/runs",
)
SLIM_DEFAULT = "/kaggle/working/viis_slim.ckpt" if os.path.isdir("/kaggle/working") else os.path.abspath("viis_slim.ckpt")


def _walk(base, max_depth=7):
    if not os.path.isdir(base):
        return
    base = os.path.abspath(base)
    for dp, dns, fns in os.walk(base):
        depth = dp[len(base):].count(os.sep)
        dns[:] = [d for d in sorted(dns) if d.lower() not in _SKIP] if depth < max_depth else []
        yield dp, dns, fns


def find_msrs(bases=("/kaggle/input", "/kaggle/working")):
    for b in bases:
        for dp, _, _ in _walk(b):
            if os.path.isdir(f"{dp}/train/vi") and os.path.isdir(f"{dp}/train/ir"):
                return dp
    return None


def find_llvip(bases=("/kaggle/input", "/kaggle/working")):
    for b in bases:
        for dp, _, _ in _walk(b):
            if os.path.isdir(f"{dp}/visible/train") and os.path.isdir(f"{dp}/infrared/train"):
                return dp
    return None


def find_ckpt(bases=("/kaggle/input", "/kaggle/working")):
    for b in bases:
        for dp, _, fns in _walk(b, max_depth=12):
            if "last.ckpt" in fns:
                return f"{dp}/last.ckpt"
    return None


def resolve():
    def pick(key, finder):
        v = os.environ.get(key)
        if v:
            return v
        d = DEFAULTS.get(key)
        if d and os.path.exists(d):
            return d
        return finder() if finder else d
    msrs = pick("MSRS", find_msrs)
    llvip = pick("LLVIP", find_llvip)
    ckpt = os.environ.get("VIIS_CKPT") or (SLIM_DEFAULT if os.path.exists(SLIM_DEFAULT) else None) or pick("VIIS_CKPT", find_ckpt)
    runs = os.environ.get("RUNS", DEFAULTS["RUNS"] if os.path.isdir("/kaggle/working") else os.path.abspath("runs"))
    return dict(MSRS=msrs, LLVIP=llvip, CKPT=ckpt, RUNS=runs)


if __name__ == "__main__":
    for k, v in resolve().items():
        ok = v and os.path.exists(v)
        note = "OK" if ok else ("(created on first run)" if k == "RUNS" else "NOT FOUND")
        if k == "CKPT" and ok and not v.endswith("viis_slim.ckpt"):
            note += "   <- original 9 GB file: run  python slim_ckpt_v13.py  once (loads become seconds instead of minutes)"
        print(f"{k:6s} {v}  {note}")
