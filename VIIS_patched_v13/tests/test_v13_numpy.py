"""CPU / numpy-only checks (no torch needed):  python tests/test_v13_numpy.py
  1. degradation: severities produce the intended brightness ordering, deterministic with a seed, light ~ near-identity
  2. alignment: ECC recovers a known synthetic misregistration
  3. metrics: TSC cannot be raised by adding random texture; it IS raised by real thermal structure; bounded
  4. data: fake MSRS / LLVIP folders -> splits are disjoint, items have the right shape/range
"""
import os, sys, shutil, tempfile
import numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from trg13 import degrade as D, align as A, metrics as M, data as DT

rgb = np.array(Image.open(f"{ROOT}/instance/rgb/01012N.png").convert("RGB"))
ir = np.array(Image.open(f"{ROOT}/instance/ir/01012N.png").convert("L"))
# the instance images are NIGHT frames; brighten to make a pseudo-"day" image for the degradation test
day = (np.clip((rgb / 255.0) ** 0.4 * 1.1, 0, 1) * 255).astype(np.uint8)
day, ir = cv2.resize(day, (256, 256), interpolation=cv2.INTER_AREA), cv2.resize(ir, (256, 256), interpolation=cv2.INTER_AREA)

# ---- 1. degradation
means = {}
for sev in ("light", "low", "dark"):
    ms = [DT.mean_lum(D.degrade(day, ir, rs=np.random.RandomState(s), severity=sev)) for s in range(12)]
    means[sev] = float(np.median(ms))
print("median luminance by severity:", {k: round(v, 3) for k, v in means.items()}, "| clean:", round(DT.mean_lum(day), 3))
assert means["light"] > means["low"] > means["dark"], means
a = D.degrade(day, ir, rs=np.random.RandomState(3), severity="dark"); b = D.degrade(day, ir, rs=np.random.RandomState(3), severity="dark")
assert np.array_equal(a, b), "degradation must be deterministic for a fixed seed"
lt = D.degrade(day, ir, rs=np.random.RandomState(0), severity="light", p_jpeg=0, p_blur=0)
print("light-severity PSNR vs clean (should be moderate, it is a mild exposure change):", round(M.psnr(lt / 255., day / 255.), 2))
att = [D.degrade(day, ir, rs=np.random.RandomState(s), severity="low", p_att=1.0, return_info=True)[1]["att"] for s in range(5)]
print("hot-object attenuation applied:", att)
print("degradation OK")

# ---- 2. alignment
r3 = np.repeat(ir[..., None], 3, 2)
self_rows = A.audit([(r3, ir)], shifts=(4, 8))
assert all(r["after"] < 0.5 for r in self_rows), ("self-pair must be recovered exactly", self_rows)
rgb_full, ir_full = np.array(Image.open(f"{ROOT}/instance/rgb/01012N.png").convert("RGB")), np.array(Image.open(f"{ROOT}/instance/ir/01012N.png").convert("L"))
rows = A.audit([(rgb_full, ir_full)], shifts=(0, 2, 4, 8))
for r in rows:
    print(f"  shift {r['shift']}: corner err before {r['before']:.2f}px  after {r['after']:.2f}px  ok={r['ok']}  "
          f"edge overlap {r['overlap_before']:.3f}->{r['overlap_after']:.3f}  (pair residual {r['residual_W0']:.2f}px)")
assert all(r["ok"] for r in rows), "ECC failed"
assert all(r["after"] < 0.5 * r["before"] for r in rows if r["shift"] >= 4), "ECC did not reduce misregistration"
print("alignment OK")

# ---- 3. metrics cannot be gamed by texture
out = day / 255.0; th = ir / 255.0
t0 = M.tsc(out, th)
noisy = np.clip(out + np.random.RandomState(0).randn(*out.shape) * 0.15, 0, 1)
t_noise = M.tsc(noisy, th)
copy = np.clip(0.5 * out + 0.5 * th[..., None], 0, 1)   # pastes thermal into the image (the 'contradictory' extreme)
t_copy = M.tsc(copy, th)
print(f"TSC clean {t0:.3f} | +random texture {t_noise:.3f} | thermal pasted {t_copy:.3f}")
assert t_noise <= t0 + 0.02, "TSC must not increase with random texture"
assert t_copy > t0, "TSC must respond to real thermal structure"
assert 0 <= t_copy <= 1
fid_c, fid_n = M.input_fidelity(out, out), M.input_fidelity(noisy, out)
print(f"input fidelity: identical {fid_c:.3f} | noisy {fid_n:.3f}")
assert fid_c > fid_n
hf_c, hf_n = M.hf_energy(out, out, th), M.hf_energy(noisy, out, th)
print(f"HF energy in unsupported regions: clean {hf_c:.4f} | texture-added {hf_n:.4f}")
assert hf_n > hf_c
print("TES:", round(M.thermal_edge_score(out, th), 3), "| OBJ:", round(M.object_contrast(out, th), 2),
      "| BOX:", round(M.box_contrast(out, [[100, 60, 140, 200]]), 2))
fr = M.full_ref_all(out, out, out, th)
assert fr["psnr"] > 90 and abs(fr["ssim"] - 1) < 1e-6 and (np.isnan(fr["tsc_gap"]) or fr["tsc_gap"] < 1e-6)
print("metrics OK")

# ---- 4. data on fake folders
tmp = tempfile.mkdtemp()
try:
    ms, lv = f"{tmp}/MSRS", f"{tmp}/LLVIP"
    for sp in ("train", "test"):
        for k in ("vi", "ir"):
            os.makedirs(f"{ms}/{sp}/{k}")
        for i in range(25):
            for kind in ("D", "N"):
                Image.fromarray(day if kind == "D" else rgb[:256, :256]).save(f"{ms}/{sp}/vi/{i:05d}{kind}.png")
                Image.fromarray(ir).save(f"{ms}/{sp}/ir/{i:05d}{kind}.png")
    for sp in ("train", "test"):
        os.makedirs(f"{lv}/visible/{sp}"); os.makedirs(f"{lv}/infrared/{sp}")
        for seq in ("01", "02", "26"):
            for i in range(5):
                Image.fromarray(rgb).save(f"{lv}/visible/{sp}/{seq}{i:04d}.jpg"); Image.fromarray(ir).save(f"{lv}/infrared/{sp}/{seq}{i:04d}.jpg")
    tr, va = DT.msrs_train_val(ms)
    assert not set(tr) & set(va) and len(va) == 3
    d = DT.MSRSPairs(ms, "train", names=tr)[0]
    assert d["gt"].shape == (256, 256, 3) and d["rgb"].dtype == np.float32 and -1 <= d["rgb"].min() and d["rgb"].max() <= 1
    v1, v2 = DT.MSRSPairs(ms, "train", names=va, train=False)[1], DT.MSRSPairs(ms, "train", names=va, train=False)[1]
    assert np.array_equal(v1["rgb"], v2["rgb"]), "validation must be deterministic"
    ntr, nva = DT.night_items(ms, lv, "train"), DT.night_items(ms, lv, "val")
    assert not {p for p, _ in ntr} & {p for p, _ in nva}, "night train/val overlap"
    assert all("/26" not in p for p, _ in ntr if "LLVIP" in p), "held-out LLVIP sequence leaked into training"
    nd = DT.NightPool(ntr)[0]; assert nd["rgb"].shape == (256, 256, 3)
    ev = DT.eval_msrs_synthetic(ms, n=4); assert len(ev) == 12
    bins = DT.sample_bins(ms, lv, k=2); print("sample bins:", {k: len(v) for k, v in bins.items()})
    print("data OK  (msrs train/val", len(tr), len(va), "| night train/val", len(ntr), len(nva), ")")
finally:
    shutil.rmtree(tmp)
print("\nALL NUMPY TESTS PASSED")
