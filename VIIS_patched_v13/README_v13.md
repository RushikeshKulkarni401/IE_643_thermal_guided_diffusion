# VIIS_patched_v13 (v13.1): Thermal-RGB guided diffusion (TRG-Diffusion v13)

**v13.1:** trajectory loss against real GT (fixes the exposure drift), validation moved to unseen MSRS-test images, checkpoints selected LLVIP-first. Commit-mode cell: `kaggle_commit_v13_1.py`. Rationale: section 6 of `V13_DESIGN.md`.

Fine-tunes pretrained **VIIS** so that, given a degraded RGB frame and its thermal frame, it reconstructs an enhanced visible image that preserves thermal-supported objects and boundaries.
- The **only** reconstruction target is real MSRS day images.
- **No VIIS zero-shot output, sharpened or not, is used anywhere as a target.**
- Why previous versions failed and how v13 fixes it: `V13_DESIGN.md`.

```
train_v13.py          training (MSRS GT + bounded real-night consistency), validation, checkpoint selection
infer_v13.py          inference + evaluation: MSRS (light/low/dark, full reference), MSRS-night, LLVIP, light/low/dark samples, user pairs
align_audit_v13.py    RGB-thermal registration audit (deliverable 1)
run_all_v13.sh        everything in one go (2 GPUs: full model + ablation, then evaluation and audit)
kaggle_v13.ipynb      the same as notebook cells
trg13/                degrade.py, align.py, data.py, maps.py, losses.py, model.py (adapter), sampler.py (tiled DDIM), metrics.py, paths.py
tests/                test_v13_numpy.py (CPU, runs anywhere), smoke_kaggle.sh (GPU, 5-10 min, every torch path)
ldm/ taming/ informationSynthesis/   VIIS code with the earlier Kaggle patches (pure-PyTorch deformable attention, taming stub,
                      checkpoint-backward fix, yaml ckpt_path null). The old v2-v12 experiment scripts were removed on purpose.
```

## Kaggle: step by step

### 0. One-time preparation
1. Upload **VIIS_patched_v13.zip** as a new Kaggle Dataset (*Datasets -> New Dataset -> upload the zip*, e.g. name it `viis-patched-v13`). Kaggle extracts it automatically.
2. Create a notebook and set **Accelerator = GPU T4 x2**, **Internet = On** (needed for `pip install lpips piq ultralytics` and the LPIPS/VGG weights).
3. **Add Input** for these four datasets (the same ones you used before):

   | dataset | content |
   |---|---|
   | `elixir4o4/viis-master` | VIIS `last.ckpt` (9 GB) |
   | `elixir4o4/msrs-main` | MSRS (`train/vi`, `train/ir`, `test/vi`, `test/ir`) |
   | `afradhossain/llvip-dataset` | LLVIP (`visible/{train,test}`, `infrared/{train,test}`, `Annotations`) |
   | `viis-patched-v13` | this code |

   Paths are found automatically. To override, set env vars `MSRS`, `LLVIP`, `VIIS_CKPT`, `RUNS` before running.
4. Upload `kaggle_v13.ipynb` (*File -> Import Notebook*), or paste the commands below into cells.

### 1. Setup (run again after every session restart)
```python
import glob, os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "omegaconf", "kornia", "lpips", "piq", "ultralytics"])
try: import pytorch_lightning
except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning"])
src = [os.path.dirname(p) for k in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * k + "train_v13.py")][0]
shutil.rmtree("/kaggle/working/VIIS13", ignore_errors=True); shutil.copytree(src, "/kaggle/working/VIIS13")
os.chdir("/kaggle/working/VIIS13")
```

### 2. Checks (1 min)
```bash
python tests/test_v13_numpy.py      # degradation, ECC alignment, metric non-gameability, leak-free splits
python -m trg13.paths               # every line must say OK
```

### 2b. One-time checkpoint conversion (2-5 min)
```bash
python slim_ckpt_v13.py     # 9 GB last.ckpt on /kaggle/input -> /kaggle/working/viis_slim.ckpt (~2.5 GB, EMA weights folded in)
```
Every later load (smoke test, both trainings, inference) then takes seconds instead of 20+ minutes. Loading the 9 GB file
memory-mapped from Kaggle's network-mounted input storage is very slow. The smoke test and `run_all_v13.sh` do this step
automatically if the file is missing; it is lost when the session ends (re-run it, or save it as a Kaggle dataset).

### 3. Smoke test (5-10 min): do not skip
```bash
bash tests/smoke_kaggle.sh          # must end with "SMOKE TEST PASSED"
```
It builds the model plus adapter and calibrates. It then runs a step-0 evaluation, 4 training steps on the MSRS and night branches, and 2 evaluations, and saves and loads checkpoints. Finally it does tiled full-frame inference with all metrics, grids and the report. If it fails, send me the traceback: the torch code was written without a GPU.

### 4. Train (about 2-3 h on two T4s in parallel)
Interactive (notebook cell 4a), on both GPUs:
```bash
mkdir -p /kaggle/working/runs
CUDA_VISIBLE_DEVICES=0 nohup python -u train_v13.py --name v13_full > /kaggle/working/runs/v13_full.log 2>&1 &
sleep 90
CUDA_VISIBLE_DEVICES=1 nohup python -u train_v13.py --name v13_no_tc --thermal_loss 0 > /kaggle/working/runs/v13_no_tc.log 2>&1 &
```
- `v13_full` is the project model.
- `v13_no_tc` is the ablation: identical except the thermal-consistency terms. It shows what the thermal loss contributes.
- With one GPU, run them one after the other.

Unattended alternative: `bash run_all_v13.sh` in a single cell, then *Save Version -> Save & Run All (Commit)*. This trains both runs, evaluates and audits; the session limit is 12 h.

Monitor:
```bash
grep -E "EVAL|NEW BEST|STOP|DONE" /kaggle/working/runs/v13_full.log | tail
```

The training is resumable: re-running the same command continues from `latest.pt`.

**Reading `EVAL` lines** (every 250 steps, EMA weights). Step 0 is pretrained VIIS, because the adapter starts at zero.

| field | meaning | healthy |
|---|---|---|
| `psnr`, `ssim`, `lpips`, `psnr_hot` | MSRS validation vs the **real** day image (light/low/dark mix) | rise above the step-0 values, then flatten |
| `psnr_dark` / `psnr_low` / `psnr_light` | per severity | `light` must not drop: good RGB must be left alone |
| `tsc_gap`, `obj_ratio` | thermal consistency / object separation vs the GT's own level | gap falls; ratio approaches 1 |
| `n_tsc`, `n_obj`, `n_fid`, `n_hf` | night validation (no GT) | `n_hf`/step-0 below 1.6 (`hf_ratio`); `fid_drop` below 0.08 |
| `flag` | hallucination guard tripped | false |

`best.pt` is the best score (`ssim + psnr/50 - lpips - 0.5*tsc_gap`) among unflagged evaluations.

### 5. Inference and evaluation (30-60 min)
```bash
python -u infer_v13.py --ckpts /kaggle/working/runs/v13_no_tc/best.pt,/kaggle/working/runs/v13_full/best.pt \
       --labels v13_no_tc,v13 --out /kaggle/working/infer_v13
python align_audit_v13.py --out /kaggle/working/align_audit
```

**What it runs on:**
- **MSRS test:** 30 day images degraded at three fixed severities (**light / low-light / dark**). This gives full-reference PSNR/SSIM/LPIPS against the real day image.
- **MSRS test night + LLVIP test:** real night, full frames, tiled sampling at long side 512. Uses thermal and object metrics, LLVIP person boxes and YOLO recall.
- **Real test frames binned by brightness** into **light / low-light / dark** (`--k_bins` per bin).
- **Your own pairs (optional):** `--samples_dir /kaggle/input/<ds>/mypairs` with `rgb/` and `ir/` folders. Pairs are ECC-aligned automatically (`--align 1`).

**Methods compared:** input, CLAHE, **VIIS zero-shot** (pretrained), `v13_no_tc`, `v13`.

**Outputs** (in `--out`):
- `report.md`: tables plus per-image paired win rate and Wilcoxon p versus VIIS zero-shot;
- `summary.json`, `metrics.csv`;
- `grid_msrs_{light,low,dark}.png`, `grid_llvip.png`, `grid_msrs_night.png`, `grid_bin_{light,low,dark}.png`, `grid_user.png`;
- every output image in `<set>/<method>/`.

**Options:**
- `--guid 1.5` turns on trained thermal guidance (stronger hot objects; check `nr_hf` and PSNR);
- `--long_side 640` gives larger full frames (slower);
- `--steps 50` gives more DDIM steps;
- `--sets real,bins` runs a subset.

### 6. Download
Notebook cell 8 zips `infer_v13/`, `align_audit/` and the run logs and previews without checkpoints. `best.pt` is about 0.6 GB per run; download it separately if you want to keep the model.

## Main hyper-parameters (all CLI flags of `train_v13.py`)
| flag | default | why |
|---|---|---|
| `--mode` | `attn_out` | Both cross-attentions + out conv + last 3 output blocks + adapter (about 74M + 2M). Global tone and detail can move. `attn2` = thermal attention only (smaller, safer). |
| `--lr`, `--lr_tpa`, `--steps`, `--warmup` | 1e-5, 3e-5, 3000, 200 | Pretrained weights / new adapter. Cosine decay to 10%, early stopping on validation. VIIS is very sensitive to Adam's first sign-like steps (4 un-warmed steps at 2e-5 already destroyed it in the smoke test), so keep the warm-up. |
| `--bs`, `--n_img`, `--t_img` | 3, 2, 300 | MSRS batch. 2 low-t samples per step get image-space losses on the decoded x0. |
| `--w_l1 --w_ssim --w_lp --hot_w` | 0.5, 0.5, 0.2, 3 | Supervised image terms; hot regions weighted x4. |
| `--lam_tc` | 0.3 | Thermal terms on MSRS. Target = the GT's own level, so the fixed point is the GT. |
| `--lam_night`, `--night_every`, `--night_t` | 0.2, 2, 150-400 | Real-night consistency block (no GT). |
| `--th_drop` | 0.1 | Thermal dropout, which enables `--guid` at inference. |
| `--thermal_loss 0` | | Ablation run. |
| `--no_tpa` | | Architecture ablation without the adapter. |

**Tuning (change one thing at a time; resume is automatic only with the same `--name`):**
- `psnr_dark` gains are small and the loss is still falling: increase `--steps` to 5000, or `--lr` to 1.5e-5 (keep `--warmup 200`).
- `flag` is true because `hf_ratio` exceeds 1.6 (texture appearing at night): lower `--lam_night` to 0.1, or use `--mode attn`.
- Hot objects are still faint on LLVIP: set `--lam_tc 0.5`, or use `--guid 1.5` at inference.
- `psnr_light` drops below step 0: the model over-processes good images. Raise the light share in `trg13/degrade.py` `DEFAULT_MIX`.
