# TRG-Diffusion (`VIIS_patched_v13`): Thermal–RGB Guided Diffusion for Consistent Visible Image Reconstruction

**TRG-Diffusion** fine-tunes the pretrained **VIIS** latent diffusion model to reconstruct clear, natural-colour visible images from severely degraded low-light RGB frames and paired thermal (infrared) imagery.

- **Live App:** [Gradio](https://dub.sh/T3425hx)
- **Demo Video:** [Drive](https://drive.google.com/file/d/1y-uVw2g_ighV7MXXE5AZgbtCLbj2as62/view?usp=sharing)
- **Report & Presentation:** [Drive](https://drive.google.com/drive/folders/1x6mmSphOa7ffEzrKczLJ9CdI4R-susJx)
- **How to run on Kaggle:** `README_v13.md`, or `kaggle_v13.ipynb` (and commit-mode script `kaggle_commit_v13_1.py`)
- **Why v3 to v12 failed and how v13.2 works:** `V13_DESIGN.md`

## Overview & Key Contributions

At night, visible RGB frames suffer from extreme darkness, sensor noise, colour casts, fog, and lamp glare, whereas thermal frames reliably reveal warm objects (pedestrians, vehicles) regardless of lighting but lack colour and fine texture. Naive fusion or zero-shot pretrained VIIS often produces hazy outputs, global exposure drift, or grey/tinted pedestrians.

TRG-Diffusion solves this through four core design principles:

1. **Real Daytime Ground Truth Only:** Only real daytime images (from MSRS) are used as reconstruction targets, paired with a physically motivated linear-light night degradation pipeline (exposure drops, sodium/LED lamp pools, colour casts, fog, glare, sensor noise, JPEG artefacts) and ECC thermal–RGB alignment. **No zero-shot VIIS output is ever used as a target.**

2. **Zero-Initialised Thermal Prior Adapter (TPA):** A lightweight adapter (\~2.9M parameters) initialised at zero (following ControlNet) injects thermal saliency ($s$) and visible quality ($q$) maps into the frozen UNet input blocks, preserving pretrained VIIS priors at step 0 while progressively learning thermal-guided structural cues.

3. **Trajectory Loss Against Exposure Bias:** A $\sqrt{\text{SNR}}$-weighted trajectory L1 loss ($\mathcal{L}_{traj}$) supervises the model's own multi-step DDIM denoising trajectory against real daytime ground truth, eliminating darkening drift during sampling.

4. **Luminance-Only Thermal Consistency (`pres_lum`):** Bounded self-supervised consistency losses on real night frames (MSRS-night and LLVIP) enforce thermal–visible structural correlation ($\mathcal{L}_{struct}$) and hot-object vs. background ring contrast ($\mathcal{L}_{pres}^{lum}$) strictly in **Rec.601 luminance (**$Y$**)** rather than CIELAB chrominance ($a^*, b^*$), preventing blue/purple colour casts on night pedestrians while preserving sharp thermal boundaries.

## Repository Structure

```
train_v13.py          Training loop (MSRS supervised + bounded real-night consistency), validation, and checkpoint selection
infer_v13.py          Tiled full-frame DDIM inference and benchmark evaluation across MSRS, LLVIP, brightness bins, and custom pairs
align_audit_v13.py    RGB–thermal ECC registration audit across datasets
slim_ckpt_v13.py      Converts the 9 GB pretrained VIIS last.ckpt into a fast-loading ~2.5 GB EMA-folded viis_slim.ckpt
run_all_v13.sh        End-to-end automated pipeline (2 GPUs: full model + ablation training, evaluation, and alignment audit)
kaggle_v13.ipynb      Interactive Kaggle notebook covering setup, checks, training, evaluation, and packaging
trg13/                Core library:
                        degrade.py  - Linear-light physical night degradation synthesis
                        align.py    - ECC parametric thermal-to-RGB registration
                        data.py     - Leak-free MSRS & LLVIP dataset loaders and brightness binning
                        maps.py     - Thermal saliency (s), visible quality (q), and edge maps
                        losses.py   - Trajectory loss, luminance-only presence loss (pres_lum), structural & night fidelity losses
                        model.py    - VIIS backbone wrapper, Thermal Prior Adapter, EMA scope, checkpoint I/O
                        sampler.py  - Full-frame tiled DDIM sampler with classifier-free thermal guidance
                        metrics.py  - Full-reference (PSNR, SSIM, LPIPS, PSNR-hot) and no-reference / YOLOv8n detection metrics
                        paths.py    - Automatic dataset and checkpoint path resolution
tests/                test_v13_numpy.py (CPU unit tests) and smoke_kaggle.sh (end-to-end GPU smoke test)
ldm/ taming/ informationSynthesis/   Patched VIIS latent diffusion codebase (pure-PyTorch deformable attention, taming stub,
                      gradient-checkpointing fix, null yaml ckpt_path)
```

## Datasets

| Dataset | Content | Split & Usage | 
| ----- | ----- | ----- | 
| **MSRS** | Paired daytime & nighttime road scenes (RGB + IR, $640 \times 480$) | **536 train day images** serve as the sole real ground truth (synthetically degraded on the fly). Train night frames enter the real-night consistency pool; **MSRS test** is held out for validation and full-reference evaluation. | 
| **LLVIP** | Paired nighttime street scenes (RGB + IR, $1280 \times 1024$) with pedestrian bounding boxes | Up to **6,000 train frames** enter the real-night consistency pool (no ground truth). **LLVIP test** frames are used for no-reference evaluation, colour-cast auditing, and YOLOv8n person recall. | 
| **VIIS Pretrained** | Pretrained VIIS checkpoint (`last.ckpt`, \~9 GB) | Provides the VQ-f4 autoencoder and latent diffusion UNet backbone. | 

## Step-by-Step Guide (Kaggle)

### 0. One-Time Preparation

1. Upload this repository as a Kaggle Dataset (e.g. `viis-patched-v13`).

2. Create a Kaggle Notebook with **Accelerator = GPU T4 x2** and **Internet = On** (required for installing `lpips`, `piq`, `ultralytics`, `gradio`, and downloading pretrained VGG/YOLO weights).

3. Attach the following four Kaggle Datasets via **Add Input**:

   * `elixir4o4/viis-master` (pretrained VIIS `last.ckpt`)

   * `elixir4o4/msrs-main` (MSRS dataset)

   * `afradhossain/llvip-dataset` (LLVIP dataset)

   * `viis-patched-v13` (this codebase)

   *(Paths are resolved automatically by `trg13/paths.py`. To override manually, set environment variables `MSRS`, `LLVIP`, `VIIS_CKPT`, or `RUNS`.)*

### 1. Environment Setup

Run at the start of each interactive Kaggle session:

```python
import glob, os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "omegaconf", "kornia", "lpips", "piq", "ultralytics"])
try:
    import pytorch_lightning
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning"])
src = [os.path.dirname(p) for k in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * k + "train_v13.py")][0]
shutil.rmtree("/kaggle/working/VIIS13", ignore_errors=True)
shutil.copytree(src, "/kaggle/working/VIIS13")
os.chdir("/kaggle/working/VIIS13")
```

### 2. Sanity Checks & Checkpoint Conversion

```bash
# 1. Verify degradation physics, ECC alignment, metric non-gameability, and split isolation
python tests/test_v13_numpy.py
python -m trg13.paths

# 2. Convert 9 GB last.ckpt -> ~2.5 GB EMA-folded viis_slim.ckpt for fast loading (2-5 min)
python slim_ckpt_v13.py
```

### 3. GPU Smoke Test (5–10 min)

```bash
bash tests/smoke_kaggle.sh
```

Builds the model and adapter, calibrates luminance contrast thresholds on MSRS ground truth, runs step-0 validation, executes 4 training steps across MSRS and real-night branches, tests checkpoint saving/loading, and runs tiled full-frame inference. Verify that it finishes with `SMOKE TEST PASSED`.

### 4. Training (\~2–3 hours on 2x T4)

Launch the full model (`v13_2_full`) and the thermal-loss ablation (`v13_no_tc`) in parallel across the two T4 GPUs:

```bash
mkdir -p /kaggle/working/runs
CUDA_VISIBLE_DEVICES=0 nohup python -u train_v13.py --name v13_2_full > /kaggle/working/runs/v13_2_full.log 2>&1 &
sleep 90
CUDA_VISIBLE_DEVICES=1 nohup python -u train_v13.py --name v13_no_tc --thermal_loss 0 > /kaggle/working/runs/v13_no_tc.log 2>&1 &
```

* **Single GPU:** Run the two commands sequentially.

* **Unattended Commit Run:** Run `bash run_all_v13.sh` in a single cell and select *Save Version -> Save & Run All (Commit)*.

* **Resuming:** Training automatically resumes from `latest.pt` if re-run with the same `--name`.

Monitor validation progress:

```bash
grep -E "EVAL|NEW BEST|STOP|DONE" /kaggle/working/runs/v13_2_full.log | tail
```

#### Reading `EVAL` Logs (every 250 steps, EMA weights)

Step 0 corresponds to zero-shot pretrained VIIS because the Thermal Prior Adapter is zero-initialised.

| Field | Meaning | Target Behaviour | 
| ----- | ----- | ----- | 
| `psnr`, `ssim`, `lpips`, `psnr_hot` | MSRS validation metrics vs. real daytime ground truth (light/low/dark mix) | Improve over step-0 VIIS and stabilise | 
| `psnr_dark`, `psnr_low`, `psnr_light` | PSNR broken down by degradation severity | `psnr_light` remains high (already-well-lit images are not over-processed) | 
| `tsc_gap`, `obj_ratio` | Thermal consistency gap and luminance object separation ($L^*$) relative to ground truth | `tsc_gap` decreases; `obj_ratio` approaches `1.0` | 
| `n_tsc`, `n_obj`, `n_fid`, `n_hf` | Real-night validation metrics (no ground truth) | `hf_ratio` (`n_hf` / step-0) $< 1.6$; `fid_drop` $< 0.08$ | 
| `flag` | Hallucination / over-sharpening guard | `False` (`best.pt` is selected only from unflagged checkpoints) | 

### 5. Evaluation, Alignment Audit & Interactive Demo

```bash
# Full benchmark evaluation across MSRS, LLVIP, and brightness bins
python -u infer_v13.py \
  --ckpts /kaggle/working/runs/v13_no_tc/best.pt,/kaggle/working/runs/v13_2_full/best.pt \
  --labels v13_no_tc,v13_2 \
  --out /kaggle/working/infer_v13_2

# RGB-thermal ECC registration audit
python align_audit_v13.py --out /kaggle/working/align_audit
```

**Evaluation Suites & Outputs (`--out`):**

* **Synthetic MSRS Test:** Full-reference PSNR, SSIM, LPIPS, and hot-region PSNR across `light`, `low`, and `dark` severities.

* **Real Night (MSRS-night & LLVIP-test):** Full-frame tiled DDIM reconstruction (`long_side=512`), thermal structural consistency, input fidelity, colour-cast checks, and YOLOv8n person detection recall.

* **Brightness Bins & Custom Pairs:** Stratified evaluation by input luminance (`--k_bins`), plus optional custom RGB/IR pairs via `--samples_dir <path>` (with `rgb/` and `ir/` subdirectories).

* **Generated Artefacts:** `report.md` (with paired win rates and Wilcoxon signed-rank $p$-values against zero-shot VIIS), `summary.json`, `metrics.csv`, comparison grids (`grid_*.png`), and individual reconstructed frames.

## Key Hyper-Parameters (`train_v13.py`)

| Flag | Default | Description | 
| ----- | ----- | ----- | 
| `--mode` | `attn_out` | Trainable parameters: cross-attention layers + output conv + last 3 decoder blocks + TPA (\~76.7M params). Use `attn` to freeze decoder blocks (\~54M params) or `attn2` for thermal cross-attention only. | 
| `--lr`, `--lr_tpa` | `1e-5`, `3e-5` | Learning rates for pretrained VIIS layers and the new Thermal Prior Adapter (with cosine decay to 10%). | 
| `--steps`, `--warmup` | `2500`, `200` | Total training steps and linear warm-up steps (warm-up prevents early Adam step instability in VIIS). | 
| `--bs`, `--n_img`, `--t_img` | `3`, `2`, `300` | MSRS batch size, number of low-timestep decoded $x_0$ image-loss samples per step, and timestep threshold. | 
| `--w_l1`, `--w_ssim`, `--w_lp` | `0.5`, `0.5`, `0.2` | Supervised image-space reconstruction weights (hot thermal regions weighted $\times 4$ via `--hot_w 3`). | 
| `--w_traj` | `1.0` | Weight of the $\sqrt{\text{SNR}}$-weighted DDIM trajectory L1 loss against exposure drift. | 
| `--lam_tc` | `0.3` | Weight of supervised thermal consistency ($\mathcal{L}_{struct} + \mathcal{L}_{pres}^{lum}$) calibrated to ground-truth luminance contrast. | 
| `--lam_night`, `--night_every` | `0.2`, `2` | Weight and cadence (every $N$ steps) of bounded real-night self-supervised consistency (`pres_lum` + input luminance fidelity). | 
| `--th_drop` | `0.1` | Thermal conditioning dropout probability during training, enabling classifier-free thermal guidance (`--guid`) at inference. | 
| `--thermal_loss` | `1` | Set to `0` to run the `v13_no_tc` ablation without thermal consistency losses. | 

## Acknowledgements

Built upon **VIIS**: *Visible and Infrared Information Synthesis for Severe Low-Light Image Enhancement* (WACV 2025).
