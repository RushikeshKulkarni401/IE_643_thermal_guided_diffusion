# VIIS_patched_v13

Thermal-RGB guided diffusion for consistent visible-image reconstruction, built on pretrained VIIS.

- **Live App:** [Gradio](https://dub.sh/T3425hx)
- **Demo Video:** [Drive](https://drive.google.com/file/d/1y-uVw2g_ighV7MXXE5AZgbtCLbj2as62/view?usp=sharing)
- **Report & Presentation:** [Drive](https://drive.google.com/drive/folders/1x6mmSphOa7ffEzrKczLJ9CdI4R-susJx)
- **How to run on Kaggle:** `README_v13.md`, or `kaggle_v13.ipynb`.
- **Why v3 to v12 failed and how v13.2 works:** `V13_DESIGN.md`.

## Overview
Fine-tunes the pretrained VIIS latent diffusion backbone using physically degraded real daytime images as ground truth, a zero-initialised **Thermal Prior Adapter (TPA)**, a $\sqrt{\text{SNR}}$-weighted **trajectory loss** against exposure bias, and bounded self-supervised night consistency losses. In **v13.2**, a Rec.601 luminance-only presence loss (`pres_lum`) eliminates blue colour cast on night pedestrians while preserving thermal contrast[cite: 66].

## Quick Start
Run on Kaggle (`2xT4`, Internet ON, with the VIIS checkpoint, MSRS, LLVIP, and this repository attached):
```bash
python tests/test_v13_numpy.py && bash tests/smoke_kaggle.sh      # checks + 5-10 min GPU smoke test
bash run_all_v13.sh                                               # train full + ablation, evaluate, audit
