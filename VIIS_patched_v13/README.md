# VIIS_patched_v13

Thermal-RGB guided diffusion for consistent visible-image reconstruction, built on pretrained VIIS.
- **How to run on Kaggle:** `README_v13.md`, or `kaggle_v13.ipynb`.
- **Why v3 to v12 failed and how v13 works:** `V13_DESIGN.md`.

Quick start (Kaggle, 2xT4, Internet on, with the VIIS checkpoint, MSRS, LLVIP and this repo attached):
```bash
python tests/test_v13_numpy.py && bash tests/smoke_kaggle.sh      # checks + 5-10 min GPU smoke test
bash run_all_v13.sh                                               # train full + ablation, evaluate, audit
```
Original VIIS: "VIIS: Visible and Infrared Information Synthesis for Severe Low-light Image Enhancement" (pretrained
checkpoint from the authors' Google Drive link; place or attach `last.ckpt`).
