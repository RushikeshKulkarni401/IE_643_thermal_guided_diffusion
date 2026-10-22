# v13: assessment of v3 to v12 and deeppro_ver1, and the design of v13

## 1. What each previous version actually optimised

| version | target on real night (LLVIP) | what went wrong | evidence |
|---|---|---|---|
| **v3** | none. Only thermal terms: one-sided edge hinge `relu(gT - gL)` (99th-percentile normalised, clamped, **unmasked**) + presence hinge, on a 4-step DDIM x0 | The hinge only penalises **missing** edges, every pixel can earn reward and nothing pulls back, so adding texture is the cheapest way down. Adam normalises the gradient scale, so the lambda warm-up does not slow the drift. | `thermal_log`: val_edge 0.053 -> 0.007 and val_pres 0.025 -> 0.003 fall monotonically while the previews turn into texture (`thermal/prev_03000.png`). The no-thermal control stays stable (val_edge 0.053 -> 0.055). This is Goodhart's law in action. |
| v4-v9 (test01 notebook) | eps-distillation to the **frozen VIIS teacher** outside hot masks + capped presence/edge | Any anchor to the frozen teacher's output caps you at the teacher. MSRS hot-PSNR moved +0.1-0.3 dB, within the noise floor the notebook measured itself (std ~0.45 dB). | `EVAL` lines of v2/v6; thermal probe: hot-region L1 0.106 (real thermal) vs 0.220 (black thermal). **VIIS already uses thermal**, so the problem was never "the model ignores thermal". |
| **v10** | `cache["istar"] = look_v10(VIIS_zero_shot(guided, guid 1.5))`: VIIS's own output, CFG-guided and passed through a tone-stretch / detail-gain / chroma x1.5 filter. Latent eps loss to `z(istar)` + L1 + Laplacian-pyramid L1 to `istar`. | **Self-distillation of "VIIS + a CPU filter".** The student's ceiling is its target. The validator measures `reach` / `vs_target` relative to the same targets (circular). Only 80 cached training images. | `v10_with.log`: `vs_zeroshot lap x2.1, col x1.5`, i.e. how well the filter was copied. `dark_full.png`: the v10 column is visually the `zero-shot+look` column. |
| **v11 / v12** | same `istar` cache (`--arb_teacher` only changes the teacher's guidance mask) + per-pixel routing `w_vis / w_th / w_fb` of the loss weights | Reconstruction weights `lam_vis = 1.0`, `lam_fb = 1.5` still pull towards `istar` = zero-shot + look, so it still depends on zero-shot. The routing itself is a sound idea. | `infer_v11` summary: every metric is a ratio to zero-shot; `v10+arb` / `v11+arb` gains come from guidance at sampling time, not from training. |
| **deeppro_ver1** | none (MSRS synthetic only) | Trains a ~10M-parameter pixel-space residual diffusion UNet **from scratch** on ~1k MSRS day images. That is not enough data to learn night -> day appearance, so it cannot match a pretrained prior. Its thermal recall loss is again one-sided (v3 family), and it never sees a real night frame. | Design-level; no run results were attached. Its good parts are reused in v13: linear-light degradation (fog, glare, shot/read noise), ECC alignment, chance-corrected thermal-edge score, 3-arm ablation, strict splits. |

The root cause of VIIS's hazy, washed-out output on LLVIP is a domain gap. VIIS was trained on MSRS day images darkened by `contrast x U(0.1,1) -> gamma U(3,10) -> sRGB noise`, which is a global curve. Real LLVIP night frames look different: lamp pools, sodium/LED colour casts, glare, JPEG and linear-light sensor noise. The legitimate way to beat zero-shot is to close that gap **with real ground truth**, not to distil a filtered copy of VIIS.

## 2. v13 in one paragraph
v13 starts from the pretrained VIIS weights (EMA). It fine-tunes cross-attention + output blocks (73.7M parameters) plus a new zero-initialised **Thermal Prior Adapter** (step 0 is exactly VIIS). The **only** training target is the real MSRS day image, paired with a **physically based** night/smoke/glare degradation that matches real night statistics. On real night frames (LLVIP train + MSRS night, which have no ground truth) it adds **bounded** self-supervised consistency terms: structure must follow the RGB input where the RGB is informative, and agree with the thermal image (up to the natural level) where it is not. No model output is ever used as a target. Model selection uses MSRS validation with real ground truth, and night guards stop texture hallucination.

## 3. Deliverables and how v13 covers them

**D1. Alignment and feature extraction**
- `trg13/align.py`: coarse-to-fine ECC on gradient maps. Gradients are used because RGB and thermal intensities are not comparable but their boundaries are. The default motion model is euclidean; full affine drifts in scale cross-modally (measured: affine moved the corners 17 px, euclidean 5.6 px, a self-pair 0.02 px). A warp is accepted only if it raises the gradient correlation.
- `align_audit_v13.py` reports the residual of the shipped pairs, the recovery of known 2/4/8 px misalignments (on the sample pair: 3-13 px before -> about 0.2 px after), and the edge overlap before and after.
- Training adds small random misregistrations (30%), so the network tolerates imperfect pairs. User pairs are ECC-aligned at inference (`--align 1`).
- Features per modality:
  - VIIS encoders: VQ latents of both modalities concatenated to x_t, plus multi-scale features cross-attended through deformable dual-modal attention.
  - v13 explicit maps: thermal luminance, thermal edges (structural), thermal saliency `s` (semantic: warm objects), visible quality `q` (where the RGB is trustworthy), and lifted RGB luminance.

**D2. Thermal-guided diffusion**
Thermal enters the diffusion latent space at three levels:
- VIIS's concatenated thermal latent and thermal cross-attention (trained);
- the TPA's zero-conv residuals into every UNet input block (ControlNet-style, `trg13/model.py`);
- classifier-free thermal dropout (10%), so `--guid > 1` at inference is a trained operation, not a hack.

**D3. Object preservation**
- Hot objects are attenuated in the RGB input (35% of training samples, keep 0-30%), so the only evidence for them is the thermal channel. Real ground truth then teaches the network to reconstruct them.
- The eps, L1 and LPIPS losses are hot-weighted (x4 inside `s`).
- `L_pres` (below) keeps hot objects separated from their surroundings.

**D4. Thermal consistency loss** (`trg13/losses.py`)
- `L_struct = mean_w relu(tau - |LNCC(Y_out, T)|)`.
  - |LNCC| is bounded and invariant to gain and offset.
  - Random texture *decorrelates*, so it **raises** the loss. This is the opposite of v3. Tested in `tests/test_v13_numpy.py`: TSC 0.489 clean, 0.460 with noise, 0.934 with thermal pasted in.
  - `tau` is the level the **real day image** has: per pixel from the GT on MSRS (fixed point = GT), a scalar calibrated on MSRS GT for night frames. So copying thermal into RGB is not rewarded beyond what natural images show.
- `L_pres = relu(tau_c - ||Lab(object) - Lab(ring)||)`: sign-agnostic separation of object and surroundings in regional means, capped at the natural level. Fine texture cannot change it.
- Both terms are weighted by `(1-q)` on night frames. The user's v12 per-pixel decision idea is kept here as a **loss router and adapter input**: thermal is consulted where the RGB is poor, the input-structure term `L_in` where the RGB is good.

**D5. Evaluation** (`infer_v13.py`, `trg13/metrics.py`)
- **MSRS test, synthetic light/low/dark (real GT):** PSNR, SSIM, LPIPS, hot-region PSNR, TSC gap to GT, object-contrast ratio to GT.
- **Real night (LLVIP test, MSRS test night):**
  - thermal consistency: TSC, chance-corrected thermal-edge score;
  - object preservation: Lab object/ring separation, LLVIP person-box contrast, YOLOv8n person recall;
  - hallucination guards: input-structure fidelity, unsupported high-frequency energy;
  - BRISQUE.
- **Real samples binned by brightness into light / low / dark.**
- Every metric is paired per image against VIIS zero-shot (win rate + Wilcoxon p), and the run includes an ablation without the thermal losses (`v13_no_tc`).

## 4. Training objective (defaults)

**MSRS (bs 3, real GT):**

`L = eps_MSE(1 + 3s) + 0.5 L1(1 + 3s) + 0.5 (1 - SSIM) + 0.2 LPIPS_vgg + ramp * 0.3 (L_struct[tau = GT] + L_pres[tau = GT])`

The image-space terms are applied to the decoded x0 estimate of 2 samples with t < 300. Thermal dropout is 0.1.

**Night (bs 2, every 2nd step, no GT):** a 4-step no-grad DDIM down to t in [150, 400], then one step with grad to x0, then the VQ decoder.

`L_n = ramp * 0.2 (L_in[q] + L_struct[(1 - q) s, tau_cal] + L_pres[(1 - q), tau_cal])`

**Optimiser:** AdamW, lr 1e-5 (pretrained weights) / 3e-5 (new adapter), 200-step warm-up, cosine decay to 10%, grad clip 1, EMA 0.999, 3000 steps.

**Validation (every 250 steps, EMA weights):**
- 12 held-out MSRS train day images with fixed light/low/dark degradations (real GT);
- 12 night frames from a held-out LLVIP sequence and MSRS night.

Score = `SSIM + PSNR/50 - LPIPS - 0.5 * TSC_gap`. A checkpoint is flagged if the night HF energy exceeds 1.6x step 0 or the input fidelity drops by more than 0.08. Two consecutive flags stop the run; 4 evaluations without improvement (after 750 steps) also stop it.

**Leak rules:**
- MSRS train/val split every 10th name;
- LLVIP held-out sequence = last 2-digit prefix;
- final numbers only on the official test splits;
- step-0 VIIS outputs appear only in previews and as the *reference for the hallucination guard*, never as a training target.

## 5. Honest limits
- The PyTorch code was written and syntax-checked without a GPU or the 9 GB checkpoint. The numpy parts (degradation, alignment, metrics, data splits) were run and tested. `tests/smoke_kaggle.sh` exercises every torch path in about 5-10 minutes; run it first.
- LLVIP has no ground truth, so "better than VIIS on LLVIP" can only be shown by no-reference, thermal and object metrics plus visual grids. The full-reference evidence comes from MSRS test.
- The calibrated `tau` values come from MSRS day images (a different camera from LLVIP). They are printed at the start of training (`calibrated natural thermal-consistency levels`).
- `q` and `s` are hand-designed maps. They gate losses and feed the adapter, but the network is free to learn its own use of them.

## 6. v13.1: what the two diagnostic sweeps showed and what changed

| sweep finding | evidence | change in v13.1 |
|---|---|---|
| The training and evaluation code is correct. | `z_sanity` (learning rate 0): every metric is identical at step 0 and step 125. | none needed |
| The losses, learning rate, adapter and timestep bias are **not** the cause of the MSRS drop. | Sweep 1 plus `e`/`f`: pure latent loss, 3x lower learning rate, uniform timesteps, with or without the adapter: all drop 4-7 dB by step 125. | uniform timesteps kept (correct anyway) |
| The drop is mostly **global exposure**: outputs drift darker. | Mean luminance of MSRS validation outputs: VIIS 0.29, MSRS-only arms 0.21-0.26, ground truth 0.36. | trajectory loss (below) |
| Arms **with** the real-night branch recover; MSRS-only arms do not. | 500 steps: `g` 19.7 dB / `h` 18.8 dB vs `e` 16.3 dB / `f` 17.0 dB. Brightness of `g` returns to 0.30. | same mechanism, now applied with GT |
| The validation set was unfair to fine-tuning. | VIIS was trained on `msrs/rgbs/train` (Image2Image.yaml). Our validation set was every 10th MSRS **train** image, which VIIS had memorised (21.8 dB there vs 18.0 dB on test). | validation = every 5th MSRS **test** image; the final evaluation uses the other test images (disjoint) |
| The full objective already helps LLVIP. | All arms raise held-out night object separation (7.6 -> 9.6-10.4) and input fidelity (0.69 -> 0.84). In the previews, people that VIIS turns into grey blobs stay visible. | checkpoints are now **selected on LLVIP first** |

**Why the night branch fixed brightness: exposure bias.** The standard diffusion loss only shows the network noisy versions of the *true* image (`q_sample(z_gt, t)`). At inference the network sees its own trajectory, starting from pure noise. LDM's linear schedule never reaches zero signal-to-noise: at t = 999, about 1.2% of the true latent is still in x_t. During training the network therefore learns to read global brightness from x_t at high t, but at inference that signal is absent. The night branch is the only term that trains on the model's own trajectory from pure noise, which is why only those arms recovered exposure.

**v13.1 trajectory loss (`--w_traj 1.0`)** applies the same idea to MSRS, where we have ground truth:
1. Take one MSRS sample per step.
2. Run a deterministic DDIM walk (up to 4 steps, no gradients) from pure noise, conditioned on its degraded RGB and thermal, down to a random `t_stop` anywhere in [0, 999].
3. Take one step with gradients to predict x0, then decode it.
4. Compare with the **real day image**: L1 at 1/8 resolution (global tone and colour) at every `t_stop`, plus L1 / SSIM / LPIPS at full resolution when `t_stop < 300`.

`t_stop = 999` trains exactly the first inference step (pure noise -> image), which is where brightness is decided. The target is always the real day image; no model output is ever a target.

**Selection (`--select night`, default):**
- **Eligibility:** a checkpoint qualifies only if MSRS-test-validation PSNR >= step-0 PSNR - 2 dB (`--msrs_tol`).
- **Score among eligible checkpoints:** night input fidelity + night thermal structure + 0.02 x night object separation + 0.5 x the MSRS score.
- **Texture guard:** raised to 3.0x, because VIIS is over-smooth on LLVIP (BRISQUE 0.4, below the raw input's 11), so 1.5-2.2x more texture is real detail.

## 7. v13.1 results and the v13.2 change

**v13.1 results (official test splits, paired against VIIS zero-shot):**
- **MSRS** (scored against the real day images): PSNR 21.92 dB vs 18.45 dB; better at every severity (light 24.9 vs 19.5, low 21.4 vs 18.4, dark 19.4 vs 17.5).
- **LLVIP + MSRS night:**
  - input fidelity: 0.858 vs 0.671;
  - YOLO person recall: 0.50 vs 0.22;
  - unsupported texture: lower than VIIS (0.0051 vs 0.0067), so no hallucinated texture.

**Artifact found in the grids:** with the thermal-consistency terms on, pedestrians on sodium-lit LLVIP frames turn **blue or purple**. On LLVIP outputs, 3.9% of pixels are strongly blue, against 1.0% without the thermal terms and 1.7% in the input; purple pixels are 0.57% against about 0%. MSRS shows no such shift.

**Cause:** `L_pres` measured object/ring separation in full L\*a\*b\*. Against yellow ground, the cheapest way to separate a dark pedestrian is to move it towards blue on the b\* axis. This also inflated the colour-based object metrics (`nr_obj` 12.96, `nr_box` 15.66).

**v13.2:**
- `L_pres` and its calibration use **L\* only** (`--pres_lum 1`). Thermal evidence says *where* an object is, not *what colour* it is.
- New metrics separate the two effects. `obj_L` / `box_L` measure brightness separation. `obj_ab` measures colour separation, to be compared with the input or ground-truth row; much higher means invented colour.
- Checkpoint selection uses `obj_L`.
