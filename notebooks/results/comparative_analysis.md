# Comparative Written Analysis — Malware Concept Drift (Latent Neural ODE vs Static Baselines)

All numbers below are **real, computed results** from training and evaluating on identical chronological BODMAS splits. No synthetic or hypothesized values are reported. Where the data does not exhibit the behavior hypothesized in the project goals, that is stated explicitly.

## 1. Experimental setup (identical chronological splits for all models)

Data: `dataset/bodmas_clean_features.parquet` — 121,870 samples, PE features `F1..F2384`, labels `Label` (0 benign / 1 malware), `timestamp`.

Chronological split (no leakage; train always precedes test):

| Split | Months | Samples |
|-------|--------|---------|
| Train | 2019-08 .. 2020-03 | 62,919 |
| Val   | 2020-04 .. 2020-05 | 20,950 |
| Test  | 2020-06, 2020-07, 2020-08, 2020-09 | 38,001 |

Test-class balance: 16,991 malware / 21,010 benign (~45% malware).

To keep training feasible on a single consumer CPU/GPU, experiments use the **top-600 drift-ranked features** (from `dataset/feature_drift_ranking.csv`). Every model (static + ODE) is trained on the same train split and evaluated on the same four test months.

## 2. Headline results (real metrics)

Comparison table (overall test window):

| Model | Type | AUROC | Macro F1 | AUT |
|-------|------|-------|----------|-----|
| LightGBM (static) | static | 0.9999 | 0.9953 | 2.9997 |
| XGBoost (static) | static | 0.9999 | 0.9943 | 2.9996 |
| MLP (static) | static | 0.9966 | 0.9787 | 2.9915 |
| Neural ODE (rk4, focal) | continuous | 0.8564 | 0.7833 | 2.5719 |
| Neural ODE (dopri5, focal) | continuous | 0.8100 | 0.7440 | 2.4200 |
| Decoupled encoder/decoder (no ODE) | continuous | 0.6745 | 0.5403 | 1.9348 |

Per-month AUROC (test window):

| Model | 2020-06 | 2020-07 | 2020-08 | 2020-09 |
|-------|---------|---------|---------|---------|
| XGBoost (static) | 0.9999 | 0.9998 | 0.9999 | 0.9999 |
| LightGBM (static) | 0.9999 | 0.9998 | 1.0000 | 0.9999 |
| MLP (static) | 0.9995 | 0.9963 | 0.9980 | 0.9950 |
| Neural ODE (dopri5) | 0.8148 | 0.8390 | 0.7782 | 0.7907 |
| Neural ODE (rk4) | 0.8409 | 0.8497 | 0.8596 | 0.8843 |
| Decoupled (no ODE) | 0.7722 | 0.7013 | 0.5527 | 0.5894 |

Visualizations (real metrics, no synthetic data):
- `graphs/auroc_vs_month.png` — month-by-month AUROC per model
- `graphs/aut_comparison.png` — AUT bar chart + cumulative AUROC curves
- `graphs/umap_latent_space.png` — initial (z₀) vs evolved (z₁) latent states from the trained dopri5 ODE

## 3. Goal-by-goal assessment vs. what the data actually shows

### Static Baseline Decay (hypothesized: up to ~50% AUROC drop)
**Not observed.** On this BODMAS window the static baselines are essentially flat:
- XGBoost / LightGBM stay at AUROC ≈ 0.9998–1.0000 across all four test months (AUT ≈ 3.0, the maximum).
- MLP declines only slightly (0.9995 → 0.9950), AUT 2.9915.

There is **no measurable decay to beat**. This matches the pre-existing EDA verdict in `dataset/drift_eda_summary.txt`: *"DRIFT IS WEAK. RECONSIDER MODEL COMPLEXITY."* The hypothesized ~50% baseline decay does not materialize on this data, so a drift-resilient model has no large gap to close here.

### Neural ODE Stability (hypothesized: AUT within ~10% of initial AUROC)
The ODE trajectories are relatively flat (no steep collapse), but in absolute terms the ODE **underperforms** the static baselines on this weak-drift data:
- dopri5: AUROC 0.7782–0.8390, AUT 2.4200
- rk4: AUROC 0.8409–0.8843, AUT 2.5719

Both are well below XGBoost/LightGBM (~1.0). So the ODE is "stable" in the sense of not decaying sharply, but it is not competitive here because there is no drift to exploit and the small CPU-trained latent model underfits relative to the powerful tree ensembles.

### Class-Imbalance Robustness (Focal Loss)
Focal Loss (α=0.25, γ=2.0) was used for all ODE runs. Test macro-F1 on the mixed (≈45% malware) window:
- ODE dopri5: 0.7440; rk4: 0.7833
- Static baselines: 0.9787–0.9953

The ODE's macro-F1 is lower than the baselines, again because the baselines already separate the classes near-perfectly on this data. Focal Loss is implemented and active, but its benefit cannot be demonstrated when class separation is already trivial.

### Computational Efficiency (adjoint sensitivity)
The ODE trains with `torchdiffeq.odeint_adjoint` (constant-memory backprop w.r.t. NFE) on **CPU** (this environment has no CUDA: `torch 2.13.0+cpu`). Full training (8 epochs, ~15k subsampled train samples, 600 features, latent_dim=8, hidden=32) completed successfully, demonstrating the adjoint method runs within consumer-grade memory constraints. On a GPU with the full dataset the memory savings of the adjoint method would be the decisive factor (constant memory vs. O(NFE) for naive backprop through the solver).

### Sudden-Drift Resilience / Neural Jump ODE (stretch goal)
**Not developed.** The Neural Jump ODE extension for discontinuous drift windows (e.g., COVID-19 onset) was not implemented. Per instructions, stretch-goal components that were not built are not added or fabricated here.

## 4. Architectural ablation studies

| Variant | Solver | ODE? | AUROC | Macro F1 | AUT | Notes |
|---------|--------|------|-------|----------|-----|-------|
| Neural ODE | dopri5 (adaptive) | yes | 0.8100 | 0.7440 | 2.4200 | default |
| Neural ODE | rk4 (fixed-step) | yes | 0.8564 | 0.7833 | 2.5719 | slightly higher AUROC |
| Decoupled enc/dec | — | no | 0.6745 | 0.5403 | 1.9348 | clear decay |

Key ablation finding — **continuous integration matters**: the decoupled encoder/decoder (no ODE) shows pronounced month-by-month AUROC decay (0.7722 → 0.7013 → 0.5527 → 0.5894), behaving like a static baseline. Adding the ODE (dopri5/rk4) flattens the trajectory (no monotonic collapse). So the ODE *does* improve drift-robustness relative to an identical static encoder/decoder — just not enough to beat the static baselines on this particular weak-drift data.

dopri5 vs rk4: comparable (0.81 vs 0.86 AUROC). rk4 is marginally better here, but neither approaches the static baselines.

## 5. Error analysis — where the ODE offers little improvement and why

The ODE model does **not** improve over static baselines on this data, and in fact performs substantially worse (0.81–0.86 vs ~1.0 AUROC). The reasons:

1. **No real drift to model.** Feature drift exists (top features drift up to 0.36 magnitude) but it is not sufficient to degrade a well-fit classifier. XGBoost/LightGBM already achieve ~1.0 AUROC in every test month, so the conditional P(y|x) is effectively stationary. A continuous-time model's advantage (extrapolating feature evolution across time) has no signal to latch onto.

2. **Underfitting from the small latent bottleneck.** The CPU-feasible configuration compresses 600 features into an 8-dimensional latent state and trains for only 8 epochs on ~15k samples. The static tree ensembles use all 600 features directly with hundreds of trees. On an easy, separable problem the high-capacity static models dominate; the low-dimensional continuous latent model underfits.

3. **Time conditioning is weak.** The decoder appends the normalized observation time `t` to the evolved latent state. With weak drift, this time signal carries little predictive information and can add noise.

4. **Class balance is not extreme.** Test malware share ≈ 45%, so Focal Loss's reweighting of hard/imbalanced examples has little headroom to improve macro-F1 beyond what balanced training already achieves.

**When the ODE would be expected to shine:** on data with strong, continuous concept drift where static models' decision boundaries go stale (the ~50% decay scenario hypothesized). That regime is simply not present in this BODMAS window. The decoupled ablation confirms the ODE mechanism works (it suppresses the decay seen in the no-ODE variant); it is the data's weak drift, not a broken ODE, that limits the absolute gain.

## 6. Latent-space visualization

`graphs/umap_latent_space.png` maps initial latent states z(t₀) vs chronologically-evolved states z(t₁) for benign/malware test samples through the trained dopri5 ODE. Because drift is weak, the benign and malware clusters remain broadly separable in both views and the malware cluster does not migrate dramatically — consistent with the AUROC results showing little temporal shift. On a strongly drifting dataset one would expect the malware cluster to drift toward/overlap the benign region at t₁; that topological migration is not evident here.

## 7. Deliverables status

| Deliverable | Status |
|-------------|--------|
| Trained Latent Neural ODE + static baselines on identical splits | ✅ Done (checkpoints in `results/`) |
| Month-by-month AUROC + AUT visualizations | ✅ `graphs/auroc_vs_month.png`, `graphs/aut_comparison.png` |
| Comparative written analysis | ✅ This document |
| Architectural ablation (dopri5 vs rk4; decoupled enc/dec) | ✅ Done |
| Latent-space UMAP (initial vs evolved) | ✅ `graphs/umap_latent_space.png` |
| Neural Jump ODE / sudden-drift stretch goal | ❌ Not developed (not fabricated) |
| Hypothesized ~50% baseline decay / ODE ~10% stability | ❌ Not observed (weak drift); reported honestly |

## 8. Reproduction

All code lives under `src/`:
- `chronology.py` — chronological split + bounded feature loading
- `ode_model.py` — Latent Neural ODE (encoder / adjoint ODE / decoder), Focal Loss, dopri5 & rk4, decoupled variant
- `train_baselines.py` — XGBoost / LightGBM / MLP training + month-by-month eval
- `train_ode.py` — ODE training (`--solver dopri5|rk4`, `--loss focal|bce`, `--decoupled`)
- `evaluate.py` — per-month AUROC, AUT (trapezoidal), macro F1
- `visualize.py` — all real-metric plots + comparison table

Run from `src/`:
```
python train_baselines.py
python train_ode.py --solver dopri5 --loss focal --epochs 8
python train_ode.py --solver rk4 --loss focal --epochs 8
python train_ode.py --decoupled --loss focal --epochs 8
python visualize.py
```
