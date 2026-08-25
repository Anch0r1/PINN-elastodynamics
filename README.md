# PINN-Elastodynamics-PyTorch

A PyTorch reimplementation of the composite Physics-Informed Neural Network (PINN)
from **Rao, Sun & Liu (2021), *"Physics-informed deep learning for computational
elastodynamics without labeled data"*** ([arXiv:2006.08472](https://arxiv.org/abs/2006.08472)),
applied to the paper's plate-with-a-circular-hole benchmark (Section 3.1: defected
plate under cyclic uni-axial tension).

The original authors released a TensorFlow 1.x implementation
([Raocp/PINN-elastodynamics](https://github.com/Raocp/PINN-elastodynamics)), included
here as `reference/train.py` for direct comparison. This repo is an independent
rewrite in modern PyTorch.

## What this solves

A quarter-symmetry square plate (0.5 m × 0.5 m) with a circular hole (r = 0.1 m) at
the origin, under cyclic uni-axial traction on the right edge:

```
Tn(t) = 0.5 · sin(2π t / T₀ + 3π/2) + 0.5,   T₀ = 5 s,  t ∈ [0, 10] s
```

Plane-stress, linear elastic (E = 20 MPa, ν = 0.25, ρ = 1.0). The network predicts
the **mixed-variable** output (u, v, σ11, σ22, σ12) directly, following the paper's
finding that this is critical for trainability over a pure-displacement formulation
(their Appendix A). Initial and boundary conditions are enforced in a **"hard" /
composite** manner:

```
U(x, y, t) = U_particular(x, y, t) + D(x, y, t) ⊙ U_general(x, y, t)
```

`U_particular` and `D` (the distance function) are pretrained, frozen auxiliary
networks; only `U_general` is trained against the PDE. No labeled/simulation data is
used at any stage — the physics network is trained purely against the governing-PDE
residual, the constitutive (Hooke's law) residual, and the traction-free condition on
the hole surface.

## Repository structure

```
.
├── README.md
├── LICENSE
├── notebook/
│   ├── plate_with_hole.ipynb    # full pipeline: point generation -> pretraining ->
│   │                             #   physics training -> inference/plots (Colab-ready)
│   ├── dist_net_final.pth       # trained distance-function network
│   ├── part_net_final.pth       # trained particular-solution network
│   └── uv_net_final.pth         # trained general-solution network
├── reference/
│   └── train.py                 # original authors' TF1 script, kept for citation/diff
└── results/
    ├── displacement_fields.png
    ├── hole_surface_stress.png          # σ11/σ22/σ12 at t=2.5,3.75,5.0s (final, Stage 2)
    ├── stress_snapshots.png
    ├── pinn_stress_animation.gif
    ├── von_mises_history.png            # (final, Stage 2)
    ├── stage1_hole_surface_stresses.png # w_hole=500 checkpoint, kept as a before/after
    ├── stage1_von_mises_history.png     #   comparison against the final run above
    └── paper_reference/
        ├── paper_s_plots.png            # paper's own Fig. 8 (FEM + their PINN)
        ├── paper_von_mises.png          # paper's own Fig. 9
        └── paper_animation.gif
```

This mirrors how the original authors structured their own repo (one training
script + saved network weights + a results folder), rather than splitting into a
Python package — appropriate for a single-case student project rather than a library
meant for reuse.

## Setup

```bash
git clone https://github.com/<you>/PINN-Elastodynamics-PyTorch.git
```

Developed and run entirely on Google Colab, using Colab's default environment
(PyTorch, NumPy, SciPy, Matplotlib preinstalled — no extra setup needed there). If
running outside Colab, you'll need at minimum: `torch`, `numpy`, `scipy`,
`matplotlib`, `pyyaml`-free (no config files used), `pillow` (for the gif).

Open `notebook/plate_with_hole.ipynb` in Colab or Jupyter. The notebook mounts Google
Drive for checkpointing — training this model in one sitting is not realistic on
free-tier Colab compute, so the notebook is structured around save/resume
checkpoints rather than a single uninterrupted run. See the notebook's own section
headers ("Resume training Helpers", "Option 1" / "Option 2") for how that works, and
the note in [Differences from the original](#differences-from-the-original-implementation)
below on which path actually produced the results in `results/`.

To reuse the already-trained networks without retraining, load the three `.pth`
files in `notebook/` directly into `DistNet` / `PartNet` / `PINN_Elastodynamics.uv_net`.

## Method summary

| Component | Architecture | Role |
|---|---|---|
| `DistNet` (𝒟) | `[3, 20, 20, 20, 20, 5]`, tanh | Pretrained distance function — zero on I/BC surfaces, nonzero in the interior. Frozen after pretraining. |
| `PartNet` (𝒰ₚ) | `[3, 20, 20, 20, 20, 5]`, tanh | Pretrained particular solution — matches prescribed I/BC values exactly. Frozen after pretraining. |
| `uv_net` (𝒰ₕ / general solution) | `[3, 70×8, 5]`, tanh | The only network trained against the PDE residual. |

Because `D = 0` on every I/BC surface by construction, `U = U_p + D ⊙ U_h` collapses
to `U_p` there automatically — boundary/initial conditions are satisfied exactly
rather than penalized softly in the loss, avoiding the soft-constraint gradient
pathology the paper documents in its Appendix B.

**Loss function** (physics-network stage; `𝒟` and `𝒰ₚ` are pretrained separately
against closed-form/prescribed targets):

```
L = w_pde · ( ‖f_u‖² + ‖f_v‖² + ‖f_σ11‖² + ‖f_σ22‖² + ‖f_σ12‖² ) + w_hole · ( ‖t_x‖² + ‖t_y‖² )
```

`f_u, f_v` are the momentum-equation (Navier–Cauchy) residuals, `f_σij` are the
Hooke's-law residuals between the network's direct stress output and the
autodiff-derived stress from strain, `t_x, t_y` is the traction on the traction-free
hole surface. This training used a **two-stage weight schedule** rather than a fixed
`w_pde = w_hole` throughout — see [Differences from the original
implementation](#differences-from-the-original-implementation) below for why.

## Results

Comparisons are against FEM reference data digitized from the paper's own figures
(see `results/paper_reference/`, which also includes the original authors' own PINN
curves for a three-way comparison).

**σ11 / σ22 / σ12 on the hole surface at t = 2.5, 3.75, 5.0 s** (paper's Fig. 8, vs.
`results/paper_reference/paper_s_plots.png`): `results/hole_surface_stress.png` (the
final, Stage 2 run) reproduces the qualitative shape and sign of all three stress
components and is close on peak magnitude at t = 2.5 s (σ11 ≈ 3.3 kPa predicted vs.
≈ 3.4 kPa FEM). Agreement is visibly weaker at t = 3.75 s and 5.0 s.
`results/stage1_hole_surface_stresses.png` is kept alongside it deliberately — it's
the `w_hole = 500` checkpoint, and shows how that early over-weighting distorted the
interior field even while satisfying the hole condition tightly. Read as a
before/after pair, not two competing "best" results.

**Von Mises stress history at (0, 0.1)** (paper's Fig. 9, vs.
`results/paper_reference/paper_von_mises.png`): `results/von_mises_history.png`
tracks the applied load's periodicity correctly (T₀ = 5 s) but the peak magnitude
(~2.7 kPa) undershoots the reference (~3.3 kPa) by roughly 18%.
`results/stage1_von_mises_history.png` is the corresponding Stage-1 checkpoint.

`results/pinn_stress_animation.gif` shows the full field over the 10 s simulation,
alongside `results/paper_reference/paper_animation.gif` for direct comparison.
`results/displacement_fields.png` and `results/stress_snapshots.png` give additional
full-field views not directly plotted in the paper's figures.

## Known limitations

1. **Not fully converged.** Training was stopped due to Google Colab compute/runtime
   limits, not because the loss plateaued. The L-BFGS loss was still decreasing when
   training was stopped.
2. **Lower collocation density than the original** — see the differences section
   below. This most likely explains the residual gap near the hole, where the
   original paper specifically doubles collocation density to capture the stress
   concentration.
3. **No convergence/mesh-refinement study**, unlike the paper's own network-size
   sweep (their Table 1) — there's no direct evidence here that the current network
   capacity is sufficient at this collocation density, only that results look
   reasonable at the one configuration tried.
4. This is a course/faculty-supervised project result, not a peer-reviewed
   reproduction — treat all numbers above as indicative, not final.

## Differences from the original implementation

Everything below documents how this repo's training diverged from
`reference/train.py`, and why — so nobody has to reverse-engineer it later.

### Framework port (TF1 → PyTorch)

Straightforward translation: `tf.placeholder` → tensors with `requires_grad=True`,
`tf.gradients` → `torch.autograd.grad`, `tf.contrib.opt.ScipyOptimizerInterface`
(L-BFGS-B) → `torch.optim.LBFGS` with `line_search_fn="strong_wolfe"`. Xavier/Glorot
init and `tanh` activations preserved. `float64` preserved throughout (the original
explicitly uses `float64`, relevant for the accuracy of the 4th-order-derivative
residuals computed via autodiff).

### Hole-boundary loss weight — two-stage schedule, not a fixed match to the original

**Original (`reference/train.py`, line 217):**
```python
self.loss = 10 * (self.loss_f_uv + self.loss_f_s + self.loss_HOLE)
```
All three residual terms — momentum-equation, constitutive, and hole-traction — share
a single multiplier of 10, fixed for the entire training run.

**This repo used a two-stage schedule instead of matching that fixed weighting
throughout:**

- **Stage 1** (`plate_with_hole.ipynb`, first physics-training cell): `w_pde = 10`,
  `w_hole = 500`. An earlier attempt at equal weighting (`w_pde = w_hole = 10`, i.e.
  matching the original from the start) collapsed toward a trivial, near-zero
  solution — the hole-traction term wasn't being satisfied strongly enough early in
  training to pull the model out of it. Upweighting `w_hole` to 500 was a deliberate
  intervention to force the traction-free hole condition to be learned first.
- **Stage 2** (the "resume training helpers" section, Option 2 path): once Stage 1
  training showed the model was no longer collapsing to the trivial solution,
  training was resumed with `w_pde = w_hole = 10` — now matching the original
  paper's fixed weighting exactly — for final refinement. This is the path that
  produced `dist_net_final.pth` / `part_net_final.pth` / `uv_net_final.pth` and every
  non-"stage1" plot in `results/`.

`results/stage1_hole_surface_stresses.png` and `results/hole_surface_stress.png` are
kept side by side deliberately to show this before/after — Stage 2 recovers
noticeably better agreement with the FEM reference on peak magnitudes.

**Open question worth investigating further:** whether the trivial-solution collapse
under equal weighting is a general property of this problem's loss landscape (in
which case the original paper's authors may have relied on their much larger
collocation point count, below, to avoid it) or specific to this repo's
initialization/point sampling. Not resolved here.

### Collocation point count — known gap, not resolved

| Point set | Original | This repo | Ratio |
|---|---|---|---|
| Global collocation (`XYT_c`, before hole removal) | 70,000 | 35,000 | 0.5x |
| Stress-concentration refinement (`x,y ∈ [0,0.15]`) | 40,000 | 20,000 | 0.5x |
| Hole surface (`HOLE`), spatial × temporal | 83 × 120 = 9,960 | 83 × 120 = 9,960 | 1.0x |
| Initial condition (`IC`) | 5,000 | 5,000 | 1.0x |
| Each symmetry/free/loaded edge (`LF`,`LW`,`UP`) | 8,000 each | 8,000 each | 1.0x |
| Loaded edge (`RT`) | 13,000 | 13,000 | 1.0x |

The interior/refinement collocation counts are at half the original's density; every
boundary and initial-condition point set matches exactly. This was a deliberate
compute-budget tradeoff for free-tier Colab, not an oversight — restoring the full
70k/40k point counts is the first thing to try if reproducing this with more compute
headroom.

### Network architecture

Matches the original exactly:

| Network | Original | This repo |
|---|---|---|
| `dist_layers` | `[3, 20, 20, 20, 20, 5]` | same |
| `part_layers` | `[3, 20, 20, 20, 20, 5]` | same |
| `uv_layers` (general solution) | `[3] + 8×[70] + [5]` | same |

### Input normalization — deliberate addition

The original network consumes raw `(x, y, t)` coordinates (a normalization line
exists in `neural_net()` but is commented out and unused in the released script).
This repo's `net_uv` normalizes inputs before the general-solution network:

```python
x_norm = (x - 0.25) / 0.25
y_norm = (y - 0.25) / 0.25
t_norm = (t - 5.0) / 5.0
```

Added because unnormalized `t ∈ [0, 10]` alongside `x, y ∈ [0, 0.5]` was suspected to
bias early gradients toward the time dimension. Not present in the original, and a
deliberate deviation — flagged here because it means initialization dynamics aren't
directly comparable between the two implementations even at identical architecture
and point counts.

### Optimizer schedule

Original: the released script's `__main__` block calls `train_bfgs()` directly for
the physics network (pure L-BFGS-B, `maxiter=70000`) — no Adam stage for this network.
This repo: a short Adam warmup (1,000 epochs, lr=1e-3) precedes L-BFGS in Stage 1,
following the common PINN heuristic of using Adam to find a good basin before L-BFGS
refines it. The dist/particular network pretraining stages do match the original's
`train_bfgs_dist()` / `train_bfgs_part()` (pure L-BFGS-B, `maxiter=20000`).

### Summary

| Difference | Type | Status |
|---|---|---|
| Two-stage hole-loss weighting (500 → 10) vs. fixed 10 throughout | Deliberate training strategy | Documented; Stage 2 matches original's fixed weighting |
| Collocation density (~50% of original) | Compute tradeoff | Open — documented, not resolved |
| Input normalization | Deliberate addition | Kept — differs from original by design |
| Adam warmup before L-BFGS (Stage 1 only) | Deliberate addition | Kept — differs from original by design |
| Architecture, non-interior point-set sizes | — | Matches original |

## Citation

If you use this code, please cite the original paper:

> Rao, C., Sun, H., & Liu, Y. (2021). Physics-informed deep learning for
> computational elastodynamics without labeled data. *Journal of Engineering
> Mechanics*. [arXiv:2006.08472](https://arxiv.org/abs/2006.08472)

This repository is an independent reimplementation and is not affiliated with the
original authors.

## License

MIT — see [LICENSE](LICENSE). `reference/train.py` is the original authors' code;
check [Raocp/PINN-elastodynamics](https://github.com/Raocp/PINN-elastodynamics) for
its license before redistributing that file independently of this repo.
