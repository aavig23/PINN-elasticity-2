# PINN_ELASTICITY_SPEC.md

Specification for a Physics-Informed Neural Network (PINN) solving **2D plane-stress elasticity** of a clamped plate under axial tension: first a static solve, then the elastodynamic problem.

This file is the single source of truth for the project. The project starts from scratch: there is no existing code to reuse. Read this file fully before planning. Where it says **CONFIRM WITH USER**, stop and ask before implementing that part.

---

## 0. How to work on this project

1. **Plan in plan mode first.** Before writing code, produce a plan and present it to the user. The plan should cover: repository and code structure, nondimensionalization, network architecture (with justification), BC/IC enforcement strategy, collocation sampling, loss weighting, optimizer schedule, configuration and experiment management, and what each run saves.
2. **The code structure is yours to design.** This spec defines the physics, the numerical requirements, and the workflow constraints. It does not prescribe files, modules, or function names.
3. **Resolve the open items in Section 8** with the user during planning. Do not silently pick values for them.
4. **Scope:** deliver the PINN code and the experiment infrastructure. **Validation is done by the user** (see Section 6). Do not implement comparisons against reference or FEM solutions.

---

## 1. Problem definition

### 1.1 Geometry
- 2D plate in the **x–z plane**.
- Length: `L = 1.0 m` (x ∈ [0, L])
- Height: `H = 0.5 m` (z ∈ [0, H])
- Aspect ratio H/L = 0.5.

### 1.2 Material (linear isotropic elastic, steel)
| Quantity | Symbol | Value |
|---|---|---|
| Young's modulus | E | 200e9 Pa |
| Poisson's ratio | ν | 0.30 |
| Density | ρ | 7800 kg/m³ |
| Lamé μ | μ = E / (2(1+ν)) | ≈ 76.92 GPa |
| 3D Lamé λ | λ = Eν / ((1+ν)(1−2ν)) | ≈ 115.38 GPa (NOT used directly) |
| Plane-stress λ* | λ* = 2λμ / (λ + 2μ) = Eν / (1−ν²) | ≈ 65.93 GPa |

### 1.3 Plane-stress assumption (FIXED)
The formulation is **plane stress**: the plate is thin in the out-of-plane (y) direction, so σ_yy = σ_xy = σ_yz = 0.

- Use the **effective Lamé constant λ*** in place of λ in all in-plane constitutive equations.
- Equivalently: λ* + 2μ = E / (1−ν²) ≈ 219.78 GPa.
- Optional post-processing: out-of-plane strain ε_yy = −(ν / (1−ν)) (ε_xx + ε_zz).

### 1.4 Unknowns
- `u`: displacement in x
- `w`: displacement in z

Displacement-only formulation: the network outputs (u, w); strains and stresses are derived by automatic differentiation.

### 1.5 Two stages (DECIDED)
The user's original time setting was `T = 1.0 s`. That is not feasible for a PINN: stress waves cross the steel plate in about 0.2 ms, so 1 s spans ~5000 wave transits and ~1266 undamped vibration cycles. It is replaced by two stages, run in this order:

- **Stage 1: static solve.** A single solve at the full load. Inputs (x, z) only, no inertia, no initial conditions, no ramp. This is the settled shape the plate would reach under the full load. It is the fastest way to get the whole pipeline working.
- **Stage 2: dynamic solve.** Full elastodynamics over a short window from t = 0 to **T̂ = 4** in nondimensional time (≈ 0.79 ms; see Section 3.2). This covers the load ramp, wave travel to the clamp and back twice, and about one full fundamental vibration period of the plate. T̂ is a config parameter and may be extended later.

A series of static solves along the ramp is **not** needed: the problem is linear, so the static field at load fraction f is exactly f × the full-load static field.

Both stages should share as much code as sensible (constitutive law, scaling, sampling, logging, outputs). The stage is selected by configuration.

---

## 2. Governing equations (dimensional form)

### 2.1 Kinematics (small strain)
```
eps_xx   = ∂u/∂x
eps_zz   = ∂w/∂z
gamma_xz = ∂u/∂z + ∂w/∂x        (engineering shear strain)
eps_xz   = 0.5 * gamma_xz       (tensorial shear strain)
```

### 2.2 Constitutive law (plane stress, isotropic)
```
lam_star = E * nu / (1 - nu**2)          # = 2*lam*mu / (lam + 2*mu)

sigma_xx = (lam_star + 2μ) ∂u/∂x + lam_star ∂w/∂z
sigma_zz = lam_star ∂u/∂x + (lam_star + 2μ) ∂w/∂z
sigma_xz = μ (∂u/∂z + ∂w/∂x)
```
Equivalent engineering form (useful as a cross-check):
```
sigma_xx = E/(1-ν²) (eps_xx + ν eps_zz)
sigma_zz = E/(1-ν²) (eps_zz + ν eps_xx)
sigma_xz = E/(2(1+ν)) gamma_xz
```

### 2.3 Equilibrium / equations of motion (no body force)
Stage 2 (dynamic):
```
ρ ∂²u/∂t² = ∂sigma_xx/∂x + ∂sigma_xz/∂z
ρ ∂²w/∂t² = ∂sigma_xz/∂x + ∂sigma_zz/∂z
```
Stage 1 (static): the same equations with the inertia terms removed (left-hand sides = 0).

Residuals (dynamic; drop the ρ terms for static):
```
R_u = ρ u_tt − ∂sigma_xx/∂x − ∂sigma_xz/∂z
R_w = ρ w_tt − ∂sigma_xz/∂x − ∂sigma_zz/∂z
```

### 2.4 Boundary conditions (both stages)
| Boundary | Location | Condition |
|---|---|---|
| Left | x = 0 | Clamped: u = 0, w = 0 |
| Bottom | z = 0 | Traction-free: sigma_zz = 0, sigma_xz = 0 |
| Top | z = H | Traction-free: sigma_zz = 0, sigma_xz = 0 |
| Right | x = L | Axial tension: sigma_xx = q, sigma_xz = 0 |

Stage 1: q = q0 (full load). Stage 2: q = q0 · f(t) (Section 4).

### 2.5 Right-edge load
- **Direction:** axial, along +x (tension).
- **Distribution:** uniform traction over the whole right edge, magnitude `q0 = P0 / H`.
- `P0 = 1000.0`, so `q0 = 1000 / 0.5 = 2.0e3 Pa`.
- **Interpretation (DECIDED):** P0 is force per unit out-of-plane thickness (N/m), so P0/H is a traction in Pa. For plane stress with a prescribed edge traction, the in-plane solution does not depend on the plate thickness.

### 2.6 Initial conditions (Stage 2 only), plate at rest
```
u(x, z, 0)   = 0
w(x, z, 0)   = 0
u_t(x, z, 0) = 0
w_t(x, z, 0) = 0
```

---

## 3. Nondimensionalization (REQUIRED, both stages)

### 3.1 Why
In SI units, stress-derivative terms are ~1e11 × (displacement gradients) while displacements are ~1e-8 m. Unscaled losses differ by many orders of magnitude and training will stall. **All training must be done in nondimensional variables.** Results are converted back to SI only for saved outputs and figures.

### 3.2 Recommended scaling
```
c    = sqrt(E / ρ)          ≈ 5064 m/s   (reference wave speed)
t_c  = L / c                ≈ 1.97e-4 s  (time for a wave to cross the plate once)
u_c  = P0 * L / (E * H)     = 1.0e-8 m
σ_c  = E * u_c / L          = P0 / H = 2.0e3 Pa

x̂ = x / L,   ẑ = z / L,   t̂ = t / t_c
û = u / u_c, ŵ = w / u_c
σ̂ = σ / σ_c
λ̂* = λ* / E ≈ 0.3297,  μ̂ = μ / E ≈ 0.3846,  λ̂* + 2μ̂ = 1/(1−ν²) ≈ 1.0989
```
t̂ counts "plate crossings": t̂ = 4 corresponds to t ≈ 4 × 0.197 ms ≈ 0.79 ms.

**Definition of T̂ (T-hat):** T̂ is the **end time of the Stage 2 simulation window, in nondimensional time**: the dynamic problem is solved for t̂ ∈ [0, T̂]. It relates to physical time by `T̂ = T / t_c`, i.e. `T = T̂ · t_c`. With the default **T̂ = 4**, the physical window is T ≈ 4 × 1.97e-4 s ≈ **0.79 ms**, which replaces the user's original T = 1.0 s (see Section 1.5). The config should store T̂ and report the corresponding T in ms in the run metadata and figure labels.

Using ẑ = z / L keeps the equations isotropic; the domain becomes x̂ ∈ [0, 1], ẑ ∈ [0, 0.5].

With this choice, the nondimensional equations have unit coefficients:
```
σ̂_xx = (λ̂* + 2μ̂) û_x̂ + λ̂* ŵ_ẑ
σ̂_zz = λ̂* û_x̂ + (λ̂* + 2μ̂) ŵ_ẑ
σ̂_xz = μ̂ (û_ẑ + ŵ_x̂)

Dynamic: û_t̂t̂ = ∂σ̂_xx/∂x̂ + ∂σ̂_xz/∂ẑ
         ŵ_t̂t̂ = ∂σ̂_xz/∂x̂ + ∂σ̂_zz/∂ẑ
Static:  0     = (same right-hand sides)

Right edge (x̂ = 1): σ̂_xx = f(t̂) (dynamic) or 1 (static), σ̂_xz = 0
```
Claude Code may propose an alternative scaling, but must show that all loss terms are O(1) and explain the choice.

---

## 4. Load time history (Stage 2, DECIDED)

Smooth ramp-then-hold load (a "soft step"):
```
f(t̂) = 0.5 * (1 − cos(π t̂ / t̂_r))   for t̂ < t̂_r
f(t̂) = 1                             for t̂ ≥ t̂_r
```
- **Default ramp time t̂_r = 1** (≈ 0.2 ms), configurable.
- f(0) = 0, so the load is compatible with the at-rest initial conditions.
- Smooth (C¹), avoiding the sharp wavefront a sudden step would create.

---

## 5. Numerical requirements

### 5.1 Stack
- Python, **PyTorch**.
- All derivatives via **automatic differentiation** (`torch.autograd.grad` with `create_graph=True` wherever a derivative is differentiated again). No finite differences.
- Automatic device selection (GPU on Colab, CPU fallback) and fixed random seeds for reproducibility.
- Consider float64 if float32 shows precision problems in second derivatives; justify the choice.

### 5.2 Dependencies across Python versions
- The user runs **Python 3.13.5 locally**. Colab's Python version is outside our control and may differ. Everything must install and run on both.
- **Do not install or pin a specific torch build on Colab.** Use Colab's preinstalled torch, which is matched to its CUDA drivers; reinstalling it can break GPU support. Specify torch only as a minimum version (e.g. `torch>=2.2`).
- Keep other dependencies minimal (e.g. numpy, matplotlib, pyyaml) and specify them with lower bounds that are known to install on Python 3.13 and on current Colab, rather than exact pins.
- Record the exact versions actually used (Python, torch, CUDA, numpy, etc.) in each run's metadata, so every run remains reproducible.

### 5.3 Network
- Stage 1 inputs: (x̂, ẑ). Stage 2 inputs: (x̂, ẑ, t̂). Outputs: (û, ŵ).
- **Architecture is for Claude Code to propose** during planning (depth, width, activation, any input normalization or Fourier features), with a brief justification. Activations must be at least twice differentiable (e.g. tanh; not ReLU).

### 5.4 Enforcing BCs and ICs
Present both options in the plan and recommend one. Hard enforcement is preferred where practical.
- **Soft:** penalty loss terms for all BCs and ICs.
- **Hard:** build the clamp (and, in Stage 2, the ICs) into the output. Stage 1 example: `û = x̂ · N_u(x̂, ẑ)`. Stage 2 example: `û = x̂ · t̂² · N_u(x̂, ẑ, t̂)`, which satisfies u = w = 0 at x = 0 and u = w = u_t = w_t = 0 at t = 0 exactly. Traction BCs remain soft.

### 5.5 Losses
```
L_total = w_pde  * (mean R̂_u² + mean R̂_w²)
        + w_free * (mean σ̂_zz² + mean σ̂_xz²)          on top and bottom
        + w_load * (mean (σ̂_xx − q̂)² + mean σ̂_xz²)    on the right edge (q̂ = f(t̂) or 1)
        + w_clamp * (mean û² + mean ŵ²)               on the left edge (if soft)
        + w_ic   * (û², ŵ², û_t̂², ŵ_t̂² terms)         at t̂ = 0 (Stage 2, if soft)
```
Loss weighting: propose a strategy (fixed weights, or adaptive such as gradient-norm balancing or learning-rate annealing) and log every term separately.

### 5.6 Sampling and training
- Sample interior collocation points over the domain ((x̂, ẑ) or (x̂, ẑ, t̂)), respecting the 2:1 aspect ratio. Use enough boundary points on each edge.
- Consider extra points near the clamped corners (x = 0, z = 0 and z = H), where stresses concentrate, and (Stage 2) along the moving wavefront.
- Propose an optimizer schedule (e.g. Adam followed by L-BFGS) and a resampling strategy if useful.
- Stage 2: if training over the full time window is hard, consider time-marching or causal weighting, and explain the choice.

### 5.7 Known pitfalls
- Unscaled losses (Section 3): never train in SI units.
- Using the 3D λ instead of λ*: this silently gives plane strain. Always use λ*.
- Missing `create_graph=True` on derivatives that are differentiated again breaks second derivatives.
- Corner singularities at the clamped corners: high local residuals are expected; do not let them dominate the loss.
- Load-IC incompatibility: a load with f(0) ≠ 0 conflicts with the at-rest initial state.

---

## 6. Scope: code only, user does validation

- Claude Code delivers the PINN code and the experiment infrastructure described in Section 7.
- **Do not implement validation** against analytical, FEM, or other reference solutions, and do not draw conclusions about accuracy. The user performs validation separately.
- To make that possible, each run exports the predicted fields in SI units (u, w, σ_xx, σ_zz, σ_xz) on a regular (x, z) grid, in a portable format, with the grid and units documented (Section 7.4).

---

## 7. Development and experiment workflow

### 7.1 Overall loop
```
Local VS Code + Claude Code
  ↓  Specification / Plan
  ↓  Implementation
  ↓  Push to GitHub
  ↓  Google Colab: clone/pull repo, run experiment
  ↓  Save results + figures + logs + checkpoints
  ↓  Push results to GitHub (from Colab)
  ↓  Claude Code reads/analyzes results (after a local pull)
  ↓  Modify code / configuration
  ↓  New experiment
  ↺  Repeat
```

### 7.2 General requirements
- **Code is written locally, but training runs on Google Colab.** Do not assume a local GPU or long local runs. Local execution is only for quick checks.
- **Colab-ready:** provide a simple, documented way to run an experiment on Colab: clone or pull the repo, install dependencies (Section 5.2), launch a run with a chosen configuration, and push the results (Section 7.5).
- **Smoke-test mode:** a fast setting (few points, few iterations, CPU-friendly) to check locally, for both stages, that the code runs end to end before pushing.
- **Configuration-driven experiments:** all physical, numerical, and training parameters (stage, T̂, t̂_r, network size, loss weights, optimizer settings, snapshot times, export grid, figure settings) live in config files, not hard-coded. A new experiment should normally need only a new or edited config.
- **Colab session limits:** Colab sessions can disconnect. Save checkpoints periodically and support resuming a run from its latest checkpoint.
- **Analysis step:** when asked to analyze a run, Claude Code reads the run folder(s), summarizes training behaviour (e.g. which loss terms stall or dominate), and proposes concrete code or config changes for the next experiment. This is about training behaviour, not validation of accuracy.

### 7.3 Repository layout for results (DECIDED)
- All run folders live under a dedicated, **committed** directory: **`runs/`**.
- The existing `.gitignore` excludes `results/` and `outputs/`. Keep those ignored and use them only as local scratch space. **Never put run folders in `results/` or `outputs/`**, or nothing will reach GitHub. Make sure `.gitignore` does not exclude `runs/` or anything inside it that must be committed.
- Each run gets its own uniquely named folder (e.g. timestamp + stage + short config name), so runs from different sessions never collide or cause merge conflicts.

### 7.4 Contents of each run folder
- A copy of the exact config used.
- Run metadata: git commit hash, start/end time, device/GPU type, random seed, library and Python versions.
- Training logs: every loss term separately, at a fixed interval, in a machine-readable format (e.g. CSV or JSON).
- A short machine-readable summary: final loss values, training time, iterations, stage.
- Figures (PNG, default 150 dpi):
  - loss curves (each term, log scale);
  - contour plots of u, w, σ_xx, σ_zz, σ_xz in SI units: Stage 1 a single set; Stage 2 one set per snapshot time;
  - PDE residual maps: Stage 1 a single map; Stage 2 at the **same snapshot times** as the field plots;
  - Stage 2 only: tip displacement history u(L, H/2, t), with time in ms.
- Exported field data for user validation:
  - default grid **101 × 51** in (x, z) (1 cm spacing), configurable;
  - Stage 1: one field set; Stage 2: default **8 snapshot times**, evenly spaced over [0, T̂], configurable;
  - compressed **float32 `.npz`**, including the grid coordinates and times (both nondimensional and SI), with units documented in the run folder.
- **Checkpoints (committed to GitHub, DECIDED):** keep only the **latest** and the **final** checkpoint per run (model and optimizer state), not every intermediate one. PINN checkpoints are expected to be well under 1 MB. If a checkpoint would exceed ~50 MB, warn and do not commit it.
- **Size target:** each run folder, checkpoints included, should stay under about **20 MB** with default settings. Log a warning if a run exceeds this.
- Stage 1 has no time dimension, so time-based outputs (snapshots, tip history, initial-condition losses) are skipped for it.

### 7.5 Pushing results from Colab (DECIDED)
- Use a **fine-grained GitHub personal access token** created by the user:
  - limited to **this one repository** only;
  - permission: **Contents: read and write**, nothing else;
  - a short expiry date.
- Store the token in **Colab Secrets** and read it at runtime (e.g. via `google.colab.userdata`). The token must **never** be hard-coded, typed into a notebook cell, printed, logged, written into any file, saved in notebook outputs, or committed. Do not leave it embedded in a persisted git remote URL or config.
- The push step should: pull the latest changes first, add only the new run folder, commit with a clear message (run name + stage), and push. If the push fails, keep the run folder intact and report the error clearly.
- Document the one-time token setup (creating the token, adding it to Colab Secrets) in the README.

---

## 8. Open items to resolve with the user during planning

1. Network architecture, and the exact BC/IC enforcement approach for each stage (Sections 5.3–5.4).

All other parameters have defaults in this spec. Claude Code may suggest changes to any of them in the plan, but must not change them silently.
