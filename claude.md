# Celestine — CLAUDE.md

## Project
EHD (electrohydrodynamic) ion propulsion AI. Air thruster + future vacuum Xe thruster.
Location: `C:\Code\Plasma\` · Flask + Groq (GPT-OSS-120B) + OpenRouter (DeepSeek-R1)

---

## Architecture

```
User → GPT-120B (interface)
     → Gemma-31B (dispatcher, future)
     → pipeline.py (controller)
     → PoissonChargeCoupler (physics engine)
     → physics_validation.py (gate)
     → ehd_force.py (thrust)
     → DeepSeek-R1 (reasoning)
     → GPT-120B (response)
```

**Rule:** Modules = pure physics (Option A). pipeline.py = orchestrator only, no physics. physics_validation.py = sole decision authority (Option B).

---

## Core Chain (the only one that runs)

```
PoissonChargeCoupler.solve()
  → E(r,z), ρ(r,z), I
  → pipeline.run_and_validate()
  → ehd_force.body_force_integral()
  → physics_validation.validate_system()
```

**Entry point:** `pipeline.run_and_validate(voltage_V, gap_m, tip_radius_m)`

---

## File Reference

### 🟢 Core (must all work together)

| File | Role | Key function |
|------|------|-------------|
| `poisson_charge_coupling.py` | Solver: E↔ρ coupled loop | `PoissonChargeCoupler.solve()` |
| `pipeline.py` | Orchestrator | `run_and_validate()`, `run_ehd_case()` |
| `physics_validation.py` | Validation + decisions | `validate_system()`, `breakdown_gate()` |
| `ehd_force.py` | Module 5: thrust | `body_force_integral()`, `estimate_thrust()` |

### 🟡 Support (optional, not in main loop)

| File | Role |
|------|------|
| `electrostatics.py` | Analytic E-field estimates, initial guesses |
| `breakdown_stability.py` | Breakdown physics, `analyse()` |
| `emitter_array.py` | Geometry design (pre-simulation) |
| `air_thruster_deepdive.py` | Empirical EHD models, benchmarking |
| `propellants_db.py` | Constants only |

### 🔴 Do NOT connect to pipeline

| File | Reason |
|------|--------|
| `corona_physics.py` | Duplicate ionization model (solver handles internally) |
| `charge_transport.py` | Duplicate PDE solver |
| `circuit_field.py` | Needs flux_total (not yet computed) |

---

## Solver: PoissonChargeCoupler

Self-consistent loop (Morrow & Lowke 1997):
```
ρ⁰ = 0
loop:
  1. Poisson → E = -∇V
  2. S = α_net(E)·ρ·μ·|E|
  3. Transport: ∇·(ρμE - D∇ρ) = S - R → ρ_new
  4. ρ = ω·ρ_new + (1-ω)·ρ  (SOR, ω=0.3)
  5. I = ∫ρ·μ·E·dA at collector
  6. |ΔI/I| < tol → converged
```

**Injection model:** Boundary-localized at emitter (i=0,1; j_tip±1).
Weights: `α(E_tip_analytic) · exp(-(r/r_scale)²) · (r + 0.5·dr)`, normalized.
`r_scale = max(5·tip_radius, dr)` — physics-based, not grid-dependent.

**Default grid:** Nr=60, Nz=90. Increase for finer tip resolution.

**Output keys:**
```python
{
  "field":   {E_peak_MV_m, E_avg_MV_m, E_tip_uncoupled_MV_m, space_charge_reduction_pct},
  "charge":  {rho_max_nC_m3, Q_total_nC, rho_injection_nC_m3},
  "current": {I_collector_uA, I_collector_mA},
  "grids":   {rho, Er, Ez, r, z},   # 2D arrays for force integral
  "solver":  {converged, outer_iterations, grid},
}
```

---

## Pipeline Output Contract

`run_and_validate()` returns:
```python
{
  "physics": {
    "E":           {E_peak_MV_m, E_avg_MV_m, E_tip_coupled_MV_m, ...},
    "rho":         {rho_max_nC_m3, Q_total_nC},
    "current":     {I_collector_uA, note},
    "thrust":      {F_z_N, F_mN, anisotropy_ratio, convergence_warning, ...},
    "performance": {power_W, thrust_per_watt_mN_W},
  },
  "gate":    validate_system() output,
  "summary": {E_peak_MV_m, ratio, regime, current_uA, thrust_mN, power_W, valid, ...},
  "proceed": bool,   # gate for DeepSeek
}
```

---

## Module 5: ehd_force.py

Two functions:
- `body_force_integral(rho, Er, Ez, r, z)` — `F = ∫ρ·E dV`, cell-averaged, non-uniform grid safe
- `estimate_thrust(I_A, gap_m, pressure_Pa)` — fallback `F ≈ I·d/μ`

Pipeline uses `body_force_integral` when grids available, else `estimate_thrust`.

**Diagnostics:**
- `rho_cells < 10` → `convergence_warning=True`
- `anisotropy_ratio > 2` AND `rho_cells < 20` → `anisotropy_suspicious=True`
- `anisotropy_ratio > 1` near needle tip is **physical** (radial field divergence), not a bug

---

## Validation: physics_validation.py

**Single authority for all decisions.**

```python
RATIO_THRESHOLDS = {"no_discharge": 3.0, "corona_max": 10.0, "streamer_max": 15.0}
ratio_to_regime(ratio)       # sole regime classifier
validate_system(results)     # global gate
breakdown_gate(bd)           # ACCEPTED/REJECTED from raw _ratio
check_dependencies(results)  # DEP_CURRENT_WITHOUT_RHO etc.
```

Modules return `_` fields only. Validation reads them. Never the reverse.

---

## Physics Constants

```python
E_BREAK_STP = 3.0e6   # V/m, air at STP
MU_ION_STP  = 2e-4    # m²/V·s, positive ions
PEEK_A      = 3.1e6   # V/m
PEEK_B      = 0.030   # m^0.5
```

Regime thresholds (Raizer 1991) — defined ONLY in `physics_validation.RATIO_THRESHOLDS`.

---

## Key Physics

**E_tip** (Peek/Sigmond): `E_tip = V / (r · ln(2d/r))`

**Thrust approximation**: `F ≈ I·d/μ` (order-of-magnitude, ideal momentum transfer)

**True thrust**: `F = ∫ρ·E dV` (body_force_integral, cylindrical: `dV = 2πr·dr·dz`)

**Onset** (Peek 1929): `E_onset = A·δ·(1 + B/√(r·δ))`

**Stability**: circuit stable when `1/R_ballast > dI_corona/dV`

---

## Build Status

| Module | Status |
|--------|--------|
| 1. Electrostatics | ✅ |
| 1+2. Poisson coupling | ✅ (solver core) |
| 2. Charge transport | ✅ (inside solver) |
| 3. Corona physics | ✅ (inside solver) |
| 4. Circuit-field | ⚠️ needs flux_total |
| 5. EHD Force | ✅ |
| 6. Fluid / N-S | 🔲 |
| 7. Breakdown | ✅ |
| 8–12. Thermal/Material/Flow/Plasma/Scaling | 🔲 |
| pipeline.py | ✅ |
| app.py connected to pipeline | 🔲 next |

---

## Environment
Kerala, India · 80% RH · 32°C · 101 kPa · PETG housing
