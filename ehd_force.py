"""
ehd_force.py — Celestine Module 5: EHD Force
Option A: pure physics. No decisions, no validation.

F = ∫ρ·E dV  (cylindrical: F_z = ∫ρ·Ez·2πr·dr·dz)
This is the direct body force from solver fields.
"""
import math

MU_ION_STP = 2e-4  # m²/V·s, positive ions in air at STP


def estimate_thrust(I_A: float,
                    gap_m: float,
                    pressure_Pa: float = 101325) -> dict:
    """
    F ≈ I·d/μ — fallback approximation when grids unavailable.
    ORDER-OF-MAGNITUDE ONLY. Use body_force_integral() when possible.
    """
    mu = MU_ION_STP * (101325 / pressure_Pa)
    F_N = (I_A * gap_m) / mu if mu > 0 else 0.0
    note = "Thrust unreliable — current under-resolved" if I_A == 0.0 else None
    return {
        "_F_canonical":  F_N,
        "F_N":           F_N,
        "F_mN":          F_N * 1000,
        "mu_ion":        mu,
        "model":         "EHD_momentum_transfer_estimate",
        "assumption":    "ideal momentum transfer — ignores losses and fluid coupling",
        "mu_note":       "pressure-only scaling (temperature and gas composition ignored)",
        "upgrade_path":  "use body_force_integral() when rho_field available",
        "note":          note,
    }


def body_force_integral(rho: list, Er: list, Ez: list,
                        r: list, z: list) -> dict:
    """
    F_z = ∫ ρ·Ez · 2πr · dr · dz   (axial thrust, cylindrical coords)
    F_r = ∫ ρ·Er · 2πr · dr · dz   (radial force component)

    Cell-averaged integration: each cell uses the average of 4 surrounding
    nodes. Cell volume: dV = 2π·r_center·dr_i·dz_j (non-uniform grid safe).

    NOTE on anisotropy_ratio (formerly symmetry_error):
    For needle-to-ring geometry, Er >> Ez near the tip is PHYSICAL (field lines
    diverge radially from tip). anisotropy_ratio > 1 does NOT indicate a bug —
    it indicates the charge is near the tip where radial fields dominate.
    True symmetry error (azimuthal asymmetry) is not detectable from 2D fields.
    """
    Nr = len(rho)
    Nz = len(rho[0]) if Nr > 0 else 0
    if Nr < 2 or Nz < 2:
        return {"_F_canonical": None, "F_N": None, "F_mN": None,
                "model": "EHD_body_force_integral", "status": "grid too small"}

    rho_max = max(abs(rho[i][j]) for i in range(Nr) for j in range(Nz))
    rho_threshold = max(rho_max * 1e-6, 1e-25)
    rho_cells = sum(1 for i in range(Nr) for j in range(Nz)
                    if abs(rho[i][j]) > rho_threshold)

    F_z = 0.0
    F_r = 0.0

    for i in range(1, Nr - 1):
        r_center = 0.5 * (r[i] + r[i - 1])
        # Non-uniform grid safe: use local cell spacing
        dr_i = r[i] - r[i - 1]

        # j loop includes 0 to capture first physical slab near electrode
        for j in range(0, Nz):
            j0 = max(j - 1, 0)
            j1 = min(j, Nz - 1)
            if j1 > j0:
                dz_j = z[j1] - z[j0]
            else:
                dz_j = z[1] - z[0]
            dV = 2.0 * math.pi * r_center * dr_i * dz_j

            rho_avg = 0.25 * (rho[i][j1]  + rho[i-1][j1] +
                               rho[i][j0]  + rho[i-1][j0])
            Ez_avg  = 0.25 * (Ez[i][j1]   + Ez[i-1][j1]  +
                               Ez[i][j0]   + Ez[i-1][j0])
            Er_avg  = 0.25 * (Er[i][j1]   + Er[i-1][j1]  +
                               Er[i][j0]   + Er[i-1][j0])

            if abs(rho_avg) < rho_threshold:
                continue

            F_z += rho_avg * Ez_avg * dV
            F_r += rho_avg * Er_avg * dV

    # anisotropy_ratio: |F_r/F_z| — expected > 1 near needle tip (physical)
    # Only meaningful as convergence check across grid refinements
    if abs(F_z) > 1e-20:
        anisotropy_ratio = abs(F_r / F_z)
    else:
        anisotropy_ratio = None

    # Convergence indicator: too few rho cells = result unreliable
    # Physical sanity check: F ≈ I·d/μ is order-of-magnitude bound
    # F_integral / F_est should be 0.5–2 when charge is well-resolved
    # Cannot compute here without I — caller should compare via pipeline summary

    # Global anisotropy: radial components should CANCEL by azimuthal symmetry
    # F_r_total >> F_z_total globally is suspicious even if locally expected near tip
    anisotropy_suspicious = (
        anisotropy_ratio is not None and
        anisotropy_ratio > 2.0 and
        rho_cells < 20
    )
    convergence_warning = rho_cells < 10 or anisotropy_suspicious

    return {
        "_F_canonical":        F_z,
        "F_z_N":               F_z,
        "F_r_N":               F_r,
        "F_N":                 F_z,
        "F_mN":                F_z * 1000,
        "anisotropy_ratio":    anisotropy_ratio,
        "anisotropy_suspicious": anisotropy_suspicious,
        "direction":           "toward_collector" if F_z < 0 else "away_from_collector",
        "rho_cells":           rho_cells,
        "convergence_warning": convergence_warning,
        "model":               "EHD_body_force_integral",
        "formula":             "F_z = ∫ρ·Ez·2πr·dr·dz (cell-averaged, non-uniform safe)",
        "assumption":          "no fluid coupling — body force only, no Navier-Stokes",
        "note": (
            "anisotropy_ratio > 1 locally near tip is physical (radial field divergence). "
            "anisotropy_ratio > 2 globally is suspicious — radial components should cancel "
            "by azimuthal symmetry; likely cause: rho under-resolved (rho_cells < 10). "
            "Fix: increase Nr/Nz until convergence_warning=False."
        ),
    }
