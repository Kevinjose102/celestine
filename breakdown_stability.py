"""
breakdown_stability.py — Celestine Physics Module 7
=====================================================
Breakdown & Stability

Tier 1 — Independent Base Module (depends only on E-field geometry).
Pure physics module (Option A).
No validation, no decisions, no gating.
All decisions handled in physics_validation.

Contains all 10 required components:
    1. Electric field limits (E_avg, E_tip, E_break ratios)
    2. Corona onset — Peek's law with air density correction
    3. Transition detection (corona → streamer → arc)
    4. Stability criterion with circuit (load line vs plasma slope)
    5. Breakdown margin quantification
    6. Space-charge stability check (ΔV_sc/V)
    7. Localised field enhancement / geometry risk (β = E_tip/E_avg)
    8. Flat physics outputs for validation layer
    9. No decision outputs
    10. Environmental corrections (pressure, temperature, humidity)

References:
    Peek (1929)        — corona onset field for wires and points
    Raizer (1991)      — Gas Discharge Physics, Ch. 3
    Sigmond (1982)     — corona I-V, J. Appl. Phys. 53, 891
    Townsend (1915)    — ionisation coefficients
    Meek (1940)        — streamer criterion
    Goldman (1978)     — corona-to-arc transitions in air
"""

import math
import json
# ── Constants ─────────────────────────────────────────────────────────────────
E_BREAK_STP = 3.0e6       # V/m  air breakdown at STP
PEEK_A      = 3.1e6       # V/m
PEEK_B      = 0.030       # m^0.5
P0          = 101325.0    # Pa  reference pressure
T0          = 293.15      # K   reference temperature (20°C)

# Townsend I-V geometry factor exponent for stability slope
EPSILON_0   = 8.854e-12
MU_POS      = 2.0e-4      # m²/V/s

# Field ratio thresholds — E_tip / E_break  (Raizer 1991, Goldman 1978)
# This is the physically correct primary safety indicator.
# E_avg/E_break is NOT sufficient — tip dominance drives breakdown.
#
# Ratio E_tip/E_break
# < 3                 | No discharge (below onset for any sharp geometry)
# 3 – 8              | Stable corona
# 8 – 12             | Transition warning band
# > 12               | Spark / arc risk
# Formula used: E_tip = V/(r·ln(2d/r)) — consistent with Peek law
# Alternative 2V/(r·ln(4d/r)) gives 1.8× higher E_tip and wrong V_onset — not used

# Current-based transition thresholds (per needle)
I_CORONA_MAX_A    = 1e-3   # 1 mA — above this: streamer risk
I_STREAMER_MAX_A  = 10e-3  # 10 mA — above this: arc risk


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENTAL CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def air_density_factor(pressure_Pa: float = P0,
                        temperature_K: float = T0,
                        humidity_pct: float = 0.0) -> float:
    """
    Air density correction factor δ (delta).

    δ = (p / p₀) · (T₀ / T)

    E_break scales with δ: E_break(p,T) = E_break_STP · δ
    Peek onset also scales: E_onset(p,T) = E_onset_STP · δ

    Humidity reduces E_break by reducing effective ionisation energy:
    dE_break/dRH ≈ -0.05% per %RH (empirical, Qiu 2014)

    Parameters:
        pressure_Pa   : ambient pressure [Pa]
        temperature_K : ambient temperature [K]
        humidity_pct  : relative humidity [%]

    Returns:
        δ (dimensionless, = 1.0 at STP)
    """
    delta = (pressure_Pa / P0) * (T0 / temperature_K)
    # Humidity correction: each 10%RH above 50% reduces E_break by ~0.5%
    rh_correction = 1.0 - max(0.0, humidity_pct - 50.0) * 0.0005
    return delta * rh_correction


def corrected_breakdown_field(pressure_Pa: float = P0,
                               temperature_K: float = T0,
                               humidity_pct: float = 0.0) -> float:
    """E_break corrected for ambient conditions [V/m]."""
    return E_BREAK_STP * air_density_factor(pressure_Pa, temperature_K, humidity_pct)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ELECTRIC FIELD LIMITS
# ═══════════════════════════════════════════════════════════════════════════════

def field_limits(voltage_V: float,
                  tip_radius_m: float,
                  gap_m: float,
                  pressure_Pa: float = P0,
                  temperature_K: float = T0,
                  humidity_pct: float = 0.0) -> dict:
    """
    Compute E_avg, E_tip, and their ratios to E_break.

    ALL INPUTS MUST BE IN SI UNITS:
        voltage_V     : volts      [V]
        tip_radius_m  : metres     [m]  ← NOT mm. 0.1mm = 1e-4 m
        gap_m         : metres     [m]  ← NOT mm. 23mm  = 2.3e-2 m

    Formula used:  E_tip = V / (r · ln(2d/r))
    This is the hyperboloid needle approximation (Peek 1929, Sigmond 1982).
    It is SELF-CONSISTENT with Peek's onset law — both use ln(2d/r).
    For r=1e-4m, d=0.023m, V=20kV: E_tip = 32.6 MV/m, V_onset = 7603V ✓

    Alternative formula 2V/(r·ln(4d/r)) (Cooperman 1981) gives 58.5 MV/m
    for the same geometry but predicts V_onset = 4231V — inconsistent with
    measured data. We do NOT use it.

    E_avg = V / d                  (uniform field, lower bound)
    E_tip = V / (r · ln(2d/r))    (hyperboloid needle tip field)
    """
    # ── SI unit sanity checks ─────────────────────────────────────────────────
    if tip_radius_m > 0.01:
        raise ValueError(
            f"tip_radius_m={tip_radius_m} looks like mm not m. "
            f"Convert: tip_radius_mm / 1000. "
            f"Example: 0.1mm → 1e-4 m."
        )
    if gap_m > 0.5:
        raise ValueError(
            f"gap_m={gap_m} looks like mm not m. "
            f"Convert: gap_mm / 1000. "
            f"Example: 23mm → 0.023 m."
        )
    if voltage_V < 10:
        raise ValueError(
            f"voltage_V={voltage_V} looks like kV not V. "
            f"Convert: voltage_kV * 1000. "
            f"Example: 20kV → 20000 V."
        )

    E_break = corrected_breakdown_field(pressure_Pa, temperature_K, humidity_pct)
    ln_term = math.log(2 * gap_m / tip_radius_m) if (2*gap_m/tip_radius_m) > 1 else 1.0
    E_avg   = voltage_V / gap_m
    E_tip   = voltage_V / (tip_radius_m * ln_term)

    return {
        "E_avg_MV_m":         round(E_avg / 1e6, 4),
        "E_tip_MV_m":         round(E_tip / 1e6, 4),
        "E_break_MV_m":       round(E_break / 1e6, 4),
        "E_avg_over_E_break": round(E_avg / E_break, 4),
        "E_tip_over_E_break": round(E_tip / E_break, 4),
        "beta":               round(E_tip / E_avg, 2),   # field enhancement factor
        "_E_avg":             E_avg,   # internal floats for downstream use
        "_E_tip":             E_tip,
        "_E_break":           E_break,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CORONA ONSET — PEEK'S LAW
# ═══════════════════════════════════════════════════════════════════════════════

def peek_onset(tip_radius_m: float,
               gap_m: float,
               pressure_Pa: float = P0,
               temperature_K: float = T0,
               humidity_pct: float = 0.0) -> dict:
    """
    Peek's law for corona onset field and voltage.

    E_onset = A · δ · (1 + B / √(r · δ))

    V_onset = E_onset · r · ln(2d/r)

    Parameters:
        tip_radius_m : emitter tip radius [m]
        gap_m        : electrode gap [m]

    Returns:
        V_onset, E_onset, margin fields
    """
    delta   = air_density_factor(pressure_Pa, temperature_K, humidity_pct)
    r       = tip_radius_m
    d       = gap_m
    ln_term = math.log(2*d/r) if (2*d/r) > 1 else 1.0

    E_onset = PEEK_A * delta * (1 + PEEK_B / math.sqrt(r * delta))
    V_onset = E_onset * r * ln_term

    return {
        "E_onset_MV_m":  round(E_onset / 1e6, 4),
        "V_onset_V":     round(V_onset),
        "delta":         round(delta, 4),
        "_E_onset":      E_onset,
        "_V_onset":      V_onset,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRANSITION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def transition_detection(voltage_V: float,
                          tip_radius_m: float,
                          gap_m: float,
                          current_A: float = 0.0,
                          pressure_Pa: float = P0,
                          temperature_K: float = T0,
                          humidity_pct: float = 0.0) -> dict:
    """
    Three-indicator transition detection:
        (A) Field-based: E_tip / E_break ratio
        (B) Current-based: I vs threshold ranges
        (C) Growth instability: dI/dV steepness (Townsend slope)
    """
    fl     = field_limits(voltage_V, tip_radius_m, gap_m,
                          pressure_Pa, temperature_K, humidity_pct)
    pk     = peek_onset(tip_radius_m, gap_m, pressure_Pa, temperature_K, humidity_pct)
    E_tip  = fl["_E_tip"]
    E_onset= pk["_E_onset"]
    E_break= fl["_E_break"]

    # Growth instability — dI/dV at current operating point
    # Using Sigmond (1982) needle formula: I = C·μ·(V-V_onset)·V/d²
    # dI/dV = C·μ·(2V - V_onset)/d²
    # Instability when slope exceeds load line slope
    r   = tip_radius_m
    d   = gap_m
    ln2 = math.log(2*d/r) if (2*d/r) > 1 else 1.0
    C   = 4 * math.pi * EPSILON_0 / (d**2 * ln2)
    V_onset = pk["_V_onset"]
    dI_dV   = C * MU_POS * (2*voltage_V - V_onset) if voltage_V > V_onset else 0.0
    # Steepness relative to a 10MΩ load line (1/R = 1e-7 A/V)
    relative_slope = dI_dV / 1e-7

    instability = relative_slope > 10  # more than 10× steeper than typical load line

    return {
        "_dI_dV":            round(dI_dV, 8),
        "_slope_ratio":      round(relative_slope, 2),
        "_instability":      instability,
        "_I_ratio_corona":   current_A / I_CORONA_MAX_A    if current_A > 0 else 0.0,
        "_I_ratio_streamer": current_A / I_STREAMER_MAX_A  if current_A > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STABILITY CRITERION (circuit)
# ═══════════════════════════════════════════════════════════════════════════════

def stability_criterion(voltage_V: float,
                         tip_radius_m: float,
                         gap_m: float,
                         R_ballast_ohm: float,
                         pressure_Pa: float = P0,
                         temperature_K: float = T0) -> dict:
    """
    Load line stability: |dI_load/dV| > |dI_plasma/dV|

    If the corona V-I slope is steeper than the load line, the system
    has positive feedback → runaway → arc.

    dI_load/dV  = -1/R_ballast         (load line slope, always negative)
    dI_plasma/dV = C·μ·(2V - V_onset)/d²  (Sigmond corona, always positive)

    For stability: 1/R_ballast > dI_plasma/dV
    → R_ballast < 1 / dI_plasma_dV
    """
    r   = tip_radius_m
    d   = gap_m
    ln2 = math.log(2*d/r) if (2*d/r) > 1 else 1.0
    pk  = peek_onset(r, d, pressure_Pa, temperature_K)
    V_onset = pk["_V_onset"]

    C   = 4 * math.pi * EPSILON_0 / (d**2 * ln2)
    dI_plasma_dV  = C * MU_POS * (2*voltage_V - V_onset) if voltage_V > V_onset else 0.0
    dI_load_dV    = 1.0 / R_ballast_ohm

    # Stability criterion for Townsend corona (positive-slope I-V curve):
    # For a stable operating point, the load line must be steeper than
    # the corona I-V curve: |dI_load/dV| > dI_corona/dV
    # → 1/R > dI_corona/dV
    # → R < R_max = 1/dI_corona/dV
    #
    # IMPORTANT: this is OPPOSITE to intuition from arc suppression.
    # For corona: LOWER R = more stable (steeper load line).
    # BUT: lower R = higher current = arc risk at high V.
    # So there is both a R_max (stability) and R_min (current limit).
    R_max_stable  = 1.0 / dI_plasma_dV if dI_plasma_dV > 0 else float('inf')
    stable        = dI_load_dV > dI_plasma_dV   # 1/R > dI_corona
    margin_pct    = (dI_load_dV - dI_plasma_dV) / dI_plasma_dV * 100 if dI_plasma_dV > 0 else 100

    return {
        "_stable":               stable,
        "_dI_load_dV":           round(dI_load_dV, 8),
        "_dI_plasma_dV":         round(dI_plasma_dV, 8),
        "_stability_margin_pct": round(margin_pct, 1),
        "_R_max_MOhm":           round(R_max_stable / 1e6, 2),
        "R_ballast_ohm":         R_ballast_ohm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BREAKDOWN MARGIN
# ═══════════════════════════════════════════════════════════════════════════════

def breakdown_margin(E_tip: float, E_break: float,
                      E_avg: float = None) -> dict:
    """
    Breakdown safety using E_tip/E_break as primary metric.

    E_tip/E_break is the correct safety indicator — not E_avg/E_break.
    Tip field drives breakdown. E_avg underestimates risk.

    Primary: R = E_tip / E_break
        R < 3    → below corona onset
        3 – 8    → stable corona ✓
        8 – 12   → streamer transition ⚠  (10.8 = your design)
        > 12     → arc / spark ❌

    Real-world: surface roughness boosts local field 2–5×.
    Apply 2× safety factor for practical assessment.
    At Kerala 80% RH: E_break_eff ≈ 2.5 MV/m (not 3 MV/m).
    """
    if E_tip <= 0:
        return {"_ratio": 0.0, "_ratio_rough": 0.0, "_E_tip": 0.0, "_E_break": E_break}
    R     = E_tip / E_break
    R_eff = R * 2.0   # 2× surface roughness conservative estimate
    M     = E_break / E_avg if E_avg and E_avg > 0 else None
    result = {
        "_ratio":       round(R, 3),
        "_ratio_rough": round(R_eff, 3),
        "_E_tip":       E_tip,
        "_E_break":     E_break,
        "E_tip_MV_m":   round(E_tip/1e6, 3),
        "E_break_MV_m": round(E_break/1e6, 3),
    }
    if M is not None:
        result["_Ebreak_Eavg"] = round(M, 3)
    return result




# ═══════════════════════════════════════════════════════════════════════════════
# 6. SPACE-CHARGE STABILITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def space_charge_stability(rho_max_C_m3: float,
                            voltage_V: float,
                            gap_m: float) -> dict:
    """
    ΔV_sc / V ratio check.

    ΔV_sc ≈ ρe_max · d² / (2·ε₀)

    If ΔV_sc / V > 0.1 (10%): field distortion is significant,
    instability risk increases (space charge can locally push E_tip
    over E_break even when the applied voltage seems safe).
    """
    delta_V = rho_max_C_m3 * gap_m**2 / (2 * EPSILON_0)
    ratio   = delta_V / voltage_V if voltage_V > 0 else 0.0
    significant = ratio > 0.1

    return {
        "_delta_V_sc_V": round(delta_V, 2),
        "_sc_ratio":     round(ratio, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LOCALISED FIELD ENHANCEMENT (geometry risk)
# ═══════════════════════════════════════════════════════════════════════════════

def geometry_risk(voltage_V: float,
                   tip_radius_m: float,
                   gap_m: float,
                   n_emitters: int = 1,
                   emitter_spacing_m: float = None) -> dict:
    """
    Detect hidden hotspots from geometry.

    β = E_tip / E_avg  (field enhancement factor)

    Expected range for needle: β = ln(2d/r) ≈ 5–13 for typical geometries.
    If β >> expected: some geometric feature is creating a local hotspot.

    For multi-emitter arrays: proximity effect between adjacent needles
    increases effective E_tip when spacing is too small.
    Proximity correction: E_effective ≈ E_tip · (1 + r/spacing)
    """
    ln_term = math.log(2*gap_m/tip_radius_m) if (2*gap_m/tip_radius_m) > 1 else 1.0
    E_avg   = voltage_V / gap_m
    E_tip   = voltage_V / (tip_radius_m * ln_term)
    beta    = E_tip / E_avg  # = ln_term (analytic check)

    # β_expected = ln(2d/r) = analytic value for hyperboloid geometry
    # (= gap_m/(tip_radius_m*ln_t) algebraically — same formula, cleaner form)
    ln_t = math.log(2*gap_m/tip_radius_m) if (2*gap_m/tip_radius_m)>1 else 1.0
    beta_expected = ln_t   # E_tip/E_avg = ln(2d/r) for hyperboloid

    proximity_factor = 1.0
    if n_emitters > 1 and emitter_spacing_m is not None:
        proximity_factor = 1 + tip_radius_m / emitter_spacing_m
        E_tip_eff = E_tip * proximity_factor
    else:
        E_tip_eff = E_tip

    return {
        "_beta":             round(beta, 2),
        "_beta_expected":    round(beta_expected, 2),
        "_E_tip":            E_tip,
        "_E_tip_eff":        E_tip_eff,
        "_proximity_factor": round(proximity_factor, 3),
        "_spacing_ratio":    (emitter_spacing_m / tip_radius_m) if emitter_spacing_m else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8+9. FULL PHYSICS AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def analyse(voltage_V: float,
             tip_radius_m: float,
             gap_m: float,
             R_ballast_ohm: float = 10e6,
             current_A: float = 0.0,
             rho_max_C_m3: float = 0.0,
             n_emitters: int = 1,
             emitter_spacing_m: float = None,
             pressure_Pa: float = P0,
             temperature_K: float = T0,
             humidity_pct: float = 0.0) -> dict:
    """
    Pure physics module (Option A).
    No validation, no decisions, no gating.
    All decisions handled in physics_validation.
    """
    fl   = field_limits(voltage_V, tip_radius_m, gap_m,
                        pressure_Pa, temperature_K, humidity_pct)
    pk   = peek_onset(tip_radius_m, gap_m, pressure_Pa, temperature_K, humidity_pct)
    td   = transition_detection(voltage_V, tip_radius_m, gap_m, current_A,
                                pressure_Pa, temperature_K, humidity_pct)
    sc   = stability_criterion(voltage_V, tip_radius_m, gap_m, R_ballast_ohm,
                               pressure_Pa, temperature_K)
    bm   = breakdown_margin(fl["_E_tip"], fl["_E_break"], fl["_E_avg"])
    geom = geometry_risk(voltage_V, tip_radius_m, gap_m, n_emitters, emitter_spacing_m)
    sc_check = (space_charge_stability(rho_max_C_m3, voltage_V, gap_m)
                if rho_max_C_m3 > 0 else {"_delta_V_sc_V": 0, "_sc_ratio": 0.0})

    return {
        # ── Core field physics ────────────────────────────────────────────
        "_E_tip":            fl["_E_tip"],
        "_E_avg":            fl["_E_avg"],
        "_E_break":          fl["_E_break"],
        "_E_onset":          pk["_E_onset"],
        "_V_onset":          pk["_V_onset"],
        # ── Derived ratios ────────────────────────────────────────────────
        "_ratio":            bm["_ratio"],
        "_ratio_rough":      bm["_ratio_rough"],
        "_Ebreak_Eavg":      bm.get("_Ebreak_Eavg", 0.0),
        "_h_r":              gap_m / tip_radius_m if tip_radius_m > 0 else 0,
        # ── Stability ─────────────────────────────────────────────────────
        "_stable":           sc["_stable"],
        "_dI_plasma_dV":     sc["_dI_plasma_dV"],
        "_dI_load_dV":       sc["_dI_load_dV"],
        "_stability_margin": sc["_stability_margin_pct"],
        "_R_max_MOhm":       sc["_R_max_MOhm"],
        # ── Growth / transition ───────────────────────────────────────────
        "_dI_dV":            td["_dI_dV"],
        "_slope_ratio":      td["_slope_ratio"],
        "_instability":      td["_instability"],
        "_I_ratio_corona":   td["_I_ratio_corona"],
        # ── Geometry ─────────────────────────────────────────────────────
        "_beta":             geom["_beta"],
        "_beta_expected":    geom["_beta_expected"],
        "_proximity_factor": geom["_proximity_factor"],
        "_spacing_ratio":    geom["_spacing_ratio"],
        # ── Space charge ──────────────────────────────────────────────────
        "_delta_V_sc_V":     sc_check["_delta_V_sc_V"],
        "_sc_ratio":         sc_check["_sc_ratio"],
        # ── Environmental ─────────────────────────────────────────────────
        "_delta":            air_density_factor(pressure_Pa, temperature_K, humidity_pct),
        "_humidity_pct":     humidity_pct,
        # ── Readable convenience (for UI/voice only) ──────────────────────
        "E_tip_MV_m":        fl["E_tip_MV_m"],
        "E_avg_MV_m":        fl["E_avg_MV_m"],
        "E_break_MV_m":      fl["E_break_MV_m"],
        "E_onset_MV_m":      pk["E_onset_MV_m"],
        "V_onset_V":         pk["V_onset_V"],
        # ── Metadata ──────────────────────────────────────────────────────
        "model": "breakdown_analysis_v1",
    }

# Keep classify_and_gate as a thin alias so app.py doesn't break
classify_and_gate = analyse

class BreakdownStabilityAI:

    @staticmethod
    def analyse(voltage_V: float,
                 tip_radius_mm: float,
                 gap_mm: float,
                 R_ballast_MOhm: float = 10.0,
                 current_uA: float = 0.0,
                 n_emitters: int = 1,
                 spacing_mm: float = None,
                 pressure_Pa: float = P0,
                 temperature_C: float = 20.0,
                 humidity_pct: float = 80.0) -> str:
        result = classify_and_gate(
            voltage_V      = voltage_V,
            tip_radius_m   = tip_radius_mm / 1000,
            gap_m          = gap_mm / 1000,
            R_ballast_ohm  = R_ballast_MOhm * 1e6,
            current_A      = current_uA * 1e-6,
            n_emitters     = n_emitters,
            emitter_spacing_m = spacing_mm / 1000 if spacing_mm else None,
            pressure_Pa    = pressure_Pa,
            temperature_K  = temperature_C + 273.15,
            humidity_pct   = humidity_pct,
        )
        return json.dumps(result, indent=2)

    @staticmethod
    def voltage_sweep(tip_radius_mm: float,
                       gap_mm: float,
                       V_min: float = 1000,
                       V_max: float = 40000,
                       n_steps: int = 10,
                       humidity_pct: float = 80.0) -> str:
        """Sweep voltage — routes through breakdown_gate for decisions."""
        from physics_validation import breakdown_gate
        results = []
        for i in range(n_steps):
            V = V_min + (V_max - V_min) * i / (n_steps - 1)
            r = analyse(
                voltage_V    = V,
                tip_radius_m = tip_radius_mm / 1000,
                gap_m        = gap_mm / 1000,
                humidity_pct = humidity_pct,
            )
            g = breakdown_gate(r)
            results.append({
                "V":       round(V),
                "ratio":   round(r["_ratio"], 2),
                "stable":  r["_stable"],
                "verdict": g["verdict"][:40],
            })
        return json.dumps({"sweep": results}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("breakdown_stability.py — Option A pure physics module")
    bd = analyse(20000, 1e-4, 0.023, humidity_pct=80)
    print(f"ratio={bd['_ratio']:.2f}  stable={bd['_stable']}  beta={bd['_beta']}")
    print(f"V_onset={bd['V_onset_V']}V  E_tip={bd['E_tip_MV_m']}MV/m")
    print("✓ raw physics only — no decisions, no warnings")
