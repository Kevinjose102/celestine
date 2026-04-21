"""
physics_validation.py — Celestine Validation & Constraint Layer
================================================================
Single schema. Single authority. All modules import from here.

SCHEMA (unified):
    Every flag is a dict: {"code": str, "severity": str, "message": str}
    severity ∈ {"info", "warning", "error"}
        "info"    → notable, result valid
        "warning" → result unreliable, use with caution
        "error"   → model invalid, do not use downstream

    model_valid = not any(f["severity"] == "error" for f in flags)
    (severity drives validity — never string-match on code)

VALIDATION APPROACH:
    - validate(result) uses already-computed fields from result dict
    - does NOT recompute physics (fix 5: no duplicate logic)
    - feeds flags into caller's flag list (fix 2: influences physics)

SYSTEM GATE:
    validate_system(results_dict) is the global enforcement point.
    Must be called before any Module 5+ computation.

References:
    All physics references in individual module files.
"""

import math

# ── Physical constants (reference only — modules own their computations) ──────
E_BREAK_STP = 3.0e6
PEEK_A      = 3.1e6
PEEK_B      = 0.030
EPSILON_0   = 8.854e-12
MU_POS      = 2.0e-4


# ═══════════════════════════════════════════════════════════════
# UNIFIED FLAG FACTORY
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# CENTRALISED THRESHOLDS (single source of truth)
# All regime decisions across the entire system use these values.
# Based on Raizer (1991) Table 3.1, humidity-adjusted for Kerala.
# ═══════════════════════════════════════════════════════════════

RATIO_THRESHOLDS = {
    "no_discharge": 3.0,    # E_tip/E_break < 3  → no discharge
    "corona_max":   10.0,   # 3 – 10  → stable corona (EHD target)
    "streamer_max": 15.0,   # 10 – 15 → streamer transition
                            # > 15    → arc / spark
}

def ratio_to_regime(ratio: float) -> str:
    """Derive regime string from raw E_tip/E_break ratio. Used by Option B only."""
    if ratio > RATIO_THRESHOLDS["streamer_max"]:
        return "ARC LIKELY"
    if ratio > RATIO_THRESHOLDS["corona_max"]:
        return "STREAMER TRANSITION"
    if ratio >= RATIO_THRESHOLDS["no_discharge"]:
        return "STABLE CORONA"
    return "NO DISCHARGE"


def flag(code: str, severity: str, message: str, fix: str = None) -> dict:
    """
    Create a validated flag dict.
    severity must be 'info', 'warning', or 'error'.
    """
    assert severity in ("info", "warning", "error"), \
        f"severity must be info/warning/error, got: {severity!r}"
    f = {"code": code, "severity": severity, "message": message}
    if fix:
        f["fix"] = fix
    return f


def normalize_flag(f: dict) -> dict:
    """
    Convert old-schema flags {"level": "CRITICAL/INVALID/WARNING/INFO"}
    to unified schema {"severity": "error/warning/info"}.
    Idempotent — safe to call on already-normalized flags.
    """
    if "severity" in f:
        return f
    level_map = {
        "CRITICAL": "error",
        "INVALID":  "error",
        "WARNING":  "warning",
        "INFO":     "info",
    }
    level = f.pop("level", "warning")
    f["severity"] = level_map.get(level, "warning")
    return f


def normalize_flags(flags: list) -> list:
    """Normalize a list of flags to unified schema."""
    return [normalize_flag(dict(f)) for f in flags]


# ═══════════════════════════════════════════════════════════════
# MODULE-LEVEL VALIDATORS
# Each reads from already-computed result fields (fix 5)
# Returns list of flags to be merged into caller's flag list (fix 2)
# ═══════════════════════════════════════════════════════════════

def validate_electrostatics(result: dict) -> list:
    """
    Read from result dict. Return flags.
    Expects flat physics keys only (no decision fields).
    """
    flags = []

    E_tip  = result.get("_E_tip", 0) or result.get("E_tip_MV_m", 0) * 1e6
    E_avg  = result.get("_E_avg", 0)
    h_r    = result.get("_h_r", 0)

    # Geometry
    if h_r > 0 and h_r < 20:
        flags.append(flag("E1_INVALID_GEOMETRY", "error",
            f"h/r = {h_r:.1f} < 20: hyperboloid formula invalid.",
            fix="Use sphere model: E_tip = V/r"))

    E_break = result.get("_E_break") or result.get("E_break_MV_m", 3.0) * 1e6
    if E_break and E_break < 100:   # MV/m → V/m
        E_break *= 1e6

    # Gap breakdown (enforce E_gap_ratio > 0.6 → arc)
    if E_tip > 0 and E_avg > 0:
        E_gap_ratio = E_avg / E_break
        if E_gap_ratio > 0.6:
            flags.append(flag("E1_GAP_BREAKDOWN", "error",
                f"E_avg/E_break = {E_gap_ratio:.2f} > 0.6: "
                f"bulk gap breakdown, not just tip corona. Arc inevitable.",
                fix="Increase gap distance"))
        elif E_gap_ratio > 0.3:
            flags.append(flag("E1_GAP_HIGH", "warning",
                f"E_avg/E_break = {E_gap_ratio:.2f} > 0.3: "
                f"gap field elevated. Streamer growth likely."))

    return flags


def validate_charge_transport(result: dict) -> list:
    """Read from charge transport result. Return flags."""
    flags = []

    if result.get("CANNOT_COMPUTE"):
        flags.append(flag("T2_PENDING", "info",
            "Module 2 awaiting Module 3 source S(r,z). "
            "Townsend α is valid. ρe is not yet computable."))
        return flags

    rho_max = result.get("rho_max_C_m3", 0)
    I_uA    = result.get("I_collector_uA", 0)
    kappa   = result.get("_kappa", 0)

    if rho_max < 0:
        flags.append(flag("T2_NEGATIVE_RHO", "error",
            "ρe < 0: non-physical. Solver diverged.",
            fix="Reduce SOR omega or increase inner iterations"))

    if rho_max > 100:
        flags.append(flag("T2_HIGH_RHO", "warning",
            f"ρe_max = {rho_max:.1f} C/m³: unusually high. "
            f"Check injection BC scaling."))

    if kappa > 1.0:
        flags.append(flag("T2_SPACE_CHARGE_DOMINATED", "warning",
            f"κ = {kappa:.2f} > 1: space charge dominant. "
            f"Laplace E-field unreliable. Use Poisson coupling.",
            fix="Use poisson_charge_coupling.analytic_space_charge_correction()"))
    elif kappa > 0.3:
        flags.append(flag("T2_SPACE_CHARGE_MODERATE", "info",
            f"κ = {kappa:.2f}: moderate space charge. "
            f"Analytic correction recommended."))

    if I_uA > 5000:
        flags.append(flag("T2_CURRENT_ARC", "error",
            f"I = {I_uA:.0f} μA > 5 mA: arc regime, not stable corona.",
            fix="Increase R_ballast or reduce voltage"))

    return flags


def validate_corona(result):
    return []


def validate_circuit(result: dict) -> list:
    """Read from circuit result. Return flags."""
    flags = []

    if result.get("CANNOT_COMPUTE"):
        return [flag("C4_PENDING", "info",
            "Current requires ρe from Module 2+3.")]

    V_op    = result.get("V_operating_V", 0)
    V_onset = result.get("V_onset_V", 0)
    V_sup   = result.get("V_supply_V", 0)
    I_uA    = result.get("I_operating_uA", 0)

    if V_op > 0 and V_onset > 0 and V_op < V_onset * 0.9:
        flags.append(flag("C4_VOP_BELOW_ONSET", "warning",
            f"V_op = {V_op}V < V_onset = {V_onset:.0f}V. "
            f"No discharge at this ballast value."))

    if V_op > V_sup > 0:
        flags.append(flag("C4_VOP_IMPOSSIBLE", "error",
            f"V_op = {V_op}V > V_supply = {V_sup}V: solver diverged.",
            fix="Check I-V model parameters"))

    if I_uA > 5000:
        flags.append(flag("C4_ARC_CURRENT", "error",
            f"I_op = {I_uA:.0f} μA > 5 mA: arc regime.",
            fix="Increase R_ballast or reduce voltage"))

    return flags


def validate_breakdown(result: dict) -> list:
    """Read from breakdown result. Return flags."""
    flags = []

    V_on   = result.get("V_onset_V", 0)
    delta  = result.get("_delta", 1.0)
    if delta < 0.6:
        flags.append(flag("B7_ENV_EXTREME", "warning",
            f"Air density δ = {delta:.2f}: Peek constants may be inaccurate."))

    if V_on > 0 and (V_on < 200 or V_on > 150000):
        flags.append(flag("B7_VONSET_RANGE", "warning",
            f"V_onset = {V_on:.0f}V outside plausible 200V–150kV. Check units."))

    return flags


# ═══════════════════════════════════════════════════════════════
# CROSS-MODULE CONSISTENCY
# ═══════════════════════════════════════════════════════════════

def cross_check(results: dict) -> list:
    """
    Detect contradictions across module outputs.
    Returns list of flags (same schema).

    results = {
        "electrostatics": {...},
        "corona":         {...},
        "breakdown":      {...},
        "circuit":        {...},
        "transport":      {...},
    }
    """
    flags = []

    e  = results.get("electrostatics", {})
    c3 = results.get("corona", {})
    b7 = results.get("breakdown", {})
    t2 = results.get("transport", {})

    # V_onset consistency across modules
    v_onset_vals = {}
    if e.get("corona_onset_V"):
        v_onset_vals["electrostatics"] = float(e["corona_onset_V"])
    if c3.get("_V_onset"):
        v_onset_vals["corona"] = float(c3["_V_onset"])
    elif c3.get("V_onset_V"):
        v_onset_vals["corona"] = float(c3["V_onset_V"])
    if b7.get("V_onset_V"):
        v_onset_vals["breakdown"] = float(b7["V_onset_V"])

    if len(v_onset_vals) >= 2:
        vals = list(v_onset_vals.values())
        spread = (max(vals) - min(vals)) / max(vals)
        if spread > 0.05:
            flags.append(flag("CROSS_VONSET_MISMATCH", "error",
                f"V_onset inconsistent: {v_onset_vals}. Spread={spread*100:.0f}%. "
                f"All modules must use identical Peek formula.",
                fix="Ensure same r, d, pressure used across all modules"))

    # E_tip consistency
    # Fix 5: only numeric fields, no string parsing
    _e_tip_raw = e.get("_E_tip", 0)
    e_tip_e = _e_tip_raw / 1e6 if _e_tip_raw and _e_tip_raw > 100 else               (e.get("E_tip_MV_m") or 0)
    e_tip_b = b7.get("E_tip_MV_m", 0)
    if e_tip_e and e_tip_b and e_tip_b > 0:
        diff = abs(float(e_tip_e) - float(e_tip_b)) / float(e_tip_b)
        if diff > 0.05:
            flags.append(flag("CROSS_ETIP_MISMATCH", "error",
                f"E_tip: electrostatics={e_tip_e} vs breakdown={e_tip_b} MV/m "
                f"({diff*100:.0f}% difference). Different formulas or inputs.",
                fix="Ensure identical V, r, d (SI) passed to both modules"))

    # Compare raw ratios across modules (not regime strings)
    ratio_e = (e.get("_E_tip", 0) / e.get("_E_break", 1)) if e.get("_E_break") else 0
    ratio_b = b7.get("_ratio", 0)
    ratio_c = c3.get("_ratio", 0)
    ratios = {k: v for k, v in
              [("electrostatics", ratio_e), ("corona", ratio_c), ("breakdown", ratio_b)]
              if v > 0}
    if len(ratios) >= 2:
        vals = list(ratios.values())
        spread = max(vals) - min(vals)
        if spread > 5.0:
            flags.append(flag("CROSS_RATIO_SPREAD", "error",
                f"E_tip/E_break ratios diverge across modules: {ratios}. "
                f"Spread={spread:.1f} — different inputs or formulas used.",
                fix="Ensure identical V, r, d (SI) passed to all modules"))
        elif spread > 2.0:
            flags.append(flag("CROSS_RATIO_MINOR", "warning",
                f"E_tip/E_break ratio spread={spread:.1f} across modules."))

    # Meek vs field-ratio indicator (flat field only)
    meek  = c3.get("_meek")
    ratio = b7.get("_ratio", 0)
    if meek is not None and ratio > 0:
        meek_streamer  = meek > 18
        ratio_streamer = ratio > 10
        if meek_streamer != ratio_streamer:
            flags.append(flag("CROSS_STREAMER_INDICATORS", "warning",
                f"Streamer indicators disagree: Meek={meek:.1f} "
                f"({'streamer' if meek_streamer else 'stable'}) vs "
                f"ratio={ratio:.1f} "
                f"({'streamer' if ratio_streamer else 'stable'}). "
                f"Meek criterion is more reliable."))

    # Space charge: does Module 1 E-field need Poisson correction?
    kappa = t2.get("_kappa", 0)
    if kappa > 0.5:
        flags.append(flag("CROSS_POISSON_NEEDED", "warning",
            f"κ = {kappa:.2f} > 0.5: space charge significant. "
            f"Module 1 Laplace field overestimates E_tip. "
            f"All downstream results (Module 5/6) will be overestimates.",
            fix="Use poisson_charge_coupling.analytic_space_charge_correction()"))

    return flags


# ═══════════════════════════════════════════════════════════════
# AGGREGATE — merge flags from any number of module results (fix 5)
# ═══════════════════════════════════════════════════════════════

def aggregate_validation(*module_results) -> dict:
    """
    Merge validation flags from multiple module outputs.
    Normalizes legacy "level"-schema flags to unified "severity" schema.

    Usage:
        agg = aggregate_validation(tip_result, solver_result, corona_result)
    """
    all_flags  = []
    confidence = 1.0

    for r in module_results:
        if not isinstance(r, dict):
            continue
        for key in ("flags", "hard_flags"):
            for f in r.get(key, []):
                if isinstance(f, dict):
                    all_flags.append(normalize_flag(dict(f)))
                elif isinstance(f, str):
                    sev = "error" if any(k in f for k in
                          ("INVALID", "ARC", "ERROR", "CRITICAL")) else "warning"
                    all_flags.append({"code": f.split(":")[0],
                                      "severity": sev, "message": f})
        # Solver sub-dict
        for f in r.get("solver", {}).get("flags", []):
            if isinstance(f, dict):
                all_flags.append(normalize_flag(dict(f)))
        # Confidence: take minimum
        for conf_key in ("confidence",):
            c = r.get(conf_key) or r.get("solver", {}).get("confidence")
            if c is not None:
                confidence = min(confidence, float(c))

    model_valid = not any(f.get("severity") == "error" for f in all_flags)
    errors      = [f for f in all_flags if f.get("severity") == "error"]
    warnings    = [f for f in all_flags if f.get("severity") == "warning"]

    return {
        "valid":      model_valid,
        "confidence": round(confidence, 3),
        "all_flags":  all_flags,
        "errors":     errors,
        "warnings":   warnings,
        "summary": (
            f"INVALID ({len(errors)} errors)" if errors else
            f"WARNING ({len(warnings)} warnings)" if warnings else
            "VALID"
        ),
    }


# ═══════════════════════════════════════════════════════════════
# SYSTEM GATE — global enforcement before Module 5+ (fix 6)
# ═══════════════════════════════════════════════════════════════

def validate_system(results: dict) -> dict:
    """
    Global gate. Must pass before any Module 5+ computation.

    results = dict of module_name → result_dict

    Runs:
        1. Per-module validators (reads from result, adds flags)
        2. Cross-module consistency checks
        3. Aggregate — final valid/confidence/summary

    Returns:
        {
            "valid":      bool,   # False = stop, do not proceed
            "confidence": float,  # 0.0–1.0
            "all_flags":  [...],  # every flag from every module
            "errors":     [...],  # error-severity flags only
            "warnings":   [...],  # warning-severity flags only
            "summary":    str,
            "per_module": {...},  # flags per module
        }
    """
    VALIDATORS = {
        "electrostatics": validate_electrostatics,
        "corona":         validate_corona,
        "breakdown":      validate_breakdown,
        "circuit":        validate_circuit,
        "transport":      validate_charge_transport,
    }

    per_module = {}
    all_new_flags = []

    for mod_name, result in results.items():
        validator = VALIDATORS.get(mod_name)
        if validator:
            mod_flags = validator(result)
            per_module[mod_name] = mod_flags
            all_new_flags.extend(mod_flags)

    # Cross-module checks
    cross_flags = cross_check(results)
    per_module["cross_module"] = cross_flags
    all_new_flags.extend(cross_flags)

    # Aggregate existing module flags (already in result dicts)
    agg = aggregate_validation(*results.values())

    # No double-counting — only add new_flags not already in agg
    existing_codes = {(f.get("code"), f.get("message")) for f in agg["all_flags"]}
    deduped_new = [f for f in all_new_flags
                   if (f.get("code"), f.get("message")) not in existing_codes]
    all_flags = agg["all_flags"] + deduped_new

    # Dependency gating — no ρe → no current → no thrust
    dep_flags = check_dependencies(results)
    per_module["dependencies"] = dep_flags
    for f in dep_flags:
        if (f.get("code"), f.get("message")) not in existing_codes:
            all_flags.append(f)

    # Mode consistency — EHD vs VACUUM, no mixing
    mode = detect_mode(results)
    mode_flags = check_mode_consistency(results, mode)
    per_module["mode_check"] = mode_flags
    for f in mode_flags:
        if (f.get("code"), f.get("message")) not in existing_codes:
            all_flags.append(f)

    errors   = [f for f in all_flags if f.get("severity") == "error"]
    warnings = [f for f in all_flags if f.get("severity") == "warning"]
    valid    = len(errors) == 0

    confidence = agg["confidence"]
    if errors:
        confidence = 0.0
    elif warnings:
        confidence = round(min(confidence, confidence * 0.7), 3)

    return {
        "valid":              valid,
        "confidence":         confidence,
        "all_flags":          all_flags,
        "errors":             errors,
        "warnings":           warnings,
        "per_module":         per_module,
        "mode":               mode,
        "flag_counts":        {"errors": len(errors), "warnings": len(warnings),
                               "total": len(all_flags)},
        "summary": (
            f"SYSTEM INVALID — {len(errors)} error(s) block downstream computation"
            if errors else
            f"SYSTEM WARNING — {len(warnings)} warning(s), confidence={confidence:.2f}"
            if warnings else
            "SYSTEM VALID — all checks passed"
        ),
        "proceed_to_module_5": valid,
    }


# ═══════════════════════════════════════════════════════════════
# DEPENDENCY GATE — enforces physics chain order (fix 2)
# ═══════════════════════════════════════════════════════════════

def check_dependencies(results: dict) -> list:
    """
    Enforces the physics chain:
        Electrostatics → Corona (S) → Charge transport (ρe) → Circuit (I) → Force → Thrust

    Returns error flags for any broken link.
    Modules expose validity via underscore fields: _rho_valid, _current_valid, etc.
    """
    flags = []
    t2 = results.get("transport", {})
    c4 = results.get("circuit", {})
    c3 = results.get("corona", {})

    # Gate 1: S not computed → ρe invalid
    if not c3.get("module_2_now_solvable", True):
        flags.append(flag("DEP_S_NOT_AVAILABLE", "error",
            "Corona module S(r,z) = 0 (below onset). ρe(r,z) cannot be solved.",
            fix="Raise voltage above corona onset V_onset"))

    # Gate 2: ρe not solved → current invalid
    rho_invalid = (t2.get("CANNOT_COMPUTE_rho")
                   or t2.get("CANNOT_COMPUTE")
                   or t2.get("_rho_valid") is False)
    if rho_invalid:
        if c4.get("I_operating_uA", 0) > 0:
            flags.append(flag("DEP_CURRENT_WITHOUT_RHO", "error",
                "I computed but ρe(r,z) not solved. "
                "I = ∫ρe·μ·E·dA is physically invalid without ρe.",
                fix="Complete Module 3 → Module 2 chain first"))
        # Propagate: mark force and thrust invalid
        flags.append(flag("DEP_FORCE_BLOCKED", "error",
            "f = ρe·E requires valid ρe. EHD force cannot be computed.",
            fix="Solve ρe first via Module 2+3"))

    # Gate 3: circuit below onset
    if c4.get("_INVALID"):
        flags.append(flag("DEP_CIRCUIT_INVALID", "error",
            c4.get("_reason", "Circuit result invalid."),
            fix="Check V_supply vs V_onset"))

    return flags


# ═══════════════════════════════════════════════════════════════
# MODE SEPARATION — EHD vs VACUUM (fix 7)
# ═══════════════════════════════════════════════════════════════

PHYSICS_MODES = {
    "EHD": {
        "description": "Air EHD thruster (corona discharge, drift-diffusion)",
        "pressure_range_Pa": (10000, 200000),
        "valid_modules": ["electrostatics","corona","breakdown","circuit","transport"],
        "invalid_modules": ["ion_thruster","vacuum_propellant","beam_optics"],
        "current_regime": "corona",
        "field_model": "laplace_or_poisson",
    },
    "VACUUM": {
        "description": "Vacuum ion thruster (plasma, beam optics)",
        "pressure_range_Pa": (0, 100),
        "valid_modules": ["ion_thruster","vacuum_propellant","beam_optics","sheath"],
        "invalid_modules": ["corona","ehd_force","air_plasma"],
        "current_regime": "plasma",
        "field_model": "poisson",
    },
}

def detect_mode(results: dict) -> str:
    """
    Detect operating mode from results.
    Returns 'EHD', 'VACUUM', or 'UNKNOWN'.
    """
    c3 = results.get("corona", {})
    p_ratio = c3.get("_p_ratio")
    pressure = (c3.get("_pressure_Pa")
                or (p_ratio * 101325 if p_ratio else None)
                or 101325)
    if pressure > 1000:
        return "EHD"
    elif pressure < 100:
        return "VACUUM"
    return "UNKNOWN"

def check_mode_consistency(results: dict, mode: str = None) -> list:
    """
    Check that modules in use are consistent with the physics mode.
    EHD modules must not be used with vacuum inputs and vice versa.
    """
    flags = []
    detected = mode or detect_mode(results)

    if detected == "EHD":
        c3 = results.get("corona", {})
        p_ratio = c3.get("_p_ratio")
        pressure = (c3.get("_pressure_Pa")
                    or (p_ratio * 101325 if p_ratio else None)
                    or 101325)
        if pressure < 1000:
            flags.append(flag("MODE_PRESSURE_MISMATCH", "error",
                f"EHD mode requires P > 1000 Pa but got {pressure} Pa. "
                f"Use VACUUM mode for low-pressure operation.",
                fix="Set mode='VACUUM' for ion thruster physics"))
    elif detected == "VACUUM":
        if "corona" in results or "ehd" in results:
            flags.append(flag("MODE_EHD_IN_VACUUM", "error",
                "Corona/EHD modules used in vacuum regime. "
                "These are only valid at atmospheric pressure.",
                fix="Remove corona module or switch to EHD mode"))

    return flags


# ═══════════════════════════════════════════════════════════════
# HARD STOP — raises on invalid (fix 6)
# ═══════════════════════════════════════════════════════════════

class PhysicsError(Exception):
    """Raised when physics validation finds an error-severity flag.
    Caller may catch to degrade gracefully or let it propagate to stop execution.
    """
    def __init__(self, system_result: dict):
        self.system_result = system_result
        errors = [f["code"] for f in system_result.get("errors", [])]
        super().__init__(
            f"Physics validation failed — {len(errors)} error(s): {errors}. "
            f"Do not proceed to Module 5+."
        )


def enforce_gate(system_result: dict) -> None:
    """
    Call after validate_system(). Raises PhysicsError if not valid.
    Use this to make the gate mandatory rather than advisory.

    Usage:
        result = validate_system({...})
        enforce_gate(result)   # raises if invalid, silent if valid
        # safe to proceed to Module 5
    """
    if not system_result.get("valid", True):
        raise PhysicsError(system_result)


# ── Convenience: backward-compatible validate() entry point ──────────────────

def validate(module: str, inputs: dict, result: dict) -> dict:
    """
    Backward-compatible entry point for existing module calls.
    Runs the module-specific validator on the result dict.
    Returns normalized flag list + summary.
    """
    VALIDATORS = {
        "electrostatics":   validate_electrostatics,
        "charge_transport": validate_charge_transport,
        "corona":           validate_corona,
        "circuit":          validate_circuit,
        "breakdown":        validate_breakdown,
    }
    fn = VALIDATORS.get(module)
    flags = fn(result) if fn else []
    flags = normalize_flags(flags)
    errors   = [f for f in flags if f["severity"] == "error"]
    warnings = [f for f in flags if f["severity"] == "warning"]
    return {
        "quality":  "INVALID" if errors else "WARNING" if warnings else "VALID",
        "summary":  f"INVALID ({len(errors)} errors)" if errors else
                    f"WARNING ({len(warnings)} warnings)" if warnings else "VALID",
        "flags":    flags,
        "module":   module,
    }



# ═══════════════════════════════════════════════════════════════
# BREAKDOWN GATE — Option B system decision layer
# Reads raw physics from breakdown_stability.analyse()
# Owns: warnings, ACCEPTED/REJECTED, verdict
# ═══════════════════════════════════════════════════════════════

def breakdown_gate(bd: dict) -> dict:
    """
    Option B gate — all decisions from raw physics fields only.
    Reads _ratio, _stable, _sc_ratio, _beta from breakdown_stability.analyse().
    NO status strings from module. Thresholds live here, not in module.
    """
    warnings = []
    rejected = False
    reasons  = []

    ratio    = bd.get("_ratio", 0)
    ratio_r  = bd.get("_ratio_rough", 0)   # 2× roughness
    stable   = bd.get("_stable", True)
    sc_r     = bd.get("_sc_ratio", 0)
    R_max    = bd.get("_R_max_MOhm", 0)
    instab   = bd.get("_instability", False)
    sp_ratio = bd.get("_spacing_ratio")  # None = single emitter
    spacing_ok = sp_ratio is None or sp_ratio >= 50
    beta     = bd.get("_beta", 0)
    beta_exp = bd.get("_beta_expected", beta)
    Ebrk_Eavg= bd.get("_Ebreak_Eavg", 99)

    # Regime from centralised thresholds — single code path
    regime = ratio_to_regime(ratio)

    # ── Warnings (thresholds live here only) ──────────────────────────────────
    if ratio > 8:
        warnings.append(f"E_tip/E_break = {ratio:.1f} (safe: 3–8). "
                        f"With 2× roughness: {ratio_r:.1f}.")
    if not stable:
        warnings.append(f"Circuit unstable: R_ballast > R_max={R_max:.1f}MΩ.")
    if sc_r > 0.1:
        warnings.append(f"Space charge ΔV/V = {sc_r*100:.0f}% — field distortion.")
    if instab:
        warnings.append("dI/dV runaway risk — steeper than load line.")
    if not spacing_ok:
        warnings.append(f"Emitter spacing ratio={sp_ratio:.0f} < 50 — proximity boost significant.")
    if Ebrk_Eavg < 1.0:
        warnings.append(f"E_avg > E_break (ratio {1/Ebrk_Eavg:.2f}) — gap breakdown risk.")

    # ── Rejection (raw ratio thresholds only) ─────────────────────────────────
    if ratio > 15:
        rejected = True
        reasons.append(f"E_tip/E_break={ratio:.1f} > 15: arc/spark zone.")
    elif ratio_r > 15:   # roughness pushes into arc
        rejected = True
        reasons.append(f"With surface roughness, effective ratio={ratio_r:.1f} > 15.")

    return {
        "ACCEPTED":         not rejected,
        "REJECTED":         rejected,
        "regime":           regime,
        "rejection_reasons": reasons,
        "warnings":         warnings,
        "verdict": (
            f"✓ ACCEPTED — {regime}, ratio={ratio:.1f}×"
            if not rejected else
            f"✗ REJECTED — {reasons[0]}"
        ),
    }

# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PHYSICS VALIDATION — unified schema self-test")
    print("=" * 60)

    # Test 1: normalize_flag converts old schema
    old = {"level": "CRITICAL", "code": "X1", "message": "test"}
    new = normalize_flag(old)
    assert new["severity"] == "error" and "level" not in new
    print("✓ normalize_flag: CRITICAL → error")

    old2 = {"level": "INVALID", "code": "X2", "message": "test"}
    new2 = normalize_flag(old2)
    assert new2["severity"] == "error"
    print("✓ normalize_flag: INVALID → error")

    # Test 2: model_valid uses severity, not string matching
    flags = [flag("SOME_ARC_CODE", "error", "test")]
    valid = not any(f["severity"] == "error" for f in flags)
    assert not valid
    print("✓ model_valid: severity-based, not string-based")

    # Test 3: validate_system with conflict
    from electrostatics import TipEnhancement
    from corona_physics import CoronaPhysics
    from breakdown_stability import classify_and_gate
    import json

    tip   = TipEnhancement.hyperboloid_tip(20000, 0.1, 23)
    E = 3e6          # V/m (electric field)
    rho = 1e-6       # C/m³ (charge density)
    pressure = 101325

    corona = CoronaPhysics.ionization_from_field(E, rho, pressure)
    bkdn  = classify_and_gate(20000, 1e-4, 0.023, humidity_pct=80)

    sys_result = validate_system({
        "electrostatics": tip,
        "corona":         corona,
        "breakdown":      bkdn,
    })
    print(f"\n✓ validate_system: valid={sys_result['valid']} "
          f"proceed_to_module_5={sys_result['proceed_to_module_5']}")
    print(f"  summary: {sys_result['summary']}")
    for mod, flags in sys_result["per_module"].items():
        if flags:
            print(f"  [{mod}]")
            for f in flags:
                print(f"    [{f['severity']}] {f['code']}: {f['message'][:65]}")

    # Test 4: aggregate handles mixed old/new schema
    old_flag = {"level": "CRITICAL", "code": "OLD", "message": "old style"}
    new_flag = {"severity": "error",  "code": "NEW", "message": "new style"}
    agg = aggregate_validation({"flags": [old_flag, new_flag], "confidence": 0.5})
    assert agg["valid"] == False
    assert agg["confidence"] == 0.5
    assert len(agg["errors"]) == 2
    print(f"\n✓ aggregate_validation: mixed schema, valid={agg['valid']}, "
          f"errors={len(agg['errors'])}")

    print("\n✓ physics_validation.py self-test complete")
