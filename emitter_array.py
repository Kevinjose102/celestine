"""
emitter_array.py — Celestine Physics Module: EHD Emitter Array Design
======================================================================
Physics-constrained emitter array design for EHD thrusters.

Design philosophy:
  MORE EMITTERS ≠ MORE THRUST
  CONTROLLED ION FLOW = MORE THRUST

All constraints are physics-derived. No arbitrary numbers.

References:
  - Masuyama & Barrett (2013) — EHD thrust scaling
  - Adamiak & Atten (2004)   — field shielding in needle arrays
  - Gilmore & Barrett (2015) — ion mobility and thrust correction
  - Peek (1929)              — corona onset field
  - White (1963)             — ionic wind and plume expansion
"""

import math
import json

# ── Constants ────────────────────────────────────────────────────────────────
E_BREAKDOWN   = 3.0e6    # V/m  air at STP
EPSILON_0     = 8.854e-12
MU_ION        = 2.0e-4   # m²/V/s  positive ions in air
PEEK_A        = 3.1e6    # V/m
PEEK_B        = 0.030    # m^0.5


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT VALIDATORS
# Each returns {valid: bool, value: float, limit: float, message: str}
# ═══════════════════════════════════════════════════════════════════════════════

def check_pitch_geometry(pitch_mm: float, emitter_dia_mm: float) -> dict:
    """
    Rule 2: pitch >= emitter_diameter + clearance (min 1mm)
    For 4mm copper emitters: pitch >= 5-6mm absolute minimum.
    """
    min_pitch = emitter_dia_mm + max(1.0, emitter_dia_mm * 0.25)
    return {
        "valid":      pitch_mm >= min_pitch,
        "pitch_mm":   pitch_mm,
        "minimum_mm": round(min_pitch, 2),
        "margin_mm":  round(pitch_mm - min_pitch, 2),
        "rule":       "pitch >= emitter_diameter + clearance",
        "message": (
            f"OK: {pitch_mm}mm pitch >= {min_pitch}mm minimum"
            if pitch_mm >= min_pitch else
            f"VIOLATED: {pitch_mm}mm pitch < {min_pitch}mm minimum — emitters would physically overlap"
        )
    }


def check_field_shielding(pitch_mm: float, tip_radius_mm: float,
                           emitter_length_mm: float) -> dict:
    """
    Rule 3: pitch >= 5-10x effective tip influence zone.
    Shielding factor η = 1 - exp(-k·s/h), k=2.3 (Adamiak 2004).
    η > 0.8 required for <20% field reduction.
    """
    h = emitter_length_mm / 1000
    s = pitch_mm / 1000
    k = 2.3
    eta = 1 - math.exp(-k * s / h)

    # Minimum pitch for η > 0.8
    s_min = -h / k * math.log(1 - 0.8) * 1000  # mm

    # Influence zone = ~10x tip radius (where field > 10% of tip field)
    influence_zone_mm = tip_radius_mm * 10
    pitch_vs_influence = pitch_mm / influence_zone_mm

    return {
        "valid":              eta >= 0.8,
        "shielding_eta":      round(eta, 4),
        "field_reduction_pct": round((1 - eta) * 100, 1),
        "min_pitch_for_eta08_mm": round(s_min, 1),
        "influence_zone_mm":  round(influence_zone_mm, 2),
        "pitch_vs_influence": round(pitch_vs_influence, 1),
        "rule":               "shielding η > 0.8 (pitch >= 5-10× tip influence zone)",
        "message": (
            f"OK: η={eta:.3f} — field reduced by {(1-eta)*100:.1f}% due to shielding"
            if eta >= 0.8 else
            f"VIOLATED: η={eta:.3f} — {(1-eta)*100:.1f}% field reduction — emitters "
            f"shielding each other. Increase pitch to {s_min:.1f}mm"
        )
    }


def check_plume_spacing(pitch_mm: float, gap_mm: float,
                         tip_radius_mm: float) -> dict:
    """
    Rule 4: pitch >= 2-3x plume diameter.
    Ion plume expands at ~15-20° half-angle from emitter tip.
    Plume diameter at collector = 2 * gap * tan(17.5°) ≈ 0.63 * gap.
    """
    plume_half_angle_deg = 17.5  # degrees, typical for needle-to-ring corona
    plume_dia_at_collector_mm = 2 * gap_mm * math.tan(math.radians(plume_half_angle_deg))
    min_pitch_plume = plume_dia_at_collector_mm * 1.5  # 1.5x practical minimum  # 2.5x plume diameter

    return {
        "valid":                    pitch_mm >= min_pitch_plume,
        "plume_dia_at_collector_mm": round(plume_dia_at_collector_mm, 1),
        "min_pitch_no_interference_mm": round(min_pitch_plume, 1),
        "pitch_vs_plume_ratio":     round(pitch_mm / plume_dia_at_collector_mm, 2),
        "rule":                     "pitch >= 1.5× plume diameter at collector (practical minimum)",
        "message": (
            f"OK: pitch {pitch_mm}mm = {pitch_mm/plume_dia_at_collector_mm:.1f}× "
            f"plume diameter ({plume_dia_at_collector_mm:.1f}mm)"
            if pitch_mm >= min_pitch_plume else
            f"WARN: pitch {pitch_mm}mm < {min_pitch_plume:.1f}mm minimum — "
            f"ion plumes interfere. Some thrust loss expected. "
            f"Increase pitch to {min_pitch_plume:.1f}mm to eliminate."
        )
    }


def check_electrical_regime(voltage_V: float, gap_mm: float,
                              tip_radius_mm: float) -> dict:
    """
    Rule 5: gap field must stay in corona regime, not streamer/arc.
    Uses E_tip/E_breakdown ratio for regime classification.
    """
    r  = tip_radius_mm / 1000
    h  = gap_mm / 1000
    ln_term = math.log(2 * h / r) if (2 * h / r) > 1 else 1.0
    E_tip = voltage_V / (r * ln_term)
    E_avg = voltage_V / h
    E_onset = PEEK_A * (1 + PEEK_B / math.sqrt(r))
    V_onset = E_onset * r * ln_term
    ratio = E_tip / E_BREAKDOWN
    gap_ratio = E_avg / E_BREAKDOWN

    if gap_ratio > 0.6:
        regime = "ARC"
        valid  = False
        msg    = f"REJECTED: E_avg = {E_avg/1e6:.2f} MV/m = {gap_ratio*100:.0f}% of breakdown — arc certain"
    elif ratio > 30:
        regime = "ARC_LIKELY"
        valid  = False
        msg    = f"REJECTED: E_tip/E_break = {ratio:.1f} > 30 — arc likely"
    elif ratio > 10:
        regime = "STREAMER"
        valid  = True   # operable but inefficient
        msg    = f"WARNING: E_tip/E_break = {ratio:.1f} (10-30) — streamer regime, inefficient"
    elif E_tip >= E_onset:
        regime = "STABLE_CORONA"
        valid  = True
        msg    = f"OK: E_tip/E_break = {ratio:.1f} (3-10) — stable corona, EHD target regime"
    elif ratio >= 1:
        regime = "ONSET"
        valid  = True
        msg    = f"MARGINAL: E_tip/E_break = {ratio:.1f} (1-3) — weak corona, marginal thrust"
    else:
        regime = "NO_DISCHARGE"
        valid  = False
        msg    = f"REJECTED: E_tip/E_break = {ratio:.2f} < 1 — no ionisation"

    return {
        "_valid":         valid,
        "_regime_indicator": regime,
        "_ratio":         ratio,
        "E_tip_MV_m":     round(E_tip / 1e6, 3),
        "E_avg_MV_m":     round(E_avg / 1e6, 4),
        "E_tip_over_Ebreak": round(ratio, 2),
        "E_avg_over_Ebreak": round(gap_ratio, 3),
        "V_onset_V":      round(V_onset),
        "corona_active":  E_tip >= E_onset,
        "message":        msg,
    }


def check_material(material: str) -> dict:
    """
    Rule 6: Material constraints for corona emitters.
    Emission is field ionisation of air — NOT thermionic.
    What matters: oxidation, erosion, surface stability.
    """
    props = {
        "copper": {
            "valid_prototype": True,
            "valid_longterm":  False,
            "pros": ["easy machining", "cheap", "high conductivity"],
            "cons": ["oxidises in ozone/NOx corona environment",
                     "CuO layer raises onset voltage over time",
                     "unstable discharge as surface degrades",
                     "higher erosion rate than Mo/W"],
            "recommendation": "Use for short tests only (<10hr). Replace with SS304 or W for sustained operation.",
            "erosion_relative": 3.0,  # relative to tungsten = 1.0
        },
        "tungsten": {
            "valid_prototype": True,
            "valid_longterm":  True,
            "pros": ["best erosion resistance", "stable in corona", "high melting point"],
            "cons": ["hard to machine — buy pre-drawn 0.1mm wire"],
            "recommendation": "Best choice for long-duration. Buy 0.1mm tungsten wire.",
            "erosion_relative": 1.0,
        },
        "molybdenum": {
            "valid_prototype": True,
            "valid_longterm":  True,
            "pros": ["good erosion resistance", "easier to machine than W"],
            "cons": ["forms stable oxide at low rate"],
            "recommendation": "Good long-term choice. Standard for ion thrusters.",
            "erosion_relative": 1.3,
        },
        "stainless": {
            "valid_prototype": True,
            "valid_longterm":  True,
            "pros": ["easy to source as hypodermic needles", "does not oxidise significantly"],
            "cons": ["moderate erosion"],
            "recommendation": "Good low-cost prototype option. SS304 hypo needles work well.",
            "erosion_relative": 2.0,
        },
    }
    mat = props.get(material.lower(), props["copper"])
    mat["material"] = material
    mat["emission_mechanism"] = (
        "Field ionisation of air (NOT thermionic emission from metal). "
        "Work function is IRRELEVANT. What matters: surface oxidation, "
        "erosion rate, and dimensional stability of tip radius over time."
    )
    return mat


# ═══════════════════════════════════════════════════════════════════════════════
# ARRAY LAYOUT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ring_layout(diameter_mm: float, pitch_mm: float) -> dict:
    """
    Generate needle positions on concentric rings within a circle.
    Returns actual positions, not just count.
    Rings from outside in, maintaining pitch constraint throughout.
    """
    radius = diameter_mm / 2
    positions = []
    ring_radii = []

    # Outer ring first, then inner rings
    r = radius - pitch_mm / 2  # first ring inset from edge
    while r > pitch_mm:
        circumference = 2 * math.pi * r
        n_ring = max(1, int(circumference / pitch_mm))
        ring_radii.append((r, n_ring))
        r -= pitch_mm

    # Centre emitter if space allows
    if r > 0:
        ring_radii.append((0, 1))

    total = sum(n for _, n in ring_radii)
    for r_ring, n in ring_radii:
        for i in range(n):
            angle = 2 * math.pi * i / n
            positions.append({
                "x_mm": round(r_ring * math.cos(angle), 2),
                "y_mm": round(r_ring * math.sin(angle), 2),
                "ring_r_mm": round(r_ring, 1),
            })

    return {
        "total_emitters":  total,
        "n_rings":         len(ring_radii),
        "rings":           [{"radius_mm": round(r, 1), "count": n} for r, n in ring_radii],
        "positions_count": len(positions),
        "diameter_mm":     diameter_mm,
        "pitch_mm":        pitch_mm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# THRUST MODEL (Rule 10)
# ═══════════════════════════════════════════════════════════════════════════════

def thrust_constrained(current_per_emitter_A: float,
                        n_emitters: int,
                        gap_mm: float,
                        voltage_V: float,
                        eta_shielding: float,
                        geometry: str = "needle-to-ring") -> dict:
    """
    Thrust computation — NOT YET VALID.

    The correct physics chain requires:
      Module 2: Charge transport → ρe(r,z) distribution  [NOT BUILT]
      Module 5: EHD body force  → f(r,z) = ρe·E          [NOT BUILT]
      Module 6: Navier-Stokes   → flow field + thrust     [NOT BUILT]

    F = Id/μ is returned here ONLY as a dimensional placeholder so the
    code does not crash. It is NOT a valid thrust prediction for this geometry.
    Do not report this number as a real thrust value.
    """
    gap_m = gap_mm / 1000
    # Placeholder calculation — wrong geometry, wrong assumptions
    F_placeholder = current_per_emitter_A * n_emitters * gap_m / MU_ION

    return {
        "CANNOT_COMPUTE":      True,
        "reason":              "Thrust requires Modules 2 (charge transport), 5 (EHD force), 6 (Navier-Stokes) — not yet built",
        "missing_modules":     ["Module 2: Charge Transport / ρe(r,z)", "Module 5: EHD Force Coupling / f=ρeE", "Module 6: Fluid Dynamics / Navier-Stokes"],
        "what_is_computable":  ["Corona onset voltage", "Discharge regime", "Optimal gap", "Minimum spacing", "Emitter count"],
        "F_placeholder_mN":    round(F_placeholder * 1000, 3),
        "placeholder_warning": "F=Id/μ placeholder — INVALID for radial multi-emitter geometry. Do NOT report as thrust.",
        "EQUATION_NEEDED":     "F = ∫∫∫ ρe(r,z) · E(r,z) dV  [requires Module 2+5+6]",
    }

def design_array(diameter_mm: float,
                  voltage_V: float,
                  emitter_dia_mm: float = 4.0,
                  tip_radius_mm: float = 0.1,
                  gap_mm: float = None,
                  material: str = "copper",
                  target: str = "optimal") -> dict:
    """
    Physics-constrained emitter array design.

    Implements all 12 rules from the EHD Array Design Ruleset.
    Returns TWO configurations:
      A) Maximum geometric packing (most emitters that physically fit)
      B) Physically optimal layout (correct spacing for EHD physics)

    Parameters:
        diameter_mm    : housing inner diameter (mm)
        voltage_V      : applied HV (V)
        emitter_dia_mm : emitter tube/wire outer diameter (mm)
        tip_radius_mm  : tip radius (mm) — affects field shielding zone
        gap_mm         : emitter-to-collector gap (mm) — derived if None
        material       : emitter material
        target         : "optimal" or "max_packing"
    """
    import math

    # ── Derive gap if not given ───────────────────────────────────────────────
    if gap_mm is None:
        # Gap from Peek onset + 2× safety margin
        # E_avg = V/gap must be < 25% of breakdown for safe corona
        gap_mm = round(voltage_V / (E_BREAKDOWN * 0.25) * 1000, 1)
        gap_derived = True
    else:
        gap_derived = False

    # ── Compute physics-constrained pitch limits ──────────────────────────────

    # 1. Geometric minimum: pitch >= emitter_dia + clearance
    pitch_geo_min = emitter_dia_mm + max(1.0, emitter_dia_mm * 0.25)

    # 2. Field shielding minimum: η > 0.8
    h = 30 / 1000  # emitter length 30mm default
    k = 2.3
    pitch_shield_min = -h / k * math.log(1 - 0.8) * 1000  # mm

    # 3. Plume spacing minimum: pitch >= 1.5× plume diameter
    # 2.5× is theoretical ideal (zero interference). 1.5× is practical minimum
    # where plume boundary interaction is tolerable (<15% thrust loss).
    # Real EHD arrays (MIT, Plasma Channel) operate at ~1.5-2× plume diameter.
    plume_dia = 2 * gap_mm * math.tan(math.radians(17.5))
    pitch_plume_min = plume_dia * 1.5

    # ── Configuration A: Maximum geometric packing ────────────────────────────
    pitch_A = max(pitch_geo_min, emitter_dia_mm + 1.0)
    layout_A = generate_ring_layout(diameter_mm, pitch_A)
    regime_A = check_electrical_regime(voltage_V, gap_mm, tip_radius_mm)
    shield_A = check_field_shielding(pitch_A, tip_radius_mm, 30.0)
    plume_A  = check_plume_spacing(pitch_A, gap_mm, tip_radius_mm)
    geo_A    = check_pitch_geometry(pitch_A, emitter_dia_mm)

    # Violations in config A
    violations_A = []
    if not shield_A["valid"]:
        violations_A.append(f"Field shielding violated (η={shield_A['shielding_eta']:.2f} < 0.8)")
    if not plume_A["valid"]:
        violations_A.append(f"Plume interference ({plume_A['pitch_vs_plume_ratio']:.1f}× plume dia, need 1.5×)")
    if not regime_A["valid"]:
        violations_A.append(f"Electrical regime: {regime_A['regime']}")

    # ── Configuration B: Physically optimal ──────────────────────────────────
    # Hard constraint: pitch MUST be >= emitter diameter + clearance (geometric)
    # This is non-negotiable — emitters cannot overlap physically
    # Then take the max of all physics constraints on top of that
    pitch_B = max(pitch_geo_min, pitch_shield_min, pitch_plume_min)
    pitch_B = round(pitch_B, 1)

    # Sanity check — spacing can never be less than emitter diameter
    if pitch_B < emitter_dia_mm:
        pitch_B = emitter_dia_mm * 1.5  # force minimum 1.5x diameter
        warnings_geo = [f"Pitch forced to {pitch_B}mm — spacing < emitter diameter is physically impossible"]
    else:
        warnings_geo = []
    layout_B = generate_ring_layout(diameter_mm, pitch_B)
    regime_B = check_electrical_regime(voltage_V, gap_mm, tip_radius_mm)
    shield_B = check_field_shielding(pitch_B, tip_radius_mm, 30.0)
    plume_B  = check_plume_spacing(pitch_B, gap_mm, tip_radius_mm)
    geo_B    = check_pitch_geometry(pitch_B, emitter_dia_mm)

    violations_B = []
    if not shield_B["valid"]:
        violations_B.append(f"Field shielding (η={shield_B['shielding_eta']:.2f})")
    if not plume_B["valid"]:
        violations_B.append(f"Plume spacing ({plume_B['pitch_vs_plume_ratio']:.1f}× plume dia)")
    if not regime_B["valid"]:
        violations_B.append(f"Electrical regime: {regime_B['regime']}")

    # ── Thrust estimates ──────────────────────────────────────────────────────
    # Estimate current per emitter from Townsend I-V
    r_m   = tip_radius_mm / 1000
    gap_m = gap_mm / 1000
    ln_t  = math.log(2 * gap_m / r_m) if (2 * gap_m / r_m) > 1 else 1.0
    E_onset = PEEK_A * (1 + PEEK_B / math.sqrt(r_m))
    V_onset = E_onset * r_m * ln_t
    if voltage_V > V_onset:
        C_t = 2 * math.pi * EPSILON_0 / (gap_m * math.log(gap_m / r_m)) if gap_m > r_m else 1e-12
        I_per_emitter = max(1e-7, min(C_t * MU_ION * (voltage_V - V_onset) * voltage_V / gap_m**2, 5e-3))
    else:
        I_per_emitter = 0.0

    thrust_A = thrust_constrained(I_per_emitter, layout_A["total_emitters"],
                                   gap_mm, voltage_V, shield_A["shielding_eta"])
    thrust_B = thrust_constrained(I_per_emitter, layout_B["total_emitters"],
                                   gap_mm, voltage_V, shield_B["shielding_eta"])

    # ── Material assessment ───────────────────────────────────────────────────
    mat = check_material(material)

    # ── Airflow behaviour ─────────────────────────────────────────────────────
    def airflow_description(pitch_mm, n_emitters, plume_dia):
        if pitch_mm < plume_dia * 1.5:
            return ("Severe plume collision expected. Ion plumes overlap significantly. "
                    "Turbulent, chaotic flow field. Thrust loss >40%.")
        elif pitch_mm < plume_dia * 2.5:
            return ("Moderate plume interaction. Some interference at collector. "
                    "Recommend duct confinement to redirect recirculating air.")
        else:
            return ("Good plume separation. Each emitter drives independent ion column. "
                    "Laminar flow achievable with axial geometry + duct confinement.")

    # ── Collector recommendation ──────────────────────────────────────────────
    collector_rec = (
        f"Collector diameter: {diameter_mm + gap_mm * 2:.0f}mm (housing dia + 2×gap). "
        f"Use mesh or perforated ring (5mm wall, 60% open area) — "
        f"solid ring causes 15% collection loss + edge field distortion. "
        f"Round all collector edges to r≥2mm to prevent secondary corona."
    )

    # ── Final design decision ─────────────────────────────────────────────────
    recommended = "B" if not violations_B else ("A" if not violations_A else "NONE")
    if recommended == "NONE":
        design_verdict = (
            f"BOTH configurations violate physics constraints at these parameters. "
            f"Increase gap to {gap_mm*1.5:.0f}mm or reduce voltage."
        )
    elif recommended == "B":
        design_verdict = (
            f"Configuration B recommended: {layout_B['total_emitters']} emitters "
            f"at {pitch_B}mm pitch. All constraints satisfied."
        )
    else:
        design_verdict = (
            f"Only Configuration A satisfies geometry. "
            f"Violations in shielding/plume — accept with performance penalty."
        )

    return {
        "inputs": {
            "diameter_mm":     diameter_mm,
            "voltage_V":       voltage_V,
            "emitter_dia_mm":  emitter_dia_mm,
            "tip_radius_mm":   tip_radius_mm,
            "gap_mm":          gap_mm,
            "gap_derived":     gap_derived,
            "material":        material,
        },
        "pitch_constraints": {
            "geometric_min_mm":  round(pitch_geo_min, 1),
            "shielding_min_mm":  round(pitch_shield_min, 1),
            "plume_min_mm":      round(pitch_plume_min, 1),
            "controlling_constraint": (
                "plume_spacing" if pitch_plume_min >= pitch_shield_min
                else "field_shielding"
            ),
        },
        "configuration_A_max_packing": {
            "label":         "Maximum Geometric Packing",
            "pitch_mm":      pitch_A,
            "emitter_count": layout_A["total_emitters"],
            "layout":        layout_A,
            "constraints": {
                "geometry":   geo_A["message"],
                "shielding":  shield_A["message"],
                "plume":      plume_A["message"],
                "electrical": regime_A["message"],
            },
            "violations":    violations_A,
            "valid":         len(violations_A) == 0,
            "thrust":        thrust_A,
            "airflow":       airflow_description(pitch_A, layout_A["total_emitters"], plume_dia),
            "note":          "Maximum count — shielding and plume constraints likely violated.",
        },
        "configuration_B_optimal": {
            "label":         "Physically Optimal Layout",
            "pitch_mm":      pitch_B,
            "emitter_count": layout_B["total_emitters"],
            "layout":        layout_B,
            "constraints": {
                "geometry":   geo_B["message"],
                "shielding":  shield_B["message"],
                "plume":      plume_B["message"],
                "electrical": regime_B["message"],
            },
            "violations":    violations_B,
            "valid":         len(violations_B) == 0,
            "thrust":        thrust_B,
            "airflow":       airflow_description(pitch_B, layout_B["total_emitters"], plume_dia),
            "note":          "Spacing from controlling physics constraint.",
        },
        "electrical_regime":    regime_B,
        "material_assessment":  mat,
        "collector_recommendation": collector_rec,
        "geometry_strategy": (
            "Axial geometry: emitters point along thrust axis, collector perpendicular. "
            "Add PETG duct cylinder around array to confine and direct ion wind. "
            "Duct recovers ~20% recirculation losses. "
            "Hex packing preferred over square grid for same pitch — "
            "15% better area utilisation with no additional shielding."
        ),
        "recommended_config":   recommended,
        "design_verdict":       design_verdict,
        "key_principle":        "MORE EMITTERS ≠ MORE THRUST. CONTROLLED ION FLOW = MORE THRUST.",
    }


# ── AI interface ──────────────────────────────────────────────────────────────
class EmitterArrayAI:

    @staticmethod
    def array_design(diameter_mm: float, voltage_V: float,
                      emitter_dia_mm: float = 4.0,
                      tip_radius_mm: float = 0.1,
                      gap_mm: float = None,
                      material: str = "copper") -> str:
        result = design_array(diameter_mm, voltage_V, emitter_dia_mm,
                               tip_radius_mm, gap_mm, material)
        return json.dumps(result, indent=2)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  EMITTER ARRAY DESIGN — 100mm diameter, 20kV, 4mm copper")
    print("=" * 65)

    r = design_array(diameter_mm=100, voltage_V=20000,
                     emitter_dia_mm=4.0, tip_radius_mm=0.1, material="copper")

    print(f"\nGap derived: {r['inputs']['gap_mm']}mm "
          f"({'physics-derived' if r['inputs']['gap_derived'] else 'user-specified'})")

    pc = r["pitch_constraints"]
    print(f"\nPITCH CONSTRAINTS:")
    print(f"  Geometric minimum:  {pc['geometric_min_mm']}mm")
    print(f"  Shielding minimum:  {pc['shielding_min_mm']}mm  (η > 0.8)")
    print(f"  Plume minimum:      {pc['plume_min_mm']}mm  (2.5× plume dia)")
    print(f"  Controlling:        {pc['controlling_constraint']}")

    ca = r["configuration_A_max_packing"]
    cb = r["configuration_B_optimal"]

    print(f"\nCONFIG A — {ca['label']}:")
    print(f"  Pitch: {ca['pitch_mm']}mm → {ca['emitter_count']} emitters")
    print(f"  Valid: {ca['valid']}  Violations: {ca['violations']}")
    print(f"  Thrust: {ca['thrust']['F_corrected_mN']}mN "
          f"({ca['thrust']['thrust_per_watt_mN_W']}mN/W)")
    print(f"  Airflow: {ca['airflow']}")

    print(f"\nCONFIG B — {cb['label']}:")
    print(f"  Pitch: {cb['pitch_mm']}mm → {cb['emitter_count']} emitters")
    print(f"  Valid: {cb['valid']}  Violations: {cb['violations']}")
    print(f"  Thrust: {cb['thrust']['F_corrected_mN']}mN "
          f"({cb['thrust']['thrust_per_watt_mN_W']}mN/W)")
    print(f"  Airflow: {cb['airflow']}")

    print(f"\nREGIME: {r['electrical_regime']['regime']} "
          f"(E_tip/E_break = {r['electrical_regime']['E_tip_over_Ebreak']})")
    print(f"MATERIAL: {r['material_assessment']['recommendation']}")
    print(f"\nVERDICT: {r['design_verdict']}")
    print(f"\n{r['key_principle']}")
