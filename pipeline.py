"""
pipeline.py — Celestine EHD Physics Pipeline
GPT-120B → pipeline → physics → validate → DeepSeek
"""
from poisson_charge_coupling import PoissonChargeCoupler
from physics_validation import validate_system, flag, ratio_to_regime
from ehd_force import estimate_thrust, body_force_integral

# Ion mobility in air at STP (m²/V·s) — positive ions

def run_ehd_case(voltage_V: float,
                 gap_m: float,
                 tip_radius_m: float,
                 pressure_Pa: float = 101325,
                 Nr: int = 30,
                 Nz: int = 45) -> dict:
    """Raw physics only. Returns stable output contract."""
    solver = PoissonChargeCoupler(Nr=Nr, Nz=Nz)
    r = solver.solve(
        voltage_V    = voltage_V,
        tip_radius_m = tip_radius_m,
        gap_m        = gap_m,
        pressure_Pa  = pressure_Pa,
    )

    I_uA = r["current"]["I_collector_uA"]
    I_A  = I_uA * 1e-6

    current_note = None
    if I_uA == 0.0:
        current_note = "Grid under-resolved — tip smaller than grid cell. Current unreliable."

    grids = r.get("grids")
    if grids and grids.get("rho"):
        thrust = body_force_integral(
            grids["rho"], grids["Er"], grids["Ez"],
            grids["r"],   grids["z"])
    else:
        thrust = estimate_thrust(I_A, gap_m, pressure_Pa)

    return {
        "E": {
            "E_peak_MV_m":               r["field"]["E_peak_MV_m"],
            "E_avg_MV_m":                r["field"]["E_avg_MV_m"],
            "E_tip_coupled_MV_m":        r["field"].get("E_peak_MV_m", 0),  # approximation: E_peak used as E_tip proxy; emitter-node extraction pending
            "E_tip_laplace_MV_m":        r["field"].get("E_tip_uncoupled_MV_m", 0),
            "space_charge_reduction_pct": r["field"].get("space_charge_field_reduction_pct", 0),
        },
        "rho": {
            "rho_max_nC_m3": r["charge"]["rho_max_nC_m3"],
            "Q_total_nC":    r["charge"]["Q_total_nC"],
        },
        "current": {
            "I_collector_uA": I_uA,
            "I_collector_mA": r["current"]["I_collector_mA"],
            "note":           current_note,
        },
        "thrust": thrust,
        "solver": {
            "converged":        r["solver"]["converged"],
            "outer_iterations": r["solver"]["outer_iterations"],
            "grid":             r["solver"]["grid"],
        },
        "performance": {
            "power_W":        voltage_V * I_A,
            "thrust_per_watt_mN_W": (thrust["F_mN"] / (voltage_V * I_A))
                              if I_A > 1e-9 and thrust["F_mN"] is not None else None,
        "power_note":     "electrical input power — losses not modeled; power from current, force from field (consistent when ρ solved)",
        },
        "inputs": {
            "voltage_V":    voltage_V,
            "gap_m":        gap_m,
            "tip_radius_m": tip_radius_m,
            "pressure_Pa":  pressure_Pa,
        },
    }


def run_and_validate(voltage_V: float,
                     gap_m: float,
                     tip_radius_m: float,
                     pressure_Pa: float = 101325) -> dict:
    """Run physics + validate + summarise. Send to DeepSeek only if proceed=True."""
    result = run_ehd_case(voltage_V, gap_m, tip_radius_m, pressure_Pa)

    E_break = 3e6
    E_tip_v = result["E"]["E_tip_coupled_MV_m"] * 1e6
    E_avg_v = result["E"]["E_avg_MV_m"] * 1e6
    ratio   = E_tip_v / E_break if E_break else 0

    flat = {
        "_E_tip":    E_tip_v,
        "_E_avg":    E_avg_v,
        "_E_break":  E_break,
        "_ratio":    ratio,
        "_h_r":      gap_m / tip_radius_m if tip_radius_m else 0,
        "_gap_mm":   gap_m * 1000,
        "_p_ratio":  pressure_Pa / 101325,
    }

    gate = validate_system({"breakdown": flat})

    if not result["solver"]["converged"]:
        gate["errors"].append(flag("SOLVER_NOT_CONVERGED", "error",
            "Poisson solver did not converge."))
        gate["valid"] = False

    if result["current"]["I_collector_uA"] == 0.0:
        gate["warnings"].append(flag("ZERO_CURRENT", "warning",
            "Current = 0 — tip under-resolved. Thrust estimate = 0 (unreliable)."))

    regime = ratio_to_regime(ratio)

    # Fix 4: thrust consistency — arc regime = thrust invalid
    if regime == "ARC LIKELY":
        result["thrust"]["F_N"]   = None
        result["thrust"]["F_mN"]  = None
        result["thrust"]["note"]  = "Thrust invalid — arc/spark regime"
        gate["warnings"].append(flag("THRUST_INVALID_REGIME", "warning",
            f"Regime={regime}: thrust estimate meaningless in arc zone."))

    summary = {
        "E_peak_MV_m": result["E"]["E_peak_MV_m"],
        "E_avg_MV_m":  result["E"]["E_avg_MV_m"],
        "ratio":       round(ratio, 2),
        "current_uA":  result["current"]["I_collector_uA"],
        "thrust_mN":   round(result["thrust"]["F_mN"], 6) if result["thrust"]["F_mN"] is not None else None,
        "thrust_note":  "F_z sign correct — negative means net force toward collector (normal EHD operation)",
        "power_W":     result["performance"]["power_W"],
        "thrust_per_watt_mN_W": result["performance"]["thrust_per_watt_mN_W"],
        "regime":      regime,
        "converged":   result["solver"]["converged"],
        "valid":       gate["valid"],
        "warnings":    [f["code"] for f in gate["warnings"]],
        "errors":      [f["code"] for f in gate["errors"]],
    }

    return {
        "physics":  result,
        "gate":     gate,
        "summary":  summary,
        "proceed":  gate["valid"],
    }


if __name__ == "__main__":
    out = run_and_validate(20000, 0.02, 1e-4)
    s = out["summary"]
    print(f"Valid:     {out['proceed']}")
    print(f"Regime:    {s['regime']}")
    print(f"E_peak:    {s['E_peak_MV_m']} MV/m")
    print(f"Ratio:     {s['ratio']}")
    print(f"Current:   {s['current_uA']} μA")
    print(f"Thrust:    {s['thrust_mN']} mN  (order-of-magnitude estimate)")
    print(f"Errors:    {s['errors']}")
    print(f"Warnings:  {s['warnings']}")
