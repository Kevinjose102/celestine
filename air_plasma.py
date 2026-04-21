"""
air_plasma.py
=============
Air Plasma Chemistry Module

Physics of ionizing air (N2/O2 mixture) instead of Xenon.
This is what makes your Plasma Channel / Integza style thruster work.

Key differences from Xenon:
  1. Higher ionization energy (N: 14.53 eV, O: 13.62 eV vs Xe: 12.13 eV)
  2. Molecular species → energy lost to dissociation and vibration
  3. Oxygen creates NEGATIVE ions (O-) — useful for self-neutralization
  4. Mixed species beam → different thrust calculation
  5. Much lower ionization efficiency vs Xe at same power

Air composition (dry, sea level):
  N2: 78.09%
  O2: 20.95%
  Ar:  0.93%
  CO2: 0.04% (negligible)
"""

import math
import json
from typing import Optional

from propellants_db import PROPELLANTS as _PROP_DB  # noqa: E402

E_CHARGE = 1.602e-19
K_BOLTZ  = 1.381e-23
ME       = 9.109e-31
G0       = 9.80665


# ─────────────────────────────────────────────
# AIR COMPOSITION DATABASE
# ─────────────────────────────────────────────

AIR_COMPOSITION = {
    "sea_level": {
        "N2": 0.7809, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0003,
        "altitude_m": 0, "pressure_Pa": 101325, "temp_K": 288,
    },
    "10km": {
        "N2": 0.7809, "O2": 0.2095, "Ar": 0.0093,
        "altitude_m": 10000, "pressure_Pa": 26500, "temp_K": 223,
    },
    "stratosphere": {
        "N2": 0.7809, "O2": 0.2095, "Ar": 0.0093,
        "altitude_m": 30000, "pressure_Pa": 1200, "temp_K": 227,
    },
}

# Ionization and collision data for air species
# All energies in eV, cross sections in m²
AIR_SPECIES = {
    "N2": {
        "mass_kg":          4.652e-26,
        "ionization_eV":    15.58,     # N2 molecular ionization
        "dissoc_eV":         9.76,     # N2 dissociation energy
        "vibration_eV":      0.29,     # vibrational spacing
        "excitation_eV":     6.17,     # first electronic excitation
        "k_iz_prefactor":    1.2e-14,
        "k_dissoc":          0.8e-14,
        "k_excit":           3.0e-14,
        "peak_cross_eV":     100.0,    # cross section peaks here
        "fraction_air":      0.7809,
    },
    "O2": {
        "mass_kg":          5.312e-26,
        "ionization_eV":    12.07,     # O2 molecular ionization
        "dissoc_eV":         5.12,     # O2 dissociation energy
        "vibration_eV":      0.19,
        "excitation_eV":     4.50,
        "k_iz_prefactor":    1.5e-14,
        "k_dissoc":          1.2e-14,
        "k_excit":           2.5e-14,
        "peak_cross_eV":     80.0,
        "fraction_air":      0.2095,
        "attachment_eV":     0.0,      # electron attachment threshold (O-)
        "k_attach":          1.0e-16,  # O- formation rate
    },
    "N":  {
        "mass_kg":          2.326e-26,
        "ionization_eV":    14.53,
        "k_iz_prefactor":   2.0e-14,
        "fraction_air":     0.0,       # formed by dissociation
    },
    "O":  {
        "mass_kg":          2.656e-26,
        "ionization_eV":    13.62,
        "k_iz_prefactor":   2.2e-14,
        "fraction_air":     0.0,       # formed by dissociation
        "attachment_eV":    0.0,
        "k_attach":         3.0e-15,   # O- formation rate (higher than O2)
    },
    "Ar": {
        # Pulled from unified propellants_db — avoids duplication
        "mass_kg":          _PROP_DB["Argon"]["mass_kg"],
        "ionization_eV":    _PROP_DB["Argon"]["ionization_eV"],
        "k_iz_prefactor":   _PROP_DB["Argon"]["k_iz_prefactor"],
        "fraction_air":     0.0093,
    },
}


# ─────────────────────────────────────────────
# 1. AIR IONIZATION MODEL
# ─────────────────────────────────────────────

class AirIonization:
    """
    Models ionization of air (N2/O2 mixture).

    Unlike Xenon (single species, single ionization path),
    air has multiple competing processes:
    - Direct ionization: N2 + e → N2+ + 2e
    - Dissociative ionization: N2 + e → N+ + N + 2e
    - Attachment: O2 + e → O- + O (REMOVES electrons)
    - Dissociation: N2 + e → 2N + e (wastes energy, no ion)

    The competition between these determines efficiency.
    """

    @staticmethod
    def ionization_rate(Te_eV: float, species: str) -> float:
        """
        Ionization rate coefficient k_iz (m³/s) for a given species.
        k_iz ≈ k_prefactor · exp(-E_iz / Te)
        """
        sp = AIR_SPECIES.get(species)
        if not sp:
            return 0.0
        return sp["k_iz_prefactor"] * math.exp(-sp["ionization_eV"] / max(Te_eV, 0.5))

    @staticmethod
    def dissociation_rate(Te_eV: float, species: str) -> float:
        """Dissociation rate coefficient (m³/s). Wastes power."""
        sp = AIR_SPECIES.get(species)
        if not sp or "k_dissoc" not in sp:
            return 0.0
        return sp["k_dissoc"] * math.exp(-sp["dissoc_eV"] / max(Te_eV, 0.5))

    @staticmethod
    def attachment_rate(Te_eV: float, species: str = "O2") -> float:
        """
        Electron attachment rate (O- formation) (m³/s).
        This REMOVES free electrons — reduces ionization efficiency.
        BUT the O- ions can provide self-neutralization of the beam.
        """
        sp = AIR_SPECIES.get(species)
        if not sp or "k_attach" not in sp:
            return 0.0
        # Attachment peaks at low electron energies (~1-3 eV)
        peak_factor = math.exp(-(Te_eV - 1.0)**2 / 4.0)
        return sp["k_attach"] * peak_factor

    @staticmethod
    def effective_ionization_rate_air(Te_eV: float,
                                       composition: dict = None) -> dict:
        """
        Effective ionization rate for air mixture.
        Accounts for all species and competing processes.

        Returns rates for each species and net efficiency.
        """
        if composition is None:
            composition = {"N2": 0.7809, "O2": 0.2095, "Ar": 0.0093}

        results      = {}
        total_iz     = 0.0
        total_dissoc = 0.0
        total_attach = 0.0

        for species, fraction in composition.items():
            if fraction == 0 or species not in AIR_SPECIES:
                continue

            k_iz     = AirIonization.ionization_rate(Te_eV, species)
            k_d      = AirIonization.dissociation_rate(Te_eV, species)
            k_a      = AirIonization.attachment_rate(Te_eV, species) if species in ["O2", "O"] else 0

            results[species] = {
                "fraction":            fraction,
                "k_iz_m3_s":           round(k_iz, 8),
                "k_dissoc_m3_s":       round(k_d, 8),
                "k_attach_m3_s":       round(k_a, 10),
                "weighted_k_iz":       round(k_iz * fraction, 8),
            }
            total_iz     += k_iz * fraction
            total_dissoc += k_d  * fraction
            total_attach += k_a  * fraction

        # Net ionization efficiency
        # Efficiency = ionization / (ionization + dissociation + attachment)
        total_processes = total_iz + total_dissoc + total_attach
        efficiency = total_iz / total_processes if total_processes > 0 else 0

        # O- fraction in beam (self-neutralization potential)
        o_neg_fraction = total_attach / (total_iz + total_attach) if (total_iz + total_attach) > 0 else 0

        # Compare to Xenon at same Te
        k_iz_xe = 1.8e-14 * math.exp(-12.13 / max(Te_eV, 0.5))
        ratio_vs_xe = total_iz / k_iz_xe if k_iz_xe > 0 else 0

        warnings = []
        if efficiency < 0.3:
            warnings.append(
                f"WARNING: Air ionization efficiency {efficiency*100:.1f}% is very low. "
                f"Most electron energy goes to dissociation and excitation, not ionization. "
                f"Increase Te to 15-20 eV to improve efficiency."
            )
        if Te_eV < 14:
            warnings.append(
                f"WARNING: Te={Te_eV}eV is below N2 ionization energy (15.58 eV). "
                f"N2 will barely ionize. O2 ionizes better at this temperature (12.07 eV). "
                f"Raise Te or rely primarily on O2 ionization."
            )

        return {
            "Te_eV":                   Te_eV,
            "species_rates":           results,
            "total_iz_rate":           round(total_iz, 8),
            "total_dissoc_rate":       round(total_dissoc, 8),
            "total_attach_rate":       round(total_attach, 10),
            "ionization_efficiency":   round(efficiency, 4),
            "ionization_pct":          round(efficiency * 100, 1),
            "o_minus_fraction_pct":    round(o_neg_fraction * 100, 2),
            "self_neutralization":     o_neg_fraction > 0.05,
            "ratio_vs_xenon":          round(ratio_vs_xe, 3),
            "xenon_comparison": (
                f"Air ionization rate is {ratio_vs_xe:.2f}x that of Xe at Te={Te_eV}eV. "
                f"{'Air is competitive.' if ratio_vs_xe > 0.5 else 'Air is significantly less efficient than Xe.'}"
            ),
            "warnings": warnings,
        }

    @staticmethod
    def energy_cost_per_ion(Te_eV: float) -> dict:
        """
        Energy cost to produce one ion in air vs Xenon (eV/ion).
        Lower is better. Xe ≈ 200-300 eV/ion for well-designed thruster.
        Air is typically 500-2000 eV/ion due to energy losses.

        This is the most important metric for air-breathing thruster efficiency.
        """
        composition = {"N2": 0.7809, "O2": 0.2095, "Ar": 0.0093}
        rates = AirIonization.effective_ionization_rate_air(Te_eV, composition)

        # Energy going into each process (per electron per collision)
        E_iz_avg    = 0.7809 * 15.58 + 0.2095 * 12.07  # weighted ionization cost
        E_dissoc    = 0.7809 * 9.76 + 0.2095 * 5.12    # dissociation cost
        E_excit     = 0.7809 * 6.17 + 0.2095 * 4.50    # excitation cost

        iz_frac     = rates["ionization_efficiency"]
        dissoc_frac = rates["total_dissoc_rate"] / max(
            rates["total_iz_rate"] + rates["total_dissoc_rate"], 1e-30)

        # Total energy per ionization event
        cost_air = E_iz_avg + (dissoc_frac / max(iz_frac, 1e-6)) * E_dissoc

        # Xenon cost for comparison
        cost_xe = 12.13 * 1.5  # Xe ionization + sheath losses

        return {
            "energy_cost_air_eV_per_ion":    round(cost_air, 1),
            "energy_cost_xenon_eV_per_ion":  round(cost_xe, 1),
            "air_vs_xe_ratio":               round(cost_air / cost_xe, 2),
            "interpretation": (
                f"Air requires {cost_air:.0f} eV per ion vs {cost_xe:.0f} eV for Xe. "
                f"Air is {cost_air/cost_xe:.1f}x less efficient in energy use. "
                f"This is fundamental — air will always need more power than Xe for same thrust."
            ),
        }


# ─────────────────────────────────────────────
# 2. AIR THRUSTER PERFORMANCE
# ─────────────────────────────────────────────

class AirThrusterPerformance:
    """
    Thrust and Isp for air-breathing ion thruster.

    Unlike Xe thrusters, the beam contains multiple species:
    N2+, N+, O2+, O+, O- (negative ions provide self-neutralization)

    Mixed beam thrust = sum of individual species contributions.
    """

    # Average ion mass for air at different altitudes
    MEAN_ION_MASS = {
        "sea_level":     4.81e-26,   # ~0.78 N2+ + 0.21 O2+
        "stratosphere":  4.75e-26,   # similar composition
        "VLEO_200km":    2.67e-26,   # mostly atomic O at 200km
    }

    @staticmethod
    def exhaust_velocity(net_voltage_V: float,
                          altitude: str = "sea_level") -> float:
        """
        Ion exhaust velocity for air propellant (m/s).
        v_ex = sqrt(2·e·V_net / m_ion_avg)
        """
        mi  = AirThrusterPerformance.MEAN_ION_MASS.get(altitude, 4.81e-26)
        return math.sqrt(2 * E_CHARGE * net_voltage_V / mi)

    @staticmethod
    def thrust_mixed_beam(beam_current_A: float,
                           net_voltage_V: float,
                           n2_fraction: float = 0.7809,
                           o2_fraction: float = 0.2095,
                           o_neg_fraction: float = 0.05) -> dict:
        """
        Thrust from mixed N2+/O2+/O- beam.

        N2+ and O2+ contribute forward thrust.
        O- ions accelerated backward contribute FORWARD thrust too
        (they're negative ions accelerated in opposite direction).

        Parameters:
            beam_current_A   : total positive ion beam current (A)
            net_voltage_V    : net accelerating voltage (V)
            n2_fraction      : fraction of beam that is N2+
            o2_fraction      : fraction of beam that is O2+
            o_neg_fraction   : fraction of negative O- ions
        """
        # Individual ion masses
        m_N2 = AIR_SPECIES["N2"]["mass_kg"]
        m_O2 = AIR_SPECIES["O2"]["mass_kg"]
        m_O  = AIR_SPECIES["O"]["mass_kg"]

        # Exhaust velocities for each species
        v_N2 = math.sqrt(2 * E_CHARGE * net_voltage_V / m_N2)
        v_O2 = math.sqrt(2 * E_CHARGE * net_voltage_V / m_O2)
        v_O  = math.sqrt(2 * E_CHARGE * net_voltage_V / m_O)

        # Mass flow for each species from beam current
        mdot_N2 = beam_current_A * n2_fraction * m_N2 / E_CHARGE
        mdot_O2 = beam_current_A * o2_fraction * m_O2 / E_CHARGE
        # O- contributes separately
        I_neg   = beam_current_A * o_neg_fraction
        mdot_O_neg = I_neg * m_O / E_CHARGE

        # Thrust contributions
        F_N2    = mdot_N2 * v_N2
        F_O2    = mdot_O2 * v_O2
        F_O_neg = mdot_O_neg * v_O  # negative ions also thrust forward

        F_total  = F_N2 + F_O2 + F_O_neg
        mdot_total = mdot_N2 + mdot_O2 + mdot_O_neg
        v_eff   = F_total / mdot_total if mdot_total > 0 else 0
        Isp     = v_eff / G0

        return {
            "thrust_total_mN":      round(F_total * 1000, 4),
            "thrust_N2_mN":         round(F_N2 * 1000, 4),
            "thrust_O2_mN":         round(F_O2 * 1000, 4),
            "thrust_O_neg_mN":      round(F_O_neg * 1000, 4),
            "effective_v_ex_m_s":   round(v_eff),
            "Isp_s":                round(Isp),
            "self_neutralized":     o_neg_fraction > 0.05,
            "neutralizer_needed":   o_neg_fraction < 0.05,
            "note": (
                "O- ions provide partial self-neutralization. "
                "External neutralizer may not be needed."
                if o_neg_fraction > 0.05 else
                "Insufficient O- for self-neutralization. Need external neutralizer."
            )
        }

    @staticmethod
    def compare_air_vs_xenon(net_voltage_V: float,
                              beam_current_A: float,
                              power_W: float) -> dict:
        """
        Direct performance comparison: air vs Xenon at same power.
        """
        # Air performance
        mi_air  = 4.81e-26
        v_air   = math.sqrt(2 * E_CHARGE * net_voltage_V / mi_air)
        mdot_air = beam_current_A * mi_air / E_CHARGE
        F_air   = mdot_air * v_air
        Isp_air = v_air / G0

        # Xe performance
        mi_xe   = 2.180e-25
        v_xe    = math.sqrt(2 * E_CHARGE * net_voltage_V / mi_xe)
        mdot_xe = beam_current_A * mi_xe / E_CHARGE
        F_xe    = mdot_xe * v_xe
        Isp_xe  = v_xe / G0

        return {
            "net_voltage_V": net_voltage_V,
            "beam_current_A": beam_current_A,
            "air": {
                "thrust_mN":    round(F_air * 1000, 4),
                "Isp_s":        round(Isp_air),
                "v_exhaust_m_s": round(v_air),
                "mdot_mg_s":    round(mdot_air * 1e6, 4),
            },
            "xenon": {
                "thrust_mN":    round(F_xe * 1000, 4),
                "Isp_s":        round(Isp_xe),
                "v_exhaust_m_s": round(v_xe),
                "mdot_mg_s":    round(mdot_xe * 1e6, 4),
            },
            "air_vs_xe": {
                "thrust_ratio":  round(F_air / F_xe, 3),
                "Isp_ratio":     round(Isp_air / Isp_xe, 3),
                "interpretation": (
                    f"Air gives {F_air/F_xe:.2f}x the thrust of Xe at same current "
                    f"({F_air*1000:.3f}mN vs {F_xe*1000:.3f}mN), but "
                    f"{Isp_air/Isp_xe:.2f}x the Isp "
                    f"({Isp_air:.0f}s vs {Isp_xe:.0f}s). "
                    f"Air has higher thrust density but wastes more propellant."
                )
            }
        }


# ─────────────────────────────────────────────
# 3. ATMOSPHERIC DISCHARGE MODEL
# ─────────────────────────────────────────────

class AtmosphericDischarge:
    """
    Models high-voltage discharge in air at atmospheric pressure.
    This is exactly what your 20kV setup does.

    Discharge modes in air:
    1. Corona discharge  — below breakdown, partial ionization at tips
    2. Glow discharge    — uniform, controllable, good for thrusters
    3. Arc discharge     — plasma channel, high current, electrode damage

    Paschen's law determines breakdown voltage.
    """

    @staticmethod
    def paschen_breakdown_voltage(pressure_Pa: float,
                                   gap_m: float) -> float:
        """
        Paschen breakdown voltage for air (V).
        V_breakdown = B·p·d / (ln(A·p·d) - ln(ln(1 + 1/γ)))

        Paschen constants for air:
        A = 11.2 cm⁻¹·Torr⁻¹ = 840 m⁻¹·Pa⁻¹ × (1/133.3)
        B = 273.8 V·cm⁻¹·Torr⁻¹
        γ = 0.01 (secondary electron emission coefficient)

        Parameters:
            pressure_Pa : gas pressure (Pa)
            gap_m       : electrode gap (m)
        Returns:
            Breakdown voltage (V)
        """
        # Convert to Torr·cm for Paschen constants
        pressure_torr = pressure_Pa / 133.322
        gap_cm        = gap_m * 100
        pd            = pressure_torr * gap_cm  # Torr·cm

        A = 11.2   # cm⁻¹·Torr⁻¹
        B = 273.8  # V·cm⁻¹·Torr⁻¹
        gamma = 0.01

        # Avoid log of negative
        apd = A * pd
        if apd <= 1:
            return float('inf')  # no breakdown possible

        ln_term = math.log(apd) - math.log(math.log(1 + 1/gamma))
        if ln_term <= 0:
            return float('inf')

        return B * pd / ln_term

    @staticmethod
    def discharge_mode(voltage_V: float,
                        pressure_Pa: float,
                        gap_m: float,
                        current_A: float) -> dict:
        """
        Determine discharge mode and characteristics.

        Parameters:
            voltage_V   : applied voltage (V)
            pressure_Pa : gas pressure (Pa)
            gap_m       : electrode gap (m)
            current_A   : discharge current (A)
        """
        V_breakdown = AtmosphericDischarge.paschen_breakdown_voltage(
            pressure_Pa, gap_m
        )

        # Current density
        if gap_m > 0:
            A_typical = (gap_m * 0.1) ** 2  # rough electrode area
            j = current_A / max(A_typical, 1e-6)
        else:
            j = 0

        # Determine mode
        V_corona  = V_breakdown * 0.3    # corona starts at ~30% of breakdown
        V_glow    = V_breakdown * 0.8    # glow starts near breakdown

        warnings = []
        if voltage_V > 20000:
            warnings.append(
                f"NOTE: Your 20kV setup is {'above' if voltage_V > V_breakdown else 'below'} "
                f"the Paschen breakdown voltage of {V_breakdown:.0f}V for this gap/pressure. "
            )

        if voltage_V < V_corona:
            mode = "sub-corona"
            description = "Below corona threshold — no significant ionization"
            thrust_possible = False
        elif voltage_V < V_glow:
            mode = "corona"
            description = "Corona discharge — partial ionization at electrode tips"
            thrust_possible = True
        elif voltage_V < V_breakdown * 1.5:
            mode = "glow"
            description = "Glow discharge — uniform plasma, controllable"
            thrust_possible = True
        else:
            mode = "arc"
            description = "Arc discharge — high current, electrode erosion, uncontrolled"
            thrust_possible = True
            warnings.append(
                "WARNING: Arc discharge — electrodes will erode rapidly. "
                "This is what happens in Plasma Channel style demos. "
                "Add current limiting resistor to stay in glow mode."
            )

        # Ionization efficiency estimate
        if mode == "corona":
            iz_efficiency = 0.001   # 0.1% — very poor
        elif mode == "glow":
            iz_efficiency = 0.05    # 5% — moderate
        elif mode == "arc":
            iz_efficiency = 0.02    # 2% — poor despite high current
        else:
            iz_efficiency = 0.0

        # Thrust estimate (very rough)
        # F ≈ I · d / (mu_0 · v_drift)  — simplified EHD thrust
        mu_i = 2.0e-4   # ion mobility in air (m²/V/s)
        E_field = voltage_V / max(gap_m, 1e-4)
        v_ion = mu_i * E_field
        rho_charge = current_A / (v_ion * gap_m * 0.01) if v_ion > 0 else 0
        F_ehd = rho_charge * E_field * gap_m * 0.01 * gap_m  # very rough

        return {
            "applied_voltage_V":    voltage_V,
            "breakdown_voltage_V":  round(V_breakdown),
            "corona_onset_V":       round(V_corona),
            "glow_onset_V":         round(V_glow),
            "discharge_mode":       mode,
            "description":          description,
            "thrust_possible":      thrust_possible,
            "ionization_efficiency": round(iz_efficiency * 100, 2),
            "estimated_EHD_thrust_mN": round(abs(F_ehd) * 1000, 4),
            "pressure_Pa":          pressure_Pa,
            "gap_m":                gap_m,
            "warnings":             warnings,
            "improvement_tips": _discharge_tips(mode, voltage_V, V_breakdown),
        }

    @staticmethod
    def optimal_gap_for_voltage(voltage_V: float,
                                 pressure_Pa: float,
                                 target_mode: str = "glow") -> dict:
        """
        Find electrode gap that puts discharge in target mode.
        For your 20kV setup: find gap that gives controlled glow discharge.
        """
        best_gap  = None
        best_diff = float('inf')

        for gap_mm in range(1, 200):
            gap_m = gap_mm / 1000
            V_bd  = AtmosphericDischarge.paschen_breakdown_voltage(pressure_Pa, gap_m)

            if target_mode == "glow":
                # Want voltage to be 80-120% of breakdown
                target_V = V_bd
            elif target_mode == "corona":
                target_V = V_bd * 0.4

            diff = abs(voltage_V - target_V)
            if diff < best_diff:
                best_diff = diff
                best_gap  = gap_mm

        if best_gap is None:
            return {"error": "No suitable gap found"}

        V_bd_best = AtmosphericDischarge.paschen_breakdown_voltage(
            pressure_Pa, best_gap/1000
        )

        return {
            "voltage_V":           voltage_V,
            "optimal_gap_mm":      best_gap,
            "breakdown_at_gap_V":  round(V_bd_best),
            "voltage_ratio":       round(voltage_V / V_bd_best, 2),
            "target_mode":         target_mode,
            "recommendation": (
                f"For your {voltage_V}V supply at {pressure_Pa}Pa: "
                f"use a {best_gap}mm electrode gap for {target_mode} discharge. "
                f"This gives V/V_breakdown = {voltage_V/V_bd_best:.2f}."
            )
        }


def _discharge_tips(mode: str, voltage_V: float, V_bd: float) -> list:
    tips = []
    if mode == "arc":
        tips.append(
            "Add a ballast resistor (10-100 kΩ) in series to limit current and "
            "maintain glow mode instead of arc mode."
        )
        tips.append(
            f"Reduce gap size to shift breakdown voltage closer to {voltage_V}V "
            f"for better-controlled glow discharge."
        )
    if mode == "corona":
        tips.append(
            "Increase electrode area or sharpen tips to increase ion production area. "
            "Corona discharge is spatially limited to electrode edges."
        )
        tips.append(
            f"Increase voltage to {V_bd:.0f}V to reach glow mode for "
            f"better ionization efficiency."
        )
    tips.append(
        "For gridded ion thruster operation: the discharge voltage (24-50V) is "
        "separate from the extraction voltage (500-2000V). "
        "Use low voltage for ionization, high voltage only for acceleration."
    )
    return tips


# ─────────────────────────────────────────────
# 4. SELF-NEUTRALIZATION ANALYSIS
# ─────────────────────────────────────────────

class SelfNeutralization:
    """
    Unique to air-breathing thrusters: the oxygen in air
    naturally forms negative ions (O-) that can neutralize
    the positive ion beam without an external neutralizer.
    This is a significant advantage over Xe thrusters.
    """

    @staticmethod
    def o_minus_production_rate(Te_eV: float,
                                 ne_m3: float,
                                 o2_density_m3: float) -> float:
        """O- production rate (m⁻³/s)."""
        k_a = AIR_SPECIES["O2"]["k_attach"]
        peak = math.exp(-(Te_eV - 1.0)**2 / 4.0)
        return ne_m3 * o2_density_m3 * k_a * peak

    @staticmethod
    def neutralization_analysis(beam_current_A: float,
                                 Te_eV: float,
                                 ne_m3: float,
                                 pressure_Pa: float) -> dict:
        """
        Check if air plasma can self-neutralize without external cathode.
        """
        # O2 density from pressure (air composition)
        nn_total = pressure_Pa / (K_BOLTZ * 500)
        o2_density = nn_total * 0.2095

        # O- production rate
        R_neg = SelfNeutralization.o_minus_production_rate(Te_eV, ne_m3, o2_density)

        # Required neutralization current
        I_neut_required = beam_current_A

        # O- current estimate
        # Very rough: I_neg ≈ R_neg · e · Volume
        volume = 1e-6  # 1 cm³ typical discharge volume
        I_neg  = R_neg * E_CHARGE * volume

        # Neutralization fraction
        neut_fraction = min(I_neg / max(I_neut_required, 1e-10), 1.0)

        return {
            "beam_current_A":        beam_current_A,
            "o_minus_current_A":     round(I_neg, 6),
            "neutralization_pct":    round(neut_fraction * 100, 1),
            "self_neutralized":      neut_fraction > 0.8,
            "external_neut_needed":  neut_fraction < 0.5,
            "recommendation": (
                "Self-neutralization sufficient — no external neutralizer needed."
                if neut_fraction > 0.8 else
                f"Partial self-neutralization ({neut_fraction*100:.0f}%). "
                "Consider supplementing with small electron emitter."
                if neut_fraction > 0.3 else
                "Insufficient self-neutralization. External neutralizer required."
            ),
            "advantage_over_xe": (
                "Air-breathing thrusters can achieve self-neutralization via O- ions. "
                "Xenon thrusters ALWAYS need an external hollow cathode neutralizer. "
                "This reduces system complexity and power consumption."
            )
        }


# ─────────────────────────────────────────────
# LLM INTERFACE
# ─────────────────────────────────────────────

class AirPlasmaAI:
    """LLM interface for air plasma modules."""

    def air_ionization(self, Te_eV: float) -> str:
        result = AirIonization.effective_ionization_rate_air(Te_eV)
        cost   = AirIonization.energy_cost_per_ion(Te_eV)
        return json.dumps({"ionization": result, "energy_cost": cost}, indent=2)

    def discharge_analysis(self, voltage_V: float, pressure_Pa: float,
                            gap_mm: float, current_A: float) -> str:
        result = AtmosphericDischarge.discharge_mode(
            voltage_V, pressure_Pa, gap_mm/1000, current_A
        )
        return json.dumps(result, indent=2)

    def optimal_gap(self, voltage_V: float, pressure_Pa: float,
                    mode: str = "glow") -> str:
        result = AtmosphericDischarge.optimal_gap_for_voltage(
            voltage_V, pressure_Pa, mode
        )
        return json.dumps(result, indent=2)

    def air_vs_xenon(self, net_V: float, beam_mA: float, power_W: float) -> str:
        result = AirThrusterPerformance.compare_air_vs_xenon(
            net_V, beam_mA/1000, power_W
        )
        return json.dumps(result, indent=2)

    def self_neutralization(self, beam_mA: float, Te_eV: float,
                             ne_m3: float, pressure_Pa: float) -> str:
        result = SelfNeutralization.neutralization_analysis(
            beam_mA/1000, Te_eV, ne_m3, pressure_Pa
        )
        return json.dumps(result, indent=2)

    def air_performance(self, net_V: float, beam_mA: float,
                         n2_frac: float = 0.78, o2_frac: float = 0.21,
                         o_neg_frac: float = 0.05) -> str:
        result = AirThrusterPerformance.thrust_mixed_beam(
            beam_mA/1000, net_V, n2_frac, o2_frac, o_neg_frac
        )
        return json.dumps(result, indent=2)


if __name__ == "__main__":

    print("\n=== TEST 1: Air ionization at different Te ===")
    for Te in [5, 10, 15, 20, 25]:
        r = AirIonization.effective_ionization_rate_air(Te)
        print(f"  Te={Te:2}eV | iz_eff={r['ionization_pct']:5.1f}% | "
              f"O- {r['o_minus_fraction_pct']:.1f}% | "
              f"vs Xe: {r['ratio_vs_xenon']:.3f}x")

    print("\n=== TEST 2: Energy cost per ion — air vs Xe ===")
    for Te in [10, 15, 20]:
        c = AirIonization.energy_cost_per_ion(Te)
        print(f"  Te={Te}eV | Air: {c['energy_cost_air_eV_per_ion']}eV/ion | "
              f"Xe: {c['energy_cost_xenon_eV_per_ion']}eV/ion | "
              f"ratio: {c['air_vs_xe_ratio']}x")

    print("\n=== TEST 3: Your 20kV setup — discharge mode analysis ===")
    d = AtmosphericDischarge.discharge_mode(
        voltage_V   = 20000,
        pressure_Pa = 101325,  # sea level
        gap_m       = 0.05,    # 5cm gap
        current_A   = 0.001,
    )
    print(json.dumps(d, indent=2))

    print("\n=== TEST 4: Optimal gap for your 20kV ===")
    g = AtmosphericDischarge.optimal_gap_for_voltage(
        voltage_V   = 20000,
        pressure_Pa = 101325,
        target_mode = "glow"
    )
    print(json.dumps(g, indent=2))

    print("\n=== TEST 5: Air vs Xe performance at 1kV, 10mA ===")
    comp = AirThrusterPerformance.compare_air_vs_xenon(1000, 0.010, 15.0)
    print(json.dumps(comp, indent=2))

    print("\n=== TEST 6: Self-neutralization check ===")
    sn = SelfNeutralization.neutralization_analysis(0.010, 15.0, 1e16, 0.1)
    print(json.dumps(sn, indent=2))
