"""
propellants_db.py
=================
Unified Propellant Database — Single Source of Truth
Imported by all Celestine physics modules.

Covers every gas that has been seriously considered for ion/EHD propulsion:
  Noble gases   : Xenon, Krypton, Argon, Neon, Helium
  Molecular     : Nitrogen (N2), Oxygen (O2), Carbon Dioxide (CO2), Hydrogen (H2)
  Mixtures      : Air (standard atmosphere), N2/O2 mixture
  Condensable   : Iodine (I2), Bismuth (Bi), Mercury (Hg) — historical / CubeSat
  EHD-specific  : Air is the primary EHD propellant (free, self-replenishing)

Each entry contains:
  mass_kg         : Ion/molecule mass (kg)
  ionization_eV   : First ionization energy (eV)
  double_ion_eV   : Second ionization energy (eV) — double-ionization threshold
  k_iz            : Ionization rate coefficient at Te=4eV (m³/s)
  k_iz_formula    : 'arrhenius' — use k_iz * exp(-E_ion/Te) scaling
  cex_sigma_m2    : Charge-exchange cross section at 1 keV (m²)
  sputter_Mo_eV   : Sputtering threshold on Molybdenum grid (eV)
  sputter_C_eV    : Sputtering threshold on Carbon grid (eV)
  Isp_1kV_s       : Specific impulse at 1kV extraction (s)
  mu_ion          : Ion mobility in gas at STP (m²/V/s) — EHD relevant
  boiling_K       : Boiling point (K) — storage/feed system design
  state_STP       : 'gas', 'liquid', 'solid' at standard conditions
  storage         : How to store/feed
  vacuum_suitable : True if good for vacuum gridded ion thruster
  ehd_suitable    : True if good for atmospheric EHD thruster
  practical_notes : Engineering notes
  rank_vacuum     : 1=best for vacuum thrusters (lower=better)
  rank_ehd        : 1=best for EHD atmospheric thrusters

References:
  - Goebel & Katz, "Fundamentals of Electric Propulsion" (2008)
  - Gilmore & Barrett, Royal Society A (2015) — EHD thrust
  - Kaufman et al. — ion thruster design heritage
  - NASA ion propulsion review papers
"""

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

E_CHARGE  = 1.602e-19   # C
AMU       = 1.661e-27   # kg per atomic mass unit
K_BOLTZ   = 1.381e-23   # J/K
import math

def isp_from_voltage(mass_kg: float, voltage_V: float) -> float:
    """Specific impulse (s) for ion accelerated through voltage_V."""
    g0 = 9.81  # m/s²
    v_exit = math.sqrt(2 * E_CHARGE * voltage_V / mass_kg)
    return v_exit / g0

# ─────────────────────────────────────────────────────────────────────────────
# MASTER PROPELLANT DATABASE
# ─────────────────────────────────────────────────────────────────────────────

PROPELLANTS = {

    # ══════════════════════════════════════════════════════
    # NOBLE GASES — vacuum ion thruster workhorses
    # ══════════════════════════════════════════════════════

    "Xenon": {
        "formula":          "Xe",
        "atomic_mass_amu":  131.29,
        "mass_kg":          2.180e-25,       # 131.29 × 1.661e-27
        "ionization_eV":    12.13,           # 1st ionization — lowest of noble gases
        "excitation_eV":    8.32,            # first metastable state
        "double_ion_eV":    21.21,           # 2nd ionization — CEX source
        "k_iz":             1.8e-14,         # ionization rate prefactor (m³/s)
        "k_iz_prefactor":   1.8e-14,         # alias — used by ion_thruster_model, plasma_physics_expanded, advanced_physics
        "k_iz_double":      5.0e-15,         # double-ionization rate prefactor
        "cex_sigma_m2":     51e-20,          # CEX cross section at 1 keV
        "cex_energy_exp":   -0.5,            # CEX cross section energy dependence exponent
        "hard_sphere_m2":   1.0e-19,         # hard-sphere collision cross section
        "sputter_Mo_eV":    35,              # Mo sputtering threshold
        "sputter_C_eV":     25,              # Carbon sputtering threshold
        "Isp_1kV_s":        isp_from_voltage(2.180e-25, 1000),
        "mu_ion":           0.6e-4,          # ion mobility in Xe gas (m²/V/s)
        "boiling_K":        165.0,
        "state_STP":        "gas",
        "storage":          "Compressed gas cylinder, 100-200 bar. Liquefies at -108°C.",
        "vacuum_suitable":  True,
        "ehd_suitable":     False,           # needs sealed chamber, expensive
        "rank_vacuum":      1,               # BEST for vacuum thrusters
        "rank_ehd":         5,
        "practical_notes": (
            "Gold standard for ion thrusters. Highest mass gives best Isp per volt. "
            "Low ionization energy means efficient discharge (low discharge loss ~180 eV/ion). "
            "Heavy mass → high thrust per beam current. "
            "Expensive (~$1000/kg) and requires pressurized storage. "
            "Used by Dawn, Hayabusa, BepiColombo, Starlink."
        ),
    },

    "Krypton": {
        "formula":          "Kr",
        "atomic_mass_amu":  83.80,
        "mass_kg":          1.391e-25,
        "ionization_eV":    13.99,
        "excitation_eV":    9.92,
        "double_ion_eV":    24.36,
        "k_iz":             1.2e-14,
        "k_iz_prefactor":   1.2e-14,
        "k_iz_double":      3.0e-15,
        "cex_sigma_m2":     41e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   8.0e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     25,
        "Isp_1kV_s":        isp_from_voltage(1.391e-25, 1000),
        "mu_ion":           0.8e-4,
        "boiling_K":        119.9,
        "state_STP":        "gas",
        "storage":          "Compressed gas cylinder. Cheaper than Xe.",
        "vacuum_suitable":  True,
        "ehd_suitable":     False,
        "rank_vacuum":      2,
        "rank_ehd":         5,
        "practical_notes": (
            "SpaceX Starlink v2 switched from Xe to Kr for cost reasons (~10x cheaper). "
            "Higher ionization energy means slightly less efficient discharge. "
            "Lower mass than Xe → higher exhaust velocity → better Isp at same voltage. "
            "Good choice when cost matters more than thrust density."
        ),
    },

    "Argon": {
        "formula":          "Ar",
        "atomic_mass_amu":  39.95,
        "mass_kg":          6.634e-26,
        "ionization_eV":    15.76,
        "excitation_eV":    11.55,
        "double_ion_eV":    27.63,
        "k_iz":             0.9e-14,
        "k_iz_prefactor":   0.9e-14,
        "k_iz_double":      2.0e-15,
        "cex_sigma_m2":     28e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   4.0e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     25,
        "Isp_1kV_s":        isp_from_voltage(6.634e-26, 1000),
        "mu_ion":           1.4e-4,
        "boiling_K":        87.3,
        "state_STP":        "gas",
        "storage":          "Compressed gas. Very cheap, widely available.",
        "vacuum_suitable":  True,
        "ehd_suitable":     True,             # can work in EHD but less efficient than air
        "rank_vacuum":      3,
        "rank_ehd":         3,
        "practical_notes": (
            "Cheapest noble gas. Good for lab testing and prototyping. "
            "High ionization energy means higher discharge losses (~600 eV/ion). "
            "Lower mass → higher Isp but lower thrust at same power. "
            "Good for hall thrusters; less common in gridded ion thrusters. "
            "Your prototype could use Ar for initial testing before buying Xe."
        ),
    },

    "Neon": {
        "formula":          "Ne",
        "atomic_mass_amu":  20.18,
        "mass_kg":          3.351e-26,
        "ionization_eV":    21.56,
        "excitation_eV":    16.62,
        "double_ion_eV":    41.07,
        "k_iz":             0.4e-14,
        "k_iz_prefactor":   0.4e-14,
        "k_iz_double":      0.5e-15,
        "cex_sigma_m2":     18e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   2.0e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     25,
        "Isp_1kV_s":        isp_from_voltage(3.351e-26, 1000),
        "mu_ion":           2.5e-4,
        "boiling_K":        27.1,
        "state_STP":        "gas",
        "storage":          "Compressed gas. Rare, moderate cost.",
        "vacuum_suitable":  True,
        "ehd_suitable":     False,
        "rank_vacuum":      4,
        "rank_ehd":         5,
        "practical_notes": (
            "Very high ionization energy — hard to ionize efficiently. "
            "Rarely used in practice. High Isp due to low mass. "
            "NSTAR tested Ne but found Xe far more efficient."
        ),
    },

    "Helium": {
        "formula":          "He",
        "atomic_mass_amu":  4.003,
        "mass_kg":          6.646e-27,
        "ionization_eV":    24.59,
        "excitation_eV":    19.82,
        "double_ion_eV":    54.42,
        "k_iz":             0.1e-14,
        "k_iz_prefactor":   0.1e-14,
        "k_iz_double":      0.05e-15,
        "cex_sigma_m2":     7e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   0.8e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     25,
        "Isp_1kV_s":        isp_from_voltage(6.646e-27, 1000),
        "mu_ion":           10.0e-4,
        "boiling_K":        4.2,
        "state_STP":        "gas",
        "storage":          "High pressure cylinder. Very low density, hard to store.",
        "vacuum_suitable":  False,           # practically unusable — too hard to ionize
        "ehd_suitable":     False,
        "rank_vacuum":      6,
        "rank_ehd":         6,
        "practical_notes": (
            "Extremely high ionization energy — practically impossible to ionize efficiently. "
            "Theoretical Isp is enormous but discharge efficiency is terrible. "
            "Not used in real ion thrusters. "
            "Very high ion mobility makes it interesting for EHD research only."
        ),
    },

    # ══════════════════════════════════════════════════════
    # MOLECULAR GASES — atmospheric / air-breathing
    # ══════════════════════════════════════════════════════

    "Air": {
        "formula":          "N2/O2 78/21%",
        "atomic_mass_amu":  28.97,           # average molar mass of air
        "mass_kg":          4.810e-26,       # average ion mass
        "ionization_eV":    14.10,           # effective average (N2: 15.58, O2: 12.07)
        "excitation_eV":    6.17,            # N2 first electronic excitation
        "double_ion_eV":    29.6,
        "k_iz":             0.5e-14,         # effective rate
        "k_iz_prefactor":   0.5e-14,
        "k_iz_double":      1.0e-15,
        "cex_sigma_m2":     40e-20,          # approximate
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   3.5e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     25,
        "Isp_1kV_s":        isp_from_voltage(4.810e-26, 1000),
        "mu_ion":           2.0e-4,          # positive ion mobility in air at STP
        "mu_ion_neg":       2.5e-4,          # O- negative ion mobility (EHD self-neutralization)
        "boiling_K":        78.8,            # N2 boiling point (dominant component)
        "state_STP":        "gas",
        "storage":          "FREE — ambient atmosphere. No storage needed.",
        "vacuum_suitable":  False,           # needs atmospheric pressure
        "ehd_suitable":     True,            # PRIMARY EHD propellant
        "rank_vacuum":      7,
        "rank_ehd":         1,               # BEST for EHD — it's free and abundant
        "composition": {
            "N2":  0.7809,
            "O2":  0.2095,
            "Ar":  0.0093,
            "CO2": 0.0004,
        },
        "ehd_physics": {
            "paschen_min_V":     327,        # Paschen minimum voltage for air
            "paschen_min_pd":    0.57,       # Torr·cm at minimum
            "breakdown_field":   3.0e6,      # V/m at STP
            "corona_onset_coeff": 0.03,      # Peek's constant (cm^0.5)
            "o_neg_fraction":    0.05,       # O- fraction enabling self-neutralization
        },
        "practical_notes": (
            "THE propellant for your EHD air-breathing prototype. "
            "Completely free, no storage, infinite supply. "
            "O2 component (12.07 eV) ionizes more easily than N2 (15.58 eV). "
            "O- ions form naturally from O2, providing self-neutralization — "
            "no external neutralizer needed unlike vacuum thrusters. "
            "High energy cost per ion (~500-2000 eV) vs Xe (~200 eV) but "
            "propellant is free so overall system efficiency can be competitive. "
            "Humidity reduces ion mobility by ~0.5% per % RH. "
            "Ozone (O3) produced as byproduct — ventilate workspace."
        ),
    },

    "Nitrogen": {
        "formula":          "N2",
        "atomic_mass_amu":  28.01,
        "mass_kg":          4.652e-26,
        "ionization_eV":    15.58,           # N2 molecular ionization
        "excitation_eV":    6.17,
        "double_ion_eV":    27.0,
        "k_iz":             0.6e-14,
        "k_iz_prefactor":   0.6e-14,
        "k_iz_double":      1.0e-15,
        "cex_sigma_m2":     38e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   3.8e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     25,
        "Isp_1kV_s":        isp_from_voltage(4.652e-26, 1000),
        "mu_ion":           1.8e-4,
        "boiling_K":        77.4,
        "state_STP":        "gas",
        "storage":          "Compressed cylinder or liquid dewar. Very cheap.",
        "vacuum_suitable":  True,            # used in some hall thrusters
        "ehd_suitable":     True,
        "rank_vacuum":      5,
        "rank_ehd":         2,
        "practical_notes": (
            "Main component of air (78%). Used in some Hall effect thrusters. "
            "Higher ionization energy than Xe but very cheap. "
            "N2+ ions are the primary charge carrier in air EHD thrusters. "
            "Good for lab testing — cheap, non-toxic, widely available. "
            "Used in early FEEP and colloid thruster research."
        ),
    },

    "Oxygen": {
        "formula":          "O2",
        "atomic_mass_amu":  32.00,
        "mass_kg":          5.314e-26,
        "ionization_eV":    12.07,           # easier than N2 — similar to Xe
        "excitation_eV":    4.50,
        "double_ion_eV":    35.12,
        "k_iz":             0.7e-14,
        "k_iz_prefactor":   0.7e-14,
        "k_iz_double":      1.2e-15,
        "cex_sigma_m2":     35e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   3.2e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     20,              # C oxidizes — lower threshold
        "Isp_1kV_s":        isp_from_voltage(5.314e-26, 1000),
        "mu_ion":           2.2e-4,
        "mu_ion_neg":       2.5e-4,          # O- mobility
        "boiling_K":        90.2,
        "state_STP":        "gas",
        "storage":          "Compressed cylinder. Oxidizer — handle with care.",
        "vacuum_suitable":  False,           # oxidizes everything
        "ehd_suitable":     True,
        "rank_vacuum":      8,               # NOT recommended — destroys electrodes
        "rank_ehd":         4,
        "practical_notes": (
            "Low ionization energy (12.07 eV) makes it easy to ionize. "
            "Forms O- negative ions naturally — excellent self-neutralization. "
            "DANGER: Pure O2 is a strong oxidizer. Burns materials, degrades electrodes. "
            "In air, the 21% O2 fraction is beneficial; pure O2 thruster is impractical. "
            "Carbon electrodes will be destroyed. Use only in air mixture."
        ),
    },

    "CO2": {
        "formula":          "CO2",
        "atomic_mass_amu":  44.01,
        "mass_kg":          7.308e-26,
        "ionization_eV":    13.78,
        "excitation_eV":    7.0,
        "double_ion_eV":    27.4,
        "k_iz":             0.5e-14,
        "k_iz_prefactor":   0.5e-14,
        "k_iz_double":      0.8e-15,
        "cex_sigma_m2":     45e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   5.0e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     25,
        "Isp_1kV_s":        isp_from_voltage(7.308e-26, 1000),
        "mu_ion":           1.1e-4,
        "boiling_K":        194.7,           # sublimation point
        "state_STP":        "gas",
        "storage":          "Compressed cylinder or dry ice. Very cheap.",
        "vacuum_suitable":  True,            # used for Mars missions (ISRU)
        "ehd_suitable":     True,
        "rank_vacuum":      4,
        "rank_ehd":         3,
        "practical_notes": (
            "Excellent for Mars ISRU (in-situ resource utilization) — "
            "Mars atmosphere is 95% CO2. "
            "Heavier than Xe per molecule but multi-atom → fragments under discharge. "
            "CO2+ and O+ ions both contribute to thrust. "
            "Used in MarCO CubeSat cold gas thruster concept. "
            "On Earth: cheap, non-toxic, available anywhere. "
            "Moderate ionization energy, good CEX cross section."
        ),
    },

    "Hydrogen": {
        "formula":          "H2",
        "atomic_mass_amu":  2.016,
        "mass_kg":          3.348e-27,
        "ionization_eV":    15.43,
        "excitation_eV":    10.20,
        "double_ion_eV":    None,            # H2 dissociates before double ionization
        "k_iz":             0.8e-14,
        "k_iz_prefactor":   0.8e-14,
        "k_iz_double":      None,
        "cex_sigma_m2":     10e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   0.5e-20,
        "sputter_Mo_eV":    35,
        "sputter_C_eV":     25,
        "Isp_1kV_s":        isp_from_voltage(3.348e-27, 1000),
        "mu_ion":           15.0e-4,         # very high mobility — lightest ion
        "boiling_K":        20.3,
        "state_STP":        "gas",
        "storage":          "High pressure cylinder or cryogenic liquid. FLAMMABLE.",
        "vacuum_suitable":  True,            # very high Isp
        "ehd_suitable":     False,           # flammable — EHD sparks are dangerous
        "rank_vacuum":      5,
        "rank_ehd":         7,               # dangerous with EHD corona discharge
        "practical_notes": (
            "Highest Isp of any propellant at given voltage (lowest mass). "
            "Theoretical Isp at 1kV is ~45,000s — extraordinary. "
            "DANGER for EHD: H2 ignites at 4% concentration in air. "
            "Corona discharge will ignite H2. Never use H2 in atmospheric EHD. "
            "Used in resistojets and arcjets. Arcjet Isp ~600-1000s. "
            "Storage is very challenging (cryogenic or very high pressure)."
        ),
    },

    # ══════════════════════════════════════════════════════
    # CONDENSABLE PROPELLANTS — CubeSat / micro thruster
    # ══════════════════════════════════════════════════════

    "Iodine": {
        "formula":          "I2",
        "atomic_mass_amu":  253.81,          # I2 molecule, singly ionized as I+
        "mass_kg":          2.107e-25,       # I+ ion mass ≈ Xe mass
        "ionization_eV":    10.45,           # atomic I (from dissociated I2)
        "excitation_eV":    7.86,
        "double_ion_eV":    19.13,
        "k_iz":             2.2e-14,         # higher than Xe due to lower ionization
        "k_iz_prefactor":   2.2e-14,
        "k_iz_double":      4.0e-15,
        "cex_sigma_m2":     60e-20,          # larger than Xe
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   1.2e-19,
        "sputter_Mo_eV":    30,              # lower threshold — corrosive
        "sputter_C_eV":     20,
        "Isp_1kV_s":        isp_from_voltage(2.107e-25, 1000),
        "mu_ion":           0.6e-4,
        "boiling_K":        457.5,
        "state_STP":        "solid",
        "storage":          "Solid at room temp. Sublimes to gas for use. Simple tank.",
        "vacuum_suitable":  True,
        "ehd_suitable":     False,
        "rank_vacuum":      2,               # excellent Xe alternative for CubeSats
        "rank_ehd":         6,
        "practical_notes": (
            "Exciting Xe alternative for CubeSats. Stores as solid, no pressure vessel. "
            "Similar mass to Xe → similar Isp and thrust. "
            "Lower ionization energy → potentially more efficient discharge. "
            "Corrosive to some metals — use Mo or C electrodes, avoid Al. "
            "BIT-3 (TU Munich) demonstrated iodine gridded ion thruster in orbit 2021. "
            "Cost: ~$100/kg vs $1000/kg for Xe. "
            "WARNING: I2 vapor is toxic. Ventilation required."
        ),
    },

    "Bismuth": {
        "formula":          "Bi",
        "atomic_mass_amu":  208.98,
        "mass_kg":          3.470e-25,
        "ionization_eV":    7.29,            # LOWEST ionization energy of any element
        "excitation_eV":    5.46,
        "double_ion_eV":    16.69,
        "k_iz":             3.5e-14,         # very high — easiest to ionize
        "k_iz_prefactor":   3.5e-14,
        "k_iz_double":      6.0e-15,
        "cex_sigma_m2":     70e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   1.5e-19,
        "sputter_Mo_eV":    30,
        "sputter_C_eV":     20,
        "Isp_1kV_s":        isp_from_voltage(3.470e-25, 1000),
        "mu_ion":           0.4e-4,
        "boiling_K":        1837.0,
        "state_STP":        "solid",
        "storage":          "Solid metal. Melts at 271°C for liquid-feed systems.",
        "vacuum_suitable":  True,
        "ehd_suitable":     False,
        "rank_vacuum":      3,
        "rank_ehd":         6,
        "practical_notes": (
            "Used in early Soviet ion thrusters (SPT-series). "
            "Lowest ionization energy of any practical propellant → very efficient discharge. "
            "Very heavy → high thrust, low Isp compared to Xe. "
            "Requires heating system to melt/vaporize (~300°C). "
            "Non-toxic solid at room temperature — simpler storage than gases. "
            "Falling out of use in favor of Xe/Kr/I2."
        ),
    },

    "Mercury": {
        "formula":          "Hg",
        "atomic_mass_amu":  200.59,
        "mass_kg":          3.331e-25,
        "ionization_eV":    10.44,
        "excitation_eV":    4.67,
        "double_ion_eV":    18.76,
        "k_iz":             2.0e-14,
        "k_iz_prefactor":   2.0e-14,
        "k_iz_double":      3.5e-15,
        "cex_sigma_m2":     65e-20,
        "cex_energy_exp":   -0.5,
        "hard_sphere_m2":   1.3e-19,
        "sputter_Mo_eV":    25,              # amalgamates with Mo — very low threshold
        "sputter_C_eV":     20,
        "Isp_1kV_s":        isp_from_voltage(3.331e-25, 1000),
        "mu_ion":           0.5e-4,
        "boiling_K":        629.9,
        "state_STP":        "liquid",
        "storage":          "Liquid at room temp. Simple tank but HIGHLY TOXIC.",
        "vacuum_suitable":  True,
        "ehd_suitable":     False,
        "rank_vacuum":      7,               # historical only — banned from spacecraft
        "rank_ehd":         7,
        "practical_notes": (
            "Used in early NASA ion thrusters (SERT-1, 1964). "
            "Liquid at room temperature → simple feed system. "
            "Low ionization energy → efficient discharge. "
            "BANNED from modern spacecraft — toxic contamination of instruments. "
            "Historical interest only. Do not use."
        ),
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_propellant(name: str) -> dict:
    """
    Get propellant data by name. Case-insensitive.
    Returns the propellant dict or raises KeyError.
    """
    for k, v in PROPELLANTS.items():
        if k.lower() == name.lower():
            return v
    # Fuzzy match on formula
    for k, v in PROPELLANTS.items():
        if v.get("formula", "").lower() == name.lower():
            return v
    raise KeyError(
        f"Unknown propellant '{name}'. "
        f"Available: {list(PROPELLANTS.keys())}"
    )


def list_vacuum_propellants() -> list:
    """Return propellants suitable for vacuum gridded ion thrusters, ranked."""
    candidates = [(k, v) for k, v in PROPELLANTS.items() if v["vacuum_suitable"]]
    return sorted(candidates, key=lambda x: x[1]["rank_vacuum"])


def list_ehd_propellants() -> list:
    """Return propellants suitable for atmospheric EHD thrusters, ranked."""
    candidates = [(k, v) for k, v in PROPELLANTS.items() if v["ehd_suitable"]]
    return sorted(candidates, key=lambda x: x[1]["rank_ehd"])


def compare_propellants(names: list, mode: str = "vacuum") -> dict:
    """
    Compare multiple propellants side by side.

    Parameters:
        names : list of propellant names
        mode  : 'vacuum' or 'ehd'

    Returns dict with comparison table.
    """
    results = {}
    for name in names:
        try:
            p = get_propellant(name)
            results[name] = {
                "mass_amu":         p["atomic_mass_amu"],
                "ionization_eV":    p["ionization_eV"],
                "Isp_1kV_s":        round(p["Isp_1kV_s"]),
                "k_iz":             p["k_iz"],
                "cex_sigma_m2":     p["cex_sigma_m2"],
                "sputter_Mo_eV":    p["sputter_Mo_eV"],
                "storage":          p["state_STP"],
                "vacuum_suitable":  p["vacuum_suitable"],
                "ehd_suitable":     p["ehd_suitable"],
                "rank":             p[f"rank_{mode}"] if f"rank_{mode}" in p else "N/A",
                "notes":            p["practical_notes"][:120] + "...",
            }
        except KeyError as e:
            results[name] = {"error": str(e)}

    # Sort by rank for the selected mode
    rank_key = f"rank_{mode}"
    ranked = dict(sorted(
        results.items(),
        key=lambda x: x[1].get("rank", 99) if isinstance(x[1].get("rank"), int) else 99
    ))

    return {
        "mode":       mode,
        "propellants": ranked,
        "best":       next(iter(ranked)) if ranked else None,
        "summary": (
            f"Best for {mode}: {next(iter(ranked))} — "
            f"ionization {results[next(iter(ranked))].get('ionization_eV', '?')} eV, "
            f"Isp {results[next(iter(ranked))].get('Isp_1kV_s', '?')} s at 1kV"
        ) if ranked else "No valid propellants"
    }


def bohm_velocity(propellant_name: str, Te_eV: float) -> float:
    """Bohm velocity u_B = sqrt(e*Te/mi) for given propellant and Te."""
    p = get_propellant(propellant_name)
    return math.sqrt(E_CHARGE * Te_eV / p["mass_kg"])


def isp_at_voltage(propellant_name: str, voltage_V: float) -> float:
    """Specific impulse (s) for propellant accelerated through voltage_V."""
    p = get_propellant(propellant_name)
    return isp_from_voltage(p["mass_kg"], voltage_V)


# ─────────────────────────────────────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("\n=== VACUUM PROPELLANTS (ranked) ===")
    for name, prop in list_vacuum_propellants():
        print(f"  {prop['rank_vacuum']}. {name:12s} | "
              f"ion={prop['ionization_eV']:5.2f}eV | "
              f"Isp@1kV={prop['Isp_1kV_s']:6.0f}s | "
              f"mass={prop['atomic_mass_amu']:6.1f}amu")

    print("\n=== EHD PROPELLANTS (ranked) ===")
    for name, prop in list_ehd_propellants():
        print(f"  {prop['rank_ehd']}. {name:12s} | "
              f"ion={prop['ionization_eV']:5.2f}eV | "
              f"mu_ion={prop.get('mu_ion', 0)*1e4:.1f}×10⁻⁴ m²/V/s | "
              f"storage={prop['state_STP']}")

    print("\n=== COMPARISON: Xe vs Kr vs Ar vs Iodine (vacuum) ===")
    comp = compare_propellants(["Xenon", "Krypton", "Argon", "Iodine"], mode="vacuum")
    for name, data in comp["propellants"].items():
        if "error" not in data:
            print(f"  {name:10s}: rank={data['rank']} | "
                  f"Isp={data['Isp_1kV_s']}s | "
                  f"ioniz={data['ionization_eV']}eV")
    print(f"  Best: {comp['best']}")

    print("\n=== EHD: Air vs N2 vs CO2 ===")
    comp2 = compare_propellants(["Air", "Nitrogen", "CO2", "Argon"], mode="ehd")
    for name, data in comp2["propellants"].items():
        if "error" not in data:
            print(f"  {name:10s}: rank={data['rank']} | "
                  f"ioniz={data['ionization_eV']}eV")

    print("\n=== BOHM VELOCITY: Xe at Te=5eV ===")
    v = bohm_velocity("Xenon", 5.0)
    print(f"  u_B = {v:.0f} m/s")

    print("\n=== ISP AT DIFFERENT VOLTAGES: Xe vs Kr vs I2 ===")
    for v_kv in [1, 2, 5]:
        print(f"  At {v_kv}kV:")
        for name in ["Xenon", "Krypton", "Iodine"]:
            isp = isp_at_voltage(name, v_kv * 1000)
            print(f"    {name:10s}: {isp:6.0f} s")

    print("\n=== AIR EHD PHYSICS ===")
    air = get_propellant("Air")
    ehd = air["ehd_physics"]
    print(f"  Paschen minimum: {ehd['paschen_min_V']}V at pd={ehd['paschen_min_pd']} Torr·cm")
    print(f"  Breakdown field: {ehd['breakdown_field']/1e6:.1f} MV/m")
    print(f"  O- fraction: {ehd['o_neg_fraction']*100:.0f}% (self-neutralization)")
    print(f"  μ_ion (positive): {air['mu_ion']*1e4:.1f}×10⁻⁴ m²/V/s")
    print(f"  μ_ion (O-):       {air['mu_ion_neg']*1e4:.1f}×10⁻⁴ m²/V/s")
