"""
electrostatics.py
Pure physics outputs only (Option A architecture).
"""

import math

EPSILON_0 = 8.854e-12
E_CHARGE = 1.602e-19
K_BOLTZ = 1.381e-23

EPS_AIR = EPSILON_0 * 1.0006
EPS_PETG = EPSILON_0 * 3.6
EPS_ABS = EPSILON_0 * 2.9

E_BREAK_STP = 3.0e6
PEEK_A = 3.1e6
PEEK_B = 0.030


class TipEnhancement:
    @staticmethod
    def hyperboloid_tip(voltage_V: float,
                        tip_radius_mm: float,
                        emitter_length_mm: float) -> dict:
        r = tip_radius_mm / 1000
        h = emitter_length_mm / 1000

        ln_term = math.log(2 * h / r) if (2 * h / r) > 1 else 1.0
        E_tip_analytic = voltage_V / (r * ln_term)
        E_tip_corrected = E_tip_analytic
        E_avg = voltage_V / h
        beta = E_tip_analytic / E_avg if E_avg > 0 else 0.0
        E_onset = PEEK_A * (1 + PEEK_B / math.sqrt(r))
        V_onset = E_onset * r * ln_term
        d_10pct = r * 10
        d_1pct = r * 100
        ratio = E_tip_corrected / E_BREAK_STP

        # ===== DEBUG BLOCK (add this) =====
        print("\n--- DEBUG: hyperboloid_tip ---")
        print(f"Input Voltage: {voltage_V} V")
        print(f"Tip radius (m): {r}")
        print(f"Emitter length (m): {h}")
        print(f"ln term: {ln_term}")

        print(f"E_tip_analytic: {E_tip_analytic:.3e}")
        print(f"E_tip_corrected: {E_tip_corrected:.3e}")
        print(f"E_avg: {E_avg:.3e}")
        print(f"beta (enhancement): {beta:.2f}")

        print(f"E_onset: {E_onset:.3e}")
        print(f"V_onset: {V_onset:.2f}")

        print(f"Breakdown ratio (E_tip / E_break): {ratio:.2f}")
        print("--------------------------------\n")
        # ===== END DEBUG =====

        return {
            "physics_outputs": {
                "E_tip": E_tip_corrected,
                "E_tip_analytic": E_tip_analytic,
                "E_tip_corrected": E_tip_corrected,
                "E_avg": E_avg,
                "beta": beta,
                "E_onset": E_onset,
                "V_onset": V_onset,
                "d_10pct": d_10pct,
                "d_1pct": d_1pct,
            },
            "_derived": {
                "_ratio": ratio,
            },
            "model_parameters": {
                "space_charge_correction_factor": 0.8,
            },
            "field_model": "hyperboloid_analytic",
            "source": "analytic",
        }

    @staticmethod
    def wire_cylinder(voltage_V: float,
                      wire_radius_mm: float,
                      cylinder_radius_mm: float) -> dict:
        a = wire_radius_mm / 1000
        R = cylinder_radius_mm / 1000

        ln_Ra = math.log(R / a)
        E_wire_surface = voltage_V / (a * ln_Ra)
        E_at_collector = voltage_V / (R * ln_Ra)
        E_onset = PEEK_A * (1 + PEEK_B / math.sqrt(a))
        V_onset = E_onset * a * ln_Ra

        radii = [a, a * 2, a * 5, (a + R) / 2, R]
        profile = []
        for rr in radii:
            Er = voltage_V / (rr * ln_Ra)
            profile.append({
                "r_m": rr,
                "E": Er,
            })

        return {
            "physics_outputs": {
                "wire_radius_m": a,
                "cylinder_radius_m": R,
                "voltage_V": voltage_V,
                "E_wire_surface": E_wire_surface,
                "E_collector_surface": E_at_collector,
                "E_onset": E_onset,
                "V_onset": V_onset,
                "radial_profile": profile,
            },
            "_derived": {
                "_ratio": E_wire_surface / E_BREAK_STP,
                "field_ratio_max_min": E_wire_surface / E_at_collector,
            },
            "field_model": "coaxial_analytic",
            "source": "analytic",
        }

    @staticmethod
    def needle_array(voltage_V: float,
                     tip_radius_mm: float,
                     emitter_length_mm: float,
                     n_needles: int,
                     spacing_mm: float) -> dict:
        single = TipEnhancement.hyperboloid_tip(
            voltage_V, tip_radius_mm, emitter_length_mm
        )
        single_fields = single["physics_outputs"]

        h = emitter_length_mm / 1000
        s = spacing_mm / 1000
        k = 2.3
        eta = 1 - math.exp(-k * s / h)

        E_single = single_fields["E_tip_corrected"]
        E_effective = E_single * eta
        s_opt_m = -h / k * math.log(1 - 0.8)
        ring_radius_m = (n_needles * s) / (2 * math.pi)

        return {
            "physics_outputs": {
                "n_needles": n_needles,
                "spacing_m": s,
                "tip_radius_m": tip_radius_mm / 1000,
                "E_single": E_single,
                "E_effective": E_effective,
                "V_onset": single_fields["V_onset"],
                "optimal_spacing_m": s_opt_m,
                "ring_radius_m": ring_radius_m,
            },
            "_derived": {
                "shielding_factor_eta": eta,
                "field_reduction_fraction": 1 - eta,
                "_ratio": E_effective / E_BREAK_STP,
            },
            "field_model": "needle_array_empirical_shielding",
            "source": "analytic",
        }


class FieldSolver2D:
    def __init__(self,
                 Nr: int = 40,
                 Nz: int = 60,
                 omega: float = 1.7):
        self.Nr = Nr
        self.Nz = Nz
        self.omega = omega

    def solve(self,
              voltage_V: float,
              gap_mm: float,
              emitter_radius_mm: float,
              collector_inner_mm: float,
              emitter_length_mm: float = 20.0,
              housing_radius_mm: float = 20.0,
              rho_e: float = 1e-6,
              max_iter: int = 2000,
              tol: float = 1e-4) -> dict:
        gap = gap_mm / 1000
        r_e = emitter_radius_mm / 1000
        r_col = collector_inner_mm / 1000
        e_len = emitter_length_mm / 1000
        r_max = housing_radius_mm / 1000

        z_max = gap + e_len + 0.01
        dr = r_max / (self.Nr - 1)
        dz = z_max / (self.Nz - 1)

        r = [i * dr for i in range(self.Nr)]
        z = [j * dz for j in range(self.Nz)]

        V = [[0.0] * self.Nz for _ in range(self.Nr)]

        collector_wall = 5e-3
        emitter_r_nodes = max(2, int(r_e / dr) + 1)

        def is_emitter(i, j):
            zj = z[j]
            return i < emitter_r_nodes and gap - dz <= zj <= gap + e_len + dz

        def is_collector(i, j):
            ri = r[i]
            zj = z[j]
            col_i_inner = max(0, int((r_col - collector_wall) / dr))
            col_i_outer = min(self.Nr - 1, int((r_col + collector_wall) / dr) + 1)
            return zj < dz * 1.5 and col_i_inner <= i <= col_i_outer

        for i in range(self.Nr):
            for j in range(self.Nz):
                if is_emitter(i, j):
                    V[i][j] = voltage_V
                elif is_collector(i, j):
                    V[i][j] = 0.0

        source_term = -rho_e / (EPSILON_0 * 1.0006)

        iterations = 0
        residual = 1.0

        for iteration in range(max_iter):
            max_change = 0.0

            for i in range(1, self.Nr - 1):
                for j in range(1, self.Nz - 1):
                    if is_emitter(i, j) or is_collector(i, j):
                        continue

                    ri = r[i]
                    coeff_r_plus = (ri + dr / 2) / (ri * dr * dr)
                    coeff_r_minus = (ri - dr / 2) / (ri * dr * dr)
                    coeff_z = 1.0 / (dz * dz)
                    coeff_centre = -(coeff_r_plus + coeff_r_minus + 2 * coeff_z)

                    V_new = (source_term - (
                        coeff_r_plus * V[i + 1][j] +
                        coeff_r_minus * V[i - 1][j] +
                        coeff_z * V[i][j + 1] +
                        coeff_z * V[i][j - 1]
                    )) / coeff_centre

                    V_sor = V[i][j] + self.omega * (V_new - V[i][j])
                    change = abs(V_sor - V[i][j])
                    if change > max_change:
                        max_change = change
                    V[i][j] = V_sor

            for j in range(self.Nz):
                V[0][j] = V[1][j]

            for j in range(self.Nz):
                V[self.Nr - 1][j] = V[self.Nr - 2][j]

            residual = max_change
            iterations = iteration + 1
            if residual < tol and iteration > 10:
                break

        Er = [[0.0] * self.Nz for _ in range(self.Nr)]
        Ez = [[0.0] * self.Nz for _ in range(self.Nr)]
        Emag = [[0.0] * self.Nz for _ in range(self.Nr)]

        for i in range(1, self.Nr - 1):
            for j in range(1, self.Nz - 1):
                Er[i][j] = -(V[i + 1][j] - V[i - 1][j]) / (2 * dr)
                Ez[i][j] = -(V[i][j + 1] - V[i][j - 1]) / (2 * dz)
                Emag[i][j] = math.sqrt(Er[i][j] ** 2 + Ez[i][j] ** 2)

        for i in range(self.Nr):
            for j in range(self.Nz):
                if i == 0 or i == self.Nr - 1 or j == 0 or j == self.Nz - 1:
                    ir = min(i, self.Nr - 2)
                    ir0 = max(i - 1, 0)
                    jz = min(j, self.Nz - 2)
                    jz0 = max(j - 1, 0)
                    Er[i][j] = -(V[ir + 1][j] - V[ir0][j]) / max((ir + 1 - ir0) * dr, dr)
                    Ez[i][j] = -(V[i][jz + 1] - V[i][jz0]) / max((jz + 1 - jz0) * dz, dz)
                    Emag[i][j] = math.sqrt(Er[i][j] ** 2 + Ez[i][j] ** 2)

        E_peak = max(Emag[i][j] for i in range(self.Nr) for j in range(self.Nz))

        # ===== DEBUG BLOCK =====
        print("\n--- DEBUG: Numerical Solver ---")
        print(f"E_peak (numerical): {E_peak:.3e} V/m")
        print(f"E_avg (expected): {(voltage_V / gap):.3e} V/m")
        print(f"Enhancement factor: {E_peak / (voltage_V / gap):.2f}")
        print("--------------------------------\n")
        # ===== END DEBUG =====

                
        axis_profile = []
        for j in range(0, self.Nz, max(1, self.Nz // 20)):
            axis_profile.append({
                "z_m": z[j],
                "Ez": Ez[0][j] if j < self.Nz else 0.0,
            })

        j_mid = int((gap / 2) / dz)
        j_mid = min(max(j_mid, 0), self.Nz - 1)
        radial_profile = []
        for i in range(0, self.Nr, max(1, self.Nr // 10)):
            radial_profile.append({
                "r_m": r[i],
                "E": Emag[i][j_mid],
            })

        return {
            "physics_outputs": {
                "V": V,
                "Er": Er,
                "Ez": Ez,
                "Emag": Emag,
                "r": r,
                "z": z,
                "E_peak": E_peak,
                "E_average_gap": voltage_V / gap,
                "axis_field_profile": axis_profile,
                "radial_field_profile": radial_profile,
            },
            "solver_info": {
                "iterations": iterations,
                "residual": residual,
            },
            "_derived": {
                "enhancement_factor": E_peak / (voltage_V / gap),
                "_ratio_peak": E_peak / E_BREAK_STP,
            },
            "field_model": "cylindrical_fd_poisson",
            "source": "numerical",
        }


class BoundaryConditions:
    @staticmethod
    def insulation_check(voltage_V: float,
                         creepage_mm: float,
                         material: str = "PETG",
                         humidity_pct: float = 50.0) -> dict:
        dielectric = {
            "PETG": {"strength_kV_mm": 16.0, "eps_r": 3.6, "tracking_index": 600},
            "ABS": {"strength_kV_mm": 15.0, "eps_r": 2.9, "tracking_index": 500},
            "PLA": {"strength_kV_mm": 12.0, "eps_r": 3.0, "tracking_index": 300},
            "Nylon": {"strength_kV_mm": 14.0, "eps_r": 3.5, "tracking_index": 400},
            "Resin": {"strength_kV_mm": 14.0, "eps_r": 3.2, "tracking_index": 350},
        }
        mat = dielectric.get(material, dielectric["PETG"])

        V_kV = voltage_V / 1000
        humidity_factor = 1.0 + (humidity_pct - 50) / 100 * 1.5
        humidity_factor = max(1.0, humidity_factor)

        creepage_req_mm = V_kV * 3.2 * humidity_factor
        wall_min_mm = V_kV / (mat["strength_kV_mm"] / 3.0)

        return {
            "physics_outputs": {
                "voltage_kV": V_kV,
                "humidity_pct": humidity_pct,
                "humidity_derating": humidity_factor,
                "creepage_provided_mm": creepage_mm,
                "creepage_required_mm": creepage_req_mm,
                "wall_thickness_min_mm": wall_min_mm,
                "dielectric_strength_kV_mm": mat["strength_kV_mm"],
                "tracking_index_CTI": mat["tracking_index"],
                "eps_r": mat["eps_r"],
            },
            "_derived": {
                "material": material,
            },
            "field_model": "insulation_surface_bulk_empirical",
            "source": "analytic",
        }


class FieldAnalysis:
    @staticmethod
    def field_map_summary(voltage_V: float,
                          gap_mm: float,
                          emitter_radius_mm: float,
                          collector_inner_mm: float,
                          emitter_length_mm: float = 20.0) -> dict:
        solver = FieldSolver2D(Nr=100, Nz=200, omega=1.7)
        numerical = solver.solve(
            voltage_V=voltage_V,
            gap_mm=gap_mm,
            emitter_radius_mm=emitter_radius_mm,
            collector_inner_mm=collector_inner_mm,
            emitter_length_mm=emitter_length_mm,
        )

        analytic = TipEnhancement.hyperboloid_tip(
            voltage_V, emitter_radius_mm, emitter_length_mm
        )

        return {
            "physics_outputs": {
                "numerical": numerical["physics_outputs"],
                "analytic": analytic["physics_outputs"],
            },
            "_derived": {
                "numerical": numerical["_derived"],
                "analytic": analytic["_derived"],
            },
            "field_model": "field_map_combined",
            "source": "numerical",
        }

    @staticmethod
    def voltage_sweep(gap_mm: float,
                      emitter_radius_mm: float,
                      collector_inner_mm: float,
                      V_start: float = 5000,
                      V_end: float = 40000,
                      V_step: float = 5000) -> dict:
        sweep = []
        V = V_start
        while V <= V_end:
            tip = TipEnhancement.hyperboloid_tip(V, emitter_radius_mm, gap_mm)
            fields = tip["physics_outputs"]
            derived = tip["_derived"]
            sweep.append({
                "voltage_V": V,
                "E_tip_analytic": fields["E_tip_analytic"],
                "E_tip_corrected": fields["E_tip_corrected"],
                "E_avg": fields["E_avg"],
                "E_onset": fields["E_onset"],
                "V_onset": fields["V_onset"],
                "_ratio": derived["_ratio"],
            })
            V += V_step

        return {
            "physics_outputs": {
                "gap_m": gap_mm / 1000,
                "emitter_radius_m": emitter_radius_mm / 1000,
                "collector_inner_m": collector_inner_mm / 1000,
                "sweep": sweep,
            },
            "_derived": {},
            "field_model": "voltage_sweep_hyperboloid",
            "source": "analytic",
        }
