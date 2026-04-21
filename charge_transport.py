import math

EPSILON_0 = 8.854e-12
E_CHARGE = 1.602e-19
K_BOLTZ = 1.381e-23
T_STP = 293.0

MU_POS = 2.0e-4
MU_NEG = 2.5e-4
MU_AVG = 2.0e-4

D_POS = MU_POS * K_BOLTZ * T_STP / E_CHARGE
ALPHA_REC = E_CHARGE * (MU_POS + MU_NEG) / EPSILON_0
BETA_REC = 1e-13


class ChargeTransport:
    @staticmethod
    def solve_transport(E_r_field: list,
                        E_z_field: list,
                        ionization_rate_field: list,
                        r_grid: list,
                        z_grid: list,
                        tip_radius_m: float,
                        gap_m: float,
                        mu: float = MU_AVG,
                        omega: float = 1.5,
                        max_iter: int = 3000,
                        tol: float = 1e-5,
                        electron_fraction: float = 1.0,  # future plasma model split
                        emitter_enhancement_factor: float = 1.0,  # future emission model
                        diffusion_multiplier: float = 1.0) -> dict:  # numerical stabilization tuning
        Nr = len(r_grid)
        Nz = len(z_grid)
        dr = (r_grid[1] - r_grid[0]) if Nr > 1 else 1e-3
        dz = (z_grid[1] - z_grid[0]) if Nz > 1 else 1e-3
        D = mu * K_BOLTZ * T_STP / E_CHARGE

        def is_emitter(i, j):
            return (r_grid[i] <= tip_radius_m * 3.0) and (abs(z_grid[j] - gap_m) < dz * 2.0)

        def is_collector(i, j):
            return z_grid[j] < dz * 1.5

        rho = [[0.0] * Nz for _ in range(Nr)]
        E_mag_max = 0.0
        for i in range(Nr):
            for j in range(Nz):
                Em = math.sqrt(E_r_field[i][j] * E_r_field[i][j] + E_z_field[i][j] * E_z_field[i][j])
                if Em > E_mag_max:
                    E_mag_max = Em
        v_max = mu * E_mag_max
        dt_adv = min(dr, dz) / (v_max + 1e-30)
        dt_diff = 0.25 * min(dr, dz) * min(dr, dz) / (D + 1e-30)
        dt = 0.1 * min(dt_adv, dt_diff)

        iterations = 0
        residual = 1.0

        for iteration in range(max_iter):
            max_change = 0.0
            for i in range(1, Nr - 1):
                for j in range(1, Nz - 1):
                    if is_collector(i, j):
                        continue

                    Er = E_r_field[i][j]
                    Ez = E_z_field[i][j]
                    S = ionization_rate_field[i][j]
                    if is_emitter(i, j):
                        S_eff = S * emitter_enhancement_factor
                    else:
                        S_eff = S
                    r = r_grid[i]
                    r_eff = max(r, 1e-6)
                    Jr_i = mu * rho[i][j] * Er
                    if Er >= 0.0:
                        Jr_im = mu * rho[i - 1][j] * E_r_field[i - 1][j]
                        dJr_dr = (r_eff * Jr_i - r_grid[i - 1] * Jr_im) / (r_eff * dr)
                    else:
                        Jr_ip = mu * rho[i + 1][j] * E_r_field[i + 1][j]
                        dJr_dr = (r_grid[i + 1] * Jr_ip - r_eff * Jr_i) / (r_eff * dr)

                    Jz_i = mu * rho[i][j] * Ez
                    Jz_jm = mu * rho[i][j - 1] * E_z_field[i][j - 1]
                    Jz_jp = mu * rho[i][j + 1] * E_z_field[i][j + 1]
                    if Ez >= 0.0:
                        dJz_dz = (Jz_i - Jz_jm) / dz
                    else:
                        dJz_dz = (Jz_jp - Jz_i) / dz

                    div_J = dJr_dr + dJz_dz

                    laplacian_rho = (
                        (rho[i + 1][j] - 2.0 * rho[i][j] + rho[i - 1][j]) / (dr * dr)
                        + (rho[i][j + 1] - 2.0 * rho[i][j] + rho[i][j - 1]) / (dz * dz)
                    )
                    diffusion_term = (D * diffusion_multiplier) * laplacian_rho
                    n_e = max(rho[i][j] / E_CHARGE, 1e10)
                    recombination = BETA_REC * n_e * n_e * E_CHARGE

                    rho_new = rho[i][j] + dt * (
                        -div_J
                        + diffusion_term
                        + S_eff
                        - recombination
                    )

                    rho_new = max(0.0, rho_new)
                    rho_sor = rho[i][j] + omega * (rho_new - rho[i][j])
                    change = abs(rho_sor - rho[i][j])
                    if change > max_change:
                        max_change = change
                    rho[i][j] = max(0.0, rho_sor)

            for j in range(Nz):
                rho[0][j] = rho[1][j]
                rho[Nr - 1][j] = rho[Nr - 2][j]

            residual = max_change
            iterations = iteration + 1
            if iteration > 10 and max_change < tol:
                break

        rho_max = max(rho[i][j] for i in range(Nr) for j in range(Nz))
        rho_mean = sum(rho[i][j] for i in range(Nr) for j in range(Nz)) / max(Nr * Nz, 1)

        drift_velocity = mu * E_mag_max

        flux_profile = []
        flux_total = 0.0
        for i in range(1, Nr):
            Ez = E_z_field[i][0]
            J = rho[i][0] * mu * Ez
            dA = 2.0 * math.pi * r_grid[i] * dr
            flux_profile.append(J * dA)
            flux_total += J * dA

        ion_density = rho_mean / E_CHARGE
        electron_density = electron_fraction * ion_density

        return {
            "physics_outputs": {
                "rho_e": rho_mean,
                "rho_field": rho,
                "ion_density": ion_density,
                "electron_density": electron_density,
                "drift_velocity": drift_velocity,
                "mobility": mu,
                "flux": flux_total,
                "flux_profile": flux_profile,
            },
            "_derived": {
                "_rho_canonical": rho_max,
                "rho_max": rho_max,
            },
            "model": "drift_diffusion_upwind",
            "source": "numerical",
        }
