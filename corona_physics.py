import math

EPSILON_0 = 8.854e-12
E_CHARGE = 1.602e-19
K_BOLTZ = 1.381e-23
T_STP = 293.0

MU_POS = 2.0e-4
MU_NEG = 2.5e-4
MU_ELECTRON = 0.04

E_BREAKDOWN = 3.0e6
PEEK_A = 3.1e6
PEEK_B = 0.030

TOWNSEND_A = 11.2
TOWNSEND_B = 273.8
ATTACH_C = 2.0
ATTACH_D = 13.0


class CoronaPhysics:
    @staticmethod
    def ionization_from_field(E_V_per_m: float,
                              rho_e_C_m3: float,
                              pressure_Pa: float = 101325) -> dict:
        p_torr = pressure_Pa / 133.322
        E_cm = max(abs(E_V_per_m) / 100.0, 1e-6)

        alpha = TOWNSEND_A * p_torr * math.exp(-TOWNSEND_B * p_torr / E_cm) * 100.0
        eta = ATTACH_C * p_torr * math.exp(-ATTACH_D * p_torr / E_cm) * 100.0

        n_e = rho_e_C_m3 / E_CHARGE
        ionization_rate = alpha * n_e * MU_ELECTRON * abs(E_V_per_m)
        attachment_rate = eta * n_e * MU_ELECTRON * abs(E_V_per_m)

        return {
            "physics_outputs": {
                "alpha": alpha,
                "ionization_rate": ionization_rate,
                "attachment_rate": attachment_rate,
            },
            "_derived": {
                "_alpha_canonical": alpha,
            },
            "model": "townsend_local_field",
            "source": "analytic",
        }

    @staticmethod
    def ionization_field(E_r_field: list,
                         E_z_field: list,
                         rho_field: list,
                         pressure_Pa: float = 101325) -> dict:
        Nr = len(E_r_field)
        Nz = len(E_r_field[0]) if Nr > 0 else 0

        alpha_field = [[0.0] * Nz for _ in range(Nr)]
        ionization_rate_field = [[0.0] * Nz for _ in range(Nr)]
        attachment_rate_field = [[0.0] * Nz for _ in range(Nr)]

        alpha_max = 0.0

        p_torr = pressure_Pa / 133.322

        for i in range(Nr):
            for j in range(Nz):
                Er = E_r_field[i][j]
                Ez = E_z_field[i][j]
                rho = rho_field[i][j]
                E_mag = math.sqrt(Er * Er + Ez * Ez)

                E_cm = max(E_mag / 100.0, 1e-6)
                alpha = TOWNSEND_A * p_torr * math.exp(-TOWNSEND_B * p_torr / E_cm) * 100.0
                eta = ATTACH_C * p_torr * math.exp(-ATTACH_D * p_torr / E_cm) * 100.0
                n_e = rho / E_CHARGE

                alpha_field[i][j] = alpha
                ionization_rate_field[i][j] = alpha * n_e * MU_ELECTRON * E_mag
                attachment_rate_field[i][j] = eta * n_e * MU_ELECTRON * E_mag
                if alpha > alpha_max:
                    alpha_max = alpha

        return {
            "physics_outputs": {
                "alpha": alpha_field,
                "ionization_rate": ionization_rate_field,
                "attachment_rate": attachment_rate_field,
            },
            "_derived": {
                "_alpha_canonical": alpha_max,
            },
            "model": "townsend_field_discretized",
            "source": "numerical",
        }
