class CircuitField:
    @staticmethod
    def current_from_flux_total(flux_total: float,
                                R_ballast_ohm: float = 0.0,
                                V_supply_V: float = 0.0) -> dict:
        current = flux_total
        voltage_drop = current * R_ballast_ohm
        voltage_discharge = V_supply_V - voltage_drop
        resistance = voltage_drop / current if abs(current) > 1e-30 else 0.0
        impedance = resistance

        return {
            "physics_outputs": {
                "current": current,
                "flux_total": flux_total,
                "voltage_drop": voltage_drop,
                "voltage_discharge": voltage_discharge,
                "resistance": resistance,
                "impedance": impedance,
            },
            "_derived": {
                "_I_canonical": current,
            },
            "model": "flux_integration",
            "source": "analytic",
        }
