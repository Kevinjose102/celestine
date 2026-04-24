from electrostatics import FieldSolver2D

def run_rho_sweep():
    voltage = 20000
    gap = 20
    radius = 0.1
    collector = 10

    rhos = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    print("\n=== RHO SWEEP TEST ===\n")

    for rho in rhos:
        solver = FieldSolver2D(Nr=100, Nz=150, omega=1.7)

        result = solver.solve(
            voltage_V=voltage,
            gap_mm=gap,
            emitter_radius_mm=radius,
            collector_inner_mm=collector,
            rho_e=rho
        )

        derived = result["_derived"]

        enhancement = derived["enhancement_factor"]
        ratio = derived["_ratio_peak"]

        print(f"rho_e = {rho:.1e}")
        print(f"  Enhancement: {enhancement:.2f}")
        print(f"  Ratio:       {ratio:.2f}")
        print("---------------------------")

if __name__ == "__main__":
    run_rho_sweep()