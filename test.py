from electrostatics import FieldAnalysis

result = FieldAnalysis.field_map_summary(
    voltage_V=20000,
    gap_mm=20,
    emitter_radius_mm=0.1,
    collector_inner_mm=10
)

analytic = result["_derived"]["analytic"]
numerical = result["_derived"]["numerical"]

print("\n--- COMPARISON ---")
print(f"Analytic ratio:   {analytic['_ratio']:.2f}")
print(f"Numerical ratio:  {numerical['_ratio_peak']:.2f}")
print(f"Numerical enhancement: {numerical['enhancement_factor']:.2f}")
print("-------------------\n")