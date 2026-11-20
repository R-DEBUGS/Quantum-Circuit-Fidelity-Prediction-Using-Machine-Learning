
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# Create a fixed 3-qubit circuit
qc = QuantumCircuit(3)

qc.h(0)
qc.ry(1.2, 1)
qc.cx(0, 1)
qc.ry(0.7, 2)
qc.cx(1, 2)
qc.rz(0.5, 0)

# Calculate ideal state
ideal_state = Statevector.from_instruction(qc)

# Noise levels
noise_levels = [
    0.000,
    0.002,
    0.004,
    0.006,
    0.008,
    0.010,
    0.012,
    0.015,
    0.018,
    0.020
]

results = []

print("STARTING NOISE SENSITIVITY EXPERIMENT\n")

for error in noise_levels:

    # Two-qubit error is twice the single-qubit error
    single_error = error
    two_error = min(2 * error, 0.04)

    noise_model = NoiseModel()

    if single_error > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(single_error, 1),
            ["h", "ry", "rz"]
        )

    if two_error > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(two_error, 2),
            ["cx"]
        )

    simulator = AerSimulator(
        method="density_matrix",
        noise_model=noise_model
    )

    noisy_circuit = qc.copy()
    noisy_circuit.save_density_matrix()

    compiled = transpile(
        noisy_circuit,
        simulator,
        optimization_level=0
    )

    # Verify noise coverage for noisy experiments
    if error > 0:
        used_gates = set(compiled.count_ops())
        covered_gates = set(noise_model.noise_instructions)

        circuit_gates = used_gates - {"save_density_matrix"}

        if not circuit_gates.issubset(covered_gates):
            raise RuntimeError(
                f"Uncovered gates: {circuit_gates - covered_gates}"
            )

    result = simulator.run(compiled).result()

    noisy_state = DensityMatrix(
        result.data(0)["density_matrix"]
    )

    fidelity = state_fidelity(
        ideal_state,
        noisy_state
    )

    results.append({
        "single_qubit_error": single_error,
        "two_qubit_error": two_error,
        "fidelity": float(fidelity)
    })

    print(
        f"Error: {error:.3f} | "
        f"Fidelity: {fidelity:.6f}"
    )


# Save results
df = pd.DataFrame(results)

df.to_csv(
    "results/noise_sensitivity.csv",
    index=False
)

# Plot
plt.figure(figsize=(9, 5))

plt.plot(
    df["single_qubit_error"],
    df["fidelity"],
    marker="o"
)

plt.xlabel("Single-Qubit Error Probability")
plt.ylabel("Circuit Fidelity")
plt.title("Quantum Circuit Fidelity vs Noise Level")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/noise_sensitivity.png",
    dpi=300
)

plt.show()

print("\nNOISE SENSITIVITY EXPERIMENT COMPLETED!")
