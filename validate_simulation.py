
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def run_simulation(single_error, two_error):

    # Fixed test circuit
    qc = QuantumCircuit(3)

    qc.h(0)
    qc.ry(1.2, 1)
    qc.cx(0, 1)
    qc.ry(0.7, 2)
    qc.cx(1, 2)
    qc.rz(0.5, 0)

    # Ideal state
    ideal_state = Statevector.from_instruction(qc)

    # Noise model
    noise_model = NoiseModel()

    if single_error > 0:
        error_1 = depolarizing_error(single_error, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1, ["h", "ry", "rz"]
        )

    if two_error > 0:
        error_2 = depolarizing_error(two_error, 2)
        noise_model.add_all_qubit_quantum_error(
            error_2, ["cx"]
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

    # Check compiled operations
    print("\nCompiled gate operations:")
    print(compiled.count_ops())

    print("\nNoise model instructions:")
    print(noise_model.noise_instructions)

    result = simulator.run(compiled).result()

    noisy_state = DensityMatrix(
        result.data(0)["density_matrix"]
    )

    fidelity = state_fidelity(
        ideal_state,
        noisy_state
    )

    return fidelity


# Experiment 1: Zero noise
print("TEST 1: ZERO NOISE")

fidelity_zero = run_simulation(0.0, 0.0)

print("Fidelity:", fidelity_zero)

assert abs(fidelity_zero - 1.0) < 1e-8

print("PASS: Ideal simulation verified")


# Experiment 2: Low noise
print("\nTEST 2: LOW NOISE")

fidelity_low = run_simulation(0.001, 0.005)

print("Fidelity:", fidelity_low)


# Experiment 3: Higher noise
print("\nTEST 3: HIGHER NOISE")

fidelity_high = run_simulation(0.02, 0.04)

print("Fidelity:", fidelity_high)

assert fidelity_low < 1.0
assert fidelity_high < fidelity_zero

print("\nSIMULATION VALIDATION COMPLETED!")
