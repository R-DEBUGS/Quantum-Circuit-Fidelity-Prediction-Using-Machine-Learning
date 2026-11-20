
import os
import random
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# ==========================================
# CONFIGURATION
# ==========================================

NUM_SAMPLES = 1000

MIN_QUBITS = 2
MAX_QUBITS = 5

MIN_DEPTH = 2
MAX_DEPTH = 8

OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "quantum_fidelity_dataset.csv"
)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==========================================
# GENERATE RANDOM QUANTUM CIRCUIT
# ==========================================

def generate_random_circuit(num_qubits, depth):

    qc = QuantumCircuit(num_qubits)

    h_count = 0
    ry_count = 0
    rz_count = 0
    cx_count = 0

    for _ in range(depth):

        # Apply random single-qubit gates
        for qubit in range(num_qubits):

            gate = random.choice(["h", "ry", "rz"])

            if gate == "h":
                qc.h(qubit)
                h_count += 1

            elif gate == "ry":
                angle = random.uniform(0, 2 * np.pi)
                qc.ry(angle, qubit)
                ry_count += 1

            elif gate == "rz":
                angle = random.uniform(0, 2 * np.pi)
                qc.rz(angle, qubit)
                rz_count += 1

        # Apply random two-qubit CNOT gates
        for control in range(num_qubits - 1):

            if random.random() < 0.8:

                target = control + 1

                qc.cx(control, target)
                cx_count += 1

    total_gates = h_count + ry_count + rz_count + cx_count

    return (
        qc,
        h_count,
        ry_count,
        rz_count,
        cx_count,
        total_gates
    )


# ==========================================
# SIMULATE NOISY CIRCUIT AND CALCULATE FIDELITY
# ==========================================

def simulate_fidelity(qc, single_error, two_error):

    """Simulate a noisy circuit and calculate state fidelity."""

    # Calculate ideal quantum state
    ideal_state = Statevector.from_instruction(qc)

    # Create noise model
    noise_model = NoiseModel()

    one_qubit_noise = depolarizing_error(single_error, 1)
    two_qubit_noise = depolarizing_error(two_error, 2)

    # Apply single-qubit noise
    noise_model.add_all_qubit_quantum_error(
        one_qubit_noise,
        ["h", "ry", "rz"]
    )

    # Apply two-qubit noise
    noise_model.add_all_qubit_quantum_error(
        two_qubit_noise,
        ["cx"]
    )

    # Create density matrix simulator
    simulator = AerSimulator(
        method="density_matrix",
        noise_model=noise_model
    )

    # Copy circuit and save density matrix
    noisy_circuit = qc.copy()
    noisy_circuit.save_density_matrix()

    # Transpile without optimization
    compiled = transpile(
        noisy_circuit,
        simulator,
        optimization_level=0
    )

    # Verify original circuit gates are covered
    original_gates = {
        instruction.operation.name
        for instruction in qc.data
    }

    noisy_gates = set(noise_model.noise_instructions)

    uncovered = original_gates - noisy_gates

    if uncovered:
        raise ValueError(
            f"Some original gates have no noise applied: {uncovered}"
        )

    # Execute noisy circuit
    result = simulator.run(
        compiled,
        shots=1
    ).result()

    # Extract noisy density matrix
    noisy_state = DensityMatrix(
        result.data(0)["density_matrix"]
    )

    # Calculate fidelity
    fidelity = state_fidelity(
        ideal_state,
        noisy_state
    )

    return float(fidelity)


# ==========================================
# GENERATE DATASET
# ==========================================

def generate_dataset():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = []

    print("=" * 50)
    print("QUANTUM CIRCUIT FIDELITY DATASET GENERATION")
    print("=" * 50)

    print(f"Total samples: {NUM_SAMPLES}")
    print("Generating quantum circuits...\n")

    for i in range(NUM_SAMPLES):

        # Random circuit parameters
        num_qubits = random.randint(
            MIN_QUBITS,
            MAX_QUBITS
        )

        depth = random.randint(
            MIN_DEPTH,
            MAX_DEPTH
        )

        # Generate circuit
        (
            qc,
            h_count,
            ry_count,
            rz_count,
            cx_count,
            total_gates
        ) = generate_random_circuit(
            num_qubits,
            depth
        )

        # Random noise parameters
        single_error = random.uniform(
            0.001,
            0.020
        )

        two_error = random.uniform(
            0.005,
            0.040
        )

        # Simulate fidelity
        fidelity = simulate_fidelity(
            qc,
            single_error,
            two_error
        )

        # Store extracted features and target
        record = {
            "qubit_count": num_qubits,
            "circuit_depth": qc.depth(),
            "total_gates": total_gates,
            "h_count": h_count,
            "ry_count": ry_count,
            "rz_count": rz_count,
            "cx_count": cx_count,
            "single_qubit_error": single_error,
            "two_qubit_error": two_error,
            "fidelity": fidelity
        }

        dataset.append(record)

        # Progress update
        if (i + 1) % 100 == 0:
            print(
                f"Generated {i + 1}/{NUM_SAMPLES} samples"
            )

    # Convert records into DataFrame
    df = pd.DataFrame(dataset)

    # Save dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 50)
    print("DATASET GENERATION COMPLETED")
    print("=" * 50)

    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print(f"Dataset shape: {df.shape}")

    print("\nFidelity statistics:")
    print(df["fidelity"].describe())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst five records:")
    print(df.head())


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    generate_dataset()
