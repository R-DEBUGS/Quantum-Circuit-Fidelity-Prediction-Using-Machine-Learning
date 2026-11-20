
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# 1. Create the ideal quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 2. Calculate the ideal quantum state
ideal_state = Statevector.from_instruction(qc)

# 3. Create a noise model
noise_model = NoiseModel()

# 1% depolarizing error on Hadamard gates
h_error = depolarizing_error(0.01, 1)

# 2% depolarizing error on CNOT gates
cx_error = depolarizing_error(0.02, 2)

noise_model.add_all_qubit_quantum_error(h_error, ['h'])
noise_model.add_all_qubit_quantum_error(cx_error, ['cx'])

# 4. Create a noisy quantum simulator
simulator = AerSimulator(
    method='density_matrix',
    noise_model=noise_model
)

# 5. Save the density matrix after circuit execution
noisy_circuit = qc.copy()
noisy_circuit.save_density_matrix()

# 6. Transpile and execute
compiled_circuit = transpile(noisy_circuit, simulator)

result = simulator.run(compiled_circuit, shots=1).result()

# 7. Extract the noisy quantum state
noisy_state = DensityMatrix(
    result.data(0)['density_matrix']
)

# 8. Calculate fidelity
fidelity = state_fidelity(ideal_state, noisy_state)

# 9. Display results
print("IDEAL QUANTUM STATE:")
print(ideal_state)

print("\nNOISY QUANTUM STATE:")
print(noisy_state)

print("\nCIRCUIT FIDELITY:")
print(f"{fidelity:.6f}")

print("\nFIDELITY PERCENTAGE:")
print(f"{fidelity * 100:.2f}%")
