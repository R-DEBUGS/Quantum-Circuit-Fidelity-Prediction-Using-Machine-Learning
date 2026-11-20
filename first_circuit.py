
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Step 1: Create a circuit with 2 qubits
qc = QuantumCircuit(2)

# Step 2: Apply Hadamard gate to qubit 0
qc.h(0)

# Step 3: Apply CNOT gate
qc.cx(0, 1)

# Step 4: Display the circuit
print("QUANTUM CIRCUIT:")
print(qc.draw())

# Step 5: Calculate the quantum state
state = Statevector.from_instruction(qc)

print("\nQUANTUM STATE:")
print(state)

# Step 6: Display probabilities
print("\nMEASUREMENT PROBABILITIES:")
print(state.probabilities_dict())
