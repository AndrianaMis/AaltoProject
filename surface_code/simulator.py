from code_generator import build_planar_surface_code
from visualize import plot_surface_code_lattice
import stim
import stim
import numpy as np

# Build your circuit
d = 5
rounds = 15
circ, meta = build_planar_surface_code(d, rounds, noisy=True)

# List of data qubit indices
data_ids = list(meta["data_qubits"].values())

# # Error probability
# p = 0.05

# # Insert X and Z noise after every round
# noisy_circ = stim.Circuit()
# noisy_circ += circ  # start with your clean circuit

# # For a quick test, append noise after each reset/measurement cycle
# for _ in range(rounds):
#     noisy_circ.append("X_ERROR", data_ids, p)
#     noisy_circ.append("Z_ERROR", data_ids, p)

# # Run
sampler = stim.CompiledDetectorSampler(circ)
res = sampler.sample(shots=1)

# Reshape into (rounds, num_stabilizers)
num_x = len(meta["x_stabilizers"])
num_z = len(meta["z_stabilizers"])
per_round = num_x + num_z
reshaped = res.reshape(rounds, per_round)
print((reshaped[1] != reshaped[0]).sum())  # Should be 0 for noiseless
print((reshaped[2] != reshaped[1]).sum())  # Should be 0 for noiseless

for r in range(rounds):
    print(f"\nRound {r+1}\nX stabs:", reshaped[r][:num_x])
    true_counts_x = reshaped[r, :num_x].sum(axis=0)
    print(f"Z stabs:", reshaped[r][num_x:])
    true_counts_z = (reshaped)[r, num_x:].sum(axis=0)
    print(f'Trues x:{true_counts_x} \t Falses x: {num_x - true_counts_x}')

    print(f'Trues z:{true_counts_z} \t Falses z: {num_z - true_counts_z}')
# plot_surface_code_lattice(
#     d,
#     meta['data_qubits'],
#     meta['x_stabilizers'],
#     meta['z_stabilizers']
# )

