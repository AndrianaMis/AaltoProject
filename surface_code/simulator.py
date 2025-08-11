from code_generator import build_planar_surface_code
from visualize import plot_surface_code_lattice
import stim
import stim
import numpy as np
from noise_injection import mask_generator

# # Build your circuit
# d = 5
# rounds = 15
# circ, meta = build_planar_surface_code(d, rounds, noisy=False)

# # List of data qubit indices
# data_ids = list(meta["data_qubits"].values())

# # # Error probability
# # p = 0.05

# # # Insert X and Z noise after every round
# # noisy_circ = stim.Circuit()
# # noisy_circ += circ  # start with your clean circuit

# # # For a quick test, append noise after each reset/measurement cycle
# # for _ in range(rounds):
# #     noisy_circ.append("X_ERROR", data_ids, p)
# #     noisy_circ.append("Z_ERROR", data_ids, p)

# # # Run
# sampler = stim.CompiledDetectorSampler(circ)
# res = sampler.sample(shots=1)

# # Reshape into (rounds, num_stabilizers)
# num_x = len(meta["x_stabilizers"])
# num_z = len(meta["z_stabilizers"])
# per_round = num_x + num_z
# reshaped = res.reshape(rounds, per_round)
# print((reshaped[1] != reshaped[0]).sum())  # Should be 0 for noiseless
# print((reshaped[2] != reshaped[1]).sum())  # Should be 0 for noiseless

# for r in range(rounds):
#     print(f"\nRound {r+1}\nX stabs:", reshaped[r][:num_x])
#     true_counts_x = reshaped[r, :num_x].sum(axis=0)
#     print(f"Z stabs:", reshaped[r][num_x:])
#     true_counts_z = (reshaped)[r, num_x:].sum(axis=0)
#     print(f'Trues x:{true_counts_x} \t Falses x: {num_x - true_counts_x}')

#     print(f'Trues z:{true_counts_z} \t Falses z: {num_z - true_counts_z}')
# # plot_surface_code_lattice(
# #     d,
# #     meta['data_qubits'],
# #     meta['x_stabilizers'],
# #     meta['z_stabilizers']
# # )

# print("Num X stabilizers:", len(meta["x_stabilizers"]))
# print("Num Z stabilizers:", len(meta["z_stabilizers"]))
# print(f'Data qubits: {len(meta["data_qubits"])}/25')
# print(f'Total qubits: {meta["total_qubits"]}/49')

# Generate the rotated surface code circuit:
distance = 3
rounds = 10
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=rounds,
    distance=distance,
    after_clifford_depolarization=0.0,
    after_reset_flip_probability=0.0

)
mr_indices = [i for i, instr in enumerate(circuit) if instr.name == 'MR']
for mr in mr_indices:
    print(f'mr: {mr}')
    circuit.insert(mr, stim.CircuitInstruction('CORRELATED_ERROR',  [stim.target_x(1), stim.target_y(1)], [0.5]))


# print(circuit.detector_error_model())
# Compile for fast sampling:
sampler = stim.CompiledDetectorSampler(circuit)
print(f'Total qubits: {circuit.num_qubits} \t Total TICKS: {circuit.num_ticks} \t Total detectors:{circuit.num_detectors} \t Total Measurements: {circuit.num_measurements} \t Total observables: {circuit.num_observables}')

# Sample a batch of detection event outcomes:
shots = 2
stbs=circuit.num_detectors //rounds
result = sampler.sample(shots)  # Result has shape (shots, detectors_per_shot)
res=result.reshape((shots, rounds, stbs))
for r in range(rounds):

    for shot in range(shots):
        x_syndromes = res[shot, r, :(stbs//2)]
        z_syndromes = res[shot, r, (stbs//2):]
        print(f"Shot {shot}, Round {r}:")
        print("  X stabilizers:", x_syndromes)
        print("  Z stabilizers:", z_syndromes)

print(circuit)


print(mr_indices)
#print(circuit.diagram('timeline-text'))
# print(circuit)


sim = stim.FlipSimulator(
    batch_size=rounds,
 
    disable_stabilizer_randomization=True,  # Usually desirable for deterministic injection
)


# for t in range(rounds):
#     for q in range(qubits):
#         pauli = mask[q, t]
#         if pauli:
#             sim.set_pauli_flip(pauli, qubit_index=q, instance_index=t)
# for t in range(rounds):
#     print(f"Round {t}: {sim.peek_pauli_flips(instance_index=t)}")




# print(circuit.num_qubits)
# coords = list(circuit.get_final_qubit_coordinates().items())
# print(coords)
# print(f'All qubits are: {len(coords)}')
# data_qubit_indices = [q for q, (x, y) in coords if (x + y) % 2 == 0]
# print(f'Data qubits ({len(data_qubit_indices)}) indices:{data_qubit_indices}')


# print(f'Data qubits ({len(data)}): {data}\n Anchillas: {anc}\n All qubits: {coor}')




def get_data_and_ancilla_ids_by_parity(circuit: stim.Circuit, distance: int):
    # 1) coords: dict {qid: (x, y)}
    coords = circuit.get_final_qubit_coordinates()
    qids, xy = zip(*coords.items())
    qids = np.array(qids, dtype=int)
    xy = np.array(xy, dtype=int)  # shape (N, 2)

    x, y = xy[:, 0], xy[:, 1]
    both_even = (x % 2 == 0) & (y % 2 == 0)
    both_odd  = (x % 2 == 1) & (y % 2 == 1)

    cand_even = qids[both_even]
    cand_odd  = qids[both_odd]

    d2 = distance * distance
    if len(cand_even) == d2 and len(cand_odd) != d2:
        data_ids = np.sort(cand_even)
    elif len(cand_odd) == d2 and len(cand_even) != d2:
        data_ids = np.sort(cand_odd)
    else:
        # Fallback: pick the parity with size closest to d^2
        # (useful if Stim changes internals; still deterministic)
        if abs(len(cand_even) - d2) <= abs(len(cand_odd) - d2):
            data_ids = np.sort(cand_even)
        else:
            data_ids = np.sort(cand_odd)

    ancilla_ids = np.array(sorted(set(qids) - set(data_ids)), dtype=int)
    return data_ids.tolist(), ancilla_ids.tolist()


d,a=get_data_and_ancilla_ids_by_parity(circuit, 3)
print(f'\n\nDistance: {distance}\nData qubits ({len(d)}): {d}\nAnchillas ({len(a)}): {a}')


###!!!!!!!!!!!!!!!!!!Dont forget to map data qubits to 0-d²-1 !!!!!!!!! 

M=mask_generator(qubits=len(d), rounds=rounds, datas=d, spatial_one_round=True, temporal_one_qubit=True)

# !!!!!!!  This is how we will be able to combine the encoding circuits, once provided, that initalize a logical state, with the already existing circuits of STIM
# def logical_state_prep() -> stim.Circuit:
#     c = stim.Circuit()
#     # Example: apply logical X or other operations to flip logical state
#     # Or build your encoding circuit here
#     # For illustration:
#     c.append("H", [0])  # logical operator acting on physical qubits (example only)
#     return c

# encoding_circuit = logical_state_prep()

# # Combine encoding with QEC rounds
# circ = encoding_circuit + circuit
# # Compile for fast sampling:
# sampler = stim.CompiledDetectorSampler(circ)


# # Pseudocode for integrating custom state preparation
# base_circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=3, 
#                                    after_clifford_depolarization=0.02 )
# # base_circ now contains: Reset data to |0>, ancilla resets, syndrome cycles, final measurements.

# # Remove the automatic data-qubit resets from base_circ, since we’ll handle those.
# # (Stim’s generated circuit starts with lines like “R …” or “RX …” for data qubits.)
# base_circ_without_init = base_circ.copy()
# base_circ_without_init.clear_range(0, 4)

# # Now build a new circuit with our custom init:
# full_circ = stim.Circuit()
# # Reset all physical qubits (or prepare in a simple known state as a baseline)
# full_circ.append_from_stim_program_text("R 0 1 2 ... all data qubit indices ...")
# # Apply your encoding gates to prepare the desired logical state on the data qubits:
# c=stim.Circuit.generated("surface code:rotated_memory_z", distance=5)
# full_circ += c 
# # Now append the rest of the surface code syndrome measurement rounds:
# full_circ += base_circ_without_init
# sampler = stim.CompiledDetectorSampler(full_circ)



'''The idea is that you perform your encoding right after the qubits are fresh/reset, then let the standard stabilizer measurement rounds run. 
Make sure that after your encoding, the system is indeed in a valid code state (i.e. all stabilizer checks would be +1). If your encoding circuit is correct,
 the first round of syndrome measurements should report no errors (all syndromes trivial), since you haven’t introduced any physical errors yet – you’ve just prepared a code state. 
 (If you do see non-trivial syndrome in round 1 with no noise, that signals your preparation left the qubit in a state outside the code space for some stabilizer – you’d want to adjust 
 the encoding procedure in that case.)'''