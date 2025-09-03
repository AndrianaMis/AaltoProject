import numpy as np, stim
from M0 import mask_generator
from helpers import extract_round_template, build_circ_by_round_from_generated

# def run_batched_data_only(circuit, M_data, data_ids):
#     circ_by_round, anc_ids = build_circ_by_round_from_generated(circuit)  # as in our wrapper
#    # print(f'circ by round: {len(circ_by_round)}')
#  #   pe, bet, af=circ_by_round[1]
#    # print(f'pre:\n{pe}\nbet.\n{bet}\nafter:{af}')
#     if M_data.ndim == 2:
#         M_data = M_data[:, :, None]
#     D, R, S = M_data.shape
#     assert D == len(data_ids) and R == len(circ_by_round)
#     pre,_,meas = circ_by_round[0]
#     print(str(meas).splitlines()[:5])  # should start with MR ... then DETECTOR lines

#     Q_total = max([0] + data_ids + anc_ids) + 1
#     sim = stim.FlipSimulator(batch_size=S, num_qubits=Q_total, disable_stabilizer_randomization=True)

#     xmask = np.zeros((Q_total, S), np.bool_)
#     ymask = np.zeros((Q_total, S), np.bool_)
#     zmask = np.zeros((Q_total, S), np.bool_)

#     for r, (pre, between, meas) in enumerate(circ_by_round):
#         # --- inject DATA errors for round r *before* the round’s entangling ---
#         mr = M_data[:, r, :]               # (D, S)
#         xmask[:] = False; ymask[:] = False; zmask[:] = False
#         xmask[data_ids, :] = (mr == 1)     # your codes: X=1, Z=2, Y=3
#         zmask[data_ids, :] = (mr == 2)
#         ymask[data_ids, :] = (mr == 3)
#         if xmask.any(): sim.broadcast_pauli_errors(pauli='X', mask=xmask)
#         if ymask.any(): sim.broadcast_pauli_errors(pauli='Y', mask=ymask)
#         if zmask.any(): sim.broadcast_pauli_errors(pauli='Z', mask=zmask)

#         # then run the round
#         sim.do(pre)
#         # (no second data window yet; between is empty for your split)
#         sim.do(meas)

#     dets = sim.get_detector_flips()    # (num_detectors, S)
#     obs  = sim.get_observable_flips()  # [] for rotated_memory_x (no observables) — expected
#     meas = sim.get_measurement_flips() # (num_measurements, S)
#     return dets, obs, meas



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









# rng = np.random.default_rng(123)
# m = np.zeros((64, 10_000), dtype=int)
# m, events = spatial_clusters(m, 64, 10_000, qubit_nums=list(range(64)),
#                             p_start=0.02, clusters_per_burst=1, rad=2, pr_to_neigh=0.4, rng=rng)
# print("len(clusters) =", len(events)) 
# m, evs=extend_clusters(m, qus=64, rounds=10_000, clusters=events, p_start=0.2)

# print("len(events) =", len(evs)) 

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



def run_batched_data_plus_anc(circuit, M_data, M_anc, data_ids, anc_ids, enable_M0: bool=True, enable_M1: bool=True):
    circ_by_round, anc_ids_auto = build_circ_by_round_from_generated(circuit)
    assert sorted(anc_ids) == sorted(anc_ids_auto)
    R = len(circ_by_round)

    # Normalize shapes
    if M_data.ndim == 2: M_data = M_data[:, :, None]
    if M_anc.ndim  == 2: M_anc  = M_anc[:, :, None]
    D, Rm, S  = M_data.shape
    A, Rm2, S2 = M_anc.shape
    assert D == len(data_ids) and A == len(anc_ids)
    assert Rm == R == Rm2 and S == S2

    Q = max([0] + data_ids + anc_ids) + 1
    sim = stim.FlipSimulator(batch_size=S, num_qubits=Q, disable_stabilizer_randomization=True)

    xmask = np.zeros((Q, S), np.bool_)
    ymask = np.zeros((Q, S), np.bool_)
    zmask = np.zeros((Q, S), np.bool_)

    for r, (pre, _, meas) in enumerate(circ_by_round):
        # --- DATA: inject before entangling ---

        if enable_M0:
            mr = M_data[:, r, :]
            xmask[:] = ymask[:] = zmask[:] = False
            xmask[data_ids, :] = (mr == 1)
            zmask[data_ids, :] = (mr == 2)
            ymask[data_ids, :] = (mr == 3)
            if xmask.any(): sim.broadcast_pauli_errors(pauli='X', mask=xmask)
            if ymask.any(): sim.broadcast_pauli_errors(pauli='Y', mask=ymask)
            if zmask.any(): sim.broadcast_pauli_errors(pauli='Z', mask=zmask)

        sim.do(pre)

        if enable_M1:
        # --- ANCILLAS: inject just before measurement ---
            ma = M_anc[:, r, :]
            xmask[:] = ymask[:] = zmask[:] = False
            xmask[anc_ids, :] = (ma == 1)
            zmask[anc_ids, :] = (ma == 2)
            ymask[anc_ids, :] = (ma == 3)
            if xmask.any(): sim.broadcast_pauli_errors(pauli='X', mask=xmask)
            if ymask.any(): sim.broadcast_pauli_errors(pauli='Y', mask=ymask)
            if zmask.any(): sim.broadcast_pauli_errors(pauli='Z', mask=zmask)

        sim.do(meas)

    dets = sim.get_detector_flips()
    obs  = sim.get_observable_flips()
    meas = sim.get_measurement_flips()
    return dets, obs, meas
