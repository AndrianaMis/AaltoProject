import stim


def build_planar_surface_code(d: int, rounds: int, noisy: bool):
    circuit = stim.Circuit()
    data_qubits = {}
    index = 0

    # Assign data qubits in dxd grid
    for row in range(d):
        for col in range(d):
            data_qubits[(row, col)] = index
            index += 1

    ancilla_index = index
    x_stabilizers = []
    z_stabilizers = []

    for row in range(d):
        for col in range(d):
            if row < d - 1 or col < d - 1:
                neighbors = []
                for dr, dc in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                    pos = (row + dr, col + dc)
                    if pos in data_qubits:
                        neighbors.append(data_qubits[pos])
                if len(neighbors) > 1:
                    if (row + col) % 2 == 0:
                        z_stabilizers.append((ancilla_index, neighbors))
                    else:
                        x_stabilizers.append((ancilla_index, neighbors))
                    ancilla_index += 1

    total_qubits = ancilla_index
    num_x = len(x_stabilizers)
    num_z = len(z_stabilizers)

    # Reset all qubits at start
    circuit.append("R", range(total_qubits))
    p=0.08
    data_ids = list(data_qubits.values())  # all data qubit indices

    # Repeat measurement rounds
    for r in range(rounds):
        # --- X stabilizers ---
        for i, (anc, targets) in enumerate(x_stabilizers):
            circuit.append("R", [anc])
            circuit.append("H", [anc])
            for t in targets:
                circuit.append("CNOT", [t,anc])
            circuit.append("H", [anc])
            circuit.append("M", [anc])
            if r == 0:
                # First round: baseline measurement
                circuit.append("DETECTOR", [stim.target_rec(-1)])
            else:

                # Compare to same ancilla's measurement in previous round
                prev_offset = -(num_x + num_z) - 1 +i

                circuit.append("DETECTOR", [
                    stim.target_rec(-1),
                    stim.target_rec(prev_offset)
                ])

        # --- Z stabilizers ---
        for i, (anc, targets) in enumerate(z_stabilizers):
            circuit.append("R", [anc])
            for t in targets:
                circuit.append("CNOT", [t,anc])
            circuit.append("M", [anc])
            if r == 0:
                circuit.append("DETECTOR", [stim.target_rec(-1)])
            else:
                
                prev_offset = -(num_x + num_z) - 1 + i +num_x
                #print(f'prev: {prev_offset}')
                circuit.append("DETECTOR", [
                    stim.target_rec(-1),
                    stim.target_rec(prev_offset)
                ])

        if noisy==True:
            print("Injecting noise!")
            # Inject Markovian noise on data qubits AFTER a full round
            circuit.append("X_ERROR", data_ids, p)
            circuit.append("Z_ERROR", data_ids, p)


    return circuit, {
        "data_qubits": data_qubits,
        "x_stabilizers": x_stabilizers,
        "z_stabilizers": z_stabilizers,
        "total_qubits": total_qubits
    }

