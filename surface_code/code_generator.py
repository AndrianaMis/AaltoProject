import stim

def build_planar_surface_code(d: int, rounds: int):
    circuit = stim.Circuit()
    data_qubits = {}
    ancillas = []
    index = 0

    # Assign data qubits in dxd grid
    for row in range(d):
        for col in range(d):
            data_qubits[(row, col)] = index
            index += 1

    # Assign ancillas (checkerboard pattern)
    ancilla_index = index
    x_stabilizers = []
    z_stabilizers = []

    for row in range(d):
        for col in range(d):
            # Stabilizers between data qubits (offset grid)
            if row < d - 1 or col < d - 1:
                # Determine neighbors based on position
                neighbors = []
                # Check all four sides (only add if in data_qubits)
                for dr, dc in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                    pos = (row + dr, col + dc)
                    if pos in data_qubits:
                        neighbors.append(data_qubits[pos])

                if len(neighbors) > 1:  # Avoid isolated ancillas
                    if (row + col) % 2 == 0:
                        z_stabilizers.append((ancilla_index, neighbors))
                    else:
                        x_stabilizers.append((ancilla_index, neighbors))
                    ancilla_index += 1

    total_qubits = ancilla_index

    # Build X stabilizer measurement subcircuit
    circ_X = stim.Circuit()
    print(f'x-stabs: {x_stabilizers}')
    for anc, targets in x_stabilizers:
        circ_X.append("R", [anc])
        circ_X.append("H", [anc])
        for t in targets:
            circ_X.append("CX", [anc, t])
        circ_X.append("H", [anc])
        circ_X.append("M", [anc])
        circ_X.append("DETECTOR", [stim.target_rec(-1)])

    # Build Z stabilizer measurement subcircuit
    circ_Z = stim.Circuit()
    for anc, targets in z_stabilizers:
        circ_Z.append("R", [anc])
        for t in targets:
            circ_Z.append("CZ", [anc, t])
        circ_Z.append("M", [anc])
        circ_Z.append("DETECTOR", [stim.target_rec(-1)])

    return circ_X, circ_Z, {
        "data_qubits": data_qubits,
        "x_stabilizers": x_stabilizers,
        "z_stabilizers": z_stabilizers,
        "total_qubits": total_qubits
    }


