import stim

def build_planar_surface_code(d: int, rounds: int, noisy: bool = False):
    assert d >= 2, "Distance must be at least 2"

    circuit = stim.Circuit()
    data_qubits = {}
    index = 0

    # Step 1: Create d x d data qubits
    for row in range(d):
        for col in range(d):
            data_qubits[(row, col)] = index
            index += 1

    ancilla_index = index
    stabilizers = []
    x_stabilizers = []
    z_stabilizers = []

    # Step 2: Place ancillas centered on plaquettes
    for row in range(d - 1):
        for col in range(d - 1):
            involved = [
                data_qubits[(row, col)],
                data_qubits[(row + 1, col)],
                data_qubits[(row, col + 1)],
                data_qubits[(row + 1, col + 1)]
            ]
            stab_type = "Z" if (row + col) % 2 == 0 else "X"
            stabilizers.append((ancilla_index, involved, stab_type))
            if stab_type == "Z":
                z_stabilizers.append((ancilla_index, involved))
            else:
                x_stabilizers.append((ancilla_index, involved))
            ancilla_index += 1

    total_qubits = ancilla_index
    num_x = len(x_stabilizers)
    num_z = len(z_stabilizers)
    p = 0.08
    data_ids = list(data_qubits.values())

    # Reset all qubits
    circuit.append("R", range(total_qubits))

    for r in range(rounds):
        # --- X stabilizers ---
        for i, (anc, targets) in enumerate(x_stabilizers):
            circuit.append("R", [anc])
            circuit.append("H", [anc])
            for t in targets:
                circuit.append("CNOT", [t, anc])  # data → ancilla
            circuit.append("H", [anc])
            circuit.append("M", [anc])

            if r == 0:
                circuit.append("DETECTOR", [stim.target_rec(-1)])
            else:
                prev_offset = -(num_x + num_z) - 1 + i
                circuit.append("DETECTOR", [
                    stim.target_rec(-1),
                    stim.target_rec(prev_offset)
                ])

        # --- Z stabilizers ---
        for i, (anc, targets) in enumerate(z_stabilizers):
            circuit.append("R", [anc])
            for t in targets:
                circuit.append("CNOT", [anc, t])  # ancilla → data
            circuit.append("M", [anc])

            if r == 0:
                circuit.append("DETECTOR", [stim.target_rec(-1)])
            else:
                prev_offset = -(num_x + num_z) - 1 + i + num_x
                circuit.append("DETECTOR", [
                    stim.target_rec(-1),
                    stim.target_rec(prev_offset)
                ])

        if noisy:
            circuit.append("X_ERROR", data_ids, p)
            circuit.append("Z_ERROR", data_ids, p)

    return circuit, {
        "data_qubits": data_qubits,
        "x_stabilizers": x_stabilizers,
        "z_stabilizers": z_stabilizers,
        "total_qubits": total_qubits
    }
