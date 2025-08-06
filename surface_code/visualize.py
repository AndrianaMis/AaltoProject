
import matplotlib.pyplot as plt

def plot_surface_code_lattice(d, data_qubits, x_stabilizers, z_stabilizers):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Helper: index -> (row, col)
    idx_to_pos = {idx: pos for pos, idx in data_qubits.items()}

    # Plot data qubits
    for (row, col), idx in data_qubits.items():
        ax.plot(col, -row, 'ko', markersize=8)  # black dot
        ax.text(col, -row, str(idx), fontsize=8, ha='center', va='center', color='white')

    # Plot X stabilizers
    for anc, neighbors in x_stabilizers:
        coords = [idx_to_pos[n] for n in neighbors]
        cx = sum(c for _, c in coords) / len(coords)
        cy = -sum(r for r, _ in coords) / len(coords)
        ax.plot(cx, cy, 'rx', markersize=10)
        for (r, c) in coords:
            ax.plot([cx, c], [cy, -r], 'r--', linewidth=0.5)

    # Plot Z stabilizers
    for anc, neighbors in z_stabilizers:
        coords = [idx_to_pos[n] for n in neighbors]
        cx = sum(c for _, c in coords) / len(coords)
        cy = -sum(r for r, _ in coords) / len(coords)
        ax.plot(cx, cy, 'bs', markersize=6)
        for (r, c) in coords:
            ax.plot([cx, c], [cy, -r], 'b--', linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_xticks(range(d))
    ax.set_yticks([-i for i in range(d)])
    ax.grid(True)
    ax.set_title(f"Planar Surface Code (d={d})")
    plt.show()

# plot_surface_code_lattice(
#     d,
#     meta['data_qubits'],
#     meta['x_stabilizers'],
#     meta['z_stabilizers']
# )