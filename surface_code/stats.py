import numpy as np

def measure_mask_stats(m: np.ndarray, actives, label: str = "M0") -> dict:
    """
    Quick telemetry for a mask M \in {0=I,1=X,2=Z,3=Y}^{qubits x rounds}.
    Returns a dict with overall and per-axis marginals; also prints a short report.
    """
    qus, rounds = m.shape
    total = qus * rounds
    nz = (m != 0)
    nnz = int(nz.sum())

    # Overall occupancy (physical error rate per space-time location)
    p_hat = nnz / total

    # Per-Pauli occupancy
    cnt_X = int((m == 1).sum())
    cnt_Z = int((m == 2).sum())
    cnt_Y = int((m == 3).sum())

    # Per-qubit and per-round marginals (means across time / space)
    per_qubit = nz.mean(axis=1)           # shape (qus,)
    per_round = nz.mean(axis=0)           # shape (rounds,)

    # Simple streak metric: longest consecutive nonzero per qubit
    def longest_run(row):
        best = cur = 0
        for v in row:
            if v != 0:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best
    longest_streaks = np.array([longest_run(m[q, :]) for q in range(qus)])
    l_max = int(longest_streaks.max()) if qus > 0 else 0
    l_mean = float(longest_streaks.mean()) if qus > 0 else 0.0

    report = {
        "label": label,
        "shape": (qus, rounds),
        "occupancy_overall": p_hat,
        "counts": {"X": cnt_X, "Z": cnt_Z, "Y": cnt_Y, "nonzero": nnz, "total": total},
        "per_qubit_mean": per_qubit,   # fraction of rounds with an error, per qubit
        "per_round_mean": per_round,   # fraction of qubits with an error, per round
        "streaks": {"l_max": l_max, "l_mean": l_mean},
    }
    if p_hat>0:
        print(f"\n\n[{label}] {qus}×{rounds} mask")
        print(f"  overall occupancy p̂ = {p_hat:.6f}  (nnz={nnz}/{total})")
        print(f"  Pauli counts: X={cnt_X}, Z={cnt_Z}, Y={cnt_Y}")
        print(f"  longest streak: max={l_max}, mean={l_mean:.2f}")
        # top-5 noisiest qubits / rounds for quick eyeballing
        top_q = np.argsort(-per_qubit)[:5].tolist()
        top_t = np.argsort(-per_round)[:5].tolist()
        print(f"  top qubits by occupancy: {[(int(q), float(per_qubit[q])) for q in top_q]}")
        print(f"  top rounds by occupancy: {[(int(t), float(per_round[t])) for t in top_t]}")
        print(f'  Cat1: {actives[0] }   Cat2: {actives[1]}   Cat3: {actives[2]}     Cat4: {actives[3]}')

       # print(f'Mask: \n{m}')

    return report




def measure_stacked_mask_stats(M, label="M_data_local", show_examples=5):
    M = np.asarray(M)
    if M.ndim == 2:  # single-shot -> add shot axis
        M = M[:, :, None]
    D, R, S = M.shape

    nz = (M != 0)
    nnz_total = int(nz.sum())
    p_hat = nnz_total / (D * R * S)

    # per-shot counts of error sites
    per_shot_counts = nz.reshape(D*R, S).sum(axis=0)
    frac_empty = float((per_shot_counts == 0).mean())

    # per-round and per-qubit marginals (averaged over the other dims)
    per_round_mean = nz.mean(axis=0).mean(axis=1)  # shape (R,)
    per_qubit_mean = nz.mean(axis=1).mean(axis=1)  # shape (D,)

    # Pauli composition
    cntX = int((M == 1).sum()); cntZ = int((M == 2).sum()); cntY = int((M == 3).sum())

    print(f"\n\n[{label}] shape={M.shape}  nnz={nnz_total}  p̂={p_hat:.6f}")
    print(f"  shots S={S}: empty_fraction={frac_empty:.3f} (target ~ exp(-D*R*p))")
    print(f"  counts: X={cntX}, Z={cntZ}, Y={cntY}")
    # quick examples
    if show_examples > 0 and nnz_total > 0:
        idx = np.argwhere(nz)
        for i in range(min(show_examples, idx.shape[0])):
            d, r, s = map(int, idx[i])
            print(f"  example[{i}]: drow={d}, round={r}, shot={s}, code={int(M[d,r,s])}")

    return {
        "shape": (D, R, S),
        "p_hat": p_hat,
        "empty_fraction": frac_empty,
        "per_round_mean": per_round_mean,
        "per_qubit_mean": per_qubit_mean,
        "counts": {"X": cntX, "Z": cntZ, "Y": cntY},
        "per_shot_counts": per_shot_counts,
    }