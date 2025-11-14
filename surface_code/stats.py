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
        # print(f"\n\n[{label}] {qus}×{rounds} mask")
        # print(f"  overall occupancy p̂ = {p_hat:.6f}  (nnz={nnz}/{total})")
        # print(f"  Pauli counts: X={cnt_X}, Z={cnt_Z}, Y={cnt_Y}")
        # print(f"  longest streak: max={l_max}, mean={l_mean:.2f}")
        # top-5 noisiest qubits / rounds for quick eyeballing
        top_q = np.argsort(-per_qubit)[:5].tolist()
        top_t = np.argsort(-per_round)[:5].tolist()
        # print(f"  top qubits by occupancy: {[(int(q), float(per_qubit[q])) for q in top_q]}")
        # print(f"  top rounds by occupancy: {[(int(t), float(per_round[t])) for t in top_t]}")
        # print(f'  Cat1: {actives[0] }   Cat2: {actives[1]}   Cat3: {actives[2]}     Cat4: {actives[3]}')

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

    # print(f"\n\n[{label}] shape={M.shape}  nnz={nnz_total}  p̂={p_hat:.6f}")
    # print(f"  shots S={S}: empty_fraction={frac_empty:.3f} (target ~ exp(-D*R*p))")
    # print(f"  counts: X={cntX}, Z={cntZ}, Y={cntY}")
    # quick examples
    if show_examples > 0 and nnz_total > 0:
        idx = np.argwhere(nz)
        for i in range(min(show_examples, idx.shape[0])):
            d, r, s = map(int, idx[i])
            # print(f"  example[{i}]: drow={d}, round={r}, shot={s}, code={int(M[d,r,s])}")

    return {
        "shape": (D, R, S),
        "p_hat": p_hat,
        "empty_fraction": frac_empty,
        "per_round_mean": per_round_mean,
        "per_qubit_mean": per_qubit_mean,
        "counts": {"X": cntX, "Z": cntZ, "Y": cntY},
        "per_shot_counts": per_shot_counts,
    }











def measure_m2_mask_stats(M, label="M2", show_examples=10):
    """
    Measure stats for a two-qubit gate error mask:
      M shape: (E, R, 2, S) with codes {0=I,1=X,2=Z,3=Y}.
    """
    M = np.asarray(M)
    if M.ndim != 4:
        raise ValueError(f"Expected 4D M (E,R,2,S); got shape {M.shape}")

    E, R, S, Q = M.shape
    if Q != 2:
        raise ValueError(f"Expected qubit axis of size 2; got {Q}")

    nz = (M != 0)
    nnz_total = int(nz.sum())
    p_hat = nnz_total / (E * R * Q * S)

    # per-shot counts of error sites across all gates/rounds/qubits
    per_shot_counts = nz.reshape(E * R * Q, S).sum(axis=0)
    frac_empty = float((per_shot_counts == 0).mean())

    # per-round marginal: average over gates and the 2 qubits -> shape (R,)
    per_round_mean = nz.mean(axis=(0, 2)).mean(axis=-1)  # eqv: nz.mean(axis=(0,2,3))
    # per-gate marginal: average over rounds and the 2 qubits -> shape (E,)
    per_gate_mean  = nz.mean(axis=(1, 2)).mean(axis=-1)  # eqv: nz.mean(axis=(1,2,3))
    # per-qubit-in-pair marginal (q=0 vs q=1), averaged over gates, rounds, shots -> shape (2,)
    per_pair_qubit_mean = nz.mean(axis=(0,1,3))          # average over E,R,S

    # Pauli composition
    cntX = int((M == 1).sum()); cntZ = int((M == 2).sum()); cntY = int((M == 3).sum())

    # print(f"\n\n[{label}] shape={M.shape}  nnz={nnz_total}  p̂={p_hat:.6f}")
    # print(f"  shots S={S}: empty_fraction={frac_empty:.3f} (target ~ exp(-E*R*2*p))")
    # print(f"  counts: X={cntX}, Z={cntZ}, Y={cntY}")

    # quick examples
    if show_examples > 0 and nnz_total > 0:
        # indices: (e, r, q, s)
        idx = np.argwhere(nz)
        for i in range(min(show_examples, idx.shape[0])):
            e, r, q, s = map(int, idx[i])
            # print(f"  example[{i}]: gate={e}, round={r}, q={q}, shot={s}, code={int(M[e, r, q, s])}")

    return {
        "shape": (E, R, Q, S),
        "p_hat": p_hat,
        "empty_fraction": frac_empty,
        "per_round_mean": per_round_mean,          # (R,)
        "per_gate_mean": per_gate_mean,            # (E,)
        "per_pair_qubit_mean": per_pair_qubit_mean,# (2,)
        "counts": {"X": cntX, "Z": cntZ, "Y": cntY},
        "per_shot_counts": per_shot_counts,        # (S,)
    }



def normalize_m2(M2):
    """Return M2 with shape (E,R,S,2). Accepts (E,R,2), (E,R,S,2) or (E,R,2,S)."""
    M2 = np.asarray(M2)
    if M2.ndim == 3 and M2.shape[-1] == 2:                 # (E,R,2)
        M2 = M2[:, :, None, :]                              # -> (E,R,1,2)
    elif M2.ndim == 4 and M2.shape[2] == 2:                 # (E,R,2,S)
        M2 = np.transpose(M2, (0, 1, 3, 2))                 # -> (E,R,S,2)
    elif M2.ndim == 4 and M2.shape[3] == 2:                 # (E,R,S,2)
        pass
    else:
        raise ValueError(f"Unexpected M2 shape {M2.shape}")
    return M2.astype(np.int8, copy=False)


def m2_stats(M2):
    """
    Reports per-leg and per-site marginals from M2; also quick diagnostics.
    - per-leg p̂_leg: mean( M2 != 0 ) over (E,R,S,legs)
    - per-site p̂_site: mean( any_leg(M2 != 0) ) over (E,R,S)
    """
    M2 = normalize_m2(M2)                 # (E,R,S,2)
    E, R, S, _ = M2.shape

    nonzero_leg  = (M2 != 0)              # (E,R,S,2)
    p_leg        = nonzero_leg.mean()

    nonzero_site = nonzero_leg.any(-1)    # (E,R,S)
    p_site       = nonzero_site.mean()

    # Counts by code, aggregated over legs
    counts = {
        "X": int((M2 == 1).sum()),
        "Z": int((M2 == 2).sum()),
        "Y": int((M2 == 3).sum()),
        "I": int((M2 == 0).sum()),
    }

    # shots with any event, and expected empty fraction from p_site
    shots_with_any = int((nonzero_site.sum(axis=(0,1)) > 0).sum())
    empty_frac     = float((nonzero_site.sum(axis=(0,1)) == 0).mean())
    approx_empty   = float(np.exp(-E * R * p_site))  # crude independence approx over sites

    # a couple of examples, if any
    examples = []
    idx = np.argwhere(M2 != 0)
    for k in range(min(5, len(idx))):
        e, r, s, leg = map(int, idx[k])
        examples.append({"gate": e, "round": r, "shot": s, "leg": leg, "code": int(M2[e,r,s,leg])})

    return {
        "shape": (E, R, S, 2),
        "p_leg": float(p_leg),
        "p_site": float(p_site),
        "nnz": int((M2 != 0).sum()),
        "counts": counts,
        "shots": S,
        "shots_with_any": shots_with_any,
        "empty_fraction": empty_frac,
        "empty_fraction≈exp(-E*R*p_site)": approx_empty,
        "examples": examples,
    }





def summarize_episode(all_action_masks, observables):
    """
    all_action_masks : list of dicts [{'X':xmask, 'Z':zmask}, ...] per round
                       each mask has shape (Q_total, S)
    observables      : np.ndarray (1, S) from env.finish_measure()

    Prints a compact summary of corrections and logical errors.
    """
    # Sum over all rounds
    x_total = sum(mask['X'].sum() for mask in all_action_masks)
    z_total = sum(mask['Z'].sum() for mask in all_action_masks)
    n_rounds = len(all_action_masks)

    # Shots count (S)
    S = observables.shape[1]

    # Per-shot averages
    x_per_shot = x_total / S
    z_per_shot = z_total / S

    # Logical error rate
    logical_errors = observables.sum()
    logical_rate = logical_errors / S


    print("\n\n====== Episode summary ======")
    print(f"Rounds: {n_rounds}, Shots: {S}")
    print(f"Mean corrections per shot: X={x_per_shot:.2f}, Z={z_per_shot:.2f}")
    print(f"Total corrections applied: X={x_total}, Z={z_total}")
    print(f"Logical error rate: {logical_rate:.3%} ({int(logical_errors)}/{S})")
    print("=============================\n")




import numpy as np

def analyze_decoding_stats(dets, obs, meas, M0, M1, M2, rounds, ancillas, circuit, slices):
    print("\n=== Decoding Statistics ===")

    print(f"Any detector flips: {np.any(dets)}")
    print(f"Sample detector flips (first shot): {dets[:,0]}")
    print(f"Sample observations (last step): {obs}")
    print(f"Any measurement flips: {np.any(meas)}")

    # Detector flip rate (fraction of flips over all detectors and shots)
    det_rate = dets.mean()
    print(f"Detector flip rate: {det_rate:.6f}")

    # Shots with any detector flip
    shots_have_flip = dets.any(axis=0)
    print(f"Shots with >=1 detector flip: {shots_have_flip.sum()} / {shots_have_flip.size}")

    # Detectors that fired at least once
    dets_fired = dets.any(axis=1)
    print(f"Detectors that fired >=1 time: {dets_fired.sum()} / {dets_fired.size}")

    # Measurement flip rate as fraction of ones
    meas_rate = meas.mean()
    print(f"Measurement 1-rate: {meas_rate:.6f}")

    print(f"\nMask shapes:")
    print(f" M0 (data) shape: {M0.shape}")
    print(f" M1 (ancilla) shape: {M1.shape}")
    print(f" M2 (two-qubit) shape: {M2.shape}")

    print(f"Dets shape: {dets.shape}")
    print(f"Meas shape: {meas.shape}")

    # Your plotting helper functions
    from visuals import corrs  # Adjust import per your project

    corrs.make_Crr_heatmaps(M0=M0, M1=M1, M2=M2, DET=dets, MR=meas, R=rounds, A=ancillas)

    print(f"\10th shot detector flips:\n{dets[:,10]}")
    print(f"10th shot measurements:\n{meas[:,:,10]}")

    # Analyze per-round detector counts and correlations:
    slices = corrs.detector_round_slices_3(circuit)

    X_det = corrs.per_round_counts_from_dets(dets, slices)  # (R,S)
    C_det = corrs.round_round_corr(X_det)
    corrs.plot_Crr_d(C_det, "C_rr — DET")

    print("\n=== End of Stats ===\n")










def summarize_noise(stats_M0, stats_M1, stats_M2):
    """
    Summarize noise levels from stacked-mask statistics.
    Input: dicts returned by measure_stacked_mask_stats or measure_m2_mask_stats
    Output: summary dict with scalar metrics.
    """

    def _summ_scalar_stats(stats):
        return {
            "p_hat": float(stats["p_hat"]),
            "empty_frac": float(stats["empty_fraction"]),
            "mean_round": float(np.mean(stats["per_round_mean"])),
            "mean_qubit": float(np.mean(stats["per_qubit_mean"])) if "per_qubit_mean" in stats else None,
            "mean_events": float(np.mean(stats["per_shot_counts"])),
            "var_events": float(np.var(stats["per_shot_counts"])),
        }

    summary = {
        "M0": _summ_scalar_stats(stats_M0),
        "M1": _summ_scalar_stats(stats_M1),

        "M2": {
            "p_hat": float(stats_M2["p_hat"]),
            "empty_frac": float(stats_M2["empty_fraction"]),
            "mean_round": float(np.mean(stats_M2["per_round_mean"])),
            "mean_gate": float(np.mean(stats_M2["per_gate_mean"])),
            "mean_pair_qubit": float(np.mean(stats_M2["per_pair_qubit_mean"])),
            "mean_events": float(np.mean(stats_M2["per_shot_counts"])),
            "var_events": float(np.var(stats_M2["per_shot_counts"])),
        }
    }

    # --- Add derived indicators: fano factor and correlation score ---
    for key in ["M0", "M1"]:
        m = summary[key]
        if m["mean_events"] > 0:
            m["fano"] = m["var_events"] / m["mean_events"]
        else:
            m["fano"] = 0.0

    # M2 fano
    if summary["M2"]["mean_events"] > 0:
        summary["M2"]["fano"] = summary["M2"]["var_events"] / summary["M2"]["mean_events"]
    else:
        summary["M2"]["fano"] = 0.0

    # Detector-style "noise level" scalar (easy to plot across episodes)
    summary["noise_level_scalar"] = (
        summary["M0"]["p_hat"] +
        summary["M1"]["p_hat"] +
        summary["M2"]["p_hat"]
    )

    return summary
