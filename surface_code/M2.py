from typing import List, Tuple, Dict, Optional

import numpy as np
import random
I, X, Z, Y = 0, 1, 2, 3
from helpers import sample_geometric_len, _sample_pair_pauli, _sample_single_pauli, _compose_xz_to_y
#First of all, intialize the mask
def mask_init_m2(E: int, R: int) -> np.ndarray:
    """Return an (E, R, 2) int8 mask; last axis is [q0_code, q1_code]."""
    return np.zeros((E, R, 2), dtype=np.int8)




def m2_t1_same_round_bursts(
    m: np.ndarray,
    E: int,
    R: int,
    *,
    p_start: float = 2e-4,
    burst_gates_min: int = 2,
    burst_gates_max: int = 6,
    pX: float = 0.5,
    pZ: float = 0.5,
    rng: Optional[np.random.Generator] = None
):
    """
    t1: SAME-ROUND multi-gate bursts (crosstalk).
    - With prob p_start per round, pick K gates in that round and apply a Pauli pair to each.
    - K ~ Uniform[burst_gates_min, burst_gates_max].
    """
    if rng is None: rng = np.random.default_rng()
    events = []  # list of (r, gate_indices, [(c0,c1)...])
   # print(f'------------------ #1 Injecting spatial cluster correlations with p_start: {p_start} -------------------------------\n\n')
    for r in range(R):
        if rng.random() < p_start:
            K = int(rng.integers(burst_gates_min, max(burst_gates_min, burst_gates_max) + 1))
            K = min(K, E)
            gates = rng.choice(E, size=K, replace=False).astype(int)
            applied = []
            for e in gates:
                c0, c1 = _sample_pair_pauli(rng, pX=pX, pZ=pZ, no_II=True)
                # compose with existing (allow Y on overlaps)
                m[e, r, 0] = _compose_xz_to_y(int(m[e, r, 0]), c0)
                m[e, r, 1] = _compose_xz_to_y(int(m[e, r, 1]), c1)
                applied.append((int(c0), int(c1)))
            events.append((r, gates.tolist(), applied))
    return m, events










def m2_t2_per_gate_streaks(
    m: np.ndarray,
    E: int,
    R: int,
    *,
    p_start: float = 8e-4,
    gamma: float = 0.8,
    max_len: Optional[int] = None,
    pX: float = 0.5,
    pZ: float = 0.5,
    rng: Optional[np.random.Generator] = None
):
    """
    t2: PER-GATE temporal streaks (primary for class-2).
    - For each gate e, try to start a streak at free rounds with prob p_start.
    - Pick one Pauli pair and extend for L ~ Geometric(gamma).
    """
    if rng is None: rng = np.random.default_rng()
    streaks = []  # list of (e, r_start, L_eff, (c0,c1))
 #   print(f'------------------ #2 Injecting Temporal one-qubit correlations with p_start: {p_start}-------------------------------\n\n')

    for e in range(E):
        t = 0
        while t < R:
            if (m[e, t, 0] != 0) or (m[e, t, 1] != 0):
                t+=1
                continue
            if rng.random() < p_start:
                L = sample_geometric_len(rng, gamma, max_len)
                c0, c1 = _sample_pair_pauli(rng, pX=pX, pZ=pZ, no_II=True)
                L_eff = 0
                for dt in range(L):
                    rr = t + dt
                    if rr >= R:
                        break
                    m[e, rr, 0] = _compose_xz_to_y(int(m[e, rr, 0]), c0)
                    m[e, rr, 1] = _compose_xz_to_y(int(m[e, rr, 1]), c1)
                    L_eff += 1
                if L_eff > 0:
                    streaks.append((e, t, L_eff, (int(c0), int(c1))))
                    t += L_eff
                else:
                    t += 1
            else:
                t += 1
    return m, streaks














def sample_pauli_code(rng: np.random.Generator, pX: float, pZ: float) -> int:
    """
    Draw a single Pauli for the whole streak. Must be exclusive.
    """
    probs = np.array([pX, pZ], dtype=float)
    probs = probs / probs.sum()
    choice = rng.choice([1, 2], p=probs)   # 1:X, 2:Z, 3:Y
    return choice
    

def mask_generator_M2(gates: list[tuple[int,int]], rounds:int, cfg, actives_list: bool=False):
    """
    gates: list of (control, target) qubit indices (data, anc)
    returns M2 mask of shape (E, R, S, 2) with Pauli codes
    """
    E = len(gates)
    M2 = np.zeros((E, rounds, 2), dtype=np.int8)
    c1=False
    c2=False
    c3=False
    c4=False
    rng = np.random.default_rng()
    if cfg["t1"]["enabled"]:
        M2, events= m2_t1_same_round_bursts(m=M2, E=len(gates), R=rounds, p_start= cfg["t1"]["p_start"] )
        # print('\n')
        # for event in events:

        #    print(f'Event---Cluster: {event}')   #mask, (round, (qubits affected))
        # print('\n--------------------------Finished injecting spatial cluster correlations ------------------')
        if len(events):
            c1=True
    if cfg["t2"]["enabled"]:
        M2, streaks= m2_t2_per_gate_streaks(m=M2, E=E, R=rounds, p_start=cfg["t2"]["p_start"])
        # print('\n')
        # for streak in streaks:
        #    print(f'Event---Streak: {streak}')
        # print('\n--------------------------Finished injecting Temporal one-qubit correlations ------------------')
        if len(streaks):
            c2=True
    if cfg["t3"]["enabled"]:
        M2, 
    

    if actives_list:
        return M2, [c1,c2,c3,c3]
    else: 
        return M2 
