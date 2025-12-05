from typing import List, Tuple, Dict, Optional

import numpy as np
import random
I, X, Z, Y = 0, 1, 2, 3
from .helpers import sample_geometric_len, _sample_pair_pauli, _sample_single_pauli, _compose_xz_to_y
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
    rng: Optional[np.random.Generator] = None,
    rng_seed=None

):
    """
    t1: SAME-ROUND multi-gate bursts (crosstalk).
    - With prob p_start per round, pick K gates in that round and apply a Pauli pair to each.
    - K ~ Uniform[burst_gates_min, burst_gates_max].
    """
    if rng is None: rng = np.random.default_rng(rng_seed)
    events = []  # list of (r, gate_indices, [(c0,c1)...])
    #print(f'------------------ #1 Injecting spatial cluster correlations with p_start: {p_start} -------------------------------\n\n')
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
    rng: Optional[np.random.Generator] = None,
    rng_seed=None
):
    """
    t2: PER-GATE temporal streaks (primary for class-2).
    - For each gate e, try to start a streak at free rounds with prob p_start.
    - Pick one Pauli pair and extend for L ~ Geometric(gamma).
    """
    if rng is None: rng = np.random.default_rng(rng_seed)
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





def m2_t3_cluster_ext(
    m: np.ndarray,
    E: int,
    R: int,
    *,
    gamma: float = 0.75,            # geometric continuation prob
    max_len: int | None = None,     # cap on extension length
    lock_pair: bool = True,         # keep same (c0,c1) while extending
    clusters: list | None = None,  # optional: events from t1 to extend; each as (r, gates, applied_pairs)
    p_start: float | None = None,   # optional: also seed new clusters if provided
    pX: float = 0.5,
    pZ: float = 0.5,
    rng: np.random.Generator | None = None,
    rng_seed=None
):
    """
    Extends *gate groups* across subsequent rounds. If t1_events is provided, we
    take those (round r, gate set G) as seeds and extend forward with geometric
    length L~Geom(gamma). If new_seed_p_start is set, we also create extra seeds
    (independent of t1) choosing random groups at a round and extending them.

    Writes onto m in-place and returns (m, events), where events are tuples:
      (seed_round, gate_indices, rounds_extended (list), pair_applied)
    """
    if rng is None: rng = np.random.default_rng(rng_seed)
    events = []

    def _compose(a:int, b:int)->int:
        if a == 0: return b
        if a == 3 or b == 3: return 3
        if (a,b) in [(1,2),(2,1)]: return 3
        return a

    # helper: sample a pair; you can plug your pair-weights here if you have them
    def _sample_pair():
        return _sample_pair_pauli(rng, pX=pX, pZ=pZ, no_II=True)
 #   print(f'------------------ #3 Injecting Temporal one-qubit correlations with p_start: {p_start}-------------------------------\n\n')

    # 1) extend from given t1_events (preferred)
    if clusters:
        for (r0, gates, applied_pairs) in clusters:
            # gates: list[int]; applied_pairs: list[(c0,c1)] aligned with gates
            for idx, e in enumerate(gates):
                c0, c1 = applied_pairs[idx] if lock_pair else _sample_pair()
                L = sample_geometric_len(rng, gamma, max_len)
                rounds_ext = []
                for dt in range(1, L):  # extend AFTER r0 (keep r0 as t1 origin)
                    rr = r0 + dt
                    if rr >= R: break
                    m[e, rr, 0] = _compose(int(m[e, rr, 0]), c0)
                    m[e, rr, 1] = _compose(int(m[e, rr, 1]), c1)
                    rounds_ext.append(rr)
                if rounds_ext:
                    events.append((r0, [int(e)], rounds_ext, (int(c0), int(c1))))
    return m,events










# ====== M2 — Category #4: multi-gate multi-round scattered clusters (non-contiguous) ======

def m2_t4_multi_gate_multi_round_scattered(
    m: np.ndarray,
    E: int,
    R: int,
    *,
    p_start: float = 1.5e-4,        # rare
    G_min: int = 2,
    G_max: int = 6,
    k_min: int = 2,                 # # of rounds ≥2 and non-contiguous
    k_max: int = 5,
    decay_model: str = "power",     # pick scattered times with heavy tail over |Δt|
    decay_n: float = 2.0,
    decay_A: float = 1.0,
    restrict_by_shared: str | None = None,  # "data" | "anc" | None (needs gate_pairs+sets)
    gate_pairs: list[tuple[int,int]] | None = None,
    data_ids: set[int] | None = None,
    anc_ids: set[int] | None = None,
    pX: float = 0.5,
    pZ: float = 0.5,
    rng: np.random.Generator | None = None,
    rng_seed=None
):
    """
    Seeds rare scattered clusters: choose a group of gates and a non-contiguous
    set of rounds (heavy-tail over gaps). Apply (possibly the same) Pauli pair
    to all chosen gates at those rounds.
    """
    if rng is None: rng = np.random.default_rng(rng_seed)
    events = []

    def _compose(a:int, b:int)->int:
        if a == 0: return b
        if a == 3 or b == 3: return 3
        if (a,b) in [(1,2),(2,1)]: return 3
        return a

    def _sample_pair():
        return _sample_pair_pauli(rng, pX=pX, pZ=pZ, no_II=True)

    def _draw_noncontiguous_times(R:int, t0:int, K:int):
        times = np.arange(R)
        pool = times[times != t0]
        if decay_model == "power":
            dt = np.abs(pool - t0).astype(float)
            dt[dt < 1.0] = 1.0
            w = decay_A / (dt ** decay_n)
        else:
            w = np.ones_like(pool, dtype=float)
        w /= w.sum()
        K = min(max(2, K), pool.size + 1)
        extra = rng.choice(pool, size=K-1, replace=False, p=w)
        return np.unique(np.concatenate([[t0], extra])).astype(int)

    def _select_gate_group() -> np.ndarray:
        G = int(rng.integers(max(2, G_min), max(G_min, G_max) + 1))
        if restrict_by_shared and gate_pairs is not None and data_ids is not None and anc_ids is not None:
            pivot = int(rng.integers(0, E))
            a,b = gate_pairs[pivot]
            if restrict_by_shared == "data":
                q = a if a in data_ids else (b if b in data_ids else None)
            elif restrict_by_shared == "anc":
                q = a if a in anc_ids else (b if b in anc_ids else None)
            else:
                q = None
            if q is None:
                cand = np.arange(E)
            else:
                cand = np.array([i for i,(u,v) in enumerate(gate_pairs) if u==q or v==q], dtype=int)
            if cand.size == 0: cand = np.arange(E)
            return rng.choice(cand, size=min(G, cand.size), replace=False)
        else:
            return rng.choice(E, size=min(G, E), replace=False)

    seeds = rng.random(R) < p_start
    for r0 in np.where(seeds)[0]:
        gates = _select_gate_group()
        K = int(rng.integers(max(2, k_min), max(k_min, k_max) + 1))
        times = _draw_noncontiguous_times(R, r0, K)
        c0, c1 = _sample_pair()
        for rr in times:
            for e in gates:
                m[e, rr, 0] = _compose(int(m[e, rr, 0]), c0)
                m[e, rr, 1] = _compose(int(m[e, rr, 1]), c1)
        events.append((int(r0), gates.tolist(), times.tolist(), (int(c0), int(c1))))

    return m, events















def sample_pauli_code(rng: np.random.Generator, pX: float, pZ: float) -> int:
    """
    Draw a single Pauli for the whole streak. Must be exclusive.
    """
    probs = np.array([pX, pZ], dtype=float)
    probs = probs / probs.sum()
    choice = rng.choice([1, 2], p=probs)   # 1:X, 2:Z, 3:Y
    return choice
    

def mask_generator_M2(gates: list[tuple[int,int]], rounds:int, cfg, actives_list: bool=False, seed=None, rng=None):
    """
    gates: list of (control, target) qubit indices (data, anc)
    returns M2 mask of shape (E, R, S, 2) with Pauli codes
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    E = len(gates)
    M2 = np.zeros((E, rounds, 2), dtype=np.int8)
    c1=False
    c2=False
    c3=False
    c4=False
    if cfg["t1"]["enabled"]:
        M2, clusters= m2_t1_same_round_bursts(m=M2, E=len(gates), R=rounds, p_start= cfg["t1"]["p_start"] , rng_seed=seed, rng=rng, burst_gates_max=cfg["t1"]["G_max"], burst_gates_min=cfg["t1"]["G_min"])
        # print('\n')
        # for event in clusters:

        #    print(f'Event---Cluster: {event}')   #mask, (round, (qubits affected))
        # print('\n--------------------------Finished injecting spatial cluster correlations ------------------')
        if len(clusters):
            c1=True
   
    if cfg["t2"]["enabled"]:
        M2, streaks= m2_t2_per_gate_streaks(m=M2, E=E, R=rounds, p_start=cfg["t2"]["p_start"], rng_seed=seed, rng=rng, gamma=cfg["t2"]["gamma"])
        # print('\n')
        # for streak in streaks:
        #    print(f'Event---Streak: {streak}')
        # print('\n--------------------------Finished injecting Temporal one-qubit correlations ------------------')
        if len(streaks):
            c2=True
    if cfg["t3"]["enabled"]:
        if not c1:
            M2, clusters= m2_t1_same_round_bursts(m=M2, E=len(gates), R=rounds, p_start= cfg["t1"]["p_start"] , rng_seed=seed, rng=rng, burst_gates_max=cfg["t1"]["G_max"], burst_gates_min=cfg["t1"]["G_min"])
            c1=c1 or len(clusters)>0
        if c1:
            M2, streaks3=m2_t3_cluster_ext(m=M2, E=len(gates), clusters=clusters, R=rounds, p_start=cfg["t3"]["p_start"], rng_seed=seed, rng=rng, gamma=cfg["t3"]["gamma"])     #< print("greeks are very racist")
            # print(f'{len(clusters)} clusters\n')
            # for st in streaks3:
            #     r,gate, rs, pairs=st
            #     print(f'\tStreak -> r={r}, gate= {gate}, rounds={rs}')
            # print('-------------------- End extending clusters----------------------------\n')
            if len(streaks3):
                c3=True

        

    if actives_list:
        return M2, [c1,c2,c3,c4]
    else: 
        return M2 
