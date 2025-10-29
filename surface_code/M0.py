#This is where the mask M will be generated form tomo 08/08
from typing import List, Tuple, Dict, Optional

import numpy as np
import random
I, X, Z, Y = 0, 1, 2, 3
#First of all, intialize the mask
def mask_init(qubits: int, rounds:int):
    mask=np.zeros((qubits, rounds), dtype=int)
  #  print('\n\n\nError Mask Initialized!\n')
    return mask

PAULI_TO_CODE: Dict[str, int] = {"X": 1, "Z": 2, "Y": 3}
CODE_TO_PAULI: Dict[int, str] = {v: k for k, v in PAULI_TO_CODE.items()}








# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Pretty sure it works well 




#1. We will generate spatially correlated errorrs, which are markovian but correlated, maybe with stim or with nearest neighbors modelling 

# m is your occupancy mask (binary) of shape (qus, rounds).
# Later you'll map this to Pauli codes {0=I,1=X,2=Z,3=Y}.

def spatial_clusters(
    m: np.ndarray,
    qus: int,
    rounds: int,
    qubit_nums: List,
    p_start: float = 0.002,     # per-round chance to activate spatial bursts
    clusters_per_burst: int = 1,        # how many clusters when a burst happens
    rad: int = 1,             # 1D neighborhood radius around a seed
    wrap: bool = False,       # treat qubits as a ring if True; else clamp at edges
    pr_to_neigh=0.3,
    pX=0.5,
    pZ=0.4,
    rng: np.random.Generator | None = None,
    rng_seed=None
) -> tuple[np.ndarray, List[Tuple[int, np.ndarray, int]]]:
    """
    Insert *one-round* spatial cluster events (Markovian) into m without overwriting
    existing entries. Returns (updated_mask, events), where events is a list of
    (t, qubit_indices) for each cluster actually placed. Use events to extend
    into spatio-temporal bursts later.

    Notes:
     It insert noise on all qubits in the neighborhood without regarding probabilty. SO if wrap == False, onyl then we might get error on only 1 neighbor. 
     Must extend to insert 1 with pr. 
     !!!!Ok, did it! Now with pr_to_neigh the neighbors get errors too
    """
    if rng is None:
        rng = np.random.default_rng(rng_seed)
    assert m.shape == (qus, rounds), "m must be shape (qus, rounds)"

    events: List[Tuple[int, np.ndarray, int]] = []
#    print(f'------------------ #1 Injecting spatial cluster correlations with p_start: {p_start} -------------------------------\n\n')

    for t in range(rounds):
        ran=rng.random()
        if ran < p_start:
#            print(f'Start injecting spatial noise in round {t}')
            for _ in range(clusters_per_burst):
                # pick a seed where the round is still empty
                empty_sites = np.flatnonzero(m[:, t] == 0)
                if empty_sites.size == 0:
                    break
                seed = rng.choice(empty_sites)
                

                # build a neighborhood around the seed
                if wrap:
                    neigh = ((np.arange(seed - rad, seed + rad + 1) + qus) % qus)
                else:
                    lo = max(0, seed - rad)
                    hi = min(qus, seed + rad + 1)
                    neigh = np.arange(lo, hi)

                # avoid overwriting anything already set at this round
                neigh = neigh[m[neigh, t] == 0]
                if neigh.size == 0:
                    continue

                
#                print(f'Neighbors within radius {rad} on round {t} are: {neigh} and we must add noise with pr to each of them')
                pauli_code = sample_pauli_code(rng, pX, pZ)
                m[seed, t]=pauli_code
                chosen=[]
                chosen.append(seed)
                for n in neigh:

            #        if rng.random()< pr_to_neigh:
                    chosen.append(n)
#                        print(f'Adding noise to neighbor: {n}')
                    m[n, t] = pauli_code  

                #the cluster
                events.append((t, np.array(chosen.copy()), int(pauli_code)))


    return m, events









#-----------------------------------------------------------------------------------------------------------------------------------------#
#pretty sure it works well, havent added pr to streak continuoing, dk if i should tho 





#2. We will generate temporal errors on a streak model, so a single qubit will have a specific error with pr on streak length l that will be initlized with geometric distribution 


##Keep in mind that Depolizing noise model: inject X,Z,Y errors with fixed probabilty. 
#But in order to model realistic quantum hardware model, we have to inject onyl X,Z errors and if they collide, then turn it into a Y

def sample_pauli_code(rng: np.random.Generator, pX: float, pZ: float) -> int:
    """
    Draw a single Pauli for the whole streak. Must be exclusive.
    """
    probs = np.array([pX, pZ], dtype=float)
    probs = probs / probs.sum()
    choice = rng.choice([1, 2], p=probs)   # 1:X, 2:Z, 3:Y
    return choice




def sample_geometric_len(rng: np.random.Generator, gamma: float, max_len: Optional[int] = None) -> int:
    """
    L >= 1, P(L=ℓ) = (1-γ) * γ^(ℓ-1). Mean = 1 / (1 - γ).
    """
    L = 1
    while rng.random() < gamma:
        L += 1
        if max_len is not None and L >= max_len:
            break
    return L






def temporal_streaks_single_qubit(
    m: np.ndarray,
    qus: int,
    rounds: int,
    *,
    p_start: float = 0.001,         # chance to *start* a streak when free
    gamma: float = 0.6,            # geometric continuation parameter
    max_len: Optional[int] = None, # optional cap on streak length
    pX: float = 0.5,               # Pauli type distribution for a streak
    pZ: float = 0.5,

    rng: np.random.Generator | None = None,
    rng_seed=None
) -> tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Add **non-Markovian temporal streaks** (per qubit) into the mask `m`
    without overwriting existing entries.

    Inputs
      m:      (qus, rounds) int mask with codes {0=I,1=X,2=Z,3=Y}
      qus:    number of data qubits (rows of m)
      rounds: number of rounds (cols of m)

    Returns
      m: updated mask
      streaks: list of (qi, t_start, L, pauli_code)
               for each placed streak (useful for debugging/metrics).
    """
    if rng is None:
        rng = np.random.default_rng(rng_seed)
    assert m.shape == (qus, rounds), "m must be shape (qus, rounds)"

    streaks: List[Tuple[int, int, int, int]] = []
 #   print(f'------------------ #2 Injecting Temporal one-qubit correlations with p_start: {p_start}-------------------------------\n\n')
    for qi in range(qus):
        t = 0

        while t < rounds:
            # if already occupied (by clusters etc), skip forward
            if m[qi, t] != 0:
#                print(f'mask on qubits {qi} and round {t} already occupied')
                t += 1
                continue

            # try to start a streak here
            if rng.random() < p_start:
                L = sample_geometric_len(rng, gamma, max_len) 
#                print(f'Sampled streak length l: {L}')
                L+=1
                pauli_code = sample_pauli_code(rng, pX, pZ)
#                print(f'The pauli we will apply: {pauli_code}')
                
                L_eff = 0
                # fill until we hit an occupied cell or run out of rounds
                for dt in range(L):
                    tt = t + dt
                    if tt >= rounds:
                        break     
                    current = m[qi, tt]
                    if current == 0:       
                        m[qi, tt] = pauli_code
                    elif (current, pauli_code) in [(1, 2), (2, 1)]:  # X then Z or Z then X                  
                        m[qi, tt] = 3
                        
  #                  print(f'Injecting error in round {tt} ')
                   
                    L_eff += 1

                if L_eff > 0:
                    streaks.append((t, qi, L_eff, int(pauli_code)))
                    t += L_eff
                else:
                    # couldn't place (occupied immediately); move on
                    t += 1
            else:
                t += 1
    
    return m, streaks







#------------------------------------------------------------------------------------------------------------------#
#It does work i think but i have made it putting -2 on the mask just so i know whats happening




#3. We weill extend the clusters that were generated on 1. on time


def extend_clusters(    
    m: np.ndarray,
    qus: int,
    rounds: int,
    clusters:List[Tuple[int, np.ndarray]] = [],  #(round, (qubits affected))
    p_start: float = 0.004,         # chance to *start* a streak when free
    gamma: float = 0.6  ,
    pX=0.5,
    pZ=0.5,
    rng: np.random.Generator | None = None,
    rng_seed=None,
    max_len: Optional[int] = None

     )-> tuple[np.ndarray, List[Tuple[List[Tuple[int, np.ndarray]], int, int]]]:
    
    if rng is None:
        rng = np.random.default_rng(rng_seed)
    assert m.shape == (qus, rounds), "m must be shape (qus, rounds)"
    streaks: List[Tuple[List[Tuple[int, np.ndarray]], int, int]] = []    #[(cluster), length, -2]
    #print(f'------------------  #3 Extending Clusters correlations with p_start: {p_start} -------------------------------\n\n')
    cand = 0; trig = 0
    for cluster in clusters:
        #print(f'\nChecking cluster: {cluster}')
        r, cl_qus, code=cluster
        if r >= rounds-1:
            continue
        cand += 1
        if rng.random() < p_start:  
                trig += 1
  #              print(f'We are gonna extend cluster {cluster}')
                L = sample_geometric_len(rng, gamma, max_len) 
   #             print(f'Sampled streak length l: {L}')
                L+=1
                pauli_code = code

                L_eff = 0
                # fill until we hit an occupied cell or run out of rounds
                for dt in range(L):
                    tt = r + dt
                    if tt >= rounds:
                        break
                    for q in cl_qus:
                        current = m[q, tt]
                        if current == 0:       
                            m[q, tt] = pauli_code
                        elif (current, pauli_code) in [(1, 2), (2, 1)]:  # X then Z or Z then X                  
                            m[q, tt] = 3
                            
 #                   print(f'Injecting error in round {tt} ')
                    L_eff += 1

                if L_eff > 0:
                    strr=[cluster, L_eff, -2]
                    streaks.append(strr)
                    #r += L_eff
    return m, streaks      
       





# ------------------------------------------------------------------------------------------------------#
                
        






#4. Multi-qubit temporal errors, not nearest neighbors, not on continuoius streaks 

def _compose_xz_to_y(existing: int, new: int) -> int:
    """
    Combine Pauli labels at one (qubit, round) without cancellation:
      - If empty, write new.
      - If existing==new or either is Y, keep existing (no double-count/erase).
      - If existing/new are {X,Z} and different, set Y.
    """
    if existing == I:
        return new
    if existing == Y or new == Y:
        return Y  # once Y, keep Y
    # X <-> Z gives Y
    if (existing == X and new == Z) or (existing == Z and new == X):
        return Y
    # same Pauli or other cases: keep the first writer
    return existing

def multi_qubit_multi_round2(
    m: np.ndarray,
    qus: int,
    rounds: int,
    *,
    p_start: float = 0.0002,          # per-(q,t) chance to seed an event
    qset_min: int = 2,
    qset_max: Optional[int] = None,     # cap group size (default below)
    gamma: float = 0.6,                 # continuation prob per next round (geometric length)
    L_max: Optional[int] = None,        # optional hard cap on streak length
    pX: float = 0.5,                    # draw X or Z only; Y arises via overlap
    pZ: float = 0.5,
    disjoint_qubit_groups: bool = False, # if True, a qubit is used by at most one event
    rng: Optional[np.random.Generator] = None,
    rng_seed=None,
    decay_model: str = "power",       # for scattered: "uniform" | "power" 
    decay_n: float = 2.0,             # n in 1/Δt^n or e^{-nΔt}
    decay_A: float = 1.0,             # global weight scale (usually fine at 1.0)
    k_min: int = 2,                   # min number of hits (times) per scattered event (≥2 recommended)
    k_max: int = 5,                   # max number of hits (times) per scattered event
) -> Tuple[np.ndarray, List[Tuple[List[int], int, int, int]]]:
    """
    Multi-qubit, multi-round *streaky* events (Detrimental Sec. III-B):
      - Sample a start (q0,t0) from a uniform probability field.
      - Choose a non-adjacent qubit group Q (includes q0).
      - Choose a Pauli P ∈ {X,Z}.
      - Apply P to all q∈Q at t0, then extend to t0+1, t0+2,... while a Bernoulli(γ) continues.
      - Stop at first occupied conflict (cell already non-I) or at L_max/time horizon.
      - Overlap rule at a cell: X+Z -> Y; keep existing otherwise.
          Decay models for scattered:
      - "uniform":   all other rounds equally likely.
      - "power":     w(Δt) ∝ A / max(Δt,1)^n      (Detrimental pairwise flavor).
    Returns:
      (updated_mask, events) where each event is (Q, t0, L_eff, P).
    """
    if rng is None:
        rng = np.random.default_rng(rng_seed)
    assert m.shape == (qus, rounds)

    if qset_max is None:
        qset_max = min(8, qus)
    qset_min = max(1, min(qset_min, qset_max))

#        print(f'Mask after [{q0,t0}] center injection: \n{m}\n')
    def draw_scattered_times(t0: int, k: int) -> np.ndarray:
        """Pick k distinct times (including t0) with decay-vs-gap weights."""
        if k <= 0:
            return np.array([t0], dtype=int)

        times = np.arange(rounds)
        mask = times != t0
        pool = times[mask]
     
        dt = np.abs(pool - t0).astype(float)
        dt[dt < 1.0] = 1.0  # guard Δt=0 (already excluded)
        if decay_model == "power":
            w = decay_A / (dt ** decay_n)
    
        else:
            # fallback to uniform if unknown
            w = np.ones_like(pool, dtype=float)

        w_sum = w.sum()
        if w_sum <= 0:
            w = np.ones_like(pool, dtype=float)
            w_sum = w.sum()
        w = w / w_sum

        pick = min(max(0, k - 1), pool.size)  # we already include t0 below
        extra = rng.choice(pool, size=pick, replace=False, p=w)
        Tset = np.unique(np.concatenate([[t0], extra])).astype(int)

        return Tset

    # Build start field and sample starts
    starts = rng.random((qus, rounds)) < p_start
 #   print(f'---------------------------- #4 Multi-qubit Multi-round random correlations with p_start: {p_start} ----------------------------------\n\n')

    centers = np.argwhere(starts)
    rng.shuffle(centers)
  #  print(f'Centers: {centers}')
    # Pauli draw (X/Z only)
    probs = np.array([pX, pZ], dtype=float)
    if probs.sum() <= 0:
        probs = np.array([1.0, 0.0])
    probs /= probs.sum()

    events: List[Tuple[List[int], int, int, int]] = []
    used_qubits: set[int] = set() if disjoint_qubit_groups else set()

    for q0, t0 in centers:
        q0 = int(q0); t0 = int(t0)

        # If disallowing reuse and q0 already used, skip
        if disjoint_qubit_groups and q0 in used_qubits:
            continue

        # Choose group Q (non-adjacent allowed)
        if disjoint_qubit_groups:
            candidates = np.array([q for q in range(qus) if q not in used_qubits and q != q0], dtype=int)
        else:
            candidates = np.delete(np.arange(qus), q0)
        if candidates.size == 0 and qset_min > 1:
            continue

        q_size = int(rng.integers(qset_min, qset_max + 1))
        q_size = min(q_size, 1 + candidates.size)  # include q0
        if q_size <= 0:
            continue

        Q = [q0]
        if q_size > 1:
            Q.extend(rng.choice(candidates, size=q_size - 1, replace=False).astype(int).tolist())

#        print(f'How many correlated qubits: {q_size}:\n{Q}')

        # Draw the event Pauli P (X or Z)
        P = int([X, Z][rng.choice([0, 1], p=probs)])

   
        k = int(rng.integers(max(1, k_min), max(k_min, k_max) + 1))
        Tset = draw_scattered_times(t0, k)
 #       print(f'Tset, the rounds we will inejct: {Tset}')
        for t in Tset:
            for q in Q:
                m[q, t] = _compose_xz_to_y(m[q, t], P)
   #             if m[q,t]==3:
  #                  print('\n\n\nGOT Y\n')
        L_eff = len(Tset)
        if L_eff > 0:
            events.append((Q, t0, L_eff, P))
            if disjoint_qubit_groups:
                used_qubits.update(Q)

    return m, events



     # # Streaky extension: contiguous rounds starting at t0
        # t = t0
        # L_eff = 0
        # while t < rounds and (L_max is None or L_eff < L_max):
        #     # Try to write P on all q∈Q at this t; if any conflict is hard (no composition change), we still proceed
        #     wrote_any = False
        #     for q in Q:
        #         new_val = _compose_xz_to_y(m[q, t], P)
        #         if new_val != m[q, t]:
        #             m[q, t] = new_val
        #             wrote_any = True
        #     if wrote_any:
        #         L_eff += 1
        #         t += 1
        #         # Continue with probability gamma
        #         if rng.random() >= gamma:
        #             break
        #     else:
        #         # nothing changed this round (all cells already had same/equivalent), still advance but may stop early
        #         t += 1
        #         L_eff += 1
        #         if rng.random() >= gamma:
        #             break
    

        # if L_eff > 0:
        #     events.append((Q, t0, L_eff, P))
        #     if disjoint_qubit_groups:
        #         used_qubits.update(Q)



# R matrix analogue: p_event plays the role of the marginal start probability; in the paper, each group’s start probability comes from R.
# T matrix analogue: the geometric continuation (gamma) is the streak temporal profile — same as Tkj being 1 for j in [t1,t2]
# Logical OR step: We prevent overwriting existing errors by skipping occupied (q,t) cells — effectively an OR of new events with the existing mask.

# Group correlation: All qubits in group share the same Pauli for the entire streak → multi-qubit, multi-round correlation.

















# Each qubit in each round should experience at most one error event. 
# The error types are exclusive: if a location is designated to have a Y error, then only a Y error is injected there (not an X and Y together, for example). 

def mask_generator( qubits:int, rounds:int, qubits_ind: List,  cfg,actives_list:bool =False, seed=None):
  
    m=mask_init(qubits=qubits, rounds=rounds)
    clusters= []
    spatial_one_round=cfg["t1"]["enabled"]
    temporal_one_qubit=cfg["t2"]["enabled"]
    spatio_temporal=cfg["t3"]["enabled"]
    multi_qubit_temporal=cfg["t4"]["enabled"]

    c1=False
    c2=False
    c3=False
    c4=False
    # for k, wk in zip(("t1","t2","t3","t4"), (0.25, 0.6, 0.1, 0.05)):
    #     if k in cfg and cfg[k].get("enabled", False) and "p_start" in cfg[k]:
    #         cfg[k]["p_start"] *= wk


    if spatial_one_round:
        
        m,clusters=spatial_clusters(m=m, qus=qubits, rounds=rounds, qubit_nums=qubits_ind, p_start=cfg["t1"]["p_start"], wrap=cfg["t1"]["wrap"], rad=cfg["t1"]["rad"], 
                                    pr_to_neigh=cfg["t1"]["pr_to_neigh"], pX=cfg["t1"]["pX"], pZ=cfg["t1"]["pZ"], rng_seed=seed )
        # print('\n')
        # for event in clusters:

        #    print(f'Event---Cluster: {event}')   #mask, (round, (qubits affected))
        # print('\n--------------------------Finished injecting spatial cluster correlations ------------------')

        # print(f'\nMask M0 after spatial Correlations: \n{m}\n\n')

        if len(clusters)>0:
            c1=True

    if temporal_one_qubit:
        m, streaks2=temporal_streaks_single_qubit(m=m, qus=qubits, rounds=rounds, gamma=cfg["t2"]["gamma"], p_start=cfg["t2"]["p_start"],  pX=cfg["t2"]["pX"], pZ=cfg["t2"]["pZ"], rng_seed=seed )    #mask, ((qubit, round, streak length, pauli code))
        # print('\n')
        # for streak in streaks:
        #    print(f'Event---Streak: {streak}')
        # print('\n--------------------------Finished injecting Temporal one-qubit correlations ------------------')
        # print(f'\nMask M0 after temporal Correlations:\n{m}\n\n')

        if len(streaks2)>0:
            c2=True

    if spatio_temporal:
        streaks3=[]
        if not clusters:
            m,clusters=spatial_clusters(m=m, qus=qubits, rounds=rounds, qubit_nums=qubits_ind, p_start=cfg["t1"]["p_start"], wrap=cfg["t1"]["wrap"], rad=cfg["t1"]["rad"], 
                                    pr_to_neigh=cfg["t1"]["pr_to_neigh"], pX=cfg["t1"]["pX"], pZ=cfg["t1"]["pZ"], rng_seed=seed )
            # print('\n')
            # for event in clusters:
            #     print(f'Event---Cluster: {event}')   #mask, (round, (qubits affected))
            
            # print('\n--------------------------Finished injecting spatial cluster correlations ------------------')

            if len(clusters)>0:
            #     print(f"\n\n[cat#1] clusters={len(clusters)} "
            #     f"eligible={sum(1 for (r,_,_) in clusters if r < rounds-1)} "
            #     f"p_ext={cfg['t3']['p_start']}")

            #    print(f'{len(clusters)} extension\n')
                m , streaks3=extend_clusters(m=m, rounds=rounds, qus=qubits, clusters=clusters, p_start=cfg["t3"]["p_start"], gamma=cfg["t3"]["gamma"], pX=cfg["t3"]["pX"], pZ=cfg["t3"]["pZ"], rng_seed=seed )
                c1=True


        if len(streaks3)>0:
            c3=True
        # print('\n--------------------------Finished extending clusters correlations ------------------')
        # print(f'\nMask M0 after cluster extensions:\n{m}\n\n')


## Im not sure if class 0 , data qubits, should be correlated in the streaky model for multi qubit multi rounds. In deterimental.pdf they are explaining 
#thisc cat only for class 1 and 2. 
    if multi_qubit_temporal:
        m, events4=multi_qubit_multi_round2(m=m, qus=qubits, rounds=rounds, p_start=cfg["t4"]["p_start"], gamma=cfg["t4"]["gamma"], qset_min=cfg["t4"]["qset_min"], qset_max=cfg["t4"]["qset_max"], 
                                 pX=cfg["t3"]["pX"], pZ=cfg["t3"]["pZ"], disjoint_qubit_groups=cfg["t4"]["disjoint_qubit_groups"], rng_seed=seed )
        # print('\n--------------------------Finished injecting random multi-qubit multi-round correlations ------------------')
        # print(f'\nMask M0 after multi qubit , multi-round:\n{m}\n\n')

    # print(f'\nFinal Mask M0:\n{m}')

        if len(events4)>0:
            c4=True
        


    if actives_list:
        return m , [c1,c2,c3,c4]
    
    return m






