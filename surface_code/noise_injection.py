#This is where the mask M will be generated form tomo 08/08
from typing import List, Tuple, Dict, Optional

import numpy as np
import random as rand
I, X, Z, Y = 0, 1, 2, 3
#First of all, intialize the mask
def mask_init(qubits: int, rounds:int):
    mask=np.zeros((qubits, rounds), dtype=int)
    print('\nError Mask Initialized!\n')
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
    probab: float = 0.02,     # per-round chance to activate spatial bursts
    clusters: int = 1,        # how many clusters when a burst happens
    rad: int = 1,             # 1D neighborhood radius around a seed
    wrap: bool = False,       # treat qubits as a ring if True; else clamp at edges
    pr_to_neigh=0.3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
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
        rng = np.random.default_rng()
    assert m.shape == (qus, rounds), "m must be shape (qus, rounds)"

    events: List[Tuple[int, np.ndarray]] = []

    for t in range(rounds):
        ran=rng.random()
       # print(f'random num: {ran} and prob: {probab}')
        if ran < probab:
            print(f'Start injecting spatial noise in round {t}')
            for _ in range(clusters):
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
                    #print(f'neigh: {neigh}, while seed: {seed}')

                # avoid overwriting anything already set at this round
                neigh = neigh[m[neigh, t] == 0]
                if neigh.size == 0:
                    continue

                
                print(f'Neighbors within radius {rad} on round {t} are: {neigh} and we must add noise with pr to each of them')
                m[seed, t]=-1
                chosen=[]
                chosen.append(seed)
                for n in neigh:

                    if rng.random()< pr_to_neigh:
                        chosen.append(n)
                        print(f'Adding noise to neighbor: {n}')
                        m[n, t] = -1  # mark occupancy (assign Pauli later)

                #the cluster
                events.append((t, np.array(chosen.copy())))
                print(qubit_nums)


    return m, events









#-----------------------------------------------------------------------------------------------------------------------------------------#
#pretty sure it works well, havent added pr to streak continuoing, dk if i should tho 





#2. We will generate temporal errors on a streak model, so a single qubit will have a specific error with pr on streak length l that will be initlized with geometric distribution 




def sample_pauli_code(rng: np.random.Generator, pX: float, pZ: float, pY: float) -> int:
    """
    Draw a single Pauli for the whole streak. Must be exclusive.
    """
    probs = np.array([pX, pZ, pY], dtype=float)
    probs = probs / probs.sum()
    choice = rng.choice([1, 2, 3], p=probs)   # 1:X, 2:Z, 3:Y
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
    p_start: float = 0.05,         # chance to *start* a streak when free
    gamma: float = 0.6,            # geometric continuation parameter
    max_len: Optional[int] = None, # optional cap on streak length
    pX: float = 0.5,               # Pauli type distribution for a streak
    pZ: float = 0.4,
    pY: float = 0.1,
    rng: np.random.Generator | None = None,
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
        rng = np.random.default_rng()
    assert m.shape == (qus, rounds), "m must be shape (qus, rounds)"

    streaks: List[Tuple[int, int, int, int]] = []

    for qi in range(qus):
        t = 0

        while t < rounds:
            # if already occupied (by clusters etc), skip forward
            if m[qi, t] != 0:
                print(f'mask on qubits {qi} and round {t} already occupied')
                t += 1
                continue

            # try to start a streak here
            if rng.random() < p_start:
                print(f'\n\nMight Inject error in round {t} starting from qubit {qi} ')
                L = sample_geometric_len(rng, gamma, max_len) 
                print(f'Sampled streak length l: {L}')
                L+=1
                pauli_code = sample_pauli_code(rng, pX, pZ, pY)
                print(f'The pauli we will apply: {pauli_code}')
                
                L_eff = 0
                # fill until we hit an occupied cell or run out of rounds
                for dt in range(L):
                    tt = t + dt
                    if tt >= rounds:
                        break
                    if m[qi, tt] != 0:        # do not overwrite clusters
                        break
                    print(f'Injecting error in round {tt} ')
                    m[qi, tt] = pauli_code
                    L_eff += 1

                if L_eff > 0:
                    streaks.append((qi, t, L_eff, pauli_code))
                    t += L_eff
                else:
                    # couldn't place (occupied immediately); move on
                    t += 1
            else:
                t += 1

    return m, streaks







#------------------------------------------------------------------------------------------------------------------#





#3. We weill extend the clusters that were generated on 1. on time


def extend_clusters(    
    m: np.ndarray,
    qus: int,
    rounds: int,
    clusters:List[Tuple[int, np.ndarray]] = []
     ):
    




#4. Multi-qubit temporal errors, not nearest neighbors, not on continuoius streaks 


# Each qubit in each round should experience at most one error event. 
# The error types are exclusive: if a location is designated to have a Y error, then only a Y error is injected there (not an X and Y together, for example). 

def mask_generator(qubits:int, rounds:int, datas: List,  spatial_one_round: bool =False, temporal_one_qubit: bool=False, spatio_temporal:bool=False, multi_qubit_temporal: bool=False):
    M=mask_init(qubits=qubits, rounds=rounds)
    clusters=List[Tuple[int, np.ndarray]] = []
    if spatial_one_round:
        m,clusters=spatial_clusters(m=M, qus=qubits, rounds=rounds, qubit_nums=datas, probab=0.2, wrap=False, rad=2, pr_to_neigh=0.4)
        print('/n')
        for event in clusters:
            print(f'Event---Cluster: {event}')   #mask, (round, (qubits affected))

    if temporal_one_qubit:
        m, streaks=temporal_streaks_single_qubit(m=M, qus=qubits, rounds=rounds)    #mask, ((qubit, round, streak length, pauli code))
        print('\n')
        for streak in streaks:
            print(f'Event---Streak: {streak}')
    if spatio_temporal:
        if spatial_clusters:

            m=extend_clusters(m=M, rounds=rounds, qus=qubits, clusters=clusters)
    print(f'\nMask:\n {m}')
    return M

