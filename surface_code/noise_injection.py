#This is where the mask M will be generated form tomo 08/08
from typing import List, Tuple, Dict, Optional

import numpy as np
import random
I, X, Z, Y = 0, 1, 2, 3
#First of all, intialize the mask
def mask_init(qubits: int, rounds:int):
    mask=np.zeros((qubits, rounds), dtype=int)
    print('\n\n\nError Mask Initialized!\n')
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
    pX=0.5,
    pZ=0.4,
    rng: np.random.Generator | None = None,
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
        rng = np.random.default_rng()
    assert m.shape == (qus, rounds), "m must be shape (qus, rounds)"

    events: List[Tuple[int, np.ndarray, int]] = []
    print('------------------ #1 Injecting spatial cluster correlations -------------------------------\n\n')

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
                pauli_code = sample_pauli_code(rng, pX, pZ)
                m[seed, t]=pauli_code
                chosen=[]
                chosen.append(seed)
                for n in neigh:

                    if rng.random()< pr_to_neigh:
                        chosen.append(n)
                        print(f'Adding noise to neighbor: {n}')
                        m[n, t] = pauli_code  

                #the cluster
                events.append((t, np.array(chosen.copy()), int(pauli_code)))
                print(qubit_nums)


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
    print('------------------ #2 Injecting Temporal one-qubit correlations -------------------------------\n\n')
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
                pauli_code = sample_pauli_code(rng, pX, pZ)
                print(f'The pauli we will apply: {pauli_code}')
                
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
                        
                    print(f'Injecting error in round {tt} ')
                   
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
    p_start: float = 0.2,         # chance to *start* a streak when free
    gamma: float = 0.6  ,
    pX=0.5,
    pZ=0.4,
    rng: np.random.Generator | None = None,
    max_len: Optional[int] = None

     )-> tuple[np.ndarray, List[Tuple[List[Tuple[int, np.ndarray]], int, int]]]:
    
    if rng is None:
        rng = np.random.default_rng()
    assert m.shape == (qus, rounds), "m must be shape (qus, rounds)"
    streaks: List[Tuple[List[Tuple[int, np.ndarray]], int, int]] = []    #[(cluster), length, -2]
    print('------------------  #3 Extending Clusters correlations -------------------------------\n\n')

    for cluster in clusters:
        print(f'\nChecking cluster: {cluster}')
        r, cl_qus, code=cluster

        if rng.random() < p_start:  
                print(f'We are gonna extend cluster {cluster}')
                L = sample_geometric_len(rng, gamma, max_len) 
                print(f'Sampled streak length l: {L}')
                L+=1
                pauli_code = sample_pauli_code(rng, pX, pZ)

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
                            
                    print(f'Injecting error in round {tt} ')
                    L_eff += 1

                if L_eff > 0:
                    strr=[cluster, L_eff, -2]
                    streaks.append(strr)
                    #r += L_eff
    return m, streaks      
       





# ------------------------------------------------------------------------------------------------------#
                
        






#4. Multi-qubit temporal errors, not nearest neighbors, not on continuoius streaks 

def multi_qubit_multi_round(
    m: np.ndarray,
    qus: int,
    rounds: int,
    *,
    group_size: int = random.randint(1,5),       # how many qubits in each correlated group
    p_event: float = 0.05,     # probability of starting an event for a given group
    gamma: float = 0.5,        # continuation probability for the streak length
    max_len: Optional[int] = None,
    pX: float = 0.5, pZ: float = 0.5, 
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, list]:
    """
    Multi-qubit, multi-round correlated streaks (Detrimental.pdf style #4).

    Steps:
      1. Partition qubits into groups of `group_size` (can overlap if desired).
      2. For each group, at each potential start round, start a streak with probability `p_event`.
      3. Sample streak length from geometric distribution with parameter gamma.
      4. Assign *same* Pauli to all qubits in group for all rounds in streak.
      5. Do not overwrite any pre-existing nonzero entries in m.

    Returns:
      m: updated mask (0=I,1=X,2=Z,3=Y)
      events: list of (group_qubits, start_round, length, pauli_code)
    """
    if rng is None:
        rng = np.random.default_rng()
    assert m.shape == (qus, rounds)

    events = []
    print('\n-------------------------- #4 Injecting random multi-qubit multi-round correlations ------------------')

    # Step 1: create groups
    groups = []

    for start in range(0, qus, group_size):
        group = list(range(start, min(start + group_size, qus)))
        if len(group) > 1:
            groups.append(group)
    print(f'We have made {len(groups)} groups:\n{ groups}')
    # Step 2: iterate over groups and possible start times
    for g in groups:
        t = 0
        while t < rounds:
            if rng.random() < p_event:
                # Step 3: choose streak length
                L = sample_geometric_len(rng, gamma, max_len)

                # Step 4: choose Pauli for the whole streak
                pauli_code = sample_pauli_code(rng, pX, pZ)

                # Step 5: inject if free
                L_eff = 0
                for dt in range(L):
                    tt = t + dt
                    if tt >= rounds:
                        break
                   
                    for qi in g:
                        current = m[qi, tt]
                        if current == 0:       
                            m[qi, tt] = pauli_code
                        elif (current, pauli_code) in [(1, 2), (2, 1)]:  # X then Z or Z then X                  
                            m[qi, tt] = 3
                            
                    print(f'Injecting error in round {tt} ')
                    L_eff += 1

                if L_eff > 0:
                    events.append((g, t, L_eff, int(pauli_code)))
                    t += L_eff
                else:
                    t += 1
            else:
                t += 1

    return m, events




# How this matches Detrimental.pdf:

# R matrix analogue: p_event plays the role of the marginal start probability; in the paper, each group’s start probability comes from R.
# T matrix analogue: the geometric continuation (gamma) is the streak temporal profile — same as Tkj being 1 for j in [t1,t2]
# Logical OR step: We prevent overwriting existing errors by skipping occupied (q,t) cells — effectively an OR of new events with the existing mask.

# Group correlation: All qubits in group share the same Pauli for the entire streak → multi-qubit, multi-round correlation.

















# Each qubit in each round should experience at most one error event. 
# The error types are exclusive: if a location is designated to have a Y error, then only a Y error is injected there (not an X and Y together, for example). 

def mask_generator(qubits:int, rounds:int, qubits_ind: List,  spatial_one_round: bool =False, temporal_one_qubit: bool=False, spatio_temporal:bool=False, multi_qubit_temporal: bool=False):
    m=mask_init(qubits=qubits, rounds=rounds)
    clusters= []
    if spatial_one_round:
        m,clusters=spatial_clusters(m=m, qus=qubits, rounds=rounds, qubit_nums=qubits_ind, probab=0.2, wrap=False, rad=2, pr_to_neigh=0.4)
        print('\n')
        for event in clusters:
            print(f'Event---Cluster: {event}')   #mask, (round, (qubits affected))
        print('\n--------------------------Finished injecting spatial cluster correlations ------------------')

        print(f'\nMask M0 after spatial Correlations: \n{m}\n\n')

    if temporal_one_qubit:
        m, streaks=temporal_streaks_single_qubit(m=m, qus=qubits, rounds=rounds)    #mask, ((qubit, round, streak length, pauli code))
        print('\n')
        for streak in streaks:
            print(f'Event---Streak: {streak}')
        print('\n--------------------------Finished injecting Temporal one-qubit correlations ------------------')
        print(f'\nMask M0 after temporal Correlations:\n{m}\n\n')

    if spatio_temporal:
        if spatial_clusters:

            extend_clusters(m=m, rounds=rounds, qus=qubits, clusters=clusters)

        else:
            m,clusters=spatial_clusters(m=m, qus=qubits, rounds=rounds, qubit_nums=qubits_ind, probab=0.2, wrap=False, rad=2, pr_to_neigh=0.4)
            print('\n')
            for event in clusters:
                print(f'Event---Cluster: {event}')   #mask, (round, (qubits affected))
            
            print('\n--------------------------Finished injecting spatial cluster correlations ------------------')


            extend_clusters(m=m, rounds=rounds, qus=qubits, clusters=clusters)
        print('\n--------------------------Finished extending clusters correlations ------------------')
        print(f'\nMask M0 after cluster extensions:\n{m}\n\n')

    if multi_qubit_temporal:
        multi_qubit_multi_round(m=m, qus=qubits, rounds=rounds)
        print('\n--------------------------Finished injecting random multi-qubit multi-round correlations ------------------')
        print(f'\nMask M0 after multi qubit , multi-round:\n{m}\n\n')

    print(f'\nFinal Mask M0:\n{m}')
    return m

