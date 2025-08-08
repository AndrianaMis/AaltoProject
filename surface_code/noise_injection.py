#This is where the mask M will be generated form tomo 08/08

import numpy as np

I, X, Z, Y = 0, 1, 2, 3
#First of all, intialize the mask
def mask_init(qubits: int, rounds:int):
    mask=np.zeros((qubits, rounds), dtype=int)
    print('Error Mask Initialized!\n')
    return mask



#1. We will generate spatially correlated errorrs, which are markovian but correlated, maybe with stim or with nearest neighbors modelling 
def spatial_clusters(m, qus: int, rounds:int, probab:float=0.02, clusters:int=1, rad:int=1):
    
    return 
#2. We will generate temporal errors on a streak model, so a single qubit will have a specific error with pr on streak length l that will be initlized with geometric distribution 
#3. We weill extend the clusters that were generated on 1. on time
#4. Multi-qubit temporal errors, not nearest neighbors, not on continuoius streaks 


# Each qubit in each round should experience at most one error event. 
# The error types are exclusive: if a location is designated to have a Y error, then only a Y error is injected there (not an X and Y together, for example). 

def mask_generator(qubits:int, rounds:int, spatial_one_round: bool =False, temporal_one_qubit: bool=False, spatio_temporal:bool=False, multi_qubit_temporal: bool=False):
    M=mask_init(qubits=qubits, rounds=rounds)
    if spatial_one_round:
        spatial_clusters(m=M, qus=qubits, rounds=rounds)
    print(f'Mask:\n {M}')
    return M

