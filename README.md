## 6/8/25:


 Created Github repo - thought about structure - began simulating the surface code distance 5 and the circuit measurements. \\ Progress: Worked in gate structure and sequence, get errors when injecting noise, X stabilizers are sketchy cause they have a pattern. Without noise we have all "False"


## 8/8/25: 

Collected info on FLipSImulator -began mask generation - used STIM's CORRELATED ERROR thing - began spatial corrs \\Spatial correlations will happen in clusters that will then be extended chronically. This will be done for data qubtis, since ancillas are getting reseted. Anchilla qubits will experience reset/measurement errors that can actually be correlated (see detrimental paper)


## 11/08/25:

Mask Generation: spatial-clusters ----> Done!   ,   temporal-one_qubit ----> Done!  , spatio_temporal-extend_clusters  -----> In progress


## 13/08/25:

Mask generation: CHose physical nosie channel, got rid of depolarizing-like features (iinjecting Y errors). Now, X+Z=Y

## 15/08/25:

Work on last function for M0. 

## 19/08/25:

Marginalized probabilities for each category based on a non-hardware-aware idle probab (empirical average per-idle-qubit error probability per round). 
Also made a functin for stats.py. Find out whether cat4 function moves the groups to contigeous rounds or not. (shouldnt). 