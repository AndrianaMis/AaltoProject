# Used only as a log for now



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

## 20/08/25:

Must change func #4 implementing a decay model (ok). Where else should the probabilities decay? Make a draft for FlipSim

## 21/08/25: 

Fix #3 never working (ok) - Re callibrated probabailities againkst 0.005. 
M0 ready

## 22/08/25:

Inject to FlipSim function. 

## 23/08/25:

How will the syndromes be generated? Will it be from shots of running the circui (steps) and then reset and do it again?

## 24/08/25: 

Check exactly to figure out what happening with FlipSim. Start M1 , at least the logic


## 03/09/25:

Inject M1 (ok), start prepping M2, which is gate errors and not qubit errors. 

## 08/09/25: 

Inject M2 (OK), marginalize ok

## 09/09/25: 

#3 and #4 categories of M2.



## 18/09/25: 


Noise injection, syndrome extraction and final details finished. Env class defined and designed. 
(MISSING: SPAM (prefix-suffix) errors (easy)). 
ML model start

## 22/09/25:
Downloaded mamba, started Kalmamba.py 


## 24/09/25:

Started PPO inegration (simple, with no Kalman filtering yet)
### Action space: 2xd+1 (simplest)
- Discrete -> logits: (B, A)
    returns -> actions: (B,) ,   logp:   (B,) (log-prob of chosen action)
- MultiDiscrete -> logits: (B, D, C) (per-qubit categorical)
    loops over the D qubits, samples a Categorical for each -> actions: (B, D) (class per qubit)    , logp: (B,) sum of per-qubit log-probs (needed for PPO)


## 25/09/25:

Does the M0,M1,M2 (esp m0, m1) apply the paulis to the correct indexed qubits? Cuase the masks are len x R   (fixed)
Fix the logic behind injection of errors and injection of corrections (2 separate functions).

## 27/09/25:

Not 2 separate functions. Fixed step_inject in decoder_helpers, so that it injects the corrections (if needed), and then injects the noise and measures. 
Fixed action mask stats, tensors, injection.
! Step Reward shaping ! 