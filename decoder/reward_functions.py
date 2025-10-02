##Apart from the final reward, whehter the logicla state was preserved or not, we will design the step reward, which is givenon every
#round, so that it is not sparse. 
# -Detector activity: penalize the number of triggered detectors in that round (they indicate physical faults not corrected yet).
# -Syndrome resolution: give positive reward if the correction clears detectors compared to the previous round.
# -Stability: penalize consecutive rounds with the same detector firing (temporal streaks left uncorrected).


#Final reward: dominant ±1 outcome for logical success/failure.

def step_reward(obs_prev_round, obs_round, alpha:float=0.1,beta:float=0.05):
    det_fired = obs_round.sum(axis=1)     # (S,)

    if obs_prev_round is not None:
        prev_det_fired = obs_prev_round.sum(axis=1)  # (S,)
        cleared = np.maximum(prev_det_fired - det_fired, 0)     #did we fix anything?
        reward_clear = alpha * cleared
    else:
        reward_clear=0

    reward_step=det_fired*(-beta)

    return reward_clear+reward_step



# If detectors disappeared compared to last round → positive reward (+α * cleared).
# If detectors persist → negative penalty (-β * curr_count).
# Tuning α > β biases the agent toward actively fixing instead of ignoring.


import numpy as np

def final_reward(obs_flips: np.ndarray) -> np.ndarray:
    """
    obs_flips: (N_obs, S) logical observable flips (0/1)
    Returns reward per shot (S,)
    """
    # For surface code memory: 1 observable = logical X/Z parity
    # If flip = 1 → logical error
    logical_error = obs_flips.sum(axis=0) > 0   # (S,)
    reward = np.where(logical_error, -1.0, +1.0)
    return reward
