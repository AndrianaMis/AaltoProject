##Apart from the final reward, whehter the logicla state was preserved or not, we will design the step reward, which is givenon every
#round, so that it is not sparse. 
# -Detector activity: penalize the number of triggered detectors in that round (they indicate physical faults not corrected yet).
# -Syndrome resolution: give positive reward if the correction clears detectors compared to the previous round.
# -Stability: penalize consecutive rounds with the same detector firing (temporal streaks left uncorrected).


#Final reward: dominant ±1 outcome for logical success/failure.

# def step_reward(obs_prev_round, obs_round, alpha:float=0.1, beta:float=0.2):
#     det_fired = obs_round.sum(axis=1)     # (S,)
    
#     if obs_prev_round is not None:
#         prev_det_fired = obs_prev_round.sum(axis=1)  # (S,)
        
#         # Did we fix anything? (Positive Reward)
#         cleared = np.maximum(prev_det_fired - det_fired, 0)
#         reward_clear = alpha * cleared
        
#         # Did we make things worse? (Negative Penalty)
#         made_worse = np.maximum(det_fired - prev_det_fired, 0)
#         reward_worse = made_worse * (-beta)
        
#         r = reward_clear + reward_worse
#     else:
#         # First round penalty (just a small constant to encourage starting correction)
#         r = det_fired * (-0.01) 
        
#     return r


# ALPHA_CLEAR = 0.5  # Increased positive signal for clearing syndrome
# BETA_PENALTY = 0.05 # Keep a small, constant penalty for active syndrome (soft time penalty)

# def step_reward(obs_prev_round, obs_round, alpha: float = ALPHA_CLEAR, beta: float = BETA_PENALTY):
#     """
#     Calculates the intermediate step reward based on syndrome reduction.

#     R_step = alpha * (syndrome_cleared) - beta * (syndrome_active)
#     """
#     # obs_round shape: (S, N_det)
    
#     # 1. Negative Reward: Penalty for all currently active syndromes
#     det_fired = obs_round.sum(axis=1, keepdims=True)  # (S, 1) total active syndrome bits per shot
#     reward_step = det_fired * (-beta)

#     # 2. Positive Reward: Reward for syndromes fixed
#     if obs_prev_round is not None:
#         # Calculate how many syndrome bits were active in the previous round but are now clear
#         cleared = ((obs_prev_round == 1) & (obs_round == 0)).sum(axis=1, keepdims=True)
#         reward_clear = alpha * cleared
#     else:
#         reward_clear = np.zeros_like(reward_step)

#     r = reward_clear + reward_step
#     return r.squeeze() # Return shape (S,)




ALPHA_CLEAR   = 0.5   # reward per bit cleared
BETA_PENALTY  = 0.05  # penalty per active bit
LAMBDA_FLIP   = 1e-3  # small penalty per flip (start tiny!)
BUDGET_K      = 1     # allow up to k flips “free-ish” per round
LAMBDA_EXCESS = 5e-3  # stronger penalty for flips beyond k

def step_reward(
    obs_prev_round,
    obs_round,
    *,
    nx: int = 0,
    nz: int = 0,
    alpha: float = ALPHA_CLEAR,
    beta: float  = BETA_PENALTY,
    lam_flip: float = LAMBDA_FLIP,
    k_budget: int   = BUDGET_K,
    lam_excess: float = LAMBDA_EXCESS,
):
    """
    R_step = alpha * (#cleared) - beta * (#active_now)
             - lam_flip * (#flips) - lam_excess * max(0, #flips - k_budget)
    """
    # Active syndrome penalty (S,1)
    det_fired = obs_round.sum(axis=1, keepdims=True)
    r_active  = -beta * det_fired

    # Cleared bonus (S,1)
    if obs_prev_round is not None:
        cleared  = ((obs_prev_round == 1) & (obs_round == 0)).sum(axis=1, keepdims=True)
        r_clear  = alpha * cleared
    else:
        r_clear  = np.zeros_like(r_active)

    # Flip penalties are per-shot scalars; broadcast to (S,1)
    n_flips   = nx + nz
    # per-flip small penalty
    r_flip    = -lam_flip * n_flips
    # extra penalty beyond soft budget k
    excess    = max(0, n_flips - k_budget)
    r_excess  = -lam_excess * excess

    # Combine; broadcast scalar penalties to all shots this round
    r = r_clear + r_active + (r_flip + r_excess)

    # Return (S,)
    return r.squeeze()



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
    logical_error = obs_flips.sum(axis=0) > 0   # (S,)  #np.where(condition, A, B) → element-wise: if condition is True → take A else → take B
    reward = np.where(logical_error, -1.0, +1.0)
    return reward
