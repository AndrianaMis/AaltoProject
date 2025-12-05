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
def round_overcorr_metrics_multidiscrete(det_prev, det_now, det_next, nx, nz, k_budget=1):
    det_prev = np.asarray(det_prev).reshape(-1)
    det_now  = np.asarray(det_now ).reshape(-1)
    det_next = np.asarray(det_next).reshape(-1)
    nx = np.asarray(nx).reshape(-1).astype(np.int32)
    nz = np.asarray(nz).reshape(-1).astype(np.int32)
    flips = nx + nz
    act_t = flips > 0

    idle_rate        = (act_t & (det_now == 0)).mean()
    improved         = det_next < det_now
    no_improve_rate  = (act_t & (~improved)).mean()
    acted_rate       = act_t.mean()
    eff_act          = (improved & act_t).sum() / max(1, act_t.sum())
    excess_flips     = np.maximum(flips - int(k_budget), 0).mean()

    return dict(
        act_rate=float(acted_rate),
        idle_rate=float(idle_rate),
        no_improve_rate=float(no_improve_rate),
        eff_act=float(eff_act),
        mean_flips=float(flips.mean()),
        mean_excess_flips=float(excess_flips),
    )










def round_overcorr_metrics_discrete(det_prev, det_now, nx, nz):
    """
    det_prev, det_now: (S,) detector counts
    nx,nz: (S,) in {0,1} for this round
    """
    det_prev = np.asarray(det_prev).reshape(-1)
    det_now  = np.asarray(det_now ).reshape(-1)
    act_t    = (np.asarray(nx).reshape(-1) + np.asarray(nz).reshape(-1)) > 0

    acted_rate = act_t.mean()
    idle_rate  = (act_t & (det_prev == 0)).mean()     # acted when prev round was already clean
    improved   = det_now < det_prev                   # effect of action in THIS round
    no_improve = (act_t & (~improved)).mean()
    eff_act    = (improved & act_t).sum() / max(1, act_t.sum())

    return dict(
        act_rate=float(acted_rate),
        idle_rate=float(idle_rate),
        no_improve_rate=float(no_improve),
        eff_act=float(eff_act),
    )




ALPHA_CLEAR   = 0.2 # reward per bit cleared
BETA_PENALTY  = 0.05  # penalty per active bit
LAMBDA_FLIP   = 1e-3  # small penalty per flip (start tiny!)
BUDGET_K      = 1     # allow up to k flips “free-ish” per round
LAMBDA_EXCESS = 5e-3  # stronger penalty for flips beyond k


'''for stage 1'''
# ALPHA_CLEAR   = 0.5  # reward per bit cleared
# BETA_PENALTY  = 0.05  # penalty per active bit
# LAMBDA_FLIP   = 1e-4  # small penalty per flip (start tiny!)
# BUDGET_K      = 1     # allow up to k flips “free-ish” per round
# LAMBDA_EXCESS = 5e-4  # stronger penalty for flips beyond k


# ALPHA_START = 0.03   # reward for cleared bits at ep=0
# ALPHA_END   = 0.20   # reward later in training

# BETA_START  = 0.15   # penalty for active dets at ep=0
# BETA_END    = 0.08   # softer penalty later
  # NOT lower than 0.06

# # reward knobs
# ALPHA_CLEAR   = 0.45   # ↓ a touch (from 0.50)
# BETA_PENALTY  = 0.08   # ↑ (from 0.05) to value active syndrome more
# LAMBDA_FLIP   = 0.003  # ↑ per-flip cost (was 1e-3)
# BUDGET_K      = 0      # important: make any flip count as "excess"
# LAMBDA_EXCESS = 0.015  # ↑ excess cost (will apply to every flip now)

def step_reward(
    obs_prev_round,        # (S, N_det) 0/1 or bool; None for first round
    obs_round,             # (S, N_det) 0/1 or bool
    *,
    nx=0,                  # number of X corrections (scalar or (S,))
    nz=0,                  # number of Z corrections (scalar or (S,))
    alpha: float = ALPHA_CLEAR,
    beta: float  = BETA_PENALTY,
    lam_flip: float = LAMBDA_FLIP,
    k_budget: int   = BUDGET_K,
    lam_excess: float = LAMBDA_EXCESS,
    ep:float =0.0
):
    """
    R_step(per shot) = alpha * (#cleared) - beta * (#active_now)
                       - lam_flip * flips - lam_excess * max(0, flips - k_budget)

    Returns: (S,) float32
    """

    """
    Suggestion: make the alpha and beta depend on the time (round) """
    # alpha_t = ALPHA_START + ep * (ALPHA_END - ALPHA_START)
    # beta_t  = BETA_START  + ep * (BETA_END  - BETA_START)

    # --- cast & shapes ---
    now  = np.asarray(obs_round, dtype=np.float32)          # (S, N_det)
    S    = now.shape[0]
    prev = np.zeros_like(now) if obs_prev_round is None \
           else np.asarray(obs_prev_round, dtype=np.float32)

    # --- detector counts ---
    det_now  = now.sum(axis=1)                              # (S,)
    det_prev = prev.sum(axis=1)                             # (S,)
    cleared  = np.maximum(det_prev - det_now, 0.0)          # (S,)

    # --- flips per shot: accept scalar or (S,) and coerce to (S,) ---
    def _vec(x):
        if isinstance(x, (int, float, np.integer, np.floating)):
            return np.full((S,), float(x), dtype=np.float32)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 1:
            return np.full((S,), float(x[0]), dtype=np.float32)
        if x.size != S:
            # best-effort broadcast/resize
            x = np.resize(x, (S,)).astype(np.float32)
        return x

    nx_v = _vec(nx)                                         # (S,)
    nz_v = _vec(nz)                                         # (S,)
    flips = nx_v + nz_v                                     # (S,)
    excess = np.maximum(flips - float(k_budget), 0.0)       # (S,)

    # --- final reward ---
    r = (alpha * cleared) \
        - (beta * det_now) \
        - (lam_flip * flips) \
        - (lam_excess * excess)

    return r.astype(np.float32)




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
    reward = np.where(logical_error, -5.0, +5.0)
    return reward
