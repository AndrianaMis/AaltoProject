
from M0 import mask_generator
from M1 import mask_generator_M1
#------------------------------------- MARGINALIZATION ----------------------------------------------------------

from typing import List, Tuple, Dict, Optional

import numpy as np

def gamma_from_mean(Lbar: float) -> float:
    """Pick gamma from desired mean streak length Lbar >= 1."""
    Lbar = max(1.0, float(Lbar))
    return 1.0 - 1.0 / Lbar

def first_guess_start_prob(p_idle: float, rho: float, gamma: float) -> float:
    """s ≈ rho * p_idle * (1 - gamma)."""
    return max(0.0, min(1.0, rho * p_idle * (1.0 - gamma)))

def empirical_marginal_rate(masks: list[np.ndarray]) -> float:
    """Average fraction of nonzero cells across a batch of masks."""
    total = sum(M.size for M in masks)
    errors = sum(int(np.count_nonzero(M)) for M in masks)
    return errors / total if total else 0.0


def calibrate_start_rates(
    build_mask_once,   # callable: lambda cfg -> np.ndarray
    cfg,               # nested dict with t1..t4 blocks
    qubits, 
    rounds, 
    qubits_ind,
    p_idle_target: float = 5e-3,
    batch: int = 64,
    iters: int = 4,
    tol: float = 0.15,
    min_scale: float = 0.3,
    max_scale: float = 3.0,
    verbose: bool = True
):
    print('\n\n\n----------------------------Beginning Marginalization of p_start ------------------------------')
    for k in range(iters):
        masks = [build_mask_once(qubits, rounds, qubits_ind, cfg) for _ in range(batch)]
        p_hat = empirical_marginal_rate(masks)
        if verbose:
            print(f"[cal] iter {k}: empirical p_hat={p_hat:.6f} (target {p_idle_target:.6f})")

        scale = (max_scale if p_hat == 0 else np.clip(p_idle_target / p_hat, min_scale, max_scale))

        # Scale each enabled block's p_start if present
        for key in ("t1", "t2", "t3", "t4"):
            if key in cfg and cfg[key].get("enabled", False) and "p_start" in cfg[key]:
                cfg[key]["p_start"] = float(np.clip(cfg[key]["p_start"] * scale, 0.0, 1.0))

        if abs(p_hat - p_idle_target) <= tol * p_idle_target:
            if verbose:
                print(f"[cal] done: within tolerance (±{int(tol*100)}%).")
            break
    print('----------------------------End Marginalization of p_start ------------------------------')
    
    return cfg



def build_m0_once(qubits:int, rounds:int, qubits_ind:List, cfg):

    return mask_generator(
        qubits=qubits,
        rounds=rounds,
        qubits_ind=qubits_ind,
        actives_list=False,
        cfg=cfg

    )

def build_m1_once(qubits:int, rounds:int, qubits_ind:List, cfg):

    return mask_generator_M1(
        qubits=qubits,
        rounds=rounds,
        qubits_ind=qubits_ind,
        actives_list=False,
        cfg=cfg

    )







cfg_data= {
    "mix" : {"w1":0.25, "w2":0.6, "w3":0.1, "w4":0.05},

    "p_idle": 0.005,   # target per-site marginal error rate (to calibrate against)
    "t1": {
        "enabled": True,
        "p_start": 0.0012,
        "rad": 2,
        "clusters_per_burst": 1,
        "wrap": False,
        "pr_to_neigh": 0.3,
        "pX": 0.5,
        "pZ": 0.5,
    },
    "t2": {
        "enabled": True,
        "p_start": 0.0007,
        "gamma": 0.6,
        "pX": 0.5,
        "pZ": 0.5,
    },
    "t3": {
        "enabled": True,
        "p_start": 0.12,
        "gamma": 0.6,
        "pX": 0.5,
        "pZ": 0.5,
    },
    "t4": {
        "enabled":True,
        "p_start": 0.00012,
        "gamma": 0.6,
        "qset_min": 2,
        "qset_max": 5,
        "pX": 0.5,
        "pZ": 0.5,
        "disjoint_qubit_groups": True,
    }
}



cfg_anch = {
    "seed": 0,                 # optional
    "p_idle": 5e-3,            # target marginal per (ancilla, round, shot)

    # Pauli draw for M1 (Z-basis MR): X with 1-fY, Y with fY
    "M1_pauli": {"fY": 0.05},

    # --- #1 Spatial (rare) ---
    # Rough cell contribution per round ~ p_start * (1 + pr_to_neigh*(2*rad)) / A
    "t1": {
        "enabled": True,
        "p_start": 2.0e-4,     # small; keep rare
        "clusters_per_burst": 1,
        "rad": 2,
        "wrap": False,
        "pr_to_neigh": 0.3
    },

    # --- #2 Temporal streaks (PRIMARY) ---
    # Mean streak length ≈ 1/(1-γ) + 1 (because of your +1) → for γ=0.8, ≈ 6
    # So p_start ≈ 0.004 / 6 ≈ 8e-4 gives ≈80–90% of the total p_idle.
    "t2": {
        "enabled": True,
        "p_start": 8.0e-4,
        "gamma": 0.8,
        "max_len": None
    },

    # --- #3 Streaky clusters (off until verified) ---
    "t3": {
        "enabled": False,       # was producing 0 events—disable for now
        "p_start": 3.0e-4,      # placeholder; ignored while disabled
        "gamma": 0.6
    },

    # --- #4 Multi-qubit multi-round scattered (tiny spice) ---
    "t4": {
        "enabled": True,
        "p_start": 1.5e-4,      # very small
        "gamma": 0.6,           # used only if you switch to contiguous extension
        "qset_min": 2,
        "qset_max": 4,
        "disjoint_qubit_groups": False,
        "decay_model": "power",
        "decay_n": 2.0,
        "decay_A": 1.0,
        "k_min": 2,
        "k_max": 5
    }
}
