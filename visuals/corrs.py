import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# -------------------- core utilities --------------------

def standardize_to_NRS(
    E: np.ndarray,
    round_axis: int,
    shot_axis: int,
) -> np.ndarray:
    """
    Return E as (N, R, S):
      - move round_axis -> -2
      - move shot_axis  -> -1
      - flatten all leading dims into N
    """
    E = np.asarray(E)
    E = np.moveaxis(E, (round_axis, shot_axis), (-2, -1))
    N = int(np.prod(E.shape[:-2]))
    R, S = E.shape[-2], E.shape[-1]
    return E.reshape(N, R, S)

def reshape_flat_round_stream(
    flat_RS: np.ndarray,
    R: int,
    N_or_A: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Try to turn a flattened (N*R, S) or (A*R, S) into (N, R, S) or (A, R, S).
    If N_or_A is None, infer N = (flat_RS.shape[0] // R) if divisible.
    Returns None if reshape isn't possible.
    """
    flat_RS = np.asarray(flat_RS)
    rows, S = flat_RS.shape
    if rows % R != 0:
        return None
    N = rows // R if N_or_A is None else N_or_A
    if N * R != rows:
        return None
    # Assumption: rows grouped by entity, then round (C-order).
    # If your packing is different, flip with swapaxes after this call.
    return flat_RS.reshape(N, R, S)

def per_round_counts(E_NRS: np.ndarray) -> np.ndarray:
    """Binary events (N,R,S) -> per-round counts X (R,S)."""
    return E_NRS.sum(axis=0)

def round_round_corr(X_RS: np.ndarray) -> np.ndarray:
    """
    X_RS: (R,S) -> C_rr: (R,R), Pearson corr across shots.
    Each row is round t's vector across shots.
    """
    X = X_RS.astype(float)
    Xm = X - X.mean(axis=1, keepdims=True)
    Xstd = X.std(axis=1, keepdims=True) + 1e-12
    Z = Xm / Xstd
    C = (Z @ Z.T) / Z.shape[1]
    return np.clip(C, -1.0, 1.0)

def plot_Crr(C: np.ndarray, title: str, ax=None, vmin=-1.0, vmax=1.0):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.8), dpi=140)
    im = ax.imshow(C, origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("round t'")
    ax.set_ylabel("round t")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("corr across shots")
    return ax

# -------------------- driver --------------------

def make_Crr_heatmaps(
    M0: np.ndarray,      # (D,R,S)
    M1: np.ndarray,      # (A,R,S)
    M2: np.ndarray,      # e.g. (24,10,1024,2)  (we'll pick axes below)
    MR: np.ndarray,      # (A,R,S) or flattened (A*R,S)
    DET: np.ndarray,     # (Ndet,R,S) or flattened (Ndet*?R?,S)
    R: int,              # number of rounds (you said 10)
    A: Optional[int] = None,    # ancillas (you said 8)
    Ndet: Optional[int] = None, # detectors (unknown per round; only needed if DET is flat and divisible)
    m2_round_axis: int = 1,
    m2_shot_axis: int  = 2,     # your M2 is (24,10,1024,2): shots at axis=2
):
    # --- Normalize M0, M1 (already N,R,S)
    E0 = standardize_to_NRS(M0, round_axis=-2, shot_axis=-1)  # (D,R,S)
    E1 = standardize_to_NRS(M1, round_axis=-2, shot_axis=-1)  # (A,R,S)

    # --- Normalize M2 (pick axes explicitly)
    # If you later change M2 to (24,10,2,1024), set m2_shot_axis=3 instead.
    E2 = standardize_to_NRS(M2, round_axis=m2_round_axis, shot_axis=m2_shot_axis)

    # --- Handle MR (either (A,R,S) or (A*R,S))
    if MR.ndim == 2:
        MR_NRS = reshape_flat_round_stream(MR, R=R, N_or_A=A)
        if MR_NRS is None:
            raise ValueError(f"Cannot reshape MR of shape {MR.shape} into (A,R,S) with R={R}, A={A}.")
    elif MR.ndim == 3:
        MR_NRS = standardize_to_NRS(MR, round_axis=-2, shot_axis=-1)
    else:
        raise ValueError(f"Unexpected MR dims: {MR.shape}")

    # # --- Handle DET (either (Ndet,R,S) or flattened)
    # if DET.ndim == 2:
    #     # Try to reshape as (Ndet,R,S) if possible.
    #     DET_NRS = reshape_flat_round_stream(DET, R=R, N_or_A=Ndet)  # N_or_A acts as Ndet here
    #     if DET_NRS is None:
    #         # If not divisible by R, we can't do C_rr for DET (no per-round split).
    #         print(f"[warn] DET shape {DET.shape} not divisible by R={R}; skipping DET C_rr.")
    #         DET_NRS = None
    # elif DET.ndim == 3:
    #     DET_NRS = standardize_to_NRS(DET, round_axis=-2, shot_axis=-1)
    # else:
    #     raise ValueError(f"Unexpected DET dims: {DET.shape}")

    # --- Build per-round counts and C_rr per component
    Xs = {}
    Crrs = {}
    DET_NRS=None
    Xs["M0"]  = per_round_counts(E0)
    Xs["M1"]  = per_round_counts(E1)
    Xs["M2"]  = per_round_counts(E2)
    Xs["MR"]  = per_round_counts(MR_NRS)
    if DET_NRS is not None:
        Xs["DET"] = per_round_counts(DET_NRS)

    for name, X in Xs.items():
        Crrs[name] = round_round_corr(X)

    # --- Combined stream
    X_all = np.zeros_like(next(iter(Xs.values())))
    for X in Xs.values():
        X_all = X_all + X
    Crrs["Combined"] = round_round_corr(X_all)

    # --- Plot
    n = len(Crrs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.4*nrows), dpi=140)
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, C) in zip(axes, Crrs.items()):
        plot_Crr(C, title=f"C_rr — {name}", ax=ax)

    for ax in axes[len(Crrs):]:
        ax.axis("off")

    plt.tight_layout()
   # plt.show()
    plt.savefig('Temporal_Heatmaps')









def detector_round_slices(circuit):
    """
    Return a list of (start, stop) row indices in dets for each round r,
    and the per-round DETECTOR counts.
    """
    from surface_code.helpers import build_circ_by_round_from_generated
    circ_by_round, _ = build_circ_by_round_from_generated(circuit)

    counts = []
    for (pre, _, meas) in circ_by_round:
        c = 0
        for inst in meas:
            if inst.name == "DETECTOR":
                c += 1
        counts.append(c)

    offs = np.cumsum([0] + counts)
    slices = [(int(offs[i]), int(offs[i+1])) for i in range(len(counts))]

    return slices, counts

def per_round_counts_from_dets(dets, slices):
    """
    dets: (N_det, S) bool/int
    slices: [(a,b)] per round
    -> X: (R,S) counts per round per shot
    """
    dets = np.asarray(dets)
    R = len(slices)
    S = dets.shape[1]
    X = np.zeros((R, S), dtype=float)
    for r, (a, b) in enumerate(slices):
        if b > a:
            X[r] = dets[a:b].sum(axis=0)
    return X

def round_round_corr(X_RS):
    X = X_RS.astype(float)
    Xm = X - X.mean(axis=1, keepdims=True)
    Xs = X.std(axis=1, keepdims=True) + 1e-12
    Z = Xm / Xs
    C = (Z @ Z.T) / Z.shape[1]
    return np.clip(C, -1, 1)

def plot_Crr_d(C, title):
    plt.figure(figsize=(4.6,4.2), dpi=140)
    im = plt.imshow(C, origin='lower', vmin=-1, vmax=1, aspect='equal')
    plt.title(title); plt.xlabel("round t'"); plt.ylabel("round t")
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("corr across shots")
    plt.tight_layout(); plt.savefig('Detector_Temporal_Corrs_tt')



def per_round_detector_blocks(dets, slices):
    """Split dets (N_det,S) into a list of per-round blocks D_r with shape (n_r,S)."""
    return [dets[a:b, :] for (a, b) in slices]

# ---------- spatial correlation heatmaps (per round) ----------
def plot_intraround_corr_heatmaps(dets, slices, max_rounds=6):
    blocks = per_round_detector_blocks(dets, slices)
    R = len(blocks)
    k = min(max_rounds, R)
    ncols = min(3, k); nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 4.2*nrows), dpi=140)
    axes = np.atleast_1d(axes).ravel()
    for i in range(k):
        D = blocks[i].astype(float)  # (n_i, S)
        if D.shape[0] < 2:
            axes[i].set_title(f"Round {i} (n={D.shape[0]})"); axes[i].axis('off'); continue
        C = np.corrcoef(D)  # (n_i, n_i) across shots
        im = axes[i].imshow(np.clip(C, -1, 1), vmin=-1, vmax=1, origin='lower', aspect='equal')
        axes[i].set_title(f"Detector corr — round {i} (n={D.shape[0]})")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    for j in range(k, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(); plt.savefig('Spatial_Det_Correlations')

   