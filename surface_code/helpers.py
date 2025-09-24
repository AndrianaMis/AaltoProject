
import re
import stim
import numpy as np

from typing import List, Tuple, Dict, Optional

import numpy as np
import random
I, X, Z, Y = 0, 1, 2, 3



def print_svg(round: stim.Circuit, strr:str):
    ric=round.diagram('timeline-svg')
    ric = str(ric)

    # Insert a white rectangle after the opening <svg ...> tag
    ric = re.sub(
        r'(<svg[^>]*>)',
        r'\1<rect width="100%" height="100%" fill="white"/>',
        ric,
        count=1
    )


    with open(f"{strr}.svg", "w") as f:
        f.write(ric)

    print(f'\nSaved svg pic in {strr}.svg!\n')

def extract_round_template_plus_suffix(circ: stim.Circuit):
    """
    Split circuit into:
      prefix    : before the first REPEAT{...}
      pre_round : inside the REPEAT body, ops before first MR/M
      meas_round: inside the REPEAT body, from first MR/M to end-of-body (includes DETECTORs)
      suffix    : after the REPEAT block
      anc_ids   : ancilla indices parsed from that first MR/M line
      repeat_count: repeat times of the REPEAT block
    """
    import stim

    # Work with text for a simple, robust split.
    lines = str(circ).splitlines()

    # Find the first REPEAT line and the matching closing brace.
    rep_start = None
    body_start = None
    body_end = None
    repeat_count = None

    # locate "REPEAT k {" line
    for i, ln in enumerate(lines):
        parts = ln.strip().split()
        if parts[:1] == ["REPEAT"] and "{" in ln:
            rep_start = i
            # repeat count is the next token
            repeat_count = int(parts[1])
            # body starts after the line with "{"
            body_start = i + 1
            break

    if rep_start is None:
        # No REPEAT: treat whole as one "round", empty suffix
        prefix_txt = ""
        body_txt = "\n".join(lines)
        suffix_txt = ""
    else:
        # find matching "}" for this REPEAT (track nesting)
        depth = 0
        for j in range(rep_start, len(lines)):
            ln = lines[j]
            depth += ln.count("{")
            depth -= ln.count("}")
            if depth == 0:
                body_end = j  # line containing the closing "}"
                break
        if body_end is None:
            raise ValueError("Unbalanced REPEAT braces in circuit text.")

        prefix_txt = "\n".join(lines[:rep_start])
        body_txt   = "\n".join(lines[body_start:body_end])  # inside braces
        suffix_txt = "\n".join(lines[body_end+1:])          # after the REPEAT block

    # Inside the body, split at the first ancilla measurement (MR or M)
    body_lines = body_txt.splitlines()
    meas_start = None
    anc_ids = []

    def parse_int(tok: str) -> int:
        return int(tok[1:]) if tok.startswith("!")==True else int(tok)

    for i, ln in enumerate(body_lines):
        parts = ln.split()
        if not parts:
            continue
        if parts[0] in ("MR", "M"):
            meas_start = i
            anc_ids = sorted(parse_int(t) for t in parts[1:])
            break

    if meas_start is None:
        pre_txt  = "\n".join(body_lines)
        meas_txt = ""
    else:
        pre_txt  = "\n".join(body_lines[:meas_start])
        meas_txt = "\n".join(body_lines[meas_start:])

    prefix    = stim.Circuit(prefix_txt) if prefix_txt.strip() else stim.Circuit()
    pre_round = stim.Circuit(pre_txt)    if pre_txt.strip()    else stim.Circuit()
    meas_round= stim.Circuit(meas_txt)   if meas_txt.strip()   else stim.Circuit()
    suffix    = stim.Circuit(suffix_txt) if suffix_txt.strip() else stim.Circuit()

    # When no REPEAT was found, set repeat_count=1 if it wasn’t parsed
    if repeat_count is None:
        repeat_count = 1

    return prefix, pre_round, meas_round, suffix, anc_ids, repeat_count

 
    


def extract_round_template(circ: stim.Circuit):
    """
    Splits a Stim-generated surface_code round template at the first ancilla measurement.
    Returns:
      prefix       : stim.Circuit (stuff before the REPEAT block)
      pre_round    : stim.Circuit (ops BEFORE first MR/M in the round)
      meas_round   : stim.Circuit (first MR/M to end-of-round; includes DETECTORs)
      anc_ids      : list[int]     (ancilla indices parsed from that first MR/M line)
      repeat_count : int           (# of rounds)
    """
    # 1) Find the REPEAT block and collect prefix instructions
    prefix = stim.Circuit()
    round_body = None
    repeat_count = None
    for inst in circ:
        if getattr(inst, "name", None) == "REPEAT":  # it's a CircuitRepeatBlock
            repeat_count = inst.repeat_count
            round_body = inst.body_copy()            # <-- correct API
            break
        else:
            prefix.append(inst)

    if round_body is None:
        # Fallback: no repeat; treat whole circuit as one "round"
        round_body = circ
        repeat_count = 1

    # 2) Split the round body at the first ancilla measurement (MR/M)
    lines = str(round_body).splitlines()

    def parse_int(tok: str) -> int:
        return int(tok[1:]) if tok.startswith('!') else int(tok)

    meas_start = None
    anc_ids = []
    for i, ln in enumerate(lines):
        parts = ln.split()
        if not parts:
            continue
        if parts[0] in ("MR", "M"):
            meas_start = i
            anc_ids = sorted(parse_int(t) for t in parts[1:])
            break

    if meas_start is None:
        pre_txt = "\n".join(lines)
        meas_txt = ""
    else:
        pre_txt = "\n".join(lines[:meas_start])
        meas_txt = "\n".join(lines[meas_start:])

    pre_round  = stim.Circuit(pre_txt)  if pre_txt.strip()  else stim.Circuit()
    meas_round = stim.Circuit(meas_txt) if meas_txt.strip() else stim.Circuit()
    return prefix, pre_round, meas_round, anc_ids, repeat_count


# simplest: a thin wrapper for readability
def make_M_anc_local_from_masks(masks, anc_ids, row_to_global=None, rounds=None):
    return make_M_data_local_from_masks(
        masks=masks, data_ids=anc_ids, row_to_global=row_to_global, rounds=rounds
    )




def get_data_and_ancilla_ids_by_parity(circuit: stim.Circuit, distance: int):
    # 1) coords: dict {qid: (x, y)}
    coords = circuit.get_final_qubit_coordinates()
    qids, xy = zip(*coords.items())
    qids = np.array(qids, dtype=int)
    xy = np.array(xy, dtype=int)  # shape (N, 2)

    x, y = xy[:, 0], xy[:, 1]
    both_even = (x % 2 == 0) & (y % 2 == 0)
    both_odd  = (x % 2 == 1) & (y % 2 == 1)

    cand_even = qids[both_even]
    cand_odd  = qids[both_odd]

    d2 = distance * distance
    if len(cand_even) == d2 and len(cand_odd) != d2:
        data_ids = np.sort(cand_even)
    elif len(cand_odd) == d2 and len(cand_even) != d2:
        data_ids = np.sort(cand_odd)
    else:
        # Fallback: pick the parity with size closest to d^2
        # (useful if Stim changes internals; still deterministic)
        if abs(len(cand_even) - d2) <= abs(len(cand_odd) - d2):
            data_ids = np.sort(cand_even)
        else:
            data_ids = np.sort(cand_odd)

    ancilla_ids = np.array(sorted(set(qids) - set(data_ids)), dtype=int)
    return data_ids.tolist(), ancilla_ids.tolist()




import numpy as np
from typing import List, Sequence, Union

ArrayLike = Union[np.ndarray, List[np.ndarray]]

def make_M_data_local_from_masks(
    masks: ArrayLike,
    data_ids: Sequence[int],
    row_to_global: Sequence[int] | None = None,
    rounds: int | None = None,
) -> np.ndarray:
    """
    Build M_data_local with shape (len(data_ids), rounds, S) and codes {0=I,1=X,2=Z,3=Y}.

    Parameters
    ----------
    masks : 
        - np.ndarray with shape (Q_or_D, R) or (Q_or_D, R, S), OR
        - list of np.ndarrays each with shape (Q_or_D, R).
        Rows can be full-qubit order (global ids 0..Q_total-1) OR data-only.
    data_ids : list[int]
        Global Stim qubit ids for DATA qubits (row order desired in the output).
    row_to_global : list[int] | None
        If your mask rows are *not* in global-id order and are *not* already data-only,
        provide a mapping: for local row i, row_to_global[i] = global_qubit_id.
        If None:
          - If first dim == max(global_id)+1, we assume rows are global-id order.
          - If first dim == len(data_ids), we assume it's already data-only.
    rounds : int | None
        Optional safeguard to assert the mask’s time dimension.

    Returns
    -------
    M_data_local : np.ndarray
        Shape (D, R, S) where D=len(data_ids), R=rounds, S=#shots.
    """
    # ---- normalize input to 3D: (Q_or_D, R, S) ----
    if isinstance(masks, list):
        assert len(masks) > 0, "Empty mask list."
        m_list = [np.asarray(m, dtype=np.int8) for m in masks]
        base_shape = m_list[0].shape
        assert all(m.shape == base_shape for m in m_list), "All masks must share the same shape."
        m = np.stack(m_list, axis=2)  # (..., R, S)
    else:
        m = np.asarray(masks, dtype=np.int8)
        if m.ndim == 2:
            m = m[:, :, None]  # add shot axis S=1
        elif m.ndim != 3:
            raise ValueError("masks must be 2D, 3D, or list of 2D arrays")

    Q_or_D, R, S = m.shape
    if rounds is not None and R != int(rounds):
        raise ValueError(f"Rounds mismatch: mask has R={R}, expected {rounds}")

    D = len(data_ids)
    data_ids = list(map(int, data_ids))
    max_global = max(data_ids) if data_ids else -1

    # ---- select rows corresponding to data_ids ----
    if row_to_global is not None:
        # Build local row index for each desired global id
        pos = {g: i for i, g in enumerate(map(int, row_to_global))}
        try:
            idx = np.array([pos[g] for g in data_ids], dtype=int)
        except KeyError as e:
            raise ValueError(f"Global id {e.args[0]} not found in row_to_global") from None
        M_local = m[idx, :, :]  # (D, R, S)

    else:
        # No mapping provided: infer layout
        if Q_or_D == (max_global + 1):
            # Full-qubit global-id row layout
            M_local = m[np.array(data_ids, dtype=int), :, :]  # (D, R, S)
        elif Q_or_D == D:
            # Already data-only in some order; assume it matches data_ids row order
            M_local = m  # (D, R, S)
        else:
            raise ValueError(
                f"Can't infer row layout: first dim={Q_or_D}, "
                f"but neither equals Q_total={max_global+1} nor D={D}. "
                f"Provide row_to_global to disambiguate."
            )

    # ---- sanity checks ----
    if not np.isin(M_local, [0, 1, 2, 3]).all():
        bad = np.unique(M_local[~np.isin(M_local, [0,1,2,3])])
        raise ValueError(f"Mask contains invalid codes: {bad}. Expected {{0,1,2,3}}.")

    return M_local.astype(np.int8, copy=False)
















def _split_at_first_mr(c: stim.Circuit):
    """Return (pre, meas, anc_ids, found). If no MR/M inside c, found=False."""
    lines = str(c).splitlines()
    def p(tok: str): return int(tok[1:]) if tok.startswith('!') else int(tok)
    i = None; anc = []
    for k, ln in enumerate(lines):
        parts = ln.split()
        if parts and parts[0] in ("MR", "M"):
            i = k
            anc = sorted(p(t) for t in parts[1:])
            break
    if i is None:
        return stim.Circuit(), stim.Circuit(), [], False
    pre  = stim.Circuit("\n".join(lines[:i])) if i > 0 else stim.Circuit()
    meas = stim.Circuit("\n".join(lines[i:])) if i < len(lines) else stim.Circuit()
    return pre, meas, anc, True

def build_circ_by_round_from_generated(circ: stim.Circuit):
    """
    Uses your extract_round_template (REPEAT body) + splits the prefix once.
    Returns:
      circ_by_round : list[(pre, between, meas)] for every actual round
      anc_ids       : list[int]
    """
    prefix, pre_tpl, meas_tpl, anc_tpl, repeat_count = extract_round_template(circ)

    pre_first, meas_first, anc_first, has_first = _split_at_first_mr(prefix)
    anc_ids = anc_first if has_first else anc_tpl

    circ_by_round = []
    if has_first:
        circ_by_round.append((pre_first, stim.Circuit(), meas_first))
    for _ in range(repeat_count):
        circ_by_round.append((pre_tpl, stim.Circuit(), meas_tpl))

    return circ_by_round, anc_ids










def extract_template_cx_pairs(circuit, distance):
    prefix, pre_round, meas_round, anc_ids, repeat_count = extract_round_template(circuit)
    data_ids, anc_ids2 = get_data_and_ancilla_ids_by_parity(circuit, distance)

    cx_pairs = []
    for inst in pre_round:
        if inst.name == "CX":
            vals = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            # group into (control, target) pairs
            for i in range(0, len(vals), 2):
                c, t = vals[i], vals[i+1]
                if (c in data_ids and t in anc_ids) or (c in anc_ids and t in data_ids):
                    cx_pairs.append((c, t))
                else:
                    # if you want *all* CX regardless of roles, just append unconditionally
                    cx_pairs.append((c, t))
    return cx_pairs, data_ids, anc_ids, repeat_count




def _compose_xz_to_y(existing: int, new: int) -> int:
    """OR-like composition without cancellation; X+Z or Z+X -> Y; once Y, keep Y."""
    if existing == I:
        return new
    if existing == Y or new == Y:
        return Y
    if (existing, new) in [(X, Z), (Z, X)]:
        return Y
    return existing  # keep first writer otherwise

def _sample_single_pauli(rng: np.random.Generator, pX: float, pZ: float) -> int:
    probs = np.array([pX, pZ], dtype=float)
    if probs.sum() <= 0:
        return X
    probs /= probs.sum()
    return int([X, Z][rng.choice([0, 1], p=probs)])

def _sample_pair_pauli(
    rng: np.random.Generator,
    *,
    mode: str = "biased_product",
    pX: float = 0.5,
    pZ: float = 0.5,
    no_II: bool = True
) -> Tuple[int, int]:
    """
    Return a pair of single-qubit Pauli codes (c0, c1) from {I,X,Z,Y} but
    we only *sample* X/Z, with Y arising from overlaps when composing hits.
    mode="biased_product": independently draw X/Z on each leg with (pX,pZ).
    If no_II, resample to avoid (I,I).
    """
    if mode == "biased_product":
        while True:
            c0 = _sample_single_pauli(rng, pX, pZ) if rng.random() < 1.0 else I
            c1 = _sample_single_pauli(rng, pX, pZ) if rng.random() < 1.0 else I
            if not (no_II and c0 == I and c1 == I):
                return c0, c1
    else:
        # fallback
        return _sample_single_pauli(rng, pX, pZ), _sample_single_pauli(rng, pX, pZ)
    









def summarize_M0_or_M1(M,name):  # M: (Q,R,S) with codes {0,1,2,3}
    nz = (M != 0)
    p_hat = nz.mean()
    per_round = nz.mean(axis=(0,2))  # (R,)
    per_qubit = nz.mean(axis=(1,2))  # (Q,)
    per_shot_counts = nz.reshape(nz.shape[0]*nz.shape[1], nz.shape[2]).sum(axis=0)
    fano = per_shot_counts.var() / max(1e-12, per_shot_counts.mean())
    print(f"[{name}] p̂={p_hat:.6f}  empty_shots={(per_shot_counts==0).mean():.3f}  Fano={fano:.2f}")
    print(f"[{name}] per-round mean (min/median/max): {per_round.min():.5f} / "
          f"{np.median(per_round):.5f} / {per_round.max():.5f}")

def summarize_M2(M2,name):  # M2: (E,R,S,2)
    occ = (M2 != 0).any(axis=-1)     # (E,R,S)
    p_site = occ.mean()              # per (gate,round,shot)
    per_round = occ.mean(axis=(0,2)) # (R,)
    print(f"[{name}] site p̂={p_site:.6f}")
    print(f"[{name}] per-round mean (min/median/max): {per_round.min():.5f} / "
          f"{np.median(per_round):.5f} / {per_round.max():.5f}")




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






def split_DET_by_round(DET, round_slices):
    DET_by_round = []
    for (a, b) in round_slices:
        DET_by_round.append(DET[a:b, :])
    return DET_by_round  # [(n_r, S), ...]


# For a given shot s, get the "syndrome sequence" of detector events:
def get_syndrome_sequence_from_DET(DET_by_round, s):
    """Return list length R; each item is (n_r,) uint8 for shot s."""
    return [blk[:, s].astype(np.uint8) for blk in DET_by_round]






def logical_error_rate(shots:int, obs):
    logical_errors=[]
    for s in range(shots):
        if obs[:,s] == 1:
            logical_errors.append(s)
    return float(len(logical_errors)/shots)