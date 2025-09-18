# --- BEGIN: syndrome export function you can call from simulator.py ---

import json, numpy as np
from pathlib import Path
from .helpers import make_M_anc_local_from_masks, make_M_data_local_from_masks, summarize_M0_or_M1, summarize_M2



def _ensure_det_shape(det_bits, *, N_det_expected=None, S_expected=None):
    db = np.asarray(det_bits)
    if db.ndim == 1:
        raise ValueError(f"det_bits is 1D {db.shape}; expected 2D (N_det, S).")
    if db.ndim != 2:
        raise ValueError(f"det_bits must be 2D, got {db.shape}")
    r0, r1 = db.shape
    if N_det_expected is not None and S_expected is not None:
        if (r0, r1) == (N_det_expected, S_expected): return db.astype(np.uint8)
        if (r0, r1) == (S_expected, N_det_expected): return db.T.astype(np.uint8)
    return (db if r0 <= r1 else db.T).astype(np.uint8)

def _reshape_mr_AR(mr_bits, *, A, R):
    mb = np.asarray(mr_bits)

    if mb.ndim != 2:
        raise ValueError(f"MR must be 2D or 3D; got {mb.shape}")

    r0, r1 = mb.shape
    if r0 == A*R:
        Sb = r1
        flat = mb
    elif r1 == A*R:
        Sb = r0
        flat = mb.T
    else:
        # Last-resort inference if it’s been reshaped oddly but sizes match
        total = r0 * r1
        if total % (A*R) != 0:
            raise ValueError(f"Cannot infer R/Sb from MR shape {mb.shape} with A*R={A*R}")
        Sb = total // (A*R)
        flat = mb.reshape(A*R, Sb)

    return flat.reshape(A, R, Sb).astype(np.uint8)

def _packbits_rows(b):
    b = np.ascontiguousarray(b.astype(np.uint8))
    return np.packbits(b, axis=1)

def export_syndrome_dataset(
    *,
    circuit,
    data_qubits,
    anchilla_qubits,
    rounds,
    data_ids,
    anc_ids,
    gate_pairs=None,
    # supply your noise builders; pass None to disable a channel
    m0_build=None,   # callable: (D, S, rng, **cfg)-> (D,R,S) codes {0,1,2,3}
    m1_build=None,   # callable: (A, S, rng, **cfg)-> (A,R,S)
    m2_build=None,   # callable: (G, S, rng, gate_pairs=..., **cfg)-> (G,R,2,S)
    m0_cfg=None,
    m1_cfg=None,
    m2_cfg=None,
    S_total=10_000,
    batch=1024,
    rng_seed=123,
    out_path="data/syndrome.npz",
    meta_path=None,
    coords=None,          # list/array or None
    round_slices=None,    # list of per-round detector indices or slice bounds
    verbose=True,
):
    """
    Runs the circuit many shots with sampled noise masks and writes:
      - DET_packed (packbits) + DET_shape
      - MR (optional) + MR_shape
      - companion .meta.json with coords/round_slices/gate_pairs, seeds, flags
    Designed to be called from simulator.py; prints progress; returns nothing.
    """
    import numpy as np
    from surface_code.inject import run_batched_data_anc_plus_m2
   # import simulator  # uses your project simulator

    rng = np.random.default_rng(rng_seed)
    A, D = len(anc_ids), len(data_ids)
    G = len(gate_pairs) if gate_pairs is not None else 0

    enable_M0 = m0_build is not None
    enable_M1 = m1_build is not None
    enable_M2 = m2_build is not None

    det_chunks, mr_chunks = [], []
    shots_done = 0

    if verbose:
        print("[export] Vectorized mode (single call)")
        print(f"  • shots S_total    : {S_total}")
        print(f"  • channels         : M0={enable_M0}  M1={enable_M1}  M2={enable_M2}")
        print(f"  • dims (D,A,G,R)   : ({D},{A},{G},{rounds})")
        try:
            print(f"  • circuit dets     : {circuit.num_detectors}")
        except Exception:
            pass
        print("[export] building stacked masks...")



    m0_stacked=[m0_build(qubits=data_qubits,
                            rounds=rounds,
                            qubits_ind=data_ids,
                            cfg=m0_cfg
                        ) 
                        for _ in range(S_total)
                       
                        
                ] if enable_M0 else None
    

    m1_stacked=[m1_build( qubits=anchilla_qubits,
                            rounds=rounds,
                            qubits_ind=anc_ids,
                            cfg=m1_cfg
                        )
                        for _ in range(S_total)
                       
                        
                ] if enable_M1 else None
    
    m2_stacked=[m2_build(gates=gate_pairs,
                          r=rounds,
                          cfg_m2=m2_cfg,
                          
                          ) 
                          for _ in range(S_total)
                          
                ] if enable_M2 else None
    

    m0_s=np.stack(m0_stacked, axis=-1).astype(np.int8)
    m1_s=np.stack(m1_stacked, axis=-1).astype(np.int8)
    m2_s=np.stack(m2_stacked, axis=-2).astype(np.int8)


    print('\n\n-----------------------------Summarization of each mask ---------------------------')
    summarize_M0_or_M1(m0_s, "M0")
    summarize_M0_or_M1(m1_s, "M1")
    summarize_M2(m2_s, "M2")

    print('---------------End of Summarization---------------------------------\n\n')

    if verbose:
        if enable_M0: print(f"  • M0_local shape: {m0_s.shape}")
        if enable_M1: print(f"  • M1_local shape: {m1_s.shape}")
        if enable_M2: print(f"  • M2_local shape: {m2_s.shape}")

    # ---------- Single vectorized simulation call ----------
    det_bits, obs, mr_bits = run_batched_data_anc_plus_m2(
        circuit=circuit,
        M_data=m0_s,
        M_anc=m1_s,
        M2=m2_s,
        gate_pairs=gate_pairs,
        data_ids=data_ids,
        anc_ids=anc_ids,
        enable_M0=enable_M0,
        enable_M1=enable_M1,
        enable_M2=enable_M2,
    )

    # ---------- Normalize shapes ----------
    N_det = getattr(circuit, "num_detectors", None)
    print(f'N_det: {N_det}')
    DET = _ensure_det_shape(det_bits, N_det_expected=N_det, S_expected=S_total)    # (N_det, S_total)
    MR  = _reshape_mr_AR(mr_bits, A=A, R=rounds) if mr_bits is not None else None # (A,R,S_total)

    # ---------- Save ----------
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    DET_packed = _packbits_rows(DET)
    np.savez_compressed(
        out_path,
        DET_packed=DET_packed,
        DET_shape=np.array(DET.shape, dtype=np.int64),
        MR=(MR if MR is not None else np.array([], dtype=np.uint8)),
        MR_shape=(np.array(MR.shape, dtype=np.int64) if MR is not None else np.array([0,0,0], dtype=np.int64)),
    )

    meta = {
        "circuit": getattr(circuit, "name", "unknown_circuit"),
        "enabled": {"M0": bool(enable_M0), "M1": bool(enable_M1), "M2": bool(enable_M2)},
        "seeds": {"rng_seed": rng_seed},
        "gate_pairs": (list(map(list, gate_pairs)) if gate_pairs is not None else None),
        "coords": (np.asarray(coords).tolist() if coords is not None else None),
        "rounds": int(rounds),
        "data_count": int(D),
        "ancilla_count": int(A),
        "shots": int(DET.shape[1]),
        "num_detectors": int(DET.shape[0]),
        "round_slices": (
            [np.asarray(x).tolist() for x in round_slices] if isinstance(round_slices, (list, tuple))
            else (np.asarray(round_slices).tolist() if round_slices is not None else None)
        ),
        "note": "syndrome export (vectorized, stacked masks)",
    }
    meta_path = Path(meta_path) if meta_path else out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    # ---------- Prints ----------
    if verbose:
        print(f"[export] wrote {out_path}")
        print(f"[export] wrote {meta_path}")
        print("\n--- FINAL STATS ---")
        print(f"N_det={DET.shape[0]}  S={DET.shape[1]}  mean(det)={DET.mean():.6f}")
        print(f"empty shot fraction = {(DET.sum(axis=0)==0).mean():.6f}")
        if MR is not None and MR.shape[1] >= 2:
            flips = (MR[:,1:,:] ^ MR[:,:-1,:]).mean(axis=(1,2))
            print("timelike flip rate per ancilla (XOR of consecutive rounds):")
            print(np.array2string(flips, precision=6, separator=", "))


    return DET, MR, obs
# --- END: syndrome export function ---
