import numpy as np
import stim

class StimDecoderEnv:
    """
    Online / per-round environment for RL decoding.
    Vectorized over shots S.
    """
    def __init__(self, circuit, data_ids, anc_ids, gate_pairs, rounds, round_slices):
        self.circuit = circuit
        self.data_ids = np.asarray(data_ids)
        self.anc_ids  = np.asarray(anc_ids)
        self.gate_pairs = gate_pairs or []
        self.rounds = rounds
        self.round_slices = list(map(tuple, round_slices))  # [(a0,b0),...]
        self.R = len(self.round_slices)
        self._exec_det=0

        # Pre-split the generated circuit once
        from surface_code.helpers import build_circ_by_round_from_generated, extract_round_template_plus_suffix
        self.circ_by_round, anc_auto = build_circ_by_round_from_generated(circuit)
        assert sorted(anc_auto) == sorted(anc_ids)

        # Include prefix/suffix so detector count matches exactly
        self.prefix, self.pre_round_t, self.meas_round_t, self.suffix, anc_idx, rep = \
            self._extract_round_template_with_suffix(circuit)

        # Masks set at reset()
        self.M0 = self.M1 = self.M2 = None
        self.sim = None
        self.S = None
        self.r = 0
        counts_all = [sum(1 for inst in meas if getattr(inst, "name", None) == "DETECTOR")
                    for (_,_,meas) in self.circ_by_round]
        self._body_offset = 1 if (counts_all and counts_all[0] == 4) else 0


    @staticmethod
    def _extract_round_template_with_suffix(circ: stim.Circuit):
        # Same extractor we discussed that returns prefix/pre/meas/suffix
        from surface_code.helpers import extract_round_template_plus_suffix  # if you upgraded it to return suffix; else paste the new impl
        prefix, pre_r, meas_r, suffix, anc_ids, repeat = extract_round_template_plus_suffix(circ)
        return prefix, pre_r, meas_r, suffix, anc_ids, repeat

    def _cnt(self,c): return sum(1 for ln in str(c).splitlines() if ln.strip().startswith("DETECTOR"))

    def reset(self, M0_local, M1_local, M2_local):
        """
        Start a new episode with pre-sampled stacked masks:
          M0_local: (D, R, S)
          M1_local: (A, R, S)
          M2_local: (E, R, S, 2)  or None if not used
        Returns the initial observation for round 0 (which is just zeros; the first real obs is after step(…)).
        """
        # Validate & store masks
        if M0_local is not None:
            D, R0, S = M0_local.shape
        else:
            D, R0, S = len(self.data_ids), self.R, (M1_local.shape[2] if M1_local is not None else M2_local.shape[2])
        if M1_local is not None:
            A, R1, S1 = M1_local.shape
            assert R1 == self.R and S1 == S
        if M2_local is not None:
            E, R2, S2, _ = M2_local.shape
            assert R2 == self.R and S2 == S

        self.M0, self.M1, self.M2 = M0_local, M1_local, M2_local
        self.S = S
        self.r = 0

        # Init simulator
        Q = max([0] + list(self.data_ids) + list(self.anc_ids) +
                [q for p in self.gate_pairs for q in p]) + 1
        self.sim = stim.FlipSimulator(batch_size=S, num_qubits=Q, disable_stabilizer_randomization=True)

        # Run prefix ONCE
        if len(self.prefix) > 0:
            # right before sim.do(self.prefix):
            dets_before = self.sim.get_detector_flips().shape[0]
            self.sim.do(self.prefix)
            dets_after  = self.sim.get_detector_flips().shape[0]
            print("DET before prefix:", dets_before, " after prefix:", dets_after)  # expect +4


        # Return a dummy observation before any round (all-zeros of current round size),
        # or simply return None and let the agent call step() to get round 0 obs.
        a0, b0 = self.round_slices[0]
        return np.zeros((self.S, b0 - a0), dtype=np.uint8)

    def _apply_paulis(self, xmask, ymask, zmask):
        if xmask is not None and xmask.any(): self.sim.broadcast_pauli_errors(pauli='X', mask=xmask)
        if ymask is not None and ymask.any(): self.sim.broadcast_pauli_errors(pauli='Y', mask=ymask)
        if zmask is not None and zmask.any(): self.sim.broadcast_pauli_errors(pauli='Z', mask=zmask)

    def _inject_M0_round(self, r):
        if self.M0 is None: return
        D = len(self.data_ids); S = self.S
        mr = self.M0[:, r, :]  # (D,S) codes {0,1,2,3}
        xmask = np.zeros((self.sim.num_qubits, S), np.bool_)
        ymask = np.zeros_like(xmask); zmask = np.zeros_like(xmask)
        xmask[self.data_ids, :] = (mr == 1) | (mr == 3)
        zmask[self.data_ids, :] = (mr == 2) | (mr == 3)
        self._apply_paulis(xmask, None, zmask)

    def _inject_M1_round(self, r):
        if self.M1 is None: return
        A = len(self.anc_ids); S = self.S
        ma = self.M1[:, r, :]  # (A,S)
        xmask = np.zeros((self.sim.num_qubits, S), np.bool_)
        ymask = np.zeros_like(xmask); zmask = np.zeros_like(xmask)
        xmask[self.anc_ids, :] = (ma == 1) | (ma == 3)
        zmask[self.anc_ids, :] = (ma == 2) | (ma == 3)
        self._apply_paulis(xmask, None, zmask)

    def _iter_twoq_pairs(self, inst: stim.CircuitInstruction):
        TWO_Q = {"CZ","CX","CY","SWAP","ISWAP","SQRT_XX","SQRT_YY","SQRT_ZZ"}

        """Return [(q0,q1), ...] for a (possibly batched) 2Q instruction; else []."""
        if inst.name not in TWO_Q:
            return []
        vs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        # Stim batches: e.g. 'CX 2 3 16 17 ...' → pairs (2,3), (16,17), ...
        return [(vs[i], vs[i+1]) for i in range(0, len(vs), 2)]
    def _run_pre_with_M2(self, pre_circ, r):
        if self.M2 is None or len(self.gate_pairs) == 0:
            self.sim.do(pre_circ)
            return
        # Walk pre_circ and inject after each 2Q gate according to gate_pairs order
        TWO_Q = {"CZ","CX","CY","SWAP","ISWAP","SQRT_XX","SQRT_YY","SQRT_ZZ"}
        SKIP = {"QUBIT_COORDS", "SHIFT_COORDS"}
        E = len(self.gate_pairs)
        e_idx = 0
        for inst in pre_circ:
            if inst.name in SKIP:
                continue
            # run instruction
            seg = stim.Circuit(); seg.append_operation(inst.name, inst.targets_copy(), inst.gate_args_copy()); self.sim.do(seg)
            for _ in self._iter_twoq_pairs(inst): # handle batched 2Q ops
                # inject M2 for the e_idx-th pair
                q0, q1 = self.gate_pairs[e_idx]
                codes0 = self.M2[e_idx, r, :, 0]  # (S,)
                codes1 = self.M2[e_idx, r, :, 1]  # (S,)
                S = self.S
                xmask = np.zeros((self.sim.num_qubits, S), np.bool_)
                ymask = np.zeros_like(xmask); zmask = np.zeros_like(xmask)
                # X/Z/Y on each leg (use your channel’s convention)
                xmask[q0, codes0==1] = True; zmask[q0, codes0==2] = True
                xmask[q1, codes1==1] = True; zmask[q1, codes1==2] = True
                self._apply_paulis(xmask, None, zmask)
                e_idx += 1
        assert e_idx == E, f"Round {r}: expected {E} 2Q pairs, saw {e_idx}"

    def step(self, action_mask=None):
        """
        One round step:
          - apply agent corrections on data qubits (optional)
          - inject M0/M2/M1 for round r
          - run pre and meas
          - return detector slice for round r: obs_r shape (S, n_r)
        """
        if self.r >= len(self.round_slices):
            raise RuntimeError("Episode already finished. Call reset().")

        # 0) agent corrections (optional), format: xmask/zmask of shape (Q,S) or (len(data_ids),S)
        if action_mask is not None:
            # adapt this to your action format; here we assume (Q,S) bool X/Z
            xmask, zmask = action_mask.get('X'), action_mask.get('Z')
            self._apply_paulis(xmask, None, zmask)

        # 1) round injections & execution
        pre, _, meas = self.circ_by_round[self._body_offset+self.r]
  #      if self.r==1: print(f'When r==0, pre is\n{pre} \n and meas is: \n{meas}')
        self._inject_M0_round(self.r)
        self._run_pre_with_M2(pre, self.r)
        self._inject_M1_round(self.r)
     #   if self.r==0:         print(f'In r 0 we have {self.sim.get_detector_flips().shape[0]} detectors (should be 4)')
        self.sim.do(meas)
    #    if self.r==0: print(f'And after first meas execution we have {self.sim.get_detector_flips().shape[0]} dets (should be 12)')
        # 2) observation = detector slice for this round (S, n_r)
        a, b = self.round_slices[self.r ]
        DET = self.sim.get_detector_flips()          # (N_det, S)
        obs_r = DET[a:b, :].T.astype(np.uint8)       # (S, n_r)

        self.r += 1
        done = (self.r == len(self.round_slices))  # suffix not run yet; we run it in finish()
        return obs_r, done

    def finish(self):
        """
        Run suffix (once), compute terminal info:
          - final detectors appended inside suffix
          - observable flips available; return them for terminal reward calculation
        """
        if len(self.suffix) > 0:
                        # right before sim.do(self.suffix):
            dets_before = self.sim.get_detector_flips().shape[0]
            self.sim.do(self.suffix)
            dets_after  = self.sim.get_detector_flips().shape[0]
            print("DET before suffix:", dets_before, " after suffix:", dets_after)  # expect +4


        dets = self.sim.get_detector_flips().astype(np.uint8)     # (N_det, S)
        obs  = self.sim.get_observable_flips().astype(np.uint8)   # (N_obs, S)
        meas = self.sim.get_measurement_flips().astype(np.uint8)  # (A*(R+1)+D, S)

        # Slice body ancilla MR only (exclude prefix A, suffix D)
        A = len(self.anc_ids); D = len(self.data_ids); R = self.R
        M_expected = A*(R+1) + D
        if meas.shape[0] != M_expected:
            raise RuntimeError(f"meas rows={meas.shape[0]} != A*(R+1)+D={M_expected}")
        mr_body = meas[A : A + A*R, :]                     # (A*R, S)
        MR = mr_body.reshape(R, A, -1).transpose(1,0,2)    # (A, R, S)

        # Typical reward: +1 if observable 0 is 0 (no logical flip), else 0
        reward = None
        if obs.shape[0] >= 1:
            reward = (1 - obs[0, :]).astype(np.float32)    # (S,)

        # Reset internal state? (your trainer typically creates a new env per episode or calls reset)
        return dets, MR, obs, reward
