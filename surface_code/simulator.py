from surface_code.code_generator import build_planar_surface_code
import stim
import numpy as np
from surface_code.marginalize import calibrate_start_rates, build_m0_once, cfg_data, cfg_anch, build_m1_once, cfg_m2, build_m2_once, calibrate_start_rates_m2
from surface_code.stats import measure_mask_stats, measure_stacked_mask_stats, measure_m2_mask_stats, m2_stats
from surface_code.inject import  run_batched_data_plus_anc, run_batched_data_anc_plus_m2
from surface_code.helpers import print_svg, extract_round_template, get_data_and_ancilla_ids_by_parity, make_M_data_local_from_masks, make_M_anc_local_from_masks, extract_template_cx_pairs, split_DET_by_round, get_syndrome_sequence_from_DET, logical_error_rate
from surface_code.M1 import mask_generator_M1
from surface_code.M2 import mask_generator_M2
from visuals.corrs import make_Crr_heatmaps
from surface_code.M0 import mask_generator, mask_init
from visuals import corrs
from .export import export_syndrome_dataset

# Generate the rotated surface code circuit:
distance = 3
rounds = 10
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=rounds,
    distance=distance
    # after_clifford_depolarization=0.0,
    # after_reset_flip_probability=0.0

)
mr_indices = [i for i, instr in enumerate(circuit) if instr.name == 'MR']
# for mr in mr_indices:
#     print(f'mr: {mr}')
#     circuit.insert(mr, stim.CircuitInstruction('CORRELATED_ERROR',  [stim.target_x(1), stim.target_y(1)], [0.5]))


# print(circuit.detector_error_model())
# Compile for fast sampling:
sampler = stim.CompiledDetectorSampler(circuit)
print(f'Total qubits: {circuit.num_qubits} \t Total TICKS: {circuit.num_ticks} \t Total detectors:{circuit.num_detectors} \t Total Measurements: {circuit.num_measurements} \t Total observables: {circuit.num_observables}')

# Sample a batch of detection event outcomes:
shots = 2
stbs=circuit.num_detectors //rounds
result = sampler.sample(shots)  # Result has shape (shots, detectors_per_shot)
res=result.reshape((shots, rounds, stbs))
for r in range(rounds):

    for shot in range(shots):
        x_syndromes = res[shot, r, :(stbs//2)]
        z_syndromes = res[shot, r, (stbs//2):]
        print(f"Shot {shot}, Round {r}:")
        print("  X stabilizers:", x_syndromes)
        print("  Z stabilizers:", z_syndromes)

print(circuit)


print(mr_indices)



data_qus,anchs=get_data_and_ancilla_ids_by_parity(circuit, distance)
print(f'\n\nDistance: {distance}\nData qubits ({len(data_qus)}): {data_qus}\nAnchillas ({len(anchs)}): {anchs}\n All of them ({len(data_qus+anchs)}): {data_qus+anchs}')
all_qubits=data_qus+anchs

###!!!!!!!!!!!!!!!!!!Dont forget to map data qubits to 0-d²-1 !!!!!!!!! 


batch_marg=256
iters_marg=25

# weights = [0.25, 0.6, 0.1, 0.05]  # sums to 1.0
# for k, wk in zip(("t1","t2","t3","t4"), (0.25, 0.6, 0.1, 0.05)):
#     if k in cfg and cfg[k].get("enabled", False) and "p_start" in cfg[k]:
#         cfg[k]["p_start"] *= wk


print(f'\n\n----------------------   DATA Stats  --------------------------------------------------\n')
repeat_body_counts=rounds-1
cfg_m0 = calibrate_start_rates(
    build_mask_once=build_m0_once,
    qubits=len(data_qus), 
    rounds=repeat_body_counts, 

    qubits_ind=data_qus,
    cfg=cfg_data,
    p_idle_target=cfg_data["p_idle"],
    batch=batch_marg,
    iters=iters_marg,
    tol=0.05,   # ±15% is plenty for training
    verbose=True
)







print(f"\n\nCalibrated start_probs for M0 (batch: {batch_marg}, iters: {iters_marg}):")
for key in ("t1", "t2", "t3", "t4"):
    if key in cfg_m0 and cfg_m0[key].get("enabled", False) and "p_start" in cfg_m0[key]:
        print(cfg_m0[key]["p_start"])



## I am thinking of generating M_data and M_anchilla and M_CNOT. this way, we will have them ckeared out, so somehow we can combine them at the end 
cat_counts=[0, 0, 0, 0]

for _ in range(1000):
    #use the calibrated cfg to generate masks
    M_data, actives=mask_generator(qubits=len(data_qus), rounds=repeat_body_counts, qubits_ind=data_qus, cfg=cfg_m0, actives_list=True)
    measure_mask_stats(m=M_data, actives=actives)
    cat_counts=[c+int(b) for c,b in zip(cat_counts, actives)]
# print(f'\nFinal Mask M0:\n{M_data}')
# print(f'Each category coutns: {cat_counts}')


# M_data=mask_generator(qubits=len(data_qus), rounds=rounds, qubits_ind=data_qus, cfg=cfg_, actives_list=False)
# M_anch=mask_init(qubits=len(anchs), rounds=rounds)





#---------------------------------  FlipSIm-------------------------------------------------

S=1024
res = [
    mask_generator(
        qubits=len(data_qus),
        rounds=repeat_body_counts,
        qubits_ind=data_qus,   # you’re already generating data-only rows
        cfg=cfg_m0,
        actives_list=True
    )
    for _ in range(S)
]

# Unpack results
Ms, cats = zip(*res)                 # Ms: tuple of (D,R) arrays; cats: tuple of [c1,c2,c3,c4]
M_data = np.stack(Ms, axis=-1).astype(np.int8)       # -> shape (D, R, S)
actives = np.array(cats, dtype=np.int32).T           # -> shape (4, S)

print("M_data shape:", M_data.shape)
print("actives shape:", actives.shape)


M0_local=make_M_data_local_from_masks(masks=M_data, data_ids=data_qus, rounds=repeat_body_counts )
# print(f'M local is of shape: {M_data_local.shape}')



# prefix, pre_round, meas_round, anc_ids, repeat_count = extract_round_template(circuit)


# circ_by_round = [(pre_round, stim.Circuit(), meas_round) for _ in range(rounds)]

# round0 = stim.Circuit(); round0 += pre_round; round0 += meas_round
# print_svg(round0, "r0")
# print_svg(circuit, "circuit")
# #print(round0)   #only one form #round 


# # for i, item1 in enumerate(circ_by_round):
# #     print(f'round?{i}:\n ')
# #     for j in circ_by_round[i]:

# #         print(f'item :\n{j}\n')



#print(f'Lets check ids:\nData: {data_qus}\nAnchillas: {anchs}')  #  (All ok!)
stats = measure_stacked_mask_stats(M0_local, "M_data_local")
# sanity vs target p_idle:
p_idle = 0.005
print("\n\ntarget p_idle =", p_idle, "  measured p̂ =", stats["p_hat"])


##Check wtf these are and the physics behind them 
cnt = stats["per_shot_counts"]            # from the meter we made
print("mean", cnt.mean(), "var", cnt.var(), "Fano", cnt.var()/max(1e-9,cnt.mean()))
# Fano >> 1 means bursty (correlated); Fano ~ 1 means near-Poisson iid.

# ## einai entaksei gia twra na xrhsimopoimv chat giati kanw ta statistika klp. Sto montelo kane kai tipota moni sou.
# #  Den einai na paizoyme me ton kwdika gia ta statistika, marginalization klp


names = ("spatial", "temporal", "cluster_ext", "multi_scattered")
print('\nCategories stats for M0 injection')
for i, name in enumerate(names):
    shots_with = int((actives[i] > 0).sum())
    total = int(actives[i].sum())
    print(f"{name}: shots_with≥1={shots_with}/{S}  total_events={total}")
print(f'\n\n')

print(f'\n --------------------------- end data stats ------------------------------------------\n')





print(f'\n\n----------------------   Anch Stats  --------------------------------------------------\n')

cfg_m1=calibrate_start_rates(
    build_mask_once=build_m1_once,
    qubits=len(anchs), 
    rounds=repeat_body_counts, 
    qubits_ind=anchs,
    cfg=cfg_anch,
    p_idle_target=cfg_anch["p_idle"],
    batch=batch_marg,
    iters=iters_marg,
    tol=0.05,   # ±15% is plenty for training
    verbose=True
)




print(f"\n\nCalibrated start_probs for M1 (batch: {batch_marg}, iters: {iters_marg}):")
for key in ("t1", "t2", "t3", "t4"):
    if key in cfg_m1 and cfg_m1[key].get("enabled", False) and "p_start" in cfg_m1[key]:
        print(cfg_m1[key]["p_start"])



## I am thinking of generating M_data and M_anchilla and M_CNOT. this way, we will have them ckeared out, so somehow we can combine them at the end 
cat_counts=[0, 0, 0, 0]

for _ in range(1000):
    #use the calibrated cfg to generate masks
    M_anch, actives=mask_generator_M1(qubits=len(anchs), rounds=repeat_body_counts, qubits_ind=anchs, cfg=cfg_m1, actives_list=True)
    measure_mask_stats(m=M_anch,  actives=actives)
    cat_counts=[c+int(b) for c,b in zip(cat_counts, actives)]
print(f'Each category coutns: {cat_counts}')


M_anch=mask_generator_M1(qubits=len(anchs), rounds=repeat_body_counts, qubits_ind=anchs, cfg=cfg_m1, actives_list=False)





#---------------------------------  FlipSIm-------------------------------------------------

res = [
    mask_generator_M1(
        qubits=len(anchs),
        rounds=repeat_body_counts,
        qubits_ind=anchs,   # you’re already generating data-only rows
        cfg=cfg_m1,
        actives_list=True
    )
    for _ in range(S)
]

# Unpack results
Ms, cats = zip(*res)                 # Ms: tuple of (D,R) arrays; cats: tuple of [c1,c2,c3,c4]
M_anch = np.stack(Ms, axis=-1).astype(np.int8)       # -> shape (D, R, S)
actives = np.array(cats, dtype=np.int32).T           # -> shape (4, S)

print("M_anch shape:", M_anch.shape)
print("actives shape:", actives.shape)


M1_local=make_M_anc_local_from_masks(masks=M_anch, anc_ids=anchs, rounds=repeat_body_counts )



prefix, pre_round, meas_round, anc_ids, repeat_count= extract_round_template(circuit)


circ_by_round = [(pre_round, stim.Circuit(), meas_round) for _ in range(rounds)]

round0 = stim.Circuit(); round0 += pre_round; round0 += meas_round
print_svg(round0, "r0")
print_svg(circuit, "circuit")
#print(round0)   #only one form #round 


# for i, item1 in enumerate(circ_by_round):
#     print(f'round?{i}:\n ')
#     for j in circ_by_round[i]:

#         print(f'item :\n{j}\n')



#print(f'Lets check ids:\nData: {data_qus}\nAnchillas: {anchs}')  #  (All ok!)
stats = measure_stacked_mask_stats(M1_local, "MANCH_local")
# sanity vs target p_idle:
p_idle = 0.005
print("\n\ntarget p_idle =", p_idle, "  measured p̂ =", stats["p_hat"])


##Check wtf these are and the physics behind them 
cnt = stats["per_shot_counts"]            # from the meter we made
print("mean", cnt.mean(), "var", cnt.var(), "Fano", cnt.var()/max(1e-9,cnt.mean()))
# Fano >> 1 means bursty (correlated); Fano ~ 1 means near-Poisson iid.

## einai entaksei gia twra na xrhsimopoimv chat giati kanw ta statistika klp. Sto montelo kane kai tipota moni sou.
#  Den einai na paizoyme me ton kwdika gia ta statistika, marginalization klp




print('\nCategories stats for M1 injection')

for i, name in enumerate(names):
    shots_with = int((actives[i] > 0).sum())
    total = int(actives[i].sum())
    print(f"{name}: shots_with≥1={shots_with}/{S}  total_events={total}")
print(f'\n\n')

print(f'\n --------------------------- end anch stats ------------------------------------------\n')




print('\n-------------------M2 stats---------------------------\n')







cx, datss, anchs, repeates=extract_template_cx_pairs(circuit, distance    )
print(f'CX gates:{ cx}\n Datas: {datss}\n Anchillas: {anchs}\n Repeats: {repeates}\n\n')

cfg_m2 = calibrate_start_rates_m2(
    build_mask_once=build_m2_once,
    cfg=cfg_m2,
    p_idle_target=0.005,   # e.g., a bit smaller than M0/M1
    batch=batch_marg,
    iters=iters_marg,
    tol=0.05,
    verbose=True,
    gates=cx,
    rounds=repeat_body_counts
)

print(f"\n\nCalibrated start_probs for M2 (batch: {batch_marg}, iters: {iters_marg}):")
for key in ("t1", "t2", "t3", "t4"):
    if key in cfg_m2 and cfg_m2[key].get("enabled", False) and "p_start" in cfg_m2[key]:
        print(cfg_m2[key]["p_start"])

cat_counts=[0, 0, 0, 0]

for _ in range(1000):
    #use the calibrated cfg to generate masks
    M2, actives=mask_generator_M2(gates=cx, rounds=repeat_body_counts,  cfg=cfg_m2, actives_list=True)
    cat_counts=[c+int(b) for c,b in zip(cat_counts, actives)]
print(f'Each category coutns: {cat_counts}\n\n')



E=len(cx)
M2, actives=mask_generator_M2(gates=cx , rounds=repeat_body_counts, cfg=cfg_m2, actives_list=True)

#print(f'M2: \n{M2[:,:,1]}')

res = [
    mask_generator_M2(
        gates=cx,
        rounds=repeat_body_counts,
        cfg=cfg_m2,
        actives_list=True
    )
    for _ in range(S)
]


Ms, cats = zip(*res)                 # Ms: tuple of (D,R) arrays; cats: tuple of [c1,c2,c3,c4]
M2_local = np.stack(Ms, axis=-2).astype(np.int8)    # -> (E,R,S,2)
print("M2 shape:", M2_local.shape)
actives = np.array(cats, dtype=np.int32).T           # -> shape (4, S)


print("M_gates shape:", M2_local.shape)
print("actives shape:", actives.shape)



stats=measure_m2_mask_stats(M2_local)
p_idle = 0.005
print("\n\ntarget p_idle =", p_idle, "  measured p̂ =", stats["p_hat"])


##Check wtf these are and the physics behind them 
cnt = stats["per_shot_counts"]            # from the meter we made
print("mean", cnt.mean(), "var", cnt.var(), "Fano", cnt.var()/max(1e-9,cnt.mean()))

print("== M2 sanity ==")
stats_m2=m2_stats(M2_local)

print(m2_stats(M2_local))



print('\n ---------------------------end M2 ----------------------------------\n\n')





from .helpers import extract_round_template_plus_suffix, does_action_mask_have_anything


prefix, pre_round, meas_round, suffix, _,_= extract_round_template_plus_suffix(circuit)



# # after measuring p_hat_final on a big sample (e.g., S=1024)
# scale_M0 = 0.005 / 0.004069  # ≈ 1.23
# scale_M1 = 0.005 / 0.004248  # ≈ 1.18
# for k in ("t1","t2","t3","t4"):
#     if cfg_data.get(k,{}).get("enabled") and "p_start" in cfg_data[k]:
#         cfg_data[k]["p_start"] = min(1.0, cfg_data[k]["p_start"] * scale_M0)
#     if cfg_anch.get(k,{}).get("enabled") and "p_start" in cfg_anch[k]:
#         cfg_anch[k]["p_start"] = min(1.0, cfg_anch[k]["p_start"] * scale_M1)

# dets, obs, meas = run_batched_data_anc_plus_m2(
#     circuit=circuit,
#     M_data=M_data_local,
#     M_anc=M_anch_local,
#     data_ids=data_qus,
#     M2=M_2,
#     gate_pairs=cx,

#     anc_ids=anchs,
#     enable_M0=True,
#     enable_M1=True,

#     enable_M2=True
# )

# # Ensure arrays
# dets = np.asarray(dets, dtype=bool)   # detector events: (N_det, S) or (S, N_det)
# meas = np.asarray(meas, dtype=bool)   # measurement results: (A*R, S) or (S, A*R)

    # corrs.plot_intraround_corr_heatmaps(dets, slices, max_rounds=6)
    # S_tot=10_000
    # det, mr, ob= export_syndrome_dataset(circuit=circuit,
    #                         data_qubits=len(data_qus),
    #                         anchilla_qubits=len(anchs),
    #                         rounds=rounds,
    #                         data_ids=data_qus,
    #                         anc_ids=anchs,
    #                         gate_pairs=cx,
    #                         m0_build=build_m0_once,
    #                         m1_build=build_m1_once,
    #                         m2_build=build_m2_once,
    #                         m0_cfg=cfg_m0,
    #                         m1_cfg=cfg_m1,
    #                         m2_cfg=cfg_m2,
    #                         S_total=S_tot,
    #                         verbose=True)




    # # Show a tiny preview for one shot
    # r_preview = min(3, len(DET_by_round))
    # print("Shot 0 — detector events per round (first 3 rounds):")
    # for r in range(r_preview):
    #     print(f"  r={r}  events={DET_by_round[r][:,0].sum()}  bits={DET_by_round[r][:,0].astype(int)}")

    # if mr is not None:
    #     print("\nShot 0 — stabilizer outcomes per round (first 3 rounds):")
    #     for r in range(r_preview):
    #         print(f"  r={r}  ones={mr[:,r,0].sum()}  bits={mr[:,r,0].astype(int)}")

slices=corrs.detector_round_slices_3(circuit)
print(f'slices: {slices}')
from decoder.decoder_helpers import StimDecoderEnv, det_syndrome_tensor, det_syndrome_sequence_for_shot, det_for_round
from decoder.KalMamba import MambaBackbone, action_to_masks, sample_from_logits


env = StimDecoderEnv(circuit, data_qus, anc_ids, cx, rounds, slices)

# env.reset(M0_local=M_data_local, M1_local=M_anch_local, M2_local=M_2)
# for r in range(env.R):  # not 'rounds'
#     obs, done = env.step_inject(None)
#     if done: break

# dets, meas,obs_final, reward = env.finish_measure()
# print(f'Detectors: {dets.shape} (should be {len(circuit.get_detector_coordinates())})\n{dets[:,0]}')
# print(f'OBSERVATIONS: {obs_final.shape}\n{obs_final}')
# print("prefix DETs =", env._cnt(env.prefix))   # expect 4
# print("suffix DETs =", env._cnt(env.suffix))   # expect 4





S_tot=10_000   #injection

from surface_code.stats import analyze_decoding_stats

# analyze_decoding_stats(dets, obs_final, meas, M0=M_data_local, M1=M_anch_local, M2=M_2, rounds=rounds, ancillas=len(anchs), circuit=circuit, slices=slices)




# SxRxD = det_syndrome_tensor(dets, slices)  # (S, R, 8)
# # print("Syndrome tensor:", SxRxD.shape)     # (1024, 9, 8)



# seq0 = det_syndrome_sequence_for_shot(dets, slices, s=0)  # list of 9 arrays, each (8,)
# # print(f'Sequence of shot 0: {len(seq0)}, {len(seq0[0])}')



# mam=MambaBackbone(d_in=env.body_detectors).cuda()
# assert syndrome.shape[2] == mam.in_proj.in_features == 8
# y=mam.forward(syndrome)
# print(y.shape)


from decoder.KalMamba import DecoderAgent, RolloutBuffer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent=DecoderAgent(d_in=env.body_detectors, n_actions=2*len(data_qus) +1).to(device)


all_qids = list(data_qus) + list(anchs)
Q_total = max([0] + all_qids)
B = S  # shots in parallel
mode="discrete"



# --- episode init ---
obs = env.reset(M0_local, M1_local, M2_local)   # (S, 8) zeros
obs = torch.from_numpy(obs).float().to(device)  # (B, 8)
agent.begin_episode(B, device=device)
buf = RolloutBuffer(obs_dim=obs.shape[1])  # store per-step data
buf.reset()

print(f'\n\n--------Shapes & Init Check---------------\n\tM0_local: {M0_local.shape}\n\tM1_local: {M1_local.shape}\n\tM2_local: {M2_local.shape}\n\tB: {B}\n\tDevice: {device}\n\tQ_total:{Q_total}\n\tobs shape: {obs.shape}\t obs dim: {obs.shape[1]}\n--------------------------------------------\n')
stats_act = {"X": 0, "Z": 0}

for t in range(env.R):   # R = 9

#chooose action from agent
    logits, V_t, h_t = agent.act(obs)         
    a_t, logp_t = sample_from_logits(logits, mode=mode)       # your MultiDiscrete or Discrete policy
    a_t_cpu = a_t.detach().cpu()
    logp_t_cpu = logp_t.detach().cpu()
 #make action mask ready for injecton with flipsim
    action_mask = action_to_masks(a_t_cpu, mode, data_qus, num_qubits=Q_total+1, shots=B, classes_per_qubit=3)
    # x_mask=action_mask.get('X')
    # z_mask=action_mask.get('Z')

    # print(f'Action mask shapes:\n\tX: {x_mask.shape}\n\tZ:{z_mask.shape}')
    #check whether these masks have anythign in them:
    did_act, nx, nz = does_action_mask_have_anything(action_mask)
    if did_act:
        print(f"Agent proposed {nx} Xs and {nz} Zs in this step")

    stats_act["X"] += int(action_mask["X"].sum())
    stats_act["Z"] += int(action_mask["Z"].sum())




                            #inject corrections BEFORE injecting the noise. THe corrections reflects the stochastic nature of the channel.
                            #perfect corretions (making the syndrome be all zeros), could potentially be overshadowed by the noise injecyion after, cause we measure after the mask injections,
                            #but that's normal 
    obs_next_np, done = env.step_inject( action_mask=action_mask )  # returns (S, 8), bool
    # r_step = compute_step_reward(obs_next)             # e.g., -λ * (#ones)
    r_step=0
   # roll.add(obs)
    obs = torch.from_numpy(obs_next_np).float().to(device)
# episode end
dets, MR, obs_final, reward_terminal = env.finish_measure()
print("Episode corrections:", stats_act)

















#roll[-1] = (*roll[-1][:-1], roll[-1][-1] + reward_terminal)  # add terminal bonus to last step reward


DET_by_round = split_DET_by_round(dets, slices)


print(f'Detectors: {dets.shape} (should be {len(circuit.get_detector_coordinates())})\n{dets[:,0]}')
print(f'OBSERVATIONS: {obs_final.shape}\n{obs_final}')
print("prefix DETs =", env._cnt(env.prefix))   # expect 4
print("suffix DETs =", env._cnt(env.suffix))   # expect 4


SxRxD = det_syndrome_tensor(dets, slices)  # (S, R, 8)

list_with_syndromes_per_round=[]
# print('Syndrome for each round')
# for r in range(env.R):
#     dets_round=det_for_round(SxRxD, r)
#     list_with_syndromes_per_round.append(dets_round)
#     print(f'\tr={r} -> {dets_round}')
print(f'\nLogical error rate: {logical_error_rate(S, obs_final)}')






import torch
syndrome=torch.from_numpy(SxRxD).float().cuda() 

# compute advantages & update PPO









#print('\n\nCircuit:\ŋ')
#print(circuit)
#print(f'\n\n\tPREFIX:\n{prefix}\n\tPRE ROUND:\n{pre_round}\n\tMEAS ROUND:\n{meas_round}\n\tSUFFIX\n{suffix}')
#





# 
# 
# 
# for s in range(S_tot):
#     syndrome=get_syndrome_sequence_from_DET(DET_by_round=DET_by_round, s=s)
#     print(f'Shot: {s}')
#     for r in range(rounds):S
#         print(f'\tRound: {r}  ->  observation:{syndrome[r]}')


# def _count_dets_in_snippet(snippet_str: str) -> int:
#     return sum(1 for line in snippet_str.splitlines() if line.strip().startswith("DETECTOR"))
# print("[exec] prefix DETs +=", _count_dets_in_snippet(prefix))
# print("[exec] pre_round DETs +=", _count_dets_in_snippet(pre_round))   # should be 0
# print("[exec] meas_round DETs +=", _count_dets_in_snippet(meas_round)) # should be 8
# print("[exec] suffix DETs +=", _count_dets_in_snippet(suffix))         # should be 4




# export_dataset(circuit=circuit,
#                M_data_local=M_data_local,
#                M_anc_local=M_anch_local,
#                M2_local=M_2,
#                data_ids=data_qus,
#                anc_ids=anchs,
#                gate_pairs=cx,
#                out_prefix='extracted')
# # force Z on first data row in round 0 for first 16 shots
# M_forced = M_data_local.copy()
# M_forced[:] = 0
# M_forced[0, 0, :16] = 2  # your codes: X=1, Z=2, Y=3
# dets_f, _, _ = run_batched_data_only(circuit, M_forced, data_ids=data_qus)

# who = np.where(dets_f.any(axis=1))[0]
# print("detectors triggered by forced Z (subset):", who[:20])




# Toy: ancilla a=0, data d=1. Measure an X-check on data via H-CNOT-H then MR(a).
# toy = stim.Circuit("""
# H 0
# CX 0 1
# H 0
# MR 0
# DETECTOR rec[-1]
# """)

# S = 16
# sim = stim.FlipSimulator(batch_size=S, num_qubits=2, disable_stabilizer_randomization=True)

# # Inject a Z on data (q=1) BEFORE the round
# xmask = np.zeros((2, S), np.bool_)
# ymask = np.zeros((2, S), np.bool_)
# zmask = np.zeros((2, S), np.bool_)
# zmask[1, :] = True

# sim.broadcast_pauli_errors(pauli='Z', mask=zmask)  # always apply
# sim.do(toy)

# dets = sim.get_detector_flips()
# print("toy dets any?", dets.any())   # EXPECT: True


