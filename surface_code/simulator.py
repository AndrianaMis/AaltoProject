from surface_code.code_generator import build_planar_surface_code
import stim
import numpy as np
from surface_code.marginalize import calibrate_start_rates, build_m0_once, cfg_data, cfg_anch, build_m1_once, cfg_m2, build_m2_once, calibrate_start_rates_m2
from surface_code.stats import measure_mask_stats, measure_stacked_mask_stats, measure_m2_mask_stats, m2_stats, summarize_episode
from surface_code.inject import  run_batched_data_plus_anc, run_batched_data_anc_plus_m2
from surface_code.helpers import print_svg, extract_round_template, get_data_and_ancilla_ids_by_parity, make_M_data_local_from_masks, make_M_anc_local_from_masks, extract_template_cx_pairs,split_DET_by_round, logical_error_rate, decode_action_index, encode_obs, set_group_lr, summary
from surface_code.M1 import mask_generator_M1
from surface_code.M2 import mask_generator_M2
from visuals.corrs import make_Crr_heatmaps
from surface_code.M0 import mask_generator, mask_init
from visuals import corrs
from .export import export_syndrome_dataset
from surface_code.stats import analyze_decoding_stats, summarize_noise
from decoder.KalMamba import DecoderAgent, RolloutBuffer, RolloutConfig
from .helpers import extract_round_template_plus_suffix, does_action_mask_have_anything, find_neighboring_qubits
import torch
from decoder.decoder_helpers import StimDecoderEnv, det_syndrome_tensor, det_syndrome_sequence_for_shot, det_for_round
from decoder.KalMamba import MambaBackbone, action_to_masks, sample_from_logits, optimize_ppo
from decoder.reward_functions import step_reward, final_reward, round_overcorr_metrics_discrete
from visuals.plots import plot_LERvsSTEPS, plot_step_reward_trends, plot_ev_kl_entropy, plot_loss_v_pi, plot_LER_rstep_finalr, plot_effectiveness, plot_coef
import time
import os
import surface_code.evaulate as evaluate
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
        # print(f"Shot {shot}, Round {r}:")
        # print("  X stabilizers:", x_syndromes)
        # print("  Z stabilizers:", z_syndromes)

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
# cat_counts=[0, 0, 0, 0]

# for _ in range(1000):
#     #use the calibrated cfg to generate masks
#     M_data, actives=mask_generator(qubits=len(data_qus), rounds=repeat_body_counts, qubits_ind=data_qus, cfg=cfg_m0, actives_list=True)
#     measure_mask_stats(m=M_data, actives=actives)
#     cat_counts=[c+int(b) for c,b in zip(cat_counts, actives)]








#---------------------------------  FlipSIm-------------------------------------------------
def generate_M0(seed=None , S:int=1024):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    res = [
        mask_generator(
            qubits=len(data_qus),
            rounds=repeat_body_counts,
            qubits_ind=data_qus,   # you’re already generating data-only rows
            cfg=cfg_m0,
            actives_list=True, 
            seed=None, 
            rng=rng
        )
        for _ in range(S)
    ]

    # Unpack results
    Ms, cats = zip(*res)                 # Ms: tuple of (D,R) arrays; cats: tuple of [c1,c2,c3,c4]
    M_data = np.stack(Ms, axis=-1).astype(np.int8)       # -> shape (D, R, S)
    actives = np.array(cats, dtype=np.int32).T           # -> shape (4, S)

    # print(f'For M0 generation we are using the following p_starts : \n')
    # for key in ("t1", "t2", "t3", "t4"):
    #         if key in cfg_m0 and cfg_m0[key].get("enabled", False) and "p_start" in cfg_m0[key]:
    #             print('\t-',cfg_m0[key]["p_start"])
    
    # print("M_data shape:", M_data.shape)
    # print("actives shape:", actives.shape)


    M0_local=make_M_data_local_from_masks(masks=M_data, data_ids=data_qus, rounds=repeat_body_counts )
    stats = measure_stacked_mask_stats(M0_local, "M_data_local")

    return M0_local  , stats 
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



# sanity vs target p_idle:
# p_idle = 0.005
# print("\n\ntarget p_idle =", p_idle, "  measured p̂ =", stats["p_hat"])


# ##Check wtf these are and the physics behind them 
# cnt = stats["per_shot_counts"]            # from the meter we made
# print("mean", cnt.mean(), "var", cnt.var(), "Fano", cnt.var()/max(1e-9,cnt.mean()))
# # Fano >> 1 means bursty (correlated); Fano ~ 1 means near-Poisson iid.

# # ## einai entaksei gia twra na xrhsimopoimv chat giati kanw ta statistika klp. Sto montelo kane kai tipota moni sou.
# # #  Den einai na paizoyme me ton kwdika gia ta statistika, marginalization klp


# names = ("spatial", "temporal", "cluster_ext", "multi_scattered")
# print('\nCategories stats for M0 injection')
# for i, name in enumerate(names):
#     shots_with = int((actives[i] > 0).sum())
#     total = int(actives[i].sum())
#     print(f"{name}: shots_with≥1={shots_with}/{S}  total_events={total}")
# print(f'\n\n')

# print(f'\n --------------------------- end data stats ------------------------------------------\n')





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
# cat_counts=[0, 0, 0, 0]

# for _ in range(1000):
#     #use the calibrated cfg to generate masks
#     M_anch, actives=mask_generator_M1(qubits=len(anchs), rounds=repeat_body_counts, qubits_ind=anchs, cfg=cfg_m1, actives_list=True)
#     measure_mask_stats(m=M_anch,  actives=actives)
#     cat_counts=[c+int(b) for c,b in zip(cat_counts, actives)]
# print(f'Each category coutns: {cat_counts}')


# M_anch=mask_generator_M1(qubits=len(anchs), rounds=repeat_body_counts, qubits_ind=anchs, cfg=cfg_m1, actives_list=False)





#---------------------------------  FlipSIm-------------------------------------------------
def generate_M1(seed=None, S:int=1024):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    res = [
        mask_generator_M1(
            qubits=len(anchs),
            rounds=repeat_body_counts,
            qubits_ind=anchs,   # you’re already generating data-only rows
            cfg=cfg_m1,
            actives_list=True, 
            seed=None,
            rng=rng

        )
        for _ in range(S)
    ]

    # Unpack results
    Ms, cats = zip(*res)                 # Ms: tuple of (D,R) arrays; cats: tuple of [c1,c2,c3,c4]
    M_anch = np.stack(Ms, axis=-1).astype(np.int8)       # -> shape (D, R, S)
    actives = np.array(cats, dtype=np.int32).T           # -> shape (4, S)

    # print("M_anch shape:", M_anch.shape)
    # print("actives shape:", actives.shape)


    # print(f'For M1 generation we are using the following p_starts : \n')
    # for key in ("t1", "t2", "t3", "t4"):
    #         if key in cfg_m1 and cfg_m1[key].get("enabled", False) and "p_start" in cfg_m1[key]:
    #             print('\t-',cfg_m1[key]["p_start"])

    M1_local=make_M_anc_local_from_masks(masks=M_anch, anc_ids=anchs, rounds=repeat_body_counts )
    stats = measure_stacked_mask_stats(M1_local, "MANCH_local")
    # cnt = stats["per_shot_counts"]            # from the meter we made
    # print("mean", cnt.mean(), "var", cnt.var(), "Fano", cnt.var()/max(1e-9,cnt.mean()))
 

    return M1_local, stats


prefix, pre_round, meas_round, anc_ids, repeat_count= extract_round_template(circuit)


circ_by_round = [(pre_round, stim.Circuit(), meas_round) for _ in range(rounds)]

round0 = stim.Circuit(); round0 += pre_round; round0 += meas_round
# print_svg(round0, "r0")
# print_svg(circuit, "circuit")




##Check wtf these are and the physics behind them 
# cnt = stats["per_shot_counts"]            # from the meter we made
# print("mean", cnt.mean(), "var", cnt.var(), "Fano", cnt.var()/max(1e-9,cnt.mean()))
# Fano >> 1 means bursty (correlated); Fano ~ 1 means near-Poisson iid.

## einai entaksei gia twra na xrhsimopoimv chat giati kanw ta statistika klp. Sto montelo kane kai tipota moni sou.
#  Den einai na paizoyme me ton kwdika gia ta statistika, marginalization klp




# print('\nCategories stats for M1 injection')

# for i, name in enumerate(names):
#     shots_with = int((actives[i] > 0).sum())
#     total = int(actives[i].sum())
#     print(f"{name}: shots_with≥1={shots_with}/{S}  total_events={total}")
# print(f'\n\n')

print(f'\n --------------------------- end anch stats ------------------------------------------\n')




print('\n-------------------M2 stats---------------------------\n')







cx, datss, anchs, repeates=extract_template_cx_pairs(circuit, distance    )
print(f'CX gates:{ cx}\n Datas: {datss}\n Anchillas: {anchs}\n Repeats: {repeates}\n\n')

cfg_m2 = calibrate_start_rates_m2(
    build_mask_once=build_m2_once,
    cfg=cfg_m2,
    p_idle_target=cfg_m2["p_idle"],   # e.g., a bit smaller than M0/M1
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

# for _ in range(1000):
#     #use the calibrated cfg to generate masks
#     M2, actives=mask_generator_M2(gates=cx, rounds=repeat_body_counts,  cfg=cfg_m2, actives_list=True)
#     cat_counts=[c+int(b) for c,b in zip(cat_counts, actives)]
# print(f'Each category coutns: {cat_counts}\n\n')




#print(f'M2: \n{M2[:,:,1]}')
def generate_M2(seed=None, S:int=1024):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    res = [
        mask_generator_M2(
            gates=cx,
            rounds=repeat_body_counts,
            cfg=cfg_m2,
            actives_list=True,
            seed=None,
            rng=rng
        )
        for _ in range(S)
    ]

    # print(f'For M2 generation we are using the following p_starts : \n')
    # for key in ("t1", "t2", "t3", "t4"):
    #         if key in cfg_m2 and cfg_m2[key].get("enabled", False) and "p_start" in cfg_m2[key]:
    #             print('\t-',cfg_m2[key]["p_start"])
    Ms, cats = zip(*res)                 # Ms: tuple of (D,R) arrays; cats: tuple of [c1,c2,c3,c4]
    M2_local = np.stack(Ms, axis=-2).astype(np.int8)    # -> (E,R,S,2)
    actives = np.array(cats, dtype=np.int32).T           # -> shape (4, S)


    stats=measure_m2_mask_stats(M2_local)

    cnt = stats["per_shot_counts"]            # from the meter we made
    print("mean", cnt.mean(), "var", cnt.var(), "Fano", cnt.var()/max(1e-9,cnt.mean()))


    return M2_local, stats

# ##Check wtf these are and the physics behind them 

# print("== M2 sanity ==")
# stats_m2=m2_stats(M2_local)

# print(m2_stats(M2_local))



print('\n ---------------------------end M2 ----------------------------------\n\n')







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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coords_det=circuit.get_detector_coordinates()
coords_qubits=circuit.get_final_qubit_coordinates()
data_ids=data_qus
anchilla_ids=anchs
data_coords = {q: coords_qubits[q] for q in data_ids}
anc_coords={q: coords_qubits[q] for q in anchilla_ids}
all_det_coords = circuit.get_detector_coordinates()
round_det_coords = []
for (a, b) in slices:
    # list of coords for detectors in this round
    coords_this_round = [coords_det[i] for i in range(a, b)]
    round_det_coords.append(coords_this_round)





all_qids = list(data_qus) + list(anchs)
Q_total = max([0] + all_qids)
B = 1024  # shots in parallel
S=1024
mode="discrete"





print("DATA qubit coords:")
for q,c in data_coords.items():
    print(q, c)

print("\nANCILLA qubit coords:")
for q,c in anc_coords.items():
    print(q, c)

# --- DETECTOR COORDS ---
det_coords = circuit.get_detector_coordinates()

print("\nDETECTORS by round:")
for r,(a,b) in enumerate(slices):
    print(f"Round {r}:")
    print(round_det_coords[r])


find_neighboring_qubits(det_coords=round_det_coords, data_ids=data_ids, data_coords= data_coords,r=0)


def train_agent(agent: DecoderAgent, env:StimDecoderEnv, episodes:int,
                    optimizer, base_seed:int=1234):
    pos_frac=[]
    neg_frac=[]
    losses_v=[]
    losses_p=[]
    kls=[]
    evs=[]
    lers=[]
    entropies=[]
    raw_rewards_list=[]
    final_rewards=[]
    no_improve=[]
    effective=[]
    act=[]
    idle=[]
    entropy_coefs=[]
    best_ler = float("inf")
    check_time = time.strftime("%H:%M", time.localtime(time.time()))

    kl_stop=0.020


    tot_ler=0
    done_mask=torch.tensor(1.0)
 
    best_ckpt_path="RANDOM"

    for ep in range(episodes):
        print(f'\n[Episode {ep}]')
        #just a check for the paramters adn the grads   
        rng_seed = base_seed + ep

        # --- episode init ---
        M0_local, stats_M0=generate_M0(seed=rng_seed)
        M1_local, stats_M1=generate_M1(seed=rng_seed)
        M2_local, stats_M2=generate_M2(seed=rng_seed)

        noise_summary=summarize_noise(stats_M0=stats_M0, stats_M1=stats_M1, stats_M2=stats_M2)
        noise_scalar = noise_summary["noise_level_scalar"]
        print(f'Noise summary \n\t-noise scalar: {noise_scalar}\n\tfano M0:{ noise_summary["M0"]["fano"]}\tfano M1:{ noise_summary["M1"]["fano"]}\tfano M2:{ noise_summary["M2"]["fano"]}')
        obs = env.reset(M0_local, M1_local, M2_local)   # (S, 8) zeros
        obs = torch.from_numpy(obs).float().to(device)  # (B, 8)
        agent.begin_episode(B, device=device)
        buf = RolloutBuffer(obs_dim=9)  # store per-step data
        buf.reset()

        if ep==0: print(f'\n\n--------Shapes & Init Check---------------\n\tM0_local: {M0_local.shape}\n\tM1_local: {M1_local.shape}\n\tM2_local: {M2_local.shape}\n\tB: {B}\n\tDevice: {device}\n\tQ_total:{Q_total}\n\tobs shape: {obs.shape}\t obs dim: {obs.shape[1]}\n--------------------------------------------\n')
        stats_act = {"X": 0, "Z": 0}

        pos_frac_ep=[]
        neg_frac_ep=[]

        progress = ep / float(episodes)   # ep = current episode index (0..total_episodes-1)
        progress = max(0.0, min(1.0, progress)) # clamp just in case


        all_action_masks=[]
        obs_prev_np=None
        feature_vector=encode_obs(obs_curr=obs, obs_prev=None, last_action= 0, round_idx=0, total_rounds=env.R)
        feature_vector = torch.from_numpy(feature_vector).to(device)  # (B, 9)
        feature_vector = (feature_vector - feature_vector.mean(dim=0, keepdim=True)) / (
                        feature_vector.std(dim=0, keepdim=True) + 1e-6)
        nx_rounds=[]
        nz_rounds=[]


    #   print(f'\tFeature vector should have very small values and is of shape: {feature_vector.shape}\n: random shot 500:\n{feature_vector[500,:]} ')
        for t in range(env.R):   # R = 9

            '''
            1. the agent.act is using the critic-value_head to return V_t which is the estimate of the reward of the state
            2. also it is giving us the logits, which are from the actor-policy_head 
            3. we then get the action from the logits , which after is giving us the r_step reward and thus the actual value of the state
            '''
            logits, V_t, h_t = agent.act(feature_vector)    
            a_t, logp_t = sample_from_logits(logits, mode=mode)       # MultiDiscrete or Discrete policy
            a_t_cpu = a_t.detach().cpu()  #should be of size 2*d without zeros? 
            logp_t_cpu = logp_t.detach().cpu()

            logp_t    = logp_t.detach()          # <- detach
            V_t       = V_t.detach()  
            action_mask = action_to_masks(a_t_cpu, mode, data_qus, num_qubits=Q_total+1, shots=B, classes_per_qubit=3)  #make action mask ready for injecton with flipsim

            gate,qubit =decode_action_index(a=a_t,D=len(data_qus), device=device )
            x_mask=action_mask.get('X')
            z_mask=action_mask.get('Z')
            did_act, _, _ = does_action_mask_have_anything(action_mask)    #check whether these masks have anythign in them:
            nx = x_mask[data_qus, :].sum(axis=0).astype(np.int32)  # (S,)
            nz = z_mask[data_qus, :].sum(axis=0).astype(np.int32) 
            nx_rounds.append(nx)
            nz_rounds.append(nz)
            stats_act["X"] += int(action_mask["X"].sum())
            stats_act["Z"] += int(action_mask["Z"].sum())
            # print(f'Shot 100 -> nx {nx[100]}\tnz {nz[100]}')   #because we are using discrete action space , we get only one correction åer round, which is in only one qubit. 
            # print(f'nx vect: {nx.sum()}, nz vect: {nz.sum()}')                        #so, vectorized, we have max 1024 correction per round

            '''
            #inject corrections BEFORE injecting the noise. THe corrections reflects the stochastic nature of the channel.
            #perfect corretions (making the syndrome be all zeros), could potentially be overshadowed by the noise injecyion after, cause we measure after the mask injections,
            #but that's normal 

            We might actually need to compress the observations (#shots, #dets) at some point, since when the code distance grows, the vector will become v v big
            SO it would be great to haev a feature vector func???
            in irder for d_in to remain 8 


            So, below im showing some important features that create a more clear picture of the curent and prev state of the env, if we make the feature vecto
            encoding thesee feature, we have a more rich representation of the env
            '''
        
            obs_current, done = env.step_inject( action_mask=action_mask )  # returns (S, 8), bool
            if t==0:
                print(f'obs fired!: {obs_current.shape}\n {obs_current.sum()}')
            r_step=step_reward(obs_prev_round=obs_prev_np, obs_round=obs_current, nx=nx, nz=nz)
            last_action=gate
            feature_vector=encode_obs(obs_curr=obs_current, obs_prev=obs_prev_np, last_action=gate, round_idx=t, total_rounds=env.R)
            feature_vector = torch.from_numpy(feature_vector).float().to(device)  # (B, 9)
            feature_vector = (feature_vector - feature_vector.mean(dim=0, keepdim=True)) / (
                        feature_vector.std(dim=0, keepdim=True) + 1e-6)

            if t==int(env.R/2):
                print(f'We got \n\tpositive reward? -> {(r_step>0).sum()} \n\tnegative reward? -> {(r_step<0).sum()}, zeros -> {(r_step==0).sum()}')
                print(f'Reward: {r_step}')

            obs_prev_np=obs_current
            pos_frac_ep.append((r_step > 0).mean())
            neg_frac_ep.append((r_step < 0).mean())
            all_action_masks.append(action_mask)
            

            '''Update buffer, compute GAE'''

            obs_c_tensor=torch.from_numpy(obs_current).float().to(device)
            r_step_tensor=torch.from_numpy(r_step).float().to(device)
            r_step_tensor = torch.nan_to_num(r_step_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            # (optional) clip tiny range first days
            r_step_tensor = torch.clamp(r_step_tensor, -1.0, 1.0)
        #  if ep==5: print(f'step reward: {r_step}')

            done_bool = (t == env.R - 1)          # shape: scalar bool
            done_mask = torch.zeros(B, device=device) if done_bool else torch.ones(B, device=device)

            buf.add(obs_t=feature_vector,action_t=a_t, logp_t=logp_t, value_t=V_t, reward_t=r_step_tensor, mask_t=done_mask)




            # Example: suppose you keep detector slices per round
            # slices = [(a0,b0), (a1,b1), ..., (aR,bR)]
        

        # episode end
        dets, MR, obs_final, reward_terminal = env.finish_measure()
        final_rew=final_reward(obs_flips=obs_final)
        final_rew_tensor = torch.as_tensor(final_rew, dtype=torch.float32, device=device)
        final_rew_tensor = torch.nan_to_num(final_rew_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        final_rew_tensor = torch.clamp(final_rew_tensor, -1.0, 1.0)         #oxi gia pnta
        buf.rewards[-1] = buf.rewards[-1] + final_rew_tensor

        det_count=np.stack([dets[a:b, :].sum(axis=0) for (a, b) in slices], axis=0)
        all_metrics = []
        for t in range(env.R - 1):
            det_prev = det_count[t - 1] if t > 0 else np.zeros(S)
            det_now  = det_count[t]

            # assume you stored per-round nx,nz arrays
            nx_t, nz_t = nx_rounds[t], nz_rounds[t]

            m = round_overcorr_metrics_discrete(det_prev, det_now, nx_t, nz_t)
            all_metrics.append(m)

        # average over rounds
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print("\nepisode over-corr:\n", avg)
        no_impr=avg['no_improve_rate']
        eff=avg["eff_act"]
        act_rate=avg["act_rate"]
        idle_rate=avg["idle_rate"]
        no_improve.append(no_impr)
        effective.append(eff)
        act.append(act_rate)
        idle.append(idle_rate)
        # corrs.plot_intraround_corr_heatmaps(dets, slices, max_rounds=6)
        # corrs.make_Crr_heatmaps(M0=M0_local, M1=M1_local, M2=M2_local, DET=dets, MR=MR, R=rounds, A=len(anc_ids))



        buf.finalize(cfg=RolloutConfig(),last_value=None)
        # After buf.finalize(...)
        advs = buf._advs
        rets = buf._rets

        # Defensive clamp (keeps training going while you debug)
        buf._advs = torch.clamp(advs, -10.0, 10.0)
        buf._rets = torch.clamp(rets, -10.0, 10.0)

        ler=logical_error_rate(S, obs_final)
        lers.append(ler)
        print(f'LER: {ler}')
        # ler is a scalar float, e.g. logical error rate for this episode
        if float(ler) < float(best_ler):
            best_ler = ler
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)

            best_ckpt_path = f"stage2_best_ler_{best_ler}_{str(check_time)}.pt"

            torch.save(
                {"agent": agent.state_dict(),
                "episode": ep,
                "noise_scalar": noise_scalar,

                "ler": best_ler},
                best_ckpt_path,
            )
            print(f"[CKPT] New best LER: {best_ler:.4f} at episode {ep}")





        #print(f'obs in the buffer: {len(buf.obs)}')   buffer obs of shape: torch.Size([9, 1024, 9])
        stats = optimize_ppo(agent, buf, optimizer,
                        clip_eps=0.04, epochs=2, batch_size=2048,
                        entropy_coef=3e-3, entropy_coef_min=1e-4,
                        update_idx=ep, total_updates=episodes, progress=progress)
        ent_coef=stats["entropy_coef_used"]
        entropy_coefs.append(ent_coef)
        # take mean across rounds in this episode

        raw_rewards = torch.stack(buf.rewards, dim=0)  # (T, B)
        raw_step_mean = raw_rewards.mean().item()
        r_step_mean=r_step.mean()
        raw_rewards_list.append(r_step_mean)
        final_rewards.append(final_rew.mean())
        


        pos_frac.append(np.mean(pos_frac_ep))
        neg_frac.append(np.mean(neg_frac_ep))
        losses_p.append(stats["loss_pi"])
        losses_v.append(stats["loss_v"])
        entropies.append(stats["entropy"])
        kls.append(stats["kl"])
        evs.append(stats["ev"])

        tot_ler+=ler

        summary(ep=ep,buf=buf,  env=env, stats_act=stats_act,
                s=stats, ler=ler, advs=advs, rets=rets, R=env.R, B=B, dets=dets, slices=slices, MR=MR, all_action_masks=all_action_masks )

    ##We are optimize and tubnig the learning rate of the actor 
        kl = stats["kl"]
        if kl is not None and kl > kl_stop:
            new_lr = set_group_lr(optimizer, "actor", factor=0.9)
            if new_lr is not None:
                print(f"[tune] KL={kl:.4f} > {kl_stop} → actor lr ↓ to {new_lr:.2e}")
            # elif kl < 0.010:
            #     new_lr = set_group_lr(optimizer, "actor", factor=1.1)




   # summarize_episode(all_action_masks=all_action_masks, observables=obs_final)




    # analyze_decoding_stats(dets, obs_final, MR, M0=M0_local, M1=M1_local, M2=M2_local, rounds=rounds, ancillas=len(anchs), circuit=circuit, slices=slices)
    plot_LERvsSTEPS(lers)
    plot_step_reward_trends(pos_frac, neg_frac)
    plot_ev_kl_entropy(ev=evs,kl=kls,entropy=entropies)
    plot_coef(coefs=entropy_coefs)
    plot_loss_v_pi(v=losses_v, pi=losses_p)
    plot_LER_rstep_finalr(ler=lers, r_step=raw_rewards_list, finals=final_rewards)
    plot_effectiveness(eff=effective, no_improve=no_improve, act=act, idle=idle)
    DET_by_round = split_DET_by_round(dets, slices)




if __name__=='__main__':
    episodes=300
    d_in=9
    env = StimDecoderEnv(circuit, data_qus, anc_ids, cx, rounds, slices)

    agent=DecoderAgent(d_in=9, n_actions=2*len(data_qus) +1).to(device)
    ckpt = torch.load("ppo_decoder_stage1_noiseless.pt", map_location=device)
    agent.load_state_dict(ckpt["policy_state_dict"])
    base_seed=1234
    eval_seed=4321
    print("Loaded Stage-1 policy + backbone weights.")
    optim_groups = [
        {"name": "actor",   "params": agent.heads.pi.parameters(),      "lr": 6e-4},
        {"name": "critic",  "params": agent.heads.v.parameters(),       "lr": 1e-3},
        {"name": "backbone","params": agent.backbone.parameters(),      "lr": 3e-4},
    ]

    optimizer = torch.optim.Adam(optim_groups, betas=(0.9, 0.999), eps=1e-8)
    train_agent(agent=agent, env=env, episodes=episodes, optimizer=optimizer, base_seed=base_seed)
    eval_env=StimDecoderEnv(circuit=circuit,data_ids=data_qus,anc_ids= anc_ids, gate_pairs=cx, rounds=rounds, round_slices=slices)
    evaluate.evaluate_agent(train_seed=base_seed, eval_seed=eval_seed, env=eval_env, device=device, agent=agent, S=S, mode=mode, data_ids=data_ids, Q_total=Q_total)


'''Save Stage 1 policy'''
# stage1_ckpt_path = "ppo_decoder_stage1_noiseless.pt"
# torch.save(
#     {
#         "policy_state_dict": agent.state_dict(),
#         # optional, if you want to reuse them:
#         # "optimizer_state_dict": optimizer.state_dict(),
#         # "config": config_dict,
#     },
#     stage1_ckpt_path,
# )
# print("Saved Stage-1 checkpoint to", stage1_ckpt_path)



'''Find syndrome for each round'''
#SxRxD = det_syndrome_tensor(dets, slices)  # (S, R, 8)
#list_with_syndromes_per_round=[]
# print('Syndrome for each round')
# for r in range(env.R):
#     dets_round=det_for_round(SxRxD, r)
#     list_with_syndromes_per_round.append(dets_round)
#     print(f'\tr={r} -> {dets_round}')
#syndrome=torch.from_numpy(SxRxD).float().cuda() 




