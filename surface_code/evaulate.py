import numpy as np
import torch
import numpy as np
from surface_code.helpers import print_svg, extract_round_template, get_data_and_ancilla_ids_by_parity, make_M_data_local_from_masks, make_M_anc_local_from_masks, extract_template_cx_pairs,split_DET_by_round, logical_error_rate, decode_action_index, encode_obs, set_group_lr, summary

from surface_code.stats import analyze_decoding_stats, summarize_noise
from decoder.KalMamba import DecoderAgent
from decoder.decoder_helpers import StimDecoderEnv
from decoder.KalMamba import action_to_masks, sample_from_logits
from decoder.reward_functions import step_reward, final_reward, round_overcorr_metrics_discrete
import random
from surface_code.simulator import generate_M0,generate_M1,generate_M2

def evaal(ids, seed:int, env:StimDecoderEnv, device, agent:DecoderAgent, S:int, mode, data_ids, Q_total:int, strr:str):
    print(f'\n\n\t\tEVALUATION ON {strr} MASKS\n')
    lers=[]
    noises=[]
    ep=0
    for id in ids:
        rng_seed_reuse=int(id)+seed  #should be seeds that have been used to create a  subset of masks that the agent has seen before
        M0_local, stats_M0=generate_M0(seed=rng_seed_reuse)
        M1_local, stats_M1=generate_M1(seed=rng_seed_reuse)
        M2_local, stats_M2=generate_M2(seed=rng_seed_reuse)
        print(f'\n--Noise Stats--')
        noise_summary=summarize_noise(stats_M0=stats_M0, stats_M1=stats_M1, stats_M2=stats_M2)
        noise_scalar = noise_summary["noise_level_scalar"]

        obs=env.reset(M0_local=M0_local, M1_local=M1_local,M2_local=M2_local)
        obs = torch.from_numpy(obs).float().to(device)  # (B, 8)

        agent.eval()
        agent.begin_episode(S, device=device)

        obs_prev_np=None
        feature_vector=encode_obs(obs_curr=obs, obs_prev=None, last_action= 0, round_idx=0, total_rounds=env.R)
        feature_vector = torch.from_numpy(feature_vector).to(device)  # (B, 9)
        feature_vector = (feature_vector - feature_vector.mean(dim=0, keepdim=True)) / (
                        feature_vector.std(dim=0, keepdim=True) + 1e-6)
        for t in range(env.R):
            logits, value, h = agent.act(feature_vector)
            action,_=sample_from_logits(logits=logits, mode=mode)
            a_t_cpu = action.detach().cpu()

            action_mask = action_to_masks(a_t_cpu, mode, data_ids=data_ids, num_qubits=Q_total+1, shots=S, classes_per_qubit=3)
            gate,qubit =decode_action_index(a=action,D=len(data_ids), device=device )

            obs_current, done = env.step_inject( action_mask=action_mask )

            feature_vector=encode_obs(obs_curr=obs_current, obs_prev=obs_prev_np, last_action=gate, round_idx=t, total_rounds=env.R)
            feature_vector = torch.from_numpy(feature_vector).float().to(device)  # (B, 9)
            feature_vector = (feature_vector - feature_vector.mean(dim=0, keepdim=True)) / (
                        feature_vector.std(dim=0, keepdim=True) + 1e-6)
            obs_prev_np=obs_current

        dets, MR, obs_final, reward_terminal = env.finish_measure()
        ler=logical_error_rate(S, obs_final)

        lers.append(ler)
        noises.append(noise_scalar)
        print(f'EP: [{ep}] ->  LER: {ler}  Noise scalar: {noise_scalar}  Seed: {id}')
        ep+=1

    lers=np.array(lers)
    noises=np.array(noises)
    print(f"\n===== {strr}-SEED EVAL SUMMARY =====")
    print(f"LER mean={lers.mean():.5f}   std={lers.std():.5f}")
    print(f"LER min={lers.min():.5f}     max={lers.max():.5f}")
    print(f"noise mean={noises.mean():.5f}")
    print("===================================\n")


@torch.no_grad()
def evaluate_agent(train_seed:int, eval_seed:int, env:StimDecoderEnv, device, agent:DecoderAgent, S:int, mode, data_ids, Q_total:int):
    eval_episode_ids=np.arange(0,50)  
    random.shuffle(eval_episode_ids)
    evaal(ids=eval_episode_ids,seed=train_seed, env=env, agent=agent, S=S, mode=mode, data_ids=data_ids, Q_total=Q_total, strr="SEEN" , device=device)
    evaal(ids=eval_episode_ids,seed=eval_seed, env=env, agent=agent, S=S, mode=mode, data_ids=data_ids, Q_total=Q_total, strr="UNSEEN", device=device )
    
    # lers=[]
    # noises=[]
    # ep=0
    # for id in eval_episode_ids:
    #     rng_seed_reuse=int(id)+train_seed  #should be seeds that have been used to create a  subset of masks that the agent has seen before
    #     M0_local, stats_M0=generate_M0(seed=rng_seed_reuse)
    #     M1_local, stats_M1=generate_M1(seed=rng_seed_reuse)
    #     M2_local, stats_M2=generate_M2(seed=rng_seed_reuse)
    #     print(f'\n--Noise Stats--')
    #     noise_summary=summarize_noise(stats_M0=stats_M0, stats_M1=stats_M1, stats_M2=stats_M2)
    #     noise_scalar = noise_summary["noise_level_scalar"]

    #     obs=env.reset(M0_local=M0_local, M1_local=M1_local,M2_local=M2_local)
    #     obs = torch.from_numpy(obs).float().to(device)  # (B, 8)

    #     agent.eval()
    #     agent.begin_episode(S, device=device)

    #     obs_prev_np=None
    #     feature_vector=encode_obs(obs_curr=obs, obs_prev=None, last_action= 0, round_idx=0, total_rounds=env.R)
    #     feature_vector = torch.from_numpy(feature_vector).to(device)  # (B, 9)
    #     feature_vector = (feature_vector - feature_vector.mean(dim=0, keepdim=True)) / (
    #                     feature_vector.std(dim=0, keepdim=True) + 1e-6)
    #     for t in range(env.R):
    #         logits, value, h = agent.act(feature_vector)
    #         action,_=sample_from_logits(logits=logits, mode=mode)
    #         a_t_cpu = action.detach().cpu()

    #         action_mask = action_to_masks(a_t_cpu, mode, data_ids=data_ids, num_qubits=Q_total+1, shots=S, classes_per_qubit=3)
    #         gate,qubit =decode_action_index(a=action,D=len(data_ids), device=device )

    #         obs_current, done = env.step_inject( action_mask=action_mask )

    #         feature_vector=encode_obs(obs_curr=obs_current, obs_prev=obs_prev_np, last_action=gate, round_idx=t, total_rounds=env.R)
    #         feature_vector = torch.from_numpy(feature_vector).float().to(device)  # (B, 9)
    #         feature_vector = (feature_vector - feature_vector.mean(dim=0, keepdim=True)) / (
    #                     feature_vector.std(dim=0, keepdim=True) + 1e-6)
    #         obs_prev_np=obs_current

    #     dets, MR, obs_final, reward_terminal = env.finish_measure()
    #     ler=logical_error_rate(S, obs_final)

    #     lers.append(ler)
    #     noises.append(noise_scalar)
    #     print(f'EP: [{ep}] ->  LER: {ler}  Noise scalar: {noise_scalar}  Seed: {id}')
    #     ep+=1

    # lers=np.array(lers)
    # noises=np.array(noises)
    # print("\n===== FIXED-SEED EVAL SUMMARY =====")
    # print(f"LER mean={lers.mean():.5f}   std={lers.std():.5f}")
    # print(f"LER min={lers.min():.5f}     max={lers.max():.5f}")
    # print(f"noise mean={noises.mean():.5f}")
    # print("===================================\n")