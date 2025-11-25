import matplotlib.pyplot as plt
import numpy as np


def plot_LERvsSTEPS(ler_list):
    episodes = np.arange(1, len(ler_list)+1)
    plt.figure(figsize=(6,4))
    plt.plot(episodes, ler_list, marker='.', label="Logical error rate")
    plt.xlabel("Episode")
    plt.ylabel("Logical error rate")
    plt.title("Logical Error Rate vs Training")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/LogicalErrorRate.png')

def plot_step_reward_trends(pos_counts, neg_counts):
    """
    pos_counts: list of mean positive rewards per round (one per episode)
    neg_counts: list of mean negative rewards per round (one per episode)
    """
    episodes = np.arange(1, len(pos_counts)+1)
    plt.figure(figsize=(6,4))
    plt.plot(episodes, pos_counts, marker='.', label="Mean positives per round")
    plt.plot(episodes, neg_counts, marker='.', label="Mean negatives per round")
    plt.xlabel("Episode")
    plt.ylabel("Mean count per round")
    plt.title("Step Reward Trends")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/RewardTrends.png')



def plot_coef(coefs):
    episodes = np.arange(1, len(coefs)+1)
    plt.figure(figsize=(6,4))

    plt.plot(episodes, coefs, marker='.', label="Entropy coeffs")
    plt.xlabel("Episode")
    plt.ylabel("Coeff")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/entropy_coeff.png')



def plot_ev_kl_entropy(ev,kl,entropy):
    assert len(ev)==len(kl)==len(entropy), "length of stats is not the same (ev,kl,h)"
    episodes = np.arange(1, len(ev)+1)
    plt.figure(figsize=(6,4))
    plt.plot(episodes, kl, marker='.', label="KL divergence")
    plt.plot(episodes, ev, marker='.', label="Explained variance")
    plt.plot(episodes, entropy, marker='.', label="Entropy")

    plt.xlabel("Episode")
    plt.ylabel("Stats")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/KL_EV_H.png')


def plot_loss_v_pi(v,pi):
    episodes = np.arange(1, len(v)+1)
    plt.figure(figsize=(6,4))
    plt.plot(episodes, v, marker='.', label="Value Loss")
    plt.plot(episodes, pi, marker='.', label="Policy Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/Losses_PPO.png')



def plot_LER_rstep_finalr(ler, r_step, finals):
    assert len(ler)==len(r_step)==len(finals), "length of stats is not the same (ev,kl,h)"
    episodes = np.arange(1, len(ler)+1)
    plt.figure(figsize=(6,4))
    plt.plot(episodes, ler, marker='.', label="LER")
    plt.plot(episodes, r_step, marker='.', label="Mean step rewards")
    plt.plot(episodes, finals, marker='.', label="Final rewards")
    plt.xlabel("Episode")
    plt.ylabel("Stats")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/L_RS_RF.png')



def plot_effectiveness(eff, no_improve, act, idle):
    assert len(eff)==len(no_improve)==len(act)==len(idle),  "length of stats is not the same (ev,kl,h)"
    episodes = np.arange(1, len(eff)+1)

    plt.figure(figsize=(6,4))
    plt.plot(episodes, eff, marker='.', label="Effective Actions Rate (acted and cleared something)")
    plt.plot(episodes, no_improve, marker='.', label="No Improvement Rate (acted but didn't clear)")
    plt.plot(episodes, act, marker='.', label="Acting Rate (X and Z actions taken)")
    plt.plot(episodes, idle, marker='.', label="Idle Rate (acted when prev round was clean)")
    plt.xlabel("Episode")
    plt.ylabel("Stats")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/effects.png')


