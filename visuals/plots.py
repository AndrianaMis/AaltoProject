import matplotlib.pyplot as plt
import numpy as np


def plot_LERvsSTEPS(ler_list):
    episodes = np.arange(1, len(ler_list)+1)
    plt.figure(figsize=(6,4))
    plt.plot(episodes, ler_list, marker='o', label="Logical error rate")
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
    plt.plot(episodes, pos_counts, marker='o', label="Mean positives per round")
    plt.plot(episodes, neg_counts, marker='s', label="Mean negatives per round")
    plt.xlabel("Episode")
    plt.ylabel("Mean count per round")
    plt.title("Step Reward Trends")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/RewardTrends.png')






def plot_ev_kl_entropy(ev,kl,entropy):
    assert len(ev)==len(kl)==len(entropy), "length of stats is not the same (ev,kl,h)"
    episodes = np.arange(1, len(ev)+1)
    plt.figure(figsize=(6,4))
    plt.plot(episodes, kl, marker='P', label="KL divergence")
    plt.plot(episodes, ev, marker='X', label="Explained variance")
    plt.plot(episodes, entropy, marker='H', label="Entropy")
    plt.xlabel("Episode")
    plt.ylabel("Stats")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/KL_EV_H.png')


def plot_loss_v_pi(v,pi):
    episodes = np.arange(1, len(v)+1)
    plt.figure(figsize=(6,4))
    plt.plot(episodes, v, marker='o', label="Value Loss")
    plt.plot(episodes, pi, marker='s', label="Policy Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig('visuals/graphs/Losses_PPO.png')



