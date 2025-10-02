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