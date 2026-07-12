# experiments.py — requirement 2: tune hyperparameters & architecture
import numpy as np
import matplotlib.pyplot as plt
import torch

from env import ContinuousMazeEnv
from dqn_agent import DQNAgent

EPISODES, MAX_STEPS, SEED = 600, 300, 42

def train_one(hidden=128, lr=1e-3, target_sync=500, eps_decay=0.995, label=""):
    np.random.seed(SEED); torch.manual_seed(SEED)
    import random as _r; _r.seed(SEED)
    env = ContinuousMazeEnv(render_mode=None)
    agent = DQNAgent(hidden=hidden, lr=lr, target_sync_every=target_sync)
    eps, rewards = 1.0, []
    for ep in range(EPISODES):
        state, _ = env.reset(); total = 0.0
        for _ in range(MAX_STEPS):
            a = agent.act(state, eps)
            ns, r, done, trunc, _ = env.step(a)
            agent.buffer.push(state, a, r, ns, float(done))
            agent.train_step()
            state = ns; total += r
            if done or trunc: break
        eps = max(0.05, eps * eps_decay)
        rewards.append(total)
    # final greedy evaluation: 10 rollouts
    succ = 0
    for _ in range(10):
        state, _ = env.reset(); total = 0.0
        for _ in range(MAX_STEPS):
            a = agent.act(state, 0.0)
            state, r, done, trunc, _ = env.step(a)
            total += r
            if done or trunc: break
        succ += total > 5
    print(f"{label:34s} -> greedy success {succ}/10")
    return rewards, succ

def sweep(name, configs):
    plt.figure(figsize=(11, 5))
    results = {}
    for label, kw in configs:
        rewards, succ = train_one(**kw, label=label)
        results[label] = succ
        w = 20
        sm = np.convolve(rewards, np.ones(w)/w, mode="valid")
        plt.plot(sm, label=f"{label}  [greedy {succ}/10]")
    plt.xlabel("Episode"); plt.ylabel("Avg reward (20-ep)")
    plt.title(f"DQN sweep: {name}")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"sweep_{name}.png", dpi=150)
    print(f"saved sweep_{name}.png\n")

if __name__ == "__main__":
    sweep("hidden_size", [
        ("hidden=32",  dict(hidden=32)),
        ("hidden=128 (baseline)", dict(hidden=128)),
        ("hidden=256", dict(hidden=256)),
    ])
    sweep("learning_rate", [
        ("lr=1e-4", dict(lr=1e-4)),
        ("lr=1e-3 (baseline)", dict(lr=1e-3)),
        ("lr=1e-2", dict(lr=1e-2)),
    ])
    sweep("target_sync", [
        ("sync=1 (no frozen target)", dict(target_sync=1)),
        ("sync=500 (baseline)", dict(target_sync=500)),
        ("sync=2000", dict(target_sync=2000)),
    ])