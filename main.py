"""
main.py — train the DQN agent on the continuous maze (Project B).

How to run:
    python main.py

Trains for 800 episodes headless (no window), saves the learned weights to
`dqn_maze.pth`, and writes the reward curve to `training_curve.png`.
Depends on `env.py` (ContinuousMazeEnv) and `dqn_agent.py` (DQNAgent).
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

from env import ContinuousMazeEnv
from dqn_agent import DQNAgent

# ---------------- Hyperparameters ----------------
EPISODES        = 800
MAX_STEPS       = 300        # our own cap: the env has none (agent could wander forever)
GAMMA           = 0.99
LR              = 1e-3
HIDDEN          = 128
BATCH_SIZE      = 64
BUFFER_CAP      = 50_000
TARGET_SYNC     = 500        # learn-steps between target-network syncs
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.995      # per-episode multiplicative decay
SEED            = 42

# ---------------- Setup ----------------
np.random.seed(SEED); torch.manual_seed(SEED)
env = ContinuousMazeEnv(render_mode=None)        # headless during training!
agent = DQNAgent(hidden=HIDDEN, lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
                 buffer_capacity=BUFFER_CAP, target_sync_every=TARGET_SYNC)

epsilon = EPS_START
episode_rewards, episode_lengths, losses = [], [], []
goals_reached = 0

# ---------------- Training loop ----------------
for ep in range(1, EPISODES + 1):
    state, _ = env.reset()
    total_reward, steps = 0.0, 0

    for _ in range(MAX_STEPS):
        action = agent.act(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)   # 5-tuple!

        agent.buffer.push(state, action, reward, next_state, float(done))
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward
        steps += 1
        if done or truncated:
            break

    if total_reward > 5:                 # reached the goal (+10 dominates)
        goals_reached += 1

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

    if ep % 20 == 0:
        recent = episode_rewards[-20:]
        print(f"ep {ep:4d} | avg reward (20) {np.mean(recent):8.2f} | "
              f"goals so far {goals_reached:3d} | eps {epsilon:.3f} | "
              f"buffer {len(agent.buffer)}")

# ---------------- Save + training curve (deliverable 3.1.2) ----------------
agent.save("dqn_maze.pth")
print("Model saved to dqn_maze.pth")

window = 20
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
plt.figure(figsize=(11, 5))
plt.plot(episode_rewards, alpha=0.25, label="per episode")
plt.plot(range(window-1, len(episode_rewards)), smoothed, label=f"{window}-ep average")
plt.xlabel("Episode"); plt.ylabel("Total reward")
plt.title("DQN on Continuous Maze — training curve")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("training_curve.png", dpi=150)
plt.show()