"""
play_dqn.py — run the trained DQN in TEST mode (greedy, no exploration).

How to run:
    python play_dqn.py

Loads `dqn_maze.pth` and opens a Pygame window showing 10 greedy rollouts,
then prints the success rate. This is the script to run to demo the final
solution in the exam.
"""
import numpy as np
import torch
from env import ContinuousMazeEnv
from dqn_agent import DQNAgent

agent = DQNAgent()
agent.load("dqn_maze.pth")

env = ContinuousMazeEnv(render_mode="human")
n_runs, successes = 10, 0

for run in range(n_runs):
    state, _ = env.reset()
    total, steps = 0.0, 0
    for _ in range(300):
        env.render()
        action = agent.act(state, epsilon=0.0)     # pure greedy
        state, r, done, trunc, _ = env.step(action)
        total += r; steps += 1
        if done or trunc:
            break
    ok = total > 5
    successes += ok
    print(f"run {run+1:2d}: steps={steps:3d}, reward={total:+.1f}, {'GOAL' if ok else 'fail'}")

print(f"\nSuccess rate: {successes}/{n_runs}")
env.close()