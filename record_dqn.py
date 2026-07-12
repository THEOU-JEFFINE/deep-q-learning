"""
record_dqn.py — record one greedy rollout of the trained DQN to GIF + MP4.

How to run:
    python record_dqn.py

Loads `dqn_maze.pth`, plays one greedy episode, and saves it to
`dqn_solution.gif` and `dqn_solution.mp4` for the presentation slides.
"""
import numpy as np
import pygame
import imageio
import torch
from env import ContinuousMazeEnv
from dqn_agent import DQNAgent

agent = DQNAgent()
agent.load("dqn_maze.pth")

env = ContinuousMazeEnv(render_mode="human")
state, _ = env.reset()
frames = []

env.render()   # opens the window / draws first frame
frames.append(pygame.surfarray.array3d(env.screen).swapaxes(0, 1).copy())

total, steps = 0.0, 0
for _ in range(300):
    action = agent.act(state, epsilon=0.0)
    state, r, done, trunc, _ = env.step(action)
    env.render()
    frames.append(pygame.surfarray.array3d(env.screen).swapaxes(0, 1).copy())
    total += r; steps += 1
    if done or trunc:
        break

# hold the final frame for a second so the ending is visible
frames.extend([frames[-1]] * 10)

imageio.mimsave("dqn_solution.gif", frames, fps=10, loop=0)
imageio.mimwrite("dqn_solution.mp4", frames, fps=10)
print(f"Recorded {steps} steps, reward {total:+.1f} -> dqn_solution.gif / .mp4")
env.close()