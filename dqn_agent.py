"""
dqn_agent.py — the DQN agent for Project B.

This module is imported by main.py / play_dqn.py / record_dqn.py; it is not run
directly. It provides three classes:
    QNetwork     — MLP approximating Q(s, ·) (2 -> hidden -> hidden -> 4).
    ReplayBuffer — stores (s, a, r, s', done) transitions, samples minibatches.
    DQNAgent     — act() (epsilon-greedy), train_step() (one gradient update),
                   save(path) / load(path) for the .pth weights.

Example:
    from dqn_agent import DQNAgent
    agent = DQNAgent()
    agent.load("dqn_maze.pth")
    action = agent.act(state, epsilon=0.0)   # greedy / test mode
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------- Part 1: the Q-Network ----------
class QNetwork(nn.Module):
    """
    Approximates Q(s, ·): input (x, y) -> output [Q_up, Q_down, Q_left, Q_right].
    The neural-network replacement for Project A's q_table.
    """
    def __init__(self, state_dim=2, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------- Part 2: the Replay Buffer ----------
class ReplayBuffer:
    """
    Stores transitions (s, a, r, s', done) and serves random minibatches.
    """
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states,      dtype=np.float32),
                np.array(actions,     dtype=np.int64),
                np.array(rewards,     dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones,       dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
    
# ---------- Part 3: the Agent ----------
class DQNAgent:
    def __init__(self, state_dim=2, action_dim=4, hidden=128,
                 lr=1e-3, gamma=0.99, batch_size=64,
                 buffer_capacity=50_000, target_sync_every=500,
                 device="cpu"):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.q_net = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(buffer_capacity)
        self.target_sync_every = target_sync_every
        self.learn_steps = 0

    def act(self, state, epsilon):
        """epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            q_values = self.q_net(s)
            return int(q_values.argmax(dim=1).item())

    def train_step(self):
        """One gradient update from one random minibatch. Returns loss or None."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        states      = torch.as_tensor(states,      device=self.device)
        actions     = torch.as_tensor(actions,     device=self.device)
        rewards     = torch.as_tensor(rewards,     device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones       = torch.as_tensor(dones,       device=self.device)

        # Q(s, a) for the actions actually taken
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a')   (0 future if done)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # periodically sync the frozen target network
        self.learn_steps += 1
        if self.learn_steps % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def save(self, path):  torch.save(self.q_net.state_dict(), path)
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())