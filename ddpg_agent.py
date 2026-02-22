import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20        # learn every N timesteps
NUM_UPDATES = 10        # number of gradient updates per learning event

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """DDPG Agent that interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process — one per agent
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay memory (shared across all agents)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Timestep counter for learning schedule
        self.timestep = 0

    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experiences from all agents and learn if it's time."""
        # Store each agent's experience individually
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i],
                            next_states[i], dones[i])

        # Learn every LEARN_EVERY timesteps, if enough samples
        self.timestep = timestep
        if timestep % LEARN_EVERY == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for all agents given their states."""
        states = torch.from_numpy(states).float().to(device)
        # Switch to inference mode for deterministic forward pass
        self.actor_local.train(False)
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train(True)
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update actor and critic using a batch of experience tuples.

        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        """
        states, actions, rewards, next_states, dones = experiences

        # --- Update critic --- #
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Update actor --- #
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update target networks --- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update: theta_target = tau*theta_local + (1 - tau)*theta_target"""
        for target_param, local_param in zip(target_model.parameters(),
                                              local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + \
            self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                   "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None])
            .astype(np.uint8)
        ).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
