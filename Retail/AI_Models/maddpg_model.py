import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class Actor(nn.Module):
    """Actor network for MADDPG."""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic network for MADDPG."""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradients."""
    def __init__(self, state_dim, action_dim, lr=0.001, tau=0.01):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.tau = tau

        # Initialize target networks
        self.update_target_networks(tau=1.0)

    def update_target_networks(self, tau=None):
        """Soft update target networks."""
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences for training."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        """Trains MADDPG using replay memory."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))

        # Critic update
        next_actions = self.target_actor(next_states)
        target_q_values = self.target_critic(next_states, next_actions).detach()
        y = rewards + 0.99 * target_q_values
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks()

    def select_action(self, state, noise_scale=0.1):
        """Selects an action given the current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        noise = noise_scale * np.random.randn(len(action))
        return np.clip(action + noise, -1, 1) 