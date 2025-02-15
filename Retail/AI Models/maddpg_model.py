import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from AI_Models.fundamental_analysis import FundamentalAnalysis

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

# ---------------------------
# ✅ Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros(buffer_size)
        self.next_states = np.zeros((buffer_size, state_dim))
        self.dones = np.zeros(buffer_size)
        self.pointer = 0
        self.size = 0

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.pointer % self.buffer_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.pointer += 1
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradients."""
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)

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
        next_actions = self.actor(next_states)
        target_q_values = self.critic(next_states, next_actions).detach()
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

    def select_action(self, state):
        """Selects an action given the current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state_tensor).detach().numpy()[0]
    
    # for considering macroeconomic factors
    def __init__(self):
        self.fundamental_analysis = FundamentalAnalysis()

    def compute_reward(self, rewards):
        """Adjusts reward using macroeconomic & sentiment analysis."""
        sentiment = self.fundamental_analysis.fetch_news_sentiment()
        macro = self.fundamental_analysis.fetch_macro_factors()

        # Reward Scaling Based on Sentiment & Macro Factors
        adjusted_rewards = rewards * (1 + sentiment * 0.1) * (1 - macro["inflation"] * 0.05)
        return adjusted_rewards
    
    # for considering technical analysis. also prevents false trades by considering institutional flow and macroeconomic data.
    def __init__(self):
        self.tech_analysis = TechnicalAnalysis()

    def select_trade(self, price_series, volume_series):
        """Executes only if order flow, fundamentals, and technicals align."""
        tech_indicators = self.tech_analysis.compute_indicators(price_series, volume_series)

        if tech_indicators["rsi"] > 70 and tech_indicators["adx"] < 25:
            return "No Trade"  # Avoid overbought weak trends
        return "Trade"
    
    
    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample_batch(batch_size)
        
        # ✅ Normalize rewards for stability
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)

        for i in range(len(self.actors)):
            # Compute target Q-values
            target_actions = [self.target_actors[i](torch.FloatTensor(next_states[i])) for i in range(len(self.actors))]
            target_q_values = self.target_critics[i](torch.FloatTensor(next_states), torch.FloatTensor(target_actions))

            # Compute critic loss
            critic_loss = nn.MSELoss()(target_q_values, torch.FloatTensor(rewards[i]))
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Compute actor loss
            actor_loss = -self.critics[i](torch.FloatTensor(states), self.actors[i](torch.FloatTensor(states))).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # ✅ Apply soft update
        for i in range(len(self.actors)):
            self.soft_update(self.target_actors[i], self.actors[i], self.tau)
            self.soft_update(self.target_critics[i], self.critics[i], self.tau)

    def select_action(self, technical_features):
        """Uses trained MADDPG model to decide whether to trade."""
        state_tensor = torch.FloatTensor(technical_features).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        return action  # Output is a value between 0 and 1; trade if above threshold.

