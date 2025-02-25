import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import schedule
from Apex.src.ai.analysis.fundamental_analysis import FundamentalAnalysis
from Apex.src.ai.forecasting.technical_analysis import TechnicalAnalysis  # Assuming TechnicalAnalysis is defined in this module
from Apex.src.Core.trading.strategies.mean_reversion import MeanReversionStrategy
from Apex.src.Core.trading.strategies.trend.momentum_breakout import MomentumBreakoutStrategy
from Apex.src.Core.trading.strategies.trend.trend_following import TrendFollowingStrategy
from Apex.src.Core.trading.strategies.regime_detection import RegimeDetectionStrategy
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """Actor network for MADDPG."""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
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
    """Multi-Agent Deep Deterministic Policy Gradients (MADDPG) for AI-driven trading."""
    def __init__(self, state_dim=10, action_dim=1, model_path="Retail/Models/maddpg_trained.pth"):
        """
        Initializes the MADDPG model.
        - Loads pre-trained model if available.
        - Includes multiple trading strategies.
        - Uses multiple technical indicators for decision-making.
        """
        self.model_path = model_path
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ✅ Initialize Actor-Critic Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)  # Target actor
        self.target_critic = Critic(state_dim, action_dim).to(device)  # Target critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        # ✅ Initialize Technical & Fundamental Analysis
        self.fundamental_analysis = FundamentalAnalysis()
        self.tech_analysis = TechnicalAnalysis()

        # ✅ Initialize Strategies
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum_breakout": MomentumBreakoutStrategy(),
            "trend_following": TrendFollowingStrategy(),
            "regime_detection": RegimeDetectionStrategy()
        }

        # ✅ Initialize Memory Buffer
        self.memory = deque(maxlen=10000)

        # ✅ Load pre-trained MADDPG model
        self.load_model()

        # Schedule daily retraining
        schedule.every().day.at("00:00").do(self.retrain_maddpg)

    def load_model(self):
        """Loads a pre-trained MADDPG model."""
        try:
            self.actor.load_state_dict(torch.load(self.model_path.replace(".pth", "_actor.pth"), map_location=device))
            self.critic.load_state_dict(torch.load(self.model_path.replace(".pth", "_critic.pth"), map_location=device))
            logger.info("✅ Loaded trained MADDPG models.")
        except Exception as e:
            logger.warning(f"🚀 Model loading issue: {e}")

    def retrain_maddpg(self):
        logger.info("🔄 Retraining MADDPG Model with Latest Market Data...")
        self.update(self.memory, batch_size=64)

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences for training."""
        self.memory.append((state, action, reward, next_state, done))

    def soft_update(self, target, source, tau=0.005):
        """Soft update for target networks."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, replay_buffer, batch_size=64, patience=10):
        """
        ✅ Trains the MADDPG model using experience replay.
        - Uses soft updates to stabilize learning.
        - Implements early stopping based on loss.
        """
        min_loss = np.inf
        patience_counter = 0

        for epoch in range(100):  # example epoch range
            if len(replay_buffer.memory) < batch_size:
                continue

            states, actions, rewards, next_states, dones = replay_buffer.sample_batch(batch_size)

            # Move tensors to device
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)

            # Critic update
            next_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, next_actions).detach()
            y = rewards + 0.99 * target_q_values
            critic_loss = nn.MSELoss()(self.critic(states, actions), y)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor update
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Updates
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)

            # Early Stopping Logic
            total_loss = actor_loss.item() + critic_loss.item()
            if total_loss < min_loss:
                min_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"⚠️ Early stopping triggered after {epoch+1} epochs with loss {total_loss:.4f}")
                    break

            # Periodic Model Saving
            if epoch % 10 == 0 or epoch == 99:  # 99 because range is 0-99
                self.save_model(best_model=False)

    def update_model(self, state, action, reward, next_state, done, batch_size=64):
        """
        ✅ Live Training Feature:
        - Stores the trade experience in memory.
        - Trains the model immediately after every trade.
        """
        self.remember(state, action, reward, next_state, done)

        if len(self.memory) < batch_size:
            return  # ✅ Not enough experiences yet to train

        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, batch_size))

        # Move tensors to device
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        # ✅ Update Critic
        next_actions = self.actor(next_states)
        target_q_values = self.critic(next_states, next_actions).detach()
        y = rewards + 0.99 * target_q_values
        critic_loss = nn.MSELoss()(self.critic(states, actions), y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ✅ Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ✅ Soft Update
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        # Periodic Model Saving
        self.save_model(best_model=False)  # Save updated model
        print("📈 RL Model Updated with Live Trading Data.")

    def save_model(self, best_model=True):
        actor_path = self.model_path.replace(".pth", "_actor_best.pth" if best_model else "_actor.pth")
        critic_path = self.model_path.replace(".pth", "_critic_best.pth" if best_model else "_critic.pth")

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        logger.info(f"💾 Models saved successfully at {actor_path} and {critic_path}")

    def select_action(self, state):
        """Selects an action given the current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.actor(state_tensor).cpu().detach().numpy()[0]

def compute_reward(self, predicted_price, true_price, action):
    """
    ✅ Optimized Reward System (Combined):
    - Rewards correct directional predictions (BUY/SELL).
    - Implements Stop Loss (-5%) & Take Profit (+5%) mechanisms.
    - Incorporates sentiment & macroeconomic adjustments.
    - Additional incentives for trend-following and penalties for incorrect trades.
    """

    price_change = true_price - predicted_price
    reward = 0

    # ✅ Reward/Penalty Based on Action
    if action == "BUY":
        reward = max(price_change, 0) * 2  # 2x reward for correct upward trades
        reward -= max(-price_change, 0) * 2  # 2x penalty for incorrect buy
    elif action == "SELL":
        reward = max(-price_change, 0) * 2  # 2x reward for correct downward trades
        reward -= max(price_change, 0) * 2  # 2x penalty for incorrect sell
    elif action == "HOLD":
        reward = -abs(price_change) * 0.1  # Small penalty for incorrect holding
    else:
        reward = -2  # Large penalty for invalid action

    # ✅ Stop Loss (-5%) & Take Profit (+5%)
    if abs(price_change) >= 0.05:  
        if price_change < 0:
            reward *= -2  # Heavy penalty if wrong beyond stop-loss threshold
        else:
            reward *= 1.5  # Boost reward for achieving take-profit target

    # ✅ Additional Reward for Strong Trend-Following
    if abs(price_change) > 0.5:
        reward *= 1.5

    # ✅ Adjust reward based on Sentiment & Macroeconomic Factors
    sentiment = self.fundamental_analysis.fetch_news_sentiment()
    macro = self.fundamental_analysis.fetch_macro_factors()
    
    # Determine sentiment weight based on volatility
    if indicators["atr"] > 1.5 * np.mean(indicators["atr"]):
        sentiment_weight = 0.3  # Increase sentiment impact
    else:
        sentiment_weight = 0.1  # Keep it low during normal markets

    # Robust reward adjustment:
    sentiment_factor = 1 + (sentiment * sentiment_weight if sentiment else 0)
    inflation_factor = 1 - (macro.get("inflation", 0) * 0.05)

    reward *= sentiment_factor * inflation_factor

    return reward

def select_action(self, market_data, true_price=None):
    """
    ✅ Uses multiple strategies + MADDPG RL model to decide whether to trade.
    - Uses **Regime Detection** to decide **which strategy** is optimal.
    """
    state_tensor = torch.FloatTensor(market_data).unsqueeze(0).to(device)
    raw_action = self.actor(state_tensor).detach().numpy()[0]  # ✅ Initial action from RL model

    # ✅ Compute Technical Indicators
    try:
        indicators = {
            "rsi": self.tech_analysis.relative_strength_index(market_data["price"]),
            "bollinger_upper": self.tech_analysis.bollinger_bands(market_data["price"])[0],
            "bollinger_lower": self.tech_analysis.bollinger_bands(market_data["price"])[1],
            "stochastic_oscillator": self.tech_analysis.stochastic_oscillator(market_data["price"]),
            "atr": self.tech_analysis.average_true_range(
                market_data["high"], market_data["low"], market_data["close"]
            ),
            "aroon_up": self.tech_analysis.aroon_indicator(market_data["high"], market_data["low"])[0],
            "aroon_down": self.tech_analysis.aroon_indicator(market_data["high"], market_data["low"])[1],
        }
    except Exception as e:
        logger.warning(f"Indicator calculation failed: {e}")
        return 0  # Default HOLD if indicators fail

    # ✅ Determine Market Regime (Trending or Ranging)
    market_regime = self.strategies["regime_detection"].detect_regime(market_data)

    # ✅ Select Strategy Based on Market Regime
    if market_regime == "TRENDING":
        selected_strategy = self.strategies["trend_following"]
    else:
        selected_strategy = self.strategies["mean_reversion"]

    # ✅ Get Strategy Signal
    strategy_decision = selected_strategy.compute_signal(market_data)

    # ✅ Adjust MADDPG action based on strategy output
    if strategy_decision == "BUY" and raw_action < 0:
        adjusted_action = 0  # BLOCK SELL
    elif strategy_decision == "SELL" and raw_action > 0:
        adjusted_action = 0  # BLOCK BUY
    else:
        if strategy_decision == "BUY":
            adjusted_action = max(raw_action, 0)  # If MADDPG was neutral, move to BUY
        elif strategy_decision == "SELL":
            adjusted_action = min(raw_action, 0)  # If MADDPG was neutral, move to SELL
        else:
            adjusted_action = 0  # HOLD

    # ✅ Apply Filters to Prevent Weak Trades
    if indicators["rsi"] > 70 and indicators["aroon_down"] < 25:
        adjusted_action = 0  # Avoid buying in weak overbought trends

    if market_data["price"][-1] > indicators["bollinger_upper"]:
        adjusted_action = min(adjusted_action, 0)  # Sell if price is above Bollinger upper band
    elif market_data["price"][-1] < indicators["bollinger_lower"]:
        adjusted_action = max(adjusted_action, 0)  # Buy if price is below Bollinger lower band

    # ✅ Calculate ATR-based Stop Loss and Take Profit
    atr = indicators["atr"]
    stop_loss = market_data["price"][-1] - (atr * 2)  # Stop loss at 2x ATR
    take_profit = market_data["price"][-1] + (atr * 3)  # Take profit at 3x ATR

    # ✅ Exit conditions based on Stop Loss and Take Profit
    if adjusted_action == "BUY" and market_data["price"][-1] > take_profit:
        adjusted_action = 0  # EXIT trade if profit target hit
    elif adjusted_action == "SELL" and market_data["price"][-1] < stop_loss:
        adjusted_action = 0  # EXIT trade if stop-loss hit

    # ✅ Adjust action based on **sentiment & macro factors**
    sentiment = self.fundamental_analysis.fetch_news_sentiment()
    macro = self.fundamental_analysis.fetch_macro_factors()
    adjusted_action *= (1 + sentiment * sentiment_weight) * (1 - macro["inflation"] * 0.05)

    final_action = np.clip(adjusted_action, -1, 1)

    # ✅ If true price is provided, update MADDPG model
    if true_price is not None:
        reward = self.compute_reward(market_data["price"], true_price, final_action)
        self.update_model(market_data, final_action, reward, market_data, False)

    return final_action
