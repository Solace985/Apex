use std::collections::HashMap;
use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use anyhow::{Result, Context};
use log::{info, warn}; // Added logging imports
use rand::seq::SliceRandom;

pub struct ExperienceReplay {
    buffer: Vec<(State, Action, f64, State)>,
    capacity: usize,
}

impl ExperienceReplay {
    pub fn new(capacity: usize) -> Self {
        ExperienceReplay {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn add(&mut self, state: State, action: Action, reward: f64, next_state: State) {
        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push((state, action, reward, next_state));
    }

    pub fn sample(&self, batch_size: usize) -> Option<Vec<(State, Action, f64, State)>> {
        if self.buffer.len() >= batch_size {
            let mut rng = rand::thread_rng();
            Some(
                self.buffer
                    .choose_multiple(&mut rng, batch_size)
                    .cloned()
                    .collect()
            )
        } else {
            warn!("Not enough experiences in buffer for sampling!");
            None
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let serialized = serde_json::to_string(&self.buffer)
            .context("Failed to serialize ExperienceReplay buffer")?;
        std::fs::write(path, serialized)
            .context("Failed to write ExperienceReplay buffer to file")?;
        Ok(())
    }

    pub fn load(path: &str, capacity: usize) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .context("Failed to read ExperienceReplay buffer file")?;
        let buffer: Vec<(State, Action, f64, State)> = serde_json::from_str(&data)
            .context("Failed to deserialize ExperienceReplay buffer")?;
        Ok(ExperienceReplay { buffer, capacity })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLearningAgent {
    q_table: HashMap<State, Array1<f64>>,
    state_space: Vec<State>,
    action_space: Vec<Action>,
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64,
    epsilon_decay: f64,
    min_epsilon: f64,
    replay_buffer: ExperienceReplay,  // ✅ Include this line
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub asset_class: String,
    pub volatility_bucket: u8,
    pub trend_direction: i8,
    pub volume_status: u8,
    pub cross_asset_corr: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Action {
    Buy,
    Hold,
    Sell,
    Hedge,
}

impl QLearningAgent {
    pub fn new(
        state_space: Vec<State>,
        action_space: Vec<Action>,
        learning_rate: f64,
        discount_factor: f64,
        initial_epsilon: f64,
        epsilon_decay: f64
    ) -> Self {
        let mut q_table = HashMap::new();
        let between = Uniform::from(0.0..0.01);
        let mut rng = rand::thread_rng();
        
        for state in &state_space {
            q_table.insert(
                state.clone(),
                Array1::from_shape_fn(action_space.len(), |_| between.sample(&mut rng))
            );
        }

        QLearningAgent {
            q_table,
            state_space,
            action_space,
            learning_rate,
            discount_factor,
            epsilon: initial_epsilon,
            epsilon_decay,
            min_epsilon: 0.01,
            replay_buffer: ExperienceReplay::new(10000),
        }
    }

    pub fn choose_action(&self, state: &State) -> Action {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            let action_idx = rng.gen_range(0..self.action_space.len());
            self.action_space[action_idx]
        } else {
            self.greedy_action(state)
        }
    }

    fn greedy_action(&self, state: &State) -> Action {
        self.q_table.get(state)
            .map(|q_values| {
                let max_idx = q_values.argmax().unwrap();
                self.action_space[max_idx]
            })
            .unwrap_or(Action::Hold)
    }

    pub fn update_q_value(
        &mut self,
        state: &State,
        action: Action,
        reward: f64,
        next_state: &State,
    ) {
        let action_idx = self.action_space.iter()
            .position(|&a| a == action)
            .unwrap();
            
        let current_q = *self.q_table.get(state)
            .and_then(|q| q.get(action_idx))
            .unwrap_or(&0.0);
            
        let max_next_q = self.q_table.get(next_state)
            .map(|q| q.fold(0.0, |acc, &x| acc.max(x)))
            .unwrap_or(0.0);
        
        // Add controlled Gaussian noise to prevent overfitting
        let mut rng = rand::thread_rng();
        let noise: f64 = rng.sample(rand_distr::Normal::new(0.0, 0.01).unwrap());
        
        let new_q = current_q + self.learning_rate * 
            (reward + self.discount_factor * max_next_q - current_q) + noise;
            
        if let Some(q_values) = self.q_table.get_mut(state) {
            q_values[action_idx] = new_q.clamp(-10.0, 10.0); // Clamping to avoid extreme values
        }

        // Log the action taken
        self.log_action(state, action, reward); // Added logging of action
    }

    pub fn log_action(&self, state: &State, action: Action, reward: f64) {
        info!("Action: {:?} | State: {:?} | Reward: {}", action, state, reward);
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay)
            .max(self.min_epsilon);
    }

    pub fn decay_learning_rate(&mut self, decay_factor: f64, min_learning_rate: f64) {
        self.learning_rate = (self.learning_rate * decay_factor).max(min_learning_rate);
    }

    pub fn calculate_reward(
        &self,
        portfolio_return: f64,
        volatility: f64,
        max_drawdown: f64,
        sharpe_ratio: f64
    ) -> f64 {
        // Normalized inputs
        let norm_return = portfolio_return.tanh();
        let norm_volatility = (volatility / 10.0).min(1.0); // Normalize volatility to a scale of 0 to 1
        let norm_drawdown = (max_drawdown / 10.0).min(1.0); // Normalize max_drawdown to a scale of 0 to 1
        let norm_sharpe = sharpe_ratio.tanh();

        let risk_penalty = norm_volatility * 0.5 + norm_drawdown * 0.3;
        (norm_sharpe * 0.6) + (norm_return * 0.4) - risk_penalty
    }

    pub fn batch_update(
        &mut self,
        experiences: Vec<(&State, Action, f64, &State)>
    ) {
        for (state, action, reward, next_state) in experiences {
            self.update_q_value(state, action, reward, next_state);
        }
        self.decay_epsilon();
    }

    pub fn save_policy(&self, path: &str) -> Result<()> {
        let serialized = serde_json::to_string(&self)
            .context("Failed to serialize QLearningAgent")?;
        std::fs::write(path, serialized)
            .context("Failed to write QLearningAgent policy to file")?;
        Ok(())
    }

    pub fn load_policy(path: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .context("Failed to read QLearningAgent policy file")?;
        let agent: QLearningAgent = serde_json::from_str(&data)
            .context("Failed to deserialize QLearningAgent policy")?;
        Ok(agent)
    }

// Mock implementations for illustration
fn generate_state_space() -> Vec<State> {
    vec![
        State {
            asset_class: "crypto".to_string(),
            volatility_bucket: 1,
            trend_direction: 1,
            volume_status: 1,
            cross_asset_corr: 0.5,
        },
        State {
            asset_class: "stock".to_string(),
            volatility_bucket: 2,
            trend_direction: -1,
            volume_status: 0,
            cross_asset_corr: -0.3,
        },
    ]
}

struct MockEnv;

impl MockEnv {
    fn step(&self) -> (State, Action, f64, State) {
        let state = generate_state_space()[0].clone();
        let next_state = generate_state_space()[1].clone();
        (state, Action::Buy, 0.05, next_state)
    }
}

// Initialize agent
let state_space = generate_state_space();
let action_space = vec![Action::Buy, Action::Sell, Action::Hold];
let mut agent = QLearningAgent::new(
    state_space,
    action_space,
    0.1,   // learning_rate
    0.9,   // discount_factor
    1.0,   // initial_epsilon
    0.995  // epsilon_decay
);

// Training loop
let env = MockEnv;
for episode in 0..1000 {
    let (state, action, reward, next_state) = env.step();
    agent.update_q_value(&state, action, reward, &next_state);
    agent.decay_epsilon();
}
    
#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> State {
        State {
            asset_class: "crypto".to_string(),
            volatility_bucket: 2,
            trend_direction: 1,
            volume_status: 1,
            cross_asset_corr: 0.75,
        }
    }

    #[test]
    fn test_q_update_no_panic() {
        let states = vec![test_state()];
        let actions = vec![Action::Buy, Action::Sell, Action::Hold];
        let mut agent = QLearningAgent::new(states.clone(), actions, 0.1, 0.9, 1.0, 0.995);
        
        let result = std::panic::catch_unwind(|| {
            agent.update_q_value(
                &states[0],
                Action::Buy,
                0.1,
                &states[0]
            );
        });
        
        assert!(result.is_ok());
    }
}
        
        let action = agent.choose_action(&test_state());
        assert!(matches!(action, Action::Buy | Action::Sell | Action::Hold));
    }}
