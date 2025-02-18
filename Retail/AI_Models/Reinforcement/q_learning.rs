use std::collections::HashMap;
use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};

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
            
        let new_q = current_q + self.learning_rate * 
            (reward + self.discount_factor * max_next_q - current_q);
            
        if let Some(q_values) = self.q_table.get_mut(state) {
            q_values[action_idx] = new_q;
        }
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay)
            .max(self.min_epsilon);
    }

    pub fn calculate_reward(
        &self,
        portfolio_return: f64,
        volatility: f64,
        max_drawdown: f64,
        sharpe_ratio: f64
    ) -> f64 {
        let risk_penalty = volatility * 0.5 + max_drawdown * 0.3;
        (sharpe_ratio * 0.6) + (portfolio_return * 0.4) - risk_penalty
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

    pub fn save_policy(&self, path: &str) -> Result<(), std::io::Error> {
        let serialized = serde_json::to_string(&self)?;
        std::fs::write(path, serialized)
    }

    pub fn load_policy(path: &str) -> Result<Self, std::io::Error> {
        let data = std::fs::read_to_string(path)?;
        let agent: QLearningAgent = serde_json::from_str(&data)?;
        Ok(agent)
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
    fn test_epsilon_greedy() {
        let states = vec![test_state()];
        let actions = vec![Action::Buy, Action::Sell, Action::Hold];
        let mut agent = QLearningAgent::new(
            states,
            actions,
            0.1,
            0.9,
            1.0,
            0.995
        );
        
        let action = agent.choose_action(&test_state());
        assert!(matches!(action, Action::Buy | Action::Sell | Action::Hold));
    }
}