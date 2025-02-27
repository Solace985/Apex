use serde::Deserialize;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use std::collections::HashSet;
use anyhow::{Result, Context};
use std::env;
use serde_yaml::Value;

/// Securely loads a YAML configuration file
pub fn load_yaml_config(file_path: &str) -> Result<HashSet<String>, String> {
    let mut file = File::open(file_path).map_err(|_| format!("Failed to open {}", file_path))?;
    let mut content = String::new();
    file.read_to_string(&mut content).map_err(|_| "Failed to read file content".to_string())?;

    let yaml: Value = serde_yaml::from_str(&content).map_err(|_| "Invalid YAML format".to_string())?;

    if let Some(assets) = yaml["assets"].as_sequence() {
        let asset_set: HashSet<String> = assets.iter().filter_map(|val| val.as_str().map(String::from)).collect();
        Ok(asset_set)
    } else {
        Err("Malformed asset universe".to_string())
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub exchanges: HashMap<String, ExchangeConfig>,
    pub risk_limits: RiskConfig,
    pub asset_universe: AssetUniverse,
    pub ai_service: AIServiceConfig, // Added AIServiceConfig
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExchangeConfig {
    pub ws_url: String,
    pub api_key_env_var: String,
    pub secret_env_var: String,
}

#[derive(Debug, Deserialize, Clone)] 
pub struct RiskConfig {
    pub max_drawdown: f64,
    pub daily_loss_limit: f64,
    pub asset_risk_weights: HashMap<String, f64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AssetUniverse {
    pub equities: Vec<String>,
    pub forex: Vec<String>,
    pub commodities: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)] // Added AIServiceConfig struct
pub struct AIServiceConfig {
    pub prediction_endpoint: String,
    pub batch_endpoint: String,
    pub api_secret: String,
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = env::var("CONFIG_PATH")
            .unwrap_or_else(|_| "Retail/config/retail.yaml".to_string());
        
        let config_str = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config at {}", config_path))?;
        
        let mut config: Config = serde_yaml::from_str(&config_str)
            .with_context(|| "Malformed YAML config")?;

        // Load secrets from environment
        for (exchange, cfg) in &mut config.exchanges {
            cfg.api_key = env::var(&cfg.api_key_env_var)
                .with_context(|| format!("Missing {} API key", exchange))?;
            cfg.secret = env::var(&cfg.secret_env_var)
                .with_context(|| format!("Missing {} secret", exchange))?;
        }

        Ok(config)
    }
}