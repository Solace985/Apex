// config.rs
#[derive(Deserialize, Clone)]
pub struct CorrelationConfig {
    #[serde(default = "default_thresholds")]
    pub thresholds: ThresholdConfig,
    
    #[serde(default = "default_ai")]
    pub ai_service: AiServiceConfig,
}

#[derive(Deserialize, Clone)]
pub struct ThresholdConfig {
    pub min_liquidity: f64,
    pub correlation_strength: f64,
    pub confidence_cutoff: f64,
}

fn default_thresholds() -> ThresholdConfig {
    ThresholdConfig {
        min_liquidity: 500_000.0,
        correlation_strength: 0.7,
        confidence_cutoff: 0.65,
    }
}