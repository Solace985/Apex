use crate::stats::{detect_volatility_clusters, detect_anomaly};
use crate::market_data::fetch_volatility;

pub struct StrategyRouter {  
    asset_profiles: HashMap<String, AssetProfile>,  
    asset_correlation_graph: Arc<CorrelationGraph>,  
}  

impl StrategyRouter {  
    pub fn new() -> Self {  
        let mut profiles = HashMap::new();  
        profiles.insert("BTC".into(), AssetProfile::crypto());  
        profiles.insert("SPY".into(), AssetProfile::equity());  
        Self { 
            asset_profiles: profiles, 
            asset_correlation_graph: Arc::new(CorrelationGraph::new()), // Assuming a new method to initialize CorrelationGraph
        }  
    }  

    pub fn select_strategy(&self, symbol: &str) -> Box<dyn Strategy> {  
        match self.asset_profiles.get(symbol) {  
            Some(AssetProfile::Crypto) => Box::new(CryptoHFT::new()),  
            Some(AssetProfile::Equity) => Box::new(VWAPArb::new()),  
            None => Box::new(RetailDefault::new()),  
        }  
    }  

    pub fn route(&self, asset: &str) -> Vec<StrategyType> {  
        let correlated_assets = self.asset_correlation_graph  
            .get_strongly_connected(asset, 0.8);

        let volatility = fetch_volatility(asset);

        // **NEW FEATURE**: If volatility is high, switch strategies to risk-adjusted trading
        if volatility > 0.06 {
            log::info!("⚠️ High volatility detected for {} → Adapting strategy", asset);
            return vec![
                StrategyType::RiskParity,
                StrategyType::VolatilityArb,
            ];
        }

        // **NEW FEATURE**: If strong correlation anomalies exist, hedge against index movements
        for correlated in &correlated_assets {
            if detect_anomaly(asset, correlated, 0.75).await {
                log::info!("🔍 Correlation anomaly detected → Adjusting strategy for {}", asset);
                return vec![StrategyType::HedgedMomentum];
            }
        }

        match asset {  
            "NIFTY50" => vec![  
                StrategyType::IndexArbitrage,  
                StrategyType::BetaHedging  
            ],  
            "BTC" => vec![  
                StrategyType::MevProtection,  
                StrategyType::OnChainFlow  
            ],  
            _ => self.default_strategies(correlated_assets)  
        }  
    }  
}  