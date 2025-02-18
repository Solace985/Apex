// NEW FILE: Strategy router for low-latency asset-specific strategies  
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
        let correlated = self.asset_correlation_graph  
            .get_strongly_connected(asset, 0.8);  

        match asset {  
            "NIFTY50" => vec![  
                StrategyType::IndexArbitrage,  
                StrategyType::BetaHedging  
            ],  
            "BTC" => vec![  
                StrategyType::MevProtection,  
                StrategyType::OnChainFlow  
            ],  
            _ => self.default_strategies(correlated)  
        }  
    }  
}  