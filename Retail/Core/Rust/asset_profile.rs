pub enum AssetClass {  
    Crypto,  
    Equity,  
    Forex,  
    Commodity,  
}  

pub struct AssetProfile {  
    pub volatility_cutoff: f64,  
    pub allowed_strategies: Vec<String>,  
}  

impl AssetProfile {  
    pub fn crypto() -> Self {  
        AssetProfile {  
            volatility_cutoff: 0.08,  
            allowed_strategies: vec!["HFT".into(), "Sentiment".into()],  
        }  
    }  
}

// Dynamic asset classification using machine learning
#[derive(Serialize, Deserialize)]
pub struct AssetDNA {
    volatility_profile: f64,
    liquidity_score: f64,
    session_sensitivity: [f64; 7], // Weekly pattern
    cross_asset_corr: HashMap<String, f64>,
}

impl AssetProfiler {
    pub fn analyze(&self, data: &MarketData) -> AssetDNA {
        let mut dna = AssetDNA::default();
        
        // Volatility clustering
        dna.volatility_profile = self.garch_model.calculate(data);
        
        // Liquidity analysis
        dna.liquidity_score = self.obv_strategy.score(data);
        
        // Weekly pattern detection
        dna.session_sensitivity = self.fourier_transform.detect_cycles(data);
        
        // Cross-asset relationships
        dna.cross_asset_corr = self.graph_neural_net.predict_linkages(data);
        
        dna
    }
}