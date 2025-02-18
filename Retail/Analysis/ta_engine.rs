// Asset-specific indicator optimization
pub struct TAEngine {
    asset_params: HashMap<AssetClass, TAParams>,
}

impl TAEngine {
    pub fn calculate(&self, data: &Data, asset: AssetClass) -> HashMap<String, f64> {
        let params = self.asset_params.get(&asset).unwrap();
        
        let mut indicators = HashMap::new();
        
        // Volatility-adjusted parameters
        indicators.insert("RSI", rsi(data.close, params.rsi_window));
        indicators.insert("MACD", macd(data, params.macd_fast, params.macd_slow));
        
        // Liquidity-sensitive indicators
        if params.needs_liquidity {
            indicators.insert("VWAP", vwap(data, params.vwap_interval));
        }
        
        indicators
    }
}

// Asset-specific parameters
pub struct TAParams {
    rsi_window: u32,        // 14 for stocks, 9 for crypto
    macd_fast: u32,         // 12 for forex, 8 for commodities
    needs_liquidity: bool,   // True for large caps
}