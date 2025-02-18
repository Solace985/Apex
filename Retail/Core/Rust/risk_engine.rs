pub fn calculate_position_size(&self, symbol: &str, volatility: f64) -> f64 {
    let params = self.asset_params.get(symbol)
        .unwrap_or_else(|| self.default_asset_params());
    (self.capital * params.risk_pct) / (volatility * params.leverage)
}  