pub fn calculate_position_size(&self, symbol: &str, volatility: f64) -> f64 {
    let params = self.asset_params.get(symbol)
        .unwrap_or_else(|| self.default_asset_params());

    let raw_size = (self.capital * params.risk_pct) / (volatility * params.leverage);

    // 🔹 Enforce minimum position size
    if raw_size < params.min_trade_size {
        return params.min_trade_size;
    }

    // 🔹 Enforce maximum position size
    if raw_size > params.max_trade_size {
        return params.max_trade_size;
    }

    raw_size
}