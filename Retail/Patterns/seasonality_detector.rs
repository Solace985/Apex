// Fractal pattern detection across timeframes
pub struct SeasonalityDetector {
    wavelet: WaveletTransform,
    lstm: LSTMForecaster,
}

impl SeasonalityDetector {
    pub fn detect(&self, data: &[f64]) -> Vec<Pattern> {
        // Wavelet analysis for multi-timeframe patterns
        let components = self.wavelet.decompose(data, 5);
        
        // LSTM for sequence prediction
        self.lstm.predict_patterns(components)
            .into_iter()
            .filter(|p| p.confidence > 0.8)
            .collect()
    }
}

// Example usage for natural gas
let ng_patterns = detector.detect(natural_gas_data);
if let Some(weekly_spike) = ng_patterns.iter().find(|p| p.name == "ThursdayVolatility") {
    adjust_position_size(weekly_spike.strength);
}