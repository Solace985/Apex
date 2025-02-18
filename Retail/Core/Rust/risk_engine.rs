// Replace Python's slow pandas-based calculation  
pub fn calculate_volatility(prices: &[f64]) -> f64 {  
    let mean = prices.iter().sum::<f64>() / prices.len() as f64;  
    let variance = prices.iter()  
        .map(|x| (x - mean).powi(2))  
        .sum::<f64>() / prices.len() as f64;  
    variance.sqrt()  
}  