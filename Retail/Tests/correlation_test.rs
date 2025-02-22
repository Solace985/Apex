// tests/correlation_test.rs
#[tokio::test]
async fn test_volatility_adaptive_calculation() {
    let matrix = CorrelationMatrix::new();
    
    let high_vol_pair = ("BTC".into(), "ETH".into());
    let low_vol_pair = ("USD".into(), "JPY".into());
    
    let high_result = matrix.calculate_correlation(high_vol_pair.0, high_vol_pair.1).await;
    let low_result = matrix.calculate_correlation(low_vol_pair.0, low_vol_pair.1).await;
    
    assert!(high_result.timeframe < low_result.timeframe);
    assert!(high_result.weight.ai < low_result.weight.ai);
}