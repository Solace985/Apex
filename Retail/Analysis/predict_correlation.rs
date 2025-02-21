// predict_correlation.rs
use reqwest::{Client, Error};
use serde_json::json;
use tokio::time::timeout;
use log::{error, info};
use crate::market_data::fetch_historical_correlation_series;

/// ✅ **Predicts AI-Based Correlation Between Two Assets**
pub async fn predict_correlation(asset1: &str, asset2: &str) -> f64 {
    let client = Client::new();
    let url = "http://localhost:5000/predict_correlation";

    // ✅ **Fetch Historical Correlation Data (Ensures Data Exists)**
    let history = match fetch_historical_correlation_series(asset1, asset2).await {
        Ok(data) if !data.is_empty() => data,
        _ => {
            error!("❌ No historical correlation data available for {}-{}", asset1, asset2);
            return 0.0;
        }
    };

    let payload = json!({
        "correlation_series": history
    });

    // ✅ **Enforce API Timeout of 3 Seconds**
    match timeout(Duration::from_secs(3), client.post(url).json(&payload).send()).await {
        Ok(Ok(response)) => {
            match response.json::<serde_json::Value>().await {
                Ok(json) if json.get("predicted_correlation").is_some() => {
                    let pred = json["predicted_correlation"].as_f64().unwrap_or(0.0);
                    info!("🔮 AI Predicted Correlation: {}-{} -> {:.4}", asset1, asset2, pred);
                    pred
                },
                _ => {
                    error!("❌ Invalid AI Response Format: {:?}", response);
                    0.0
                }
            }
        }
        Ok(Err(e)) => {
            error!("❌ API Request Failed: {}", e);
            0.0
        }
        Err(_) => {
            error!("⏳ API Request Timed Out (3s)");
            0.0
        }
    }
}
