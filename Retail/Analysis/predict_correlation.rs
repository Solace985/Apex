// predict_correlation.rs
use reqwest::Client;
use serde_json::json;
use tokio::time::Duration;
use log::{error, info};

pub async fn predict_correlation(asset1: &str, asset2: &str) -> f64 {
    let client = Client::new();
    let url = "http://localhost:5000/predict_correlation";

    let history = fetch_historical_data(asset1, asset2).await.unwrap_or(vec![0.0; 30]);

    let payload = json!({
        "correlation_series": history
    });

    match client.post(url)
        .json(&payload)
        .timeout(Duration::from_secs(3))
        .send()
        .await {
            Ok(response) => {
                if let Ok(json) = response.json::<serde_json::Value>().await {
                    if let Some(pred) = json["predicted_correlation"].as_f64() {
                        info!("🔮 AI Predicted Correlation: {}-{} -> {:.4}", asset1, asset2, pred);
                        return pred;
                    }
                }
                error!("❌ Failed to parse AI response");
                0.0
            },
            Err(e) => {
                error!("❌ AI Prediction API Error: {}", e);
                0.0
            }
        }
}
