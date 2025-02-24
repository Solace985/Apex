use reqwest::{Client, Error, RequestBuilder};
use serde_json::json;
use tokio::time::timeout;
use log::{error, info};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::market_data::fetch_historical_correlation_series;
use hmac_sha256::hmac_sha256; // Ensure to include the hmac_sha256 crate

fn sign_request(request: RequestBuilder, key: &str) -> RequestBuilder {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    
    let signature = hmac_sha256(key, format!("{}{}", nonce, request));
    
    request
        .header("X-Auth-Nonce", nonce)
        .header("X-Auth-Signature", signature)
}

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
    match timeout(Duration::from_secs(3), sign_request(client.post(url), "your_secret_key").json(&payload).send()).await {
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

/// ✅ **Batch Predicts AI-Based Correlation Between Multiple Asset Pairs**
pub async fn batch_predict_correlations(pairs: &[(String, String)]) -> HashMap<(String, String), f64> {
    let client = Client::new();
    let url = "http://localhost:5000/batch_predict";
    
    let payload = json!({
        "pairs": pairs,
        "historical_window": "720h"
    });
    
    match client.post(url)
        .json(&payload)
        .send()
        .await {
            Ok(response) => {
                match response.json::<HashMap<(String, String), f64>>().await {
                    Ok(predictions) => predictions,
                    Err(e) => {
                        error!("❌ Failed to parse response JSON: {}", e);
                        HashMap::new()
                    }
                }
            },
            Err(e) => {
                error!("❌ API Request Failed: {}", e);
                HashMap::new()
            }
        }
}
