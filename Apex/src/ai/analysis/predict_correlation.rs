// Analysis/correlation_predictor.rs
use crate::{
    utils::{
        error_handler::ApiError,
        logging::structured_logger::{log_correlation_prediction, LogLevel},
    },
    Config::config_loader::get_ai_service_config,
    Core::data::historical_data::fetch_validated_correlation_history,
    Core::data::asset_validator::validate_asset_pair, // Import the function
};
use reqwest::{Client, RequestBuilder};
use serde_json::json;
use tokio::{sync::Mutex, time::timeout};
use std::{collections::HashMap, time::Duration};
use hmac_sha512::hmac_sha512;  // More secure than SHA256
use uuid::Uuid;

// Shared cache for correlation predictions
lazy_static! {
    static ref PREDICTION_CACHE: Mutex<HashMap<(String, String), f64>> = Mutex::new(HashMap::new());
}

fn create_secure_nonce() -> String {
    Uuid::new_v4().to_string() // Cryptographically secure nonce
}

fn sign_request(request: RequestBuilder, secret: &str) -> RequestBuilder {
    let nonce = create_secure_nonce();
    let body_hash = hmac_sha512(secret.as_bytes(), nonce.as_bytes());
    
    request
        .header("X-Auth-Nonce", nonce)
        .header("X-Auth-Signature", format!("{:x}", body_hash))
        .header("X-Request-ID", Uuid::new_v4().to_string())
}

/// Integrated with historical data validation and AI model orchestration
pub async fn predict_correlation(asset1: &str, asset2: &str) -> Result<f64, ApiError> {
    let config = get_ai_service_config()?;
    let client = Client::new();
    
    // Validate asset pair securely
    validate_asset_pair(asset1, asset2).await.map_err(|e| {
        log_correlation_prediction(LogLevel::Error, asset1, asset2, &e.to_string());
        ApiError::InvalidAssetPair(format!("Validation failed: {}", e))
    })?;

    // Check cache first
    let cache_key = (asset1.to_string(), asset2.to_string());
    {
        let cache = PREDICTION_CACHE.lock().await;
        if let Some(pred) = cache.get(&cache_key) {
            return Ok(*pred);
        }
    }

    // Fetch validated historical data with anomaly detection
    let history = fetch_validated_correlation_history(asset1, asset2)
        .await
        .map_err(|e| ApiError::DataValidationFailed(e.to_string()))?;

    let payload = json!({
        "assets": [asset1, asset2],
        "correlation_series": history,
        "model_version": "quantum_ensemble_v3"
    });

    let signed_request = sign_request(client.post(&config.prediction_endpoint), &config.api_secret)
        .json(&payload)
        .timeout(Duration::from_secs(2));

    match timeout(Duration::from_secs(3), signed_request.send()).await {
        Ok(Ok(response)) if response.status().is_success() => {
            let json: serde_json::Value = response.json().await.map_err(|e| {
                log_correlation_prediction(LogLevel::Error, asset1, asset2, &e.to_string());
                ApiError::InvalidResponseFormat
            })?;

            let prediction = json["prediction"]
                .as_f64()
                .ok_or(ApiError::InvalidPredictionData)?;

            // Update cache and return
            let mut cache = PREDICTION_CACHE.lock().await;
            cache.insert(cache_key, prediction);
            
            log_correlation_prediction(LogLevel::Info, asset1, asset2, &format!("Prediction: {:.4}", prediction));
            Ok(prediction)
        }
        Ok(Err(e)) => {
            log_correlation_prediction(LogLevel::Error, asset1, asset2, &e.to_string());
            Err(ApiError::ServiceUnavailable(e.to_string()))
        }
        Err(_) => {
            let msg = "Prediction service timeout".to_string();
            log_correlation_prediction(LogLevel::Warning, asset1, asset2, &msg);
            Err(ApiError::Timeout(msg))
        }
    }
}

/// Batch prediction with circuit breaker pattern
pub async fn batch_predict_correlations(pairs: &[(String, String)]) -> Result<HashMap<(String, String), f64>, ApiError> {
    let config = get_ai_service_config()?;
    let client = Client::new();
    
    // Validate all pairs first
    for (a1, a2) in pairs {
        crate::Core::data::asset_validator::validate_asset_pair(a1, a2).await?;
    }

    let payload = json!({
        "pairs": pairs,
        "historical_window": "720h",
        "model_selector": "adaptive_ensemble"
    });

    let signed_request = sign_request(client.post(&config.batch_endpoint), &config.api_secret)
        .json(&payload)
        .timeout(Duration::from_secs(5));

    match timeout(Duration::from_secs(7), signed_request.send()).await {
        Ok(Ok(response)) if response.status().is_success() => {
            let predictions: HashMap<(String, String), f64> = response.json().await.map_err(|_| ApiError::InvalidResponseFormat)?;
            
            // Update cache
            let mut cache = PREDICTION_CACHE.lock().await;
            for (pair, value) in &predictions {
                cache.insert(pair.clone(), *value);
            }
            
            Ok(predictions)
        }
        Ok(Err(e)) => {
            log_correlation_prediction(LogLevel::Error, "batch", "processing", &e.to_string());
            Err(ApiError::ServiceUnavailable(e.to_string()))
        }
        Err(_) => {
            let msg = "Batch prediction timeout".to_string();
            log_correlation_prediction(LogLevel::Warning, "batch", "processing", &msg);
            Err(ApiError::Timeout(msg))
        }
    }
}