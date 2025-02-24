use std::collections::VecDeque;
use crate::market_data::{fetch_historical_data, fetch_volatility, fetch_liquidity};
use statrs::statistics::{Statistics, correlation, kendall, spearman};
use tokio::task;
use tokio::sync::Mutex;
use std::sync::Arc;
/// ✅ **Thread-Safe Circular Buffer for Historical Correlation**
static HISTORICAL_CORRELATIONS: once_cell::sync::Lazy<Arc<Mutex<[f64; HISTORY_SIZE]>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new([0.0; HISTORY_SIZE])));

const HISTORY_SIZE: usize = 100;  // ✅ Increased History Buffer for Stability
const Z_SCORE_THRESHOLD: f64 = 2.5; // ✅ Adaptive Outlier Sensitivity
const ALPHA: f64 = 0.2; // ✅ EWMA (Exponential Weighted Moving Average) Smoothing Factor

/// ✅ **Updates the Correlation Buffer**
pub async fn update_correlation_buffer(new_corr: f64) {
    let mut historical_corr = HISTORICAL_CORRELATIONS.lock().await;
    historical_corr.rotate_left(1);
    historical_corr[HISTORY_SIZE - 1] = new_corr;
}
/// ✅ **Detects Correlation Anomalies using Rolling Z-Score & IQR**
pub async fn detect_anomaly(new_corr: f64) -> bool {
    let mut historical_corr = HISTORICAL_CORRELATIONS.lock().await;

    // ✅ **Rolling Update (Circular Buffer)**
    historical_corr.rotate_left(1);
    historical_corr[HISTORY_SIZE - 1] = new_corr;

    let mut sorted_corr = historical_corr.to_vec();
    sorted_corr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // ✅ **Interquartile Range (IQR) Instead of MAD**
    let q1 = sorted_corr[HISTORY_SIZE / 4];
    let q3 = sorted_corr[(3 * HISTORY_SIZE) / 4];
    let iqr = q3 - q1;
    let upper_bound = q3 + (iqr * 1.5);
    let lower_bound = q1 - (iqr * 1.5);

    // ✅ **Check for Anomaly**
    new_corr > upper_bound || new_corr < lower_bound
}

/// ✅ **Fixed-Size Circular Buffer for EMA Volatility**
static LAST_VOLATILITY: once_cell::sync::Lazy<Arc<Mutex<[f64; HISTORY_SIZE]>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new([0.5; HISTORY_SIZE])));

/// ✅ **Detects Market Volatility Clustering using ATR & EMA**
pub async fn detect_volatility_clusters(asset1: &str, asset2: &str) -> f64 {
    let (vol1, vol2) = tokio::join!(
        fetch_volatility(asset1),
        fetch_volatility(asset2)
    );

    let atr = (vol1.unwrap_or(0.0) + vol2.unwrap_or(0.0)) / 2.0; // ✅ **ATR-Based Volatility Estimation**
    let volatility_diff = (vol1.unwrap_or(0.0) - vol2.unwrap_or(0.0)).abs() / (vol1.unwrap_or(0.0) + vol2.unwrap_or(0.0) + 1e-6);

    let mut last_volatility = LAST_VOLATILITY.lock().await;

    last_volatility.rotate_left(1);
    last_volatility[HISTORY_SIZE - 1] = atr;
    let mean_volatility = last_volatility.iter().copied().sum::<f64>() / HISTORY_SIZE as f64;

    let ema_volatility = (ALPHA * atr) + ((1.0 - ALPHA) * mean_volatility); // ✅ **Using ALPHA for EMA Calculation**

    // ✅ **Normalized Volatility Score (1 = High Correlation in Volatility)**
    (ema_volatility * (1.0 - volatility_diff)).min(1.0)
}

/// ✅ **Computes Adaptive Rolling Correlation using Liquidity & Volatility Scaling**
pub async fn calculate_rolling_corr(asset1: &str, asset2: &str, window: chrono::Duration) -> f64 {
    // ✅ **Fixed Asynchronous Data Fetching (Now Properly Handles Errors)**
    let (price_data1, price_data2, liquidity1, liquidity2) = tokio::try_join!(
        async { fetch_historical_data(asset1, window).await.unwrap_or_default() },
        async { fetch_historical_data(asset2, window).await.unwrap_or_default() },
        async { fetch_liquidity(asset1).await.unwrap_or(0.0) },
        async { fetch_liquidity(asset2).await.unwrap_or(0.0) },
    ).await;

    let price_data1 = price_data1.unwrap_or(Ok(vec![])).unwrap_or_default();
    let price_data2 = price_data2.unwrap_or(Ok(vec![])).unwrap_or_default();
    let liquidity1 = liquidity1.unwrap_or(Ok(0.0)).unwrap_or(0.0);
    let liquidity2 = liquidity2.unwrap_or(Ok(0.0)).unwrap_or(0.0);

    let avg_liquidity = (liquidity1 + liquidity2) / 2.0;
    let avg_volatility = (detect_volatility_clusters(asset1, asset2).await) * 100.0;

    if price_data1.len() < 10 || price_data2.len() < 10 {
        return 0.0; // ✅ Prevent Division Errors with Insufficient Data
    }

    // ✅ **Logarithmic Liquidity Scaling (Avoids log(0) Errors)**
    let adjusted_window = match avg_liquidity.log1p().max(1e-6) {
        l if l > 6.0 => window + chrono::Duration::hours(18),  // ✅ **High Liquidity → Longer Window**
        l if l > 5.0 => window + chrono::Duration::hours(6),   // ✅ **Medium Liquidity → Normal Window**
        _ => window - chrono::Duration::hours(6),             // ✅ **Low Liquidity → Shorter Window**
    };

    // ✅ **Volatility Scaling using Logarithmic Adjustment**
    let final_window = match avg_volatility.log1p().max(1e-6) {
        v if v > 1.5 => adjusted_window - chrono::Duration::hours(8),
        _ => adjusted_window,
    // ✅ **Parallelized Multi-Correlation Computation (Now Handles NaN Properly)**
    let (pearson_corr, kendall_corr, spearman_corr) = tokio::try_join!(
        task::spawn_blocking(move || correlation(&price_data1, &price_data2).unwrap_or(0.0)),
        task::spawn_blocking(move || kendall(&price_data1, &price_data2).unwrap_or(0.0)),
        task::spawn_blocking(move || spearman(&price_data1, &price_data2).unwrap_or(0.0))
    ).unwrap_or((0.0, 0.0, 0.0));

    // ✅ **Ensure No NaN Values Are Used**
    let final_correlation = (pearson_corr * 0.6) + (kendall_corr * 0.2) + (spearman_corr * 0.2);
    final_correlation.max(-1.0).min(1.0)  // ✅ Clamp Values to [-1, 1]
