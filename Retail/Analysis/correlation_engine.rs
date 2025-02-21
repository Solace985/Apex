use rayon::prelude::*;
use tokio::task;
use tokio::sync::RwLock;
use chrono::{Duration, Utc};
use lru::LruCache;
use std::collections::HashMap;
use std::sync::Arc;
use crate::market_data::{fetch_fundamental_data, fetch_historical_data, fetch_volatility};
use crate::stats::{calculate_dtw_correlation};
use crate::ai_models::predict_correlation;
use crate::graph_db::store_correlation;
use log::{warn, error, info};

pub struct CorrelationEngine {
    asset_sources: HashMap<&'static str, Vec<&'static str>>,
    ai_forecast_cache: Arc<RwLock<LruCache<(String, String), (f64, f64, i64)>>>, // ✅ AI Cache (Prediction, Confidence, Expiry)
}

impl CorrelationEngine {
    pub fn new() -> Self {
        let mut sources = HashMap::new();

        sources.insert("equity", vec!["sec_filings", "earnings_calls", "insider_trades"]);
        sources.insert("crypto", vec!["on_chain_flow", "protocol_revenue", "whale_alerts"]);
        sources.insert("forex", vec!["central_bank_sentiment", "cot_report", "geopolitical_risk"]);

        Self {
            asset_sources: sources,
            ai_forecast_cache: Arc::new(RwLock::new(LruCache::new(200))), // ✅ Increased Cache Size
        }
    }

    /// **🔄 AI + Historical Data-Based Correlation Detection**
    pub async fn auto_detect_relationships(&self) {
        let assets: Vec<&str> = self.asset_sources.keys().cloned().collect();

        assets.into_par_iter().for_each(|asset1| {
            let ai_cache = Arc::clone(&self.ai_forecast_cache);

            assets.iter().for_each(|&asset2| {
                if asset1 != asset2 {
                    task::spawn(async move {
                        let correlation = self.calculate_correlation(asset1, asset2, ai_cache.clone()).await;
                        if correlation > 0.7 {
                            self.add_relationship(asset1, asset2, correlation);
                        }
                    });
                }
            });
        });
    }

    /// **🔍 AI-Driven Hybrid Correlation Calculation**
    async fn calculate_correlation(
        &self,
        asset1: &str,
        asset2: &str,
        ai_cache: Arc<RwLock<LruCache<(String, String), (f64, f64, i64)>>>,
    ) -> f64 {
        // ✅ **Batch Fetch Fundamental & Volatility Data**
        let (data1, data2, vol1, vol2) = tokio::join!(
            fetch_fundamental_data(asset1),
            fetch_fundamental_data(asset2),
            fetch_volatility(asset1),
            fetch_volatility(asset2)
        );

        let data1 = data1.unwrap_or(vec![]);
        let data2 = data2.unwrap_or(vec![]);
        let vol1 = vol1.unwrap_or(0.0);
        let vol2 = vol2.unwrap_or(0.0);
        let avg_volatility = (vol1 + vol2) / 2.0;

        if data1.is_empty() || data2.is_empty() {
            error!("❌ No fundamental data available for {} or {}", asset1, asset2);
            return 0.0;
        }

        // ✅ **AI-Driven Forecasting with Smart TTL Expiry**
        let (ai_pred_corr, confidence, last_updated) = {
            let mut cache = ai_cache.write().await;
            let now = Utc::now().timestamp();

            match cache.get(&(asset1.to_string(), asset2.to_string())) {
                Some(&(pred, conf, expiry)) if now < expiry => (pred, conf, expiry), // ✅ Use Cached Value if Valid
                _ => {
                    let (pred, conf) = predict_correlation(asset1, asset2).await;
                    let expiry = now + if avg_volatility > 0.07 { 6 * 3600 } else { 18 * 3600 }; 
                    cache.put((asset1.to_string(), asset2.to_string()), (pred, conf, expiry));
                    (pred, conf, expiry)
                }
            }
        };

        if confidence < 0.65 {
            warn!(
                "⚠️ AI prediction for {}-{} has low confidence: {:.2}",
                asset1, asset2, confidence
            );
            return 0.0; // ✅ Ignore low-confidence correlations
        }

        // ✅ **Optimized DTW Computation**
        let dtw_corr = tokio::task::spawn_blocking(move || calculate_dtw_correlation(data1, data2))
            .await
            .unwrap_or(0.0);

        // ✅ **AI + Historical Data Weighting (Now Volatility-Aware)**
        let final_corr = match confidence {
            c if c > 0.85 && avg_volatility < 0.05 => (ai_pred_corr * 0.9) + (dtw_corr * 0.1), // ✅ AI Dominates in Low-Volatility, High-Confidence Predictions
            c if c > 0.75 => (ai_pred_corr * 0.7) + (dtw_corr * 0.3),  // ✅ Balanced Weighting
            _ => (ai_pred_corr * 0.6) + (dtw_corr * 0.4),             // ✅ More Reliance on Historical Data
        };

        // ✅ **Filter Out Weak Correlations**
        if final_corr.abs() > 0.75 {
            info!(
                "📈 AI-Enhanced DTW Correlation for {}-{}: {:.4} (AI: {:.4}, Confidence: {:.2}, Volatility: {:.3})",
                asset1, asset2, final_corr, ai_pred_corr, confidence, avg_volatility
            );

            store_correlation(asset1, asset2, final_corr).await; // ✅ Store in Graph DB
        } else {
            warn!(
                "⚠️ Correlation too weak ({}-{}): {:.4} (AI: {:.4}, Volatility: {:.3})",
                asset1, asset2, final_corr, ai_pred_corr, avg_volatility
            );
        }

        final_corr
    }

    /// **📊 Store High-Impact Correlations in Graph DB**
    fn add_relationship(&self, a1: &str, a2: &str, score: f64) {
        info!(
            "🔗 Strong Fundamental Correlation: {} <-> {} (Score: {:.4})",
            a1, a2, score
        );
    }
}