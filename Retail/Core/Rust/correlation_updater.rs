use rayon::prelude::*;
use tokio::task;
use chrono::Duration;
use lru::LruCache;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::market_data::{fetch_historical_data, fetch_volatility, fetch_liquidity};
use crate::stats::{calculate_rolling_corr, detect_anomaly, detect_volatility_clusters};
use crate::graph_db::store_correlation; // ✅ Store in Graph DB
use log::{warn, error, info};

fn generate_asset_pairs(assets: &[String]) -> Vec<(String, String)> {
    let mut pairs = Vec::with_capacity(assets.len() * (assets.len() - 1) / 2);
    
    assets.par_iter().enumerate().for_each(|(i, a1)| {
        for a2 in &assets[i+1..] {
            if should_compare(a1, a2) {
                pairs.push((a1.clone(), a2.clone()));
            }
        }
    });
    
    pairs
}

fn should_compare(a1: &str, a2: &str) -> bool {
    // Implement asset class filtering
    !(is_crypto(a1) && is_forex(a2)) &&
    correlation_coverage(a1, a2) > 0.8
}

pub async fn update_correlations(&self) {
    let assets = self.load_asset_universe();
    let pairs = generate_asset_pairs(&assets);

    let volatility_cache = Arc::new(Mutex::new(LruCache::new(200))); // ✅ Increased Cache Size

    pairs.into_par_iter().for_each(|(a1, a2)| {
        let volatility_cache = Arc::clone(&volatility_cache);

        task::spawn(async move {
            // ✅ Batch Fetch Volatility
            let (volatility1, volatility2) = {
                let mut cache = volatility_cache.lock().unwrap();
                
                let v1 = match cache.get(&a1) {
                    Some(v) => *v,
                    None => {
                        let v = fetch_volatility(a1).await.unwrap_or_else(|| {
                            error!("Volatility fetch failed for {}", a1);
                            f64::NAN
                        });
                        if v.is_normal() {  // ✅ Only cache valid numbers
                            cache.put(a1.clone(), v);
                        }
                        v
                    }
                };
                
                let v2 = match cache.get(&a2) {
                    Some(v) => *v,
                    None => {
                        let v = fetch_volatility(a2).await.unwrap_or_else(|| {
                            error!("Volatility fetch failed for {}", a2);
                            f64::NAN
                        });
                        if v.is_normal() {  // ✅ Only cache valid numbers
                            cache.put(a2.clone(), v);
                        }
                        v
                    }
                };

                (v1, v2)
            };

            let avg_volatility = (volatility1 + volatility2) / 2.0;

            // 🔄 **Improved Adaptive Timeframes**
            let timeframe = match avg_volatility {
                v if v > 0.1 => Duration::hours(6),
                v if v > 0.05 => Duration::hours(12),
                _ => Duration::hours(24),
            };

            // ✅ Calculate correlation safely
            let corr = match calculate_rolling_corr(a1, a2, timeframe).await {
                Ok(value) if value.is_finite() => value,
                _ => {
                    error!("❌ Correlation calculation failed for {}-{}", a1, a2);
                    return;
                }
            };

            // ✅ Improved Anomaly Detection
            if detect_anomaly(a1, a2, corr).await {
                warn!("⚠️ Correlation anomaly detected: {}-{}", a1, a2);
            }

            // ✅ Batch Fetch Liquidity
            let liquidity1 = fetch_liquidity(a1).await.unwrap_or(0.0);
            let liquidity2 = fetch_liquidity(a2).await.unwrap_or(0.0);
            let min_liquidity_threshold = 500_000.0;

            if (corr.abs() > 0.6) && (liquidity1 > min_liquidity_threshold && liquidity2 > min_liquidity_threshold) {
                self.correlation_graph.update(a1, a2, corr);
                store_correlation(a1, a2, corr).await; // ✅ Store in Graph DB
                info!("🔗 Correlation updated for {}-{}: {:.4}", a1, a2, corr);
            }
        });
    });
}
