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
                (
                    *cache.get(&a1).unwrap_or(&fetch_volatility(a1).await.unwrap_or(0.0)),
                    *cache.get(&a2).unwrap_or(&fetch_volatility(a2).await.unwrap_or(0.0)),
                )
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
