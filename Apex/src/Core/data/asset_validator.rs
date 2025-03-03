use crate::config::assets::asset_universe::AssetUniverse;
use crate::core::data::{
    insider_monitor::InsiderActivityAnalyzer,
    market_data::MarketDataAPI,
    order_book_analyzer::OrderBookAnalyzer,
    correlation_monitor::CorrelationMonitor
};
use crate::core::trading::{
    risk::risk_engine::RiskEngine,
    strategies::arbitrage::CrossExchangeArbitrageDetector,
    execution::market_impact::MarketImpactCalculator
};
use crate::utils::{
    error_handler::{ApiError, Result},
    logging::structured_logger::{Logger, ValidationMetric}
};
use crate::ai::analysis::{
    fraud_detector::FraudDetectionModel,
    asset_scoring_engine::AssetScoringEngine,
    market_regime_classifier::MarketRegimeClassifier
};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Defines the thresholds for validation which can be adjusted based on market regimes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationThresholds {
    pub min_liquidity: f64,
    pub max_spread: f64,
    pub min_order_book_depth: f64,
    pub max_price_impact: f64,
    pub max_fraud_score: f64,
    pub max_wash_trading_risk: f64,
    pub min_market_cap: f64,
}

impl Default for ValidationThresholds {
    fn default() -> Self {
        Self {
            min_liquidity: 100_000.0,
            max_spread: 0.05,
            min_order_book_depth: 500_000.0,
            max_price_impact: 0.02,
            max_fraud_score: 0.7,
            max_wash_trading_risk: 0.65,
            min_market_cap: 50_000_000.0,
        }
    }
}

/// Represents comprehensive validation result with detailed metrics
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResult {
    pub metrics: ValidationMetric,
    pub score: AssetScore,
    pub is_valid: bool,
    pub validation_timestamp: i64,
    pub market_regime: String,
}

/// Multi-dimensional asset score with component breakdowns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetScore {
    pub liquidity: f64,
    pub safety: f64,
    pub profitability: f64,
    pub market_structure: f64,
    pub institutional_interest: f64,
    pub composite: f64,
}

/// Blockchain analysis results structure
#[derive(Debug, Default, Clone)]
pub struct BlockchainAnalysis {
    pub large_transactions: u32,
    pub whale_concentration: f64,
    pub recent_contract_interactions: Vec<String>,
    pub average_transaction_size: f64,
}

/// Compliance status enum representing regulatory compliance checks
#[derive(Debug, Clone, PartialEq)]
pub enum ComplianceStatus {
    Compliant,
    Restricted,
    Banned,
    UnderInvestigation,
    Unknown,
}

/// Interface for regulatory database operations
pub trait RegulatoryDatabase: Send + Sync {
    fn check_asset_compliance(&self, asset: &str) -> Result<ComplianceStatus>;
    fn get_jurisdiction_restrictions(&self, asset: &str) -> Result<Vec<String>>;
}

/// Regulatory database implementation
pub struct ComplianceChecker {
    regulatory_db: Arc<dyn RegulatoryDatabase>,
    compliance_cache: dashmap::DashMap<String, (ComplianceStatus, i64)>, // Asset -> (Status, Timestamp)
    cache_ttl_seconds: i64,
}

impl ComplianceChecker {
    pub fn new(regulatory_db: Arc<dyn RegulatoryDatabase>) -> Self {
        Self {
            regulatory_db,
            compliance_cache: dashmap::DashMap::new(),
            cache_ttl_seconds: 3600, // 1 hour cache
        }
    }
    
    pub fn check_asset_compliance(&self, asset: &str) -> Result<ComplianceStatus> {
        let now = chrono::Utc::now().timestamp();
        
        // Check cache first
        if let Some(entry) = self.compliance_cache.get(asset) {
            let (status, timestamp) = entry.value().clone();
            if now - timestamp < self.cache_ttl_seconds {
                return Ok(status);
            }
        }
        
        // Cache miss or expired, get fresh data
        let status = self.regulatory_db.check_asset_compliance(asset)?;
        self.compliance_cache.insert(asset.to_string(), (status.clone(), now));
        Ok(status)
    }
}

/// Institutional-grade asset validation system
pub struct AssetValidator {
    asset_universe: Arc<AssetUniverse>,
    market_data: Arc<MarketDataAPI>,
    risk_engine: Arc<RiskEngine>,
    order_book_analyzer: Arc<OrderBookAnalyzer>,
    arbitrage_detector: Arc<CrossExchangeArbitrageDetector>,
    fraud_detector: Arc<FraudDetectionModel>,
    insider_monitor: Arc<InsiderActivityAnalyzer>,
    scoring_engine: Arc<AssetScoringEngine>,
    market_regime_classifier: Arc<MarketRegimeClassifier>,
    correlation_monitor: Arc<CorrelationMonitor>,
    market_impact_calculator: Arc<MarketImpactCalculator>,
    logger: Arc<Logger>,
    compliance_checker: Arc<ComplianceChecker>,
    thresholds: RwLock<ValidationThresholds>,
    validation_cache: dashmap::DashMap<String, (ValidationResult, i64)>, // Asset pair -> (Result, Timestamp)
    cache_ttl_ms: i64,
}

impl AssetValidator {
    pub async fn new(
        asset_universe: Arc<AssetUniverse>,
        market_data: Arc<MarketDataAPI>,
        risk_engine: Arc<RiskEngine>,
        regulatory_db: Arc<dyn RegulatoryDatabase>
    ) -> Result<Self> {
        let compliance_checker = Arc::new(ComplianceChecker::new(regulatory_db));
        
        Ok(Self {
            asset_universe,
            market_data,
            risk_engine,
            order_book_analyzer: Arc::new(OrderBookAnalyzer::new()),
            arbitrage_detector: Arc::new(CrossExchangeArbitrageDetector::new()),
            fraud_detector: Arc::new(FraudDetectionModel::load_model().await?),
            insider_monitor: Arc::new(InsiderActivityAnalyzer::new()),
            scoring_engine: Arc::new(AssetScoringEngine::new()),
            market_regime_classifier: Arc::new(MarketRegimeClassifier::new().await?),
            correlation_monitor: Arc::new(CorrelationMonitor::new()),
            market_impact_calculator: Arc::new(MarketImpactCalculator::new()),
            logger: Arc::new(Logger::new("AssetValidator")),
            compliance_checker,
            thresholds: RwLock::new(ValidationThresholds::default()),
            validation_cache: dashmap::DashMap::new(),
            cache_ttl_ms: 500, // 500ms cache for HFT
        })
    }

    /// Dynamically adjust validation thresholds based on current market regime
    pub async fn adjust_thresholds_for_market_regime(&self) -> Result<()> {
        let current_regime = self.market_regime_classifier.get_current_regime().await?;
        let mut thresholds = self.thresholds.write().await;
        
        match current_regime.as_str() {
            "BULL" => {
                thresholds.min_liquidity *= 0.8;  // Less stringent in bull markets
                thresholds.max_spread *= 1.2;
            },
            "BEAR" => {
                thresholds.min_liquidity *= 1.5;  // More stringent in bear markets
                thresholds.max_spread *= 0.8;
                thresholds.max_fraud_score *= 0.9; // Lower fraud tolerance in bear markets
            },
            "VOLATILE" => {
                thresholds.min_liquidity *= 2.0;  // Much more stringent in volatile markets
                thresholds.max_price_impact *= 0.7;
                thresholds.min_order_book_depth *= 1.5;
            },
            _ => {} // Default thresholds for normal or unknown regimes
        }
        
        Ok(())
    }

    /// Validate an asset pair with caching for HFT performance
    pub async fn validate_asset_pair(&self, base: &str, quote: &str) -> Result<ValidationResult> {
        let pair_key = format!("{}-{}", base, quote);
        let now = chrono::Utc::now().timestamp_millis();
        
        // Check cache first for ultra-low latency
        if let Some(entry) = self.validation_cache.get(&pair_key) {
            let (result, timestamp) = entry.value();
            if now - timestamp < self.cache_ttl_ms {
                return Ok(result.clone());
            }
        }
        
        // Adjust thresholds based on current market conditions
        self.adjust_thresholds_for_market_regime().await?;
        
        let mut metrics = ValidationMetric::new();
        
        // Core validation pipeline - all running concurrently for speed
        let universe_fut = self.check_asset_universe(base, quote, &mut metrics);
        let liquidity_fut = self.analyze_liquidity(base, quote, &mut metrics);
        let compliance_fut = self.check_compliance(base, quote, &mut metrics);
        let risk_fut = self.assess_risk(base, quote, &mut metrics);
        let correlation_fut = self.detect_correlations(base, quote, &mut metrics);
        let market_structure_fut = self.analyze_market_structure(base, quote, &mut metrics);
        let smart_money_fut = self.detect_smart_money(base, quote, &mut metrics);
        let fraud_fut = self.scan_for_fraud(base, quote, &mut metrics);
        
        // Execute all checks concurrently
        tokio::try_join!(
            universe_fut, liquidity_fut, compliance_fut, risk_fut,
            correlation_fut, market_structure_fut, smart_money_fut, fraud_fut
        )?;
        
        // Generate final score
        let score = self.scoring_engine.calculate_score(&metrics).await?;
        
        // Check if passing all thresholds
        let thresholds = self.thresholds.read().await;
        let is_valid = 
            metrics.get("volume_24h").unwrap_or(&0.0) >= &thresholds.min_liquidity &&
            metrics.get("bid_ask_spread").unwrap_or(&1.0) <= &thresholds.max_spread &&
            metrics.get("order_book_depth").unwrap_or(&0.0) >= &thresholds.min_order_book_depth &&
            metrics.get("price_impact_100k").unwrap_or(&1.0) <= &thresholds.max_price_impact &&
            metrics.get("fraud_score").unwrap_or(&1.0) <= &thresholds.max_fraud_score &&
            metrics.get("wash_trading_risk").unwrap_or(&1.0) <= &thresholds.max_wash_trading_risk;
        
        let current_regime = self.market_regime_classifier.get_current_regime().await?;
        
        let result = ValidationResult {
            metrics,
            score,
            is_valid,
            validation_timestamp: now,
            market_regime: current_regime,
        };
        
        // Update cache
        self.validation_cache.insert(pair_key, (result.clone(), now));
        
        Ok(result)
    }
    
    /// Return only assets passing validation as a filtered universe
    pub async fn get_validated_universe(&self) -> Result<Vec<(String, String)>> {
        let asset_pairs = self.asset_universe.get_all_asset_pairs().await?;
        let mut valid_pairs = Vec::new();
        
        let mut validation_futures = Vec::new();
        
        // Create concurrent validation tasks
        for (base, quote) in asset_pairs {
            let validator = self.clone();
            let base_clone = base.clone();
            let quote_clone = quote.clone();
            
            let future = tokio::spawn(async move {
                let result = validator.validate_asset_pair(&base_clone, &quote_clone).await;
                (base_clone, quote_clone, result)
            });
            
            validation_futures.push(future);
        }
        
        // Gather results
        for future in validation_futures {
            let (base, quote, result) = future.await.map_err(|e| ApiError::TaskJoinError(e.to_string()))?;
            
            match result {
                Ok(validation) if validation.is_valid => {
                    valid_pairs.push((base, quote));
                }
                Ok(_) => {
                    // Asset failed validation
                }
                Err(e) => {
                    self.logger.error(&format!("Validation error for {}/{}: {}", base, quote, e));
                }
            }
        }
        
        Ok(valid_pairs)
    }

    async fn check_asset_universe(&self, base: &str, quote: &str, metrics: &mut ValidationMetric) -> Result<()> {
        let exists = self.asset_universe.asset_pair_exists(base, quote).await?;
        let base_market_cap = self.market_data.get_market_cap(base).await.unwrap_or(0.0);
        
        metrics.insert("exists_in_universe", if exists { 1.0 } else { 0.0 });
        metrics.insert("market_cap", base_market_cap);
        
        if !exists {
            return Err(ApiError::AssetNotInUniverse);
        }
        
        Ok(())
    }

    async fn analyze_liquidity(&self, base: &str, quote: &str, metrics: &mut ValidationMetric) -> Result<()> {
        let base_liquidity = self.market_data.get_liquidity_metrics(base).await?;
        let quote_liquidity = self.market_data.get_liquidity_metrics(quote).await?;
        
        // Advanced order book analysis
        let order_book_depth = self.order_book_analyzer.analyze_depth(base).await?;
        let spread_analysis = self.order_book_analyzer.analyze_spread_trends(base).await?;
        
        // Calculate buy-side and sell-side liquidity imbalance
        let imbalance = self.order_book_analyzer.calculate_liquidity_imbalance(base).await?;
        
        metrics.insert("bid_ask_spread", spread_analysis.current_spread);
        metrics.insert("spread_volatility", spread_analysis.spread_volatility);
        metrics.insert("order_book_depth", order_book_depth);
        metrics.insert("volume_24h", base_liquidity.volume);
        metrics.insert("turnover_ratio", base_liquidity.turnover_ratio);
        metrics.insert("liquidity_imbalance", imbalance);

        Ok(())
    }
    
    async fn check_compliance(&self, base: &str, quote: &str, metrics: &mut ValidationMetric) -> Result<()> {
        let base_status = self.compliance_checker.check_asset_compliance(base)?;
        let quote_status = self.compliance_checker.check_asset_compliance(quote)?;
        
        let base_compliant = base_status == ComplianceStatus::Compliant;
        let quote_compliant = quote_status == ComplianceStatus::Compliant;
        
        metrics.insert("base_compliant", if base_compliant { 1.0 } else { 0.0 });
        metrics.insert("quote_compliant", if quote_compliant { 1.0 } else { 0.0 });
        
        if !base_compliant || !quote_compliant {
            self.logger.warning(&format!("Compliance issues detected for {}/{}", base, quote));
            return Err(ApiError::ComplianceIssue);
        }
        
        Ok(())
    }
    
    async fn assess_risk(&self, base: &str, quote: &str, metrics: &mut ValidationMetric) -> Result<()> {
        let risk_profile = self.risk_engine.analyze_asset_risk(base).await?;
        
        // Extract key risk metrics
        let volatility = risk_profile.volatility;
        let var_95 = risk_profile.value_at_risk_95;
        let max_drawdown = risk_profile.max_drawdown;
        
        metrics.insert("volatility", volatility);
        metrics.insert("var_95", var_95);
        metrics.insert("max_drawdown", max_drawdown);
        metrics.insert("risk_score", risk_profile.composite_risk);
        
        Ok(())
    }
    
    async fn detect_correlations(&self, base: &str, quote: &str, metrics: &mut ValidationMetric) -> Result<()> {
        let correlations = self.correlation_monitor.get_correlations(base).await?;
        let top_correlation = correlations.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(asset, corr)| (asset.clone(), *corr));
            
        if let Some((top_asset, top_corr)) = top_correlation {
            metrics.insert("top_correlation", top_corr);
            metrics.insert("is_correlated_pair", if top_asset == quote { 1.0 } else { 0.0 });
        }
        
        // Detect cointegration for pair trading opportunities
        let cointegration = self.correlation_monitor.test_cointegration(base, quote).await?;
        metrics.insert("cointegration", cointegration);
        
        Ok(())
    }

    async fn detect_smart_money(&self, base: &str, quote: &str, metrics: &mut ValidationMetric) -> Result<()> {
        // Institutional flow tracking
        let insider_activity = self.insider_monitor.check_insider_activity(base).await?;
        let institutional_flows = self.insider_monitor.get_institutional_flows(base).await?;
        
        // Advanced whale analysis
        let whale_pattern = self.insider_monitor.detect_whale_accumulation_patterns(base).await?;
        
        // Blockchain analysis for crypto assets
        let on_chain_analysis = if self.asset_universe.is_crypto(base).await? {
            self.market_data.get_blockchain_analytics(base).await?
        } else {
            BlockchainAnalysis::default()
        };

        metrics.insert("insider_activity_score", insider_activity.score);
        metrics.insert("institutional_flow", institutional_flows.net_flow);
        metrics.insert("whale_transactions", on_chain_analysis.large_transactions as f64);
        metrics.insert("whale_accumulation", whale_pattern.confidence);
        metrics.insert("smart_money_conviction", whale_pattern.conviction * institutional_flows.direction);

        Ok(())
    }

    async fn scan_for_fraud(&self, base: &str, quote: &str, metrics: &mut ValidationMetric) -> Result<()> {
        // AI-powered anomaly detection
        let fraud_score = self.fraud_detector.analyze_asset(base).await?;
        let wash_trading_risk = self.fraud_detector.detect_wash_trading(base).await?;
        let pump_dump_pattern = self.fraud_detector.detect_pump_and_dump(base).await?;
        let behavioral_anomalies = self.fraud_detector.analyze_behavioral_patterns(base).await?;
        
        metrics.insert("fraud_score", fraud_score);
        metrics.insert("wash_trading_risk", wash_trading_risk);
        metrics.insert("pump_dump_risk", pump_dump_pattern);
        metrics.insert("behavioral_anomaly_score", behavioral_anomalies);

        if fraud_score > 0.7 || wash_trading_risk > 0.65 {
            self.logger.alert(&format!("High fraud risk detected: {}/{}", base, quote));
            return Err(ApiError::FraudulentAssetDetected);
        }

        Ok(())
    }

    async fn analyze_market_structure(&self, base: &str, quote: &str, metrics: &mut ValidationMetric) -> Result<()> {
        // HFT and market microstructure analysis
        let hft_presence = self.order_book_analyzer.detect_hft_patterns(base).await?;
        let price_impact = self.market_impact_calculator.simulate_market_impact(base, 100_000.0).await?;
        let tick_by_tick = self.order_book_analyzer.analyze_microstructure(base).await?;
        
        // Market maker analysis
        let market_maker_dominance = self.order_book_analyzer.detect_market_maker_dominance(base).await?;

        metrics.insert("hft_activity_score", hft_presence.score);
        metrics.insert("price_impact_100k", price_impact);
        metrics.insert("tick_stability", tick_by_tick.stability_score);
        metrics.insert("market_maker_dominance", market_maker_dominance);

        Ok(())
    }

    pub async fn clone(&self) -> Self {
        Self {
            asset_universe: self.asset_universe.clone(),
            market_data: self.market_data.clone(),
            risk_engine: self.risk_engine.clone(),
            order_book_analyzer: self.order_book_analyzer.clone(),
            arbitrage_detector: self.arbitrage_detector.clone(),
            fraud_detector: self.fraud_detector.clone(),
            insider_monitor: self.insider_monitor.clone(),
            scoring_engine: self.scoring_engine.clone(),
            market_regime_classifier: self.market_regime_classifier.clone(),
            correlation_monitor: self.correlation_monitor.clone(),
            market_impact_calculator: self.market_impact_calculator.clone(),
            logger: self.logger.clone(),
            compliance_checker: self.compliance_checker.clone(),
            thresholds: RwLock::new(self.thresholds.read().await.clone()),
            validation_cache: self.validation_cache.clone(),
            cache_ttl_ms: self.cache_ttl_ms,
        }
    }
}