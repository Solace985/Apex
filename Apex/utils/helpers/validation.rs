use crate::Core::asset_repository::{Asset, AssetRepository};
use crate::utils::asset_error::AssetError;
use crate::Core::data::realtime::market_data::{MarketDataService, PriceSource};
use crate::Core::trading::risk::regulatory_checker::RegulatoryCompliance;
use crate::Core::strategies::statistical_analyzer::{StatisticalReport, ZScoreAnalysis};
use crate::Core::trading::risk::liquidity_assessor::LiquidityAssessor;
use crate::utils::logging::structured_logger::{Logger, LogLevel, LogCategory};
use crate::utils::crypto::hash::Hasher;
use crate::Config::config::ConfigManager;

use chrono::{DateTime, Duration, Utc};
use regex::Regex;
use std::collections::HashSet;
use std::sync::Arc;
use lazy_static::lazy_static;

// Validation thresholds for different asset classes
struct ValidationThresholds {
    max_price_deviation: f64,       // Maximum allowed deviation from reference price
    max_price_staleness: i64,       // Maximum age of price data in seconds
    min_volume_threshold: u64,      // Minimum required volume
    max_zscore_threshold: f64,      // Maximum allowed statistical deviation
    min_sources_required: usize,    // Minimum number of price sources required
    max_bid_ask_spread: f64,        // Maximum allowable bid-ask spread as percentage
}

lazy_static! {
    // Compile regex patterns only once
    static ref SYMBOL_PATTERN: Regex = Regex::new(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,4})?$").unwrap();
    static ref CRYPTO_SYMBOL_PATTERN: Regex = Regex::new(r"^[A-Z]{2,6}-[A-Z]{2,6}$").unwrap();
    static ref OPTION_SYMBOL_PATTERN: Regex = Regex::new(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$").unwrap();
    static ref FUTURES_SYMBOL_PATTERN: Regex = Regex::new(r"^[A-Z]{1,3}[FGHJKMNQUVXZ]\d{1,2}$").unwrap();
}

/// Asset validator that performs comprehensive validation on financial assets
/// before they can be used in the trading system.
pub struct AssetValidator {
    asset_repo: Arc<AssetRepository>,
    market_data: Arc<MarketDataService>,
    regulatory: Arc<RegulatoryCompliance>,
    stats_analyzer: Arc<ZScoreAnalysis>,
    liquidity_assessor: Arc<LiquidityAssessor>,
    config: Arc<ConfigManager>,
    logger: Logger,
    hasher: Hasher,
    // Cache for recently validated assets to prevent repeated validations
    validated_cache: HashSet<String>,
    thresholds: ValidationThresholds,
}

impl AssetValidator {
    /// Creates a new AssetValidator with the specified dependencies
    pub fn new(
        asset_repo: Arc<AssetRepository>,
        market_data: Arc<MarketDataService>,
        regulatory: Arc<RegulatoryCompliance>,
        stats_analyzer: Arc<ZScoreAnalysis>,
        liquidity_assessor: Arc<LiquidityAssessor>,
        config: Arc<ConfigManager>,
        logger: Logger,
    ) -> Self {
        // Load validation thresholds from configuration
        let thresholds = ValidationThresholds {
            max_price_deviation: config.get_f64("validation.max_price_deviation").unwrap_or(0.05),
            max_price_staleness: config.get_i64("validation.max_price_staleness_seconds").unwrap_or(30),
            min_volume_threshold: config.get_u64("validation.min_volume_threshold").unwrap_or(1000),
            max_zscore_threshold: config.get_f64("validation.max_zscore_threshold").unwrap_or(3.0),
            min_sources_required: config.get_usize("validation.min_price_sources").unwrap_or(2),
            max_bid_ask_spread: config.get_f64("validation.max_bid_ask_spread_pct").unwrap_or(0.03),
        };

        Self {
            asset_repo,
            market_data,
            regulatory,
            stats_analyzer,
            liquidity_assessor,
            config,
            logger,
            hasher: Hasher::new(),
            validated_cache: HashSet::new(),
            thresholds,
        }
    }

    /// Validates an asset against all validation criteria
    /// Returns statistical report if asset passes validation
    pub fn validate_asset(&mut self, asset: &Asset) -> Result<StatisticalReport, AssetError> {
        // Check cache first to avoid redundant validation
        let cache_key = format!("{}:{}", asset.symbol, asset.timestamp.timestamp());
        if self.validated_cache.contains(&cache_key) {
            self.logger.log(
                LogLevel::Debug,
                LogCategory::Validation,
                &format!("Asset {} already validated, using cached result", asset.symbol),
            );
            
            // Still need to return a statistical report
            return self.stats_analyzer.get_cached_report(&asset.symbol)
                .ok_or_else(|| AssetError::AnalysisUnavailable(format!(
                    "Statistical report not found for validated asset {}", asset.symbol
                )));
        }
        
        self.logger.log(
            LogLevel::Info,
            LogCategory::Validation,
            &format!("Validating asset: {}", asset.symbol),
        );
        
        // Perform validation pipeline
        self.validate_identifiers(asset)?;
        self.validate_pricing(asset)?;
        self.validate_market_conditions(asset)?;
        let report = self.validate_statistics(asset)?;
        self.validate_regulatory(asset)?;
        self.validate_integrity(asset)?;
        self.validate_cross_reference(asset)?;
        
        // Add to validation cache with TTL
        self.validated_cache.insert(cache_key);
        if self.validated_cache.len() > 10000 {
            // Simple cache eviction strategy - clear if too large
            self.validated_cache.clear();
        }
        
        self.logger.log(
            LogLevel::Info,
            LogCategory::Validation,
            &format!("Asset {} successfully validated", asset.symbol),
        );
        
        Ok(report)
    }
    
    /// Validates asset identifiers, symbol format, and basic properties
    fn validate_identifiers(&self, asset: &Asset) -> Result<(), AssetError> {
        // Symbol format validation based on asset type
        let symbol_valid = match asset.asset_type.as_str() {
            "equity" => SYMBOL_PATTERN.is_match(&asset.symbol),
            "crypto" => CRYPTO_SYMBOL_PATTERN.is_match(&asset.symbol),
            "option" => OPTION_SYMBOL_PATTERN.is_match(&asset.symbol),
            "futures" => FUTURES_SYMBOL_PATTERN.is_match(&asset.symbol),
            _ => SYMBOL_PATTERN.is_match(&asset.symbol), // Default pattern
        };
        
        if !symbol_valid {
            return Err(AssetError::InvalidFormat(format!(
                "Symbol {} does not match pattern for asset type {}", 
                asset.symbol, asset.asset_type
            )));
        }
        
        // CUSIP/ISIN validation for equities and bonds
        if asset.asset_type == "equity" || asset.asset_type == "bond" {
            if let Some(ref isin) = asset.isin {
                if isin.len() != 12 || !isin.chars().all(|c| c.is_alphanumeric()) {
                    return Err(AssetError::InvalidFormat(format!("Invalid ISIN format: {}", isin)));
                }
                
                // Verify ISIN checksum (simplified implementation)
                if !self.verify_isin_checksum(isin) {
                    return Err(AssetError::InvalidData(format!("ISIN checksum failed: {}", isin)));
                }
            }
        }
        
        // Validate issuer information
        if asset.issuer.is_empty() {
            return Err(AssetError::MissingData(format!("Missing issuer for {}", asset.symbol)));
        }
        
        Ok(())
    }
    
    /// Validates price, bid/ask spread, and data freshness
    fn validate_pricing(&self, asset: &Asset) -> Result<(), AssetError> {
        // Price reasonableness check
        if asset.price <= 0.0 || asset.price.is_nan() || asset.price.is_infinite() {
            return Err(AssetError::InvalidData(format!("Invalid price for {}", asset.symbol)));
        }
        
        // Volume validation
        if asset.volume < self.thresholds.min_volume_threshold {
            return Err(AssetError::IlliquidAsset(format!(
                "Insufficient volume for {}: {} < {}", 
                asset.symbol, asset.volume, self.thresholds.min_volume_threshold
            )));
        }
        
        // Bid-ask spread validation
        if let (Some(bid), Some(ask)) = (asset.bid, asset.ask) {
            if bid <= 0.0 || ask <= 0.0 || bid > ask {
                return Err(AssetError::InvalidData(format!(
                    "Invalid bid-ask prices for {}: bid={}, ask={}", asset.symbol, bid, ask
                )));
            }
            
            // Check if spread is too wide
            let spread_pct = (ask - bid) / ((ask + bid) / 2.0);
            if spread_pct > self.thresholds.max_bid_ask_spread {
                return Err(AssetError::ExcessiveSpread(format!(
                    "Bid-ask spread too wide for {}: {:.2}%", asset.symbol, spread_pct * 100.0
                )));
            }
        }
        
        // Timestamp freshness check
        let max_age = Duration::seconds(self.thresholds.max_price_staleness);
        if Utc::now() - asset.timestamp > max_age {
            return Err(AssetError::StaleData(format!(
                "Stale data for {}: age = {} seconds", 
                asset.symbol, (Utc::now() - asset.timestamp).num_seconds()
            )));
        }
        
        Ok(())
    }
    
    /// Validates market conditions including trading status and circuit breakers
    fn validate_market_conditions(&self, asset: &Asset) -> Result<(), AssetError> {
        // Check trading halts using market data service
        match self.market_data.is_halted(&asset.symbol) {
            Ok(true) => {
                return Err(AssetError::MarketHalt(format!(
                    "Trading halted for {}", asset.symbol
                )));
            }
            Err(e) => {
                self.logger.log(
                    LogLevel::Warning,
                    LogCategory::Validation,
                    &format!("Failed to check halt status for {}: {}", asset.symbol, e)
                );
                // Continue validation but note the issue
            }
            _ => {} // Not halted, continue
        }
        
        // Check market open status for relevant asset types
        if asset.asset_type == "equity" || asset.asset_type == "option" {
            if !self.market_data.is_market_open(&asset.asset_type) {
                return Err(AssetError::MarketClosed(format!(
                    "Market closed for {} ({})", asset.symbol, asset.asset_type
                )));
            }
        }
        
        // Check for circuit breakers
        if self.market_data.is_circuit_breaker_active(&asset.asset_exchange) {
            return Err(AssetError::CircuitBreaker(format!(
                "Circuit breaker active for exchange {}", asset.asset_exchange
            )));
        }
        
        // Check expiry dates for derivatives
        if let Some(expiry) = asset.expiry {
            if expiry < Utc::now() {
                return Err(AssetError::ExpiredContract(format!(
                    "Contract {} has expired at {}", asset.symbol, expiry
                )));
            }
            
            // Warn on near-expiry contracts
            let warning_period = Duration::days(3);
            if expiry - Utc::now() < warning_period {
                self.logger.log(
                    LogLevel::Warning,
                    LogCategory::Validation,
                    &format!("Contract {} expiring soon at {}", asset.symbol, expiry)
                );
            }
        }
        
        // Check liquidity requirements based on asset type
        let liquidity_status = self.liquidity_assessor.assess_liquidity(
            &asset.symbol, 
            asset.volume, 
            &asset.asset_type
        )?;
        
        if !liquidity_status.is_sufficient {
            return Err(AssetError::IlliquidAsset(format!(
                "Insufficient liquidity for {}: {}", 
                asset.symbol, liquidity_status.reason
            )));
        }
        
        Ok(())
    }
    
    /// Performs statistical analysis on the asset
    fn validate_statistics(&self, asset: &Asset) -> Result<StatisticalReport, AssetError> {
        // Get price history for analysis
        let history = self.asset_repo.get_price_history(&asset.symbol, 30)?;
        
        // If there's not enough historical data, flag it but don't fail validation
        if history.len() < 10 {
            self.logger.log(
                LogLevel::Warning,
                LogCategory::Validation,
                &format!("Limited price history for {}: only {} data points", asset.symbol, history.len())
            );
        }
        
        // Perform statistical analysis
        let report = self.stats_analyzer.analyze_asset(asset, &history)?;
        
        // Check for significant statistical anomalies
        if report.z_score.abs() > self.thresholds.max_zscore_threshold {
            return Err(AssetError::StatisticalAnomaly(format!(
                "Statistical anomaly detected for {}: Z-score = {:.2}", 
                asset.symbol, report.z_score
            )));
        }
        
        // Check for volatility exceeding thresholds
        if report.volatility > self.config.get_f64("validation.max_volatility").unwrap_or(0.10) {
            self.logger.log(
                LogLevel::Warning,
                LogCategory::Validation,
                &format!("High volatility for {}: {:.2}%", asset.symbol, report.volatility * 100.0)
            );
        }
        
        // Additional time-series anomaly checks
        if report.anomaly_probability > 0.8 {
            return Err(AssetError::StatisticalAnomaly(format!(
                "Time-series anomaly detected for {}: probability = {:.2}", 
                asset.symbol, report.anomaly_probability
            )));
        }
        
        Ok(report)
    }
    
    /// Validates regulatory compliance
    fn validate_regulatory(&self, asset: &Asset) -> Result<(), AssetError> {
        // Check sanctioned entities and assets
        if let Ok(true) = self.regulatory.is_sanctioned(&asset.issuer) {
            return Err(AssetError::RegulatoryRestriction(format!(
                "Sanctioned issuer: {}", asset.issuer
            )));
        }
        
        // Check restricted asset classes based on jurisdiction
        if let Some(ref jurisdiction) = asset.jurisdiction {
            if !self.regulatory.is_asset_allowed_in_jurisdiction(
                &asset.asset_type, 
                jurisdiction
            )? {
                return Err(AssetError::RegulatoryRestriction(format!(
                    "Asset type {} not allowed in jurisdiction {}", 
                    asset.asset_type, jurisdiction
                )));
            }
        }
        
        // Check if asset is listed on a verified exchange
        if !self.regulatory.is_listed_on_verified_exchange(&asset.symbol)? {
            return Err(AssetError::RegulatoryRestriction(format!(
                "Asset {} not listed on verified exchange", asset.symbol
            )));
        }
        
        // Check for any trading restrictions
        if let Ok(Some(restriction)) = self.regulatory.get_trading_restrictions(&asset.symbol) {
            self.logger.log(
                LogLevel::Warning,
                LogCategory::Validation,
                &format!("Trading restriction for {}: {}", asset.symbol, restriction)
            );
            
            if restriction.severity == "high" {
                return Err(AssetError::RegulatoryRestriction(format!(
                    "High severity trading restriction: {}", restriction.reason
                )));
            }
        }
        
        // Check for relevant corporate actions
        if let Ok(Some(actions)) = self.asset_repo.get_corporate_actions(&asset.symbol) {
            for action in actions {
                if action.action_type == "merger" || action.action_type == "acquisition" {
                    return Err(AssetError::CorporateAction(format!(
                        "{} affected by {} on {}", 
                        asset.symbol, action.action_type, action.effective_date
                    )));
                }
            }
        }
        
        Ok(())
    }
    
    /// Validates data integrity using checksums and cross-reference checks
    fn validate_integrity(&self, asset: &Asset) -> Result<(), AssetError> {
        // Check data integrity via checksum (if provided)
        if let Some(checksum) = asset.checksum {
            let computed_checksum = self.compute_asset_checksum(asset);
            if checksum != computed_checksum {
                return Err(AssetError::DataTampering(format!(
                    "Checksum mismatch for {}: expected {}, got {}", 
                    asset.symbol, checksum, computed_checksum
                )));
            }
        }
        
        // Check digital signature if present
        if let Some(ref signature) = asset.signature {
            if !self.verify_signature(asset, signature) {
                return Err(AssetError::DataTampering(format!(
                    "Invalid digital signature for {}", asset.symbol
                )));
            }
        }
        
        // Validate all required fields are present
        self.validate_required_fields(asset)?;
        
        Ok(())
    }
    
    /// Cross-references asset data with multiple sources
    fn validate_cross_reference(&self, asset: &Asset) -> Result<(), AssetError> {
        // Get price data from multiple sources
        let sources = self.market_data.get_prices_from_multiple_sources(&asset.symbol)?;
        
        // Ensure minimum number of sources
        if sources.len() < self.thresholds.min_sources_required {
            return Err(AssetError::InsufficientDataSources(format!(
                "Insufficient price sources for {}: {} < {}", 
                asset.symbol, sources.len(), self.thresholds.min_sources_required
            )));
        }
        
        // Compare price against multiple sources
        let mut max_deviation = 0.0;
        let mut deviant_source = String::new();
        
        for source in &sources {
            let deviation = (asset.price - source.price).abs() / source.price;
            
            if deviation > max_deviation {
                max_deviation = deviation;
                deviant_source = source.source.clone();
            }
        }
        
        if max_deviation > self.thresholds.max_price_deviation {
            return Err(AssetError::DataMismatch(format!(
                "Price deviation exceeds threshold for {}: {:.2}% from {}", 
                asset.symbol, max_deviation * 100.0, deviant_source
            )));
        }
        
        // Cross-reference with official exchange data when available
        if let Ok(official_price) = self.market_data.get_official_price(&asset.symbol) {
            let official_deviation = (asset.price - official_price).abs() / official_price;
            
            if official_deviation > self.thresholds.max_price_deviation {
                return Err(AssetError::DataMismatch(format!(
                    "Price deviates {:.2}% from official exchange data for {}", 
                    official_deviation * 100.0, asset.symbol
                )));
            }
        }
        
        Ok(())
    }
    
    /// Validates that all required fields are present and correctly formatted
    fn validate_required_fields(&self, asset: &Asset) -> Result<(), AssetError> {
        // For equities, check for required fields
        if asset.asset_type == "equity" {
            if asset.market_cap.is_none() {
                self.logger.log(
                    LogLevel::Warning,
                    LogCategory::Validation,
                    &format!("Missing market cap for equity {}", asset.symbol)
                );
            }
            
            if asset.sector.is_none() || asset.industry.is_none() {
                self.logger.log(
                    LogLevel::Warning,
                    LogCategory::Validation,
                    &format!("Missing sector/industry for equity {}", asset.symbol)
                );
            }
        }
        
        // For options, check for required fields
        if asset.asset_type == "option" {
            if asset.strike_price.is_none() {
                return Err(AssetError::MissingData(format!(
                    "Missing strike price for option {}", asset.symbol
                )));
            }
            
            if asset.expiry.is_none() {
                return Err(AssetError::MissingData(format!(
                    "Missing expiry date for option {}", asset.symbol
                )));
            }
            
            if asset.option_type.is_none() {
                return Err(AssetError::MissingData(format!(
                    "Missing option type (call/put) for {}", asset.symbol
                )));
            }
        }
        
        // For futures, check for required fields
        if asset.asset_type == "futures" {
            if asset.expiry.is_none() {
                return Err(AssetError::MissingData(format!(
                    "Missing expiry date for futures contract {}", asset.symbol
                )));
            }
            
            if asset.underlying.is_none() {
                return Err(AssetError::MissingData(format!(
                    "Missing underlying asset for futures contract {}", asset.symbol
                )));
            }
        }
        
        Ok(())
    }
    
    /// Computes checksum for an asset to verify data integrity
    fn compute_asset_checksum(&self, asset: &Asset) -> u64 {
        // Use the Hasher utility to generate a hash of the asset's key fields
        let mut hasher = self.hasher.new_hasher();
        
        // Hash key fields to generate checksum
        hasher.update(asset.symbol.as_bytes());
        hasher.update(&asset.price.to_le_bytes());
        hasher.update(&asset.volume.to_le_bytes());
        hasher.update(asset.timestamp.to_rfc3339().as_bytes());
        
        if let Some(bid) = asset.bid {
            hasher.update(&bid.to_le_bytes());
        }
        
        if let Some(ask) = asset.ask {
            hasher.update(&ask.to_le_bytes());
        }
        
        hasher.finish()
    }
    
    /// Verifies digital signature of asset data
    fn verify_signature(&self, asset: &Asset, signature: &str) -> bool {
        // In a real implementation, this would use proper cryptographic verification
        // For this example, we'll just check if the signature is in a valid format
        
        // Check if signature has the correct format (example: base64 encoded)
        signature.len() >= 64 && signature.chars().all(|c| c.is_alphanumeric() || c == '+' || c == '/' || c == '=')
    }
    
    /// Verifies ISIN checksum
    fn verify_isin_checksum(&self, isin: &str) -> bool {
        // Example ISIN validation algorithm (simplified)
        // Real implementation would follow ISO 6166 standard
        
        if isin.len() != 12 {
            return false;
        }
        
        // Last digit is the check digit
        let check_digit = isin.chars().last().unwrap().to_digit(10).unwrap_or(0);
        
        // Calculate checksum (simplified example)
        let mut sum = 0;
        for (i, c) in isin.chars().take(11).enumerate() {
            let digit = match c.to_digit(36) {
                Some(d) => d,
                None => return false,
            };
            
            // Double every other digit
            let value = if i % 2 == 0 { digit } else { digit * 2 };
            
            // Sum the digits (if value > 9, sum its digits)
            sum += if value > 9 { value / 10 + value % 10 } else { value };
        }
        
        // Check if checksum matches
        (10 - (sum % 10)) % 10 == check_digit
    }
    
    /// Clears validation cache to force revalidation of assets
    pub fn clear_validation_cache(&mut self) {
        self.validated_cache.clear();
        self.logger.log(
            LogLevel::Info,
            LogCategory::Validation,
            "Validation cache cleared"
        );
    }
    
    /// Updates validation thresholds based on market conditions
    pub fn update_thresholds(&mut self, market_volatility: f64) {
        // Dynamically adjust thresholds based on market volatility
        let base_zscore = self.config.get_f64("validation.base_zscore_threshold").unwrap_or(3.0);
        let volatility_factor = self.config.get_f64("validation.volatility_adjustment").unwrap_or(0.5);
        
        // Adjust Z-score threshold based on market volatility
        // During high volatility, we allow larger price movements
        self.thresholds.max_zscore_threshold = base_zscore * (1.0 + market_volatility * volatility_factor);
        
        self.logger.log(
            LogLevel::Info,
            LogCategory::Validation,
            &format!("Updated Z-score threshold to {:.2} based on market volatility {:.2}%",
                    self.thresholds.max_zscore_threshold, market_volatility * 100.0)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use mockall::predicate::*;
    use mockall::mock;
    
    // Generate mock implementations for dependencies
    mock! {
        AssetRepository {}
        impl AssetRepository {
            fn get_price_history(&self, symbol: &str, days: i32) -> Result<Vec<f64>, AssetError>;
            fn get_corporate_actions(&self, symbol: &str) -> Result<Option<Vec<CorporateAction>>, AssetError>;
        }
    }
    
    mock! {
        MarketDataService {}
        impl MarketDataService {
            fn is_halted(&self, symbol: &str) -> Result<bool, AssetError>;
            fn is_market_open(&self, asset_type: &str) -> bool;
            fn is_circuit_breaker_active(&self, exchange: &str) -> bool;
            fn get_prices_from_multiple_sources(&self, symbol: &str) -> Result<Vec<PriceSource>, AssetError>;
            fn get_official_price(&self, symbol: &str) -> Result<f64, AssetError>;
        }
    }
    
    #[test]
    fn test_validate_asset_success() {
        // Set up test data and mocks
        let mut mock_repo = MockAssetRepository::new();
        mock_repo.expect_get_price_history()
            .with(eq("AAPL"), eq(30))
            .returning(|_, _| Ok(vec![150.0, 151.0, 152.0, 153.0, 154.0]));
            
        // Additional test implementation
    }
    
    #[test]
    fn test_validate_asset_invalid_price() {
        // Test with invalid price data
        // Implementation details
    }
    
    // Additional test cases for other validation scenarios
}