use serde::{Serialize, Deserialize};
use chrono_tz::Tz;
use statrs::distribution::{Normal, StudentsT};
use crate::market_data::{MarketData, DataQuality};
use crate::risk_models::CorrelationMatrix;
use crate::strategy::StrategyLibrary;
use std::collections::{HashMap, HashSet};
use ring::signature::{self, Ed25519KeyPair, KeyPair};
use ring::signature::UnparsedPublicKey;
use thiserror::Error; // Add this line for error handling

#[derive(Debug, Error)] // Updated to use thiserror for error handling
pub enum AnalysisError {
    #[error("Insufficient data quality: {0}")]
    DataQuality(#[from] DataQualityError),
    
    #[error("Volatility modeling failed: {0}")]
    VolatilityAnalysis(#[from] VolatilityError),
    
    #[error("Liquidity assessment error: {0}")]
    LiquidityError(#[from] LiquidityError),
    
    #[error("ML engine failure: {0}")]
    MlEngineError(String),
    
    #[error("Crisis prediction timeout")]
    Timeout,
    
    #[error("Cross-asset correlation matrix invalid")]
    CorrelationMatrixError,
}

impl From<rayon::ThreadPoolBuildError> for AnalysisError {
    fn from(_: rayon::ThreadPoolBuildError) -> Self {
        AnalysisError::MlEngineError("Thread pool initialization failed".into())
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssetClass {
    Crypto {
        category: CryptoCategory,
        consensus_mechanism: Option<String>,
    },
    Equity {
        market_cap: f64,  // In USD billions
        sector: Sector,
        region: Region,
    },
    Forex {
        pair_type: ForexPairType,
        liquidity_tier: u8,
    },
    Commodity {
        category: CommodityCategory,
        delivery_mechanism: String,
    },
    Derivative {
        underlying: String,
        contract_type: DerivativeType,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetProfile {
    pub volatility_regimes: HashMap<String, VolatilityRegime>,
    pub allowed_strategies: StrategySet,
    pub market_microstructure: MarketMicrostructureProfile,
    pub regulatory_constraints: RegulatoryProfile,
    pub tax_considerations: TaxProfile,
    pub dna: AssetDNA,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetDNA {
    #[serde(with = "hex")]
    pub genetic_signature: Vec<u8>,
    pub volatility_clusters: Vec<Cluster>,
    pub liquidity_profile: LiquidityDNA,
    pub session_sensitivity: SessionProfile,
    pub crisis_behavior: CrisisResponse,
    pub cross_asset_linkages: CorrelationMatrix,
}

impl AssetDNA {
    pub fn sign(&self, private_key: &[u8]) -> Result<Vec<u8>, SignatureError> {
        let key_pair = Ed25519KeyPair::from_pkcs8(private_key)
            .map_err(|_| SignatureError::InvalidKey)?;
        
        let message = serde_json::to_vec(self)
            .map_err(|_| SignatureError::SerializationFailed)?;
        
        Ok(key_pair.sign(&message).as_ref().to_vec())
    }

    pub fn verify(&self, public_key: &[u8], signature: &[u8]) -> bool {
        let peer_public_key = UnparsedPublicKey::new(&signature::ED25519, public_key);
        let message = serde_json::to_vec(self).unwrap();
        
        peer_public_key.verify(&message, signature.as_ref()).is_ok()
    }
}

#[derive(Debug)]
pub enum SignatureError {
    InvalidKey,
    SerializationFailed,
    VerificationFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityDNA {
    pub obv_score: f64,
    pub vwap_deviation: f64,
    pub order_book_depth: OrderBookProfile,
    pub market_impact_cost: f64,
    pub flash_crash_resilience: f64,
}

impl AssetProfile {
    pub fn builder(class: AssetClass) -> AssetProfileBuilder {
        AssetProfileBuilder::new(class)
    }

    pub fn validate(&self) -> Result<(), ProfileError> {
        // Validate regulatory constraints
        if self.regulatory_constraints.restricted_jurisdictions.len() > 50 {
            return Err(ProfileError::Overconstrained);
        }
        
        // Check strategy compatibility
        if self.allowed_strategies.is_empty() {
            return Err(ProfileError::NoValidStrategies);
        }

        Ok(())
    }

    pub fn deep_validate(&self) -> Result<(), ProfileError> { // Enhanced profile validation
        self.validate()?;
        
        // Validate DNA integrity
        let recomputed_signature = self.ml_engine.generate_signature(&self.dna);
        if recomputed_signature != self.dna.genetic_signature {
            return Err(ProfileError::TamperedDna);
        }
        
        // Check volatility cluster validity
        let total_weight: f64 = self.dna.volatility_clusters
            .iter()
            .map(|c| c.weight)
            .sum();
            
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(ProfileError::InvalidVolatilityDistribution);
        }
        
        Ok(())
    }

    pub fn allowed_strategy(&self, strategy: &str) -> bool {
        self.allowed_strategies.contains(strategy)
    }

    pub fn crisis_correlation(&self, other: &AssetDNA) -> f64 {
        self.dna.crisis_behavior.correlation(&other.crisis_behavior)
    }
}

#[derive(Debug)]
pub enum ProfileError {
    Overconstrained,
    NoValidStrategies,
    InvalidTaxStructure,
    TamperedDna, // New variant added
    InvalidVolatilityDistribution, // New variant added
    ExpiredProfile,
    BacktestDivergence,
}

pub struct AssetProfiler {
    garch_models: Vec<Box<dyn VolatilityModel>>,
    liquidity_models: HashMap<String, Box<dyn LiquidityModel>>,
    regime_detector: ChangePointDetector,
    ml_engine: AssetGraphNN,
}

impl AssetProfiler {
    pub fn analyze(&self, data: &MarketData) -> Result<AssetDNA, AnalysisError> {
        // Validate data quality first
        data.validate_quality(DataQuality::ResearchGrade)?;

        // Parallel feature extraction
        let (volatility, liquidity, sessions, crisis) = rayon::join(
            || self.calculate_volatility_profile(data),
            || self.calculate_liquidity_dna(data),
            || self.detect_session_patterns(data),
            || self.analyze_crisis_behavior(data),
        );

        Ok(AssetDNA {
            genetic_signature: self.ml_engine.generate_signature(data),
            volatility_clusters: volatility?,
            liquidity_profile: liquidity?,
            session_sensitivity: sessions?,
            crisis_behavior: crisis?,
            cross_asset_linkages: self.ml_engine.predict_linkages(data),
        })
    }

    fn calculate_volatility_profile(&self, data: &MarketData) -> Result<Vec<Cluster>, VolatilityError> {
        // Ensemble volatility modeling
        let mut clusters = Vec::new();
        for model in &self.garch_models {
            clusters.extend(model.detect_regimes(data)?);
        }
        self.regime_detector.consensus_clusters(&clusters)
    }
}

// Add to AssetGraphNN implementation
impl AssetGraphNN {
    pub fn train(
        &mut self,
        training_data: &[MarketData],
        epochs: usize,
        learning_rate: f32,
    ) -> Result<(), TrainingError> {
        use burn::tensor;
        use burn::optim::Adam;
        
        let device = tensor::backend::Backend::device(&self.backend);
        let mut optimizer = Adam::new(learning_rate);
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for data in training_data {
                let inputs = self.preprocess(data);
                let targets = self.generate_targets(data);
                
                let loss = self.model.forward(inputs).backward();
                optimizer.step(&self.model.parameters());
                total_loss += loss.value();
            }
            
            println!("Epoch {}: Loss {:.4}", epoch, total_loss);
        }
        
        Ok(())
    }
}

pub struct AssetProfileBuilder {
    class: AssetClass,
    strategies: StrategySet,
    regulatory: RegulatoryProfile,
    tax: TaxProfile,
}

impl AssetProfileBuilder {
    pub fn new(class: AssetClass) -> Self {
        let mut builder = Self {
            class,
            strategies: StrategySet::default(),
            regulatory: RegulatoryProfile::default(),
            tax: TaxProfile::default(),
        };

        // Apply class-specific defaults
        match &builder.class {
            AssetClass::Crypto { .. } => {
                builder.strategies.insert("HFT".into());
                builder.strategies.insert("Arbitrage".into());
            },
            AssetClass::Equity { .. } => {
                builder.strategies.insert("ValueInvesting".into());
                builder.strategies.insert("DividendCapture".into());
            },
            _ => {}
        }

        builder
    }

    pub fn with_strategy(mut self, strategy: &str) -> Result<Self, StrategyError> {
        if StrategyLibrary::validate_compatibility(&self.class, strategy) {
            self.strategies.insert(strategy.into());
            Ok(self)
        } else {
            Err(StrategyError::Incompatible)
        }
    }

    pub fn build(self) -> Result<AssetProfile, ProfileError> {
        let profile = AssetProfile {
            volatility_regimes: HashMap::new(),
            allowed_strategies: self.strategies,
            market_microstructure: MarketMicrostructureProfile::default(),
            regulatory_constraints: self.regulatory,
            tax_considerations: self.tax,
            dna: AssetDNA::default(),
        };

        profile.validate()?;
        Ok(profile)
    }
}