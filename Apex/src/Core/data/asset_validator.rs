use crate::Config::config_loader::load_yaml_config;
use crate::utils::error_handler::ApiError;
use std::collections::HashSet;

/// Validate if a given asset pair exists in `asset_universe.yaml`
pub async fn validate_asset_pair(a1: &str, a2: &str) -> Result<(), ApiError> {
    // Load asset universe from configuration
    let asset_universe: HashSet<String> = load_yaml_config("assets/asset_universe.yaml")
        .map_err(|e| ApiError::DataValidationFailed(format!("Failed to load asset universe: {}", e)))?;

    // Check if both assets exist in the asset universe
    if !asset_universe.contains(a1) || !asset_universe.contains(a2) {
        return Err(ApiError::InvalidAssetPair(format!(
            "Invalid asset pair: {} - {}", a1, a2
        )));
    }

    Ok(())
}
