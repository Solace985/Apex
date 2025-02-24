use crate::Core::asset_repository::Asset;
use crate::utils::asset_error::AssetError;

pub fn validate_asset(asset: &Asset) -> Result<(), AssetError> {
    if asset.price <= 0.0 {
        return Err(AssetError::InvalidData(format!("Invalid price for {}", asset.symbol)));
    }
    if asset.volume == 0 {
        return Err(AssetError::InvalidData(format!("Zero volume for {}", asset.symbol)));
    }
    Ok(())
}
