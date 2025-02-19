use std::fs::File;
use std::io::{self, BufRead, BufReader};
use serde::{Deserialize, Serialize};
use serde_json;
use csv::Reader;
use std::collections::HashMap;
use rayon::prelude::*; // For parallel processing
use log::{info, warn, error};
use env_logger;


/// Structure to represent a financial asset
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Asset {
    pub symbol: String,
    pub name: String,
    pub asset_class: String,  // Stocks, Crypto, Forex
    pub exchange: String,
    #[serde(default = "default_price")]  // ✅ Adds default value for missing price
    pub price: f64,
    pub volume: u64,
}

// Define a default value function for price
fn default_price() -> f64 {
    0.0
}

/// Asset Loader - Loads assets from various sources (CSV, JSON, Database)
pub struct AssetLoader {
    assets: HashMap<String, Asset>,
}

impl AssetLoader {
    /// Create a new AssetLoader instance
    pub fn new() -> Self {
        AssetLoader {
            assets: HashMap::new(),
        }
    }

    /// Load assets from a CSV file with improved error handling and parallel processing
    pub fn load_from_csv(&mut self, file_path: &str) -> Result<(), io::Error> {
        let mut reader = Reader::from_path(file_path)?;
        
        let assets: Vec<Asset> = reader.deserialize()
            .par_bridge()  // Converts the iterator to a parallel stream
            .filter_map(|result| result.ok())
            .collect();

        self.assets.extend(assets.into_iter().map(|asset| (asset.symbol.clone(), asset)));
        
        info!("✅ Loaded {} assets from CSV", self.assets.len());
        Ok(())
    }

    /// Load assets from a JSON file with improved error handling
    pub fn load_from_json(&mut self, file_path: &str) -> Result<(), io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let assets: Vec<Asset> = serde_json::from_reader(reader).unwrap_or_else(|_| vec![]);

        self.assets.extend(
            assets.into_par_iter().map(|asset| (asset.symbol.clone(), asset))
        );

        info!("✅ Loaded {} assets from JSON", self.assets.len());
        Ok(())
    }

    /// Load assets from an SQLite database
    pub fn load_from_db(&mut self, db_path: &str) -> Result<()> {
        let conn = Connection::open(db_path)?;
        let mut stmt = conn.prepare("SELECT symbol, name, asset_class, exchange, price, volume FROM assets")?;
        let asset_iter = stmt.query_map([], |row| {
            Ok(Asset {
                symbol: row.get(0)?,
                name: row.get(1)?,
                asset_class: row.get(2)?,
                exchange: row.get(3)?,
                price: row.get(4)?,
                volume: row.get(5)?,
            })
        })?;

        for asset in asset_iter {
            let asset = asset?;
            self.assets.insert(asset.symbol.clone(), asset);
        }

        info!("✅ Loaded {} assets from DB", self.assets.len());
        Ok(())
    }

    /// Fetch an asset by symbol with additional logging
    pub fn get_asset(&self, symbol: &str) -> Result<&Asset, String> {
        self.assets.get(symbol).ok_or_else(|| format!("❌ Asset {} not found.", symbol))
    }
    
    /// List all assets in sorted order by symbol
    pub fn list_assets(&self) -> Vec<&Asset> {
        let mut asset_list: Vec<&Asset> = self.assets.values().collect();
        asset_list.sort_by(|a, b| a.symbol.cmp(&b.symbol));
        asset_list
    }

    /// Filter assets by asset class
    pub fn filter_assets_by_class(&self, asset_class: &str) -> Vec<&Asset> {
        self.assets.values()
            .filter(|asset| asset.asset_class.eq_ignore_ascii_case(asset_class))
            .collect()
    }
}

fn main() {
    let mut loader = AssetLoader::new();

    // Example: Load assets from CSV and JSON
    if let Err(err) = loader.load_from_csv("assets.csv") {
        eprintln!("⚠️ Error loading CSV: {}", err);
    }
    if let Err(err) = loader.load_from_json("assets.json") {
        eprintln!("⚠️ Error loading JSON: {}", err);
    }
    // Example: Load assets from SQLite database
    if let Err(err) = loader.load_from_db("assets.db") {
        eprintln!("⚠️ Error loading database: {}", err);
    }

    // Fetch and print an asset
    if let Some(asset) = loader.get_asset("AAPL") {
        info!("📊 Asset found: {:?}", asset);
    }
}
