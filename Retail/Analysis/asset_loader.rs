use std::{collections::HashMap, path::Path, sync::Arc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, BufReader};
use anyhow::{Context, Result};
use sqlx::{SqlitePool, sqlite::SqliteConnectOptions};
use tracing::{info, warn, instrument};
use async_trait::async_trait;
use thiserror::Error;

/// Custom error type for asset loading operations
#[derive(Error, Debug)]
pub enum AssetError {
    #[error("Invalid asset data: {0}")]
    InvalidData(String),
    #[error("Database connection error")]
    DbConnection(#[from] sqlx::Error),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("Serialization error")]
    Serialization(#[from] serde_json::Error),
}

/// Financial asset structure with validation
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Asset {
    #[serde(rename = "sym")]
    symbol: String,
    name: String,
    #[serde(rename = "class")]
    asset_class: String,
    exchange: String,
    #[serde(default = "default_price")]
    price: f64,
    volume: u64,
    #[serde(skip)]
    metadata: HashMap<String, String>,
}

impl Asset {
    /// Validate asset data integrity
    pub fn validate(&self) -> Result<(), AssetError> {
        if self.price <= 0.0 {
            return Err(AssetError::InvalidData(
                format!("Invalid price for {}: {}", self.symbol, self.price)
            ));
        }
        if self.volume == 0 {
            return Err(AssetError::InvalidData(
                format!("Zero volume for {}", self.symbol)
            ));
        }
        Ok(())
    }
}

fn default_price() -> f64 {
    1.0
}

/// Thread-safe asset repository with async capabilities
#[derive(Debug, Clone)]
pub struct AssetRepository {
    assets: Arc<DashMap<String, Asset>>,
    db_pool: SqlitePool,
}

impl AssetRepository {
    /// Create new repository with database connection
    pub async fn new(db_url: &str) -> Result<Self> {
        let options = SqliteConnectOptions::new()
            .filename(db_url)
            .create_if_missing(true);

        let pool = SqlitePool::connect_with(options).await?;
        sqlx::migrate!().run(&pool).await?;

        Ok(Self {
            assets: Arc::new(DashMap::new()),
            db_pool: pool,
        })
    }

    /// Load assets from multiple sources in parallel
    #[instrument(skip(self))]
    pub async fn load_assets(&self, paths: &[&Path]) -> Result<()> {
        let mut handles = vec![];

        for path in paths {
            let path = path.to_path_buf();
            let repo = self.clone();
            
            handles.push(tokio::spawn(async move {
                match path.extension().and_then(|ext| ext.to_str()) {
                    Some("csv") => repo.load_csv(&path).await,
                    Some("json") => repo.load_json(&path).await,
                    _ => {
                        warn!("Unsupported file format: {:?}", path);
                        Ok(())
                    }
                }
            }));
        }

        futures::future::join_all(handles).await;
        self.load_db().await?;
        
        Ok(())
    }

    /// Load assets from CSV with streaming and parallel processing
    #[instrument(skip(self))]
    async fn load_csv(&self, path: &Path) -> Result<()> {
        let mut reader = csv_async::AsyncReaderBuilder::new()
            .flexible(true)
            .create_deserializer(BufReader::new(File::open(path).await?));

        let mut records = reader.deserialize::<Asset>();
        let mut batch = Vec::with_capacity(1000);

        while let Some(record) = records.next().await {
            let mut asset = record?;
            asset.validate()?;
            batch.push(asset);

            if batch.len() >= 1000 {
                self.process_batch(batch.drain(..).collect()).await;
            }
        }

        if !batch.is_empty() {
            self.process_batch(batch).await;
        }

        Ok(())
    }

    /// Process batch of assets with database insertion
    async fn process_batch(&self, batch: Vec<Asset>) {
        let repo = self.clone();
        tokio::spawn(async move {
            let mut transaction = repo.db_pool.begin().await.unwrap();
            
            for asset in batch {
                sqlx::query!(
                    r#"INSERT OR IGNORE INTO assets 
                    (symbol, name, class, exchange, price, volume)
                    VALUES (?, ?, ?, ?, ?, ?)"#,
                    asset.symbol,
                    asset.name,
                    asset.asset_class,
                    asset.exchange,
                    asset.price,
                    asset.volume as i64
                )
                .execute(&mut transaction)
                .await
                .unwrap();

                repo.assets.insert(asset.symbol.clone(), asset);
            }

            transaction.commit().await.unwrap();
        });
    }

    /// Load assets from JSON with streaming
    #[instrument(skip(self))]
    async fn load_json(&self, path: &Path) -> Result<()> {
        let mut file = File::open(path).await?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).await?;

        let assets: Vec<Asset> = serde_json::from_str(&contents)?;
        for asset in assets {
            asset.validate()?;
            self.assets.insert(asset.symbol.clone(), asset);
        }

        Ok(())
    }

    /// Load assets from database with caching
    #[instrument(skip(self))]
    async fn load_db(&self) -> Result<()> {
        let records = sqlx::query_as!(
            Asset,
            r#"SELECT 
                symbol, 
                name, 
                class as asset_class, 
                exchange, 
                price, 
                volume 
            FROM assets"#
        )
        .fetch_all(&self.db_pool)
        .await?;

        for asset in records {
            self.assets.insert(asset.symbol.clone(), asset);
        }

        Ok(())
    }

    /// Get asset with read-through cache
    pub async fn get_asset(&self, symbol: &str) -> Option<Asset> {
        if let Some(asset) = self.assets.get(symbol) {
            return Some(asset.value().clone());
        }

        let asset = sqlx::query_as!(
            Asset,
            r#"SELECT 
                symbol, 
                name, 
                class as asset_class, 
                exchange, 
                price, 
                volume 
            FROM assets 
            WHERE symbol = ?"#,
            symbol
        )
        .fetch_optional(&self.db_pool)
        .await
        .unwrap()?;

        self.assets.insert(symbol.to_string(), asset.clone());
        Some(asset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_asset_loading() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        
        let repo = AssetRepository::new(db_path.to_str().unwrap())
            .await
            .unwrap();

        let csv_path = dir.path().join("assets.csv");
        tokio::fs::write(&csv_path, "sym,name,class,exchange,price,volume\nAAPL,Apple Inc.,STOCK,NASDAQ,150.0,1000000")
            .await
            .unwrap();

        repo.load_assets(&[csv_path.as_path()])
            .await
            .unwrap();

        let asset = repo.get_asset("AAPL").await.unwrap();
        assert_eq!(asset.name, "Apple Inc.");
        assert_eq!(asset.price, 150.0);
    }
}