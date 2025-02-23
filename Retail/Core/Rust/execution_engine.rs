use std::collections::{HashSet, HashMap};
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration, Instant};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use log::{info, warn, error, debug};
use tracing::{info as tracing_info, warn as tracing_warn, error as tracing_error, debug as tracing_debug};
use tracing_subscriber;
use thiserror::Error;
use rand::Rng;
use reqwest::Client;
use serde_json::json;

// Custom Error Types
#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("Max failures reached")]
    MaxFailures,
    #[error("Broker error: {0}")]
    BrokerError(#[from] Box<dyn std::error::Error>),
}

// Order State Enum
#[derive(Clone, Copy, PartialEq)]
pub enum OrderState {
    Pending,
    PartiallyFilled(f64),  // Track filled quantity
    Filled,
    Canceled,
    Failed(String),
}

// Rate Limiter
struct RateLimiter {
    semaphore: Arc<Semaphore>,
    max_requests: u32,
    interval: Duration,
}

impl RateLimiter {
    fn new(max_requests: u32, interval_secs: u64) -> Self {
        RateLimiter {
            semaphore: Arc::new(Semaphore::new(max_requests as usize)),
            max_requests,
            interval: Duration::from_secs(interval_secs),
        }
    }

    async fn allow_request(&self) -> bool {
        let permit = self.semaphore.acquire().await.ok();
        tokio::time::sleep(self.interval).await;
        permit.is_some()
    }

    async fn allow_request_with_adaptive_delay(&self, market_volatility: f64) -> bool {
        let base_delay = if market_volatility > 0.8 {
            1  // Aggressive trading (1 request/sec)
        } else if market_volatility > 0.5 {
            3  // Moderate trading
        } else {
            5  // Conservative trading
        };

        let permit = self.semaphore.acquire().await.ok();
        tokio::time::sleep(Duration::from_secs(base_delay)).await;
        permit.is_some()
    }
}

// Core Execution Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub amount: f64,
    pub price: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub confidence: Option<f64>,
}

pub struct ExecutionEngine<B: BrokerAPI + Send + Sync> {
    broker: B,
    risk_manager: Arc<dyn RiskManager>,
    rate_limiter: Arc<RateLimiter>,
    failed_order_count: AtomicU32,
    executed_orders: Arc<RwLock<HashSet<String>>>,
    config: ExecutionConfig,
    liquidity_manager: Arc<dyn LiquidityManager>,
    strategy_selector: Arc<dyn StrategySelector>,
    session_detector: Arc<dyn SessionDetector>,
    market_impact_analyzer: Arc<dyn MarketImpactAnalyzer>,
    order_states: HashMap<String, OrderState>, // Add order states tracking
}

impl<B: BrokerAPI + Send + Sync> ExecutionEngine<B> {
    pub fn new(
        broker: B,
        risk_manager: Arc<dyn RiskManager>,
        config: ExecutionConfig,
        liquidity_manager: Arc<dyn LiquidityManager>,
        strategy_selector: Arc<dyn StrategySelector>,
        session_detector: Arc<dyn SessionDetector>,
        market_impact_analyzer: Arc<dyn MarketImpactAnalyzer>,
    ) -> Self {
        ExecutionEngine {
            broker,
            risk_manager,
            rate_limiter: Arc::new(RateLimiter::new(10, 1)),
            failed_order_count: AtomicU32::new(0),
            executed_orders: Arc::new(Mutex::new(HashSet::new())),
            config,
            liquidity_manager,
            strategy_selector,
            session_detector,
            market_impact_analyzer,
            order_states: HashMap::new(), // Initialize order states
        }
    }

    // Add to impl block
    pub async fn update_order_state(&mut self, order_id: &str, new_state: OrderState) {
        self.order_states.insert(order_id.to_string(), new_state);
    }

    // ✅ Selects the best broker dynamically (Based on execution cost & speed)
    pub async fn select_best_broker(&self, symbol: &str) -> Result<&dyn BrokerAPI, ExecutionError> {
        let brokers = self.broker.get_all_brokers().await?;

        brokers.iter()
            .min_by_key(|b| {
                let execution_cost = b.get_execution_cost(symbol);
                let execution_latency = b.get_execution_latency(symbol);
                (execution_cost + execution_latency) as u64
            })
            .cloned()
            .ok_or(ExecutionError::BrokerError("No available brokers".into()))
    }

    pub async fn execute_order(&self, order: &Order) -> Result<(), ExecutionError> {
        self.validate_order(order)?;
        
        // Check duplicate execution
        if self.executed_orders.read().await.contains(&order.id) {
            warn!("Order {} already executed", order.id);
            return Ok(());
        }

        // Split large orders
        if order.amount > self.config.max_order_size {
            self.split_and_execute(order).await?;
            return Ok(());
        }

        tracing_info!(
            "🛒 Executing Order: ID={}, Symbol={}, Amount={}, Price={}, StopLoss={:?}, TakeProfit={:?}, Confidence={:?}",
            order.id,
            order.symbol,
            order.amount,
            order.price,
            order.stop_loss,
            order.take_profit,
            order.confidence
        );
        

        // Liquidity check
        let liquidity = self.liquidity_manager.get_liquidity(&order.symbol).await?;
        if liquidity < order.amount {
            warn!("Insufficient liquidity for {}", order.symbol);
            return Err(ExecutionError::InvalidOrder("Insufficient liquidity".into()));
        }

        // Whale activity detection
        if self.liquidity_manager.detect_whale_activity(&order.symbol).await? {
            warn!("Whale activity detected for {}", order.symbol);
            return Ok(());
        }

        // Trading session check
        let session = self.session_detector.current_session().await;
        if !session.is_active() {
            warn!("Market closed for session {:?}", session);
            return Ok(());
        }

        // Strategy selection
        let strategy = self.strategy_selector.select_strategy(&order.symbol).await?;
        let indicators = strategy.get_indicators();
        let fundamentals = strategy.get_fundamentals();

        // Sentiment analysis
        let sentiment = strategy.analyze_sentiment().await?;
        if !sentiment.should_execute() {
            warn!("Negative sentiment for {}", order.symbol);
            return Ok(());
        }

        // Market impact analysis
        let impact = self.market_impact_analyzer.analyze(order).await?;
        if impact.cost > self.config.max_impact_cost {
            warn!("High market impact cost: {}", impact.cost);
            return Ok(());
        }

        // Risk management
        if !self.risk_manager.validate(order).await? {
            warn!("Risk validation failed for {}", order.id);
            return Ok(());
        }

        // Rate limiting
        let market_volatility = self.get_market_volatility().await.unwrap_or(0.5);
        if !self.rate_limiter.allow_request_with_adaptive_delay(market_volatility).await {
                    warn!("Rate limit exceeded");
            return Err(ExecutionError::RateLimitExceeded);
        }

        // Execute order with retries
        self.execute_with_retries(order).await
    }

    pub async fn execute_trade(
        &self,
        symbol: &str,
        order_type: &str,
        volatility: f64,
        broker_api_url: &str,
        api_key: &str
    ) -> Result<(), ExecutionError> {
        // 🔹 Get position size from risk engine
        let position_size = self.risk_manager.calculate_position_size(symbol, volatility);

        if position_size <= 0.0 {
            error!("❌ Trade aborted due to risk engine restrictions.");
            return Err(ExecutionError::InvalidOrder("Position size is zero".to_string()));
        }

        // 🔹 Place order via send_order()
        self.send_order(symbol, order_type, position_size, None, broker_api_url, api_key).await?;

        Ok(())
    }

    async fn execute_with_retries(&self, order: &Order) -> Result<(), ExecutionError> {
        let mut attempts = 0;
        let max_retries = 3;
        let mut rng = rand::thread_rng();

        loop {
            match self.broker.place_order(order).await {
                Ok(_) => {
                    self.executed_orders.write().await.insert(order.id.clone());
                    self.failed_order_count.store(0, Ordering::SeqCst);
                    return Ok(());
                }
                Err(e) => {
                    self.failed_order_count.fetch_add(1, Ordering::SeqCst);
                    attempts += 1;
                    tracing_error!(
                        "❌ Order Execution Failed for Order ID={} after {} attempts. Error: {:?}",
                        order.id,
                        attempts,
                        e
                    );
                    self.failed_order_count.fetch_add(1, Ordering::SeqCst);
                    if self.failed_order_count.load(Ordering::SeqCst) >= self.config.max_failures {
                        tracing_error!("🚨 Max failure limit reached. Stopping order execution for safety.");
                        return Err(ExecutionError::MaxFailures);
                    }
                    

                    if attempts >= max_retries || self.failed_order_count.load(Ordering::SeqCst) >= self.config.max_failures {
                        error!("Failed to execute order {}: {}", order.id, e);
                        return Err(e.into());
                    }

                    let backoff = rng.gen_range(1..=5);
                    sleep(Duration::from_secs(backoff)).await;
                }
            }
        }
    }

    async fn split_and_execute(&self, order: &Order) -> Result<(), ExecutionError> {
        let parts = (order.amount / self.config.max_order_size).ceil() as usize;
        info!("Splitting order {} into {} parts", order.id, parts);

        for i in 0..parts {
            let mut split_order = order.clone();
            split_order.id = format!("{}-{}", order.id, i);
            let remaining_amount = order.amount - (i as f64 * self.config.max_order_size);
            split_order.amount = remaining_amount.min(self.config.max_order_size);
            self.execute_order(&split_order).await?;
        }

        Ok(())
    }

    fn validate_order(&self, order: &Order) -> Result<(), ExecutionError> {
        if order.amount <= 0.0 {
            return Err(ExecutionError::InvalidOrder("Invalid amount".into()));
        }

        if !self.config.allowed_symbols.contains(&order.symbol) {
            return Err(ExecutionError::InvalidOrder("Symbol not allowed".into()));
        }

        Ok(())
    }

    pub async fn execute_orders_concurrently(&self, orders: Vec<Order>) {
        let tasks: Vec<_> = orders.into_iter()
            .map(|order| {
                let execution_clone = self.clone();
                tokio::spawn(async move {
                    execution_clone.execute_order(&order).await
                })
            })
            .collect();

        for result in futures::future::join_all(tasks).await {
            if let Err(e) = result {
                tracing_error!("❌ Order Execution Failed: {:?}", e);
            }
        }
    }
    
    pub async fn execute_hft_orders(&self, orders: Vec<Order>) {
        let results: Vec<_> = orders.into_iter()
            .map(|order| {
                let execution_clone = self.clone();
                tokio::spawn(async move {
                    let order_id = order.id.clone();
                    let result = execution_clone.execute_order(&order).await;
                    if let Err(e) = &result {
                        tracing_error!("❌ HFT Order Failed: Order ID={}, Error={:?}", order_id, e);
                    }
                    result
                })
            })
            .collect();
    
        // ✅ Wait for all orders to finish & collect results
        let results = futures::future::join_all(results).await;
    
        // ✅ Retry failed orders
        let failed_orders: Vec<_> = results.iter().filter_map(|r| r.as_ref().err()).collect();
        if !failed_orders.is_empty() {
            tracing_warn!("⚠ Retrying {} failed HFT orders...", failed_orders.len());
            for failed_order in failed_orders {
                tokio::spawn(async move {
                    self.execute_order(&failed_order).await.unwrap_or_else(|e| {
                        tracing_error!("🚨 Retry Failed for Order {:?}: {:?}", failed_order, e);
                    });
                });
            }
        }
    
        let total_orders = results.len();
        let failed_count = results.iter().filter(|r| r.is_err()).count();
        if failed_count > 0 {
            tracing_warn!("⚠ HFT Execution Summary: {}/{} orders failed.", failed_count, total_orders);
        } else {
            tracing_info!("✅ HFT Execution Complete: All {} orders executed successfully.", total_orders);
        }
    }
    
    pub async fn send_order(
        &self,
        symbol: &str,
        order_type: &str,
        size: f64,
        price: Option<f64>,
        broker_api_url: &str,
        api_key: &str
    ) -> Result<(), ExecutionError> {
        let client = Client::new();
        let payload = json!({
            "symbol": symbol,
            "order_type": order_type,
            "size": size,
            "price": price
        });

        let response = client.post(broker_api_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await?;

        if response.status().is_success() {
            info!("✅ Order placed successfully: {:?}", payload);
            Ok(())
        } else {
            error!("❌ Order placement failed: {:?}", response.text().await?);
            Err(ExecutionError::BrokerError(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Order failed",
            ))))
        }
    }

    pub async fn start_hft_websocket(url: &str) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            match connect_async(Url::parse(url)?).await {
                Ok((mut ws_stream, _)) => {
                    tracing_info!("✅ Connected to HFT WebSocket: {}", url);
    
                    // 🔄 Send heartbeat every 30 seconds to maintain connection
                    let heartbeat = tokio::spawn(async move {
                        loop {
                            sleep(Duration::from_secs(30)).await;
                            if let Err(e) = ws_stream.send(Message::Text("ping".to_string())).await {
                                tracing_error!("⚠ WebSocket Heartbeat Failed: {:?}", e);
                                break;
                            }
                        }
                    });
    
                    while let Some(msg) = ws_stream.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                tracing_info!("📩 WebSocket Message: {}", text);
                            }
                            Err(e) => {
                                tracing_error!("⚠ WebSocket Error: {:?}. Retrying in 5s...", e);
                                heartbeat.abort();
                                sleep(Duration::from_secs(5)).await;
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing_error!("⚠ WebSocket Connection Failed: {:?}. Retrying in 5s...", e);
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
    
    // Additional methods for concurrent execution, monitoring, etc.
}

// Broker API Trait
#[async_trait]
pub trait BrokerAPI: Send + Sync {
    async fn place_order(&self, order: &Order) -> Result<(), ExecutionError>;
    async fn cancel_order(&self, order_id: &str) -> Result<(), ExecutionError>;
    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatus, ExecutionError>;
    async fn get_all_brokers(&self) -> Result<Vec<Box<dyn BrokerAPI>>, ExecutionError>;
    fn get_execution_cost(&self, symbol: &str) -> f64;
    fn get_execution_latency(&self, symbol: &str) -> f64;
}

// Supporting types and traits
pub struct ExecutionConfig {
    pub max_order_size: f64,
    pub allowed_symbols: HashSet<String>,
    pub max_impact_cost: f64,
    pub max_failures: u32,
}

#[async_trait]
pub trait RiskManager: Send + Sync {
    async fn validate(&self, order: &Order) -> Result<bool, ExecutionError>;
    async fn calculate_position_size(&self, symbol: &str, volatility: f64) -> f64;
}

#[async_trait]
pub trait LiquidityManager: Send + Sync {
    async fn get_liquidity(&self, symbol: &str) -> Result<f64, ExecutionError>;
    async fn detect_whale_activity(&self, symbol: &str) -> Result<bool, ExecutionError>;
}

#[async_trait]
pub trait StrategySelector: Send + Sync {
    async fn select_strategy(&self, symbol: &str) -> Result<Arc<dyn TradingStrategy>, ExecutionError>;
}

#[async_trait]
pub trait TradingStrategy: Send + Sync {
    async fn get_indicators(&self) -> Vec<String>;
    async fn get_fundamentals(&self) -> Vec<String>;
    async fn analyze_sentiment(&self) -> Result<SentimentResult, ExecutionError>;
}

// Other supporting types and implementations...

// Logger setup with rotating logs (like Python’s `logging.handlers.RotatingFileHandler`)
struct Logger {
    file: Mutex<std::fs::File>,
}

impl Logger {
    fn new(filename: &str) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .expect("Failed to create log file");

        Logger {
            file: Mutex::new(file),
        }
    }

    fn log(&self, message: &str) {
        let mut file = self.file.lock().unwrap();
        writeln!(file, "{}", message).unwrap();
    }
}

// Trade order structure
#[derive(Debug, Serialize, Deserialize, Clone)]
struct TradeOrder {
    order_id: String,
    symbol: String,
    quantity: f64,
    order_type: String,
    price: Option<f64>,
    status: String,
}

// Shared execution state across threads
#[derive(Clone)]
struct ExecutionState {
    orders: Arc<Mutex<HashMap<String, TradeOrder>>>,
    failed_order_count: Arc<AtomicUsize>,
}

impl ExecutionState {
    fn new() -> Self {
        ExecutionState {
            orders: Arc::new(Mutex::new(HashMap::new())),
            failed_order_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn add_order(&self, order: TradeOrder) {
        let mut orders = self.orders.lock().unwrap();
        orders.insert(order.order_id.clone(), order);
    }

    fn update_order_status(&self, order_id: &str, status: &str) {
        let mut orders = self.orders.lock().unwrap();
        if let Some(order) = orders.get_mut(order_id) {
            order.status = status.to_string();
        }
    }

    fn increment_failed_orders(&self) {
        self.failed_order_count.fetch_add(1, Ordering::SeqCst);
    }

    fn reset_failed_orders(&self) {
        self.failed_order_count.store(0, Ordering::SeqCst);
    }

    fn get_failed_orders(&self) -> usize {
        self.failed_order_count.load(Ordering::SeqCst)
    }
}

// WebSocket connection for real-time trade execution
async fn start_websocket_execution(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        match connect_async(Url::parse(url)?).await {
            Ok((mut ws_stream, _)) => {
                tracing_info!("✅ Connected to WebSocket: {}", url);

                while let Some(msg) = ws_stream.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            tracing_info!("📩 WebSocket Message: {}", text);
                        }
                        Err(e) => {
                            tracing_error!("⚠ WebSocket Error: {:?}. Retrying in 5s...", e);
                            sleep(Duration::from_secs(5)).await;
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                tracing_error!("⚠ WebSocket Connection Failed: {:?}. Retrying in 5s...", e);
                sleep(Duration::from_secs(5)).await;
            }
        }
    }
}


// Function to execute an order via REST API
async fn execute_order(order: TradeOrder, client: Client, execution_state: ExecutionState, logger: Arc<Logger>) -> Result<(), reqwest::Error> {
    let api_url = "https://api.broker.com/execute_trade";
    let response = client.post(api_url)
        .json(&order)
        .send()
        .await?;

    if response.status().is_success() {
        logger.log(&format!("Order executed successfully: {:?}", order.order_id));
        execution_state.update_order_status(&order.order_id, "Executed");
        execution_state.reset_failed_orders();
    } else {
        logger.log(&format!("Order execution failed: {:?}", order.order_id));
        execution_state.increment_failed_orders();
    }

    Ok(())
}

// Task to handle new trade orders
async fn order_handler(mut receiver: mpsc::Receiver<TradeOrder>, client: Client, execution_state: ExecutionState, logger: Arc<Logger>) {
    while let Some(order) = receiver.recv().await {
        let exec_state_clone = execution_state.clone();
        let client_clone = client.clone();
        let logger_clone = logger.clone();
        
        task::spawn(async move {
            let _ = execute_order(order, client_clone, exec_state_clone, logger_clone).await;
        });
    }
}
//Main Function
#[tokio::main]
async fn main() {
    // ✅ Initializes structured logging (tracing)
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let execution_state = ExecutionState::new();
    let client = Client::new();
    
    let logger = Arc::new(Logger::new("execution_log.txt"));
    let rate_limiter = RateLimiter::new(10);

    // ✅ Start WebSocket Execution in Background
    tokio::spawn(async move {
        let _ = start_websocket_execution("wss://api.broker.com/trading").await;
    });

    // ✅ Improved Logging
    tracing_info!("🚀 Execution Engine Initialized");

    sleep(Duration::from_secs(10)).await;
}
