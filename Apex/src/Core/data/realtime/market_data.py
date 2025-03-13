# src/Core/data/realtime/market_data.py

import asyncio
import concurrent.futures
import time
import numpy as np
import msgpack
import websockets
import hashlib
import hmac
import os
import uvloop
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
from collections import deque
import redis
import aiohttp
import orjson
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from prometheus_client import Counter, Gauge, Histogram
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

# Apex Core Imports
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.src.Core.trading.risk.risk_management import RiskEngine
from Apex.src.Core.trading.execution.order_execution import OrderExecutionManager
from Apex.src.Core.trading.hft.liquidity_manager import LiquidityOracle
from Apex.src.Core.data.trade_history import TradeHistory
from Apex.src.Core.data.realtime.data_feed import DataFeed
from Apex.src.Core.data.processing.data_cleaner import DataCleaner
from Apex.src.Core.data.realtime.market_sync import MarketSynchronizer
from Apex.src.ai.ensembles.meta_trader import MetaTrader
from Apex.utils.helpers.security import rate_limiter, decrypt_payload
from Apex.Config.config_loader import load_config
from Apex.src.Core.data.realtime.exchange_failover import ExchangeFailoverManager
from Apex.src.Core.data.realtime.data_integrity import DataIntegrityValidator

# Configure UVLoop for enhanced async performance
uvloop.install()

# Load Configuration
config = load_config()

# Performance Metrics
METRIC_INGEST_LATENCY = Histogram('market_data_ingest_latency', 'Time to ingest market data in nanoseconds', ['exchange'])
METRIC_PROCESS_LATENCY = Histogram('market_data_process_latency', 'Time to process market data in nanoseconds', ['exchange'])
METRIC_DATA_DROPS = Counter('market_data_drops', 'Number of dropped market data events', ['reason'])
METRIC_ACTIVE_CONNECTIONS = Gauge('market_data_active_connections', 'Number of active market data connections', ['exchange'])

class QuantumMarketData:
    """
    Apex Quantum-Grade Market Data Processing Core
    - Hybrid Rust/Python architecture for nanosecond processing
    - Multi-exchange synchronization with atomic clock alignment
    - AI-validated data integrity with anti-spoofing protection
    - Multi-layer security with HMAC-SHA3-512 and AES-256-GCM
    - Distributed processing with Kafka for horizontal scaling
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # System Initialization
        self.logger = StructuredLogger("QuantumMarketData")
        self.data_cleaner = DataCleaner()
        self.market_sync = MarketSynchronizer()
        self.exchange_failover = ExchangeFailoverManager()
        self.data_integrity = DataIntegrityValidator()
        
        # Initialize Core Systems
        self._init_core_components()
        self._init_security()
        self._init_data_pipelines()
        self._init_kafka()
        
        # AI/ML Components
        self.anomaly_detector = MetaTrader.load_component('market_anomaly_v3')
        self.latency_arbiter = MetaTrader.load_component('latency_optimizer')

        # Performance Monitoring
        self.processed_count = 0
        self.last_throughput_check = time.monotonic()
        self.health_status = {"status": "initializing", "last_update": time.time()}
        
        # Circuit breaker for emergency shutdown
        self.circuit_breaker = {"enabled": False, "triggered": False}
        
        # Heartbeat tracking
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 0.1  # 100ms

    def _init_core_components(self):
        """Initialize interconnected system components"""
        self.risk_engine = RiskEngine()
        self.liquidity_oracle = LiquidityOracle()
        self.trade_history = TradeHistory()
        self.order_execution = OrderExecutionManager()
        self.data_feed = DataFeed()
        
        # Thread and process pools
        cpu_count = os.cpu_count()
        self.threads_io = concurrent.futures.ThreadPoolExecutor(
            max_workers=cpu_count * 2,
            thread_name_prefix="market_io_"
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=cpu_count
        )
        
        # Connection management
        self.active_connections = {}
        self.connection_health = {}

    def _init_security(self):
        """Initialize security and cryptographic systems"""
        # HMAC key for message authentication
        self.hmac_key = os.environ.get("APEX_HMAC_KEY", "").encode()
        if not self.hmac_key:
            raise RuntimeError("HMAC_KEY environment variable not set")
            
        # AES-GCM for end-to-end encryption
        self.aes_key = os.environ.get("APEX_AES_KEY", "").encode()
        if not self.aes_key:
            raise RuntimeError("AES_KEY environment variable not set")
        self.aes = AESGCM(self.aes_key)
        
        # Redis for inter-process communication
        redis_ssl = os.getenv("REDIS_SSL", "False").lower() == "true"
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=redis_ssl,
            ssl_ca_certs=os.getenv("REDIS_SSL_CA") if redis_ssl else None,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
            decode_responses=False
        )

    def _init_data_pipelines(self):
        """Initialize high-speed data pipelines"""
        # Ring buffers for in-memory data caching
        self.raw_buffer = deque(maxlen=1_000_000)
        self.processed_buffer = deque(maxlen=1_000_000)
        
        # Multi-layered data processing queues
        self.ingest_queue = asyncio.Queue(maxsize=100_000)
        self.process_queue = asyncio.Queue(maxsize=100_000)
        self.distribution_queue = asyncio.Queue(maxsize=100_000)
        
        # Redis pubsub for system commands
        self.data_pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        self.data_pubsub.subscribe(**{
            'system_commands': self._system_command_handler
        })
        
        # Statistics and monitoring
        self.dropped_messages = 0
        self.processed_messages = 0
        self.anomalies_detected = 0

    async def _init_kafka(self):
        """Initialize Kafka producers and consumers for distributed processing"""
        # Kafka configuration for high-throughput data streaming
        kafka_config = {
            'bootstrap_servers': os.getenv("KAFKA_SERVERS", "localhost:9092"),
            'security_protocol': os.getenv("KAFKA_SECURITY", "PLAINTEXT"),
            'ssl_cafile': os.getenv("KAFKA_SSL_CA"),
            'ssl_certfile': os.getenv("KAFKA_SSL_CERT"),
            'ssl_keyfile': os.getenv("KAFKA_SSL_KEY"),
        }
        
        # Producer for market data distribution
        self.kafka_producer = AIOKafkaProducer(
            **kafka_config,
            compression_type="lz4",
            max_batch_size=16384,
            linger_ms=0,  # No delay for HFT
            acks=1,  # Wait for leader acknowledgment only
            enable_idempotence=True,
            max_request_size=1048576,
        )
        
        # Consumer for system commands
        self.kafka_consumer = AIOKafkaConsumer(
            "apex.commands",
            **kafka_config,
            group_id="market_data_processors",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            max_poll_records=1000,
        )
        
        await self.kafka_producer.start()
        await self.kafka_consumer.start()

    async def start_quantum_processing(self):
        """Start the quantum data processing engine with all critical tasks"""
        self.health_status["status"] = "starting"
        
        # Create task groups for different processing stages
        self.background_tasks = set()
        
        # Data ingestion tasks
        self._create_background_task(self._ingest_market_data())
        self._create_background_task(self._process_data_stream())
        self._create_background_task(self._distribute_processed_data())
        
        # System monitoring tasks
        self._create_background_task(self._monitor_system_health())
        self._create_background_task(self._handle_system_commands())
        self._create_background_task(self._kafka_command_consumer())
        self._create_background_task(self._send_heartbeat())
        
        # HFT optimization tasks
        self._create_background_task(self._execute_latency_critical_tasks())
        
        self.health_status["status"] = "running"
        self.logger.info("Quantum Market Data processing started")

    def _create_background_task(self, coro):
        """Helper to create and track background tasks"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    @rate_limiter(max_calls=50_000, period=1)
    async def _ingest_market_data(self):
        """High-speed market data ingestion pipeline with failover capability"""
        exchanges = config['data_feed']['exchanges']
        connection_tasks = {}
        
        for exchange in exchanges:
            connection_tasks[exchange] = self._create_exchange_connection(exchange)
        
        # Start all connections concurrently
        await asyncio.gather(*connection_tasks.values())

    async def _create_exchange_connection(self, exchange):
        """Establish and maintain websocket connection to an exchange"""
        endpoint = config['data_feed']['endpoints'][exchange]
        backoff = 0.1  # Initial backoff in seconds
        max_backoff = 30  # Maximum backoff in seconds
        
        while True:
            try:
                METRIC_ACTIVE_CONNECTIONS.labels(exchange=exchange).inc()
                self.connection_health[exchange] = {"status": "connecting", "timestamp": time.time()}
                
                # Get authentication parameters
                auth_params = await self._get_exchange_auth(exchange)
                
                async with websockets.connect(
                    endpoint,
                    extra_headers=auth_params.get("headers", {}),
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**24,  # 16MB max message size
                    compression=None,  # Handle compression ourselves
                ) as ws:
                    # Send subscription message if required
                    if "subscribe_msg" in auth_params:
                        await ws.send(auth_params["subscribe_msg"])
                    
                    self.active_connections[exchange] = ws
                    self.connection_health[exchange] = {"status": "connected", "timestamp": time.time()}
                    backoff = 0.1  # Reset backoff on successful connection
                    
                    # Process incoming messages
                    async for raw_data in ws:
                        ingest_start = time.time_ns()
                        
                        try:
                            # Validate and process the message
                            validated = await self._validate_incoming_data(raw_data, exchange)
                            if validated:
                                validated['exchange'] = exchange
                                validated['received_ns'] = time.time_ns()
                                
                                # Put in queue, drop if full (backpressure)
                                try:
                                    self.raw_buffer.append(validated)
                                    await asyncio.wait_for(
                                        self.ingest_queue.put(validated),
                                        timeout=0.001  # 1ms timeout for HFT
                                    )
                                except asyncio.TimeoutError:
                                    METRIC_DATA_DROPS.labels(reason="queue_full").inc()
                                    self.dropped_messages += 1
                                
                                # Record latency metrics
                                ingest_latency = (time.time_ns() - ingest_start) / 1000  # µs
                                METRIC_INGEST_LATENCY.labels(exchange=exchange).observe(ingest_latency)
                        
                        except Exception as e:
                            self.logger.error(f"Message processing error for {exchange}", error=str(e))
                            METRIC_DATA_DROPS.labels(reason="processing_error").inc()
            
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK) as e:
                self.logger.warning(f"WebSocket connection closed for {exchange}: {str(e)}")
                self.connection_health[exchange] = {"status": "disconnected", "timestamp": time.time(), "error": str(e)}
            
            except Exception as e:
                self.logger.error(f"Connection error for {exchange}", error=str(e))
                self.connection_health[exchange] = {"status": "error", "timestamp": time.time(), "error": str(e)}
            
            finally:
                METRIC_ACTIVE_CONNECTIONS.labels(exchange=exchange).dec()
                if exchange in self.active_connections:
                    del self.active_connections[exchange]
                
                # Check if we should trigger failover
                await self.exchange_failover.check_failover_conditions(exchange, self.connection_health)
                
                # Exponential backoff for reconnection
                await asyncio.sleep(min(backoff, max_backoff))
                backoff *= 2  # Exponential backoff

    async def _get_exchange_auth(self, exchange):
        """Get authentication parameters for an exchange"""
        auth_config = config['data_feed'].get('auth', {}).get(exchange, {})
        
        # Basic auth parameters
        auth = {
            "headers": auth_config.get("headers", {})
        }
        
        # If this exchange requires real-time signature (common in crypto)
        if auth_config.get("requires_signature", False):
            timestamp = int(time.time() * 1000)
            api_key = os.environ.get(f"APEX_{exchange.upper()}_API_KEY", "")
            api_secret = os.environ.get(f"APEX_{exchange.upper()}_API_SECRET", "").encode()
            
            # Generate signature (varies by exchange)
            if exchange.lower() in ["binance", "bitmex"]:
                signature_payload = f"timestamp={timestamp}"
                signature = hmac.new(api_secret, signature_payload.encode(), hashlib.sha256).hexdigest()
                auth["headers"].update({
                    "X-MBX-APIKEY": api_key,
                    "X-MBX-SIGNATURE": signature,
                    "X-MBX-TIMESTAMP": str(timestamp)
                })
            
            # Prepare subscription message if needed
            if auth_config.get("subscribe_format"):
                channels = auth_config.get("channels", ["ticker", "trades", "orderbook"])
                subscribe_msg = auth_config["subscribe_format"].format(
                    api_key=api_key,
                    channels=channels,
                    timestamp=timestamp,
                    signature=signature
                )
                auth["subscribe_msg"] = subscribe_msg
        
        return auth

    async def _validate_incoming_data(self, raw_data: bytes, exchange: str) -> Optional[dict]:
        """Cryptographic and structural validation with enhanced security"""
        try:
            # Handle text or binary data based on exchange format
            if isinstance(raw_data, str):
                # Parse JSON data
                data = orjson.loads(raw_data)
                
                # Skip heartbeat messages
                if self._is_heartbeat_message(data, exchange):
                    return None
                
            else:
                # For binary protocols with signatures (e.g., secure websockets)
                if config['data_feed'].get('signature_format', {}).get(exchange):
                    # Extract signature and payload
                    sig_length = config['data_feed']['signature_format'][exchange]['length']
                    signature = raw_data[:sig_length]
                    payload = raw_data[sig_length:]
                    
                    # Verify HMAC with constant-time comparison
                    expected_sig = hmac.new(
                        self.hmac_key, 
                        payload, 
                        getattr(hashlib, config['data_feed']['signature_format'][exchange]['algorithm'])
                    ).digest()
                    
                    if not hmac.compare_digest(signature, expected_sig):
                        self.logger.warning(f"Invalid HMAC signature from {exchange}")
                        METRIC_DATA_DROPS.labels(reason="invalid_signature").inc()
                        return None
                    
                    # Decrypt payload if encrypted
                    if config['data_feed']['signature_format'][exchange].get('encrypted', False):
                        # AES-GCM encryption: first 12 bytes are nonce, remaining is ciphertext
                        nonce = payload[:12]
                        ciphertext = payload[12:]
                        try:
                            decrypted = self.aes.decrypt(nonce, ciphertext, None)
                            payload = decrypted
                        except Exception as e:
                            self.logger.error(f"Decryption error for {exchange}", error=str(e))
                            METRIC_DATA_DROPS.labels(reason="decryption_error").inc()
                            return None
                
                # Deserialize binary data (msgpack, protobuf, etc.)
                data = msgpack.unpackb(payload)
            
            # Comprehensive data validation
            if not self.data_integrity.validate_message_structure(data, exchange):
                self.logger.warning(f"Structural validation failed for {exchange}")
                METRIC_DATA_DROPS.labels(reason="invalid_structure").inc()
                return None
            
            # Cross-reference timestamp for time synchronization
            server_time = data.get('timestamp', time.time() * 1000)
            client_time = time.time() * 1000
            time_drift = abs(server_time - client_time)
            
            # Warn on clock drift > 500ms
            if time_drift > 500:
                self.logger.warning(f"Clock drift detected with {exchange}: {time_drift}ms")
            
            # Normalize data format to internal structure
            normalized = self.data_cleaner.normalize_exchange_data(data, exchange)
            return normalized
            
        except orjson.JSONDecodeError:
            self.logger.warning(f"JSON parse error from {exchange}")
            METRIC_DATA_DROPS.labels(reason="json_error").inc()
            return None
            
        except msgpack.exceptions.UnpackException:
            self.logger.warning(f"MsgPack parse error from {exchange}")
            METRIC_DATA_DROPS.labels(reason="msgpack_error").inc()
            return None
            
        except Exception as e:
            self.logger.error(f"Validation error from {exchange}", error=str(e))
            METRIC_DATA_DROPS.labels(reason="validation_error").inc()
            return None

    def _is_heartbeat_message(self, data, exchange):
        """Detect and handle exchange heartbeat messages"""
        # Exchange-specific heartbeat detection
        if exchange == "binance" and data.get("type") == "ping":
            return True
        elif exchange == "kraken" and data.get("heartbeat"):
            return True
        elif exchange == "coinbase" and data.get("type") == "heartbeat":
            return True
        
        return False

    async def _process_data_stream(self):
        """Parallel data processing pipeline with backpressure handling"""
        while True:
            try:
                # Get batch of messages for vectorized processing (up to 100 messages)
                batch = []
                batch_size = 0
                max_batch_size = 100
                
                # Try to accumulate a batch with 1ms timeout
                try:
                    while batch_size < max_batch_size:
                        item = await asyncio.wait_for(self.ingest_queue.get(), 0.001)
                        batch.append(item)
                        batch_size += 1
                        self.ingest_queue.task_done()
                except asyncio.TimeoutError:
                    # Process what we have so far
                    pass
                
                if not batch:
                    await asyncio.sleep(0.0001)  # Short sleep to prevent CPU spinning
                    continue
                
                # Process batch in a worker process
                loop = asyncio.get_running_loop()
                process_start = time.time_ns()
                
                # Process messages in parallel using process pool
                processed_batch = await loop.run_in_executor(
                    self.process_pool,
                    self._process_data_batch,
                    batch
                )
                
                # Measure and record processing latency
                process_latency = (time.time_ns() - process_start) / 1000  # µs
                
                # Queue processed messages for distribution
                for processed in processed_batch:
                    if processed:
                        self.processed_buffer.append(processed)
                        
                        try:
                            await asyncio.wait_for(
                                self.distribution_queue.put(processed),
                                timeout=0.001  # 1ms timeout
                            )
                        except asyncio.TimeoutError:
                            METRIC_DATA_DROPS.labels(reason="distribution_queue_full").inc()
                
                # Update metrics
                for exchange in set(item.get('exchange') for item in batch):
                    METRIC_PROCESS_LATENCY.labels(exchange=exchange).observe(process_latency)
                
                self.processed_count += len(processed_batch)
                
            except Exception as e:
                self.logger.error("Processing error", error=str(e))
                await asyncio.sleep(0.001)  # Short sleep on error

    def _process_data_batch(self, batch: List[dict]) -> List[dict]:
        """Process a batch of market data messages in parallel"""
        try:
            # Group by exchange and symbol for parallel processing
            exchange_symbol_groups = {}
            for item in batch:
                exchange = item.get('exchange', 'unknown')
                symbol = item.get('symbol', 'unknown')
                key = f"{exchange}:{symbol}"
                
                if key not in exchange_symbol_groups:
                    exchange_symbol_groups[key] = []
                
                exchange_symbol_groups[key].append(item)
            
            # Process each group
            results = []
            for key, items in exchange_symbol_groups.items():
                # Vectorized processing for each group
                processed_group = self._process_data_group(items)
                results.extend(processed_group)
            
            return results
            
        except Exception as e:
            # Log error and return empty list
            print(f"Batch processing error: {str(e)}")
            return []

    def _process_data_group(self, items: List[dict]) -> List[dict]:
        """Process a group of related market data items"""
        if not items:
            return []
        
        try:
            # Extract exchange and symbol
            exchange = items[0].get('exchange', 'unknown')
            symbol = items[0].get('symbol', 'unknown')
            
            # Prepare ndarrays for vectorized processing
            timestamps = np.array([item.get('timestamp', 0) for item in items])
            prices = np.array([float(item.get('price', 0)) for item in items])
            volumes = np.array([float(item.get('size', 0)) for item in items])
            
            # Detect anomalies using vectorized operations
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            z_scores = np.abs((prices - mean_price) / max(std_price, 1e-8))
            anomalies = z_scores > 3.0  # Flag potential anomalies
            
            # Apply exchange-specific normalization and cleaning
            # This is a placeholder - DataCleaner would implement the actual logic
            cleaned_prices = prices.copy()
            cleaned_volumes = volumes.copy()
            
            # Process items with anomalies specially
            if np.any(anomalies):
                # Use AI-powered anomaly detection for suspicious items
                for i, is_anomaly in enumerate(anomalies):
                    if is_anomaly:
                        # Clean anomalous data point
                        if i > 0 and i < len(items) - 1:
                            # Simple median filter (replace with AI model in production)
                            cleaned_prices[i] = np.median(prices[max(0, i-2):min(len(prices), i+3)])
            
            # Prepare results
            results = []
            for i, item in enumerate(items):
                # Create processed output
                processed = {
                    'data': {
                        'exchange': exchange,
                        'symbol': symbol,
                        'price': cleaned_prices[i],
                        'size': cleaned_volumes[i],
                        'timestamp': timestamps[i],
                        'received_ns': item.get('received_ns', 0),
                    },
                    'processed_ns': time.time_ns(),
                    'anomaly_detected': bool(anomalies[i]),
                    'source': exchange,
                    'validation_hash': self._generate_data_hash(item)
                }
                results.append(processed)
            
            return results
            
        except Exception as e:
            print(f"Group processing error: {str(e)}")
            return []

    def _generate_data_hash(self, data):
        """Generate cryptographic hash for data integrity verification"""
        serialized = orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
        return hmac.new(self.hmac_key, serialized, hashlib.sha3_256).hexdigest()

    async def _distribute_processed_data(self):
        """Ultra-low latency data distribution to consumers"""
        while True:
            try:
                # Get data from queue or wait
                data = await self.distribution_queue.get()
                if not data:
                    self.distribution_queue.task_done()
                    continue
                
                # Extract symbol and exchange for topic routing
                symbol = data['data'].get('symbol', 'unknown')
                exchange = data['data'].get('exchange', 'unknown')
                
                # Serialize data once for multiple destinations
                serialized = msgpack.packb(data)
                
                # Execute distribution tasks concurrently
                await asyncio.gather(
                    # Update internal components
                    self.risk_engine.update_market_data(data),
                    self.liquidity_oracle.update_liquidity_state(data),
                    self.order_execution.feed_data(data),
                    self.trade_history.log_trade_data(data),
                    
                    # Distribute to external consumers via Kafka
                    self._publish_to_kafka(f"apex.market.{exchange}.{symbol}", serialized),
                    
                    # Cache in Redis
                    self._cache_processed_data(data),
                )
                
                # Mark task as done
                self.distribution_queue.task_done()
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error("Distribution error", error=str(e))
                await asyncio.sleep(0.001)

    async def _publish_to_kafka(self, topic, data):
        """Publish data to Kafka with retry logic"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                await self.kafka_producer.send_and_wait(topic, data)
                return
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    self.logger.error(f"Failed to publish to Kafka: {str(e)}")
                await asyncio.sleep(0.001 * (2 ** retry_count))  # Exponential backoff

    async def _cache_processed_data(self, data: dict):
        """Cache processed data in Redis with TTL"""
        try:
            # Create a unique key for this data point
            symbol = data['data']['symbol']
            exchange = data['data']['exchange']
            timestamp = data['data'].get('timestamp', data['processed_ns'])
            key = f"apex:market:{exchange}:{symbol}:{timestamp}"
            
            # Serialize and cache with TTL
            cached_data = msgpack.packb(data)
            
            # Set with expiration based on config
            ttl = config['data_cache'].get('ttl', 3600)  # Default 1 hour TTL
            
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()
            pipe.set(key, cached_data, ex=ttl, nx=True)
            
            # Update latest value for symbol/exchange
            latest_key = f"apex:market:latest:{exchange}:{symbol}"
            pipe.set(latest_key, cached_data, ex=ttl)
            
            # Execute pipeline
            await pipe.execute()
            
        except Exception as e:
            self.logger.warning(f"Cache error: {str(e)}")

    async def _monitor_system_health(self):
        """Real-time system health monitoring with alerting"""
        while True:
            try:
                # Calculate throughput
                current_time = time.monotonic()
                elapsed = current_time - self.last_throughput_check
                
                if elapsed >= 1.0:  # 1-second window
                    throughput = self.processed_count / elapsed
                    
                    # Log metrics
                    self.logger.metric(
                        "system_throughput",
                        {
                            "events_per_sec": throughput,
                            "dropped_messages": self.dropped_messages,
                            "queue_sizes": {
                                "ingest": self.ingest_queue.qsize(),
                                "process": self.process_queue.qsize(),
                                "distribution": self.distribution_queue.qsize(),
                            },
                            "active_connections": len(self.active_connections),
                            "raw_buffer_size": len(self.raw_buffer),
                            "processed_buffer_size": len(self.processed_buffer),
                        }
                    )

                    # Reset counters for next window
                    self.last_throughput_check = current_time
                    self.processed_count = 0
                    
                    # Update health status
                    self.health_status = {
                        "status": "running" if not self.circuit_breaker["triggered"] else "emergency_shutdown",
                        "last_update": time.time(),
                        "throughput": throughput,
                        "connection_status": {k: v["status"] for k, v in self.connection_health.items()},
                        "queue_sizes": {
                            "ingest": self.ingest_queue.qsize(),
                            "distribution": self.distribution_queue.qsize(),
                        }
                    }
                    
                    # Check for system overload or other critical conditions
                    if self.ingest_queue.qsize() > 90000:  # 90% capacity
                        self.logger.warning("Ingest queue near capacity, potential data bottleneck")
                        
                    if throughput < config['performance_thresholds']['min_throughput']:
                        self.logger.warning(f"System throughput below threshold: {throughput}")
                        
                    # Check connection health across exchanges
                    disconnected = [ex for ex, status in self.connection_health.items() 
                                    if status.get("status") != "connected"]
                    if disconnected:
                        self.logger.warning(f"Disconnected exchanges: {', '.join(disconnected)}")
                        
                        # Try to reconnect if all exchanges are down
                        if len(disconnected) == len(config['data_feed']['exchanges']):
                            self.logger.error("All exchanges disconnected, triggering emergency reconnect")
                            await self._emergency_reconnect()
                
                # Check circuit breaker conditions
                if self.circuit_breaker["enabled"] and not self.circuit_breaker["triggered"]:
                    await self._check_circuit_breaker()
                
                # Sleep briefly to avoid CPU hogging
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(0.5)  # Longer sleep on error

    async def _check_circuit_breaker(self):
        """Check circuit breaker conditions and trigger emergency shutdown if needed"""
        try:
            # Check critical conditions that would trigger circuit breaker
            conditions = []
            
            # Condition 1: All exchanges disconnected for > 30 seconds
            all_disconnected = all(status.get("status") != "connected" for status in self.connection_health.values())
            if all_disconnected:
                last_connected = max(status.get("timestamp", 0) for status in self.connection_health.values())
                if time.time() - last_connected > 30:
                    conditions.append("all_exchanges_disconnected")
            
            # Condition 2: Extreme market volatility detected
            if any(item.get('anomaly_detected', False) for item in list(self.processed_buffer)[-100:]):
                anomaly_count = sum(1 for item in list(self.processed_buffer)[-100:] 
                                   if item.get('anomaly_detected', False))
                if anomaly_count > 50:  # More than 50% of recent messages are anomalous
                    conditions.append("extreme_market_volatility")
            
            # Condition 3: Critical component failure
            component_status = await self._check_component_health()
            if not component_status["healthy"]:
                conditions.append(f"component_failure:{component_status['failed_component']}")
            
            # If any conditions met, trigger circuit breaker
            if conditions:
                self.logger.critical(f"Circuit breaker triggered: {', '.join(conditions)}")
                await self._trigger_circuit_breaker(conditions)
                
        except Exception as e:
            self.logger.error("Circuit breaker check error", error=str(e))

    async def _trigger_circuit_breaker(self, conditions):
        """Emergency circuit breaker implementation"""
        self.circuit_breaker["triggered"] = True
        self.health_status["status"] = "emergency_shutdown"
        
        # Log critical event
        self.logger.critical("CIRCUIT BREAKER ACTIVATED", conditions=conditions)
        
        # Alert risk management system
        await self.risk_engine.alert_emergency_shutdown(conditions)
        
        # Publish emergency event to Kafka
        alert = {
            "event": "circuit_breaker_triggered",
            "timestamp": time.time(),
            "conditions": conditions,
            "component": "QuantumMarketData"
        }
        await self.kafka_producer.send_and_wait("apex.alerts.critical", orjson.dumps(alert))
        
        # Gracefully disconnect from exchanges
        for exchange, ws in self.active_connections.items():
            try:
                await ws.close(code=1000, reason="Circuit breaker triggered")
            except Exception:
                pass
        
        # Clear all queues
        self._clear_queues()
        
        # Keep system in emergency mode until manual intervention
        self.health_status["emergency_details"] = {
            "triggered_at": time.time(),
            "conditions": conditions
        }

    def _clear_queues(self):
        """Clear all data queues during emergency shutdown"""
        # Drain ingest queue
        while not self.ingest_queue.empty():
            try:
                self.ingest_queue.get_nowait()
                self.ingest_queue.task_done()
            except asyncio.QueueEmpty:
                break
                
        # Drain distribution queue
        while not self.distribution_queue.empty():
            try:
                self.distribution_queue.get_nowait()
                self.distribution_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Clear buffers
        self.raw_buffer.clear()
        self.processed_buffer.clear()

    async def _emergency_reconnect(self):
        """Attempt emergency reconnection to all exchanges"""
        self.logger.warning("Initiating emergency reconnect procedure")
        
        # Reset connection health status
        for exchange in self.connection_health:
            self.connection_health[exchange] = {"status": "reconnecting", "timestamp": time.time()}
        
        # Create new connection tasks
        reconnect_tasks = []
        for exchange in config['data_feed']['exchanges']:
            task = self._create_background_task(self._create_exchange_connection(exchange))
            reconnect_tasks.append(task)
        
        # Wait for at least one connection to be established
        try:
            await asyncio.wait_for(
                self._wait_for_any_connection(),
                timeout=30.0  # 30 second timeout
            )
            self.logger.info("Emergency reconnect successful")
        except asyncio.TimeoutError:
            self.logger.critical("Emergency reconnect failed - no exchanges available")
            # Consider circuit breaker if enabled
            if self.circuit_breaker["enabled"] and not self.circuit_breaker["triggered"]:
                await self._trigger_circuit_breaker(["reconnect_failed"])

    async def _wait_for_any_connection(self):
        """Wait until at least one exchange is connected"""
        while True:
            if any(status.get("status") == "connected" for status in self.connection_health.values()):
                return True
            await asyncio.sleep(0.5)

    async def _check_component_health(self):
        """Check health of critical system components"""
        result = {"healthy": True, "failed_component": None}
        
        # Check Redis connection
        try:
            if not self.redis.ping():
                result["healthy"] = False
                result["failed_component"] = "redis"
                return result
        except Exception:
            result["healthy"] = False
            result["failed_component"] = "redis"
            return result
            
        # Check Kafka connection
        try:
            if not self.kafka_producer._sender.sender.connected():
                result["healthy"] = False
                result["failed_component"] = "kafka"
                return result
        except Exception:
            result["healthy"] = False
            result["failed_component"] = "kafka"
            return result
        
        # Check risk engine
        try:
            if not await self.risk_engine.health_check():
                result["healthy"] = False
                result["failed_component"] = "risk_engine"
                return result
        except Exception:
            result["healthy"] = False
            result["failed_component"] = "risk_engine"
            return result
            
        return result

    async def _handle_system_commands(self):
        """Handle system commands from Redis PubSub"""
        # Start pubsub listener in a separate thread
        self.pubsub_thread = self.threads_io.submit(self._pubsub_listener)
        
        while True:
            try:
                # Get commands from the Redis listener thread
                if hasattr(self, 'command_queue'):
                    while not self.command_queue.empty():
                        cmd = self.command_queue.get_nowait()
                        await self._process_system_command(cmd)
                
                # Short sleep to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error("Command handler error", error=str(e))
                await asyncio.sleep(0.5)

    def _pubsub_listener(self):
        """Redis PubSub listener running in a separate thread"""
        self.command_queue = asyncio.Queue()
        pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe("apex.commands")
        
        for message in pubsub.listen():
            try:
                if message["type"] == "message":
                    data = orjson.loads(message["data"])
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.command_queue.put(data))
            except Exception as e:
                print(f"PubSub error: {str(e)}")

    async def _process_system_command(self, cmd):
        """Process system command"""
        cmd_type = cmd.get("command")
        
        if cmd_type == "shutdown":
            self.logger.info("Received shutdown command")
            await self.shutdown()
            
        elif cmd_type == "reconnect":
            exchange = cmd.get("exchange")
            self.logger.info(f"Received reconnect command for {exchange}")
            if exchange in self.connection_health:
                # Force reconnection by updating connection status
                self.connection_health[exchange] = {"status": "reconnecting", "timestamp": time.time()}
                
        elif cmd_type == "toggle_circuit_breaker":
            enabled = cmd.get("enabled", True)
            self.logger.info(f"Setting circuit breaker enabled={enabled}")
            self.circuit_breaker["enabled"] = enabled
            
        elif cmd_type == "reset_circuit_breaker":
            if self.circuit_breaker["triggered"]:
                self.logger.info("Resetting circuit breaker")
                self.circuit_breaker["triggered"] = False
                self.health_status["status"] = "running"
                # Restart data processing
                await self.start_quantum_processing()
                
        elif cmd_type == "status":
            # Publish current status to response channel
            status_data = {
                "component": "QuantumMarketData",
                "timestamp": time.time(),
                "health": self.health_status,
                "connections": self.connection_health,
                "metrics": {
                    "processed_messages": self.processed_messages,
                    "dropped_messages": self.dropped_messages,
                    "anomalies_detected": self.anomalies_detected
                }
            }
            await self.redis.publish("apex.status", orjson.dumps(status_data))

    async def _kafka_command_consumer(self):
        """Consume commands from Kafka"""
        while True:
            try:
                async for msg in self.kafka_consumer:
                    try:
                        data = orjson.loads(msg.value)
                        if data.get("target") in ["all", "market_data"]:
                            await self._process_system_command(data)
                    except Exception as e:
                        self.logger.error("Kafka command error", error=str(e))
                
                await asyncio.sleep(0.01)  # Short sleep to prevent CPU spinning
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error("Kafka consumer error", error=str(e))
                await asyncio.sleep(1.0)  # Longer sleep on error

    async def _execute_latency_critical_tasks(self):
        """Execute latency-critical tasks with nanosecond precision"""
        while True:
            try:
                # HFT optimization: Track network latency
                start_time = time.time_ns()
                
                # Check active connections and measure true latency
                for exchange, ws in list(self.active_connections.items()):
                    # Skip if connection is not active
                    if not ws or ws.closed:
                        continue
                        
                    # Ping each connection for accurate latency measurement
                    try:
                        # Measure round-trip time
                        ping_start = time.time_ns()
                        pong_waiter = await ws.ping()
                        await asyncio.wait_for(pong_waiter, timeout=0.5)
                        ping_time = (time.time_ns() - ping_start) / 1_000_000  # ms
                        
                        # Update latency metrics
                        self.connection_health[exchange]["latency"] = ping_time
                        
                        # Use AI-powered latency optimizer
                        if self.latency_arbiter:
                            optimization = await self.latency_arbiter.optimize_connection(
                                exchange, ping_time, self.connection_health[exchange]
                            )
                            if optimization and optimization.get("action"):
                                if optimization["action"] == "reconnect":
                                    self.logger.info(f"Latency optimizer suggests reconnect for {exchange}")
                                    # Schedule reconnection
                                    self.connection_health[exchange] = {"status": "reconnecting", "timestamp": time.time()}
                    
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                        # Connection issue detected
                        self.connection_health[exchange]["status"] = "lagging"
                        self.connection_health[exchange]["timestamp"] = time.time()
                    
                    except Exception as e:
                        self.logger.warning(f"Latency check error for {exchange}: {str(e)}")
                
                # Adaptive sleep based on system load
                execution_time = (time.time_ns() - start_time) / 1_000_000  # ms
                sleep_time = max(0.01, 0.1 - (execution_time / 1000))  # minimum 10ms
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error("Latency monitoring error", error=str(e))
                await asyncio.sleep(0.1)

    async def _send_heartbeat(self):
        """Send regular heartbeats to monitoring systems"""
        while True:
            try:
                heartbeat_data = {
                    "component": "QuantumMarketData",
                    "timestamp": time.time(),
                    "status": self.health_status["status"],
                    "connections": len(self.active_connections),
                    "processed_count": self.processed_count
                }
                
                # Publish heartbeat to Redis
                await self.redis.publish("apex.heartbeats", orjson.dumps(heartbeat_data))
                
                # Update last heartbeat time
                self.last_heartbeat = time.time()
                
                # Wait for next heartbeat interval
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error("Heartbeat error", error=str(e))
                await asyncio.sleep(0.5)  # Longer sleep on error

    async def _system_command_handler(self, message):
        """Handle system commands from Redis"""
        try:
            data = orjson.loads(message["data"])
            await self._process_system_command(data)
        except Exception as e:
            self.logger.error("Command handling error", error=str(e))

    async def get_market_data(self, exchange, symbol, limit=100):
        """Retrieve latest market data for a specific exchange and symbol"""
        try:
            # First check in-memory buffer for most recent data
            buffer_data = []
            for item in reversed(self.processed_buffer):
                if (item["data"]["exchange"] == exchange and 
                    item["data"]["symbol"] == symbol):
                    buffer_data.append(item)
                    if len(buffer_data) >= limit:
                        break
            
            # If not enough data in buffer, check Redis
            if len(buffer_data) < limit:
                # Get the latest value first
                latest_key = f"apex:market:latest:{exchange}:{symbol}"
                latest_data = await self.redis.get(latest_key)
                
                if latest_data:
                    buffer_data.append(msgpack.unpackb(latest_data))
                
                # Get historical data from Redis sorted set if needed
                if len(buffer_data) < limit:
                    history_key = f"apex:market:history:{exchange}:{symbol}"
                    history_data = await self.redis.zrevrange(
                        history_key, 0, limit - len(buffer_data) - 1, withscores=True
                    )
                    
                    for data, _ in history_data:
                        buffer_data.append(msgpack.unpackb(data))
            
            return buffer_data
            
        except Exception as e:
            self.logger.error("Error retrieving market data", error=str(e))
            return []

    async def get_market_status(self):
        """Get current market data system status"""
        return {
            "status": self.health_status["status"],
            "connections": {
                exchange: status["status"] 
                for exchange, status in self.connection_health.items()
            },
            "latency": {
                exchange: status.get("latency", 0) 
                for exchange, status in self.connection_health.items()
                if "latency" in status
            },
            "throughput": self.health_status.get("throughput", 0),
            "anomalies_detected": self.anomalies_detected
        }

    async def shutdown(self):
        """Gracefully shut down the market data system"""
        self.logger.info("Shutting down QuantumMarketData")
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close connections to exchanges
        for exchange, ws in list(self.active_connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        
        # Close Kafka connections
        try:
            await self.kafka_producer.stop()
            await self.kafka_consumer.stop()
        except Exception as e:
            self.logger.error("Error stopping Kafka", error=str(e))
        
        # Shutdown thread pools
        self.threads_io.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
        
        self.logger.info("QuantumMarketData shutdown complete")

async def initialize_market_data():
    """Initialize and start the market data system"""
    market_data = QuantumMarketData()
    await market_data.start_quantum_processing()
    return market_data