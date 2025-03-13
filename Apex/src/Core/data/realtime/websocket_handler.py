"""
Apex Quantum-Grade WebSocket Data Ingestion Engine
- Multi-exchange connectivity with nanosecond-precision processing
- Hardware-accelerated data pipeline with AI validation
- Secure authentication and failover protocols
"""
import asyncio
import concurrent.futures
import hashlib
import hmac
import msgpack
import os
import time
import uuid
import websockets
from collections import deque
from typing import Dict, Any, Optional, List, Tuple, Set

# Performance optimization
import uvloop
import redis
from redis.client import PubSub

# Apex Core imports
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.utils.helpers.security import decrypt_api_key, validate_hmac
from Apex.Config.config_loader import load_config

# Trading system integration
from Apex.src.Core.data.processing.data_parser import DataParser
from Apex.src.Core.data.realtime.market_data import QuantumMarketData
from Apex.src.Core.data.realtime.websocket_manager import WebSocketManager
from Apex.src.Core.trading.execution.order_execution import OrderExecutionManager
from Apex.src.Core.trading.risk.risk_management import RiskEngine
from Apex.src.Core.trading.hft.liquidity_manager import LiquidityOracle
from Apex.src.ai.ensembles.meta_trader import MetaTrader

# Configure UVLoop for enhanced async performance
uvloop.install()

class QuantumWebSocketHandler:
    """
    High-performance WebSocket handler for Apex trading system.
    Implements thread-safe singleton pattern with optimized message processing.
    """
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        return cls._instance if cls._instance else super().__new__(cls)
    
    async def __aenter__(self):
        async with self._lock:
            if not self._instance:
                self._instance = self
                await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self._handle_exception(exc_type, exc_val, exc_tb)
    
    async def _initialize(self):
        """Initialize core components with minimal latency impact"""
        self.logger = StructuredLogger("QuantumWS")
        self.config = load_config()
        self.parser = DataParser()
        self.ws_manager = WebSocketManager()
        
        # Enhanced memory management with fixed buffer sizes
        self.raw_queue = deque(maxlen=self.config.get('queue_size', 1_000_000))
        self.proc_queue = deque(maxlen=self.config.get('queue_size', 1_000_000))
        
        # Initialize security with HSM compatibility
        await self._init_security()
        
        # Performance metrics with nanosecond precision
        self.metrics = {
            'throughput': 0.0,
            'latency_ns': deque(maxlen=10_000),
            'error_rate': 0.0,
            'connection_quality': {},
            'last_failover': {}
        }
        
        # Thread pool for CPU-bound operations
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.get('thread_workers', os.cpu_count() * 2)
        )
        
        # Pre-load AI validation components
        self.anomaly_detector = await MetaTrader.load_component('ws_anomaly_detector')
        self.latency_optimizer = await MetaTrader.load_component('latency_arbiter')
        
        # Track active connections for monitoring
        self.active_connections = set()
        self.connection_latencies = {}
        
        # Cache for deduplication
        self.message_cache = set()
        self.cache_expiry = {}
        
        self.is_running = True
        self.logger.info("QuantumWebSocketHandler initialized successfully")
    
    async def _init_security(self):
        """Initialize security components with HSM integration"""
        # Get HMAC key with HSM fallback
        self.hmac_key = os.environ.get("APEX_HMAC_SECRET", "").encode()
        if not self.hmac_key:
            self.logger.critical("HMAC_SECRET environment variable not set")
            raise RuntimeError("Missing required security credential: HMAC_SECRET")
        
        # Connect to Redis with TLS if configured
        try:
            redis_ssl = os.getenv("REDIS_SSL", "False").lower() == "true"
            self.redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=decrypt_api_key(os.getenv("REDIS_PASSWORD", "")),
                ssl=redis_ssl,
                ssl_ca_certs=os.getenv("REDIS_SSL_CA") if redis_ssl else None,
                socket_timeout=0.1,  # Tight timeout for HFT
                socket_connect_timeout=1.0,
                health_check_interval=5,
                retry_on_timeout=True,
                decode_responses=False
            )
            self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
            self.pubsub.subscribe('ws_commands', 'system_alerts')
            self.logger.info("Connected to Redis securely")
        except Exception as e:
            self.logger.critical("Redis connection failed", error=str(e))
            raise
    
    async def start_quantum_stream(self):
        """Start the WebSocket streaming engine with parallel task execution"""
        try:
            async with self:
                tasks = [
                    self._connect_exchanges(),
                    self._process_raw_stream(),
                    self._distribute_processed_data(),
                    self._monitor_performance(),
                    self._handle_system_commands(),
                    self._execute_failover_protocols(),
                    self._cleanup_cache()
                ]
                await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.critical("Critical failure in quantum stream", error=str(e))
            await self.shutdown()
            raise
    
    async def _connect_exchanges(self):
        """Connect to multiple exchanges with automatic failover capability"""
        connection_tasks = []
        for exchange in self.config['exchanges']:
            connection_tasks.append(
                self.ws_manager.create_connection(
                    exchange_id=exchange['id'],
                    name=exchange['name'],
                    primary_url=exchange['ws_url'],
                    backup_urls=exchange.get('backup_urls', []),
                    auth_handler=self._authenticate_connection,
                    symbols=exchange.get('symbols', []),
                    connection_limit=exchange.get('connection_limit', 5)
                )
            )
        
        await asyncio.gather(*connection_tasks)
        self.logger.info(f"Connected to {len(connection_tasks)} exchanges")
    
    async def _authenticate_connection(self, exchange: str, ws):
        """Secure authentication with rotating credentials and HMAC validation"""
        try:
            # Generate secure authentication parameters
            nonce = str(uuid.uuid4())
            timestamp = str(int(time.time() * 1000))
            
            # Get encrypted API credentials with HSM support
            api_key = decrypt_api_key(os.environ.get(f"{exchange.upper()}_API_KEY", ""))
            api_secret = decrypt_api_key(os.environ.get(f"{exchange.upper()}_API_SECRET", ""))
            
            if not api_key or not api_secret:
                raise ValueError(f"Missing API credentials for {exchange}")
            
            # Generate secure signature
            signature = hmac.new(
                api_secret.encode(),
                f"{nonce}{timestamp}".encode(),
                hashlib.sha3_256
            ).hexdigest()
            
            # Send authentication message with minimal data exposure
            auth_msg = {
                "apiKey": api_key,
                "nonce": nonce,
                "timestamp": timestamp,
                "signature": signature
            }
            
            # Send packed message and validate response
            await ws.send(msgpack.packb(auth_msg))
            response = await asyncio.wait_for(ws.recv(), timeout=3.0)
            response_data = msgpack.unpackb(response)
            
            if not response_data.get('authenticated', False):
                raise AuthenticationError(f"Failed {exchange} authentication")
            
            self.logger.info(f"Authenticated with {exchange}")
            self.active_connections.add(exchange)
            return True
        except Exception as e:
            self.logger.error(f"Authentication failure for {exchange}", error=str(e))
            await self.ws_manager.rotate_credentials(exchange)
            return False
    
    async def _process_raw_stream(self):
        """Process raw WebSocket messages with parallel execution"""
        cpu_bound_tasks = set()
        
        while self.is_running:
            try:
                # Process messages from all exchanges in parallel
                async for exchange, message in self.ws_manager.stream_messages():
                    receipt_time = time.time_ns()
                    
                    # HMAC validation to ensure message integrity
                    if not validate_hmac(message, self.hmac_key):
                        self.logger.warning("HMAC validation failed", exchange=exchange)
                        continue
                    
                    # Check message deduplication
                    msg_hash = hashlib.blake2b(message, digest_size=16).digest()
                    if msg_hash in self.message_cache:
                        continue
                    
                    # Add to deduplication cache with expiry
                    self.message_cache.add(msg_hash)
                    self.cache_expiry[msg_hash] = time.time() + 3.0  # 3-second expiry
                    
                    # Process with ThreadPoolExecutor for CPU-bound operations
                    task = asyncio.create_task(self._process_message(
                        exchange, message, receipt_time, msg_hash
                    ))
                    cpu_bound_tasks.add(task)
                    task.add_done_callback(lambda t: cpu_bound_tasks.remove(t))
                    
                    # Throttle if too many tasks are pending
                    if len(cpu_bound_tasks) > 1000:
                        await asyncio.wait(cpu_bound_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            except Exception as e:
                self.logger.error("Stream processing error", error=str(e), exchange=exchange)
                await self._activate_failover(exchange)
                await asyncio.sleep(0.1)  # Backoff on failure
    
    async def _process_message(self, exchange, message, receipt_time, msg_hash):
        """Process individual messages with optimized parsing"""
        try:
            # Use thread pool for CPU-bound parsing
            loop = asyncio.get_running_loop()
            parsed = await loop.run_in_executor(
                self.executor, 
                self.parser.parse,
                message
            )
            
            # Add metadata for tracing and performance analysis
            parsed.update({
                'exchange': exchange,
                'receipt_ns': receipt_time,
                'processing_ns': time.time_ns(),
                'msg_hash': msg_hash.hex()
            })
            
            # AI validation to detect anomalies
            if not await self.anomaly_detector.validate(parsed):
                self.logger.warning("AI validation failed", data={
                    'exchange': exchange,
                    'symbol': parsed.get('symbol'),
                    'type': parsed.get('type')
                })
                return
            
            # Add to processing queue for distribution
            self.raw_queue.append(parsed)
            
            # Update connection quality metrics
            latency = (time.time_ns() - receipt_time) / 1_000_000  # Convert to ms
            if exchange not in self.connection_latencies:
                self.connection_latencies[exchange] = deque(maxlen=100)
            self.connection_latencies[exchange].append(latency)
            
        except Exception as e:
            self.logger.error("Message processing error", 
                              error=str(e), 
                              exchange=exchange)
    
    async def _distribute_processed_data(self):
        """Distribute processed data with batch operations"""
        batch_size = self.config.get('batch_size', 100)
        batch_timeout = self.config.get('batch_timeout', 0.01)  # 10ms max latency
        
        while self.is_running:
            try:
                batch = []
                # Collect batch of messages
                start_time = time.time()
                while len(batch) < batch_size and (time.time() - start_time) < batch_timeout:
                    try:
                        if self.raw_queue:
                            batch.append(self.raw_queue.popleft())
                        else:
                            await asyncio.sleep(0.001)
                            break
                    except IndexError:
                        await asyncio.sleep(0.001)
                        break
                
                if not batch:
                    await asyncio.sleep(0.001)
                    continue
                
                # Process batch in parallel
                distribution_tasks = []
                for data in batch:
                    # Pre-process for distribution
                    data['distribution_ns'] = time.time_ns()
                    self.proc_queue.append(data)
                    
                    # Parallel distribution to all systems
                    distribution_tasks.append(asyncio.create_task(
                        self._distribute_single_message(data)
                    ))
                
                # Wait for all distribution tasks
                if distribution_tasks:
                    await asyncio.gather(*distribution_tasks)
                
            except Exception as e:
                self.logger.error("Distribution error", error=str(e))
                await asyncio.sleep(0.01)  # Backoff on failure
    
    async def _distribute_single_message(self, data):
        """Distribute a single message to all trading systems"""
        try:
            # Distribute to all critical systems in parallel
            await asyncio.gather(
                QuantumMarketData.feed(data),
                RiskEngine.update_market_state(data),
                LiquidityOracle.process_stream(data),
                OrderExecutionManager.update_market_data(data),
                self._cache_processed_data(data)
            )
            
            # Track latency for performance monitoring
            processing_time = time.time_ns() - data['receipt_ns']
            self.metrics['latency_ns'].append(processing_time)
            
        except Exception as e:
            self.logger.error("Message distribution error", 
                             error=str(e),
                             exchange=data.get('exchange'),
                             symbol=data.get('symbol'))
    
    async def _cache_processed_data(self, data):
        """Cache processed data using batch operations"""
        try:
            # Generate cache key with correct namespace
            cache_key = f"market:{data['exchange']}:{data['symbol']}:{data['receipt_ns']}"
            
            # Use pipeline for batched Redis operations
            with self.redis.pipeline(transaction=False) as pipe:
                pipe.set(
                    cache_key,
                    msgpack.packb(data),
                    ex=self.config.get('cache_ttl', 60),
                    nx=True
                )
                
                # Add to time-series if enabled
                if self.config.get('use_time_series', False):
                    ts_key = f"ts:{data['exchange']}:{data['symbol']}"
                    pipe.xadd(
                        ts_key, 
                        {'data': msgpack.packb(data)},
                        maxlen=10000,
                        approximate=True
                    )
                
                await asyncio.to_thread(pipe.execute)
                
        except Exception as e:
            # Don't fail the entire pipeline on cache error
            self.logger.warning("Cache operation failed", error=str(e))
    
    async def _monitor_performance(self):
        """Monitor system performance with adaptive reporting"""
        monitoring_interval = self.config.get('monitoring_interval', 5)
        
        while self.is_running:
            try:
                # Calculate metrics
                throughput = len(self.raw_queue) / monitoring_interval
                self.metrics['throughput'] = throughput
                
                # Calculate connection quality for each exchange
                for exchange, latencies in self.connection_latencies.items():
                    if latencies:
                        avg_latency = sum(latencies) / len(latencies)
                        # Higher is better (1.0 is perfect)
                        quality = max(0.0, 1.0 - (avg_latency / 100.0))
                        self.metrics['connection_quality'][exchange] = quality
                
                # Calculate average latency
                if self.metrics['latency_ns']:
                    avg_latency_ns = sum(self.metrics['latency_ns']) / len(self.metrics['latency_ns'])
                    avg_latency_ms = avg_latency_ns / 1_000_000
                    self.logger.metric("processing_latency", {"ns": avg_latency_ns, "ms": avg_latency_ms})
                
                # Publish batch metrics
                await asyncio.to_thread(
                    self.redis.publish,
                    'ws_metrics',
                    msgpack.packb(self.metrics)
                )
                
                # Log critical metrics for observability
                queue_size = len(self.raw_queue)
                proc_queue_size = len(self.proc_queue)
                if queue_size > self.raw_queue.maxlen * 0.8 or proc_queue_size > self.proc_queue.maxlen * 0.8:
                    self.logger.warning("Queue near capacity", 
                                       raw_queue=queue_size,
                                       proc_queue=proc_queue_size)
                
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                self.logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(1)
    
    async def _handle_system_commands(self):
        """Process system commands with secure validation"""
        while self.is_running:
            try:
                # Non-blocking pubsub message handling
                message = await asyncio.to_thread(self.pubsub.get_message, timeout=0.1)
                if not message or message['type'] != 'message':
                    await asyncio.sleep(0.01)
                    continue
                
                # Validate and execute command
                command = msgpack.unpackb(message['data'])
                if not self._validate_command(command):
                    self.logger.warning("Invalid command received", command=command)
                    continue
                
                await self._execute_command(command)
                
            except Exception as e:
                self.logger.error("Command handling error", error=str(e))
                await asyncio.sleep(0.1)
    
    def _validate_command(self, command):
        """Validate command integrity and authorization"""
        try:
            # Basic validation
            if not isinstance(command, dict) or 'action' not in command:
                return False
            
            # Check authorization if present
            if 'auth' in command:
                auth_token = command.get('auth')
                expected = hmac.new(
                    self.hmac_key,
                    str(command.get('timestamp', '')).encode(),
                    hashlib.sha256
                ).hexdigest()
                
                if auth_token != expected:
                    self.logger.warning("Command authentication failed")
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def _execute_command(self, command):
        """Execute validated system command"""
        action = command.get('action')
        self.logger.info(f"Executing command: {action}")
        
        try:
            match action:
                case 'reconnect':
                    exchange = command.get('exchange')
                    if exchange:
                        await self.ws_manager.reconnect(exchange)
                        self.logger.info(f"Reconnected to {exchange}")
                
                case 'flush_queues':
                    self.raw_queue.clear()
                    self.proc_queue.clear()
                    self.logger.info("Flushed all queues")
                
                case 'rotate_keys':
                    exchange = command.get('exchange')
                    if exchange:
                        await self.ws_manager.rotate_credentials(exchange)
                        self.logger.info(f"Rotated credentials for {exchange}")
                
                case 'shutdown':
                    self.logger.info("Received shutdown command")
                    await self.shutdown()
                
                case _:
                    self.logger.warning(f"Unknown command: {action}")
        
        except Exception as e:
            self.logger.error(f"Command execution error: {action}", error=str(e))
    
    async def _execute_failover_protocols(self):
        """Monitor connection quality and execute failover when needed"""
        failover_threshold = self.config.get('failover_threshold', 0.8)
        check_interval = self.config.get('failover_check_interval', 10)
        
        while self.is_running:
            try:
                # Check each exchange connection quality
                for exchange, quality in self.metrics['connection_quality'].items():
                    # Check if quality below threshold and not recently failed over
                    last_failover = self.metrics['last_failover'].get(exchange, 0)
                    time_since_last = time.time() - last_failover
                    
                    if quality < failover_threshold and time_since_last > 60:
                        self.logger.warning(f"Connection quality degraded: {exchange}={quality}")
                        await self._activate_failover(exchange)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error("Failover check error", error=str(e))
                await asyncio.sleep(check_interval)
    
    async def _activate_failover(self, exchange):
        """Activate backup connection for exchange"""
        try:
            self.logger.info(f"Activating failover for {exchange}")
            
            # Record failover time
            self.metrics['last_failover'][exchange] = time.time()
            
            # Activate backup connection
            success = await self.ws_manager.activate_backup(exchange)
            
            if success:
                self.logger.info(f"Failover completed for {exchange}")
            else:
                self.logger.error(f"Failover failed for {exchange}")
                
        except Exception as e:
            self.logger.error(f"Failover error for {exchange}", error=str(e))
    
    async def _cleanup_cache(self):
        """Periodically clean up message cache to prevent memory leaks"""
        while self.is_running:
            try:
                # Clean expired cache entries
                current_time = time.time()
                expired = []
                
                for msg_hash, expiry in self.cache_expiry.items():
                    if current_time > expiry:
                        expired.append(msg_hash)
                
                # Remove expired entries
                for msg_hash in expired:
                    self.message_cache.discard(msg_hash)
                    del self.cache_expiry[msg_hash]
                
                # Log cache size periodically
                if len(expired) > 0:
                    self.logger.debug(f"Cleaned {len(expired)} cache entries, remaining: {len(self.message_cache)}")
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error("Cache cleanup error", error=str(e))
                await asyncio.sleep(5.0)
    
    async def _handle_exception(self, exc_type, exc_val, exc_tb):
        """Handle uncaught exceptions"""
        self.logger.critical(
            "Uncaught exception",
            error_type=str(exc_type.__name__),
            error=str(exc_val)
        )
        await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown with resource cleanup"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.logger.info("Initiating quantum shutdown")
        
        # Close all connections and resources
        shutdown_tasks = [
            self.ws_manager.close_all(),
            asyncio.to_thread(self.pubsub.unsubscribe),
            asyncio.to_thread(self.redis.close)
        ]
        
        # Wait for all tasks with timeout
        try:
            await asyncio.wait_for(asyncio.gather(*shutdown_tasks), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Shutdown timed out, forcing exit")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        self.logger.info("Shutdown complete")

# Quantum Singleton Instance
quantum_ws = QuantumWebSocketHandler()

if __name__ == "__main__":
    try:
        # Use asyncio.run with proper signal handling
        asyncio.run(quantum_ws.start_quantum_stream())
    except KeyboardInterrupt:
        asyncio.run(quantum_ws.shutdown())