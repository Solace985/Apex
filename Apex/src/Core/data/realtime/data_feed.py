import asyncio
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Deque, Tuple, Any, Set
from collections import deque
import aiohttp
import websockets
import orjson
import xxhash
import redis.asyncio as redis
import rocksdb
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory
import msgpack
import zstandard as zstd
import hmac
import hashlib
from functools import lru_cache
import warnings
import platform
import os
import sys

# Import optional GPU acceleration if available
try:
    import cupy as cp
    import cudf
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    warnings.warn("GPU acceleration not available. Install RAPIDS for optimal performance.")

# Optional Rust bindings for critical path operations
try:
    import apex_rust_core  # Custom Rust module via PyO3
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    warnings.warn("Rust acceleration not available. Install apex_rust_core for optimal performance.")

# Constants optimized for institutional-grade HFT performance
DATA_BUFFER_SIZE = 50_000_000  # 50MB shared memory buffer for IPC
MAX_QUEUE_SIZE = 5_000_000     # Increased queue size for extreme market conditions
RECONNECT_BASE_DELAY = 0.01    # Faster reconnection (10ms base)
MAX_RECONNECT_DELAY = 2.0      # Reduced maximum delay (2s)
HEALTH_CHECK_INTERVAL = 0.5    # Health check interval (500ms)
PROCESSING_THREADS = min(32, mp.cpu_count() * 2)  # Optimized thread count
GPU_BATCH_SIZE = 10000         # Batch size for GPU operations
ANOMALY_THRESHOLD = 4.0        # Z-score threshold for anomaly detection 
DATA_RETENTION_TICKS = 2000    # Increased for better statistical analysis
CACHE_SIZE = 1000              # Larger cache for performance
ROCKS_DB_WRITE_BUFFER = 256    # 256MB write buffer for RocksDB
LATENCY_STATS_WINDOW = 10000   # Window size for latency statistics
EXCHANGE_TIMEOUT = 1.0         # Exchange connection timeout (1s)
KAFKA_BATCH_SIZE = 65536       # 64KB batch size for Kafka
REDIS_POOL_SIZE = 10           # Connection pool size for Redis
WEBSOCKET_MAX_SIZE = 10 * 1024 * 1024  # 10MB max message size

# Market data normalization constants
NORMALIZATION_FIELDS = {'price', 'volume', 'timestamp', 'size', 'bid', 'ask', 'last'}
TICKER_FIELDS = {'symbol', 'price', 'bid', 'ask', 'volume', 'timestamp'}
ORDERBOOK_FIELDS = {'symbol', 'bids', 'asks', 'timestamp'}
TRADE_FIELDS = {'symbol', 'price', 'volume', 'side', 'timestamp'}
REQUIRED_FIELDS = {
    'ticker': TICKER_FIELDS,
    'orderbook': ORDERBOOK_FIELDS, 
    'trade': TRADE_FIELDS
}

class QuantumDataFeed:
    """
    Ultra-low latency market data pipeline for Apex AI Trading System
    Optimized for institutional-grade high-frequency trading with nanosecond precision
    
    Core features:
    - Multi-exchange data integration with cross-validation
    - AI-powered anomaly detection and data normalization
    - Hardware-accelerated processing (GPU/SIMD/Rust)
    - Distributed state with Redis and Kafka
    - Fault-tolerant with automatic failover
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the high-performance market data feed
        
        Args:
            config_path: Path to configuration file (optional, can use env vars)
        """
        # Core initialization with absolute minimal overhead
        self._shutdown_event = asyncio.Event()
        self._setup_logging()
        
        # System capability detection
        self.has_gpu = HFT = HAS_GPU
        self.has_rust = HAS_RUST
        self.is_linux = platform.system() == "Linux"
        self.cpu_count = mp.cpu_count()
        
        # Start timestamp for performance tracking
        self.start_time = time.time_ns()
        
        # Configuration
        self.config_path = config_path
        self.config = self._load_config() if config_path else {}
        
        # Connection management
        self.ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.last_heartbeats: Dict[str, int] = {}  # Using ns precision timestamps
        self.connection_status: Dict[str, bool] = {}
        self.reconnect_attempts: Dict[str, int] = {}
        self.active_feeds = 0
        self.feed_stats: Dict[str, Dict[str, Any]] = {}
        
        # Exchange reliability metrics 
        self.exchange_latency: Dict[str, Deque[float]] = {}
        self.exchange_reliability: Dict[str, float] = {}  # 0.0-1.0 score
        self.exchange_weights: Dict[str, float] = {}  # For weighted price validation
        
        # Ultra-low latency processing queues with increased capacity
        self.raw_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.proc_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.persist_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        
        # Data structures optimized for specific market data types
        # Using memory-efficient NumPy arrays where possible
        self.price_series: Dict[str, Dict[str, np.ndarray]] = {}  # symbol -> {exchange -> prices}
        self.tick_cache: Dict[str, Dict[str, Any]] = {}  # symbol -> latest tick
        self.orderbook_cache: Dict[str, Dict[str, Any]] = {}  # symbol -> latest orderbook
        self.trade_cache: Dict[str, Deque[Dict[str, Any]]] = {}  # symbol -> recent trades
        self.volatility_cache: Dict[str, float] = {}  # symbol -> volatility
        self.spread_cache: Dict[str, float] = {}  # symbol -> current spread
        self.cross_validation: Dict[str, Dict[str, float]] = {}  # symbol -> {exchange -> price}
        
        # Anomaly detection
        self.anomaly_counters: Dict[str, int] = {}  # exchange -> anomaly count
        self.blacklisted_exchanges: Set[str] = set()  # Temporarily ignored exchanges
        
        # Performance monitoring with nanosecond precision
        self.processing_latency = deque(maxlen=LATENCY_STATS_WINDOW)
        self.ingestion_latency = deque(maxlen=LATENCY_STATS_WINDOW)
        self.serialization_latency = deque(maxlen=LATENCY_STATS_WINDOW)
        self.throughput_counter = 0
        self.message_counters: Dict[str, int] = {}  # message type -> count
        self.error_counters: Dict[str, int] = {}  # error type -> count
        self.last_throughput_check = time.time_ns()
        
        # Rate limiting with adaptive algorithm
        self.api_request_counts: Dict[str, Dict[int, int]] = {}
        self.rate_limit_windows: Dict[str, int] = {}
        self.backoff_multipliers: Dict[str, float] = {}
        
        # Thread and process pools for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=PROCESSING_THREADS)
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, self.cpu_count // 2))
        
        # Initialize GPU context if available
        self.gpu_context = self._setup_gpu() if self.has_gpu else None
        
        # Initialize shared memory for IPC
        self.shm = self._setup_shared_memory()
        self.shm_index = 0  # Current write position
        
        # Initialize distributed state systems
        self.redis_pool = self._setup_redis_pool()
        self.rocks_db = self._setup_rocksdb()
        self.kafka_producer = None  # Initialized in start()
        self.kafka_consumer = None  # Initialized in start()
        
        # Load exchange configurations with validation
        self._load_exchange_configs()
        
        # Cache frequently used config values
        self._cache_config_values()
        
        # Spin up atomic clock synchronization
        self._time_synchronization_offset = 0
        asyncio.create_task(self._sync_atomic_clock())
        
        # Symbol metadata (tick size, lot size, etc.)
        self.symbol_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize normalizers
        self._init_normalizers()
        
        self.logger.info(
            f"QuantumDataFeed initialized: {self.cpu_count} CPUs, "
            f"GPU: {self.has_gpu}, Rust: {self.has_rust}, "
            f"Exchanges: {len(self.exchange_configs)}"
        )
    
    def _setup_logging(self):
        """Efficient logging with structured output"""
        self.logger = logging.getLogger("QuantumDataFeed")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler with a higher log level
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Create formatter with timestamp
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
        
        # Reduce unnecessary log propagation
        self.logger.propagate = False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with validation"""
        if not self.config_path:
            return {}
            
        try:
            if self.config_path.endswith('.json'):
                with open(self.config_path, 'rb') as f:
                    return orjson.loads(f.read())
            else:
                # Assume Python module
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", self.config_path)
                if not spec or not spec.loader:
                    raise ImportError(f"Could not load config from {self.config_path}")
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                return {k: v for k, v in config_module.__dict__.items() 
                        if not k.startswith('_')}
        except Exception as e:
            self.logger.error(f"Config loading error: {e}")
            return {}
    
    def _setup_gpu(self):
        """Set up GPU context for accelerated operations"""
        if not self.has_gpu:
            return None
            
        try:
            # Initialize GPU context
            mem_pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(mem_pool.malloc)
            
            # Get GPU info
            dev = cp.cuda.Device(0)  # Use first GPU
            self.logger.info(f"GPU acceleration enabled: {dev.name}, {dev.mem_info[1]/1024**3:.2f}GB memory")
            
            # Pre-compile GPU kernels 
            # This reduces latency during critical operations
            if hasattr(apex_rust_core, 'precompile_gpu_kernels'):
                apex_rust_core.precompile_gpu_kernels()
                
            return {'device': dev, 'memory_pool': mem_pool}
        except Exception as e:
            self.logger.error(f"GPU initialization error: {e}")
            self.has_gpu = False
            return None
    
    def _setup_shared_memory(self):
        """Set up shared memory for inter-process communication"""
        try:
            # Clean up any existing shared memory with same name
            try:
                existing_shm = shared_memory.SharedMemory(name="apex_market_data", create=False)
                existing_shm.close()
                existing_shm.unlink()
            except Exception:
                pass
                
            # Create new shared memory block
            return shared_memory.SharedMemory(
                name="apex_market_data", 
                create=True, 
                size=DATA_BUFFER_SIZE
            )
        except Exception as e:
            self.logger.error(f"Shared memory error: {e}")
            return None
    
    def _setup_redis_pool(self):
        """Set up Redis connection pool for distributed state"""
        try:
            host = os.environ.get("REDIS_HOST", "localhost")
            port = int(os.environ.get("REDIS_PORT", "6379"))
            password = os.environ.get("REDIS_PASSWORD", None)
            
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                max_connections=REDIS_POOL_SIZE,
                decode_responses=False,  # Raw binary for performance
                health_check_interval=10
            )
            return pool
        except Exception as e:
            self.logger.error(f"Redis pool setup error: {e}")
            return None
    
    async def get_redis(self):
        """Get a Redis connection from the pool"""
        if not self.redis_pool:
            return None
            
        try:
            return redis.Redis(connection_pool=self.redis_pool)
        except Exception as e:
            self.logger.error(f"Redis connection error: {e}")
            return None
    
    def _setup_rocksdb(self):
        """Set up RocksDB for high-performance local persistence"""
        try:
            # Directory for RocksDB files
            db_path = os.environ.get("ROCKSDB_PATH", "apex_market_data.db")
            
            # Optimize RocksDB for HFT workloads
            opts = rocksdb.Options()
            opts.create_if_missing = True
            opts.max_open_files = 500000  # Handle many small files
            opts.write_buffer_size = ROCKS_DB_WRITE_BUFFER * 1024 * 1024  # Write buffer in MB
            opts.max_write_buffer_number = 6  # More write buffers for better write performance
            opts.target_file_size_base = 64 * 1024 * 1024  # 64MB target file size
            opts.max_background_jobs = max(4, self.cpu_count)  # Parallel compaction
            opts.compression = rocksdb.CompressionType.lz4_compression  # Faster than zstd
            opts.compression_opts = (9, 0, 0)  # Level 9 compression
            opts.enable_pipelined_write = True  # Pipelined writes for better throughput
            opts.db_write_buffer_size = 512 * 1024 * 1024  # Global write buffer (512MB)
            
            # Configure bloom filters for faster lookups
            table_options = rocksdb.BlockBasedTableFactory(
                filter_policy=rocksdb.BloomFilterPolicy(10),
                block_cache=rocksdb.LRUCache(1 * 1024 * 1024 * 1024),  # 1GB cache
                block_size=16 * 1024,  # 16KB blocks
            )
            opts.table_factory = table_options
            
            # Configure WAL (Write-Ahead Log) for durability vs performance tradeoff
            opts.wal_dir = os.path.join(db_path, "wal")
            os.makedirs(opts.wal_dir, exist_ok=True)
            
            # Set WAL size limit
            opts.max_total_wal_size = 512 * 1024 * 1024  # 512MB
            
            # Configure compaction style
            opts.compaction_style = rocksdb.CompactionStyle.level
            opts.level0_file_num_compaction_trigger = 8
            
            # Create RocksDB instance with optimized options
            return rocksdb.DB(db_path, opts)
        except Exception as e:
            self.logger.error(f"RocksDB setup error: {e}")
            return None
    
    async def _setup_kafka_producer(self):
        """Set up Kafka producer for data persistence"""
        try:
            bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            client_id = os.environ.get("KAFKA_CLIENT_ID", "apex_data_feed")
            
            # High-performance Kafka producer
            producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                client_id=client_id,
                compression_type="lz4",  # Faster than zstd for streaming data
                acks=0,  # Fire and forget for maximum throughput
                batch_size=KAFKA_BATCH_SIZE,
                linger_ms=1,  # 1ms linger for batching
                request_timeout_ms=2000,  # 2s timeout
                max_request_size=10 * 1024 * 1024,  # 10MB max request
                enable_idempotence=False,  # Disable for performance
                max_in_flight_requests_per_connection=1000  # Allow many in-flight requests
            )
            
            await producer.start()
            self.logger.info(f"Kafka producer connected to {bootstrap_servers}")
            return producer
        except Exception as e:
            self.logger.error(f"Kafka producer setup error: {e}")
            return None
    
    async def _setup_kafka_consumer(self):
        """Set up Kafka consumer for backup data source"""
        try:
            bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            group_id = os.environ.get("KAFKA_GROUP_ID", "apex_data_feed")
            topics = ["apex.control", "apex.command"]  # Control channel
            
            consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=1000,
                fetch_max_wait_ms=100
            )
            
            await consumer.start()
            self.logger.info(f"Kafka consumer subscribed to control topics")
            return consumer
        except Exception as e:
            self.logger.error(f"Kafka consumer setup error: {e}")
            return None

    def _load_exchange_configs(self):
        """Load exchange configurations with enhanced validation"""
        self.exchange_configs = {}
        
        # Load exchanges from environment
        exchanges = [x.split('_')[0].lower() for x in os.environ.keys() 
                    if x.endswith('_EXCHANGE_URL')]
        
        # Also check for exchanges in config file
        if hasattr(self, 'config') and self.config and 'exchanges' in self.config:
            exchanges.extend(self.config['exchanges'].keys())
            
        exchanges = list(set(exchanges))  # Remove duplicates
        
        if not exchanges:
            self.logger.error("No exchange configurations found")
            raise ValueError("At least one exchange configuration required")
        
        # Process each exchange
        for exchange in exchanges:
            # Try config file first, then environment variables
            config_entry = (self.config.get('exchanges', {}).get(exchange, {}) 
                           if hasattr(self, 'config') and self.config else {})
            
            # Get URL (required)
            url_key = f"{exchange.upper()}_EXCHANGE_URL"
            url = config_entry.get('url') or os.environ.get(url_key)
            if not url:
                self.logger.error(f"Missing URL for {exchange}")
                continue
            
            # Get credentials
            api_key = config_entry.get('api_key') or os.environ.get(f"{exchange.upper()}_API_KEY", "")
            secret = config_entry.get('secret') or os.environ.get(f"{exchange.upper()}_SECRET", "")
            
            # Get performance settings
            timeout = float(config_entry.get('timeout') or 
                           os.environ.get(f"{exchange.upper()}_TIMEOUT", EXCHANGE_TIMEOUT))
            
            priority = int(config_entry.get('priority') or 
                          os.environ.get(f"{exchange.upper()}_PRIORITY", "1"))
            
            rate_limit = int(config_entry.get('rate_limit') or 
                            os.environ.get(f"{exchange.upper()}_RATE_LIMIT", "20"))
            
            rate_window = int(config_entry.get('rate_window') or 
                             os.environ.get(f"{exchange.upper()}_RATE_WINDOW", "1"))
            
            # Handle historical data endpoints (optional)
            historical_url = config_entry.get('historical_url') or os.environ.get(
                f"{exchange.upper()}_HISTORICAL_URL", "")
            
            # Get exchange-specific subscription channels
            channels = config_entry.get('channels') or os.environ.get(
                f"{exchange.upper()}_CHANNELS", "ticker,orderbook,trade").split(',')
            
            # Get symbols to subscribe
            symbols = config_entry.get('symbols') or os.environ.get(
                f"{exchange.upper()}_SYMBOLS", "").split(',')
            if symbols == [""]:  # Handle empty string case
                symbols = []
            
            # Store exchange config
            self.exchange_configs[exchange] = {
                "url": url,
                "api_key": api_key,
                "secret": secret,
                "timeout": timeout,
                "priority": priority,
                "rate_limit": rate_limit,
                "rate_window": rate_window,
                "historical_url": historical_url,
                "channels": channels,
                "symbols": symbols,
                # Exchange-specific message format
                "message_format": config_entry.get('message_format', 'json'),
                # Authentication type
                "auth_type": config_entry.get('auth_type', 'hmac'),
                # Max connection attempts
                "max_retries": int(config_entry.get('max_retries') or 
                                  os.environ.get(f"{exchange.upper()}_MAX_RETRIES", "10")),
                # Is this exchange a primary or backup
                "is_primary": priority <= 2
            }
            
            # Initialize tracking for this exchange
            self.api_request_counts[exchange] = {}
            self.rate_limit_windows[exchange] = rate_window
            self.reconnect_attempts[exchange] = 0
            self.backoff_multipliers[exchange] = 1.0
            self.exchange_latency[exchange] = deque(maxlen=100)
            self.exchange_reliability[exchange] = 1.0  # Start with full reliability
            self.feed_stats[exchange] = {
                "connected": False,
                "last_message": 0,
                "message_count": 0,
                "error_count": 0,
                "latency_ms": 0.0
            }
        
        self.logger.info(f"Loaded {len(self.exchange_configs)} exchange configs")

    def _cache_config_values(self):
        """Cache frequently accessed configuration values"""
        # Group exchanges by priority for failover
        self.priority_exchanges = {}
        for e, c in self.exchange_configs.items():
            priority = c["priority"]
            if priority not in self.priority_exchanges:
                self.priority_exchanges[priority] = []
            self.priority_exchanges[priority].append(e)
            
        # Primary exchanges (priority 1) are the default data source
        self.primary_exchanges = self.priority_exchanges.get(1, [])
        if not self.primary_exchanges and self.priority_exchanges:
            # If no priority 1, use the lowest available priority
            min_priority = min(self.priority_exchanges.keys())
            self.primary_exchanges = self.priority_exchanges[min_priority]
            
        # Pre-calculate priorities by exchange
        self.exchange_priorities = {e: c["priority"] for e, c in self.exchange_configs.items()}
        
        # Pre-calculate rate limits by exchange
        self.exchange_rate_limits = {e: c["rate_limit"] for e, c in self.exchange_configs.items()}
        
        # Initialize exchange weights equally
        total_exchanges = len(self.exchange_configs)
        for exchange in self.exchange_configs:
            self.exchange_weights[exchange] = 1.0 / total_exchanges if total_exchanges else 0.0
    
    def _init_normalizers(self):
        """Initialize data normalizers for consistent output across exchanges"""
        # Each exchange may have a different data format
        # We need to normalize data for consistent processing
        self.data_normalizers = {}
        
        # Load custom normalizers if available
        for exchange in self.exchange_configs:
            # First check for custom normalizer in subdirectory
            normalizer_name = f"{exchange}_normalizer.py"
            if os.path.exists(os.path.join("normalizers", normalizer_name)):
                try:
                    # Import custom normalizer
                    import sys
                    import importlib.util
                    normalizer_path = os.path.join("normalizers", normalizer_name)
                    spec = importlib.util.spec_from_file_location(
                        f"{exchange}_normalizer", normalizer_path)
                    if spec and spec.loader:
                        normalizer_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(normalizer_module)
                        self.data_normalizers[exchange] = normalizer_module.normalize
                        self.logger.info(f"Loaded custom normalizer for {exchange}")
                except Exception as e:
                    self.logger.error(f"Error loading normalizer for {exchange}: {e}")
        
        # Register the default normalizer for exchanges without custom ones
        for exchange in self.exchange_configs:
            if exchange not in self.data_normalizers:
                self.data_normalizers[exchange] = self._default_normalizer
    
    def _default_normalizer(self, data, message_type=None, exchange=None):
        """Default normalizer for exchange data"""
        if not data or not isinstance(data, dict):
            return None
        
        # If message_type not specified, try to infer it
        if not message_type:
            if 'type' in data:
                message_type = data['type']
            elif 'e' in data:  # Some exchanges use 'e' for event type
                message_type = data['e']
            else:
                # Can't determine type, return as-is
                return data
        
        normalized = {
            "type": message_type,
            "exchange": exchange,
            "raw": data,  # Preserve raw data
            "timestamp": int(time.time() * 1000)  # Default timestamp
        }
        
        # Extract common fields based on message type
        if message_type == 'ticker':
            # Extract symbol
            symbol = data.get('symbol', data.get('s', data.get('pair', '')))
            normalized['symbol'] = symbol
            
            # Extract price
            price = data.get('price', data.get('p', data.get('last', 0.0)))
            normalized['price'] = float(price) if price else 0.0
            
            # Extract bid/ask
            bid = data.get('bid', data.get('b', 0.0))
            ask = data.get('ask', data.get('a', 0.0))
            normalized['bid'] = float(bid) if bid else 0.0
            normalized['ask'] = float(ask) if ask else 0.0
            
            # Extract volume
            volume = data.get('volume', data.get('v', data.get('vol', 0.0)))
            normalized['volume'] = float(volume) if volume else 0.0
            
            # Extract timestamp if available
            ts = data.get('timestamp', data.get('ts', data.get('time', 0)))
            if ts:
                # Different exchanges use different timestamp formats
                # Normalize to milliseconds
                if isinstance(ts, str):
                    try:
                        ts = float(ts)
                    except:
                        # Try ISO format
                        try:
                            from datetime import datetime
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp() * 1000
                        except:
                            ts = time.time() * 1000
                
                # Convert to milliseconds if in seconds
                if ts < 1e12:  # Likely in seconds
                    ts *= 1000
                
                normalized['timestamp'] = int(ts)
                
        elif message_type == 'trade':
            # Extract symbol
            symbol = data.get('symbol', data.get('s', data.get('pair', '')))
            normalized['symbol'] = symbol
            
            # Extract price
            price = data.get('price', data.get('p', 0.0))
            normalized['price'] = float(price) if price else 0.0
            
            # Extract volume/size
            volume = data.get('size', data.get('q', data.get('amount', data.get('volume', 0.0))))
            normalized['volume'] = float(volume) if volume else 0.0
            
            # Extract side
            side = data.get('side', data.get('S', ''))
            if side.lower() in ('buy', 'b'):
                normalized['side'] = 'buy'
            elif side.lower() in ('sell', 's'):
                normalized['side'] = 'sell'
            else:
                normalized['side'] = 'unknown'
                
            # Extract timestamp
            ts = data.get('timestamp', data.get('ts', data.get('time', 0)))
            if ts:
                # Normalize to milliseconds
                if isinstance(ts, str):
                    try:
                        ts = float(ts)
                    except:
                        try:
                            from datetime import datetime
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp() * 1000
                        except:
                            ts = time.time() * 1000
                
                # Convert to milliseconds if in seconds
                if ts < 1e12:  # Likely in seconds
                    ts *= 1000
                
                normalized['timestamp'] = int(ts)
                
        elif message_type == 'orderbook':
            # Extract symbol
            symbol = data.get('symbol', data.get('s', data.get('pair', '')))
            normalized['symbol'] = symbol
            
            # Extract bids
            bids = data.get('bids', data.get('b', []))
            if bids:
                if isinstance(bids[0], list):
                    # Format is [[price, size], ...]
                    normalized['bids'] = [[float(b[0]), float(b[1])] for b in bids]
                elif isinstance(bids[0], dict):
                    # Format is [{"price": x, "size": y}, ...]
                    normalized['bids'] = [[float(b.get('price', 0)), float(b.get('size', 0))] for b in bids]
                else:
                    # Unknown format, store as-is
                    normalized['bids'] = bids
            else:
                normalized['bids'] = []
                
            # Extract asks
            asks = data.get('asks', data.get('a', []))
            if asks:
                if isinstance(asks[0], list):
                    # Format is [[price, size], ...]
                    normalized['asks'] = [[float(a[0]), float(a[1])] for a in asks]
                elif isinstance(asks[0], dict):
                    # Format is [{"price": x, "size": y}, ...]
                    normalized['asks'] = [[float(a.get('price', 0)), float(a.get('size', 0))] for a in asks]
                else:
                    # Unknown format, store as-is
                    normalized['asks'] = asks
            else:
                normalized['asks'] = []
                
            # Extract timestamp
            ts = data.get('timestamp', data.get('ts', data.get('time', 0)))
            if ts:
                # Normalize to milliseconds
                if isinstance(ts, str):
                    try:
                        ts = float(ts)
                    except:
                        try:
                            from datetime import datetime
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp() * 1000
                        except:
                            ts = time.time() * 1000
                
                # Convert to milliseconds if in seconds
                if ts < 1e12:  # Likely in seconds
                    ts *= 1000
                
                normalized['timestamp'] = int(ts)
                
        return normalized

    async def start(self):
        """Start the market data feed with all exchanges"""
        self.logger.info("Starting QuantumDataFeed...")
        
        # Initialize Kafka producer
        self.kafka_producer = await self._setup_kafka_producer()
        
        # Initialize Kafka consumer for control messages
        self.kafka_consumer = await self._setup_kafka_consumer()
        
        # Start the processing workers
        # Each worker runs in its own task
        for _ in range(PROCESSING_THREADS):
            asyncio.create_task(self._process_worker())
            
        # Start the persistence workers
        for _ in range(max(1, PROCESSING_THREADS // 2)):
            asyncio.create_task(self._persistence_worker())
            
        # Start the control message consumer
        asyncio.create_task(self._control_consumer())
        
        # Start health check and metrics collection
        asyncio.create_task(self._health_check())
        asyncio.create_task(self._collect_metrics())
        
        # Connect to all exchanges
        connect_tasks = []
        for exchange in self.exchange_configs:
            connect_task = asyncio.create_task(
                self._connect_exchange(exchange)
            )
            connect_tasks.append(connect_task)
        
        # Wait for initial connections (with timeout)
        _, pending = await asyncio.wait(
            connect_tasks, 
            timeout=5.0,
            return_when=asyncio.ALL_COMPLETED
        )
        
        if pending:
            self.logger.warning(f"{len(pending)} exchange connections still pending")
            
        # Report connection status
        connected = sum(1 for e, status in self.connection_status.items() if status)
        self.logger.info(f"Connected to {connected}/{len(self.exchange_configs)} exchanges")
        
        # Set active status
        self.running = True
        self.logger.info("QuantumDataFeed started successfully")
    
    async def _connect_exchange(self, exchange):
        """Connect to an exchange websocket with automatic reconnection"""
        config = self.exchange_configs.get(exchange)
        if not config:
            self.logger.error(f"No configuration found for exchange: {exchange}")
            self.connection_status[exchange] = False
            return
            
        url = config["url"]
        self.logger.info(f"Connecting to {exchange} at {url}")
        
        # Update connection status
        self.connection_status[exchange] = False
        self.active_feeds += 1
        
        # Authentication headers/params if needed
        headers = {}
        if config.get("api_key") and config.get("secret"):
            if config["auth_type"] == "hmac":
                # HMAC-based authentication
                timestamp = str(int(time.time() * 1000))
                signature_payload = f"{timestamp}GET{url.split('/', 3)[-1]}"
                
                # Create signature
                signature = hmac.new(
                    config["secret"].encode(), 
                    signature_payload.encode(), 
                    hashlib.sha256
                ).hexdigest()
                
                # Add to headers
                headers = {
                    "API-Key": config["api_key"],
                    "API-Timestamp": timestamp,
                    "API-Signature": signature
                }
            elif config["auth_type"] == "jwt":
                # JWT-based authentication
                # This would need a JWT library import
                # Since libraries may vary, this is a placeholder
                self.logger.warning(f"JWT auth not fully implemented for {exchange}")
                # headers = {"Authorization": f"Bearer {jwt_token}"}
        
        # Set up retry loop with exponential backoff
        max_retries = config.get("max_retries", 10)
        retry_count = 0
        
        while retry_count < max_retries and not self._shutdown_event.is_set():
            try:
                # Establish connection with timeout
                self.logger.debug(f"Connecting to {exchange}, attempt {retry_count+1}/{max_retries}")
                
                async with websockets.connect(
                    url,
                    extra_headers=headers,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=WEBSOCKET_MAX_SIZE,
                    compression=None  # Disable for HFT performance
                ) as websocket:
                    # Store the connection
                    self.ws_connections[exchange] = websocket
                    self.connection_status[exchange] = True
                    self.reconnect_attempts[exchange] = 0
                    self.feed_stats[exchange]["connected"] = True
                    self.last_heartbeats[exchange] = time.time_ns()
                    
                    self.logger.info(f"Connected to {exchange}")
                    
                    # Subscribe to channels
                    await self._subscribe_channels(exchange, websocket)
                    
                    # Reset backoff on successful connection
                    self.backoff_multipliers[exchange] = 1.0
                    
                    # Process messages in a loop
                    while not self._shutdown_event.is_set():
                        try:
                            # Set timeout for receiving messages
                            message = await asyncio.wait_for(
                                websocket.recv(), 
                                timeout=config["timeout"]
                            )
                            
                            # Process received message
                            start_time = time.time_ns()
                            await self._handle_message(exchange, message)
                            
                            # Update heartbeat
                            self.last_heartbeats[exchange] = time.time_ns()
                            
                            # Measure latency
                            latency_ms = (time.time_ns() - start_time) / 1e6
                            self.exchange_latency[exchange].append(latency_ms)
                            
                            # Update stats
                            self.feed_stats[exchange]["message_count"] += 1
                            self.feed_stats[exchange]["last_message"] = time.time_ns()
                            self.feed_stats[exchange]["latency_ms"] = (
                                sum(self.exchange_latency[exchange]) / 
                                len(self.exchange_latency[exchange])
                            )
                            
                        except asyncio.TimeoutError:
                            # Check if the connection is still alive
                            try:
                                pong = await websocket.ping()
                                await asyncio.wait_for(pong, timeout=5)
                                self.logger.debug(f"{exchange} ping successful")
                            except:
                                self.logger.warning(f"{exchange} connection timed out")
                                break
                        except websockets.exceptions.ConnectionClosedError as e:
                            self.logger.warning(f"{exchange} connection closed: {e}")
                            break
                        except Exception as e:
                            self.logger.error(f"Error processing {exchange} message: {e}")
                            self.feed_stats[exchange]["error_count"] += 1
                            self.error_counters[str(type(e).__name__)] = (
                                self.error_counters.get(str(type(e).__name__), 0) + 1
                            )
                
            except (websockets.exceptions.InvalidStatusCode, 
                    websockets.exceptions.InvalidHandshake) as e:
                self.logger.error(f"{exchange} connection error (invalid handshake): {e}")
                retry_count += 1
                
            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.ConnectionClosedError) as e:
                self.logger.warning(f"{exchange} connection closed: {e}")
                retry_count += 1
                
            except Exception as e:
                self.logger.error(f"Unexpected error connecting to {exchange}: {e}")
                retry_count += 1
            
            # Update connection status
            self.connection_status[exchange] = False
            self.feed_stats[exchange]["connected"] = False
            
            # Clean up connection reference
            self.ws_connections.pop(exchange, None)
            
            # Exponential backoff with jitter
            if retry_count < max_retries and not self._shutdown_event.is_set():
                # Calculate backoff time
                backoff = min(
                    RECONNECT_BASE_DELAY * (2 ** retry_count) * self.backoff_multipliers[exchange],
                    MAX_RECONNECT_DELAY
                )
                
                # Add jitter (±20%)
                jitter = backoff * 0.2 * (random.random() * 2 - 1)
                backoff += jitter
                
                self.logger.info(f"Reconnecting to {exchange} in {backoff:.2f}s (attempt {retry_count+1})")
                await asyncio.sleep(backoff)
                
                # Increase backoff multiplier
                self.backoff_multipliers[exchange] = min(
                    self.backoff_multipliers[exchange] * 1.2, 
                    5.0
                )
        
        # If we've exhausted retries
        if retry_count >= max_retries:
            self.logger.error(f"Failed to connect to {exchange} after {max_retries} attempts")
            # Blacklist the exchange if all retries failed
            self.blacklisted_exchanges.add(exchange)
            
        self.active_feeds -= 1
    
    async def _subscribe_channels(self, exchange, websocket):
        """Subscribe to exchange channels based on configuration"""
        config = self.exchange_configs.get(exchange)
        if not config:
            return
            
        # Get configured channels and symbols
        channels = config.get("channels", [])
        symbols = config.get("symbols", [])
        
        # If no symbols specified, this might be an exchange that sends all data
        if not symbols:
            self.logger.info(f"{exchange}: No symbols specified, assuming full data feed")
            return
            
        try:
            # Different exchanges have different subscription formats
            # We use exchange-specific subscription methods if available
            subscription_method = f"_subscribe_{exchange}"
            if hasattr(self, subscription_method):
                # Use exchange-specific subscription logic
                subscription_func = getattr(self, subscription_method)
                await subscription_func(websocket, channels, symbols)
            else:
                # Use generic subscription format for common exchanges
                # This is a basic implementation that works for many exchanges
                for channel in channels:
                    for symbol in symbols:
                        # Construct subscription message
                        subscription = {
                            "method": "subscribe",
                            "params": [f"{channel}:{symbol}"],
                            "id": xxhash.xxh32(f"{exchange}:{channel}:{symbol}").intdigest()
                        }
                        
                        # Send subscription
                        await websocket.send(orjson.dumps(subscription))
                        self.logger.debug(f"Subscribed to {exchange} {channel}:{symbol}")
                        
                        # Rate limit subscriptions to avoid flooding
                        await asyncio.sleep(0.05)
            
            self.logger.info(f"Subscribed to {exchange} channels: {', '.join(channels)}")
        except Exception as e:
            self.logger.error(f"Error subscribing to {exchange} channels: {e}")
    
    async def _handle_message(self, exchange, message):
        """Process incoming websocket message with automatic format detection"""
        if not message:
            return
            
        # Get exchange configuration
        config = self.exchange_configs.get(exchange)
        if not config:
            return
        
        # Parse message based on format
        try:
            message_format = config.get("message_format", "json")
            
            if message_format == "json":
                # Most exchanges use JSON
                data = orjson.loads(message)
            elif message_format == "messagepack":
                # Some exchanges use MessagePack
                data = msgpack.unpackb(message)
            elif message_format == "raw":
                # Some exchanges use custom binary formats
                # In this case, we need exchange-specific handler
                handler_method = f"_handle_{exchange}_raw"
                if hasattr(self, handler_method):
                    handler = getattr(self, handler_method)
                    data = handler(message)
                else:
                    self.logger.warning(f"No raw handler for {exchange}, skipping message")
                    return
            else:
                # Default to JSON
                data = orjson.loads(message)
                
            # Determine message type
            message_type = None
            
            # Check for heartbeat/ping messages
            if self._is_heartbeat(data, exchange):
                # Process heartbeat
                self._process_heartbeat(data, exchange)
                return
                
            # Try to determine message type
            if 'type' in data:
                message_type = data['type']
            elif 'e' in data:  # Some exchanges use 'e' for event type
                message_type = data['e']
            elif 'channel' in data:  # Some use channel
                message_type = data['channel']
            elif 'table' in data:  # Some use table
                message_type = data['table']
                
            # If still can't determine, try exchange-specific logic
            if not message_type:
                message_type = self._infer_message_type(data, exchange)
                
            # If we still can't determine, log and return
            if not message_type:
                if not self._is_control_message(data, exchange):
                    self.logger.debug(f"Unknown message type from {exchange}: {data}")
                return
                
            # Normalize the data format
            normalized = self.data_normalizers[exchange](data, message_type, exchange)
            if not normalized:
                return
                
            # Check for required fields
            if message_type in REQUIRED_FIELDS:
                missing_fields = [f for f in REQUIRED_FIELDS[message_type] 
                                if f not in normalized]
                if missing_fields:
                    self.logger.warning(
                        f"Missing fields in {exchange} {message_type}: {missing_fields}"
                    )
                    return
            
            # Add reception timestamp for latency calculation
            normalized['recv_time'] = time.time_ns()
            
            # Add message to processing queue
            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(
                    self.raw_queue.put(normalized),
                    timeout=0.1
                )
                
                # Update throughput counter
                self.throughput_counter += 1
                
                # Update message type counter
                self.message_counters[message_type] = self.message_counters.get(message_type, 0) + 1
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Queue full, dropping {exchange} {message_type} message")
                
        except Exception as e:
            self.logger.error(f"Error processing {exchange} message: {e}")
            self.feed_stats[exchange]["error_count"] += 1
            self.error_counters[str(type(e).__name__)] = (
                self.error_counters.get(str(type(e).__name__), 0) + 1
            )
    
    def _is_heartbeat(self, data, exchange):
        """Check if a message is a heartbeat/ping message"""
        # Common heartbeat patterns
        if isinstance(data, dict):
            # Check for common heartbeat fields
            if 'ping' in data:
                return True
            if 'op' in data and data['op'] == 'ping':
                return True
            if 'type' in data and data['type'] in ('heartbeat', 'ping'):
                return True
            if 'event' in data and data['event'] == 'heartbeat':
                return True
                
        # Check for empty messages that some exchanges use as heartbeats
        if isinstance(data, dict) and not data:
            return True
            
        # Exchange-specific heartbeat detection
        heartbeat_method = f"_is_{exchange}_heartbeat"
        if hasattr(self, heartbeat_method):
            heartbeat_checker = getattr(self, heartbeat_method)
            return heartbeat_checker(data)
            
        return False
        
    def _process_heartbeat(self, data, exchange):
        """Process a heartbeat message from an exchange"""
        # Some exchanges expect a pong response
        if isinstance(data, dict) and 'ping' in data:
            try:
                ws = self.ws_connections.get(exchange)
                if ws and not ws.closed:
                    # Create pong message (exchange-specific)
                    if exchange in ('binance', 'okex', 'huobi'):
                        pong = {'pong': data['ping']}
                    else:
                        pong = {'pong': int(time.time() * 1000)}
                        
                    # Send pong asynchronously
                    asyncio.create_task(ws.send(orjson.dumps(pong)))
            except Exception as e:
                self.logger.error(f"Error sending pong to {exchange}: {e}")
                
        # Update heartbeat timestamp
        self.last_heartbeats[exchange] = time.time_ns()
        
    def _is_control_message(self, data, exchange):
        """Check if a message is a control/status message (not market data)"""
        if isinstance(data, dict):
            # Check for common control message patterns
            if 'id' in data and 'result' in data:
                return True  # Response to a request
            if 'event' in data and data['event'] in ('info', 'status', 'subscribed'):
                return True
            if 'type' in data and data['type'] in ('subscribed', 'info', 'status'):
                return True
                
        # Exchange-specific control message detection
        control_method = f"_is_{exchange}_control"
        if hasattr(self, control_method):
            control_checker = getattr(self, control_method)
            return control_checker(data)
            
        return False
        
    def _infer_message_type(self, data, exchange):
        """Infer message type from data structure for unknown formats"""
        # Check common structures
        if isinstance(data, dict):
            # Look for common order book fields
            if ('bids' in data and 'asks' in data) or ('b' in data and 'a' in data):
                return 'orderbook'
                
            # Look for common ticker fields
            if ('last' in data or 'price' in data) and ('bid' in data or 'ask' in data):
                return 'ticker'
                
            # Look for common trade fields
            if ('price' in data or 'p' in data) and ('size' in data or 'q' in data or 'amount' in data):
                if 'side' in data or 'S' in data:
                    return 'trade'
        
        # Exchange-specific type inference
        infer_method = f"_infer_{exchange}_type"
        if hasattr(self, infer_method):
            type_inferrer = getattr(self, infer_method)
            return type_inferrer(data)
            
        return None
        
    async def _process_worker(self):
        """Worker for processing market data"""
        while not self._shutdown_event.is_set():
            try:
                # Get next item from queue
                item = await self.raw_queue.get()
                
                # Process item
                start_time = time.time_ns()
                
                # Extract key fields
                message_type = item.get('type')
                symbol = item.get('symbol')
                exchange = item.get('exchange')
                
                if not all([message_type, symbol, exchange]):
                    self.raw_queue.task_done()
                    continue
                
                # Update cross-exchange price comparison
                if message_type in ('ticker', 'trade') and 'price' in item:
                    price = item.get('price', 0.0)
                    if price > 0:
                        # Update price history
                        if symbol not in self.price_series:
                            self.price_series[symbol] = {}
                        if exchange not in self.price_series[symbol]:
                            self.price_series[symbol][exchange] = np.zeros(DATA_RETENTION_TICKS)
                            
                        # Shift array and add new price
                        self.price_series[symbol][exchange] = np.roll(self.price_series[symbol][exchange], -1)
                        self.price_series[symbol][exchange][-1] = price
                        
                        # Update cross validation
                        if symbol not in self.cross_validation:
                            self.cross_validation[symbol] = {}
                        self.cross_validation[symbol][exchange] = price
                        
                        # Calculate price deviation (only if we have enough exchanges)
                        if symbol in self.cross_validation and len(self.cross_validation[symbol]) >= 2:
                            # Calculate weighted median price
                            exchanges = list(self.cross_validation[symbol].keys())
                            prices = [self.cross_validation[symbol][e] for e in exchanges]
                            weights = [self.exchange_weights.get(e, 1.0) for e in exchanges]
                            
                            if self.has_gpu:
                                # Use GPU for weighted median
                                try:
                                    median_price = self._gpu_weighted_median(prices, weights)
                                except:
                                    # Fallback to CPU
                                    median_price = self._weighted_median(prices, weights)
                            else:
                                median_price = self._weighted_median(prices, weights)
                                
                            # Calculate deviation from median
                            deviation = abs(price - median_price) / median_price if median_price > 0 else 0
                            
                            # Mark anomaly if deviation exceeds threshold
                            if deviation > ANOMALY_THRESHOLD / 100:  # Convert percentage to fraction
                                self.anomaly_counters[exchange] = self.anomaly_counters.get(exchange, 0) + 1
                                if self.anomaly_counters[exchange] > 10:
                                    # Reduce exchange reliability
                                    self.exchange_reliability[exchange] = max(
                                        0.1, self.exchange_reliability[exchange] * 0.9
                                    )
                                    
                                    # Update weights based on reliability
                                    self._update_exchange_weights()
                                    
                                    # Blacklist exchange if reliability drops too low
                                    if self.exchange_reliability[exchange] < 0.3:
                                        self.logger.warning(
                                            f"Blacklisting {exchange} due to anomalous data "
                                            f"(reliability: {self.exchange_reliability[exchange]:.2f})"
                                        )
                                        self.blacklisted_exchanges.add(exchange)
                                        
                                if deviation > ANOMALY_THRESHOLD / 50:  # Very high deviation
                                    self.logger.warning(
                                        f"Anomalous price on {exchange} for {symbol}: "
                                        f"{price} (median: {median_price}, deviation: {deviation:.2%})"
                                    )
                
                # Update cache based on message type
                if message_type == 'ticker':
                    # Update ticker cache
                    if symbol not in self.tick_cache:
                        self.tick_cache[symbol] = {}
                    self.tick_cache[symbol][exchange] = {
                        'price': item.get('price', 0.0),
                        'bid': item.get('bid', 0.0),
                        'ask': item.get('ask', 0.0),
                        'volume': item.get('volume', 0.0),
                        'timestamp': item.get('timestamp', int(time.time() * 1000))
                    }
                    
                    # Update spread
                    bid = item.get('bid', 0.0)
                    ask = item.get('ask', 0.0)
                    if bid > 0 and ask > 0:
                        self.spread_cache[symbol] = (ask - bid) / ask
                    
                elif message_type == 'orderbook':
                    # Update orderbook cache
                    if symbol not in self.orderbook_cache:
                        self.orderbook_cache[symbol] = {}
                    self.orderbook_cache[symbol][exchange] = {
                        'bids': item.get('bids', []),
                        'asks': item.get('asks', []),
                        'timestamp': item.get('timestamp', int(time.time() * 1000))
                    }
                    
                    # Calculate rough volatility using order book depth
                    try:
                        bids = item.get('bids', [])
                        asks = item.get('asks', [])
                        
                        if bids and asks:
                            best_bid = bids[0][0]
                            best_ask = asks[0][0]
                            mid_price = (best_bid + best_ask) / 2
                            
                            # Calculate depth at 1% from mid price
                            depth_boundary = mid_price * 0.01
                            bid_depth = sum(b[1] for b in bids if mid_price - b[0] <= depth_boundary)
                            ask_depth = sum(a[1] for a in asks if a[0] - mid_price <= depth_boundary)
                            
                            # Volatility is inversely proportional to depth
                            total_depth = bid_depth + ask_depth
                            if total_depth > 0:
                                self.volatility_cache[symbol] = 1.0 / total_depth
                            else:
                                self.volatility_cache[symbol] = 1.0
                    except Exception:
                        pass
                    
                elif message_type == 'trade':
                    # Update trade cache
                    if symbol not in self.trade_cache:
                        self.trade_cache[symbol] = deque(maxlen=DATA_RETENTION_TICKS)
                    
                    self.trade_cache[symbol].append({
                        'price': item.get('price', 0.0),
                        'volume': item.get('volume', 0.0),
                        'side': item.get('side', 'unknown'),
                        'timestamp': item.get('timestamp', int(time.time() * 1000)),
                        'exchange': exchange
                    })
                    
                    # Calculate volatility from recent trades
                    # (standard deviation of price changes)
                    if len(self.trade_cache[symbol]) >= 10:
                        prices = [t['price'] for t in self.trade_cache[symbol]]
                        if self.has_gpu:
                            try:
                                volatility = float(cp.std(cp.array(prices)) / cp.mean(cp.array(prices)))
                            except:
                                volatility = float(np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0)
                        else:
                            volatility = float(np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0)
                        
                        self.volatility_cache[symbol] = volatility
                
                # Add to persistence queue if successful
                await self.persist_queue.put(item)
                
                # Calculate processing latency
                latency = (time.time_ns() - start_time) / 1e6  # ns to ms
                self.processing_latency.append(latency)
                
                # Mark task as done
                self.raw_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in process worker: {e}")
                # Mark task as done even if it failed
                try:
                    self.raw_queue.task_done()
                except:
                    pass
    
    def _weighted_median(self, data, weights):
        """Calculate weighted median of a dataset"""
        if not data or not weights:
            return 0.0
            
        # Sort data and weights together
        sorted_pairs = sorted(zip(data, weights))
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.0
            
        # Find the weighted median
        cumulative_weight = 0
        for value, weight in sorted_pairs:
            cumulative_weight += weight
            if cumulative_weight >= total_weight / 2:
                return value
                
        return 0.0
        
    def _gpu_weighted_median(self, data, weights):
        """Calculate weighted median using GPU acceleration"""
        if not data or not weights or not self.has_gpu:
            return self._weighted_median(data, weights)
            
        # Convert to GPU arrays
        data_gpu = cp.array(data)
        weights_gpu = cp.array(weights)
        
        # Sort data and get indices
        sorted_indices = cp.argsort(data_gpu)
        
        # Reorder weights according to sorted data
        sorted_weights = weights_gpu[sorted_indices]
        
        # Compute cumulative weights
        cumulative_weights = cp.cumsum(sorted_weights)
        
        # Find the weighted median
        median_idx = cp.searchsorted(cumulative_weights, cp.sum(weights_gpu) / 2)
        
        # Get the value at the median index
        if median_idx < len(data):
            return float(data_gpu[sorted_indices[median_idx]])
        else:
            return 0.0
    
    def _update_exchange_weights(self):
        """Update exchange weights based on reliability metrics"""
        # Calculate total reliability
        total_reliability = sum(self.exchange_reliability.values())
        
        if total_reliability > 0:
            # Update weights proportionally to reliability
            for exchange in self.exchange_reliability:
                self.exchange_weights[exchange] = (
                    self.exchange_reliability[exchange] / total_reliability
                )
        else:
            # Reset to equal weights if all reliabilities are 0
            count = len(self.exchange_reliability)
            if count > 0:
                weight = 1.0 / count
                for exchange in self.exchange_reliability:
                    self.exchange_weights[exchange] = weight
    
    async def _persistence_worker(self):
        """Worker for persisting market data"""
        while not self._shutdown_event.is_set():
            try:
                # Get next item from queue
                item = await self.persist_queue.get()
                                # Process item with nanosecond precision
                start_time = time.time_ns()
                
                try:
                    # Serialize data with zero-copy optimizations
                    serialized = msgpack.packb(item, use_bin_type=True)
                    compressed = zstd.compress(serialized, level=3)
                    
                    # Generate unique key using symbol + timestamp + hash
                    key = f"{item['symbol']}-{item['timestamp']}-{xxhash.xxh64(serialized).hexdigest()}".encode()
                    
                    # Atomic write to RocksDB with WAL enabled
                    if self.rocks_db:
                        self.rocks_db.put(key, compressed)
                    
                    # Update Redis with latest data using pipelining
                    async with await self.redis_pool as r:
                        pipe = r.pipeline()
                        
                        # Store in sorted set for time-series access
                        pipe.zadd(
                            f"apex:data:{item['symbol']}:{item['exchange']}:{item['type']}",
                            {compressed: item['timestamp']}
                        )
                        
                        # Update volatility cache
                        if item['type'] == 'trade' and 'volatility' in item:
                            pipe.hset(
                                f"apex:volatility:{item['symbol']}",
                                mapping={str(item['timestamp']): item['volatility']}
                            )
                        
                        # Execute pipeline
                        await pipe.execute()
                    
                    # Publish to Kafka for downstream systems
                    if self.kafka_producer:
                        await self.kafka_producer.send(
                            topic=f"apex.market.{item['type']}",
                            key=key,
                            value=compressed,
                            headers=[
                                ("symbol", item['symbol'].encode()),
                                ("exchange", item['exchange'].encode()),
                                ("type", item['type'].encode())
                            ]
                        )
                    
                    # Update shared memory buffer for HFT strategies
                    if self.shm:
                        try:
                            # Use memoryview for zero-copy buffer access
                            buffer = memoryview(self.shm.buf)
                            start = self.shm_index % DATA_BUFFER_SIZE
                            end = start + len(compressed)
                            
                            if end > DATA_BUFFER_SIZE:
                                # Handle wrap-around
                                buffer[start:DATA_BUFFER_SIZE] = compressed[:DATA_BUFFER_SIZE-start]
                                buffer[0:end-DATA_BUFFER_SIZE] = compressed[DATA_BUFFER_SIZE-start:]
                            else:
                                buffer[start:end] = compressed
                            
                            self.shm_index = end % DATA_BUFFER_SIZE
                        except Exception as e:
                            self.logger.error(f"Shared memory write error: {e}")
                
                except rocksdb.RocksDBError as dbe:
                    self.logger.error(f"RocksDB write error: {dbe}")
                    self.error_counters["rocksdb"] = self.error_counters.get("rocksdb", 0) + 1
                except redis.RedisError as re:
                    self.logger.error(f"Redis update error: {re}")
                    self.error_counters["redis"] = self.error_counters.get("redis", 0) + 1
                except Exception as e:
                    self.logger.error(f"Persistence error: {e}")
                    self.error_counters["persistence"] = self.error_counters.get("persistence", 0) + 1
                
                # Record serialization latency
                latency = (time.time_ns() - start_time) / 1e6  # Convert to ms
                self.serialization_latency.append(latency)
                
                # Mark task as done
                self.persist_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Critical persistence failure: {e}")
                try:
                    self.persist_queue.task_done()
                except:
                    pass
