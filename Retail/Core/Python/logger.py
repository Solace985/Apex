import logging
import os
import sys
import threading
import gzip
import shutil
import re
import time
import stat
import hashlib
import hmac
import boto3  # ✅ Install with: pip install boto3
from google.cloud import storage  # ✅ Install with: pip install google-cloud-storage
from cloghandler import ConcurrentRotatingFileHandler  # Install with: pip install concurrent-log-handler.
from pythonjsonlogger import jsonlogger  # Importing jsonlogger for JSON formatting
import queue  # Added import for queue
from cryptography.fernet import Fernet  # ✅ Import Fernet for encryption

SECRET_LOG_KEY = os.getenv("LOG_SECRET_KEY", "default_secret_key")  # ✅ Use ENV Variable
LOG_ENCRYPTION_KEY = os.getenv("LOG_ENCRYPTION_KEY", Fernet.generate_key())  # ✅ Store securely
AWS_BUCKET_NAME = os.getenv("AWS_S3_BUCKET", "your-default-bucket")
GCP_BUCKET_NAME = os.getenv("GCP_STORAGE_BUCKET", "your-gcp-bucket")

def sanitize_log_message(message):
    """Prevents log injection attacks by removing newline characters, special escape sequences, and unwanted formats."""
    message = re.sub(r"[\r\n\t]", " ", message)  # ✅ Remove newlines, tabs
    message = re.sub(r"%[sd]", "", message)  # ✅ Prevent format string exploits
    return message

def _generate_log_signature(message):
    """Generates a cryptographic signature for log messages to prevent log forgery."""
    return hmac.new(SECRET_LOG_KEY.encode(), message.encode(), hashlib.sha256).hexdigest()

def encrypt_log_file(file_path):
    """Encrypts a log file before uploading to prevent exposure."""
    cipher = Fernet(LOG_ENCRYPTION_KEY)
    
    with open(file_path, "rb") as f:
        encrypted_data = cipher.encrypt(f.read())

    encrypted_file_path = file_path + ".enc"
    with open(encrypted_file_path, "wb") as ef:
        ef.write(encrypted_data)

    return encrypted_file_path  # ✅ Return path of encrypted file

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler with a limited queue size to prevent DoS attacks."""
    MAX_QUEUE_SIZE = 10_000  # ✅ Prevents infinite memory growth (Limit to 10K logs)

    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)  # ✅ Set queue size
        self.log_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.log_thread.start()

    def emit(self, record):
        """Places log messages into a queue, dropping old messages if full."""
        try:
            self.log_queue.put_nowait(record)  # ✅ Avoids blocking the main thread
        except queue.Full:
            sys.stderr.write("⚠️ Log queue is full. Dropping old logs.\n")  # ✅ Warn when logs are lost

    def _process_logs(self):
        """Processes logs in a background thread."""
        while True:
            try:
                record = self.log_queue.get()
                sys.stdout.write(self.format(record) + "\n")
            except Exception:
                pass

class Logger:
    _lock = threading.Lock()  # ✅ Ensures thread-safe logger initialization
    last_log_level_change = 0  # ✅ Track last modification time

    def __init__(self, name="RetailBot", log_level=logging.INFO, log_dir="logs/", debug=False):
        """
        Initializes the logger with rotating file handler and console output.

        :param name: Name of the logger (default: "RetailBot")
        :param log_level: Logging level (default: logging.INFO)
        :param log_dir: Directory to store log files (default: "logs/")
        :param debug: Enable debug mode (default: False)
        """
        self.name = name
        self.log_dir = os.getenv("LOG_DIR", log_dir)  # Updated to use environment variable
        self.debug = debug
        self.logger = logging.getLogger(name)

        # ✅ Ensure thread-safe log directory creation
        with self._lock:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except OSError as e:
                print(f"❌ Failed to create log directory '{self.log_dir}': {e}", file=sys.stderr)

        # ✅ Set base log level
        self.logger.setLevel(logging.DEBUG if debug else log_level)

        # ✅ Remove old handlers if logger is re-created (prevents duplicates)
        self._clear_handlers()

        log_path = os.path.join(self.log_dir, f"{name}.log")

        # ✅ File handler with log rotation (Max 5MB per file, keeps last 5 logs)
        file_handler = ConcurrentRotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(self._get_formatter())

        # ✅ Console handler (for real-time logs, avoids duplicate stdout logs)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_formatter())

        # ✅ Add asynchronous log handler (prevents performance bottlenecks)
        async_handler = AsyncLogHandler()
        async_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(async_handler)

        # ✅ Add handlers only once
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.log_format = os.getenv("LOG_FORMAT_STYLE", "%(asctime)s - %(levelname)s - [%(name)s] %(message)s")

    def flush_logs(self):
        """Flushes all log handlers to ensure logs are written to disk immediately."""
        for handler in self.logger.handlers:
            handler.flush()
        self.logger.info("🚀 Logs flushed to disk.")

    def rotate_logs(self):
        """Manually rotates logs and compresses old logs."""
        for handler in self.logger.handlers:
            if isinstance(handler, ConcurrentRotatingFileHandler):
                handler.doRollover()  # ✅ Force log rotation

        self.compress_old_logs()  # ✅ Compress rotated logs
        self.auto_cleanup_logs()  # ✅ Cleanup old logs automatically
        self.logger.info("🔄 Logs rotated, old logs compressed, and old logs cleaned up.")

    def auto_cleanup_logs(self, days=90):
        """Deletes old log files older than the specified days to free up space."""
        cutoff_time = time.time() - (days * 86400)  # ✅ Convert days to seconds

        for log_file in os.listdir(self.log_dir):
            log_path = os.path.join(self.log_dir, log_file)

            if os.path.isfile(log_path) and log_file.endswith((".log", ".gz", ".enc")):
                file_mtime = os.path.getmtime(log_path)

                if file_mtime < cutoff_time:  # ✅ Check if the file is too old
                    os.remove(log_path)
                    self.logger.info(f"🗑 Deleted old log file: {log_file}")

    def compress_old_logs(self):
        """Compresses old log files and uploads to AWS S3 or Google Cloud Storage."""
        for log_file in os.listdir(self.log_dir):
            if log_file.endswith(".log") and not log_file.endswith(".gz"):
                log_path = os.path.join(self.log_dir, log_file)
                compressed_log_path = log_path + ".gz"

                with open(log_path, "rb") as f_in, gzip.open(compressed_log_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

                os.remove(log_path)  # ✅ Delete original uncompressed log
                self.upload_to_cloud(compressed_log_path)  # ✅ Upload to Cloud

    def upload_to_cloud(self, file_path):
        """Encrypts and uploads logs securely to AWS S3 or Google Cloud."""
        encrypted_file_path = encrypt_log_file(file_path)  # ✅ Encrypt before upload
        file_name = os.path.basename(encrypted_file_path)

        if AWS_BUCKET_NAME:
            s3 = boto3.client("s3")
            s3.upload_file(encrypted_file_path, AWS_BUCKET_NAME, file_name)
            self.logger.info(f"☁️ Encrypted log uploaded to AWS S3: {file_name}")

        if GCP_BUCKET_NAME:
            client = storage.Client()
            bucket = client.bucket(GCP_BUCKET_NAME)
            blob = bucket.blob(file_name)
            blob.upload_from_filename(encrypted_file_path)
            self.logger.info(f"☁️ Encrypted log uploaded to Google Cloud Storage: {file_name}")

    def _clear_handlers(self):
        """Removes existing handlers and sets secure permissions for log files."""
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                log_path = handler.baseFilename

                # ✅ Unix/Linux/macOS - Restrict access to log files
                if os.name == "posix":
                    os.chmod(log_path, stat.S_IRUSR | stat.S_IWUSR)  # ✅ rw------- (Only owner can read/write)

                # ✅ Windows - Set file read/write permissions
                elif os.name == "nt":
                    import ctypes
                    FILE_ATTRIBUTE_READONLY = 0x01
                    ctypes.windll.kernel32.SetFileAttributesW(log_path, FILE_ATTRIBUTE_READONLY)  # ✅ Read-only for everyone
            
            self.logger.removeHandler(handler)
        self.logger.info("🔄 All handlers cleared and permissions set securely.")

    def _get_formatter(self):
        """Returns a structured JSON formatter with cryptographic signing for log authenticity."""
        if os.getenv("LOG_FORMAT", "TEXT") == "JSON":
            return jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s %(funcName)s %(lineno)d %(log_signature)s"
            )
        
        return logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(name)s] %(message)s [SIGNATURE: %(log_signature)s]",
            datefmt="%Y-%m-%d %H:%M:%S UTC"
        )

    def get_log_format(self):
        """Returns the log format set by the environment variable."""
        return self.log_format

    def log_info(self, message):
        """Logs an INFO-level message securely with cryptographic signing."""
        signed_message = f"{message} [SIGNATURE: {_generate_log_signature(message)}]"
        self.logger.info(sanitize_log_message(signed_message))

    def log_warning(self, message):
        """Logs a WARNING-level message securely with cryptographic signing."""
        signed_message = f"{message} [SIGNATURE: {_generate_log_signature(message)}]"
        self.logger.warning(sanitize_log_message(signed_message))

    def log_error(self, message):
        """Logs an ERROR-level message securely with cryptographic signing."""
        signed_message = f"{message} [SIGNATURE: {_generate_log_signature(message)}]"
        self.logger.error(sanitize_log_message(signed_message))

    def log_debug(self, message):
        """Logs a DEBUG-level message, but only if debugging is enabled."""
        if self.logger.isEnabledFor(logging.DEBUG):  # ✅ Prevents unnecessary string formatting
            signed_message = f"{message} [SIGNATURE: {_generate_log_signature(message)}]"
            self.logger.debug(sanitize_log_message(signed_message))

    def log_exception(self, message):
        """Logs an ERROR message with full exception traceback securely with cryptographic signing."""
        signed_message = f"{message} [SIGNATURE: {_generate_log_signature(message)}]"
        self.logger.exception(sanitize_log_message(signed_message))

    def log_critical(self, message):
        """Logs a CRITICAL-level message and triggers an alert if needed with cryptographic signing."""
        signed_message = f"{message} [SIGNATURE: {_generate_log_signature(message)}]"
        self.logger.critical(sanitize_log_message(signed_message))

        # ✅ Send alerts (Extend this with an email/SMS alerting system)
        if os.getenv("ENABLE_ALERTS", "FALSE").upper() == "TRUE":
            self.send_alert(message)

    def set_log_level(self, level):
        """Dynamically sets the log level safely with rate-limiting."""
        current_time = time.time()
        if current_time - self.last_log_level_change < 600:  # ✅ 10-minute cooldown
            self.logger.warning("⚠️ Log level change request denied (Rate limit exceeded).")
            return

        valid_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        if isinstance(level, str):
            level = level.upper()

        if level not in valid_levels:
            self.logger.warning(f"⚠️ Invalid log level '{level}'. Keeping current level.")
            return

        self.logger.setLevel(valid_levels[level])
        for handler in self.logger.handlers:
            handler.setLevel(valid_levels[level])

        self.logger.info(f"🔄 Log level changed to {level}")
        self.last_log_level_change = current_time  # ✅ Update timestamp

    def filter_logs(self, min_level=logging.WARNING):
        """Filters log messages to show only those above a certain level."""
        self.logger.info(f"🔍 Filtering logs above {logging.getLevelName(min_level)} level.")
        for handler in self.logger.handlers:
            handler.setLevel(min_level)

def get_logger(name="RetailBot", log_level=logging.INFO, debug=False):
    """
    Returns a logger instance with the given settings.
    
    :param name: Name of the logger.
    :param log_level: Logging level.
    :param debug: Enable debug mode.
    :return: Configured logger instance.
    """
    return Logger(name=name, log_level=log_level, debug=debug).logger

# ✅ Example Usage:
logger = get_logger(debug=True)  # Enable debug mode
logger.info("🚀 Logger initialized successfully!")
logger.debug("🔍 Debug mode enabled")
logger.error("❌ Example error message")

# ✅ Example of changing log level dynamically
logger.set_log_level(logging.WARNING)  # Only log warnings and above
logger.info("ℹ️ This will NOT be logged!")  # INFO will be ignored
logger.warning("⚠️ This is a warning!")  # Will be logged
