import os
import yaml

class Config:
    """Loads bot configuration from a YAML file dynamically."""

    def __init__(self, config_file="../config.yaml"):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # Load configurations from YAML
        self.data_feed_interval = config.get("data_feed_interval", 1)
        self.risk_threshold = config.get("risk_threshold", 0.02)
        self.database_path = config.get("database_path", "storage/trade_history.db")
        self.default_broker = config.get("default_broker", "dummy")  
        self.supported_brokers = config.get("supported_brokers", [])
        self.api_config = config.get("api_config", {})

    def get_broker_api_keys(self, broker_name):
        """Fetch API keys for the selected broker."""
        return self.api_config.get(broker_name, {})
