import logging
import json
from brokers.broker_factory import BrokerFactory
from core.config import Config

# This file runs the bot by calling the retail_core.py file that is the core archetecture of the bot.

def suggest_broker():
    """Suggests the best broker based on multiple factors."""
    # Dummy logic: Selects first broker from config
    config = Config()
    suggested_broker = config.supported_brokers[0] if config.supported_brokers else "dummy"

    print(f"⚡ Suggested Broker: {suggested_broker}")
    return suggested_broker

def get_user_selected_broker():
    """Prompts user for broker selection."""
    config = Config()
    print(f"Supported Brokers: {', '.join(config.supported_brokers)}")
    
    broker_name = input("Enter your preferred broker (or press Enter to use suggested): ").strip()
    return broker_name if broker_name else suggest_broker()

def generate_api_keys(broker_name):
    """Generates API keys dynamically for the selected broker."""
    print(f"Generating API keys for {broker_name}...")
    # Dummy API key generation
    return {"api_key": f"generated_key_{broker_name}", "api_secret": f"generated_secret_{broker_name}"}

def main():
    config = Config()
    selected_broker = get_user_selected_broker()
    
    print(f"✅ Integrating with {selected_broker}...")
    
    # Generate API keys dynamically
    api_keys = generate_api_keys(selected_broker)
    
    # Store API keys in a config.json file (not hardcoded)
    with open("api_config.json", "w") as file:
        json.dump(api_keys, file)
    
    print(f"✅ API Integration Complete! Trading will now begin on {selected_broker}")

if __name__ == "__main__":
    main()
