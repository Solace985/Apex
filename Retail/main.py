import logging
import json
from Brokers.broker_factory import BrokerFactory
from Retail.Core.Python.config import load_config
from Retail.AI_Models.ai_engine_wrapper import predict_lstm  # Importing the LSTM prediction function
from AI_Models.ai_engine_wrapper import predict

# Load configuration at startup
config = load_config()

print(f"Starting bot in {config.mode} mode...")  # Logs the current trading mode

def suggest_broker():
    """Suggests the best broker based on multiple factors."""
    # Dummy logic: Selects first broker from config
    suggested_broker = config.supported_brokers[0] if config.supported_brokers else "dummy"

    print(f"⚡ Suggested Broker: {suggested_broker}")
    return suggested_broker

def get_user_selected_broker():
    """Prompts user for broker selection."""
    print(f"Supported Brokers: {', '.join(config.supported_brokers)}")
    
    broker_name = input("Enter your preferred broker (or press Enter to use suggested): ").strip()
    return broker_name if broker_name else suggest_broker()

def generate_api_keys(broker_name):
    """Generates API keys dynamically for the selected broker."""
    print(f"Generating API keys for {broker_name}...")
    # Dummy API key generation
    return {"api_key": f"generated_key_{broker_name}", "api_secret": f"generated_secret_{broker_name}"}

def main():
    selected_broker = get_user_selected_broker()
    
    print(f"✅ Integrating with {selected_broker}...")
    
    # Generate API keys dynamically
    api_keys = generate_api_keys(selected_broker)
    
    # Store API keys in a config.json file (not hardcoded)
    with open("api_config.json", "w") as file:
        json.dump(api_keys, file)
    
    print(f"✅ API Integration Complete! Trading will now begin on {selected_broker}")

# Choose a model dynamically
model_choice = "maddpg"  # Can be "lstm", "cnn", "qlearning", etc.

example_input = [[0.1] * 8] * 30
prediction = predict(model_choice, example_input)

print(f"Prediction using {model_choice}: {prediction}")

if __name__ == "__main__":
    main()
