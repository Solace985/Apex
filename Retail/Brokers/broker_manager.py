import json
import requests
import os
import logging
import webbrowser

logger = logging.getLogger(__name__)

class BrokerManager:
    """Handles broker selection and OAuth authentication."""

    def __init__(self):
        self.user_data_file = "config/user_broker.json"
        self.broker_file = "config/brokers.json"
        self.user_broker_file = "config/user_brokers.json"

    def load_brokers(self):
        """Loads all brokers (default + user-defined)"""
        try:
            with open(self.broker_file, "r") as file:
                brokers = json.load(file)
        except FileNotFoundError:
            brokers = {}

        try:
            with open(self.user_broker_file, "r") as file:
                user_brokers = json.load(file)
        except FileNotFoundError:
            user_brokers = {}

        brokers.update(user_brokers)  # Merge user-defined brokers with default ones
        return brokers

    def add_broker(self, broker_name, base_url, auth_type, oauth_url=None, endpoints=None):
        """Allows users to add a new broker dynamically."""
        brokers = self.load_brokers()

        if broker_name in brokers:
            return {"status": "FAILED", "error": "Broker already exists"}

        new_broker = {
            "base_url": base_url,
            "auth_type": auth_type,
            "oauth_url": oauth_url,
            "endpoints": endpoints or {
                "place_order": "/orders",
                "get_balance": "/balance"
            }
        }

        # Save new broker to user_brokers.json
        brokers[broker_name] = new_broker
        with open(self.user_broker_file, "w") as file:
            json.dump(brokers, file, indent=4)

        return {"status": "SUCCESS", "message": f"Broker {broker_name} added successfully!"}

    def get_broker(self, broker_name):
        """Returns broker details."""
        brokers = self.load_brokers()
        return brokers.get(broker_name, None)

    def get_oauth_url(self, broker_name):
        """Fetch OAuth URL from brokers.json."""
        with open("config/brokers.json", "r") as file:
            brokers = json.load(file)
        
        broker_data = brokers.get(broker_name, {})
        return broker_data.get("oauth_url", None)

    def authenticate_broker(self, user_id, broker_name):
        """Redirects user to broker’s OAuth page and stores authentication token."""
        oauth_url = self.get_oauth_url(broker_name)
        
        if not oauth_url:
            logger.error(f"❌ OAuth not supported for broker: {broker_name}")
            return {"status": "FAILED", "error": "OAuth not available for this broker"}

        logger.info(f"🔗 Redirecting user to OAuth login: {oauth_url}")
        webbrowser.open(oauth_url)  # Opens OAuth URL in browser

        # Simulate receiving an access token (In real-world, use a callback URL)
        access_token = input(f"🔑 Enter the OAuth token from {broker_name}: ")

        if not access_token:
            logger.error("❌ Authentication failed: No token received")
            return {"status": "FAILED", "error": "Authentication failed"}

        # Save credentials
        self.save_broker(user_id, broker_name, access_token)

        logger.info(f"✅ Authentication successful for {broker_name}")
        return {"status": "SUCCESS", "broker": broker_name}

    def save_broker(self, user_id, broker_name, access_token):
        """Saves the user's selected broker and OAuth token."""
        user_data = {
            "broker_name": broker_name,
            "access_token": access_token
        }
        with open(self.user_data_file, "w") as file:
            json.dump({user_id: user_data}, file)

    def load_broker(self, user_id):
        """Loads the user's selected broker and credentials."""
        with open(self.user_data_file, "r") as file:
            data = json.load(file)
            return data.get(user_id, None)
