# src/ai/sentiment_analysis.py
import re
import json
import time
import numpy as np
from typing import Dict, Any, Deque
from datetime import datetime
from collections import deque
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from core.trading.risk.config import load_config  # Use existing config
from utils.logging.structured_logger import StructuredLogger  # Use existing logger
from utils.helpers.error_handler import handle_api_error  # Use existing error handler

class SentimentAnalyzer:
    """Integrated market sentiment analysis pipeline with fail-safes"""
    
    def __init__(self, market_data_bus: Any):
        self.config = load_config()['sentiment']
        self.logger = StructuredLogger(__name__)
        self.market_data_bus = market_data_bus  # Shared data bus with trading core
        
        # Initialize models with safety checks
        self._init_nlp_models()
        
        # Configure rolling windows from config
        self.sentiment_queues = {
            'news': deque(maxlen=self.config['window_size']),
            'social': deque(maxlen=self.config['window_size'])
        }

    def _init_nlp_models(self):
        """Safe model initialization with fallbacks"""
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.nlp = pipeline("text-classification", 
                              model="ProsusAI/finbert",
                              use_auth_token=self.config.get('hf_token', ''))
        except Exception as e:
            self.logger.critical("NLP model initialization failed", error=str(e))
            raise RuntimeError("Sentiment analysis unavailable")

    def _clean_text(self, text: str) -> str:
        """Sanitize input text with aggressive filtering"""
        text = re.sub(r"http\S+|@\w+|#\w+", "", text)  # Remove links/mentions/hashtags
        text = re.sub(r"[^a-zA-Z0-9\s\.\-,]", "", text)  # Allow basic punctuation
        return text.strip()[:512]  # Truncate for model safety

    @handle_api_error(retries=3, cooldown=5)
    def _fetch_news(self) -> list:
        """Fetch news using configured sources with circuit breaker"""
        from core.data.fetch_data import NewsFetcher  # Use existing data module
        
        fetcher = NewsFetcher(
            sources=self.config['news_sources'],
            api_keys=self.config['api_keys']
        )
        return fetcher.fetch_news()

    def _analyze_content(self, text: str) -> float:
        """Robust sentiment scoring with model consensus"""
        text = self._clean_text(text)
        if not text:
            return 0.0

        # Get model consensus
        vader_score = self.sia.polarity_scores(text)["compound"]
        try:
            finbert = self.nlp(text)[0]
            finbert_score = finbert['score'] * (1 if finbert['label'] == 'positive' else -1)
        except Exception as e:
            self.logger.warning("FinBERT failed, using VADER only", error=str(e))
            finbert_score = 0

        return (vader_score + finbert_score) / 2

    def _get_social_sentiment(self) -> float:
        """Integrated social analysis using existing data feeds"""
        from services.web.social_features import SocialMonitor  # Use existing social module
        
        monitor = SocialMonitor()
        posts = monitor.get_recent_posts(limit=50)
        scores = [self._analyze_content(post['content']) for post in posts]
        return np.mean(scores) if scores else 0.0

    def update_market_sentiment(self):
        """Main analysis loop integrated with market data bus"""
        try:
            # Get news sentiment
            news_items = self._fetch_news()
            news_scores = [self._analyze_content(item['title'] + ' ' + item['summary']) 
                         for item in news_items]
            news_avg = np.mean(news_scores) if news_scores else 0.0
            
            # Get social sentiment
            social_avg = self._get_social_sentiment()
            
            # Update rolling windows
            self.sentiment_queues['news'].append(news_avg)
            self.sentiment_queues['social'].append(social_avg)
            
            # Calculate weighted score
            weights = self.config['weights']
            combined_score = (
                weights['news'] * np.mean(self.sentiment_queues['news']) +
                weights['social'] * np.mean(self.sentiment_queues['social'])
            )
            
            # Update shared market data
            self.market_data_bus.update({
                'sentiment': {
                    'score': combined_score,
                    'confidence': self._calculate_confidence(),
                    'timestamp': datetime.utcnow().isoformat()
                }
            })
            
            self.logger.info("Market sentiment updated", 
                           score=combined_score,
                           source_counts=len(news_items))
            
        except Exception as e:
            self.logger.error("Sentiment update failed", error=str(e))
            # Provide neutral signal on failure
            self.market_data_bus.update({'sentiment': {'score': 0.0, 'confidence': 'LOW'}})

    def _calculate_confidence(self) -> str:
        """Dynamic confidence scoring based on variance and volume"""
        variances = {
            'news': np.var(list(self.sentiment_queues['news'])),
            'social': np.var(list(self.sentiment_queues['social']))
        }
        avg_variance = np.mean(list(variances.values()))
        
        if avg_variance < 0.05:
            return 'HIGH'
        elif avg_variance < 0.15:
            return 'MEDIUM'
        return 'LOW'

    def get_current_sentiment(self) -> Dict[str, Any]:
        """Get latest sentiment from shared data bus"""
        return self.market_data_bus.get('sentiment', {})