import numpy as np
from typing import Dict, Any, Deque, List
from datetime import datetime, timedelta
from collections import deque
from sklearn.linear_model import LinearRegression
from Core.data.realtime.fetch_data import MacroDataFetcher, InstitutionalDataFetcher
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import handle_api_error
from src.ai.forecasting.sentiment_analysis import SentimentAnalyzer
from src.Core.trading.ai.config import load_config
from Core.trading.execution.retail_core import market_data_bus


class FundamentalAnalysis:
    """Enhanced fundamental analysis with all original features preserved"""
    
    def __init__(self, market_data_bus: Any):
        self.config = self._load_config()
        self.logger = StructuredLogger(__name__)
        self.market_data_bus = market_data_bus
        self.data_fetcher = MacroDataFetcher()
        self.institutional_fetcher = InstitutionalDataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer(market_data_bus)
        
        # State management
        self.cache = {}
        self.sentiment_window = deque(maxlen=self.config['sentiment_window'])
        self.historical_macro_data = deque(maxlen=500)
        self._last_update = datetime.min

    def _load_config(self) -> Dict[str, Any]:
        return load_config()['fundamental_analysis']

    @handle_api_error(retries=3, cooldown=5)
    def _fetch_cached_data(self, endpoint: str) -> Any:
        """Enhanced caching with TTL"""
        if endpoint in self.cache:
            cached = self.cache[endpoint]
            if datetime.utcnow() < cached['expiry']:
                return cached['data']
        return None

    def _store_cache(self, endpoint: str, data: Any, ttl: int = 1800):
        """Universal caching method"""
        self.cache[endpoint] = {
            'data': data,
            'expiry': datetime.utcnow() + timedelta(seconds=ttl)
        }

    def _compute_moving_average(self) -> float:
        """Preserved moving average calculation"""
        return np.mean(self.sentiment_window) if self.sentiment_window else 0.0

    def fetch_news_sentiment(self) -> float:
        """Enhanced news sentiment with original moving average logic"""
        try:
            articles = self.data_fetcher.get_financial_news(limit=5)
            sentiments = [self.sentiment_analyzer.analyze_text(a['title'] + ' ' + a['summary']) 
                        for a in articles]
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            self.sentiment_window.append(avg_sentiment)
            return self._compute_moving_average()
        except Exception as e:
            self.logger.error("News sentiment failed", error=str(e))
            return 0.0

    def fetch_macro_factors(self) -> Dict[str, float]:
        """Preserved macro factors with enhanced caching"""
        cached = self._fetch_cached_data('macro_factors')
        if cached: return cached
        
        data = self.data_fetcher.get_macro_indicators()
        self._store_cache('macro_factors', data)
        self.historical_macro_data.append(data)
        return data

    def fetch_institutional_activity(self) -> Dict[str, float]:
        """Preserved institutional data with dark pool analysis"""
        cached = self._fetch_cached_data('institutional_activity')
        if cached: return cached
        
        data = self.institutional_fetcher.get_institutional_flows()
        enhanced_data = {
            **data,
            'dark_pool_volume': data.get('dark_pool', 0.0)
        }
        self._store_cache('institutional_activity', enhanced_data)
        return enhanced_data

    def get_smart_money_trend(self, data: Dict[str, float]) -> str:
        """Preserved original smart money logic"""
        net_inflow = (
            data["Whale Buying"] - data["Whale Selling"] +
            0.5 * data["Hedge Fund Activity"] -
            0.5 * data["Retail Sentiment"]
        )
        return "BULLISH" if net_inflow > 0 else "BEARISH"

    def _normalize_weights(self, values: Dict[str, float]) -> Dict[str, float]:
        """Preserved weight normalization with volatility scaling"""
        volatility = self._get_historical_volatility()
        total = sum(abs(v) for v in values.values()) + 1e-10
        normalized = {k: v/total for k, v in values.items()}
        
        # Volatility adjustment
        return {
            k: v * (1 + volatility.get(k, 0))
            for k, v in normalized.items()
        }

    def analyze_fundamentals(self) -> Dict[str, Any]:
        """Preserved composite scoring logic"""
        macro_data = self.fetch_macro_factors()
        institutional_data = self.fetch_institutional_activity()
        
        macro_score = sum(
            self.config['macro_weights'][k] * macro_data[k] 
            for k in self.config['macro_weights']
        )
        
        institutional_score = (
            0.4 * institutional_data['dark_pool_volume'] +
            0.3 * institutional_data['Hedge Fund Activity'] -
            0.3 * institutional_data['Retail Sentiment']
        )
        
        sentiment_score = self.fetch_news_sentiment()
        
        composite = (
            self.config['weights']['macro'] * macro_score +
            self.config['weights']['institutional'] * institutional_score +
            self.config['weights']['sentiment'] * sentiment_score
        )
        
        return {
            'macro_score': macro_score,
            'institutional_score': institutional_score,
            'sentiment_score': sentiment_score,
            'composite_score': composite
        }

    def detect_market_risk(self) -> str:
        """Preserved risk detection with historical data"""
        if not self.historical_macro_data:
            return "LOW"
            
        current = self.fetch_macro_factors()
        means = {k: np.mean([d[k] for d in self.historical_macro_data]) 
               for k in current}
        stds = {k: np.std([d[k] for d in self.historical_macro_data]) 
              for k in current}
        
        risk_score = sum(
            1 for k in ['Inflation', 'Interest Rate', 'GDP Growth']
            if abs(current[k] - means[k]) > stds[k]
        )
        return "HIGH" if risk_score >=2 else "MEDIUM" if risk_score==1 else "LOW"

    def detect_market_shocks(self) -> List[str]:
        """Preserved Z-score shock detection"""
        shocks = []
        current = self.fetch_macro_factors()
        
        for factor in ['Inflation', 'Interest Rate', 'GDP Growth']:
            history = [d[factor] for d in self.historical_macro_data]
            if len(history) < 10: continue
            
            z = (current[factor] - np.mean(history)) / (np.std(history) + 1e-10)
            if abs(z) > 2.0:
                shocks.append(f"{factor}_SHOCK")
        
        return shocks

    def forecast_fundamental_trend(self) -> str:
        """Preserved linear regression forecasting"""
        if len(self.historical_macro_data) < 10:
            return "NEUTRAL"
            
        X = np.arange(len(self.historical_macro_data)).reshape(-1, 1)
        scores = []
        
        for factor in ['GDP Growth', 'Inflation', 'Interest Rate']:
            y = np.array([d[factor] for d in self.historical_macro_data])
            model = LinearRegression().fit(X, y)
            scores.append(model.coef_[0])
        
        weighted_score = (
            0.5 * scores[0] - 
            0.3 * scores[1] - 
            0.2 * scores[2]
        )
        
        return "BULLISH" if weighted_score >0.5 else "BEARISH" if weighted_score<-0.5 else "NEUTRAL"

    def update_market_data(self):
        """Update the market data bus with fundamental insights"""
        if datetime.utcnow() - self._last_update < timedelta(minutes=5):
            return  # Prevent frequent updates
        
        try:
            analysis = self.analyze_fundamentals()
            analysis.update({
                'risk_level': self.detect_market_risk(),
                'shocks': self.detect_market_shocks(),
                'trend': self.forecast_fundamental_trend(),
                'timestamp': datetime.utcnow().isoformat()
            })

            # ✅ Push updated fundamentals into the shared market data bus
            market_data_bus.update({'fundamentals': analysis})

            self._last_update = datetime.utcnow()
            self.logger.info("Fundamental data updated", analysis=analysis)

        except Exception as e:
            self.logger.error("Fundamental update failed", error=str(e))
            market_data_bus.update({'fundamentals': {'error': str(e)}})