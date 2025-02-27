import httpx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
from sklearn.ensemble import IsolationForest
from Apex.utils.helpers import validate_input, secure_api_call
from Apex.src.Core.fundamental.fundamental_engine import FundamentalAnalyzer
from Apex.src.ai.forecasting.sentiment_analysis import FinBertProcessor
from Apex.src.ai.reinforcement.q_learning import QLearningAgent

class InstitutionalInsiderMonitor:
    """Real-time insider trading detection with dark pool integration"""
    
    def __init__(self):
        self.fundamental = FundamentalAnalyzer()
        self.sentiment = FinBertProcessor()
        self.rl_agent = QLearningAgent(state_size=3, action_size=3)  # State: [Position, Trade Size, Market Reaction]
        self._setup_datastores()
        self._init_thresholds()

    def _setup_datastores(self):
        """Initialize data structures without external dependencies"""
        self.insider_activity = deque(maxlen=1000)
        self.dark_pool_cache = {}
        self.cluster_models = self._load_cluster_models()

    def _init_thresholds(self):
        """Dynamic thresholds from config"""
        self.thresholds = {
            'ceo_weight': 0.85,
            'cfo_weight': 0.75,
            'director_weight': 0.65,
            'cluster_cutoff': 0.9,
            'dark_pool_alert': 0.0001  # 0.01% of market cap
        }

    async def monitor_insiders(self):
        """Main monitoring loop with multiple data sources"""
        async with httpx.AsyncClient() as client:
            while True:
                await self._fetch_sec_filings(client)
                await self._poll_dark_pools(client)
                await self._process_13f_filings()
                await asyncio.sleep(300)  # 5 minute interval

    async def _fetch_sec_filings(self, client: httpx.AsyncClient):
        """SEC Edgar API integration with sanitization"""
        url = "https://api.sec.gov/submissions/"
        response = await secure_api_call(client, url)
        if response:
            filings = self._parse_edgar_response(response.json())
            for filing in filings:
                if self._validate_filing(filing):
                    processed = self._process_filing(filing)
                    self.insider_activity.append(processed)

    def _process_filing(self, filing: Dict) -> Dict:
        """Multi-stage filing processing pipeline"""
        processed = {
            'executive': filing.get('name', ''),
            'position': filing.get('title', ''),
            'transaction': self._normalize_transaction(filing),
            'sentiment': self._analyze_filing_text(filing.get('text', '')),
            'confidence': self._calculate_confidence(filing)
        }
        return self._detect_anomalies(processed)

    def _calculate_confidence(self, filing: Dict) -> float:
        """Reinforcement Learning-based Confidence Scoring System"""

        base_score = {
            'CEO': self.thresholds['ceo_weight'],
            'CFO': self.thresholds['cfo_weight'],
            'Director': self.thresholds['director_weight']
        }.get(filing.get('title', ''), 0.5)

        size_multiplier = min(1.0, filing['shares'] / 1e6)  # Normalize trade size
        timing_penalty = 0.8 if self._near_earnings(filing['date']) else 1.0

        # Convert to RL State (Position, Trade Size, Market Reaction)
        state = [base_score, size_multiplier, self.fundamental.get_market_reaction(filing['symbol'])]

        # RL-Based Confidence Adjustment
        action = self.rl_agent.select_action(state)  # RL decides weight adjustment
        confidence_adjustment = {0: -0.1, 1: 0.0, 2: 0.1}  # Adjust weight dynamically

        adjusted_score = base_score * size_multiplier * timing_penalty
        return max(0, min(1, adjusted_score + confidence_adjustment[action]))

    def update_rl_agent(self, filing: Dict, trade_outcome: float):
        """Update RL agent based on trade success or failure"""
        state = [filing['confidence'], filing['transaction']['size'], self.fundamental.get_market_reaction(filing['symbol'])]
        reward = trade_outcome  # Positive if trade was profitable, negative if it was a bad trade
        self.rl_agent.update_q_table(state, reward)

    def _analyze_filing_text(self, text: str) -> Dict:
        """Integrated sentiment analysis"""
        return self.sentiment.analyze(text)

    def _detect_anomalies(self, filing: Dict) -> Optional[Dict]:
        """Cluster analysis for pattern detection across multiple companies in the same sector"""
        
        features = [
            filing['transaction']['size'],
            filing['confidence'],
            filing['sentiment']['score']
        ]
        cluster_score = self.cluster_models['dbscan'].fit_predict([features])[0]

        sector = self.fundamental.get_sector(filing['symbol'])
        sector_insiders = [f for f in self.insider_activity if self.fundamental.get_sector(f['symbol']) == sector]

        # Sector-wide cluster detection
        if len(sector_insiders) > 3:  # If multiple insiders in the same sector are active
            sector_cluster_score = np.mean([f['confidence'] for f in sector_insiders])
            if sector_cluster_score > self.thresholds['cluster_cutoff']:
                self._alert_fundamental_engine(filing, sector_alert=True)
                return filing

        if cluster_score >= self.thresholds['cluster_cutoff']:
            self._alert_fundamental_engine(filing)
            return filing
        return None

    async def _poll_dark_pools(self, client: httpx.AsyncClient):
        """Dark pool trade monitoring + ML-based manipulation detection"""
        url = "https://api.finra.org/darkpool/data"
        response = await secure_api_call(client, url)
        if response:
            trades = response.json().get('data', [])
            for trade in trades:
                if self._significant_dark_pool_trade(trade):
                    self._update_dark_pool_cache(trade)
                    self._detect_dark_pool_manipulation(trade)

    def _detect_dark_pool_manipulation(self, trade: Dict):
        """AI Model to detect front-running or suspicious accumulation"""
        symbol = trade['symbol']
        order_size = trade['size']
        market_cap = self.fundamental.get_market_cap(symbol)

        if symbol not in self.dark_pool_cache:
            self.dark_pool_cache[symbol] = deque(maxlen=50)

        # Extract past dark pool trades for this symbol
        past_trades = np.array([t['size'] for t in list(self.dark_pool_cache[symbol])])

        if len(past_trades) > 10:
            anomaly_detector = IsolationForest(contamination=0.1)
            anomaly_score = anomaly_detector.fit_predict(past_trades.reshape(-1, 1))

            if anomaly_score[-1] == -1 and order_size > (market_cap * self.thresholds['dark_pool_alert']):
                self._alert_fundamental_engine({"symbol": symbol, "dark_pool_alert": True})

    def _significant_dark_pool_trade(self, trade: Dict) -> bool:
        """Check against market cap threshold"""
        market_cap = self.fundamental.get_market_cap(trade['symbol'])
        return trade['size'] > (market_cap * self.thresholds['dark_pool_alert'])

    def _update_dark_pool_cache(self, trade: Dict):
        """Maintain rolling window of dark pool activity"""
        symbol = trade['symbol']
        if symbol not in self.dark_pool_cache:
            self.dark_pool_cache[symbol] = deque(maxlen=50)
        self.dark_pool_cache[symbol].append(trade)

    # Integration points
    def _alert_fundamental_engine(self, filing: Dict):
        """Push significant filings to fundamental analysis"""
        from Apex.src.Core.fundamental.fundamental_engine import handle_insider_alert
        handle_insider_alert({
            'symbol': filing['symbol'],
            'confidence': filing['confidence'],
            'direction': filing['transaction']['type']
        })

    def get_insider_signal(self, symbol: str) -> Dict:
        """API for external modules to access insider data"""
        return {
            'activity': [f for f in self.insider_activity if f['symbol'] == symbol],
            'dark_pool': list(self.dark_pool_cache.get(symbol, [])[-10:])
        }

    # Security and validation
    @validate_input
    def _normalize_transaction(self, filing: Dict) -> Dict:
        """Sanitize transaction data"""
        return {
            'type': 'BUY' if filing['transactionCode'].startswith('P') else 'SELL',
            'size': abs(int(filing['shares'])),
            'price': secure_float(filing['price'])
        }

    def _near_earnings(self, date: str) -> bool:
        """Check if trade is near earnings report"""
        report_dates = self.fundamental.get_earnings_dates(date['symbol'])
        trade_date = datetime.fromisoformat(date)
        return any(abs(trade_date - rd) < timedelta(days=14) for rd in report_dates)