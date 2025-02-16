import requests
from textblob import TextBlob
from AI_Models.sentiment_analysis import SentimentAnalyzer

class FundamentalAnalysis:
    """Combines economic indicators, institutional moves, and sentiment scores."""

    def __init__(self):
        self.api_key = "YOUR_NEWS_API_KEY"  # Get from newsapi.org
        self.sentiment_analyzer = SentimentAnalyzer()

    def fetch_news_sentiment(self):
        """Analyzes sentiment from latest financial news headlines."""
        try:
            response = requests.get(f"https://newsapi.org/v2/everything?q=finance&apiKey={self.api_key}")
            articles = response.json().get("articles", [])[:5]  # Top 5 headlines
            sentiments = [TextBlob(article["title"]).sentiment.polarity for article in articles]
            return sum(sentiments) / len(sentiments) if sentiments else 0
        except:
            return 0

    def fetch_macro_factors(self):
        """Fetches key macroeconomic indicators like CPI, interest rates, and unemployment."""
        macro_data = {
            "inflation": 2.5,  # Placeholder
            "interest_rate": 0.75,  # Placeholder
        }
        return macro_data

    def fetch_economic_data(self):
        """Fetches fundamental macroeconomic data (GDP, CPI, FOMC). Placeholder for API integration."""
        economic_data = {"GDP Growth": 3.2, "CPI": 6.4, "FOMC Rate Decision": 5.25}
        return economic_data

    def analyze_fundamentals(self):
        """Generates a fundamental sentiment score weighted with economic data."""
        economic_data = self.fetch_economic_data()
        sentiment_score = self.sentiment_analyzer.get_sentiment_score()

        macroeconomic_weight = 0.4
        institutional_weight = 0.3
        sentiment_weight = 0.3

        weighted_score = (
            macroeconomic_weight * economic_data["GDP Growth"]
            - institutional_weight * economic_data["FOMC Rate Decision"]
            + sentiment_weight * sentiment_score
        )

        return weighted_score
