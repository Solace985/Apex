import requests
import re
import json
import nltk
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Any

nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """Extracts sentiment from financial news, social media, and institutional reports."""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fetch_news_sentiment(self):
        """Fetches financial news sentiment from API sources and assigns sentiment scores."""
        sources = [
            "https://newsapi.org/v2/top-headlines?category=business&apiKey=YOUR_API_KEY",
            "https://finnhub.io/api/v1/news?category=general&token=YOUR_API_KEY"
        ]
        sentiments = []
        for url in sources:
            try:
                response = requests.get(url)
                articles = response.json()
                for article in articles[:10]:  # Analyze top 10 articles for better accuracy
                    sentiment_score = self.analyze_text(article.get("title", "") + " " + article.get("description", ""))
                    sentiments.append(sentiment_score)
            except Exception as e:
                print(f"Error fetching news: {e}")

        return np.mean(sentiments) if sentiments else 0

    def fetch_twitter_sentiment(self, keyword="stock market"):
        """Placeholder for fetching Twitter sentiment. This requires an API key."""
        tweets = ["Market is booming!", "Inflation is rising, stocks will fall.", "Tech stocks are looking great."]
        sentiments = [self.analyze_text(tweet) for tweet in tweets]
        return np.mean(sentiments)

    def fetch_reddit_sentiment(self, subreddit="stocks"):
        """Placeholder for fetching Reddit sentiment. This requires API access."""
        reddit_posts = ["Everyone is bullish on AAPL!", "Bearish market coming soon!", "Stock split rumors on Tesla."]
        sentiments = [self.analyze_text(post) for post in reddit_posts]
        return np.mean(sentiments)

    def analyze_text(self, text: str):
        """Uses NLP to extract sentiment scores from text."""
        text = re.sub(r"http\S+", "", text)  # Remove links
        text_blob_score = TextBlob(text).sentiment.polarity
        vader_score = self.sia.polarity_scores(text)["compound"]
        return (text_blob_score + vader_score) / 2  # Average sentiment score

    def get_sentiment_score(self):
        """Calculates a weighted sentiment score for the market based on news, Twitter, and Reddit sentiment."""
        news_score = self.fetch_news_sentiment()
        twitter_score = self.fetch_twitter_sentiment()
        reddit_score = self.fetch_reddit_sentiment()

        sentiment_score = (0.5 * news_score) + (0.25 * twitter_score) + (0.25 * reddit_score)
        return sentiment_score

    def integrate_sentiment_with_trading(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrates sentiment analysis with trading strategy to make informed decisions."""
        sentiment_score = self.get_sentiment_score()
        if sentiment_score > 0.3:
            market_data['sentiment_signal'] = 'buy'
        elif sentiment_score < -0.3:
            market_data['sentiment_signal'] = 'sell'
        else:
            market_data['sentiment_signal'] = 'hold'
        return market_data

    def log_sentiment_analysis(self, sentiment_score: float):
        """Logs the sentiment analysis results for further analysis and auditing."""
        print(f"Sentiment Score: {sentiment_score}")
        if sentiment_score > 0.3:
            print("Positive sentiment detected. Consider buying.")
        elif sentiment_score < -0.3:
            print("Negative sentiment detected. Consider selling.")
        else:
            print("Neutral sentiment. Hold position.")
