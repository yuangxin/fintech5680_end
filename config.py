# Configuration file for API keys and settings
# This file contains all the necessary API keys and configuration

# Polygon.io API Key for stock data
POLYGON_API_KEY = "l7eyBqnxp9XsobMxIBIVHx69zqRlY5rc"

# News API settings
NEWS_API_KEY = "l7eyBqnxp9XsobMxIBIVHx69zqRlY5rc"
NEWS_BASE_URL = "https://api.massive.com/v2/reference/news"

# Model settings
LSTM_MODEL_PATH = "model/LSTM_FINTECH.pth"
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Other settings
DEFAULT_STOCKS = [
    {'Ticker': 'AAPL', 'Name': 'Apple Inc.'},
    {'Ticker': 'MSFT', 'Name': 'Microsoft Corp.'},
    {'Ticker': 'GOOGL', 'Name': 'Alphabet Inc.'}
]