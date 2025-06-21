import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Alpaca API
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Risk Management
    DEFAULT_RISK_PERCENT = float(os.getenv('DEFAULT_RISK_PERCENT', 0.02))
    MAX_POSITION_SIZE = int(os.getenv('MAX_POSITION_SIZE', 1000))
    
    # Trading Parameters
    WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META']
    SCAN_INTERVAL = 300  # 5 minutes
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/trading_bot.log'

# Validate configuration
def validate_config():
    """Validate that all required configuration is present"""
    if not Config.ALPACA_API_KEY or not Config.ALPACA_SECRET_KEY:
        raise ValueError("Alpaca API credentials not found. Please check your .env file.")
    
    if Config.ALPACA_API_KEY == "your_api_key_here":
        raise ValueError("Please update your .env file with actual Alpaca API credentials")
    
    return True