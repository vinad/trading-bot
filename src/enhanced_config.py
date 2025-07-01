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
    DEFAULT_RISK_PERCENT = float(os.getenv('DEFAULT_RISK_PERCENT', 0.01))  # Reduced to 1% for safety
    MAX_POSITION_SIZE = int(os.getenv('MAX_POSITION_SIZE', 100))  # Reduced for paper trading
    
    # Trading Parameters
    WATCHLIST = [
        # High-volume stocks that support extended hours
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META',
        'SPY', 'QQQ',  # ETFs for market-wide exposure
        # Add crypto-related stocks for 24-hour exposure
        'COIN', 'MSTR'  # These trade during regular hours but crypto-related
    ]
    
    # Extended Hours Trading
    ENABLE_EXTENDED_HOURS = bool(os.getenv('ENABLE_EXTENDED_HOURS', 'True').lower() == 'true')
    EXTENDED_HOURS_SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META',
        'SPY', 'QQQ', 'COIN', 'MSTR'
    ]
    
    # Timing Configuration
    SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL', 300))  # 5 minutes default
    EXTENDED_HOURS_SCAN_INTERVAL = int(os.getenv('EXTENDED_HOURS_SCAN_INTERVAL', 600))  # 10 minutes for extended hours
    
    # Strategy Configuration
    STRATEGIES_REGULAR_HOURS = ['mean_reversion', 'momentum', 'rsi']
    STRATEGIES_EXTENDED_HOURS = ['momentum', 'rsi']  # Fewer strategies for extended hours
    
    # Performance and Monitoring
    PERFORMANCE_REPORT_INTERVAL = int(os.getenv('PERFORMANCE_REPORT_INTERVAL', 12))  # Every 12 scans
    PORTFOLIO_SUMMARY_INTERVAL = int(os.getenv('PORTFOLIO_SUMMARY_INTERVAL', 3))   # Every 3 scans
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/trading_bot.log')
    
    # Alert Configuration (Optional - for email alerts)
    ENABLE_EMAIL_ALERTS = bool(os.getenv('ENABLE_EMAIL_ALERTS', 'False').lower() == 'true')
    EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', 587))
    EMAIL_FROM = os.getenv('EMAIL_FROM')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')  # Use app password for Gmail
    EMAIL_TO = os.getenv('EMAIL_TO')
    
    # 24-Hour Market Simulation (for testing when markets are closed)
    SIMULATE_24H_TRADING = bool(os.getenv('SIMULATE_24H_TRADING', 'False').lower() == 'true')

# Validate configuration
def validate_config():
    """Validate that all required configuration is present"""
    if not Config.ALPACA_API_KEY or not Config.ALPACA_SECRET_KEY:
        raise ValueError("Alpaca API credentials not found. Please check your .env file.")
    
    if Config.ALPACA_API_KEY == "your_api_key_here":
        raise ValueError("Please update your .env file with actual Alpaca API credentials")
    
    # Validate email configuration if enabled
    if Config.ENABLE_EMAIL_ALERTS:
        if not all([Config.EMAIL_FROM, Config.EMAIL_PASSWORD, Config.EMAIL_TO]):
            print("Warning: Email alerts enabled but configuration incomplete. Disabling email alerts.")
            Config.ENABLE_EMAIL_ALERTS = False
    
    # Validate scan intervals
    if Config.SCAN_INTERVAL < 60:
        print(f"Warning: Scan interval {Config.SCAN_INTERVAL}s is very short. Consider using 300s (5min) or higher.")
    
    return True

def get_email_config():
    """Get email configuration if enabled"""
    if not Config.ENABLE_EMAIL_ALERTS:
        return None
    
    return {
        'smtp_server': Config.EMAIL_SMTP_SERVER,
        'smtp_port': Config.EMAIL_SMTP_PORT,
        'from_email': Config.EMAIL_FROM,
        'password': Config.EMAIL_PASSWORD,
        'to_email': Config.EMAIL_TO
    }