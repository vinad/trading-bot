import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot import TradingBot, TradingEngine
from config import Config, validate_config
import logging
import time

def setup_logging():
    """Set up logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the trading bot"""
    try:
        # Validate configuration
        validate_config()
        
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Trading Bot...")
        
        # Initialize bot
        bot = TradingBot(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL
        )
        
        engine = TradingEngine(bot)
        
        # Show initial account info
        logger.info("Bot initialized successfully!")
        
        # Main trading loop
        while True:
            try:
                # Check if market is open
                clock = bot.api.get_clock()
                
                if clock.is_open:
                    logger.info("Market is open - running strategies")
                    engine.run_multi_strategy_scan(Config.WATCHLIST)
                    engine.get_portfolio_summary()
                else:
                    logger.info("Market is closed - waiting for next market open")
                
                # Wait before next scan
                logger.info(f"Waiting {Config.SCAN_INTERVAL} seconds before next scan...")
                time.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info("Waiting 60 seconds before retrying...")
                time.sleep(60)  # Wait 1 minute before retrying
                
    except Exception as e:
        print(f"Failed to start bot: {e}")
        print("Please check your configuration and API credentials")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())