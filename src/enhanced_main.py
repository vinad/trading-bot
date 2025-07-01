import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_trading_bot import TradingBot, AlertSystem, AdvancedTradingEngine
from enhanced_config import Config, validate_config
import logging
import time
import signal
from datetime import datetime
import json

# Global logger variable
logger = None

def setup_logging():
    """Set up enhanced logging configuration"""
    global logger
    os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important info
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(detailed_formatter)
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set module logger
    logger = logging.getLogger(__name__)

def create_performance_report(bot: TradingBot, engine: AdvancedTradingEngine):
    """Create and save performance report"""
    try:
        account = bot.api.get_account()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'account_equity': float(account.equity),
            'trade_count': len(bot.trade_history),
            'recent_trades': []
        }
        
        # Add recent trades
        for trade in bot.trade_history[-10:]:  # Last 10 trades
            report['recent_trades'].append({
                'timestamp': trade['timestamp'].isoformat() if hasattr(trade['timestamp'], 'isoformat') else str(trade['timestamp']),
                'symbol': trade['symbol'],
                'signal': trade['signal'],
                'strategy': trade.get('strategy', 'unknown'),
                'market_session': trade.get('market_session', 'regular')
            })
        
        # Save report
        report_file = f"logs/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved: {report_file}")
        
    except Exception as e:
        logger.error(f"Error creating performance report: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    global running
    running = False

def main():
    """Enhanced main function with better monitoring and control"""
    global running, logger
    running = True
    
    try:
        # Validate configuration
        validate_config()
        
        # Set up logging
        setup_logging()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("="*60)
        logger.info("ENHANCED TRADING BOT STARTING")
        logger.info("="*60)
        
        # Initialize alert system
        alert_system = AlertSystem()
        
        # Initialize bot
        bot = TradingBot(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL
        )
        
        # Initialize trading engine
        engine = AdvancedTradingEngine(bot, alert_system)
        
        # Show initial account info
        logger.info("Bot initialized successfully!")
        alert_system.send_alert("Trading bot started successfully", "INFO")
        
        # Initial portfolio summary
        engine.get_enhanced_portfolio_summary()
        
        # Performance tracking
        start_equity = float(bot.account.equity)
        scan_count = 0
        
        logger.info(f"Starting main trading loop...")
        logger.info(f"Watchlist: {Config.WATCHLIST}")
        logger.info(f"Scan interval: {Config.SCAN_INTERVAL} seconds")
        
        # Main trading loop
        while running:
            try:
                scan_count += 1
                logger.info(f"\n🔄 SCAN #{scan_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run intelligent scan using the AdvancedTradingEngine
                engine.run_intelligent_scan(Config.WATCHLIST)
                
                # Show portfolio summary every 3rd scan or if there are positions
                positions = bot.api.list_positions()
                if scan_count % 3 == 0 or len(positions) > 0:
                    engine.get_enhanced_portfolio_summary()
                
                # Create performance report every hour (12 scans at 5min intervals)
                if scan_count % 12 == 0:
                    create_performance_report(bot, engine)
                
                # Show progress
                current_equity = float(bot.api.get_account().equity)
                performance = ((current_equity - start_equity) / start_equity) * 100
                logger.info(f"💰 Performance since start: {performance:+.2f}% (${current_equity - start_equity:+.2f})")
                
                if not running:
                    break
                
                # Wait before next scan
                logger.info(f"⏰ Waiting {Config.SCAN_INTERVAL} seconds before next scan...")
                
                # Sleep in smaller intervals to allow graceful shutdown
                for _ in range(Config.SCAN_INTERVAL):
                    if not running:
                        break
                    time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user (Ctrl+C)")
                break
            except Exception as e:
                error_msg = f"Error in main loop: {e}"
                logger.error(error_msg)
                alert_system.send_alert(error_msg, "ERROR")
                
                logger.info("Waiting 60 seconds before retrying...")
                for _ in range(60):
                    if not running:
                        break
                    time.sleep(1)
        
        # Shutdown procedures
        logger.info("\n" + "="*60)
        logger.info("SHUTTING DOWN TRADING BOT")
        logger.info("="*60)
        
        # Final portfolio summary
        engine.get_enhanced_portfolio_summary()
        
        # Final performance report
        create_performance_report(bot, engine)
        
        # Final statistics
        final_equity = float(bot.api.get_account().equity)
        total_performance = ((final_equity - start_equity) / start_equity) * 100
        
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"   Total scans completed: {scan_count}")
        logger.info(f"   Total trades executed: {len(bot.trade_history)}")
        logger.info(f"   Starting equity: ${start_equity:,.2f}")
        logger.info(f"   Final equity: ${final_equity:,.2f}")
        logger.info(f"   Total performance: {total_performance:+.2f}% (${final_equity - start_equity:+.2f})")
        
        alert_system.send_alert(f"Trading bot shutdown. Final performance: {total_performance:+.2f}%", "INFO")
        
        logger.info("Trading bot shutdown complete.")
        
    except Exception as e:
        error_msg = f"Failed to start bot: {e}"
        print(error_msg)
        if logger:
            logger.error(error_msg)
        print("Please check your configuration and API credentials")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())