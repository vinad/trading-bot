import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import ta
import json
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pytz
from enhanced_config import Config


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketSession(Enum):
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"

class TradingBot:
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        """Initialize the trading bot with Alpaca API credentials"""
        try:
            self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            self.account = self.api.get_account()
            self.positions = {}
            self.strategies = {}
            self.trade_history = []
            
            logger.info(f"Bot initialized. Account equity: ${float(self.account.equity):.2f}")
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise
    
    def get_market_session(self) -> MarketSession:
        """Determine current market session with fixed timezone handling"""
        try:
            # If 24-hour trading simulation is enabled, always return REGULAR session
            if Config.SIMULATE_24H_TRADING:
                # Always return REGULAR session for 24-hour simulation
                logger.info("24-hour simulation enabled - treating as regular market hours")
                return MarketSession.REGULAR

            clock = self.api.get_clock()
            
            # Get current time in Eastern timezone
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            
            if clock.is_open:
                return MarketSession.REGULAR
            
            # Define market hours in Eastern time
            today = now.date()
            market_open_naive = datetime.combine(today, datetime.strptime("09:30", "%H:%M").time())
            market_close_naive = datetime.combine(today, datetime.strptime("16:00", "%H:%M").time())
            pre_market_start_naive = datetime.combine(today, datetime.strptime("04:00", "%H:%M").time())
            after_hours_end_naive = datetime.combine(today, datetime.strptime("20:00", "%H:%M").time())

            market_open_time = eastern.localize(market_open_naive)
            market_close_time = eastern.localize(market_close_naive)
            pre_market_start = eastern.localize(pre_market_start_naive)
            after_hours_end = eastern.localize(after_hours_end_naive)
            
            # Check market session
            if pre_market_start <= now < market_open_time:
                return MarketSession.PRE_MARKET
            elif market_close_time <= now <= after_hours_end:
                return MarketSession.AFTER_HOURS
            else:
                return MarketSession.CLOSED
                
        except Exception as e:
            logger.error(f"Error determining market session: {e}")
            # Fallback to simple time-based detection
            current_hour = datetime.now().hour
            if 4 <= current_hour < 9:
                return MarketSession.PRE_MARKET
            elif 9 <= current_hour < 16:
                return MarketSession.REGULAR
            elif 16 <= current_hour < 20:
                return MarketSession.AFTER_HOURS
            else:
                return MarketSession.CLOSED
    
    def can_trade_extended_hours(self, symbol: str) -> bool:
        """Check if symbol supports extended hours trading"""
        # Most major stocks support extended hours
        extended_hours_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'SPY', 'QQQ', 'COIN', 'MSTR']
        return symbol in extended_hours_symbols
    
    def get_market_data(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> pd.DataFrame:
        """Get historical market data for a symbol"""
        try:
            # Use yfinance for more reliable historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            return data.tail(limit)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, symbol: str, risk_percent: float = 0.02) -> int:
        """Calculate position size based on account equity and risk management"""
        equity = float(self.account.equity)
        risk_amount = equity * risk_percent
        
        try:
            # # Get current price
            # quote = self.api.get_latest_quote(symbol)
            # current_price = float(quote.ask_price) if quote.ask_price else float(quote.bid_price)
            
            #Get current price with fallback
            quote = self.api.get_latest_quote(symbol)
            if quote.ask_price:
                current_price = float(quote.ask_price)
            elif quote.bid_price:
                current_price = float(quote.bid_price)
            else:
                # Fallback to last trade price
                try:
                    last_trade = self.api.get_latest_trade(symbol)
                    current_price = float(last_trade.price)
                except:
                    logger.error(f"Could not get price for {symbol}")
                    return 1


            # Simple position sizing (can be enhanced with volatility-based sizing)
            shares = int(risk_amount / (current_price * 0.02))  # 2% stop loss assumption
            return max(1, min(shares, 100))  # Cap at 100 shares for safety
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 1
    
    def scan_and_trade(self, symbol: str):
        """Simple scan and trade method for compatibility"""
        try:
            data = self.get_market_data(symbol)
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Simple RSI strategy for compatibility
            if len(data) < 14:
                return
            
            data['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
            current_rsi = data['rsi'].iloc[-1]
            
            logger.info(f"{symbol}: RSI = {current_rsi:.1f}")
            
            # Simple trading logic
            if current_rsi < 30:
                logger.info(f"{symbol}: RSI oversold signal")
            elif current_rsi > 70:
                logger.info(f"{symbol}: RSI overbought signal")
            else:
                logger.info(f"{symbol}: RSI neutral")
                
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")

class AlertSystem:
    def __init__(self, email_config: Dict = None):
        self.email_config = email_config
        self.alerts = []
    
    def send_alert(self, message: str, alert_type: str = "INFO"):
        """Send alert via logging and optionally email"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = f"[{alert_type}] {timestamp}: {message}"
        
        self.alerts.append(alert)
        logger.info(f"ALERT: {alert}")
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Optional: Send email alert for important events
        if alert_type in ["TRADE", "ERROR"] and self.email_config:
            self.send_email_alert(alert)
    
    def send_email_alert(self, message: str):
        """Send email alert (optional - requires email configuration)"""
        try:
            if not self.email_config:
                return
                
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = "Trading Bot Alert"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['from_email'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_email'], text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

class AdvancedTradingEngine:
    def __init__(self, bot: TradingBot, alert_system: AlertSystem = None):
        self.bot = bot
        self.alert_system = alert_system or AlertSystem()
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(),
            'momentum': MomentumStrategy(),
            'rsi': RSIStrategy()
        }
        self.portfolio = {}
        self.strategy_performance = {}
        self.last_signals = {}  # Track last signals to avoid conflicts
    
    def execute_strategy(self, symbol: str, strategy_name: str, market_session: MarketSession):
        """Execute a specific strategy for a symbol"""
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return
        
        strategy = self.strategies[strategy_name]
        data = self.bot.get_market_data(symbol)
        
        if data.empty:
            logger.error(f"No data available for {symbol}")
            return
        
        signal = strategy.generate_signals(data)
        signal_key = f"{symbol}_{strategy_name}"
        
        # Check if this is a new signal (avoid repeated trades)
        if signal_key in self.last_signals and self.last_signals[signal_key] == signal['signal']:
            logger.debug(f"Same signal for {symbol} with {strategy_name}, skipping")
            return
        
        self.last_signals[signal_key] = signal['signal']
        
        logger.info(f"{strategy.name} strategy for {symbol}: {signal['signal']} - {signal['reason']}")
        
        # Execute trade based on signal
        if signal['signal'] in ['buy', 'sell']:
            self.execute_trade(symbol, signal['signal'], strategy_name, market_session)
    
    def execute_trade(self, symbol: str, signal: str, strategy_name: str, market_session: MarketSession):
        """Execute buy/sell orders based on signals"""
        try:
            # Check current position
            try:
                position = self.bot.api.get_position(symbol)
                current_qty = int(position.qty)
            except:
                current_qty = 0
            
            # Determine if we can trade in current market session
            extended_hours = market_session != MarketSession.REGULAR
            if extended_hours and not self.bot.can_trade_extended_hours(symbol):
                logger.info(f"Extended hours trading not available for {symbol}")
                return
            
            order_params = {
                'symbol': symbol,
                'type': 'market',
                'time_in_force': 'day'
            }
            
            # Add extended hours parameter if needed
            if extended_hours:
                order_params['time_in_force'] = 'day'  # or 'gtc' for good-till-canceled
            else:
                order_params['time_in_force'] = 'day'
            
            trade_executed = False
            
            if signal == "buy" and current_qty <= 0:
                qty = self.bot.calculate_position_size(symbol)
                order_params.update({'qty': qty, 'side': 'buy'})
                
                order = self.bot.api.submit_order(**order_params)
                
                trade_msg = f"BUY order submitted: {qty} shares of {symbol} using {strategy_name} in {market_session.value}"
                logger.info(trade_msg)
                self.alert_system.send_alert(trade_msg, "TRADE")
                trade_executed = True
                
            elif signal == "sell" and current_qty > 0:
                order_params.update({'qty': abs(current_qty), 'side': 'sell'})
                
                order = self.bot.api.submit_order(**order_params)
                
                trade_msg = f"SELL order submitted: {abs(current_qty)} shares of {symbol} using {strategy_name} in {market_session.value}"
                logger.info(trade_msg)
                self.alert_system.send_alert(trade_msg, "TRADE")
                trade_executed = True
            
            if trade_executed:
                # Record trade for performance tracking
                self.record_trade(symbol, signal, strategy_name, market_session)
            else:
                logger.info(f"No action taken for {symbol} - {signal} signal with current position: {current_qty}")
                
        except Exception as e:
            error_msg = f"Error executing trade for {symbol}: {e}"
            logger.error(error_msg)
            self.alert_system.send_alert(error_msg, "ERROR")
    
    def record_trade(self, symbol: str, signal: str, strategy_name: str, market_session: MarketSession):
        """Record trade for performance analysis"""
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'strategy': strategy_name,
            'market_session': market_session.value
        }
        self.bot.trade_history.append(trade_record)
    
    def run_intelligent_scan(self, symbols: List[str]):
        """Run strategies intelligently to avoid conflicts"""
        market_session = self.bot.get_market_session()
        
        logger.info(f"Starting scan in {market_session.value} session...")
        self.alert_system.send_alert(f"Market scan started - {market_session.value} session", "INFO")
        
        if market_session == MarketSession.CLOSED:
            logger.info("Market completely closed - no trading available")
            return
        
        # Prioritize strategies based on market session
        if market_session == MarketSession.REGULAR:
            # Use all strategies during regular hours
            active_strategies = list(self.strategies.keys())
        else:
            # Use only momentum and RSI for extended hours (less noise)
            active_strategies = ['momentum', 'rsi']
        
        for symbol in symbols:
            logger.info(f"\n--- Analyzing {symbol} in {market_session.value} ---")
            
            # Check if symbol supports current market session
            if market_session != MarketSession.REGULAR and not self.bot.can_trade_extended_hours(symbol):
                logger.info(f"{symbol} not available for extended hours trading")
                continue
            
            # Run strategies with priority (avoid conflicts)
            for strategy_name in active_strategies:
                self.execute_strategy(symbol, strategy_name, market_session)
                time.sleep(0.5)  # Rate limiting
    
    def get_enhanced_portfolio_summary(self):
        """Get detailed portfolio summary with proper error handling"""
        try:
            account = self.bot.api.get_account()
            positions = self.bot.api.list_positions()
            
            logger.info("\n" + "="*50)
            logger.info("📊 PORTFOLIO SUMMARY")
            logger.info("="*50)
            
            # Account information
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            # Calculate day P&L using position data since account doesn't have unrealized_pl
            total_unrealized_pl = 0
            if positions:
                total_unrealized_pl = sum(float(pos.unrealized_pl) for pos in positions)
            
            logger.info(f"💰 Account Equity: ${equity:,.2f}")
            logger.info(f"💸 Buying Power: ${buying_power:,.2f}")
            logger.info(f"📈 Unrealized P&L: ${total_unrealized_pl:+,.2f}")
            logger.info(f"📦 Open Positions: {len(positions)}")
            
            if positions:
                logger.info("\n📋 CURRENT POSITIONS:")
                logger.info("-" * 50)
                total_position_value = 0
                
                for position in positions:
                    symbol = position.symbol
                    qty = int(position.qty)
                    market_value = float(position.market_value)
                    unrealized_pl = float(position.unrealized_pl)
                    unrealized_plpc = float(position.unrealized_plpc) * 100
                    
                    total_position_value += market_value
                    
                    # Format P&L with color indicators
                    pl_indicator = "🟢" if unrealized_pl >= 0 else "🔴"
                    
                    logger.info(f"{symbol:8} | {qty:4} shares | ${market_value:8,.2f} | {pl_indicator}${unrealized_pl:+7.2f} ({unrealized_plpc:+5.1f}%)")
                
                logger.info("-" * 50)
                logger.info(f"Total Position Value: ${total_position_value:,.2f}")
            else:
                logger.info("\nNo current positions")
            
            # Recent trades
            if self.bot.trade_history:
                recent_trades = self.bot.trade_history[-5:]  # Last 5 trades
                logger.info(f"\nRecent Trades:")
                logger.info("-" * 30)
                for trade in recent_trades:
                    timestamp_str = trade['timestamp'].strftime('%H:%M') if hasattr(trade['timestamp'], 'strftime') else str(trade['timestamp'])
                    logger.info(f"{timestamp_str} - {trade['signal'].upper()} {trade['symbol']} ({trade['strategy']})")
            
            logger.info("="*50)
            
            self.alert_system.send_alert(f"Portfolio update: Equity ${equity:,.2f}, Unrealized P&L ${total_unrealized_pl:,.2f}", "INFO")
            
        except Exception as e:
            error_msg = f"Error getting portfolio summary: {e}"
            logger.error(error_msg)
            self.alert_system.send_alert(error_msg, "ERROR")

# Strategy classes
class MeanReversionStrategy:
    def __init__(self, lookback_period: int = 20, z_threshold: float = 2.0):
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.name = "Mean Reversion"
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        if not self._validate_data(data):
            return {"signal": "hold", "reason": "Invalid data format"}
        
        """Generate buy/sell signals based on mean reversion"""
        if len(data) < self.lookback_period:
            return {"signal": "hold", "reason": "Insufficient data"}
        
        # Calculate moving average and standard deviation
        data['sma'] = data['Close'].rolling(window=self.lookback_period).mean()
        data['std'] = data['Close'].rolling(window=self.lookback_period).std()
        
        # Calculate Z-score
        data['z_score'] = (data['Close'] - data['sma']) / data['std']
        
        current_z = data['z_score'].iloc[-1]
        
        if current_z < -self.z_threshold:
            return {"signal": "buy", "reason": f"Z-score {current_z:.2f} below -{self.z_threshold}"}
        elif current_z > self.z_threshold:
            return {"signal": "sell", "reason": f"Z-score {current_z:.2f} above {self.z_threshold}"}
        else:
            return {"signal": "hold", "reason": f"Z-score {current_z:.2f} within threshold"}

class MomentumStrategy:
    def __init__(self, short_window: int = 10, long_window: int = 30):
        self.short_window = short_window
        self.long_window = long_window
        self.name = "Momentum"
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        if not self._validate_data(data):
            return {"signal": "hold", "reason": "Invalid data format"}

        """Generate signals based on moving average crossover"""
        if len(data) < self.long_window:
            return {"signal": "hold", "reason": "Insufficient data"}
        
        data['sma_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['sma_long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Current and previous values
        current_short = data['sma_short'].iloc[-1]
        current_long = data['sma_long'].iloc[-1]
        prev_short = data['sma_short'].iloc[-2]
        prev_long = data['sma_long'].iloc[-2]
        
        # Check for crossover
        if prev_short <= prev_long and current_short > current_long:
            return {"signal": "buy", "reason": "Golden cross - short MA crossed above long MA"}
        elif prev_short >= prev_long and current_short < current_long:
            return {"signal": "sell", "reason": "Death cross - short MA crossed below long MA"}
        else:
            trend = "bullish" if current_short > current_long else "bearish"
            return {"signal": "hold", "reason": f"No crossover, trend is {trend}"}

class RSIStrategy:
    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.name = "RSI"

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        if not self._validate_data(data):
            return {"signal": "hold", "reason": "Invalid data format"}
        
        """Generate signals based on RSI levels"""
        if len(data) < self.rsi_period + 1:
            return {"signal": "hold", "reason": "Insufficient data"}
        
        # Calculate RSI
        data['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=self.rsi_period).rsi()
        
        current_rsi = data['rsi'].iloc[-1]
        
        if current_rsi < self.oversold:
            return {"signal": "buy", "reason": f"RSI {current_rsi:.1f} below {self.oversold} (oversold)"}
        elif current_rsi > self.overbought:
            return {"signal": "sell", "reason": f"RSI {current_rsi:.1f} above {self.overbought} (overbought)"}
        else:
            return {"signal": "hold", "reason": f"RSI {current_rsi:.1f} in neutral zone"}