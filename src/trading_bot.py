import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import ta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        """
        Initialize the trading bot with Alpaca API credentials
        For paper trading, use: https://paper-api.alpaca.markets
        """
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.account = self.api.get_account()
        self.positions = {}
        self.strategies = {}
        
        logger.info(f"Bot initialized. Account equity: ${float(self.account.equity):.2f}")
    
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
        
        # Get current price
        quote = self.api.get_latest_quote(symbol)
        current_price = float(quote.ask_price)
        
        # Simple position sizing (can be enhanced with volatility-based sizing)
        shares = int(risk_amount / (current_price * 0.02))  # 2% stop loss assumption
        return max(1, shares)

class MeanReversionStrategy:
    def __init__(self, lookback_period: int = 20, z_threshold: float = 2.0):
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.name = "Mean Reversion"
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
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
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
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
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
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

class TradingEngine:
    def __init__(self, bot: TradingBot):
        self.bot = bot
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(),
            'momentum': MomentumStrategy(),
            'rsi': RSIStrategy()
        }
        self.portfolio = {}
    
    def execute_strategy(self, symbol: str, strategy_name: str):
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
        logger.info(f"{strategy.name} strategy for {symbol}: {signal['signal']} - {signal['reason']}")
        
        # Execute trade based on signal
        self.execute_trade(symbol, signal['signal'], strategy_name)
    
    def execute_trade(self, symbol: str, signal: str, strategy_name: str):
        """Execute buy/sell orders based on signals"""
        try:
            # Check current position
            try:
                position = self.bot.api.get_position(symbol)
                current_qty = int(position.qty)
            except:
                current_qty = 0
            
            if signal == "buy" and current_qty <= 0:
                qty = self.bot.calculate_position_size(symbol)
                order = self.bot.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"BUY order submitted for {qty} shares of {symbol} using {strategy_name}")
                
            elif signal == "sell" and current_qty > 0:
                order = self.bot.api.submit_order(
                    symbol=symbol,
                    qty=abs(current_qty),
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"SELL order submitted for {abs(current_qty)} shares of {symbol} using {strategy_name}")
            
            else:
                logger.info(f"No action taken for {symbol} - {signal} signal with current position: {current_qty}")
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def run_multi_strategy_scan(self, symbols: List[str]):
        """Run multiple strategies across multiple symbols"""
        logger.info("Starting multi-strategy scan...")
        
        for symbol in symbols:
            logger.info(f"\n--- Analyzing {symbol} ---")
            for strategy_name in self.strategies.keys():
                self.execute_strategy(symbol, strategy_name)
                time.sleep(1)  # Rate limiting
    
    def get_portfolio_summary(self):
        """Get current portfolio summary"""
        account = self.bot.api.get_account()
        positions = self.bot.api.list_positions()
        
        logger.info(f"\n--- Portfolio Summary ---")
        logger.info(f"Account Equity: ${float(account.equity):.2f}")
        logger.info(f"Buying Power: ${float(account.buying_power):.2f}")
        logger.info(f"Day P&L: ${float(account.unrealized_pl):.2f}")
        
        if positions:
            logger.info("\nCurrent Positions:")
            for position in positions:
                logger.info(f"{position.symbol}: {position.qty} shares, "
                          f"Value: ${float(position.market_value):.2f}, "
                          f"P&L: ${float(position.unrealized_pl):.2f}")
        else:
            logger.info("No current positions")

# Example usage
def main():
    # Replace with your Alpaca paper trading credentials
    API_KEY = "YOUR_API_KEY"
    SECRET_KEY = "YOUR_SECRET_KEY"
    
    # Initialize bot
    bot = TradingBot(API_KEY, SECRET_KEY)
    engine = TradingEngine(bot)
    
    # Define watchlist
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Run strategies
    try:
        while True:
            # Check market hours
            clock = bot.api.get_clock()
            if clock.is_open:
                engine.run_multi_strategy_scan(watchlist)
                engine.get_portfolio_summary()
            else:
                logger.info("Market is closed")
            
            # Wait before next scan (adjust frequency as needed)
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")

if __name__ == "__main__":
    main()