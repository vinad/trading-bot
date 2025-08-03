import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import json
import requests
import time
import random

# Import your existing strategy classes
from enhanced_trading_bot import MeanReversionStrategy, MomentumStrategy, RSIStrategy, MACDStrategy, BollingerBandsStrategy, VWAPStrategy, StochasticStrategy, ParabolicSARStrategy, BreakoutStrategy
from enhanced_config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol: {'qty': int, 'avg_price': float}
        self.trade_history = []
        self.equity_curve = []
        # self.strategies = {
        #     'mean_reversion': MeanReversionStrategy(),
        #     'momentum': MomentumStrategy(),
        #     'rsi': RSIStrategy()
        # }
        self.strategies = {
            'mean_reversion': MeanReversionStrategy()
            ,'momentum': MomentumStrategy()
            ,'rsi': RSIStrategy()
            ,'macd': MACDStrategy()
            ,'bollinger_bands': BollingerBandsStrategy()
            ,'vwap': VWAPStrategy()
            ,'stochastic': StochasticStrategy()
            ,'parabolic_sar': ParabolicSARStrategy()
            ,'breakout': BreakoutStrategy()
        }
        # Add this session configuration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
        """Get historical data for backtesting with enhanced error handling"""
        for attempt in range(max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                if attempt > 0:
                    delay = random.uniform(1, 3)
                    time.sleep(delay)
                
                # Create ticker WITHOUT custom session - let yfinance handle it
                ticker = yf.Ticker(symbol)
                
                # Get data with error handling
                data = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if not data.empty and len(data) > 5:  # Ensure we have enough data
                    logger.info(f"Successfully retrieved {len(data)} days of data for {symbol}")
                    return data
                else:
                    logger.warning(f"Empty or insufficient data for {symbol} on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get data for {symbol} after {max_retries} attempts")
                    
        return pd.DataFrame()

    
    def calculate_position_size(self, symbol: str, current_price: float, risk_percent: float = 0.02) -> int:
        """Calculate position size for backtesting"""
        risk_amount = self.current_capital * risk_percent
        shares = int(risk_amount / (current_price * 0.02))  # Assuming 2% stop loss
        return max(1, min(shares, Config.MAX_POSITION_SIZE))
    
    def execute_backtest_trade(self, symbol: str, signal: str, price: float, date: datetime, strategy: str):
        """Execute a trade in the backtest"""
        commission = 0  # Assuming no commission for simplicity
        
        if signal == 'buy':
            # Only buy if we don't have a position
            if symbol not in self.positions or self.positions[symbol]['qty'] <= 0:
                qty = self.calculate_position_size(symbol, price)
                cost = qty * price + commission
                
                if cost <= self.current_capital:
                    self.current_capital -= cost
                    self.positions[symbol] = {'qty': qty, 'avg_price': price}
                    
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'qty': qty,
                        'price': price,
                        'value': cost,
                        'strategy': strategy,
                        'capital_after': self.current_capital
                    }
                    self.trade_history.append(trade)
                    logger.info(f"{date.strftime('%Y-%m-%d')}: BUY {qty} {symbol} @ ${price:.2f} ({strategy})")
        
        elif signal == 'sell':
            # Only sell if we have a position
            if symbol in self.positions and self.positions[symbol]['qty'] > 0:
                qty = self.positions[symbol]['qty']
                proceeds = qty * price - commission
                
                self.current_capital += proceeds
                
                # Calculate P&L
                buy_price = self.positions[symbol]['avg_price']
                pnl = (price - buy_price) * qty
                pnl_pct = (price - buy_price) / buy_price * 100
                
                trade = {
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'qty': qty,
                    'price': price,
                    'value': proceeds,
                    'strategy': strategy,
                    'buy_price': buy_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_after': self.current_capital
                }
                self.trade_history.append(trade)
                
                # Remove position
                self.positions[symbol] = {'qty': 0, 'avg_price': 0}
                
                logger.info(f"{date.strftime('%Y-%m-%d')}: SELL {qty} {symbol} @ ${price:.2f} ({strategy}) - P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
    
    def calculate_portfolio_value(self, date: datetime, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        cash = self.current_capital
        stock_value = 0
        
        for symbol, position in self.positions.items():
            if position['qty'] > 0 and symbol in current_prices:
                stock_value += position['qty'] * current_prices[symbol]
        
        return cash + stock_value
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str):
        """Run the backtest"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Get data for all symbols
        all_data = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, start_date, end_date)
            if not data.empty:
                all_data[symbol] = data
        
        if not all_data:
            logger.error("No data available for backtesting")
            return
        
        # Get all unique dates
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index.date)
        all_dates = sorted(list(all_dates))
        
        # Run backtest day by day
        for date in all_dates:
            date_dt = datetime.combine(date, datetime.min.time())
            current_prices = {}
            
            # Process each symbol for this date
            for symbol in symbols:
                if symbol not in all_data:
                    continue
                
                data = all_data[symbol]
                
                # Get data up to current date for strategy calculation
                historical_data = data[data.index.date <= date].copy()
                
                if len(historical_data) < 30:  # Need enough data for strategies
                    continue
                
                current_price = historical_data['Close'].iloc[-1]
                current_prices[symbol] = current_price
                
                # In your run_backtest method, replace the strategy loop section with this:

                # Run each strategy
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal_data = strategy.generate_signals(historical_data)
                        signal = signal_data.get('signal', 'hold')  # Default to 'hold' if no signal
                        
                        # Debug logging
                        if date == all_dates[0]:  # Log on first date for each symbol
                            logger.info(f"DEBUG - {symbol} {strategy_name}: signal='{signal}', data_points={len(historical_data)}")
                        
                        if signal in ['buy', 'sell']:
                            logger.info(f"SIGNAL - {date}: {signal.upper()} {symbol} @ ${current_price:.2f} ({strategy_name})")
                            self.execute_backtest_trade(symbol, signal, current_price, date_dt, strategy_name)
                    
                    except Exception as e:
                        logger.error(f"Error running {strategy_name} for {symbol} on {date}: {e}")
            
            # Record equity curve
            portfolio_value = self.calculate_portfolio_value(date_dt, current_prices)
            self.equity_curve.append({
                'date': date_dt,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital
            })
    
    def generate_backtest_report(self) -> Dict:
        """Generate comprehensive backtest report"""
        if not self.equity_curve:
            return {"error": "No backtest data available"}
        
        # Calculate performance metrics
        initial_value = self.initial_capital
        final_value = self.equity_curve[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate daily returns for risk metrics
        daily_values = [point['portfolio_value'] for point in self.equity_curve]
        daily_returns = np.diff(daily_values) / daily_values[:-1]
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized
        sharpe_ratio = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
        
        # Max drawdown
        peak = daily_values[0]
        max_dd = 0
        for value in daily_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        # Trade statistics
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        total_trades = len([t for t in self.trade_history if 'pnl' in t])
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Strategy performance
        strategy_performance = {}
        for strategy_name in self.strategies.keys():
            strategy_trades = [t for t in self.trade_history if t.get('strategy') == strategy_name and 'pnl' in t]
            if strategy_trades:
                strategy_pnl = sum(t['pnl'] for t in strategy_trades)
                strategy_performance[strategy_name] = {
                    'total_trades': len(strategy_trades),
                    'total_pnl': strategy_pnl,
                    'avg_pnl': strategy_pnl / len(strategy_trades)
                }
        
        report = {
            'period': f"{self.equity_curve[0]['date'].strftime('%Y-%m-%d')} to {self.equity_curve[-1]['date'].strftime('%Y-%m-%d')}",
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_return_dollar': final_value - initial_value,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_dd * 100,
            'strategy_performance': strategy_performance,
            'trade_history': self.trade_history,
            'equity_curve': self.equity_curve
        }
        
        return report
    
    def print_backtest_results(self):
        """Print formatted backtest results"""
        report = self.generate_backtest_report()
        
        if "error" in report:
            print(report["error"])
            return
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {report['period']}")
        print(f"Initial Capital: ${report['initial_capital']:,.2f}")
        print(f"Final Value: ${report['final_value']:,.2f}")
        print(f"Total Return: {report['total_return_pct']:+.2f}% (${report['total_return_dollar']:+,.2f})")
        print(f"Total Trades: {report['total_trades']}")
        print(f"Win Rate: {report['win_rate_pct']:.1f}%")
        print(f"Average Win: ${report['avg_win']:+.2f}")
        print(f"Average Loss: ${report['avg_loss']:+.2f}")
        print(f"Volatility: {report['volatility_pct']:.1f}%")
        print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {report['max_drawdown_pct']:.1f}%")
        
        print("\nStrategy Performance:")
        print("-" * 40)
        for strategy, perf in report['strategy_performance'].items():
            print(f"{strategy:15} | {perf['total_trades']:3} trades | ${perf['total_pnl']:+8.2f} | ${perf['avg_pnl']:+6.2f}/trade")
        
        print("\nRecent Trades:")
        print("-" * 40)
        for trade in report['trade_history'][-10:]:  # Last 10 trades
            if 'pnl' in trade:
                print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['action']} {trade['symbol']} | ${trade['pnl']:+7.2f}")

    def debug_strategy_signals(self, symbols: List[str], start_date: str, end_date: str):
        """Debug why strategies aren't generating signals"""
        print("\n" + "="*60)
        print("DEBUGGING STRATEGY SIGNALS")
        print("="*60)
        
        # Get data for all symbols
        all_data = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, start_date, end_date)
            if not data.empty:
                all_data[symbol] = data
                print(f"\n{symbol} Data Shape: {data.shape}")
                print(f"Date Range: {data.index[0].date()} to {data.index[-1].date()}")
                print(f"Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        if not all_data:
            print("ERROR: No data available for any symbols!")
            return
        
        # Test strategies on the last available data for each symbol
        for symbol in symbols:
            if symbol not in all_data:
                continue
                
            data = all_data[symbol]
            print(f"\n--- Testing {symbol} ---")
            
            # Test each strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    signal_data = strategy.generate_signals(data)
                    signal = signal_data.get('signal', 'No signal')
                    
                    print(f"{strategy_name:15}: {signal}")
                    
                    # Print additional debug info if available
                    if isinstance(signal_data, dict):
                        for key, value in signal_data.items():
                            if key != 'signal' and isinstance(value, (int, float)):
                                print(f"  {key}: {value:.4f}")
                                
                except Exception as e:
                    print(f"{strategy_name:15}: ERROR - {e}")

# Replace your run_backtest_example function with this debug version:
def run_backtest_example():
    """Example of how to run a backtest with debugging"""
    # Create backtest engine
    backtest = BacktestEngine(initial_capital=100000)
    
    # Define test parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA','UBER','AMD','NVDA','INTC','IBM','F','GM','GE','BA','CAT','MMM','XOM','CVX','BP','T','VZ']
    end_date = datetime.now().strftime('%Y-%m-%d')#'2025-04-17'
    start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')#'2025-01-01'
    
    # First, debug the strategies
    backtest.debug_strategy_signals(symbols, start_date, end_date)
    
    # Then run the actual backtest
    print("\n" + "="*60)
    print("RUNNING BACKTEST")
    print("="*60)
    
    backtest.run_backtest(symbols, start_date, end_date)
    backtest.print_backtest_results()
    
    # Save detailed report
    report = backtest.generate_backtest_report()
    with open(f'backtest_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

if __name__ == "__main__":
    run_backtest_example()