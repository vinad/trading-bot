#!/usr/bin/env python3
"""
Trading Bot Dashboard - Real-time monitoring and control
Run this alongside your main trading bot to monitor performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_trading_bot import TradingBot, AlertSystem
from enhanced_config import Config, validate_config
import time
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
import threading
import logging

# Disable some logging for cleaner dashboard
logging.getLogger('yfinance').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

class TradingDashboard:
    def __init__(self):
        self.bot = None
        self.running = False
        self.last_update = None
        self.start_time = datetime.now()
        
    def initialize_bot(self):
        """Initialize connection to trading bot"""
        try:
            validate_config()
            self.bot = TradingBot(
                Config.ALPACA_API_KEY,
                Config.ALPACA_SECRET_KEY,
                Config.ALPACA_BASE_URL
            )
            return True
        except Exception as e:
            print(f"❌ Failed to connect to trading bot: {e}")
            return False
    
    def get_market_status(self):
        """Get current market status"""
        try:
            clock = self.bot.api.get_clock()
            session = self.bot.get_market_session()
            
            return {
                'is_open': clock.is_open,
                'session': session.value,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'current_time': datetime.now()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_account_summary(self):
        """Get account summary"""
        try:
            account = self.bot.api.get_account()
            positions = self.bot.api.list_positions()
            
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'day_pl': float(account.unrealized_pl),
                'total_pl': float(account.unrealized_pl) + float(account.realized_pl),
                'position_count': len(positions),
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'qty': int(pos.qty),
                        'market_value': float(pos.market_value),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_pl_pct': float(pos.unrealized_plpc) * 100
                    }
                    for pos in positions
                ]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_recent_orders(self, limit=10):
        """Get recent orders"""
        try:
            orders = self.bot.api.list_orders(
                status='all',
                limit=limit,
                direction='desc'
            )
            
            return [
                {
                    'symbol': order.symbol,
                    'side': order.side,
                    'qty': int(order.qty),
                    'status': order.status,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
                }
                for order in orders
            ]
        except Exception as e:
            return {'error': str(e)}
    
    def display_dashboard(self):
        """Display the main dashboard"""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("🤖 TRADING BOT DASHBOARD")
        print("=" * 80)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Runtime: {datetime.now() - self.start_time}")
        print()
        
        # Market Status
        market_status = self.get_market_status()
        if 'error' not in market_status:
            print("📊 MARKET STATUS")
            print("-" * 40)
            session_emoji = {
                'regular': '🟢',
                'pre_market': '🟡',
                'after_hours': '🟠',
                'closed': '🔴'
            }
            emoji = session_emoji.get(market_status['session'], '❓')
            print(f"{emoji} Session: {market_status['session'].upper()}")
            print(f"🕐 Current: {market_status['current_time'].strftime('%H:%M:%S')}")
            if not market_status['is_open']:
                print(f"🔜 Next Open: {market_status['next_open'].strftime('%Y-%m-%d %H:%M')}")
            print()
        
        # Account Summary
        account = self.get_account_summary()
        if 'error' not in account:
            print("💰 ACCOUNT SUMMARY")
            print("-" * 40)
            print(f"💎 Equity: ${account['equity']:,.2f}")
            print(f"💸 Buying Power: ${account['buying_power']:,.2f}")
            print(f"📈 Day P&L: ${account['day_pl']:+,.2f}")
            print(f"📊 Total P&L: ${account['total_pl']:+,.2f}")
            print(f"📦 Positions: {account['position_count']}")
            print()
            
            # Positions
            if account['positions']:
                print("📋 CURRENT POSITIONS")
                print("-" * 60)
                print(f"{'Symbol':<8} {'Qty':<6} {'Value':<12} {'P&L':<12} {'%':<8}")
                print("-" * 60)
                for pos in account['positions']:
                    pnl_color = "🟢" if pos['unrealized_pl'] >= 0 else "🔴"
                    print(f"{pos['symbol']:<8} {pos['qty']:<6} "
                          f"${pos['market_value']:>10.2f} "
                          f"{pnl_color}${pos['unrealized_pl']:>9.2f} "
                          f"{pos['unrealized_pl_pct']:>6.1f}%")
                print()
        
        # Recent Orders
        orders = self.get_recent_orders(5)
        if 'error' not in orders and orders:
            print("📝 RECENT ORDERS (Last 5)")
            print("-" * 70)
            print(f"{'Time':<8} {'Symbol':<8} {'Side':<4} {'Qty':<6} {'Status':<10} {'Price':<10}")
            print("-" * 70)
            for order in orders:
                time_str = order['submitted_at'].strftime('%H:%M') if order['submitted_at'] else 'N/A'
                side_emoji = "🟢" if order['side'] == 'buy' else "🔴"
                status_emoji = "✅" if order['status'] == 'filled' else "⏳"
                price_str = f"${order['filled_avg_price']:.2f}" if order['filled_avg_price'] else "N/A"
                
                print(f"{time_str:<8} {order['symbol']:<8} {side_emoji}{order['side']:<3} "
                      f"{order['qty']:<6} {status_emoji}{order['status']:<9} {price_str:<10}")
            print()
        
        # Performance Metrics
        if hasattr(self, 'start_equity'):
            current_equity = account.get('equity', 0)
            performance = ((current_equity - self.start_equity) / self.start_equity) * 100
            print("📈 PERFORMANCE")
            print("-" * 40)
            print(f"🚀 Since Start: {performance:+.2f}% (${current_equity - self.start_equity:+.2f})")
        
        # Instructions
        print("\n" + "=" * 80)
        print("🔧 CONTROLS: Press Ctrl+C to exit | Bot runs independently")
        print("💡 TIP: Your main bot should be running in another terminal")
        print("🔄 Dashboard updates every 30 seconds")
        print("=" * 80)
    
    def monitor_log_file(self):
        """Monitor log file for real-time updates"""
        log_file = Config.LOG_FILE
        if not os.path.exists(log_file):
            return
        
        try:
            with open(log_file, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                while self.running:
                    line = f.readline()
                    if line:
                        # Check for important events
                        if any(keyword in line.lower() for keyword in ['trade', 'buy', 'sell', 'error']):
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            print(f"\n🚨 {timestamp} - {line.strip()}")
                    time.sleep(1)
        except Exception as e:
            pass
    
    def run(self):
        """Run the dashboard"""
        if not self.initialize_bot():
            return
        
        # Get starting equity for performance tracking
        try:
            account = self.get_account_summary()
            self.start_equity = account.get('equity', 0)
        except:
            self.start_equity = 0                                                  
        
        self.running = True
        
        # Start log monitoring in background
        log_thread = threading.Thread(target=self.monitor_log_file, daemon=True)
        log_thread.start()
        
        print("🚀 Starting Trading Bot Dashboard...")
        print("💡 Make sure your main trading bot is running in another terminal!")
        time.sleep(3)
        
        try:
            while self.running:
                self.display_dashboard()
                
                # Update every 30 seconds
                for _ in range(30):
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Dashboard stopped by user")
            self.running = False
        except Exception as e:
            print(f"\n❌ Dashboard error: {e}")
        
        print("👋 Dashboard shutdown complete")

def main():
    """Main function"""
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()