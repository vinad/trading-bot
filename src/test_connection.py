#!/usr/bin/env python3
"""
Simple script to test Alpaca API connection
"""

import sys
import os
sys.path.append('src')

try:
    from trading_bot import TradingBot
    from config import Config, validate_config
    
    print("Testing Alpaca API connection...")
    print(f"API Key: {Config.ALPACA_API_KEY[:8]}...")
    print(f"Base URL: {Config.ALPACA_BASE_URL}")
    
    # Validate config
    validate_config()
    print("✓ Configuration is valid")
    
    # Test connection
    bot = TradingBot(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY, Config.ALPACA_BASE_URL)
    print("✓ Successfully connected to Alpaca API")
    
    # Get account info
    account = bot.account
    print(f"✓ Account equity: ${float(account.equity):,.2f}")
    print(f"✓ Buying power: ${float(account.buying_power):,.2f}")
    print(f"✓ Account status: {account.status}")
    
    # Test market data
    try:
        clock = bot.api.get_clock()
        print(f"✓ Market is {'OPEN' if clock.is_open else 'CLOSED'}")
        print(f"✓ Next market open: {clock.next_open}")
    except Exception as e:
        print(f"⚠ Market data error: {e}")
    
    print("\n🎉 All tests passed! Your bot is ready to run.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have created all the required files")
    
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    print("Please check your .env file and API credentials")
    
except Exception as e:
    print(f"❌ Connection error: {e}")
    print("Possible issues:")
    print("1. Check your API credentials in .env file")
    print("2. Make sure you're using PAPER trading credentials")
    print("3. Verify internet connection")