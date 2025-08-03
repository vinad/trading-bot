#Now you can run it anytime with: ~/check_bot_status.sh

cat << 'EOF' > ~/check_bot_status.sh
#!/bin/bash
echo "🤖 TRADING BOT STATUS CHECK"
echo "=========================="
echo "📅 Current Time: $(date)"
echo ""

echo "🖥️  Screen Sessions:"
screen -ls 2>/dev/null || echo "No screen sessions found"
echo ""

echo "🐍 Python Processes:"
ps aux | grep "python.*enhanced_main.py" | grep -v grep || echo "No trading bot process found"
echo ""

echo "💾 System Resources:"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')"
echo ""

echo "📊 Recent Activity (if log exists):"
if [ -f "trading_bot.log" ]; then
    tail -5 trading_bot.log
else
    echo "No log file found"
fi
EOF

chmod +x ~/check_bot_status.sh
