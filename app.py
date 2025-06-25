"""
Grid Trading Bot - Simplified Web Interface
Updated for Signal Strategy instead of misleading Grid Strategy.
"""
import os
import sys
import logging
import json
import argparse
from typing import Dict, List, Any

from flask import Flask, render_template, request, redirect, url_for, jsonify

from core.exchange import Exchange
from core.data_store import DataStore
from core.signal_manager import SignalManager  # Updated import

# Setup Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global variables
signal_manager = None  # Renamed from grid_manager
logger = None

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = log_level_map.get(log_level.upper(), logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join('logs', 'signal_bot.log'),  # Updated filename
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    logging.getLogger("core.adaptive_crypto_signals").setLevel(logging.DEBUG)  # Set specific logger to DEBUG for detailed output
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def load_config(config_file: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

# Flask routes
@app.route('/')
def index():
    """Home page route."""
    try:
        strategies = signal_manager.get_all_strategies_status()
        active_symbols = signal_manager.get_active_symbols_data()
        return render_template('index.html', strategies=strategies, active_symbols=active_symbols)  # Fixed variable name
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return render_template('error.html', error='Failed to load page data')

@app.route('/api/strategies')  # Updated route name
def get_strategies():
    """API route to get all strategies."""
    strategies = signal_manager.get_all_strategies_status()
    return jsonify(strategies)

@app.route('/api/strategy/<strategy_id>')  # Updated route name
def get_strategy(strategy_id):
    """API route to get a specific strategy."""
    strategy = signal_manager.get_strategy_status(strategy_id)
    if strategy:
        return jsonify(strategy)
    return jsonify({'error': 'Strategy not found'}), 404

@app.route('/sync-active-symbols', methods=['POST'])  # New route for manual sync
def sync_active_symbols():
    """Manually sync strategies with active symbols."""
    try:
        # Force update active symbols and auto-manage strategies
        signal_manager.force_update_active_symbols()
        signal_manager._auto_manage_active_strategies()
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error syncing active symbols: {e}")
        return render_template('error.html', error=str(e))

@app.route('/strategy/start/<strategy_id>')  # Updated route name
def start_strategy(strategy_id):
    """Start a strategy."""
    try:
        if signal_manager.start_strategy(strategy_id):
            return redirect(url_for('index'))
        return render_template('error.html', error=f"Failed to start strategy {strategy_id}")
    except Exception as e:
        logger.error(f"Error starting strategy: {e}")
        return render_template('error.html', error=str(e))

@app.route('/strategy/stop/<strategy_id>')  # Updated route name
def stop_strategy(strategy_id):
    """Stop a strategy."""
    try:
        if signal_manager.stop_strategy(strategy_id):
            return redirect(url_for('index'))
        return render_template('error.html', error=f"Failed to stop strategy {strategy_id}")
    except Exception as e:
        logger.error(f"Error stopping strategy: {e}")
        return render_template('error.html', error=str(e))

@app.route('/strategy/delete/<strategy_id>')  # Updated route name
def delete_strategy(strategy_id):
    """Delete a strategy."""
    try:
        if signal_manager.delete_strategy(strategy_id):
            return redirect(url_for('index'))
        return render_template('error.html', error=f"Failed to delete strategy {strategy_id}")
    except Exception as e:
        logger.error(f"Error deleting strategy: {e}")
        return render_template('error.html', error=str(e))
@app.route('/api/active-symbols')
def get_active_symbols_api():
    """API route to get active symbols data."""
    try:
        active_symbols = signal_manager.get_active_symbols_data()
        return jsonify(active_symbols)
    except Exception as e:
        logger.error(f"Error getting active symbols: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/active-symbols/refresh')
def refresh_active_symbols_api():
    """API route to force refresh active symbols."""
    try:
        success = signal_manager.force_update_active_symbols()
        if success:
            active_symbols = signal_manager.get_active_symbols_data()
            return jsonify({'success': True, 'data': active_symbols})
        else:
            return jsonify({'success': False, 'error': 'Failed to refresh'})
    except Exception as e:
        logger.error(f"Error refreshing active symbols: {e}")
        return jsonify({'success': False, 'error': str(e)})
@app.route('/create_strategy', methods=['GET', 'POST'])
def create_strategy():
    """Create new strategy with strategy selection."""
    try:
        if request.method == 'GET':
            # Get available symbols for dropdown
            available_symbols = signal_manager.exchange.get_available_symbols()[:50]  # Limit for UI
            
            # Define available strategies (from test_strategies.py)
            available_strategies = {
                'qqe_supertrend_fixed': {
                    'name': 'QQE + Supertrend (Current)',
                    'description': 'Current QQE + Supertrend with fixed logic',
                    'params': {'qqe_length': 12, 'qqe_smooth': 5, 'supertrend_period': 10, 'supertrend_multiplier': 2.8}
                },
                'qqe_supertrend_fast': {
                    'name': 'Fast QQE + Supertrend', 
                    'description': 'Faster QQE + Supertrend for 3m crypto',
                    'params': {'qqe_length': 8, 'qqe_smooth': 3, 'supertrend_period': 6, 'supertrend_multiplier': 2.2}
                },
                'rsi_macd': {
                    'name': 'RSI + MACD',
                    'description': 'RSI Oversold/Overbought + MACD Confirmation',
                    'params': {'rsi_length': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9}
                },
                'tsi_vwap': {
                    'name': 'TSI + VWAP',
                    'description': 'Fast TSI + VWAP Trend Following', 
                    'params': {'tsi_fast': 8, 'tsi_slow': 15, 'tsi_signal': 6}
                }
            }
            
            return render_template('create_strategy.html', 
                                 available_symbols=available_symbols,
                                 available_strategies=available_strategies)
        
        elif request.method == 'POST':
            symbol = request.form.get('symbol', '').strip()
            strategy_type = request.form.get('strategy_type', 'qqe_supertrend_fixed')
            auto_start = 'auto_start' in request.form
            
            if not symbol:
                return render_template('create_strategy.html', error='Symbol is required')
            
            # Create strategy with selected type
            strategy_id = signal_manager.create_strategy(symbol, strategy_type=strategy_type)
            
            if not strategy_id:
                return render_template('create_strategy.html', error='Failed to create strategy')
            
            # Auto-start if requested
            if auto_start:
                signal_manager.start_strategy(strategy_id)
            
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error in create_strategy: {e}")
        return render_template('error.html', error=str(e))
# Replace the create_templates_directory() function in app.py

def create_templates_directory():
    """Create templates directory and HTML files with proper UTF-8 encoding."""
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Updated index.html template for multi-strategy auto-management
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Strategy Signal Trading Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        h1 { color: #333; margin-bottom: 10px; }
        .subtitle { color: #666; margin-bottom: 20px; }
        
        /* Container for main content */
        .main-container { display: flex; gap: 20px; flex-wrap: wrap; }
        .left-panel { flex: 2; min-width: 600px; }
        .right-panel { flex: 1; min-width: 300px; }
        
        /* Strategy table styles */
        .strategies-section { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        
        /* Strategy badge styles */
        .strategy-badge {
            background: #e9ecef;
            border: 1px solid #ced4da;
            border-radius: 12px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: bold;
            color: #495057;
        }
        
        /* Auto-sync controls */
        .auto-sync-info { background: #e8f5e8; border: 1px solid #c3e6c3; border-radius: 4px; padding: 15px; margin-bottom: 15px; }
        .auto-sync-info h3 { margin-top: 0; color: #2d5016; }
        .sync-button { 
            background-color: #28a745; color: white; border: none; padding: 10px 20px; 
            border-radius: 4px; cursor: pointer; font-weight: bold; margin-right: 10px;
        }
        .sync-button:hover { background-color: #218838; }
        .create-button {
            background-color: #007bff; color: white; border: none; padding: 10px 20px;
            border-radius: 4px; cursor: pointer; font-weight: bold; text-decoration: none; display: inline-block;
        }
        .create-button:hover { background-color: #0056b3; }
        
        /* Active symbols section */
        .active-symbols-section { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .symbols-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px; }
        .symbol-category { background: #f8f9fa; border-radius: 6px; padding: 15px; }
        .symbol-category h4 { margin: 0 0 10px 0; color: #333; }
        .symbol-item { background: white; border-radius: 4px; padding: 8px 12px; margin-bottom: 8px; border-left: 4px solid #ddd; display: flex; justify-content: space-between; align-items: center; }
        .symbol-item:last-child { margin-bottom: 0; }
        
        /* Color coding for changes */
        .positive { border-left-color: #28a745 !important; }
        .negative { border-left-color: #dc3545 !important; }
        .change-positive { color: #28a745; font-weight: bold; }
        .change-negative { color: #dc3545; font-weight: bold; }
        
        /* Active strategy indicator */
        .active-strategy { border-left-color: #007bff !important; background-color: #f0f8ff; }
        .strategy-indicator { font-size: 12px; color: #007bff; font-weight: bold; }
        
        /* Symbol info layout */
        .symbol-info { display: flex; flex-direction: column; }
        .symbol-name { font-weight: bold; font-size: 14px; }
        .symbol-price { font-size: 12px; color: #666; }
        
        /* Button styles */
        .button { 
            display: inline-block; padding: 8px 16px; text-decoration: none; 
            color: white; background-color: #4CAF50; border-radius: 4px; margin-right: 5px;
        }
        .button.red { background-color: #f44336; }
        .button.blue { background-color: #2196F3; }
        .action-cell { white-space: nowrap; }
        
        /* Refresh controls */
        .refresh-info { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .last-update { font-size: 12px; color: #666; }
        .refresh-btn { background: none; border: 1px solid #ddd; border-radius: 4px; padding: 5px 10px; cursor: pointer; }
        .refresh-btn:hover { background: #f0f0f0; }
        .loading { opacity: 0.6; }
        
        /* Category icons */
        .category-icon { display: inline-block; margin-right: 5px; }
        .active-icon { color: #ff6b35; }
        .gain-icon { color: #28a745; }
        .loss-icon { color: #dc3545; }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main-container { flex-direction: column; }
            .left-panel, .right-panel { min-width: auto; }
            .symbols-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <h1>Multi-Strategy Signal Trading Bot</h1>
    <p class="subtitle">Auto-managed multi-strategy signal trading with QQE+ST, RSI+MACD, TSI+VWAP</p>
    
    <div class="main-container">
        <!-- Left Panel: Auto-Managed Strategies -->
        <div class="left-panel">
            <div class="strategies-section">
                <div class="auto-sync-info">
                    <h3>🤖 Multi-Strategy Management</h3>
                    <p>System supports multiple signal strategies: QQE+Supertrend, RSI+MACD, TSI+VWAP. Each strategy auto-manages positions with $1 fixed sizing and 20x leverage.</p>
                    <form action="/sync-active-symbols" method="post" style="display: inline;">
                        <button type="submit" class="sync-button">🔄 Sync Active Symbols</button>
                    </form>
                    <a href="/create_strategy" class="create-button">+ Create New Strategy</a>
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Strategy</th>
                            <th>Real-time PnL</th>
                            <th>Trades</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for strategy in strategies %}
                        <tr>
                            <td><strong>{{ strategy.display_symbol or strategy.symbol }}</strong></td>
                            <td>
                                <span class="strategy-badge">{{ strategy.strategy_display or strategy.strategy_type }}</span>
                            </td>
                            <td>
                                <span class="{{ 'change-positive' if strategy.pnl >= 0 else 'change-negative' }}">
                                    ${{ "%.2f"|format(strategy.pnl) }} ({{ "%.2f"|format(strategy.pnl_percentage) }}%)
                                </span>
                            </td>
                            <td>{{ strategy.trades_count or 0 }}</td>
                            <td>
                                {% if strategy.running %}
                                    {% if strategy.has_position %}
                                        <span style="color: #28a745;">🟢 Auto-Running (Position)</span>
                                    {% else %}
                                        <span style="color: #28a745;">🟢 Auto-Running</span>
                                    {% endif %}
                                {% else %}
                                    <span style="color: #dc3545;">🔴 Auto-Stopped</span>
                                {% endif %}
                            </td>
                            <td class="action-cell">
                                {% if strategy.running %}
                                    <a href="/strategy/stop/{{ strategy.strategy_id }}" class="button red" onclick="return confirm('Stop this strategy?')">Stop</a>
                                {% else %}
                                    <a href="/strategy/start/{{ strategy.strategy_id }}" class="button">Start</a>
                                {% endif %}
                                <a href="/strategy/delete/{{ strategy.strategy_id }}" class="button red" onclick="return confirm('Delete this strategy permanently?')">Delete</a>
                            </td>
                        </tr>
                        {% else %}
                        <tr>
                            <td colspan="6">No strategies found. Click "Create New Strategy" to get started.</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Right Panel: Active Symbols -->
        <div class="right-panel">
            <div class="active-symbols-section">
                <div class="refresh-info">
                    <h3 style="margin: 0;">
                        <span class="category-icon active-icon">📈</span>Market Overview
                    </h3>
                    <div>
                        <button class="refresh-btn" onclick="refreshActiveSymbols()">🔄</button>
                        <div class="last-update" id="lastUpdate">
                            Last: {{ active_symbols.last_updated_formatted if active_symbols else 'Never' }}
                        </div>
                    </div>
                </div>
                
                <div class="symbols-grid" id="symbolsContainer">
                    <!-- Top Active Symbols -->
                    <div class="symbol-category">
                        <h4><span class="category-icon active-icon">🤖</span>Top Active</h4>
                        <div id="topActiveList">
                            {% if active_symbols and active_symbols.top_active %}
                                {% for symbol in active_symbols.top_active[:5] %}
                                <div class="symbol-item">
                                    <div class="symbol-info">
                                        <div class="symbol-name">{{ symbol.symbol }}</div>
                                        <div class="symbol-price">${{ "%.6f"|format(symbol.current_price) }}</div>
                                    </div>
                                    <div class="symbol-change {{ 'change-positive' if symbol.price_change_pct > 0 else 'change-negative' }}">
                                        {{ "{:+.2f}".format(symbol.price_change_pct) }}%
                                    </div>
                                </div>
                                {% endfor %}
                            {% else %}
                            <div style="text-align: center; color: #666; padding: 20px;">Loading active symbols...</div>
                            {% endif %}
                        </div>
                    </div>
                    
                    <!-- Strategy Stats -->
                    <div class="symbol-category">
                        <h4><span class="category-icon gain-icon">📊</span>Multi-Strategy Stats</h4>
                        <div>
                            <p style="font-size: 12px; color: #666; margin: 5px 0;">
                                🎯 QQE+ST: Technical momentum
                            </p>
                            <p style="font-size: 12px; color: #666; margin: 5px 0;">
                                📈 RSI+MACD: Range detection  
                            </p>
                            <p style="font-size: 12px; color: #666; margin: 5px 0;">
                                🌊 TSI+VWAP: Trend following
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh page every 30 seconds
        setTimeout(function() { location.reload(); }, 30000);
        
        // Auto-refresh active symbols every 15 seconds
        setInterval(updateActiveSymbols, 15000);
        
        function refreshActiveSymbols() {
            const container = document.getElementById('symbolsContainer');
            const lastUpdate = document.getElementById('lastUpdate');
            
            container.classList.add('loading');
            lastUpdate.textContent = 'Refreshing...';
            
            fetch('/api/active-symbols/refresh')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateSymbolsDisplay(data.data);
                        lastUpdate.textContent = 'Last: ' + data.data.last_updated_formatted;
                    } else {
                        lastUpdate.textContent = 'Refresh failed';
                    }
                })
                .catch(error => {
                    lastUpdate.textContent = 'Error refreshing';
                })
                .finally(() => {
                    container.classList.remove('loading');
                });
        }
        
        function updateActiveSymbols() {
            fetch('/api/active-symbols')
                .then(response => response.json())
                .then(data => {
                    updateSymbolsDisplay(data);
                    document.getElementById('lastUpdate').textContent = 'Last: ' + data.last_updated_formatted;
                })
                .catch(error => {
                    console.error('Error updating symbols:', error);
                });
        }
        
        function updateSymbolsDisplay(data) {
            updateTopActiveList(data.top_active);
        }
        
        function updateTopActiveList(symbols) {
            const container = document.getElementById('topActiveList');
            if (!container || !symbols) return;
            
            if (symbols.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No active symbols</div>';
                return;
            }
            
            const html = symbols.slice(0, 5).map(symbol => {
                const isPositive = symbol.price_change_pct > 0;
                const changeClass = isPositive ? 'change-positive' : 'change-negative';
                
                return `
                    <div class="symbol-item">
                        <div class="symbol-info">
                            <div class="symbol-name">${symbol.symbol}</div>
                            <div class="symbol-price">$${symbol.current_price.toFixed(6)}</div>
                        </div>
                        <div class="symbol-change ${changeClass}">
                            ${symbol.price_change_pct > 0 ? '+' : ''}${symbol.price_change_pct.toFixed(2)}%
                        </div>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = html;
        }
    </script>
</body>
</html>
        """)
    
    # Strategy creation template with proper strategy selection
    with open('templates/create_strategy.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Create New Signal Strategy</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #333; }
        .subtitle { color: #666; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select, input[type="text"] { 
            width: 100%; padding: 8px; box-sizing: border-box; 
            border: 1px solid #ddd; border-radius: 4px; height: 35px;
        }
        .button { 
            display: inline-block; padding: 10px 20px; text-decoration: none; 
            color: white; background-color: #4CAF50; border: none;
            border-radius: 4px; cursor: pointer; margin-right: 10px;
        }
        .button.cancel { background-color: #f44336; }
        .error { color: red; margin-bottom: 15px; }
        .checkbox-group { margin-bottom: 15px; }
        .checkbox-group label { display: inline; margin-left: 5px; font-weight: normal; }
        .info-box { background: #e8f4f8; border: 1px solid #bee5eb; border-radius: 4px; padding: 15px; margin-bottom: 20px; }
        .info-box h3 { margin-top: 0; color: #0c5460; }
        .strategy-info { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px; margin-top: 10px; font-size: 14px; }
        .strategy-params { font-family: monospace; color: #6c757d; font-size: 12px; margin-top: 5px; }
    </style>
    <script>
        function updateStrategyInfo() {
            const strategySelect = document.getElementById('strategy_type');
            const strategyInfo = document.getElementById('strategy_info');
            const selectedStrategy = strategySelect.value;
            
            const strategies = {
                'qqe_supertrend_fixed': {
                    description: 'Current QQE + Supertrend with fixed logic. Balanced approach for most market conditions.',
                    params: 'QQE(12,5), Supertrend(10, 2.8)',
                    exit: 'Both indicators must turn against position'
                },
                'qqe_supertrend_fast': {
                    description: 'Faster QQE + Supertrend optimized for 3-minute crypto trading. More responsive to quick moves.',
                    params: 'QQE(8,3), Supertrend(6, 2.2)',
                    exit: 'Both indicators must turn against position'
                },
                'rsi_macd': {
                    description: 'RSI oversold/overbought levels with MACD trend confirmation. Good for ranging markets.',
                    params: 'RSI(14), MACD(12,26,9)',
                    exit: 'RSI extreme levels + MACD reversal'
                },
                'tsi_vwap': {
                    description: 'Fast TSI momentum with VWAP trend following. Effective for trending markets.',
                    params: 'TSI(8,15,6), VWAP',
                    exit: 'TSI reversal + price moves away from VWAP'
                }
            };
            
            if (strategies[selectedStrategy]) {
                strategyInfo.innerHTML = `
                    <strong>Description:</strong> ${strategies[selectedStrategy].description}<br>
                    <div class="strategy-params"><strong>Parameters:</strong> ${strategies[selectedStrategy].params}</div>
                    <div class="strategy-params"><strong>Exit Logic:</strong> ${strategies[selectedStrategy].exit}</div>
                `;
                strategyInfo.style.display = 'block';
            } else {
                strategyInfo.style.display = 'none';
            }
        }
        
        window.onload = function() {
            updateStrategyInfo();
        };
    </script>
</head>
<body>
    <h1>Create New Signal Strategy</h1>
    <p class="subtitle">Choose strategy type and symbol for automated signal trading</p>
    
    <div class="info-box">
        <h3>Fixed Parameters</h3>
        <ul>
            <li><strong>Position Size:</strong> $1 USD per strategy</li>
            <li><strong>Leverage:</strong> 20x</li>
            <li><strong>Risk Management:</strong> Built-in stop loss and take profit orders</li>
            <li><strong>Exit Logic:</strong> Strategy-specific technical exit conditions</li>
        </ul>
    </div>
    
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    
    <form method="post">
        <div class="form-group">
            <label for="strategy_type">Strategy Type:</label>
            <select id="strategy_type" name="strategy_type" required onchange="updateStrategyInfo()">
                <option value="qqe_supertrend_fixed">QQE + Supertrend (Current)</option>
                <option value="qqe_supertrend_fast">Fast QQE + Supertrend</option>
                <option value="rsi_macd">RSI + MACD</option>
                <option value="tsi_vwap">TSI + VWAP</option>
            </select>
            <div id="strategy_info" class="strategy-info"></div>
        </div>
        
        <div class="form-group">
            <label for="symbol">Symbol:</label>
            <select id="symbol" name="symbol" required>
                <option value="">Select a symbol</option>
                {% for symbol in available_symbols %}
                <option value="{{ symbol }}">{{ symbol }}</option>
                {% endfor %}
            </select>
        </div>
        
        <div class="checkbox-group">
            <input type="checkbox" id="auto_start" name="auto_start" checked>
            <label for="auto_start">Start Strategy Automatically</label>
        </div>
        
        <div class="form-group">
            <button type="submit" class="button">Create Strategy</button>
            <a href="/" class="button cancel">Cancel</a>
        </div>
    </form>
</body>
</html>
        """)
    
    # Error template
    with open('templates/error.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #f44336; }
        .error-message { margin-bottom: 20px; }
        .button { 
            display: inline-block; padding: 10px 20px; text-decoration: none; 
            color: white; background-color: #2196F3; border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>Error</h1>
    <div class="error-message">{{ error }}</div>
    <a href="/" class="button">Back to Home</a>
</body>
</html>
        """)

def initialize_app(config_file, log_level):
    """Initialize the application."""
    global signal_manager, logger  # Updated variable name
    
    # Setup logging
    logger = setup_logging(log_level)
    logger.info("Starting TSI Signal Trading Bot (Web Interface)")  # Updated log message
    
    # Load configuration
    config = load_config(config_file)
    
    if not config:
        logger.error("Failed to load configuration")
        return False
    
    # Check for required configuration
    if 'api_key' not in config or 'api_secret' not in config:
        logger.error("API key and secret not found in configuration")
        return False
    
    # Initialize components
    try:
        # Initialize exchange
        logger.info("Initializing exchange connection")
        exchange = Exchange(
            api_key=config['api_key'],
            api_secret=config['api_secret']
        )
        
        # Initialize data store
        data_dir = config.get('data_dir', 'data')
        logger.info(f"Initializing data store in directory: {data_dir}")
        data_store = DataStore(data_dir=data_dir)
        
        # Initialize signal manager
        logger.info("Initializing signal strategy manager")  # Updated log message
        signal_manager = SignalManager(exchange, data_store)  # Updated class and variable
        
        # Create templates directory and HTML files
        create_templates_directory()
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        return False
    
    logger.info("Signal trading application initialization complete")  # Updated log message
    return True

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TSI Signal Trading Bot (Web Interface)')  # Updated description
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the web server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web server on')
    args = parser.parse_args()
    
    # Initialize application
    if not initialize_app(args.config, args.log_level):
        return 1
    
    try:
        # Run web server
        app.run(host=args.host, port=args.port, debug=False)
        return 0
        
    except Exception as e:
        logger.exception(f"Error running TSI Signal Trading Bot: {e}")  # Updated log message
        return 1
    finally:
        # Clean up
        if signal_manager:  # Updated variable name
            signal_manager.stop_all_strategies()  # Updated method name
            logger.info("TSI Signal Trading Bot stopped")  # Updated log message

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Set root logger to DEBUG to catch everything
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.exit(main())