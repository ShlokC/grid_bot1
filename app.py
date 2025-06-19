"""
Grid Trading Bot - Simplified Web Interface
No hedge mode, no SAMIG, just simple grid trading.
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
from core.grid_manager import GridManager

# Setup Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global variables
grid_manager = None
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
        os.path.join('logs', 'grid_bot.log'), 
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
        grids = grid_manager.get_all_grids_status()
        active_symbols = grid_manager.get_active_symbols_data()
        return render_template('index.html', grids=grids, active_symbols=active_symbols)
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return render_template('error.html', error='Failed to load page data')

@app.route('/api/grids')
def get_grids():
    """API route to get all grids."""
    grids = grid_manager.get_all_grids_status()
    return jsonify(grids)

@app.route('/api/grid/<grid_id>')
def get_grid(grid_id):
    """API route to get a specific grid."""
    grid = grid_manager.get_grid_status(grid_id)
    if grid:
        return jsonify(grid)
    return jsonify({'error': 'Grid not found'}), 404

@app.route('/grid/create', methods=['GET', 'POST'])
def create_grid():
    """Create a new grid."""
    if request.method == 'POST':
        try:
            logger.debug(f"Form data: {request.form}")
            
            # Get string fields with validation
            symbol = request.form.get('symbol', '').strip()
            if not symbol:
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error='Symbol is required',
                                      available_symbols=available_symbols)
            
            # Get numeric fields with validation and type conversion
            try:
                grid_number = int(request.form.get('grid_number', 0))
                investment = float(request.form.get('investment', 0))
                take_profit_pnl = float(request.form.get('take_profit_pnl', 5.0))
                stop_loss_pnl = float(request.form.get('stop_loss_pnl', 3.0))
                leverage = float(request.form.get('leverage', 20.0))
            except ValueError as e:
                logger.error(f"Type conversion error: {e}")
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error=f'Invalid numeric value: {str(e)}',
                                      available_symbols=available_symbols)
            
            # Get boolean fields
            enable_grid_adaptation = 'enable_grid_adaptation' in request.form
            auto_start = 'auto_start' in request.form
            
            # Validate inputs
            if grid_number <= 0:
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error='Number of grids must be positive',
                                      available_symbols=available_symbols)
                
            if investment <= 0:
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error='Investment amount must be positive',
                                      available_symbols=available_symbols)
            
            if leverage < 1 or leverage > 125:
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error='Leverage must be between 1x and 125x',
                                      available_symbols=available_symbols)
            
            # Log the parameters
            logger.info(f"Creating simplified grid for {symbol} with {grid_number} levels")
            logger.info(f"Investment: ${investment:.2f}, Leverage: {leverage}x")
            logger.info(f"Grid adaptation: {'enabled' if enable_grid_adaptation else 'disabled'}")
            
            # Create grid
            grid_id = grid_manager.create_grid(
                symbol=symbol,                
                grid_number=grid_number,
                investment=investment,
                take_profit_pnl=take_profit_pnl,
                stop_loss_pnl=stop_loss_pnl,
                leverage=leverage,
                enable_grid_adaptation=enable_grid_adaptation
            )
            
            if not grid_id:
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error='Failed to create grid. See logs for details.',
                                      available_symbols=available_symbols)
            
            # Start grid if requested
            if auto_start:
                success = grid_manager.start_grid(grid_id)
                if not success:
                    logger.warning(f"Grid {grid_id} was created but failed to start")
            
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"Error creating grid: {e}", exc_info=True)
            available_symbols = grid_manager.exchange.get_available_symbols()
            return render_template('create_grid.html', 
                                  error=f'An error occurred: {str(e)}',
                                  available_symbols=available_symbols)
    
    # Load available symbols for the dropdown
    available_symbols = grid_manager.exchange.get_available_symbols()
    return render_template('create_grid.html', available_symbols=available_symbols)
@app.route('/api/active-symbols')
def get_active_symbols():
    """API route to get active symbols data."""
    try:
        symbols_data = grid_manager.get_active_symbols_data()
        return jsonify(symbols_data)
    except Exception as e:
        logger.error(f"Error getting active symbols: {e}")
        return jsonify({'error': 'Failed to get active symbols'}), 500

@app.route('/api/active-symbols/refresh')
def refresh_active_symbols():
    """API route to force refresh active symbols data."""
    try:
        success = grid_manager.force_update_active_symbols()
        if success:
            symbols_data = grid_manager.get_active_symbols_data()
            return jsonify({
                'success': True,
                'message': 'Active symbols refreshed',
                'data': symbols_data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to refresh active symbols'
            }), 500
    except Exception as e:
        logger.error(f"Error refreshing active symbols: {e}")
        return jsonify({'error': 'Failed to refresh active symbols'}), 500
@app.route('/grid/start/<grid_id>')
def start_grid(grid_id):
    """Start a grid."""
    try:
        if grid_manager.start_grid(grid_id):
            return redirect(url_for('index'))
        return render_template('error.html', error=f"Failed to start grid {grid_id}")
    except Exception as e:
        logger.error(f"Error starting grid: {e}")
        return render_template('error.html', error=str(e))

@app.route('/grid/stop/<grid_id>')
def stop_grid(grid_id):
    """Stop a grid."""
    try:
        if grid_manager.stop_grid(grid_id):
            return redirect(url_for('index'))
        return render_template('error.html', error=f"Failed to stop grid {grid_id}")
    except Exception as e:
        logger.error(f"Error stopping grid: {e}")
        return render_template('error.html', error=str(e))

@app.route('/grid/delete/<grid_id>')
def delete_grid(grid_id):
    """Delete a grid."""
    try:
        if grid_manager.delete_grid(grid_id):
            return redirect(url_for('index'))
        return render_template('error.html', error=f"Failed to delete grid {grid_id}")
    except Exception as e:
        logger.error(f"Error deleting grid: {e}")
        return render_template('error.html', error=str(e))

@app.route('/grid/edit/<grid_id>', methods=['GET', 'POST'])
def edit_grid(grid_id):
    """Edit a grid's config."""
    grid = grid_manager.get_grid_status(grid_id)
    
    if not grid:
        return render_template('error.html', error=f"Grid {grid_id} not found")
    
    if request.method == 'POST':
        try:
            take_profit_pnl = float(request.form['take_profit_pnl'])
            stop_loss_pnl = float(request.form['stop_loss_pnl'])
            enable_grid_adaptation = 'enable_grid_adaptation' in request.form
            
            logger.info(f"Updating grid {grid_id}: TP={take_profit_pnl}%, SL={stop_loss_pnl}%, "
                       f"Adaptation={enable_grid_adaptation}")
            
            if grid_manager.update_grid_config(
                grid_id=grid_id,
                take_profit_pnl=take_profit_pnl,
                stop_loss_pnl=stop_loss_pnl,
                enable_grid_adaptation=enable_grid_adaptation
            ):
                return redirect(url_for('index'))
            return render_template('error.html', error=f"Failed to update grid {grid_id}")
        except Exception as e:
            logger.error(f"Error updating grid: {e}")
            return render_template('edit_grid.html', grid=grid, error=str(e))
    
    return render_template('edit_grid.html', grid=grid)

def create_templates_directory():
    """Create templates directory and HTML files with proper UTF-8 encoding."""
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # FIXED: Use UTF-8 encoding and simpler symbols for better compatibility
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Simplified Grid Trading Bot</title>
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
        
        /* Grid table styles */
        .grids-section { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        
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
    <h1>Simplified Grid Trading Bot</h1>
    <p class="subtitle">Buy/Sell orders at grid intervals - No hedge mode</p>
    
    <div class="main-container">
        <!-- Left Panel: Grids -->
        <div class="left-panel">
            <div class="grids-section">
                <a href="/grid/create" class="button">Create New Grid</a>
                
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Symbol</th>
                            <th>Grids</th>
                            <th>Investment</th>
                            <th>Leverage</th>
                            <th>PnL</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for grid in grids %}
                        <tr>
                            <td>{{ grid.grid_id[:8] }}</td>
                            <td>{{ grid.display_symbol or grid.symbol }}</td>
                            <td>{{ grid.grid_number }}</td>
                            <td>{{ "%.2f"|format(grid.investment) }}</td>
                            <td>{{ "%.1f"|format(grid.leverage) }}x</td>
                            <td>{{ "%.2f"|format(grid.pnl) }} ({{ "%.2f"|format(grid.pnl_percentage) }}%)</td>
                            <td>{{ "Running" if grid.running else "Stopped" }}</td>
                            <td class="action-cell">
                                {% if grid.running %}
                                <a href="/grid/stop/{{ grid.grid_id }}" class="button red">Stop</a>
                                {% else %}
                                <a href="/grid/start/{{ grid.grid_id }}" class="button blue">Start</a>
                                {% endif %}
                                <a href="/grid/edit/{{ grid.grid_id }}" class="button">Edit</a>
                                <a href="/grid/delete/{{ grid.grid_id }}" class="button red" onclick="return confirm('Are you sure?')">Delete</a>
                            </td>
                        </tr>
                        {% else %}
                        <tr>
                            <td colspan="8">No grids found</td>
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
                        <span class="category-icon active-icon">&#9202;</span>Market Activity
                    </h3>
                    <div>
                        <button class="refresh-btn" onclick="refreshActiveSymbols()">&#8634;</button>
                        <div class="last-update" id="lastUpdate">
                            Last: {{ active_symbols.last_updated_formatted if active_symbols else 'Never' }}
                        </div>
                    </div>
                </div>
                
                <div class="symbols-grid" id="symbolsContainer">
                    <!-- Top Active Symbols -->
                    <div class="symbol-category">
                        <h4><span class="category-icon active-icon">&#9733;</span>Most Active (24hr)</h4>
                        <div id="topActiveList">
                            {% if active_symbols and active_symbols.top_active %}
                                {% for symbol in active_symbols.top_active %}
                                <div class="symbol-item {{ 'positive' if symbol.price_change_pct > 0 else 'negative' }}">
                                    <div class="symbol-info">
                                        <div class="symbol-name">{{ symbol.symbol }}</div>
                                        <div class="symbol-price">${{ "%.6f"|format(symbol.current_price) }}</div>
                                        <div class="symbol-volume" style="font-size: 10px; color: #999;">Vol: {{ "%.0f"|format(symbol.volume) if symbol.volume else 'N/A' }}</div>
                                    </div>
                                    <div class="symbol-change {{ 'change-positive' if symbol.price_change_pct > 0 else 'change-negative' }}">
                                        {{ "{:+.2f}".format(symbol.price_change_pct) }}%
                                    </div>
                                </div>
                                {% endfor %}
                            {% else %}
                            <div style="text-align: center; color: #666; padding: 20px;">Loading...</div>
                            {% endif %}
                        </div>
                    </div>
                    
                    <!-- Top Gainers -->
                    <div class="symbol-category">
                        <h4><span class="category-icon gain-icon">&#9650;</span>Top Gainers (24hr)</h4>
                        <div id="gainersList">
                            {% if active_symbols and active_symbols.gainers %}
                                {% for symbol in active_symbols.gainers %}
                                <div class="symbol-item positive">
                                    <div class="symbol-info">
                                        <div class="symbol-name">{{ symbol.symbol }}</div>
                                        <div class="symbol-price">${{ "%.6f"|format(symbol.current_price) }}</div>
                                        <div class="symbol-volume" style="font-size: 10px; color: #999;">Vol: {{ "%.0f"|format(symbol.volume) if symbol.volume else 'N/A' }}</div>
                                    </div>
                                    <div class="symbol-change change-positive">
                                        +{{ "%.2f"|format(symbol.price_change_pct) }}%
                                    </div>
                                </div>
                                {% endfor %}
                            {% else %}
                            <div style="text-align: center; color: #666; padding: 20px;">Loading...</div>
                            {% endif %}
                        </div>
                    </div>
                    
                    <!-- Top Losers -->
                    <div class="symbol-category">
                        <h4><span class="category-icon loss-icon">&#9660;</span>Top Losers (24hr)</h4>
                        <div id="losersList">
                            {% if active_symbols and active_symbols.losers %}
                                {% for symbol in active_symbols.losers %}
                                <div class="symbol-item negative">
                                    <div class="symbol-info">
                                        <div class="symbol-name">{{ symbol.symbol }}</div>
                                        <div class="symbol-price">${{ "%.6f"|format(symbol.current_price) }}</div>
                                        <div class="symbol-volume" style="font-size: 10px; color: #999;">Vol: {{ "%.0f"|format(symbol.volume) if symbol.volume else 'N/A' }}</div>
                                    </div>
                                    <div class="symbol-change change-negative">
                                        {{ "%.2f"|format(symbol.price_change_pct) }}%
                                    </div>
                                </div>
                                {% endfor %}
                            {% else %}
                            <div style="text-align: center; color: #666; padding: 20px;">Loading...</div>
                            {% endif %}
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
            updateSymbolList('topActiveList', data.top_active, true);
            updateSymbolList('gainersList', data.gainers, false);
            updateSymbolList('losersList', data.losers, false);
        }
        
        function updateSymbolList(containerId, symbols, showDirection) {
            const container = document.getElementById(containerId);
            if (!container || !symbols) return;
            
            if (symbols.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No data</div>';
                return;
            }
            
            const html = symbols.map(symbol => {
                const isPositive = symbol.price_change_pct > 0;
                const changeClass = isPositive ? 'change-positive' : 'change-negative';
                const itemClass = isPositive ? 'positive' : 'negative';
                const changePrefix = showDirection ? (isPositive ? '+' : '') : (isPositive ? '+' : '');
                
                return `
                    <div class="symbol-item ${itemClass}">
                        <div class="symbol-info">
                            <div class="symbol-name">${symbol.symbol}</div>
                            <div class="symbol-price">$${symbol.current_price.toFixed(6)}</div>
                        </div>
                        <div class="symbol-change ${changeClass}">
                            ${changePrefix}${symbol.price_change_pct.toFixed(2)}%
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
    
    # FIXED: Also add UTF-8 encoding to other template files
    with open('templates/create_grid.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Create New Grid</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #333; }
        .subtitle { color: #666; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input[type="text"], input[type="number"], select { 
            width: 100%; padding: 8px; box-sizing: border-box; 
            border: 1px solid #ddd; border-radius: 4px;
        }
        select { height: 35px; }
        .button { 
            display: inline-block; padding: 10px 20px; text-decoration: none; 
            color: white; background-color: #4CAF50; border: none;
            border-radius: 4px; cursor: pointer; margin-right: 10px;
        }
        .button.cancel { background-color: #f44336; }
        .error { color: red; margin-bottom: 15px; }
        .checkbox-group { margin-bottom: 15px; }
        .checkbox-group label { display: inline; margin-left: 5px; }
    </style>
</head>
<body>
    <h1>Create New Grid</h1>
    <p class="subtitle">Simplified grid - places buy orders below price, sell orders above price</p>
    
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    
    <form method="post">
        <div class="form-group">
            <label for="symbol">Symbol:</label>
            <select id="symbol" name="symbol" required>
                <option value="">Select a symbol</option>
                {% for symbol in available_symbols %}
                <option value="{{ symbol }}">{{ symbol }}</option>
                {% endfor %}
            </select>
        </div>
        
        <div class="form-group">
            <label for="grid_number">Number of Grid Levels:</label>
            <input type="number" id="grid_number" name="grid_number" min="2" max="20" value="10" required>
        </div>
        
        <div class="form-group">
            <label for="investment">Investment Amount (USD):</label>
            <input type="number" id="investment" name="investment" step="any" min="10" max="5000" value="100" required>
        </div>
        
        <div class="form-group">
            <label for="leverage">Leverage:</label>
            <input type="number" id="leverage" name="leverage" step="0.1" min="1" max="125" value="20.0" required>
        </div>
        
        <div class="form-group">
            <label for="take_profit_pnl">Take Profit (%):</label>
            <input type="number" id="take_profit_pnl" name="take_profit_pnl" step="any" value="10.0" required>
        </div>
        
        <div class="form-group">
            <label for="stop_loss_pnl">Stop Loss (%):</label>
            <input type="number" id="stop_loss_pnl" name="stop_loss_pnl" step="any" value="5.0" required>
        </div>
        
        <div class="checkbox-group">
            <input type="checkbox" id="enable_grid_adaptation" name="enable_grid_adaptation" checked>
            <label for="enable_grid_adaptation">Enable Grid Adaptation</label>
        </div>
        
        <div class="checkbox-group">
            <input type="checkbox" id="auto_start" name="auto_start">
            <label for="auto_start">Start Grid Automatically</label>
        </div>
        
        <div class="form-group">
            <button type="submit" class="button">Create</button>
            <a href="/" class="button cancel">Cancel</a>
        </div>
    </form>
</body>
</html>
        """)
    
    # FIXED: Add UTF-8 encoding to edit grid template
    with open('templates/edit_grid.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Edit Grid</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #333; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input[type="text"], input[type="number"] { 
            width: 100%; padding: 8px; box-sizing: border-box; 
            border: 1px solid #ddd; border-radius: 4px;
        }
        .button { 
            display: inline-block; padding: 10px 20px; text-decoration: none; 
            color: white; background-color: #4CAF50; border: none;
            border-radius: 4px; cursor: pointer; margin-right: 10px;
        }
        .button.cancel { background-color: #f44336; }
        .error { color: red; margin-bottom: 15px; }
        .grid-details { margin-bottom: 20px; }
        .grid-details p { margin: 5px 0; }
        .checkbox-group { margin-bottom: 15px; }
        .checkbox-group label { display: inline; margin-left: 5px; }
    </style>
</head>
<body>
    <h1>Edit Grid</h1>
    
    <div class="grid-details">
        <p><strong>Grid ID:</strong> {{ grid.grid_id }}</p>
        <p><strong>Symbol:</strong> {{ grid.display_symbol or grid.symbol }}</p>
        <p><strong>Grid Levels:</strong> {{ grid.grid_number }}</p>
        <p><strong>Investment:</strong> ${{ "%.2f"|format(grid.investment) }}</p>
        <p><strong>Leverage:</strong> {{ "%.1f"|format(grid.leverage) }}x</p>
        <p><strong>PnL:</strong> ${{ "%.2f"|format(grid.pnl) }} ({{ "%.2f"|format(grid.pnl_percentage) }}%)</p>
        <p><strong>Status:</strong> {{ "Running" if grid.running else "Stopped" }}</p>
    </div>
    
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    
    <form method="post">
        <div class="form-group">
            <label for="take_profit_pnl">Take Profit (%):</label>
            <input type="number" id="take_profit_pnl" name="take_profit_pnl" step="any" value="{{ grid.take_profit_pnl }}" required>
        </div>
        
        <div class="form-group">
            <label for="stop_loss_pnl">Stop Loss (%):</label>
            <input type="number" id="stop_loss_pnl" name="stop_loss_pnl" step="any" value="{{ grid.stop_loss_pnl }}" required>
        </div>
        
        <div class="checkbox-group">
            <input type="checkbox" id="enable_grid_adaptation" name="enable_grid_adaptation" 
                   {% if grid.get('enable_grid_adaptation', False) %}checked{% endif %}>
            <label for="enable_grid_adaptation">Enable Grid Adaptation</label>
        </div>
        
        <div class="form-group">
            <button type="submit" class="button">Update</button>
            <a href="/" class="button cancel">Cancel</a>
        </div>
    </form>
</body>
</html>
        """)
def initialize_app(config_file, log_level):
    """Initialize the application."""
    global grid_manager, logger
    
    # Setup logging
    logger = setup_logging(log_level)
    logger.info("Starting Simplified Grid Trading Bot (Web Interface)")
    
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
        
        # Initialize grid manager
        logger.info("Initializing simplified grid manager")
        grid_manager = GridManager(exchange, data_store)
        
        # Create templates directory and HTML files
        create_templates_directory()
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        return False
    
    logger.info("Simplified application initialization complete")
    return True

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simplified Grid Trading Bot (Web Interface)')
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
        logger.exception(f"Error running Simplified Grid Trading Bot: {e}")
        return 1
    finally:
        # Clean up
        if grid_manager:
            grid_manager.stop_all_grids()
            logger.info("Simplified Grid Trading Bot stopped")

if __name__ == "__main__":
    sys.exit(main())