"""
Grid Trading Bot - Web Interface Version
"""
import os
import sys
import logging
import json
import argparse
from typing import Dict, List, Any
import threading

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
    """Setup logging configuration with Unicode support."""
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
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        os.path.join('logs', 'grid_bot.log'), 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Console handler with UTF-8 encoding and error handling
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
    
    # Set console encoding for Windows
    if sys.platform.startswith('win'):
        try:
            # Try to set console to UTF-8
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
        except:
            # Fallback: just avoid special characters
            pass
    
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
    grids = grid_manager.get_all_grids_status()
    return render_template('index.html', grids=grids)

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
            # Get form data with explicit type conversion and validation
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
                price_lower = float(request.form.get('price_lower', 0))
                price_upper = float(request.form.get('price_upper', 0))
                grid_number = int(request.form.get('grid_number', 0))
                investment = float(request.form.get('investment', 0))
                take_profit_pnl = float(request.form.get('take_profit_pnl', 5.0))
                stop_loss_pnl = float(request.form.get('stop_loss_pnl', 3.0))
                leverage = float(request.form.get('leverage', 25.0))
            except ValueError as e:
                logger.error(f"Type conversion error: {e}")
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error=f'Invalid numeric value: {str(e)}',
                                      available_symbols=available_symbols)
            
            # Get boolean fields
            enable_grid_adaptation = 'enable_grid_adaptation' in request.form
            enable_samig = 'enable_samig' in request.form
            
            # Validate inputs
            if price_lower >= price_upper:
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error='Lower price must be less than upper price',
                                      available_symbols=available_symbols)
                
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
            
            if leverage < 1 or leverage > 75:
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error='Leverage must be between 1x and 75x',
                                      available_symbols=available_symbols)
            
            # Log the parameters for debugging
            logger.info(f"Creating grid for {symbol} with {grid_number} levels")
            logger.info(f"Price range: {price_lower} - {price_upper}")
            logger.info(f"Investment: {investment}, Leverage: {leverage}x")
            logger.info(f"Grid adaptation: {'enabled' if enable_grid_adaptation else 'disabled'}")
            logger.info(f"SAMIG: {'enabled' if enable_samig else 'disabled'}")
            
            # Create grid
            grid_id = grid_manager.create_grid(
                symbol=symbol,                
                grid_number=grid_number,
                investment=investment,
                take_profit_pnl=take_profit_pnl,
                stop_loss_pnl=stop_loss_pnl,
                leverage=leverage,
                enable_grid_adaptation=enable_grid_adaptation,
                enable_samig=enable_samig  
            )
            
            if not grid_id:
                available_symbols = grid_manager.exchange.get_available_symbols()
                return render_template('create_grid.html', 
                                      error='Failed to create grid. See logs for details.',
                                      available_symbols=available_symbols)
            
            # Start grid if requested
            if 'auto_start' in request.form:
                success = grid_manager.start_grid(grid_id)
                if not success:
                    # Grid was created but not started
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
            enable_samig = 'enable_samig' in request.form
            
            logger.info(f"Updating grid {grid_id}: TP={take_profit_pnl}%, SL={stop_loss_pnl}%, "
                       f"Adaptation={enable_grid_adaptation}, SAMIG={enable_samig}")
            
            if grid_manager.update_grid_config(
                grid_id=grid_id,
                take_profit_pnl=take_profit_pnl,
                stop_loss_pnl=stop_loss_pnl,
                enable_grid_adaptation=enable_grid_adaptation,
                enable_samig=enable_samig
            ):
                return redirect(url_for('index'))
            return render_template('error.html', error=f"Failed to update grid {grid_id}")
        except Exception as e:
            logger.error(f"Error updating grid: {e}")
            return render_template('edit_grid.html', grid=grid, error=str(e))
    
    return render_template('edit_grid.html', grid=grid)

def create_templates_directory():
    """Create templates directory and HTML files."""
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Grid Trading Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .button { 
            display: inline-block; padding: 8px 16px; text-decoration: none; 
            color: white; background-color: #4CAF50; border-radius: 4px; margin-right: 5px;
        }
        .button.red { background-color: #f44336; }
        .button.blue { background-color: #2196F3; }
        .action-cell { white-space: nowrap; }
    </style>
</head>
<body>
    <h1>Grid Trading Bot</h1>
    <a href="/grid/create" class="button">Create New Grid</a>
    
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Symbol</th>
                <th>Price Range</th>
                <th>Grids</th>
                <th>Investment</th>
                <th>PnL</th>
                <th>Status</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {% for grid in grids %}
            <tr>
                <td>{{ grid.grid_id[:8] }}</td>
                <td>{{ grid.symbol }}</td>
                <td>{{ grid.price_lower }} - {{ grid.price_upper }}</td>
                <td>{{ grid.grid_number }}</td>
                <td>{{ "%.2f"|format(grid.investment) }}</td>
                <td>{{ "%.2f"|format(grid.pnl) }} ({{ "%.2f"|format(grid.pnl_percentage) }}%)</td>
                <td>{{ "Running" if grid.running else "Stopped" }}</td>
                <td class="action-cell">
                    {% if grid.running %}
                    <a href="/grid/stop/{{ grid.grid_id }}" class="button red">Stop</a>
                    {% else %}
                    <a href="/grid/start/{{ grid.grid_id }}" class="button blue">Start</a>
                    {% endif %}
                    <a href="/grid/edit/{{ grid.grid_id }}" class="button">Edit</a>
                    <a href="/grid/delete/{{ grid.grid_id }}" class="button red" onclick="return confirm('Are you sure you want to delete this grid?')">Delete</a>
                </td>
            </tr>
            {% else %}
            <tr>
                <td colspan="8">No grids found</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <script>
        // Auto-refresh the page every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
        """)
    
    # Create create_grid.html with SAMIG support
    with open('templates/create_grid.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Create New Grid</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #333; }
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
            <label for="price_lower">Lower Price:</label>
            <input type="number" id="price_lower" name="price_lower" step="any" required>
        </div>
        
        <div class="form-group">
            <label for="price_upper">Upper Price:</label>
            <input type="number" id="price_upper" name="price_upper" step="any" required>
        </div>
        
        <div class="form-group">
            <label for="grid_number">Number of Grids:</label>
            <input type="number" id="grid_number" name="grid_number" required>
        </div>
        
        <div class="form-group">
            <label for="investment">Investment Amount:</label>
            <input type="number" id="investment" name="investment" step="any" required>
        </div>
        
        <div class="form-group">
            <label for="leverage">Leverage:</label>
            <input type="number" id="leverage" name="leverage" step="0.1" min="1" max="75" value="25.0" required>
        </div>
        
        <div class="form-group">
            <label for="take_profit_pnl">Take Profit (%):</label>
            <input type="number" id="take_profit_pnl" name="take_profit_pnl" step="any" value="5.0" required>
        </div>
        
        <div class="form-group">
            <label for="stop_loss_pnl">Stop Loss (%):</label>
            <input type="number" id="stop_loss_pnl" name="stop_loss_pnl" step="any" value="3.0" required>
        </div>
        
        <div class="checkbox-group">
            <input type="checkbox" id="enable_grid_adaptation" name="enable_grid_adaptation" checked>
            <label for="enable_grid_adaptation">Enable Grid Adaptation</label>
        </div>
        
        <div class="checkbox-group">
            <input type="checkbox" id="enable_samig" name="enable_samig">
            <label for="enable_samig">Enable SAMIG (Self-Adaptive Market Intelligence)</label>
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
    
    # Create edit_grid.html with SAMIG support
    with open('templates/edit_grid.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Edit Grid</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
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
        <p><strong>Symbol:</strong> {{ grid.symbol }}</p>
        <p><strong>Price Range:</strong> {{ grid.price_lower }} - {{ grid.price_upper }}</p>
        <p><strong>Grid Number:</strong> {{ grid.grid_number }}</p>
        <p><strong>Investment:</strong> {{ "%.2f"|format(grid.investment) }}</p>
        <p><strong>PnL:</strong> {{ "%.2f"|format(grid.pnl) }} ({{ "%.2f"|format(grid.pnl_percentage) }}%)</p>
        <p><strong>Status:</strong> {{ "Running" if grid.running else "Stopped" }}</p>
        {% if grid.get('enable_samig') %}
        <p><strong>SAMIG Status:</strong> {{ "Active" if grid.get('samig_active', False) else "Inactive" }}</p>
        {% if grid.get('samig_active') %}
        <p><strong>Adaptations:</strong> {{ grid.get('adaptation_count', 0) }}</p>
        <p><strong>Market Volatility:</strong> {{ "%.2f"|format(grid.get('volatility_regime', 1.0)) }}</p>
        <p><strong>Momentum:</strong> {{ "%.3f"|format(grid.get('momentum', 0.0)) }}</p>
        {% endif %}
        {% endif %}
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
        
        <div class="checkbox-group">
            <input type="checkbox" id="enable_samig" name="enable_samig" 
                   {% if grid.get('enable_samig', False) %}checked{% endif %}>
            <label for="enable_samig">Enable SAMIG (Self-Adaptive Market Intelligence)</label>
        </div>
        
        <div class="form-group">
            <button type="submit" class="button">Update</button>
            <a href="/" class="button cancel">Cancel</a>
        </div>
    </form>
</body>
</html>
        """)
    
    # Create error.html
    with open('templates/error.html', 'w') as f:
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
    """Initialize the application with enhanced grid loading."""
    global grid_manager, logger
    
    # Setup logging
    logger = setup_logging(log_level)
    logger.info("Starting Grid Trading Bot (Web Interface)")
    
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
        
        # Initialize grid manager (which loads saved grids)
        logger.info("Initializing grid manager and loading saved grids")
        grid_manager = GridManager(exchange, data_store)
        
        # Count loaded grids
        num_grids = len(grid_manager.grids)
        if num_grids > 0:
            logger.info(f"Successfully loaded {num_grids} saved grid(s)")
            
            # Log each grid's status
            for grid_id, grid in grid_manager.grids.items():
                status = "Running" if grid.running else "Stopped"
                logger.info(f"Grid {grid_id} ({grid.symbol}): {status}")
        
        # Ensure monitor thread is running
        logger.info("Starting grid monitor thread")
        grid_manager._ensure_monitor_running()
        
        # Create templates directory and HTML files
        create_templates_directory()
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        return False
    
    logger.info("Application initialization complete")
    return True

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Grid Trading Bot (Web Interface)')
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
        logger.exception(f"Error running Grid Trading Bot: {e}")
        return 1
    finally:
        # Clean up
        if grid_manager:
            grid_manager.stop_all_grids()
            logger.info("Grid Trading Bot stopped")

if __name__ == "__main__":
    sys.exit(main())