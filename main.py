"""
Grid Trading Bot - Main Entry Point
"""
import os
import sys
import logging
import json
import argparse
from typing import Dict, List, Any

from core.exchange import Exchange
from core.data_store import DataStore
from core.grid_manager import GridManager
from ui.ui import GridBotUI

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

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Grid Trading Bot')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--testnet', action='store_true', help='Use testnet')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Grid Trading Bot")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        if not config:
            logger.error("Failed to load configuration")
            return 1
        
        # Check for required configuration
        if 'api_key' not in config or 'api_secret' not in config:
            logger.error("API key and secret not found in configuration")
            return 1
        
        # Initialize components
        exchange = Exchange(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            testnet=args.testnet or config.get('testnet', False)
        )
        
        data_store = DataStore(data_dir=config.get('data_dir', 'data'))
        
        grid_manager = GridManager(exchange, data_store)
        
        # Initialize UI
        ui = GridBotUI(grid_manager)
        
        # Run UI
        ui.run()
        
        # Clean up
        grid_manager.stop_all_grids()
        
        logger.info("Grid Trading Bot stopped")
        return 0
        
    except Exception as e:
        logger.exception(f"Error running Grid Trading Bot: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())