"""
Grid Manager module for managing multiple grid strategies.
"""
import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional

from core.exchange import Exchange
from core.grid_strategy import GridStrategy, MarketIntelligenceEngine, AdaptiveParameterManager
from core.data_store import DataStore

class GridManager:
    def __init__(self, exchange: Exchange, data_store: DataStore):
        """
        Initialize the grid manager.
        
        Args:
            exchange: Exchange instance
            data_store: Data store instance
        """
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.data_store = data_store
        
        # Dictionary to store grid instances
        self.grids: Dict[str, GridStrategy] = {}
        
        # Thread for monitoring grids
        self.monitor_thread = None
        self.running = False
        
        # Load existing grids from data store
        self._load_grids()
    
    def _load_grids(self) -> None:
        """Load existing grids from data store with enhanced error handling."""
        try:
            self.logger.info("Loading saved grids from data store")
            grid_data = self.data_store.get_all_grids()
            if not grid_data:
                self.logger.info("No saved grids found")
                return
            self.logger.info(f"Found {len(grid_data)} saved grids")
            loaded_count = 0
            for grid_id, grid_config in grid_data.items():
                try:
                    # Make sure grid_number is an integer
                    grid_number = int(grid_config.get('grid_number', 0))
                    # Create grid instance
                    grid = GridStrategy(
                        exchange=self.exchange,
                        symbol=grid_config['symbol'],
                        price_lower=float(grid_config['price_lower']),
                        price_upper=float(grid_config['price_upper']),
                        grid_number=grid_number,
                        investment=float(grid_config['investment']),
                        take_profit_pnl=float(grid_config['take_profit_pnl']),
                        stop_loss_pnl=float(grid_config['stop_loss_pnl']),
                        grid_id=grid_id
                    )
                    # Load orders if available
                    orders = self.data_store.get_orders(grid_id)
                    if orders:
                        grid.grid_orders = orders
                    # Restore other state variables
                    grid.pnl = float(grid_config.get('pnl', 0))
                    grid.trades_count = int(grid_config.get('trades_count', 0))
                    grid.running = bool(grid_config.get('running', False))
                    # Add to grids dictionary
                    self.grids[grid_id] = grid
                    loaded_count += 1
                    self.logger.info(f"Loaded grid: {grid_id} ({grid_config['symbol']})")
                except Exception as e:
                    self.logger.error(f"Error loading grid {grid_id}: {e}")
            self.logger.info(f"Successfully loaded {loaded_count} of {len(grid_data)} grids")
        except Exception as e:
            self.logger.error(f"Error in _load_grids: {e}")
    
    def create_grid(self, 
                   symbol: str, 
                   price_lower: float, 
                   price_upper: float,
                   grid_number: int,
                   investment: float,
                   take_profit_pnl: float,
                   stop_loss_pnl: float,
                   leverage: float = 20.0,
                   enable_grid_adaptation: bool = True,
                    enable_samig: bool = True) -> str:
        """
        Create a new grid strategy.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            price_lower: Lower price boundary
            price_upper: Upper price boundary
            grid_number: Number of grid levels
            investment: Total investment amount
            take_profit_pnl: Take profit PnL percentage
            stop_loss_pnl: Stop loss PnL percentage
            leverage: Trading leverage (e.g., 1.0, 10.0, etc.)
            enable_grid_adaptation: Whether to adapt grid when price moves outside boundaries
            
        Returns:
            str: Grid ID
        """
        try:
            # Generate unique ID for this grid
            grid_id = str(uuid.uuid4())
            
            # Create grid instance
            grid = GridStrategy(
                exchange=self.exchange,
                symbol=symbol,
                price_lower=float(price_lower),
                price_upper=float(price_upper),
                grid_number=int(grid_number),
                investment=float(investment),
                take_profit_pnl=float(take_profit_pnl),
                stop_loss_pnl=float(stop_loss_pnl),
                grid_id=grid_id,
                leverage=float(leverage),
                enable_grid_adaptation=enable_grid_adaptation,
                 enable_samig=enable_samig  # Add this
            )
            
            # Add to grids dictionary
            self.grids[grid_id] = grid
            
            # Save grid configuration to data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            self.logger.info(f"Created grid: {grid_id} ({symbol}), adaptation: {'enabled' if enable_grid_adaptation else 'disabled'}")
            return grid_id
        except Exception as e:
            self.logger.error(f"Error creating grid: {e}")
            return ""
    
    def start_grid(self, grid_id: str) -> bool:
        """
        Start a grid strategy with enhanced error handling.
        
        Args:
            grid_id: Grid identifier
                
        Returns:
            bool: Success status
        """
        try:
            if grid_id not in self.grids:
                self.logger.error(f"Grid {grid_id} not found")
                return False
                
            grid = self.grids[grid_id]
            
            # Make sure the grid is not already running
            if grid.running:
                self.logger.info(f"Grid {grid_id} is already running")
                return True
                
            # Set up initial grid parameters in case they need validation
            grid.price_lower = float(grid.price_lower)
            grid.price_upper = float(grid.price_upper)
            grid.grid_number = int(grid.grid_number)
            grid.investment = float(grid.investment)
            grid.take_profit_pnl = float(grid.take_profit_pnl)
            grid.stop_loss_pnl = float(grid.stop_loss_pnl)
            grid.leverage = float(grid.leverage) if hasattr(grid, 'leverage') else 20.0
            
            # Setup grid orders
            self.logger.info(f"Starting grid {grid_id} with {grid.grid_number} levels")
            self.logger.info(f"Price range: {grid.price_lower} - {grid.price_upper}")
            self.logger.info(f"Investment: {grid.investment}, Leverage: {grid.leverage}x")
            
            # Call setup_grid to place the orders
            grid.setup_grid()
            
            # Check if grid was started successfully
            if not grid.running:
                self.logger.error(f"Grid {grid_id} setup failed to start the grid")
                return False
                
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            # Save initial orders
            self.data_store.save_orders(grid_id, grid.grid_orders)
            
            # Start monitor thread if not running
            self._ensure_monitor_running()
            
            self.logger.info(f"Started grid: {grid_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error starting grid {grid_id}: {e}")
            # Include traceback for better debugging
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def stop_grid(self, grid_id: str) -> bool:
        """
        Stop a grid strategy with enhanced persistence.
        
        Args:
            grid_id: Grid identifier
                
        Returns:
            bool: Success status
        """
        try:
            if grid_id not in self.grids:
                self.logger.error(f"Grid {grid_id} not found")
                return False
            grid = self.grids[grid_id]
            # Stop grid
            grid.stop_grid()
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            # Clear orders but keep a record
            self.data_store.save_orders(grid_id, {})
            self.logger.info(f"Stopped grid: {grid_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping grid {grid_id}: {e}")
            return False
    
    def delete_grid(self, grid_id: str) -> bool:
        """
        Delete a grid strategy.
        
        Args:
            grid_id: Grid identifier
            
        Returns:
            bool: Success status
        """
        try:
            if grid_id not in self.grids:
                self.logger.error(f"Grid {grid_id} not found")
                return False
            grid = self.grids[grid_id]
            # Make sure grid is stopped
            if grid.running:
                self.stop_grid(grid_id)
            # Remove from grids dictionary
            del self.grids[grid_id]
            # Remove from data store
            self.data_store.delete_grid(grid_id)
            self.data_store.delete_orders(grid_id)
            self.data_store.delete_positions(grid_id)
            self.logger.info(f"Deleted grid: {grid_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting grid {grid_id}: {e}")
            return False
    
    def get_grid_status(self, grid_id: str) -> Optional[Dict]:
        """Get the status of a grid strategy."""
        try:
            if grid_id not in self.grids:
                return None
            return self.grids[grid_id].get_status()
        except Exception as e:
            self.logger.error(f"Error getting grid status for {grid_id}: {e}")
            return None
    
    def get_all_grids_status(self) -> List[Dict]:
        """Get the status of all grid strategies."""
        try:
            return [grid.get_status() for grid in self.grids.values()]
        except Exception as e:
            self.logger.error(f"Error getting all grids status: {e}")
            return []
    
    def _ensure_monitor_running(self) -> None:
        """Ensure monitor thread is running."""
        try:
            if self.monitor_thread is None or not self.monitor_thread.is_alive():
                self.running = True
                self.monitor_thread = threading.Thread(target=self._monitor_grids)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
                self.logger.info("Grid monitor thread started")
        except Exception as e:
            self.logger.error(f"Error ensuring monitor running: {e}")
    
    def _monitor_grids(self) -> None:
        """Monitor and update all running grid strategies."""
        try:
            while self.running:
                for grid_id, grid in list(self.grids.items()):
                    if grid.running:
                        try:
                            # Update grid
                            grid.update_grid()
                            # Save updated grid status and orders
                            self.data_store.save_grid(grid_id, grid.get_status())
                            self.data_store.save_orders(grid_id, grid.grid_orders)
                        except Exception as e:
                            self.logger.error(f"Error updating grid {grid_id}: {e}")
                # Check every 10 seconds
                time.sleep(10)
        except Exception as e:
            self.logger.error(f"Error in monitor_grids: {e}")
    
    def stop_all_grids(self) -> None:
        """Stop all grid strategies and monitor thread."""
        try:
            self.logger.info("Stopping all grids")
            # Stop monitor thread
            self.running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(30)  # Wait up to 30 seconds
            # Stop all grids
            for grid_id in list(self.grids.keys()):
                self.stop_grid(grid_id)
            self.logger.info("All grids stopped")
        except Exception as e:
            self.logger.error(f"Error stopping all grids: {e}")
    
    def update_grid_config(self, 
                  grid_id: str, 
                  take_profit_pnl: Optional[float] = None,
                  stop_loss_pnl: Optional[float] = None,
                  enable_grid_adaptation: Optional[bool] = None,
                  enable_samig: Optional[bool] = None) -> bool:
        """
        Update grid configuration parameters.
        
        Note: Only parameters that can be changed while a grid is running.
        
        Args:
            grid_id: Grid identifier
            take_profit_pnl: New take profit PnL percentage (optional)
            stop_loss_pnl: New stop loss PnL percentage (optional)
            enable_grid_adaptation: Enable/disable grid adaptation (optional)
            enable_samig: Enable/disable SAMIG (optional)
            
        Returns:
            bool: Success status
        """
        try:
            if grid_id not in self.grids:
                self.logger.error(f"Grid {grid_id} not found")
                return False
                
            grid = self.grids[grid_id]
            
            # Update parameters if provided
            if take_profit_pnl is not None:
                grid.take_profit_pnl = float(take_profit_pnl)
                
            if stop_loss_pnl is not None:
                grid.stop_loss_pnl = float(stop_loss_pnl)
                
            if enable_grid_adaptation is not None:
                grid.enable_grid_adaptation = bool(enable_grid_adaptation)
                self.logger.info(f"Grid adaptation for {grid_id} set to: {'enabled' if enable_grid_adaptation else 'disabled'}")
            
            # Update SAMIG parameter
            if enable_samig is not None:
                grid.enable_samig = bool(enable_samig)
                self.logger.info(f"SAMIG for {grid_id} set to: {'enabled' if enable_samig else 'disabled'}")
                
                if enable_samig and not hasattr(grid, 'market_intelligence'):
                    # Initialize SAMIG components if enabling
                    from collections import deque
                    grid.market_intelligence = MarketIntelligenceEngine(grid.original_symbol)
                    grid.parameter_manager = AdaptiveParameterManager()
                    grid.performance_tracker = deque(maxlen=50)
                    grid.current_market_snapshot = None
                    grid.adaptation_count = 0
                    self.logger.info(f"SAMIG components initialized for grid {grid_id}")
            
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            self.logger.info(f"Updated grid {grid_id} configuration")
            return True
        except Exception as e:
            self.logger.error(f"Error updating grid config for {grid_id}: {e}")
            return False