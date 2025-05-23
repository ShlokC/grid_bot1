"""
Grid Manager module for managing multiple grid strategies with STRICT LIMITS.
Fixed to prevent investment multiplication and grid count bugs.
"""
import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional

from core.exchange import Exchange
from core.grid_strategy import GridStrategy
from core.data_store import DataStore

class GridManager:
    def __init__(self, exchange: Exchange, data_store: DataStore):
        """
        Initialize the grid manager with enhanced limit enforcement.
        
        Args:
            exchange: Exchange instance
            data_store: Data store instance
        """
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.data_store = data_store
        
        # Dictionary to store grid instances
        self.grids: Dict[str, GridStrategy] = {}
        
        # CRITICAL: Global investment and grid tracking
        self.total_investment_across_all_grids = 0.0
        self.total_grids_created = 0
        self.max_concurrent_grids = 3  # Reasonable limit for most users
        self.max_total_investment = 1000.0  # Default safety limit
        
        # Thread for monitoring grids
        self.monitor_thread = None
        self.running = False
        
        # Load existing grids from data store
        self._load_grids()
        
        self.logger.info(f"GridManager initialized with limits:")
        self.logger.info(f"  Max concurrent grids: {self.max_concurrent_grids}")
        self.logger.info(f"  Max total investment: ${self.max_total_investment:.2f}")
    
    def _load_grids(self) -> None:
        """Load existing grids from data store with enhanced validation."""
        try:
            self.logger.info("Loading saved grids from data store")
            grid_data = self.data_store.get_all_grids()
            if not grid_data:
                self.logger.info("No saved grids found")
                return
            
            self.logger.info(f"Found {len(grid_data)} saved grids")
            loaded_count = 0
            total_loaded_investment = 0.0
            
            for grid_id, grid_config in grid_data.items():
                try:
                    # Validate grid configuration before loading
                    if not self._validate_grid_config_for_loading(grid_config):
                        self.logger.warning(f"Skipping invalid grid config: {grid_id}")
                        continue
                    
                    # Extract and validate parameters
                    grid_number = int(grid_config.get('grid_number', 0))
                    investment = float(grid_config.get('investment', 0))
                    
                    if grid_number <= 0 or investment <= 0:
                        self.logger.warning(f"Skipping grid {grid_id} with invalid parameters")
                        continue
                    
                    # Create grid instance
                    grid = GridStrategy(
                        exchange=self.exchange,
                        symbol=grid_config['symbol'],
                        price_lower=float(grid_config['price_lower']),
                        price_upper=float(grid_config['price_upper']),
                        grid_number=grid_number,
                        investment=investment,
                        take_profit_pnl=float(grid_config['take_profit_pnl']),
                        stop_loss_pnl=float(grid_config['stop_loss_pnl']),
                        grid_id=grid_id,
                        leverage=float(grid_config.get('leverage', 20.0)),
                        enable_grid_adaptation=bool(grid_config.get('enable_grid_adaptation', True)),
                        enable_samig=bool(grid_config.get('enable_samig', False))
                    )
                    
                    # Restore state variables
                    grid.total_pnl = float(grid_config.get('pnl', 0))
                    grid.total_trades = int(grid_config.get('trades_count', 0))
                    grid.running = bool(grid_config.get('running', False))
                    
                    # Add to grids dictionary
                    self.grids[grid_id] = grid
                    loaded_count += 1
                    total_loaded_investment += investment
                    
                    self.logger.info(f"Loaded grid: {grid_id} ({grid_config['symbol']}) - ${investment:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading grid {grid_id}: {e}")
            
            # Update global tracking
            self.total_investment_across_all_grids = total_loaded_investment
            self.total_grids_created = loaded_count
            
            self.logger.info(f"Successfully loaded {loaded_count} of {len(grid_data)} grids")
            self.logger.info(f"Total investment across all grids: ${self.total_investment_across_all_grids:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in _load_grids: {e}")
    
    def _validate_grid_config_for_loading(self, grid_config: Dict) -> bool:
        """Validate grid configuration before loading."""
        try:
            required_fields = ['symbol', 'price_lower', 'price_upper', 'grid_number', 'investment']
            
            for field in required_fields:
                if field not in grid_config:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate numeric values
            if float(grid_config['price_lower']) >= float(grid_config['price_upper']):
                self.logger.error("Invalid price range")
                return False
            
            if int(grid_config['grid_number']) <= 0:
                self.logger.error("Invalid grid number")
                return False
            
            if float(grid_config['investment']) <= 0:
                self.logger.error("Invalid investment amount")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating grid config: {e}")
            return False
    
    def _validate_new_grid_limits(self, investment: float) -> bool:
        """Validate that creating a new grid won't exceed global limits."""
        try:
            # Check concurrent grid limit
            active_grids = sum(1 for grid in self.grids.values() if grid.running)
            if active_grids >= self.max_concurrent_grids:
                self.logger.error(f"Cannot create grid: active grids limit reached ({active_grids}/{self.max_concurrent_grids})")
                return False
            
            # Check total investment limit
            if self.total_investment_across_all_grids + investment > self.max_total_investment:
                remaining = self.max_total_investment - self.total_investment_across_all_grids
                self.logger.error(f"Cannot create grid: would exceed total investment limit")
                self.logger.error(f"Current: ${self.total_investment_across_all_grids:.2f}, "
                                f"Requested: ${investment:.2f}, "
                                f"Limit: ${self.max_total_investment:.2f}, "
                                f"Remaining: ${remaining:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating new grid limits: {e}")
            return False
    
    def _validate_grid_parameters(self, **kwargs) -> bool:
        """Validate grid parameters before creation."""
        try:
            # Extract parameters
            symbol = kwargs.get('symbol', '')
            price_lower = float(kwargs.get('price_lower', 0))
            price_upper = float(kwargs.get('price_upper', 0))
            grid_number = int(kwargs.get('grid_number', 0))
            investment = float(kwargs.get('investment', 0))
            leverage = float(kwargs.get('leverage', 1.0))
            
            # Validate symbol
            if not symbol or len(symbol.strip()) == 0:
                self.logger.error("Symbol cannot be empty")
                return False
            
            # Validate price range
            if price_lower <= 0 or price_upper <= 0:
                self.logger.error("Prices must be positive")
                return False
            
            if price_lower >= price_upper:
                self.logger.error(f"Lower price ({price_lower}) must be less than upper price ({price_upper})")
                return False
            
            # Validate grid number
            if grid_number <= 0:
                self.logger.error(f"Grid number ({grid_number}) must be positive")
                return False
            
            if grid_number > 50:  # Reasonable maximum
                self.logger.error(f"Grid number ({grid_number}) exceeds maximum of 50")
                return False
            
            # Validate investment
            if investment <= 0:
                self.logger.error(f"Investment ({investment}) must be positive")
                return False
            
            if investment > 10000:  # Reasonable maximum for safety
                self.logger.error(f"Investment ({investment}) exceeds safety maximum of $10,000")
                return False
            
            # Validate leverage
            if leverage < 1 or leverage > 125:
                self.logger.error(f"Leverage ({leverage}) must be between 1x and 125x")
                return False
            
            # Check for duplicate symbols (prevent multiple grids on same symbol)
            for existing_grid in self.grids.values():
                if existing_grid.original_symbol.upper() == symbol.upper() and existing_grid.running:
                    self.logger.error(f"Active grid already exists for symbol {symbol}")
                    return False
            
            return True
            
        except ValueError as e:
            self.logger.error(f"Invalid parameter type: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating grid parameters: {e}")
            return False
    
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
                   enable_samig: bool = False) -> str:
        """
        Create a new grid strategy with STRICT validation and limits.
        
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
            enable_samig: Enable SAMIG intelligent adaptation
            
        Returns:
            str: Grid ID if successful, empty string if failed
        """
        try:
            # CRITICAL: Validate all parameters first
            if not self._validate_grid_parameters(
                symbol=symbol,
                price_lower=price_lower,
                price_upper=price_upper,
                grid_number=grid_number,
                investment=investment,
                leverage=leverage
            ):
                self.logger.error("Grid creation failed: invalid parameters")
                return ""
            
            # CRITICAL: Validate global limits
            if not self._validate_new_grid_limits(investment):
                self.logger.error("Grid creation failed: would exceed global limits")
                return ""
            
            # Generate unique ID for this grid
            grid_id = str(uuid.uuid4())
            
            self.logger.info(f"Creating grid with VALIDATED parameters:")
            self.logger.info(f"  Symbol: {symbol}")
            self.logger.info(f"  Price Range: ${price_lower:.6f} - ${price_upper:.6f}")
            self.logger.info(f"  Grid Count: {grid_number} (VALIDATED)")
            self.logger.info(f"  Investment: ${investment:.2f} (VALIDATED)")
            self.logger.info(f"  Leverage: {leverage}x")
            self.logger.info(f"  Grid Adaptation: {'enabled' if enable_grid_adaptation else 'disabled'}")
            self.logger.info(f"  SAMIG: {'enabled' if enable_samig else 'disabled'}")
            
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
                enable_samig=enable_samig
            )
            
            # Add to grids dictionary
            self.grids[grid_id] = grid
            
            # Update global tracking
            self.total_investment_across_all_grids += investment
            self.total_grids_created += 1
            
            # Save grid configuration to data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            self.logger.info(f"Grid created successfully: {grid_id}")
            self.logger.info(f"Global stats - Total grids: {len(self.grids)}, "
                           f"Total investment: ${self.total_investment_across_all_grids:.2f}")
            
            return grid_id
            
        except Exception as e:
            self.logger.error(f"Error creating grid: {e}")
            return ""
    
    def start_grid(self, grid_id: str) -> bool:
        """
        Start a grid strategy with enhanced validation and safety checks.
        
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
            
            # Additional safety check - verify grid limits before starting
            if not self._validate_grid_before_start(grid):
                self.logger.error(f"Grid {grid_id} failed pre-start validation")
                return False
            
            self.logger.info(f"Starting grid {grid_id} with VALIDATED parameters:")
            self.logger.info(f"  Symbol: {grid.original_symbol}")
            self.logger.info(f"  Grid count: {grid.user_grid_number} (STRICT LIMIT)")
            self.logger.info(f"  Investment: ${grid.user_total_investment:.2f} (STRICT LIMIT)")
            self.logger.info(f"  Price range: ${grid.user_price_lower:.6f} - ${grid.user_price_upper:.6f}")
            self.logger.info(f"  Leverage: {grid.user_leverage}x")
            
            # Call setup_grid to place the orders
            grid.setup_grid()
            
            # Check if grid was started successfully
            if not grid.running:
                self.logger.error(f"Grid {grid_id} setup failed to start the grid")
                return False
                
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            # Start monitor thread if not running
            self._ensure_monitor_running()
            
            self.logger.info(f"Grid started successfully: {grid_id}")
            self._log_global_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting grid {grid_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _validate_grid_before_start(self, grid: GridStrategy) -> bool:
        """Validate grid configuration before starting."""
        try:
            # Check if exchange connection is working
            try:
                ticker = self.exchange.get_ticker(grid.symbol)
                current_price = float(ticker['last'])
                
                # Verify price is within reasonable range of grid boundaries
                if current_price < grid.user_price_lower * 0.5 or current_price > grid.user_price_upper * 2.0:
                    self.logger.warning(f"Current price ${current_price:.6f} is very far from grid range "
                                      f"${grid.user_price_lower:.6f} - ${grid.user_price_upper:.6f}")
                    # Don't fail, just warn
                    
            except Exception as e:
                self.logger.error(f"Cannot fetch current price for {grid.symbol}: {e}")
                return False
            
            # Verify grid parameters are still valid
            if grid.user_grid_number <= 0 or grid.user_total_investment <= 0:
                self.logger.error(f"Invalid grid parameters: grids={grid.user_grid_number}, "
                                f"investment=${grid.user_total_investment:.2f}")
                return False
            
            # Check account balance (if possible)
            try:
                balance = self.exchange.get_balance()
                # Basic balance check - implementation depends on exchange structure
                self.logger.debug(f"Account balance check completed")
            except Exception as e:
                self.logger.warning(f"Could not verify account balance: {e}")
                # Don't fail on balance check as it might not be critical
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating grid before start: {e}")
            return False
    
    def _log_global_status(self):
        """Log current global status."""
        try:
            active_grids = sum(1 for grid in self.grids.values() if grid.running)
            total_grids = len(self.grids)
            
            self.logger.info(f"Global GridManager Status:")
            self.logger.info(f"  Active grids: {active_grids}/{total_grids}")
            self.logger.info(f"  Total investment: ${self.total_investment_across_all_grids:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error logging global status: {e}")
    
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
            
            self.logger.info(f"Stopped grid: {grid_id}")
            self._log_global_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping grid {grid_id}: {e}")
            return False
    
    def delete_grid(self, grid_id: str) -> bool:
        """
        Delete a grid strategy with proper cleanup.
        
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
            
            # Get investment amount before deletion for tracking
            grid_investment = grid.user_total_investment
            
            # Make sure grid is stopped
            if grid.running:
                self.stop_grid(grid_id)
            
            # Remove from grids dictionary
            del self.grids[grid_id]
            
            # Update global tracking
            self.total_investment_across_all_grids -= grid_investment
            self.total_investment_across_all_grids = max(0, self.total_investment_across_all_grids)  # Prevent negative
            
            # Remove from data store
            self.data_store.delete_grid(grid_id)
            self.data_store.delete_orders(grid_id)
            self.data_store.delete_positions(grid_id)
            
            self.logger.info(f"Deleted grid: {grid_id}")
            self.logger.info(f"Released investment: ${grid_investment:.2f}")
            self._log_global_status()
            
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
        """Get the status of all grid strategies with global summary."""
        try:
            grid_statuses = [grid.get_status() for grid in self.grids.values()]
            
            # Add global summary to each status (for debugging)
            for status in grid_statuses:
                status['global_investment_used'] = self.total_investment_across_all_grids
                status['global_grid_count'] = len(self.grids)
                status['global_active_count'] = sum(1 for grid in self.grids.values() if grid.running)
            
            return grid_statuses
            
        except Exception as e:
            self.logger.error(f"Error getting all grids status: {e}")
            return []
    
    def get_global_status(self) -> Dict:
        """Get global GridManager status."""
        try:
            active_grids = sum(1 for grid in self.grids.values() if grid.running)
            total_investment = sum(grid.user_total_investment for grid in self.grids.values())
            
            return {
                'total_grids': len(self.grids),
                'active_grids': active_grids,
                'total_investment': total_investment,
                'max_concurrent_grids': self.max_concurrent_grids,
                'max_total_investment': self.max_total_investment,
                'remaining_investment_capacity': self.max_total_investment - total_investment
            }
            
        except Exception as e:
            self.logger.error(f"Error getting global status: {e}")
            return {}
    
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
                active_count = 0
                
                for grid_id, grid in list(self.grids.items()):
                    if grid.running:
                        active_count += 1
                        try:
                            # Update grid
                            grid.update_grid()
                            # Save updated grid status
                            self.data_store.save_grid(grid_id, grid.get_status())
                        except Exception as e:
                            self.logger.error(f"Error updating grid {grid_id}: {e}")
                
                # Log periodic status
                if active_count > 0:
                    self.logger.debug(f"Monitor cycle complete: {active_count} active grids")
                
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
            self._log_global_status()
            
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
            
            # Validate parameters before updating
            if take_profit_pnl is not None:
                if take_profit_pnl <= 0 or take_profit_pnl > 100:
                    self.logger.error(f"Invalid take profit: {take_profit_pnl}%")
                    return False
                grid.take_profit_pnl = float(take_profit_pnl)
                
            if stop_loss_pnl is not None:
                if stop_loss_pnl <= 0 or stop_loss_pnl > 50:
                    self.logger.error(f"Invalid stop loss: {stop_loss_pnl}%")
                    return False
                grid.stop_loss_pnl = float(stop_loss_pnl)
                
            if enable_grid_adaptation is not None:
                grid.enable_grid_adaptation = bool(enable_grid_adaptation)
                self.logger.info(f"Grid adaptation for {grid_id} set to: {'enabled' if enable_grid_adaptation else 'disabled'}")
            
            if enable_samig is not None:
                grid.enable_samig = bool(enable_samig)
                self.logger.info(f"SAMIG for {grid_id} set to: {'enabled' if enable_samig else 'disabled'}")
                
                if enable_samig and not hasattr(grid, 'market_intel'):
                    from core.grid_strategy import MarketIntelligence
                    grid.market_intel = MarketIntelligence(grid.original_symbol)
                    self.logger.info(f"SAMIG components initialized for grid {grid_id}")
            
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            self.logger.info(f"Updated grid {grid_id} configuration")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating grid config for {grid_id}: {e}")
            return False