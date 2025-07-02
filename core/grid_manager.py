"""
Simplified Grid Manager for managing multiple grid strategies.
No hedge mode, no complex investment tracking, just simple grid management.
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
        """Initialize the simplified grid manager."""
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.data_store = data_store
        
        # Dictionary to store grid instances
        self.grids: Dict[str, GridStrategy] = {}
        
        # Simple limits
        self.max_concurrent_grids = 10
        self.max_total_investment = 10000.0
        
        # Thread for monitoring grids
        self.monitor_thread = None
        self.running = False
        # ENHANCEMENT: Add active symbols tracking
        self.active_symbols_data = {
            'top_active': [],
            'gainers': [],
            'losers': [],
            'last_updated': 0
        }
        self.active_symbols_update_interval = 30  # Update every 30 seconds
        
            # Load existing grids from data store
        self._load_grids()
        
        # FIXED: Start monitor if any loaded grids are running
        running_grids = sum(1 for grid in self.grids.values() if grid.running)
        if running_grids > 0:
            self.logger.info(f"Found {running_grids} running grids, starting monitor thread")
            self._ensure_monitor_running()
        
        self.logger.info(f"Grid Manager initialized: {len(self.grids)} grids loaded, "
                        f"Max grids: {self.max_concurrent_grids}, Max investment: ${self.max_total_investment:.2f}")
        
    def _load_grids(self) -> None:
        """Load existing grids from data store."""
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
                    # Validate grid configuration
                    if not self._validate_grid_config_for_loading(grid_config):
                        self.logger.warning(f"Skipping invalid grid config: {grid_id}")
                        continue
                    
                    # Create grid instance
                    grid = GridStrategy(
                        exchange=self.exchange,
                        symbol=grid_config['symbol'],
                        grid_number=int(grid_config['grid_number']),
                        investment=float(grid_config['investment']),
                        take_profit_pnl=float(grid_config['take_profit_pnl']),
                        stop_loss_pnl=float(grid_config['stop_loss_pnl']),
                        grid_id=grid_id,
                        leverage=float(grid_config.get('leverage', 10.0)),
                        enable_grid_adaptation=bool(grid_config.get('enable_grid_adaptation', True))
                    )
                    
                    # Restore state
                    grid.total_pnl = float(grid_config.get('pnl', 0))
                    grid.total_trades = int(grid_config.get('trades_count', 0))
                    grid.running = bool(grid_config.get('running', False))
                    
                    # Add to grids dictionary
                    self.grids[grid_id] = grid
                    loaded_count += 1
                    
                    self.logger.info(f"Loaded grid: {grid_id} ({grid_config['symbol']}) - ${grid_config['investment']:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading grid {grid_id}: {e}")
            
            self.logger.info(f"Successfully loaded {loaded_count} of {len(grid_data)} grids")
            
        except Exception as e:
            self.logger.error(f"Error in _load_grids: {e}")
    
    def _validate_grid_config_for_loading(self, grid_config: Dict) -> bool:
        """Validate grid configuration before loading."""
        try:
            required_fields = ['symbol', 'grid_number', 'investment']
            
            for field in required_fields:
                if field not in grid_config:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate numeric values
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
        """Validate that creating a new grid won't exceed limits."""
        try:
            # Check concurrent grid limit
            if len(self.grids) >= self.max_concurrent_grids:
                self.logger.error(f"Cannot create grid: grid limit reached ({len(self.grids)}/{self.max_concurrent_grids})")
                return False
            
            # Check total investment limit
            current_investment = sum(grid.user_total_investment for grid in self.grids.values())
            if current_investment + investment > self.max_total_investment:
                remaining = self.max_total_investment - current_investment
                self.logger.error(f"Cannot create grid: would exceed total investment limit")
                self.logger.error(f"Current: ${current_investment:.2f}, Requested: ${investment:.2f}, "
                                f"Limit: ${self.max_total_investment:.2f}, Remaining: ${remaining:.2f}")
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
            grid_number = int(kwargs.get('grid_number', 0))
            investment = float(kwargs.get('investment', 0))
            leverage = float(kwargs.get('leverage', 1.0))
            
            # Validate symbol
            if not symbol or len(symbol.strip()) == 0:
                self.logger.error("Symbol cannot be empty")
                return False
            
            # Validate grid number
            if grid_number <= 0:
                self.logger.error(f"Grid number ({grid_number}) must be positive")
                return False
            
            if grid_number > 20:  # Reasonable maximum for simple grid
                self.logger.error(f"Grid number ({grid_number}) exceeds maximum of 20")
                return False
            
            # Validate investment
            if investment <= 0:
                self.logger.error(f"Investment ({investment}) must be positive")
                return False
            
            if investment > 5000:  # Reasonable maximum for safety
                self.logger.error(f"Investment ({investment}) exceeds safety maximum of $5,000")
                return False
            
            # Validate leverage
            if leverage < 1 or leverage > 125:
                self.logger.error(f"Leverage ({leverage}) must be between 1x and 125x")
                return False
            
            # Check for duplicate symbols
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
                   grid_number: int,
                   investment: float,
                   take_profit_pnl: float,
                   stop_loss_pnl: float,
                   leverage: float = 10.0,
                   enable_grid_adaptation: bool = True,
                   enable_samig: bool = False) -> str:
        """Create a new grid strategy."""
        try:
            # Validate all parameters
            if not self._validate_grid_parameters(
                symbol=symbol,
                grid_number=grid_number,
                investment=investment,
                leverage=leverage
            ):
                self.logger.error("Grid creation failed: invalid parameters")
                return ""
            
            # Validate limits
            if not self._validate_new_grid_limits(investment):
                self.logger.error("Grid creation failed: would exceed limits")
                return ""
            
            # Generate unique ID
            grid_id = str(uuid.uuid4())
            
            self.logger.info(f"Creating simplified grid:")
            self.logger.info(f"  Symbol: {symbol}")
            self.logger.info(f"  Grid Count: {grid_number}")
            self.logger.info(f"  Investment: ${investment:.2f}")
            self.logger.info(f"  Leverage: {leverage}x")
            
            # Create grid instance
            grid = GridStrategy(
                exchange=self.exchange,
                symbol=symbol,
                grid_number=int(grid_number),
                investment=float(investment),
                take_profit_pnl=float(take_profit_pnl),
                stop_loss_pnl=float(stop_loss_pnl),
                grid_id=grid_id,
                leverage=float(leverage),
                enable_grid_adaptation=enable_grid_adaptation
            )
            
            # Add to grids dictionary
            self.grids[grid_id] = grid
            
            # Save grid configuration to data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            self.logger.info(f"Grid created successfully: {grid_id}")
            
            return grid_id
            
        except Exception as e:
            self.logger.error(f"Error creating grid: {e}")
            return ""
    
    def start_grid(self, grid_id: str) -> bool:
        """Start a grid strategy."""
        try:
            if grid_id not in self.grids:
                self.logger.error(f"Grid {grid_id} not found")
                return False
                
            grid = self.grids[grid_id]
            
            if grid.running:
                self.logger.info(f"Grid {grid_id[:8]} is already running")
                return True

            self.logger.info(f"🚀 STARTING GRID: {grid_id[:8]}")
            
            # Setup the grid
            if grid.setup_grid():
                # Update grid status in data store
                self.data_store.save_grid(grid_id, grid.get_status())
                
                # Start monitor thread if not running
                self._ensure_monitor_running()
                
                self.logger.info(f"✅ Grid started successfully: {grid_id[:8]}")
                return True
            else:
                self.logger.error(f"❌ Grid setup failed: {grid_id[:8]}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error starting grid {grid_id}: {e}")
            return False
    
    def stop_grid(self, grid_id: str) -> bool:
        """Stop a grid strategy."""
        try:
            if grid_id not in self.grids:
                self.logger.error(f"Grid {grid_id} not found")
                return False
                
            grid = self.grids[grid_id]
            
            self.logger.info(f"🛑 STOPPING GRID: {grid_id[:8]}")
            
            # Stop grid
            grid.stop_grid()
            
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            self.logger.info(f"✅ Grid stopped successfully: {grid_id[:8]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping grid {grid_id}: {e}")
            return False
    
    def delete_grid(self, grid_id: str) -> bool:
        """Delete a grid strategy."""
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
        """Get status of all grids and trigger updates for running grids."""
        try:
            # FIXED: Ensure monitor is running before getting status
            running_grids = sum(1 for grid in self.grids.values() if grid.running)
            if running_grids > 0:
                self._ensure_monitor_running()
            
            grid_statuses = []
            
            for grid in self.grids.values():
                try:
                    # FIXED: Update running grids before getting status
                    if grid.running:
                        grid.update_grid()
                        # Save updated status
                        self.data_store.save_grid(grid.grid_id, grid.get_status())
                    
                    status = grid.get_status()
                    grid_statuses.append(status)
                    
                except Exception as e:
                    self.logger.error(f"Error getting status for grid {grid.grid_id}: {e}")
                    # Add error status
                    grid_statuses.append({
                        'grid_id': grid.grid_id,
                        'symbol': getattr(grid, 'original_symbol', 'unknown'),
                        'error': str(e)
                    })
            
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
        """FIXED: Ensure monitor thread is running with better checking and logging."""
        try:
            # Check if thread exists and is alive
            thread_running = (self.monitor_thread is not None and 
                            self.monitor_thread.is_alive())
            
            self.logger.debug(f"Monitor check: thread_running={thread_running}, self.running={self.running}")
            
            if not thread_running:
                # Stop any existing thread
                if self.running:
                    self.logger.info("🔍 Stopping existing monitor thread...")
                    self.running = False
                    if self.monitor_thread and self.monitor_thread.is_alive():
                        self.monitor_thread.join(timeout=2)
                
                # Start new thread
                self.logger.info("🔍 STARTING NEW MONITOR THREAD...")
                self.running = True
                self.monitor_thread = threading.Thread(target=self._monitor_grids, daemon=True)
                self.monitor_thread.start()
                
                # Verify thread started
                if self.monitor_thread.is_alive():
                    self.logger.info("✅ Grid monitor thread STARTED successfully")
                else:
                    self.logger.error("❌ Grid monitor thread FAILED to start")
            else:
                self.logger.debug("🔍 Grid monitor thread already running")
                
        except Exception as e:
            self.logger.error(f"❌ Error ensuring monitor running: {e}")
    
    def _monitor_grids(self) -> None:
        """Enhanced monitor grids and update active symbols periodically."""
        self.logger.info("🔍 ===== ENHANCED GRID MONITOR THREAD STARTED =====")
        
        cycle_count = 0
        try:
            while self.running:
                cycle_count += 1
                monitor_start = time.time()
                
                # Get active grids
                active_grids = [(grid_id, grid) for grid_id, grid in self.grids.items() if grid.running]
                
                # Log monitor activity every 10 cycles (50 seconds)
                if cycle_count % 10 == 1:
                    self.logger.info(f"🔍 Monitor cycle #{cycle_count}: {len(active_grids)} active grids")
                
                # Update grids
                if active_grids:
                    for grid_id, grid in active_grids:
                        try:
                            grid.update_grid()
                            # Save updated status
                            self.data_store.save_grid(grid_id, grid.get_status())
                        except Exception as e:
                            self.logger.error(f"❌ Error updating grid {grid_id[:8]}: {e}")
                else:
                    if cycle_count % 20 == 1:  # Log less frequently when no active grids
                        self.logger.info(f"🔍 Monitor cycle #{cycle_count}: No active grids")
                
                # ENHANCEMENT: Update active symbols data periodically
                try:
                    self._update_active_symbols_if_needed()
                except Exception as e:
                    self.logger.error(f"❌ Error updating active symbols: {e}")
                
                monitor_duration = time.time() - monitor_start
                
                # Log slow monitor cycles
                if monitor_duration > 2.0:
                    self.logger.warning(f"⚠️ Slow monitor cycle: {monitor_duration:.2f}s")
                
                # Wait before next update
                time.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            self.logger.error(f"❌ Critical error in monitor_grids: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.logger.warning("🔍 ===== ENHANCED GRID MONITOR THREAD STOPPED =====")
    
    def stop_all_grids(self) -> None:
        """Stop all grid strategies."""
        try:
            self.logger.info("Stopping all grids")
            
            # Stop monitor thread
            self.running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(10)  # Wait up to 10 seconds
            
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
        """Update grid configuration parameters."""
        try:
            if grid_id not in self.grids:
                self.logger.error(f"Grid {grid_id} not found")
                return False
                
            grid = self.grids[grid_id]
            
            # Update parameters
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
            
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            self.logger.info(f"Updated grid {grid_id} configuration")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating grid config for {grid_id}: {e}")
            return False
    def _update_active_symbols_if_needed(self) -> None:
        """OPTIMIZED: Update active symbols data using single bulk ticker API call instead of 435 individual calls."""
        try:
            current_time = time.time()
            last_update = self.active_symbols_data.get('last_updated', 0)
            
            # Only update if enough time has passed
            if current_time - last_update < self.active_symbols_update_interval:
                return
            
            self.logger.info("📊 Updating active symbols data (24hr bulk ticker - OPTIMIZED)...")
            
            # OPTIMIZATION: These now use single API call instead of 435 individual calls
            top_active = self.exchange.get_top_active_symbols(limit=10)
            gainers_losers = self.exchange.get_top_gainers_losers(limit=5)
            
            # Update stored data
            self.active_symbols_data = {
                'top_active': top_active,
                'gainers': gainers_losers.get('gainers', []),
                'losers': gainers_losers.get('losers', []),
                'last_updated': current_time,
                'timeframe': '24hr'  # Updated to reflect actual data
            }
            
            self.logger.info(f"📊 Active symbols updated (24hr bulk data): {len(top_active)} active, "
                            f"{len(gainers_losers.get('gainers', []))} gainers, "
                            f"{len(gainers_losers.get('losers', []))} losers")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating active symbols data: {e}")

    def get_active_symbols_data(self) -> Dict:
        """Get the current active symbols data with correct timeframe info."""
        try:
            # Return a copy to prevent external modification
            return {
                'top_active': self.active_symbols_data.get('top_active', []).copy(),
                'gainers': self.active_symbols_data.get('gainers', []).copy(),
                'losers': self.active_symbols_data.get('losers', []).copy(),
                'last_updated': self.active_symbols_data.get('last_updated', 0),
                'timeframe': self.active_symbols_data.get('timeframe', '24hr'),  # Updated default
                'last_updated_formatted': time.strftime('%H:%M:%S', 
                    time.localtime(self.active_symbols_data.get('last_updated', 0)))
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting active symbols data: {e}")
            return {
                'top_active': [],
                'gainers': [],
                'losers': [],
                'last_updated': 0,
                'timeframe': '24hr',
                'last_updated_formatted': 'Never'
            }

    def force_update_active_symbols(self) -> bool:
        """Force immediate update of active symbols data."""
        try:
            self.logger.info("🔄 Force updating active symbols...")
            
            # Reset last update time to force refresh
            self.active_symbols_data['last_updated'] = 0
            
            # Trigger update
            self._update_active_symbols_if_needed()
            
            return len(self.active_symbols_data.get('top_active', [])) > 0
            
        except Exception as e:
            self.logger.error(f"❌ Error force updating active symbols: {e}")
            return False