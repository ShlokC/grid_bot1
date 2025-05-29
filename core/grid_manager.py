"""
Grid Manager module for managing multiple grid strategies with STRICT LIMITS.
Fixed to prevent investment multiplication and grid count bugs.
"""
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
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
        """Initialize the grid manager with parallel execution support."""
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.data_store = data_store
        
        # Dictionary to store grid instances
        self.grids: Dict[str, GridStrategy] = {}
        
        # ADDED: Thread pool for parallel grid processing
        self.thread_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="GridWorker")
        
        # CRITICAL: Global investment and grid tracking
        self.total_investment_across_all_grids = 0.0
        self.total_grids_created = 0
        self.max_concurrent_grids = 10  # INCREASED for multi-symbol
        self.max_total_investment = 5000.0  # INCREASED for multi-symbol
        self.monitor_cycle_count = 0
        self.last_monitor_summary = 0
        
        # Thread for monitoring grids
        self.monitor_thread = None
        self.running = False
        
        # Load existing grids from data store
        self._load_grids()
        self._log_manager_initialization()
        
    def _log_manager_initialization(self):
            """Enhanced manager initialization logging"""
            self.logger.info("=" * 100)
            self.logger.info("🏭 GRID MANAGER INITIALIZED")
            self.logger.info("=" * 100)
            self.logger.info(f"📊 LIMITS & CONFIGURATION:")
            self.logger.info(f"   Max Concurrent Grids: {self.max_concurrent_grids}")
            self.logger.info(f"   Max Total Investment: ${self.max_total_investment:.2f}")
            self.logger.info(f"   Current Grids Loaded: {len(self.grids)}")
            self.logger.info(f"   Total Investment Used: ${self.total_investment_across_all_grids:.2f}")
            
            if self.grids:
                self.logger.info(f"🎯 LOADED GRIDS:")
                for grid_id, grid in self.grids.items():
                    status = "🟢 RUNNING" if grid.running else "🔴 STOPPED"
                    self.logger.info(f"   {grid_id[:8]} ({grid.original_symbol}): ${grid.user_total_investment:.2f} [{status}]")
            
            self.logger.info("=" * 100)
    
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
        """Create a new grid strategy with symbol duplication check."""
        try:
            # ADDED: Check for duplicate symbols
            for existing_grid in self.grids.values():
                if existing_grid.original_symbol.upper() == symbol.upper() and existing_grid.running:
                    self.logger.error(f"Active grid already exists for symbol {symbol}")
                    return ""
            
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
        """Enhanced grid starting with comprehensive validation and logging"""
        try:
            if grid_id not in self.grids:
                self.logger.error(f"❌ Grid {grid_id} not found")
                return False
                
            grid = self.grids[grid_id]
            
            if grid.running:
                self.logger.info(f"ℹ️ Grid {grid_id[:8]} is already running")
                return True

            self.logger.info("=" * 80)
            self.logger.info(f"🚀 STARTING GRID: {grid_id[:8]}")
            self.logger.info("=" * 80)
            
            # ENHANCED: Pre-start validation with detailed logging
            if not self._validate_grid_before_start(grid):
                self.logger.error(f"❌ Grid {grid_id[:8]} failed pre-start validation")
                return False
            
            # ENHANCED: Check exchange connectivity
            try:
                ticker = self.exchange.get_ticker(grid.symbol)
                current_price = float(ticker['last'])
                
                self.logger.info(f"📊 PRE-START MARKET CHECK:")
                self.logger.info(f"   Symbol: {grid.original_symbol} ({grid.symbol})")
                self.logger.info(f"   Current Price: ${current_price:.6f}")
                self.logger.info(f"   Grid Range: ${grid.user_price_lower:.6f} - ${grid.user_price_upper:.6f}")
                
                if current_price < grid.user_price_lower or current_price > grid.user_price_upper:
                    self.logger.warning(f"⚠️ Current price is outside grid range!")
                    self.logger.warning(f"   Price may need to move into range for optimal performance")
                else:
                    self.logger.info(f"✅ Current price is within grid range")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to get market data for {grid.symbol}: {e}")
                return False
            
            # ENHANCED: Account balance check
            try:
                balance = self.exchange.get_balance()
                if 'USDT' in balance.get('free', {}):
                    available_balance = float(balance['free']['USDT'])
                    self.logger.info(f"💰 Account Balance Check:")
                    self.logger.info(f"   Available USDT: ${available_balance:.2f}")
                    self.logger.info(f"   Required Investment: ${grid.user_total_investment:.2f}")
                    
                    if available_balance < grid.user_total_investment:
                        self.logger.warning(f"⚠️ Insufficient balance for full investment")
                        self.logger.warning(f"   Consider reducing investment amount")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not verify account balance: {e}")
            
            self.logger.info(f"🎯 STARTING GRID SETUP...")
            
            # Call setup_grid to place the orders
            grid.setup_grid()
            
            # Check if grid was started successfully
            if not grid.running:
                self.logger.error(f"❌ Grid {grid_id[:8]} setup failed to start the grid")
                return False
                
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            # Start monitor thread if not running
            self._ensure_monitor_running()
            
            self.logger.info(f"✅ GRID STARTED SUCCESSFULLY: {grid_id[:8]}")
            self._log_global_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting grid {grid_id}: {e}")
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
        """Enhanced global status logging"""
        try:
            active_grids = sum(1 for grid in self.grids.values() if grid.running)
            total_grids = len(self.grids)
            total_pnl = sum(grid.total_pnl for grid in self.grids.values())
            
            self.logger.info("📊 GLOBAL STATUS SUMMARY:")
            self.logger.info(f"   Active Grids: {active_grids}/{total_grids}")
            self.logger.info(f"   Total Investment: ${self.total_investment_across_all_grids:.2f}")
            self.logger.info(f"   Available Capacity: ${self.max_total_investment - self.total_investment_across_all_grids:.2f}")
            self.logger.info(f"   Combined PnL: ${total_pnl:.2f}")
            
            if total_grids > 0:
                avg_pnl_pct = (total_pnl / self.total_investment_across_all_grids * 100) if self.total_investment_across_all_grids > 0 else 0
                self.logger.info(f"   Average PnL: {avg_pnl_pct:.2f}%")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging global status: {e}")
    
    def stop_grid(self, grid_id: str) -> bool:
        """Enhanced grid stopping with detailed logging"""
        try:
            if grid_id not in self.grids:
                self.logger.error(f"❌ Grid {grid_id} not found")
                return False
                
            grid = self.grids[grid_id]
            
            self.logger.info("=" * 80)
            self.logger.info(f"🛑 STOPPING GRID: {grid_id[:8]}")
            self.logger.info("=" * 80)
            
            # Get final status before stopping
            final_status = grid.get_status()
            
            self.logger.info(f"📊 FINAL GRID STATUS:")
            self.logger.info(f"   Symbol: {grid.original_symbol}")
            self.logger.info(f"   Investment: ${grid.user_total_investment:.2f}")
            self.logger.info(f"   Final PnL: ${final_status.get('pnl', 0):.2f} ({final_status.get('pnl_percentage', 0):.2f}%)")
            self.logger.info(f"   Total Trades: {final_status.get('trades_count', 0)}")
            self.logger.info(f"   Active Orders: {final_status.get('orders_count', 0)}")
            self.logger.info(f"   Active Positions: {final_status.get('active_positions', 0)}")
            
            # Stop grid
            grid.stop_grid()
            
            # Update grid status in data store
            self.data_store.save_grid(grid_id, grid.get_status())
            
            self.logger.info(f"✅ Grid {grid_id[:8]} stopped successfully")
            self._log_global_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping grid {grid_id}: {e}")
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
        """Enhanced grid status retrieval with comprehensive information"""
        try:
            grid_statuses = []
            
            for grid in self.grids.values():
                try:
                    status = grid.get_status()
                    
                    # ENHANCED: Add additional computed fields
                    status['global_investment_used'] = self.total_investment_across_all_grids
                    status['global_grid_count'] = len(self.grids)
                    status['global_active_count'] = sum(1 for g in self.grids.values() if g.running)
                    status['investment_utilization_pct'] = (self.total_investment_across_all_grids / self.max_total_investment * 100)
                    
                    # Add health indicators
                    pnl_pct = status.get('pnl_percentage', 0)
                    if pnl_pct >= 2.0:
                        status['health_status'] = 'excellent'
                    elif pnl_pct >= 0:
                        status['health_status'] = 'good'
                    elif pnl_pct >= -2.0:
                        status['health_status'] = 'caution'
                    else:
                        status['health_status'] = 'critical'
                    
                    grid_statuses.append(status)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error getting status for grid {grid.grid_id}: {e}")
                    # Add error status
                    grid_statuses.append({
                        'grid_id': grid.grid_id,
                        'symbol': getattr(grid, 'original_symbol', 'unknown'),
                        'error': str(e),
                        'health_status': 'error'
                    })
            
            return grid_statuses
            
        except Exception as e:
            self.logger.error(f"❌ Error getting all grids status: {e}")
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
        """FIXED: Enhanced grid monitoring with proper parallel processing and timeout handling"""
        self.logger.info("🔍 Grid monitor thread started with parallel processing")
        
        try:
            while self.running:
                monitor_start_time = time.time()
                self.monitor_cycle_count += 1
                
                # Get active grids
                active_grids = [(grid_id, grid) for grid_id, grid in list(self.grids.items()) if grid.running]
                
                should_log_summary = (
                    time.time() - self.last_monitor_summary > 1800 or  # Every 30 minutes
                    self.monitor_cycle_count % 60 == 1  # Every 60 cycles
                )
                
                if should_log_summary and active_grids:
                    self.logger.info(f"🔍 MONITOR CYCLE #{self.monitor_cycle_count}")
                    self.logger.info(f"   Active Grids: {len(active_grids)}/{len(self.grids)}")
                    self.logger.info(f"   Total Investment: ${self.total_investment_across_all_grids:.2f}")
                
                if active_grids:
                    # FIXED: Process grids in parallel using ThreadPoolExecutor with proper timeout handling
                    futures = []
                    for grid_id, grid in active_grids:
                        future = self.thread_pool.submit(self._update_single_grid, grid_id, grid)
                        futures.append((future, grid_id))
                    
                    # FIXED: Use wait() instead of as_completed() to handle timeouts properly
                    done_futures, pending_futures = concurrent.futures.wait(
                        [f[0] for f in futures], 
                        timeout=60,  # Global timeout for all grids
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
                    
                    success_count = 0
                    error_count = 0
                    timeout_count = 0
                    
                    # Process completed futures
                    for future, grid_id in futures:
                        try:
                            if future in done_futures:
                                # Future completed (either success or exception)
                                result = future.result(timeout=1)  # Quick result retrieval
                                if result:
                                    success_count += 1
                                else:
                                    error_count += 1
                            else:
                                # Future didn't complete in time
                                self.logger.warning(f"Grid {grid_id[:8]} update timed out")
                                timeout_count += 1
                                # Cancel the pending future
                                future.cancel()
                                
                        except concurrent.futures.TimeoutError:
                            self.logger.warning(f"Grid {grid_id[:8]} result retrieval timed out")
                            timeout_count += 1
                        except Exception as e:
                            self.logger.error(f"Grid {grid_id[:8]} update failed: {e}")
                            error_count += 1
                    
                    monitor_duration = time.time() - monitor_start_time
                    
                    # ENHANCED: Log monitor performance with timeout info
                    if should_log_summary:
                        self.logger.info(f"🔍 Parallel monitor cycle complete: {success_count} success, "
                                    f"{error_count} errors, {timeout_count} timeouts, {monitor_duration:.2f}s duration")
                        self.last_monitor_summary = time.time()
                    
                    # Log slow monitor cycles or timeout issues
                    if monitor_duration > 30.0 or timeout_count > 0:
                        self.logger.warning(f"⚠️ Monitor cycle issues: {monitor_duration:.2f}s duration, {timeout_count} timeouts")
                
                # Check every 10 seconds
                time.sleep(10)
                
        except Exception as e:
            self.logger.error(f"❌ Critical error in monitor_grids: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.logger.warning("🔍 Grid monitor thread stopped")
    def _update_single_grid(self, grid_id: str, grid: GridStrategy) -> bool:
        """ADDED: Update a single grid (called by thread pool)"""
        try:
            # Update grid with timing
            update_start = time.time()
            grid.update_grid()
            update_duration = time.time() - update_start
            
            # Log slow updates
            if update_duration > 5.0:
                self.logger.warning(f"⚠️ Slow grid update: {grid_id[:8]} took {update_duration:.2f}s")
            
            # Save updated grid status
            self.data_store.save_grid(grid_id, grid.get_status())
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating grid {grid_id[:8]}: {e}")
            return False
    def stop_all_grids(self) -> None:
        """Stop all grid strategies and cleanup thread pool."""
        try:
            self.logger.info("Stopping all grids")
            
            # Stop monitor thread
            self.running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(30)  # Wait up to 30 seconds
            
            # Stop all grids
            for grid_id in list(self.grids.keys()):
                self.stop_grid(grid_id)
            
            # ADDED: Shutdown thread pool
            self.thread_pool.shutdown(wait=True, timeout=30)
            
            self.logger.info("All grids stopped and thread pool shutdown")
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