"""
Signal Strategy Manager - Manages multiple TSI-based signal strategies.
Simplified replacement for GridManager with fixed position sizing.
"""
import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional

from core.exchange import Exchange
from core.data_store import DataStore
from core.signal_strategy import SignalStrategy  # Import the new strategy

class SignalManager:
    def __init__(self, exchange: Exchange, data_store: DataStore):
        """Initialize signal strategy manager."""
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.data_store = data_store
        
        # Strategy instances
        self.strategies: Dict[str, SignalStrategy] = {}
        
        # Simple limits
        self.max_concurrent_strategies = 20  # Increased since strategies are smaller
        self.position_size_usd = 0.5  # Fixed $1 per strategy
        self.leverage = 20.0  # Fixed 10x leverage
        
        # Monitor thread
        self.monitor_thread = None
        self.running = False
        
        # Active symbols data (keeping this from original)
        self.active_symbols_data = {
            'top_active': [],
            'gainers': [],
            'losers': [],
            'last_updated': 0
        }
        self.active_symbols_update_interval = 30
        
        # Load existing strategies
        self._load_strategies()
        
        # Start monitor if needed
        running_strategies = sum(1 for strategy in self.strategies.values() if strategy.running)
        if running_strategies > 0:
            self._ensure_monitor_running()
        
        # self.logger.info(f"Signal Manager initialized: {len(self.strategies)} strategies loaded")
    
    def _load_strategies(self):
        """Load existing strategies from data store."""
        try:
            strategy_data = self.data_store.get_all_grids()  # Reuse existing storage
            if not strategy_data:
                return
            
            loaded_count = 0
            for strategy_id, config in strategy_data.items():
                try:
                    # Validate config
                    if not config.get('symbol'):
                        continue
                    
                    # Create strategy instance
                    strategy = SignalStrategy(
                        exchange=self.exchange,
                        symbol=config['symbol'],
                        strategy_id=strategy_id,
                        position_size_usd=self.position_size_usd,
                        leverage=self.leverage
                    )
                    
                    # Restore state
                    strategy.total_pnl = float(config.get('pnl', 0))
                    strategy.total_trades = int(config.get('trades_count', 0))
                    strategy.running = bool(config.get('running', False))
                    
                    self.strategies[strategy_id] = strategy
                    loaded_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error loading strategy {strategy_id}: {e}")
            
            # self.logger.info(f"Loaded {loaded_count} strategies")
            
        except Exception as e:
            self.logger.error(f"Error loading strategies: {e}")
    
    def _auto_manage_active_strategies(self):
        """FIXED: Auto-manage strategies - NEVER stop strategies with active positions."""
        try:
            # Force fresh data from exchange instead of cached data
            # self.logger.info("Getting fresh top 10 active symbols from exchange...")
            fresh_symbols = self.exchange.get_top_active_symbols(limit=10)
            
            if not fresh_symbols:
                self.logger.warning("No fresh active symbols received from exchange")
                return
            
            top_symbols = [symbol['symbol'] for symbol in fresh_symbols]
            # self.logger.info(f"Auto-managing fresh top 10: {top_symbols}")
            
            # FIXED: Create snapshots to avoid iteration errors during modification
            strategies_snapshot = dict(self.strategies)
            current_active_symbols = set()
            strategies_to_remove = []
            
            # Check existing strategies against top active symbols
            for strategy_id, strategy in strategies_snapshot.items():
                # Check if strategy still exists
                if strategy_id not in self.strategies:
                    continue
                    
                symbol = strategy.original_symbol
                
                if symbol in top_symbols:
                    # Symbol is in top 10 - keep it and ensure it's running
                    if not strategy.running:
                        self.start_strategy(strategy_id)
                        # self.logger.info(f"Started strategy for top active symbol: {symbol}")
                    current_active_symbols.add(symbol)
                else:
                    # Symbol not in top 10 - check for position before any action
                    has_position = self._check_strategy_has_position(strategy)
                    if not has_position:
                        # No position and not active - safe to remove
                        strategies_to_remove.append(strategy_id)
                        # self.logger.info(f"Marking {symbol} for removal (not active, no position)")
                    else:
                        # CRITICAL FIX: Has position - NEVER stop it, just log
                        # self.logger.info(f"Keeping {symbol} strategy (has position but not in top active)")
                        # DO NOT call strategy.stop_strategy() - let it keep running for exit evaluation
                        current_active_symbols.add(symbol)  # Keep it as "active" for position management
            
            # Remove inactive strategies safely (only those without positions)
            for strategy_id in strategies_to_remove:
                if strategy_id in self.strategies:  # Double-check it still exists
                    self.stop_strategy(strategy_id)
                    self.delete_strategy(strategy_id)
                    # self.logger.info(f"Deleted inactive strategy: {strategy_id}")
            
            # Create missing strategies for top active symbols
            for symbol in top_symbols:
                if symbol not in current_active_symbols:
                    strategy_id = self.create_strategy(symbol, auto_created=True, strategy_type='roc_multi_timeframe')
                    if strategy_id:
                        self.start_strategy(strategy_id)
                        # self.logger.info(f"Created and started strategy for: {symbol}")
            
            # Update cached data with fresh data
            self.active_symbols_data['top_active'] = fresh_symbols
            self.active_symbols_data['last_updated'] = time.time()
            
            # Log final state
            total_strategies = len(self.strategies)
            running_strategies = sum(1 for s in self.strategies.values() if s.running)
            # self.logger.info(f"Auto-management complete: {running_strategies}/{total_strategies} strategies running")
                            
        except Exception as e:
            self.logger.error(f"Error in auto-manage strategies: {e}")

    def _check_strategy_has_position(self, strategy) -> bool:
        """FIXED: Robust position check for strategy."""
        try:
            positions = self.exchange.get_positions(strategy.symbol)
            for pos in positions:
                pos_symbol = pos.get('info', {}).get('symbol', '').upper()
                our_symbol = strategy.symbol.upper()
                
                if pos_symbol == our_symbol:
                    size = float(pos.get('contracts', 0))
                    if abs(size) >= 0.001:  # Has meaningful position
                        self.logger.debug(f"Position found for {our_symbol}: {size:.6f}")
                        return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error checking position for {strategy.symbol}: {e}")
            return False  # Assume no position on error
    
    def create_strategy(self, symbol: str, auto_created: bool = True, strategy_type: str = 'roc_multi_timeframe') -> str:
        """Create a new signal strategy for a symbol with specified strategy type."""
        try:
            # Validate limits
            if len(self.strategies) >= self.max_concurrent_strategies:
                self.logger.error(f"Cannot create strategy: limit reached ({len(self.strategies)}/{self.max_concurrent_strategies})")
                return ""
            
            # Check for duplicate symbols
            for strategy in self.strategies.values():
                if strategy.original_symbol.upper() == symbol.upper() and strategy.running:
                    self.logger.error(f"Active strategy already exists for {symbol}")
                    return ""
            
            # Generate unique ID
            strategy_id = str(uuid.uuid4())
            
            # Create strategy with specified type
            strategy = SignalStrategy(
                exchange=self.exchange,
                symbol=symbol,
                strategy_id=strategy_id,
                position_size_usd=self.position_size_usd,
                leverage=self.leverage,
                strategy_type=strategy_type  # Pass strategy type
            )
            
            self.strategies[strategy_id] = strategy
            
            # Save to data store
            self.data_store.save_grid(strategy_id, strategy.get_status())  # Reuse existing method
            
            # self.logger.info(f"Created signal strategy: {strategy_id} for {symbol} using {strategy_type}")
            return strategy_id
            
        except Exception as e:
            self.logger.error(f"Error creating strategy: {e}")
            return ""
    
    def start_strategy(self, strategy_id: str) -> bool:
        """Start a signal strategy."""
        try:
            if strategy_id not in self.strategies:
                return False
            
            strategy = self.strategies[strategy_id]
            if strategy.start_strategy():
                self.data_store.save_grid(strategy_id, strategy.get_status())
                self._ensure_monitor_running()
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting strategy {strategy_id}: {e}")
            return False
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a signal strategy."""
        try:
            if strategy_id not in self.strategies:
                return False
            
            strategy = self.strategies[strategy_id]
            strategy.stop_strategy()
            self.data_store.save_grid(strategy_id, strategy.get_status())
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy {strategy_id}: {e}")
            return False
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a signal strategy."""
        try:
            if strategy_id not in self.strategies:
                return False
            
            strategy = self.strategies[strategy_id]
            if strategy.running:
                strategy.stop_strategy()
            
            del self.strategies[strategy_id]
            self.data_store.delete_grid(strategy_id)  # Reuse existing method
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting strategy {strategy_id}: {e}")
            return False
    
    def get_all_strategies_status(self) -> List[Dict]:
        """FIXED: Get status and update ALL strategies with positions."""
        try:
            # Ensure monitor is running
            running_strategies = sum(1 for strategy in self.strategies.values() if strategy.running)
            if running_strategies > 0:
                self._ensure_monitor_running()
            
            strategy_statuses = []
            
            # FIXED: Create a snapshot of strategies to avoid iteration errors
            strategies_snapshot = dict(self.strategies)
            
            for strategy_id, strategy in strategies_snapshot.items():
                try:
                    # Check if strategy still exists (might have been deleted)
                    if strategy_id not in self.strategies:
                        continue
                    
                    # CRITICAL FIX: Update if running OR has position
                    should_update = strategy.running
                    
                    if not should_update:
                        # Check if strategy has active position
                        has_position = self._check_strategy_has_position(strategy)
                        if has_position:
                            should_update = True
                            self.logger.debug(f"Status update for {strategy.symbol} (has position but not running)")
                    
                    if should_update:
                        strategy.update_strategy()
                        self.data_store.save_grid(strategy_id, strategy.get_status())
                    
                    status = strategy.get_status()
                    strategy_statuses.append(status)
                    
                except Exception as e:
                    self.logger.error(f"Error getting status for strategy {strategy_id}: {e}")
                    # Add error status instead of skipping
                    strategy_statuses.append({
                        'strategy_id': strategy_id,
                        'symbol': getattr(strategy, 'symbol', 'unknown'),
                        'error': str(e),
                        'running': False
                    })
            
            return strategy_statuses
            
        except Exception as e:
            self.logger.error(f"Error getting all strategies status: {e}")
            return []
    
    def _ensure_monitor_running(self):
        """Ensure monitor thread is running."""
        try:
            thread_running = (self.monitor_thread is not None and self.monitor_thread.is_alive())
            
            if not thread_running:
                if self.running:
                    self.running = False
                    if self.monitor_thread and self.monitor_thread.is_alive():
                        self.monitor_thread.join(timeout=2)
                
                self.running = True
                self.monitor_thread = threading.Thread(target=self._monitor_strategies, daemon=True)
                self.monitor_thread.start()
                # self.logger.info("✅ Strategy monitor started")
                
        except Exception as e:
            self.logger.error(f"Error ensuring monitor running: {e}")
    
    def _monitor_strategies(self):
        """FIXED: Monitor ALL strategies with positions, regardless of running status."""
        # self.logger.info("Monitor started - Will update ALL strategies with positions")
        
        cycle_count = 0
        last_active_symbols_sync = 0
        
        try:
            while self.running:
                cycle_count += 1
                
                # Update active symbols data first
                try:
                    self._update_active_symbols_if_needed()
                    
                    # Immediate sync when active symbols are updated
                    current_active_update = self.active_symbols_data.get('last_updated', 0)
                    if current_active_update > last_active_symbols_sync:
                        # self.logger.info(f"Active symbols updated - immediate strategy sync")
                        self._auto_manage_active_strategies()
                        last_active_symbols_sync = current_active_update
                    
                except Exception as e:
                    self.logger.error(f"Error updating active symbols: {e}")
                
                # FIXED: Update ALL strategies with positions OR running status
                try:
                    strategies_snapshot = dict(self.strategies)
                    strategies_to_update = []
                    
                    for strategy_id, strategy in strategies_snapshot.items():
                        if strategy_id not in self.strategies:  # Strategy was deleted
                            continue
                        
                        # CRITICAL FIX: Update if running OR has position
                        should_update = strategy.running
                        
                        if not should_update:
                            # Check if strategy has active position
                            has_position = self._check_strategy_has_position(strategy)
                            if has_position:
                                should_update = True
                                self.logger.debug(f"Updating {strategy.symbol} (has position but not running)")
                        
                        if should_update:
                            strategies_to_update.append((strategy_id, strategy))
                    
                    if cycle_count % 10 == 1:
                        running_count = sum(1 for _, s in strategies_to_update if s.running)
                        position_count = len(strategies_to_update) - running_count
                        # self.logger.info(f"Monitor cycle #{cycle_count}: {running_count} running + {position_count} with positions = {len(strategies_to_update)} total")
                    
                    # Update all strategies that need updating
                    for strategy_id, strategy in strategies_to_update:
                        try:
                            # Double-check strategy still exists
                            if strategy_id not in self.strategies:
                                continue
                                
                            strategy.update_strategy()
                            self.data_store.save_grid(strategy_id, strategy.get_status())
                        except Exception as e:
                            self.logger.error(f"Error updating strategy {strategy_id}: {e}")
                
                except Exception as e:
                    self.logger.error(f"Error in strategy monitoring: {e}")
                
                time.sleep(5)  # 5 second intervals
                
        except Exception as e:
            self.logger.error(f"Critical error in monitor: {e}")
        finally:
            self.logger.warning("Strategy monitor stopped")
    def test_all_strategies_setup(self) -> bool:
        """Test trading setup for all active strategies."""
        try:
            # self.logger.info("🧪 Testing trading setup for all strategies...")
            
            success_count = 0
            total_count = 0
            
            for strategy_id, strategy in self.strategies.items():
                total_count += 1
                if strategy.test_trading_setup():
                    success_count += 1
            
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            # self.logger.info(f"📊 Setup test results: {success_count}/{total_count} passed ({success_rate:.1f}%)")
            
            if success_count == total_count:
                # self.logger.info("🎉 All strategy setups passed!")
                return True
            else:
                self.logger.warning("⚠️ Some strategy setups failed!")
                return False
                
        except Exception as e:
            self.logger.error(f"Error testing all strategies setup: {e}")
            return False
    def _update_active_symbols_if_needed(self):
        """FIXED: Update active symbols with faster sync for real-time strategy management."""
        try:
            current_time = time.time()
            last_update = self.active_symbols_data.get('last_updated', 0)
            
            # FIXED: Reduced update interval for better responsiveness (15 seconds instead of 30)
            if current_time - last_update < 15:  # Update every 15 seconds for real-time
                return
            
            # self.logger.info("Updating active symbols data (1m real-time volume spike detection)")
            
            # FIXED: Use 5-minute timeframe for better small crypto detection
            top_active = self.exchange.get_top_active_symbols(limit=10, timeframe_minutes=5)
            gainers_losers = self.exchange.get_top_gainers_losers(limit=5, timeframe_minutes=5)
            
            # Update stored data
            self.active_symbols_data = {
                'top_active': top_active,
                'gainers': gainers_losers.get('gainers', []),
                'losers': gainers_losers.get('losers', []),
                'last_updated': current_time,
                'timeframe': '5min'  # FIXED: Reflect actual timeframe
            }
            
            # self.logger.info(f"Active symbols updated (5min real-time): {len(top_active)} active, "
            #                 f"{len(gainers_losers.get('gainers', []))} gainers, "
            #                 f"{len(gainers_losers.get('losers', []))} losers")
            
        except Exception as e:
            self.logger.error(f"Error updating active symbols data: {e}")
    
    def get_active_symbols_data(self) -> Dict:
        """FIXED: Get current active symbols data with correct timeframe info."""
        try:
            # Return a copy to prevent external modification
            return {
                'top_active': self.active_symbols_data.get('top_active', []).copy(),
                'gainers': self.active_symbols_data.get('gainers', []).copy(),
                'losers': self.active_symbols_data.get('losers', []).copy(),
                'last_updated': self.active_symbols_data.get('last_updated', 0),
                'timeframe': self.active_symbols_data.get('timeframe', '5min'),  # FIXED: Real timeframe
                'last_updated_formatted': time.strftime('%H:%M:%S', 
                    time.localtime(self.active_symbols_data.get('last_updated', 0)))
            }
        except Exception as e:
            self.logger.error(f"Error getting active symbols data: {e}")
            return {
                'top_active': [],
                'gainers': [],
                'losers': [],
                'last_updated': 0,
                'timeframe': '5min',
                'last_updated_formatted': 'Never'
            }
    
    def force_update_active_symbols(self) -> bool:
        """Force update of active symbols data."""
        try:
           # self.logger.info("Force syncing strategies with active symbols...")
            
            # Force active symbols update
            self.active_symbols_data['last_updated'] = 0
            self._update_active_symbols_if_needed()
            
            # Force strategy management
            self._auto_manage_active_strategies()
            
            return True
        except Exception as e:
            self.logger.error(f"Error force syncing: {e}")
            return False
    
    def stop_all_strategies(self):
        """Stop all strategies."""
        try:
            self.running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(10)
            
            for strategy_id in list(self.strategies.keys()):
                self.stop_strategy(strategy_id)
            
            self.logger.info("All strategies stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping all strategies: {e}")
    
    def get_global_status(self) -> Dict:
        """Get global manager status."""
        try:
            active_strategies = sum(1 for strategy in self.strategies.values() if strategy.running)
            total_investment = len(self.strategies) * self.position_size_usd
            
            return {
                'total_strategies': len(self.strategies),
                'active_strategies': active_strategies,
                'total_investment': total_investment,
                'position_size_per_strategy': self.position_size_usd,
                'leverage': self.leverage,
                'max_concurrent_strategies': self.max_concurrent_strategies
            }
            
        except Exception as e:
            self.logger.error(f"Error getting global status: {e}")
            return {}