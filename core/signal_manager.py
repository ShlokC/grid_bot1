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
        self.position_size_usd = 1.0  # Fixed $1 per strategy
        self.leverage = 20.0  # Fixed 20x leverage
        
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
        
        self.logger.info(f"Signal Manager initialized: {len(self.strategies)} strategies loaded")
    
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
            
            self.logger.info(f"Loaded {loaded_count} strategies")
            
        except Exception as e:
            self.logger.error(f"Error loading strategies: {e}")
    
    def _auto_manage_active_strategies(self):
        """Auto-create and manage strategies for top 10 most active symbols."""
        try:
            # Force fresh data from exchange instead of cached data
            self.logger.info("🔄 Getting fresh top 10 active symbols from exchange...")
            fresh_symbols = self.exchange.get_top_active_symbols(limit=10)
            
            if not fresh_symbols:
                self.logger.warning("No fresh active symbols received from exchange")
                return
            
            top_symbols = [symbol['symbol'] for symbol in fresh_symbols]
            self.logger.info(f"🎯 Auto-managing fresh top 10: {top_symbols}")
            
            # Get current running strategies  
            current_running = set()
            strategies_to_remove = []
            
            for strategy_id, strategy in list(self.strategies.items()):
                symbol = strategy.original_symbol
                
                if symbol in top_symbols:
                    # Symbol is in top 10 - keep it
                    if not strategy.running:
                        self.start_strategy(strategy_id)
                        self.logger.info(f"🚀 Started strategy for top symbol: {symbol}")
                    current_running.add(symbol)
                else:
                    # Symbol not in top 10 - check for position
                    has_position = self._check_strategy_has_position(strategy)
                    if not has_position:
                        # No position, remove strategy
                        strategies_to_remove.append(strategy_id)
                        self.logger.info(f"🗑️ Removing {symbol} (not in top 10, no position)")
            
            # Remove old strategies
            for strategy_id in strategies_to_remove:
                self.stop_strategy(strategy_id)
                self.delete_strategy(strategy_id)
            
            # Create missing strategies for top 10
            for symbol in top_symbols:
                if symbol not in current_running:
                    strategy_id = self.create_strategy(symbol, auto_created=True)
                    if strategy_id:
                        self.start_strategy(strategy_id)
                        self.logger.info(f"🆕 Created and started: {symbol}")
            
            # Update cached data with fresh data
            self.active_symbols_data['top_active'] = fresh_symbols
            self.active_symbols_data['last_updated'] = time.time()
                            
        except Exception as e:
            self.logger.error(f"❌ Error in auto-manage strategies: {e}")
    
    def _check_strategy_has_position(self, strategy) -> bool:
        """Check if strategy has active position."""
        try:
            positions = self.exchange.get_positions(strategy.symbol)
            for pos in positions:
                if pos.get('info', {}).get('symbol', '') == strategy.symbol:
                    size = float(pos.get('contracts', 0))
                    if abs(size) >= 0.001:
                        return True
            return False
        except Exception:
            return False
    
    def create_strategy(self, symbol: str, auto_created: bool = True) -> str:
        """Create a new signal strategy for a symbol."""
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
            
            # Create strategy
            strategy = SignalStrategy(
                exchange=self.exchange,
                symbol=symbol,
                strategy_id=strategy_id,
                position_size_usd=self.position_size_usd,
                leverage=self.leverage
            )
            
            self.strategies[strategy_id] = strategy
            
            # Save to data store
            self.data_store.save_grid(strategy_id, strategy.get_status())  # Reuse existing method
            
            self.logger.info(f"Created signal strategy: {strategy_id} for {symbol}")
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
    
    def get_strategy_status(self, strategy_id: str) -> Optional[Dict]:
        """Get status of a specific strategy."""
        try:
            if strategy_id not in self.strategies:
                return None
            return self.strategies[strategy_id].get_status()
        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return None
    
    def get_all_strategies_status(self) -> List[Dict]:
        """Get status of all strategies."""
        try:
            # Ensure monitor is running
            running_strategies = sum(1 for strategy in self.strategies.values() if strategy.running)
            if running_strategies > 0:
                self._ensure_monitor_running()
            
            strategy_statuses = []
            for strategy in self.strategies.values():
                try:
                    if strategy.running:
                        strategy.update_strategy()
                        self.data_store.save_grid(strategy.strategy_id, strategy.get_status())
                    
                    status = strategy.get_status()
                    strategy_statuses.append(status)
                    
                except Exception as e:
                    self.logger.error(f"Error getting status for strategy {strategy.strategy_id}: {e}")
            
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
                self.logger.info("✅ Strategy monitor started")
                
        except Exception as e:
            self.logger.error(f"Error ensuring monitor running: {e}")
    
    def _monitor_strategies(self):
        """Monitor strategies and auto-manage based on active symbols."""
        self.logger.info("🔍 Auto-strategy monitor thread started")
        
        cycle_count = 0
        try:
            while self.running:
                cycle_count += 1
                
                # Update active symbols data first
                try:
                    self._update_active_symbols_if_needed()
                    # Auto-manage strategies every 10 cycles (50 seconds)
                    if cycle_count % 10 == 1:
                        self._auto_manage_active_strategies()
                except Exception as e:
                    self.logger.error(f"Error updating active symbols: {e}")
                
                # Get active strategies
                active_strategies = [(sid, strategy) for sid, strategy in self.strategies.items() if strategy.running]
                
                if cycle_count % 10 == 1:
                    self.logger.info(f"🔍 Monitor cycle #{cycle_count}: {len(active_strategies)} active strategies")
                
                # Update strategies
                for strategy_id, strategy in active_strategies:
                    try:
                        strategy.update_strategy()
                        self.data_store.save_grid(strategy_id, strategy.get_status())
                    except Exception as e:
                        self.logger.error(f"Error updating strategy {strategy_id}: {e}")
                
                time.sleep(5)  # 5 second intervals
                
        except Exception as e:
            self.logger.error(f"Critical error in monitor: {e}")
        finally:
            self.logger.warning("🔍 Strategy monitor stopped")
    
    def _update_active_symbols_if_needed(self):
        """Update active symbols data periodically."""
        try:
            current_time = time.time()
            if current_time - self.active_symbols_data.get('last_updated', 0) < self.active_symbols_update_interval:
                return
            
            # Get market data
            top_active = self.exchange.get_top_active_symbols(limit=5)
            gainers_losers = self.exchange.get_top_gainers_losers(limit=3)
            
            self.active_symbols_data = {
                'top_active': top_active,
                'gainers': gainers_losers.get('gainers', []),
                'losers': gainers_losers.get('losers', []),
                'last_updated': current_time,
                'timeframe': '24hr'
            }
            
        except Exception as e:
            self.logger.error(f"Error updating active symbols: {e}")
    
    def get_active_symbols_data(self) -> Dict:
        """Get active symbols data."""
        try:
            return {
                'top_active': self.active_symbols_data.get('top_active', []).copy(),
                'gainers': self.active_symbols_data.get('gainers', []).copy(),
                'losers': self.active_symbols_data.get('losers', []).copy(),
                'last_updated': self.active_symbols_data.get('last_updated', 0),
                'timeframe': '24hr',
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
                'timeframe': '24hr',
                'last_updated_formatted': 'Never'
            }
    
    def force_update_active_symbols(self) -> bool:
        """Force update of active symbols data."""
        try:
            self.active_symbols_data['last_updated'] = 0
            self._update_active_symbols_if_needed()
            return len(self.active_symbols_data.get('top_active', [])) > 0
        except Exception as e:
            self.logger.error(f"Error force updating active symbols: {e}")
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