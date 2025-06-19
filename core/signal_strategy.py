"""
Simplified Signal Trading Strategy - TSI-based trading with fixed position sizing.
Replaces the misleading "Grid" implementation with clear signal-based trading.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional
from core.exchange import Exchange
from core.adaptive_tsi import integrate_adaptive_tsi

class SignalStrategy:
    """
    TSI Signal-based Trading Strategy with fixed position sizing.
    Simplified replacement for the misleading GridStrategy.
    """
    
    def __init__(self, 
                 exchange: Exchange, 
                 symbol: str,
                 strategy_id: str,
                 position_size_usd: float = 1.0,
                 leverage: float = 20.0,
                 take_profit_pct: float = 2.0,
                 stop_loss_pct: float = 2.0):
        """Initialize signal strategy with fixed parameters."""
        
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        self.strategy_id = strategy_id
        
        # Fixed parameters (simplified)
        self.position_size_usd = position_size_usd
        self.leverage = leverage
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        
        # Fetch market info
        self._fetch_market_info()
        
        # State tracking
        self.running = False
        self.total_trades = 0
        self.total_pnl = 0.0
        self.filled_orders = []
        self.active_order_ids = set()
        self.last_update_time = 0
        
        # Threading
        self.update_lock = threading.Lock()
        
        # Integrate TSI system
        integrate_adaptive_tsi(self)
        
        self.logger.info(f"Signal Strategy initialized: {symbol}, Size: ${position_size_usd}, Leverage: {leverage}x")
    
    def _fetch_market_info(self):
        """Fetch market precision and limits."""
        try:
            market_info = self.exchange.get_market_info(self.symbol)
            precision_info = market_info.get('precision', {})
            limits = market_info.get('limits', {})
            
            self.price_precision = int(precision_info.get('price', 6))
            self.amount_precision = int(precision_info.get('amount', 6))
            self.min_amount = float(limits.get('amount', {}).get('min', 0.0001))
            self.min_cost = float(limits.get('cost', {}).get('min', 1.0))
            
        except Exception as e:
            self.logger.error(f"Error fetching market info: {e}")
            # Defaults
            self.price_precision = 6
            self.amount_precision = 6
            self.min_amount = 0.0001
            self.min_cost = 1.0
    
    def _calculate_position_amount(self, price: float) -> float:
        """Calculate position amount for fixed USD size."""
        try:
            if price <= 0:
                return self.min_amount
            
            # Calculate quantity for fixed USD amount with leverage
            notional_value = self.position_size_usd * self.leverage
            quantity = notional_value / price
            
            # Round to precision
            rounded_amount = float(f"{quantity:.{self.amount_precision}f}")
            return max(self.min_amount, rounded_amount)
            
        except Exception as e:
            self.logger.error(f"Error calculating position amount: {e}")
            return self.min_amount
    
    def start_strategy(self) -> bool:
        """Start the signal-based strategy."""
        try:
            if self.running:
                return True
            
            self.logger.info(f"🚀 Starting Signal Strategy: {self.symbol}")
            
            # Cancel any existing orders
            try:
                cancelled = self.exchange.cancel_all_orders(self.symbol)
                if cancelled:
                    self.logger.info(f"Cancelled {len(cancelled)} existing orders")
                time.sleep(1)
            except Exception as e:
                self.logger.warning(f"Failed to cancel existing orders: {e}")
            
            self.running = True
            self.logger.info(f"✅ Signal Strategy started: {self.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting strategy: {e}")
            return False
    
    def update_strategy(self):
        """Main strategy update loop."""
        try:
            with self.update_lock:
                if not self.running:
                    return
                
                # Check current position
                self._check_position_and_signals()
                
                # Update PnL
                self._update_pnl()
                
                # Check TP/SL conditions  
                self._check_tp_sl_conditions()
                
                self.last_update_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Error updating strategy: {e}")
    
    def _check_position_and_signals(self):
        """Check position and handle signals for THIS symbol only."""
        try:
            # Get positions for this specific symbol
            positions = self.exchange.get_positions(self.symbol)
            current_position = None
            
            for pos in positions:
                pos_symbol = pos.get('info', {}).get('symbol', '')
                if pos_symbol == self.symbol:
                    size = float(pos.get('contracts', 0))
                    if abs(size) >= 0.001:
                        current_position = pos
                        break
            
            # Get technical signal
            tech_signal = self._get_technical_direction()
            
            if current_position:
                # Has position - check for exit signal
                self._handle_position_management(current_position, tech_signal)
            else:
                # No position - check for entry signal
                self._handle_entry_signals(tech_signal)
                
        except Exception as e:
            self.logger.error(f"Error checking position and signals: {e}")
    
    def _handle_entry_signals(self, signal: str):
        """Handle entry signals when no position exists."""
        try:
            if signal in ['buy', 'sell']:
                # CRITICAL: Double-check no position exists before entry
                if self._has_active_position():
                    self.logger.debug(f"Skipping {signal} signal - position already exists")
                    return
                
                # Check recent orders throttle
                if hasattr(self, '_last_entry_time'):
                    if time.time() - self._last_entry_time < 60:
                        return
                
                # Get current price and calculate amount
                ticker = self.exchange.get_ticker(self.symbol)
                current_price = float(ticker['last'])
                amount = self._calculate_position_amount(current_price)
                
                if amount >= self.min_amount:
                    # Final position check before order
                    if self._has_active_position():
                        self.logger.warning(f"Position detected during order preparation - cancelling {signal}")
                        return
                    
                    # Place market order
                    order = self.exchange.create_market_order(self.symbol, signal, amount)
                    if order and 'id' in order:
                        self._last_entry_time = time.time()
                        self.total_trades += 1
                        
                        # Record the fill
                        fill_price = float(order.get('average', current_price))
                        self.filled_orders.append({
                            'id': order['id'],
                            'side': signal,
                            'price': fill_price,
                            'amount': amount,
                            'timestamp': time.time(),
                            'type': 'entry'
                        })
                        
                        self.logger.info(f"✅ Entry: {signal.upper()} {amount:.6f} @ ${fill_price:.6f}")
                        
        except Exception as e:
            self.logger.error(f"Error handling entry signals: {e}")
    
    def _handle_position_management(self, position: Dict, signal: str):
        """Handle position management and exit signals."""
        try:
            position_size = float(position.get('contracts', 0))
            
            # Determine position side from size (not from position.side which can be unreliable)
            if position_size > 0:
                position_side = 'long'
                close_side = 'sell'
            elif position_size < 0:
                position_side = 'short' 
                close_side = 'buy'
            else:
                return  # No position
            
            close_amount = abs(position_size)
            
            # Check for opposing signal (exit condition)
            should_exit = False
            if position_side == 'long' and signal == 'sell':
                should_exit = True
            elif position_side == 'short' and signal == 'buy':
                should_exit = True
            
            if should_exit:
                self.logger.info(f"🚨 Closing {position_side.upper()} position: {close_side.upper()} {close_amount:.6f}")
                
                try:
                    exit_order = self.exchange.create_market_order(self.symbol, close_side, close_amount)
                    if exit_order and 'id' in exit_order:
                        self.total_trades += 1
                        
                        # Record the exit
                        fill_price = float(exit_order.get('average', 0))
                        self.filled_orders.append({
                            'id': exit_order['id'],
                            'side': close_side,
                            'price': fill_price,
                            'amount': close_amount,
                            'timestamp': time.time(),
                            'type': 'exit'
                        })
                        
                        self.logger.info(f"✅ Position closed: {close_side.upper()} {close_amount:.6f} @ ${fill_price:.6f}")
                        
                except Exception as e:
                    self.logger.error(f"Error closing position: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in position management: {e}")
    
    def _update_pnl(self):
        """Update PnL from live exchange data for THIS symbol only."""
        try:
            # Get real-time unrealized PnL from open positions for this symbol
            unrealized_pnl = 0.0
            try:
                positions = self.exchange.get_positions(self.symbol)
                for pos in positions:
                    pos_symbol = pos.get('info', {}).get('symbol', '')
                    if pos_symbol == self.symbol:
                        unrealized_pnl += float(pos.get('unrealizedPnl', 0))
            except Exception as e:
                self.logger.debug(f"Could not get unrealized PnL for {self.symbol}: {e}")
            
            # Calculate realized PnL from completed trades
            realized_pnl = 0.0
            if len(self.filled_orders) >= 2:
                for i in range(0, len(self.filled_orders) - 1, 2):
                    if i + 1 >= len(self.filled_orders):
                        break
                        
                    entry_order = self.filled_orders[i]
                    exit_order = self.filled_orders[i + 1]
                    
                    if entry_order.get('type') == 'entry' and exit_order.get('type') == 'exit':
                        if entry_order['side'] == 'buy' and exit_order['side'] == 'sell':
                            pnl = (exit_order['price'] - entry_order['price']) * entry_order['amount']
                        elif entry_order['side'] == 'sell' and exit_order['side'] == 'buy':
                            pnl = (entry_order['price'] - exit_order['price']) * entry_order['amount']
                        else:
                            continue
                        
                        realized_pnl += pnl
            
            # Total PnL = realized + unrealized
            self.total_pnl = realized_pnl + unrealized_pnl
            
        except Exception as e:
            self.logger.error(f"Error updating PnL for {self.symbol}: {e}")
    
    def _check_tp_sl_conditions(self):
        """Check take profit and stop loss conditions."""
        try:
            if self.position_size_usd <= 0:
                return
            
            pnl_percentage = (self.total_pnl / self.position_size_usd) * 100
            
            # Check take profit
            if pnl_percentage >= self.take_profit_pct:
                self.logger.info(f"Take profit reached: {pnl_percentage:.2f}%")
                self.stop_strategy()
                return
            
            # Check stop loss  
            if pnl_percentage <= -self.stop_loss_pct:
                self.logger.info(f"Stop loss reached: {pnl_percentage:.2f}%")
                self.stop_strategy()
                return
                
        except Exception as e:
            self.logger.error(f"Error checking TP/SL: {e}")
    
    def stop_strategy(self):
        """Stop the strategy and close positions."""
        try:
            if not self.running:
                return
            
            self.running = False
            
            # Cancel all orders
            try:
                cancelled = self.exchange.cancel_all_orders(self.symbol)
                if cancelled:
                    self.logger.info(f"Cancelled {len(cancelled)} orders")
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
            
            # Close any open positions
            try:
                positions = self.exchange.get_positions(self.symbol)
                for position in positions:
                    if position.get('info', {}).get('symbol', '') == self.symbol:
                        size = float(position.get('contracts', 0))
                        if abs(size) >= 0.001:
                            close_side = 'sell' if size > 0 else 'buy'
                            self.exchange.create_market_order(self.symbol, close_side, abs(size))
                            self.logger.info(f"Closed position: {close_side.upper()} {abs(size):.6f}")
            except Exception as e:
                self.logger.error(f"Error closing positions: {e}")
            
            self.logger.info("✅ Strategy stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy: {e}")
            self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status with position-based running state."""
        try:
            with self.update_lock:
                pnl_percentage = (self.total_pnl / self.position_size_usd * 100) if self.position_size_usd > 0 else 0.0
                
                # Check if has active position
                has_position = self._has_active_position()
                # Strategy is "running" if it has active position OR is internally running
                effective_running = has_position or self.running
                
                return {
                    'strategy_id': self.strategy_id,
                    'symbol': self.symbol,
                    'display_symbol': self.original_symbol,
                    'position_size_usd': self.position_size_usd,
                    'leverage': self.leverage,
                    'take_profit_pct': self.take_profit_pct,
                    'stop_loss_pct': self.stop_loss_pct,
                    'running': effective_running,
                    'has_position': has_position,
                    'trades_count': self.total_trades,
                    'pnl': self.total_pnl,
                    'pnl_percentage': pnl_percentage,
                    'last_update': self.last_update_time,
                }
                
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {
                'strategy_id': self.strategy_id,
                'symbol': self.symbol,
                'running': False,
                'error': str(e)
            }
    
    def _has_active_position(self) -> bool:
        """Check if strategy has active position for THIS symbol only."""
        try:
            positions = self.exchange.get_positions(self.symbol)
            for pos in positions:
                # More robust symbol matching
                pos_symbol = pos.get('info', {}).get('symbol', '')
                if pos_symbol == self.symbol:
                    size = float(pos.get('contracts', 0))
                    if abs(size) >= 0.001:
                        return True
            return False
        except Exception:
            return False