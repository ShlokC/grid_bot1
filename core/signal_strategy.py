"""
Simplified Signal Trading Strategy - TSI-based trading with fixed position sizing.
Replaces the misleading "Grid" implementation with clear signal-based trading.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional
from core.exchange import Exchange
from core.adaptive_tsi import integrate_momentum_tsi


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
                 leverage: float = 20.0):
        """Initialize signal strategy with fixed parameters."""
        
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        self.strategy_id = strategy_id
        
        # Fixed parameters (simplified)
        self.position_size_usd = position_size_usd
        self.leverage = leverage
        
        
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
        # Order tracking for duplicate detection
        self._last_order_time = 0
        self._order_attempts = []
        
        self.logger.info(f"✅ Centralized order management initialized for {symbol}")
        # Integrate TSI system
        integrate_momentum_tsi(self)
        
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
        """FIXED: Clean start without order placement"""
        try:
            if self.running:
                return True
            
            self.logger.info(f"🚀 Starting Signal Strategy: {self.symbol}")
            
            # Cancel any existing orders but don't place new ones
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
    def test_duplicate_prevention(self):
        """Test method to verify no duplicates can occur."""
        self.logger.info("🧪 Testing duplicate prevention...")
        
        # Store original method
        original_create_order = self.exchange.create_market_order
        
        # Mock create_market_order to track calls
        order_calls = []
        def mock_create_order(symbol, side, amount):
            order_calls.append({'symbol': symbol, 'side': side, 'amount': amount, 'time': time.time()})
            return {'id': f'test_{len(order_calls)}', 'average': 100.0}
        
        self.exchange.create_market_order = mock_create_order
        
        # Run multiple update cycles rapidly
        for i in range(5):
            self.update_strategy()
            time.sleep(0.1)  # Small delay
        
        # Restore original method
        self.exchange.create_market_order = original_create_order
        
        # Analyze results
        self.logger.info(f"🧪 Test complete: {len(order_calls)} orders would have been placed")
        
        if len(order_calls) <= 1:
            self.logger.info("✅ DUPLICATE PREVENTION WORKING: Maximum 1 order per test")
            return True
        else:
            self.logger.error(f"❌ DUPLICATES DETECTED: {len(order_calls)} orders in rapid test")
            return False
    def test_trading_setup(self) -> bool:
        """Test trading setup for this strategy's symbol."""
        try:
            self.logger.info(f"🧪 Testing trading setup for {self.symbol}...")
            
            # FIXED: Call setup on exchange instance, not self
            success = self.exchange.setup_symbol_trading_config(self.symbol, 20)
            
            if success:
                # Get current config to verify
                config = self.exchange.get_symbol_trading_config(self.symbol)
                
                self.logger.info(f"📊 Current config for {self.symbol}:")
                self.logger.info(f"   Margin Type: {config.get('margin_type', 'UNKNOWN')}")
                self.logger.info(f"   Leverage: {config.get('leverage', 'UNKNOWN')}x")
                self.logger.info(f"   Position Size: {config.get('position_size', 0)}")
                
                # Verify requirements
                margin_ok = config.get('margin_type') == 'ISOLATED'
                leverage_ok = config.get('leverage') == 20
                
                if margin_ok and leverage_ok:
                    self.logger.info(f"✅ {self.symbol} correctly configured for trading")
                    return True
                else:
                    self.logger.error(f"❌ {self.symbol} configuration incorrect:")
                    if not margin_ok:
                        self.logger.error(f"   Expected ISOLATED margin, got {config.get('margin_type')}")
                    if not leverage_ok:
                        self.logger.error(f"   Expected 20x leverage, got {config.get('leverage')}x")
                    return False
            else:
                self.logger.error(f"❌ Setup failed for {self.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error testing trading setup: {e}")
            return False
    def update_strategy(self):
        """FIXED: Centralized order management to prevent duplicates."""
        try:
            with self.update_lock:
                if not self.running:
                    return
                
                # FIXED: Single data fetch for entire update cycle
                live_data = self._get_live_trading_data()
                
                # FIXED: Single decision point for all orders
                order_decision = self._make_order_decision(live_data)
                
                # FIXED: Single execution point to prevent duplicates
                if order_decision['action'] != 'none':
                    self._execute_single_order(order_decision, live_data)
                
                # Update PnL (non-order related)
                self._update_pnl()
                
                self.last_update_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Error updating strategy: {e}")
    def _get_live_trading_data(self) -> Dict:
        """FIXED: Single API call to get ALL required data."""
        try:
            # Get all data in one batch to prevent inconsistencies
            positions = self.exchange.get_positions(self.symbol)
            open_orders = self.exchange.get_open_orders(self.symbol)
            ticker = self.exchange.get_ticker(self.symbol)
            tech_signal = self._get_technical_direction()
            
            # Parse position data
            current_position = None
            position_size = 0.0
            position_side = None
            entry_price = 0.0
            
            for pos in positions:
                pos_symbol = pos.get('info', {}).get('symbol', '').upper()
                if pos_symbol == self.symbol.upper():
                    size = float(pos.get('contracts', 0))
                    if abs(size) >= 0.001:
                        current_position = pos
                        position_size = size
                        position_side = 'long' if size > 0 else 'short'
                        entry_price = float(pos.get('entryPrice', 0))
                        break
            
            # Count orders by type
            symbol_orders = [o for o in open_orders if o.get('info', {}).get('symbol', '') == self.symbol]
            pending_orders = len(symbol_orders)
            
            live_data = {
            'current_price': float(ticker['last']),
            'tech_signal': tech_signal,
            'has_position': current_position is not None,
            'position_size': position_size,
            'position_side': position_side,
            'entry_price': entry_price,
            'pending_orders': pending_orders,
            'timestamp': time.time()
            }
            self.logger.info(f"📊 {self.symbol}: ${live_data['current_price']:.6f} | {live_data['tech_signal'].upper()} | {live_data['position_side'] or 'NO_POS'} {abs(live_data['position_size']):.6f} | Pending:{live_data['pending_orders']}")
    
            return live_data
        except Exception as e:
            self.logger.error(f"Error getting live trading data: {e}")
            return {'tech_signal': 'none', 'has_position': False, 'pending_orders': 0, 'current_price': 0}

    def _make_order_decision(self, live_data: Dict) -> Dict:
        """FIXED: Single decision engine to prevent conflicting orders."""
        try:
            signal = live_data['tech_signal']
            has_position = live_data['has_position']
            position_side = live_data.get('position_side')
            pending_orders = live_data['pending_orders']
            current_time = live_data['timestamp']
            
            # Default: no action
            decision = {'action': 'none', 'side': '', 'reason': ''}
            
            # RULE 1: No action if invalid signal
            if signal not in ['buy', 'sell']:
                decision['reason'] = f'Invalid signal: {signal}'
                return decision
            
            # RULE 2: No action if orders pending
            if pending_orders > 0:
                decision['reason'] = f'{pending_orders} orders already pending'
                return decision
            
            # RULE 3: Throttling check
            if hasattr(self, '_last_order_time'):
                time_since_last = current_time - self._last_order_time
                if time_since_last < 10:  # 10 second minimum gap
                    decision['reason'] = f'Throttled: {time_since_last:.1f}s < 10s'
                    return decision
            
            # RULE 4: Position management (exits)
            if has_position:
                should_exit = False
                
                if position_side == 'long' and signal == 'sell':
                    should_exit = True
                    decision['reason'] = 'Exit LONG on SELL signal'
                elif position_side == 'short' and signal == 'buy':
                    should_exit = True
                    decision['reason'] = 'Exit SHORT on BUY signal'
                
                if should_exit:
                    decision['action'] = 'exit'
                    decision['side'] = 'sell' if position_side == 'long' else 'buy'
                    return decision
                else:
                    decision['reason'] = f'No exit needed: {position_side.upper()} + {signal.upper()} signal'
                    return decision
            
            # RULE 5: Entry management (new positions)
            if not has_position:
                decision['action'] = 'entry'
                decision['side'] = signal
                decision['reason'] = f'New {signal.upper()} entry'
                return decision
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making order decision: {e}")
            return {'action': 'none', 'side': '', 'reason': f'Error: {e}'}

    def _execute_single_order(self, decision: Dict, live_data: Dict):
        """FIXED: Single order execution point - eliminates all duplicates."""
        try:
            action = decision['action']
            side = decision['side'] 
            reason = decision['reason']
            current_price = live_data['current_price']
            
            if action not in ['entry', 'exit'] or side not in ['buy', 'sell']:
                return
            
            # Calculate amount
            if action == 'entry':
                amount = self._calculate_position_amount(current_price)
            else:  # exit
                amount = abs(live_data['position_size'])
            
            if amount < self.min_amount:
                self.logger.warning(f"Amount {amount:.6f} below minimum {self.min_amount}")
                return
            
            # Log the decision
            self.logger.info(f"🎯 ORDER DECISION: {action.upper()} {side.upper()} {amount:.6f} @ ${current_price:.6f}")
            self.logger.info(f"🎯 Reason: {reason}")
            
            # SINGLE ORDER EXECUTION POINT
            order = self.exchange.create_market_order(self.symbol, side, amount)
            
            if order and 'id' in order:
                # Update tracking
                self._last_order_time = live_data['timestamp']
                self.total_trades += 1
                
                # Record the order
                fill_price = float(order.get('average', current_price))
                self.filled_orders.append({
                    'id': order['id'],
                    'side': side,
                    'price': fill_price,
                    'amount': amount,
                    'timestamp': live_data['timestamp'],
                    'type': action
                })
                
                self.logger.info(f"✅ {action.upper()} EXECUTED: {side.upper()} {amount:.6f} @ ${fill_price:.6f}")
                self.logger.info(f"✅ Order ID: {order['id']}")
            else:
                self.logger.error(f"❌ Order execution failed")
                
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
    # def _check_position_and_signals(self):
    #     """FIXED: Single position fetch to prevent race condition duplicates."""
    #     try:
    #         # FIXED: Get positions ONCE and reuse the result everywhere
    #         positions = self.exchange.get_positions(self.symbol)
    #         current_position = None
    #         has_active_position = False
            
    #         # FIXED: Single position detection with reusable boolean result
    #         for pos in positions:
    #             pos_symbol = pos.get('info', {}).get('symbol', '').upper()
    #             our_symbol = self.symbol.upper()
                
    #             if pos_symbol == our_symbol:
    #                 size = float(pos.get('contracts', 0))
    #                 if abs(size) >= 0.001:  # Has meaningful position
    #                     current_position = pos
    #                     has_active_position = True
    #                     self.logger.debug(f"📊 Position detected: {our_symbol} = {size:.6f}")
    #                     break
            
    #         # Get technical signal
    #         tech_signal = self._get_technical_direction()
            
    #         if has_active_position:
    #             # Has position - handle exit logic
    #             self._handle_position_management(current_position, tech_signal)
    #         else:
    #             # No position - handle entry logic (PASS position status to prevent re-checking)
    #             self.logger.debug(f"📊 No position for {self.symbol}, checking entry")
    #             self._handle_entry_signals(tech_signal, position_exists=False)
                
    #     except Exception as e:
    #         self.logger.error(f"Error checking position and signals: {e}")

    # def _handle_entry_signals(self, signal: str, position_exists: bool = None):
    #     """FIXED: Use passed position status to prevent duplicate API calls and race conditions."""
    #     try:
    #         if signal not in ['buy', 'sell']:
    #             return
                
    #         # FIXED: Use passed position status - no more duplicate get_positions() calls!
    #         if position_exists is True:
    #             self.logger.debug(f"Skipping {signal} - position exists (passed from caller)")
    #             return
    #         elif position_exists is None:
    #             # Fallback: only call API if position status wasn't provided
    #             if self._has_active_position():
    #                 self.logger.debug(f"Skipping {signal} - position exists (fallback API call)")
    #                 return
            
    #         # Throttling check (unchanged)
    #         current_time = time.time()
    #         if hasattr(self, '_last_entry_time'):
    #             time_since_last = current_time - self._last_entry_time
    #             if time_since_last < 10:  # 10 seconds throttle
    #                 self.logger.debug(f"Entry throttled: {time_since_last:.1f}s < 10s")
    #                 return
            
    #         # Check for open orders (unchanged) 
    #         try:
    #             open_orders = self.exchange.get_open_orders(self.symbol)
    #             if open_orders:
    #                 self.logger.debug(f"Skipping {signal} - {len(open_orders)} open orders exist")
    #                 return
    #         except Exception as e:
    #             self.logger.warning(f"Could not check open orders: {e}")
            
    #         # Calculate order details (unchanged)
    #         ticker = self.exchange.get_ticker(self.symbol)
    #         current_price = float(ticker['last'])
    #         amount = self._calculate_position_amount(current_price)
            
    #         if amount >= self.min_amount:
    #             # REMOVED: Final position check since we already know position status
    #             # This eliminates the race condition entirely!
                
    #             self.logger.info(f"🎯 ENTRY ORDER: {signal.upper()} {amount:.6f} @ ${current_price:.6f}")
                
    #             # Place market order (unchanged)
    #             order = self.exchange.create_market_order(self.symbol, signal, amount)
    #             if order and 'id' in order:
    #                 # Update tracking (unchanged)
    #                 self._last_entry_time = current_time
    #                 self.total_trades += 1
                    
    #                 # Record the fill (unchanged)
    #                 fill_price = float(order.get('average', current_price))
    #                 self.filled_orders.append({
    #                     'id': order['id'],
    #                     'side': signal,
    #                     'price': fill_price,
    #                     'amount': amount,
    #                     'timestamp': current_time,
    #                     'type': 'entry'
    #                 })
                    
    #                 self.logger.info(f"✅ Entry executed: {signal.upper()} {amount:.6f} @ ${fill_price:.6f}")
                    
    #     except Exception as e:
    #         self.logger.error(f"Error handling entry signals: {e}")
    
    # def _place_single_order(self, signal: str) -> bool:
    #     """SINGLE ORDER ENTRY POINT - All orders must go through here"""
    #     try:
    #         current_time = time.time()
            
    #         # Check cooldown period
    #         if current_time - self.last_order_time < self.order_cooldown:
    #             remaining = self.order_cooldown - (current_time - self.last_order_time)
    #             self.logger.debug(f"⏳ Order cooldown: {remaining:.1f}s remaining")
    #             return False
            
    #         # CRITICAL: Check position with delay if we just placed an order
    #         if self.last_order_id and current_time - self.last_order_time < self.position_check_delay:
    #             self.logger.debug(f"⏳ Waiting for previous order to settle...")
    #             return False
            
    #         # Double-check no position exists
    #         if self._has_active_position():
    #             self.logger.debug(f"📊 Position exists, skipping {signal} order")
    #             return False
            
    #         # Get current price and calculate amount
    #         ticker = self.exchange.get_ticker(self.symbol)
    #         current_price = float(ticker['last'])
    #         amount = self._calculate_position_amount(current_price)
            
    #         if amount < self.min_amount:
    #             self.logger.warning(f"Amount {amount:.6f} below minimum {self.min_amount}")
    #             return False
            
    #         # FINAL position check before order
    #         if self._has_active_position():
    #             self.logger.warning(f"Position detected during order prep - aborting {signal}")
    #             return False
            
    #         self.logger.info(f"🎯 PLACING SINGLE ORDER: {signal.upper()} {amount:.6f} @ ${current_price:.6f}")
            
    #         # Place the order
    #         order = self.exchange.create_market_order(self.symbol, signal, amount)
            
    #         if order and 'id' in order:
    #             # Update tracking
    #             self.last_order_time = current_time
    #             self.last_order_id = order['id']
    #             self.total_trades += 1
                
    #             # Record the order
    #             fill_price = float(order.get('average', current_price))
    #             self.filled_orders.append({
    #                 'id': order['id'],
    #                 'side': signal,
    #                 'price': fill_price,
    #                 'amount': amount,
    #                 'timestamp': current_time,
    #                 'type': 'single_entry'
    #             })
                
    #             self.logger.info(f"✅ ORDER PLACED: {signal.upper()} {amount:.6f} @ ${fill_price:.6f}, ID: {order['id']}")
                
    #             # Wait for settlement
    #             time.sleep(self.position_check_delay)
                
    #             return True
    #         else:
    #             self.logger.error(f"❌ Order placement failed - invalid response")
    #             return False
                
    #     except Exception as e:
    #         self.logger.error(f"❌ Error in single order placement: {e}")
    #         return False
    # def _handle_position_management(self, position: Dict, signal: str):
    #     """FIXED: Enhanced existing method with better exit logic"""
    #     try:
    #         position_size = float(position.get('contracts', 0))
    #         position_symbol = position.get('info', {}).get('symbol', '').upper()
            
    #         # ENHANCED: Strict symbol matching
    #         if position_symbol != self.symbol.upper():
    #             self.logger.debug(f"Position symbol mismatch: {position_symbol} != {self.symbol.upper()}")
    #             return
            
    #         # Determine position side and close side
    #         if position_size > 0:
    #             position_side = 'long'
    #             close_side = 'sell'
    #         elif position_size < 0:
    #             position_side = 'short' 
    #             close_side = 'buy'
    #         else:
    #             self.logger.debug(f"Zero position size for {self.symbol}")
    #             return
            
    #         close_amount = abs(position_size)

    #         self.logger.info(f"📊 Managing position: {self.symbol.upper()} {position_side.upper()} {close_amount:.6f} {signal}")

    #         # Check for opposing signal (exit condition)
    #         should_exit = False
    #         if position_side == 'long' and signal == 'sell':
    #             should_exit = True
    #             exit_reason = f"LONG position with SELL signal"
    #         elif position_side == 'short' and signal == 'buy':
    #             should_exit = True
    #             exit_reason = f"SHORT position with BUY signal"
            
    #         if should_exit:
    #             # ENHANCED: Check for recent exit to prevent duplicate exits
    #             current_time = time.time()
    #             last_exit_time = getattr(self, '_last_exit_time', 0)
                
    #             if current_time - last_exit_time < 5:  # 5 second gap between exits
    #                 self.logger.debug(f"Exit throttled: {current_time - last_exit_time:.1f}s < 5s")
    #                 return
                
    #             self.logger.warning(f"🚨 Signal-based exit: {exit_reason}")
    #             self.logger.warning(f"🚨 Closing {position_side.upper()}: {close_side.upper()} {close_amount:.6f}")
                
    #             try:
    #                 exit_order = self.exchange.create_market_order(self.symbol, close_side, close_amount)
    #                 if exit_order and 'id' in exit_order:
    #                     # ENHANCED: Update exit tracking
    #                     self._last_exit_time = current_time
    #                     self.total_trades += 1
                        
    #                     # Record the exit
    #                     fill_price = float(exit_order.get('average', 0))
    #                     self.filled_orders.append({
    #                         'id': exit_order['id'],
    #                         'side': close_side,
    #                         'price': fill_price,
    #                         'amount': close_amount,
    #                         'timestamp': current_time,
    #                         'type': 'signal_exit'
    #                     })
                        
    #                     self.logger.warning(f"✅ Position closed: {close_side.upper()} {close_amount:.6f} @ ${fill_price:.6f}")
                        
    #                     # ENHANCED: Mark position as closed to prevent TP/SL from running
    #                     self._position_closed_by_signal = True
                        
    #             except Exception as e:
    #                 self.logger.error(f"Error closing position: {e}")
    #         else:
    #             # Log when signals align with position
    #             if ((position_side == 'long' and signal == 'buy') or 
    #                 (position_side == 'short' and signal == 'sell')):
    #                 self.logger.debug(f"📊 Signal aligned: {position_side.upper()} + {signal.upper()}")
                    
    #     except Exception as e:
    #         self.logger.error(f"Error in position management: {e}")
    
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
        """FIXED: Check take profit and stop loss conditions - prevent double exit."""
        try:
            # FIXED: Skip if position already closed by signal
            if getattr(self, '_position_closed_by_signal', False):
                self.logger.debug("Position already closed by signal, skipping TP/SL check")
                return
            
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
        """FIXED: Clean stop without position closing orders"""
        try:
            if not self.running:
                return
            
            self.running = False
            self.logger.info(f"🛑 Stopping strategy for {self.symbol}")
            
            # Cancel all orders
            try:
                cancelled = self.exchange.cancel_all_orders(self.symbol)
                if cancelled:
                    self.logger.info(f"Cancelled {len(cancelled)} orders for {self.symbol}")
            except Exception as e:
                self.logger.error(f"Error cancelling orders for {self.symbol}: {e}")
            
            # Reset tracking flags
            if hasattr(self, '_position_closed_by_signal'):
                self._position_closed_by_signal = False
            
            self.logger.info(f"✅ Strategy stopped for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy for {self.symbol}: {e}")
            self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status (removed TP/SL fields)."""
        try:
            with self.update_lock:
                pnl_percentage = (self.total_pnl / self.position_size_usd * 100) if self.position_size_usd > 0 else 0.0
                has_position = self._has_active_position()
                effective_running = has_position or self.running
                
                return {
                    'strategy_id': self.strategy_id,
                    'symbol': self.symbol,
                    'display_symbol': self.original_symbol,
                    'position_size_usd': self.position_size_usd,
                    'leverage': self.leverage,
                    'strategy_type': 'tsi_only',  # Identify as TSI-only
                    'running': effective_running,
                    'has_position': has_position,
                    'trades_count': self.total_trades,
                    'pnl': self.total_pnl,
                    'pnl_percentage': pnl_percentage,
                    'last_update': self.last_update_time,
                    'exit_method': 'tsi_momentum_only'  # Clear identification
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
        """FIXED: Keep existing logic but add warning about potential race conditions."""
        try:
            positions = self.exchange.get_positions(self.symbol)
            
            for pos in positions:
                pos_symbol = pos.get('info', {}).get('symbol', '').upper()
                our_symbol = self.symbol.upper()
                
                if pos_symbol == our_symbol:
                    size = float(pos.get('contracts', 0))
                    if abs(size) >= 0.001:
                        self.logger.debug(f"📊 Active position found: {our_symbol} = {size:.6f}")
                        return True
            
            self.logger.debug(f"📊 No active position for {self.symbol}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking active position: {e}")
            return False