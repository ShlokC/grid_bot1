"""
Simplified Signal Trading Strategy - TSI-based trading with fixed position sizing.
Replaces the misleading "Grid" implementation with clear signal-based trading.
"""

import logging
import time
import threading
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional
from core.exchange import Exchange
from core.adaptive_crypto_signals import integrate_adaptive_crypto_signals


class SignalStrategy:
    """
    TSI Signal-based Trading Strategy with fixed position sizing.
    Simplified replacement for the misleading GridStrategy.
    """
    
    def __init__(self, 
                 exchange: Exchange, 
                 symbol: str,
                 strategy_id: str,
                 position_size_usd: float = 0.5,
                 leverage: float = 20.0,
                 strategy_type: str = 'roc_multi_timeframe',
                 enable_llm: bool = True):  # Add this parameter
        """Initialize signal strategy with LLM support."""
        
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.enable_llm = enable_llm  # Store LLM setting
        
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
        self._last_order_time = 0
        self._order_attempts = []
        
        # self.logger.info(f"✅ Centralized order management initialized for {symbol}")
        
        # Integrate signal system with strategy type and LLM setting
        integrate_adaptive_crypto_signals(self, strategy_type=strategy_type, enable_llm=enable_llm)
        
        llm_status = "with LLM fusion" if enable_llm else "traditional only"
        # self.logger.info(f"Signal Strategy initialized: {symbol}, Size: ${position_size_usd}, "
        #                 f"Leverage: {leverage}x, Strategy: {strategy_type}, {llm_status}")
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
            
            # self.logger.info(f"🚀 Starting Signal Strategy: {self.symbol}")
            
            # Cancel any existing orders but don't place new ones
            try:
                cancelled = self.exchange.cancel_all_orders(self.symbol)
                # if cancelled:
                    # self.logger.info(f"Cancelled {len(cancelled)} existing orders")
                time.sleep(1)
            except Exception as e:
                self.logger.warning(f"Failed to cancel existing orders: {e}")
            
            self.running = True
            # self.logger.info(f"✅ Signal Strategy started: {self.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting strategy: {e}")
            return False
    def test_duplicate_prevention(self):
        """Test method to verify no duplicates can occur."""
        # self.logger.info("🧪 Testing duplicate prevention...")
        
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
        # self.logger.info(f"🧪 Test complete: {len(order_calls)} orders would have been placed")
        
        if len(order_calls) <= 1:
            # self.logger.info("✅ DUPLICATE PREVENTION WORKING: Maximum 1 order per test")
            return True
        else:
            self.logger.error(f"❌ DUPLICATES DETECTED: {len(order_calls)} orders in rapid test")
            return False
    def test_trading_setup(self) -> bool:
        """Test trading setup for this strategy's symbol."""
        try:
            # self.logger.info(f"🧪 Testing trading setup for {self.symbol}...")
            
            # FIXED: Call setup on exchange instance, not self
            success = self.exchange.setup_symbol_trading_config(self.symbol, 20)
            
            if success:
                # Get current config to verify
                config = self.exchange.get_symbol_trading_config(self.symbol)
                
                # self.logger.info(f"📊 Current config for {self.symbol}:")
                # self.logger.info(f"   Margin Type: {config.get('margin_type', 'UNKNOWN')}")
                # self.logger.info(f"   Leverage: {config.get('leverage', 'UNKNOWN')}x")
                # self.logger.info(f"   Position Size: {config.get('position_size', 0)}")
                
                # Verify requirements
                margin_ok = config.get('margin_type') == 'ISOLATED'
                leverage_ok = config.get('leverage') == 20
                
                if margin_ok and leverage_ok:
                    # self.logger.info(f"✅ {self.symbol} correctly configured for trading")
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
        """FIXED: Single API call to get ALL required data using adaptive crypto signals."""
        try:
            # Get all data in one batch to prevent inconsistencies
            positions = self.exchange.get_positions(self.symbol)
            open_orders = self.exchange.get_open_orders(self.symbol)
            ticker = self.exchange.get_ticker(self.symbol)
            
            # FIXED: Use adaptive crypto signals system for technical direction
            tech_signal = self.get_technical_direction(self.exchange)
            
            # Parse position data
            current_position = None
            position_size = 0.0
            position_side = None
            entry_price = 0.0
            
            for pos in positions:
                pos_symbol = pos.get('info', {}).get('symbol', '').upper()
                if pos_symbol == self.symbol.upper():
                    position_side = pos.get('side', '') 
                    size = float(pos.get('contracts', 0))
                    if abs(size) >= 0.001:
                        current_position = pos
                        position_size = size
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
        """FIXED: Always check exit when position exists, using adaptive signals consistently."""
        try:
            entry_signal = live_data['tech_signal']
            has_position = live_data['has_position']
            position_side = live_data.get('position_side')
            position_entry_price = live_data.get('entry_price')
            current_price = live_data.get('current_price')
            pending_orders = live_data['pending_orders']
            
            decision = {'action': 'none', 'side': '', 'reason': ''}
            
            # FIXED: ALWAYS check exit FIRST when position exists - using adaptive_signals consistently
            if has_position and hasattr(self, 'adaptive_signals'):
                exit_decision = self.adaptive_signals.evaluate_exit_conditions(
                    position_side=position_side,
                    entry_price=position_entry_price,
                    current_price=current_price
                )
                
                if exit_decision.get('should_exit', False):
                    decision.update({
                        'action': 'exit',
                        'side': 'sell' if position_side == 'long' else 'buy',
                        'reason': exit_decision.get('exit_reason', 'Exit signal')
                    })
                    return decision
            
            # SIMPLE CHECK: No NEW orders if already pending (only applies to entries)
            if pending_orders > 0:
                decision['reason'] = f'{pending_orders} orders pending'
                return decision
            
            # SIMPLE ENTRY CHECK: No position, check entry
            if not has_position and entry_signal in ['buy', 'sell']:
                decision.update({
                    'action': 'entry',
                    'side': entry_signal,
                    'reason': f'{entry_signal.upper()} signal'
                })
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in simplified decision: {e}")
            return {'action': 'none', 'side': '', 'reason': f'Error: {e}'}


    def _execute_single_order(self, decision: Dict, live_data: Dict):
        """FIXED: Single order execution point using adaptive signals consistently."""
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
                if amount == 0:
                    self.logger.warning(f"[{self.symbol}] Attempting to exit with zero position size. Aborting exit.")
                    return

            if amount < self.min_amount:
                self.logger.warning(f"[{self.symbol}] Amount {amount:.6f} below minimum {self.min_amount}. Order not placed for {action} {side}.")
                return
            
            self.logger.info(f"[{self.symbol}] 🎯 ORDER DECISION TO EXECUTE: {action.upper()} {side.upper()} {amount:.6f} @ ${current_price:.6f}")
            self.logger.info(f"[{self.symbol}] 🎯 Reason: {reason}")
            
            order = self.exchange.create_market_order(self.symbol, side, amount)
            
            if order and 'id' in order:
                self._last_order_time = live_data['timestamp']
                self.total_trades += 1
                
                fill_price = float(order.get('average', current_price))
                self.filled_orders.append({
                    'id': order['id'], 'side': side, 'price': fill_price,
                    'amount': amount, 'timestamp': live_data['timestamp'], 'type': action
                })
                
                self.logger.info(f"[{self.symbol}] ✅ {action.upper()} EXECUTED: {side.upper()} {amount:.6f} @ ${fill_price:.6f}. Order ID: {order['id']}")

                # FIXED: Use adaptive_signals consistently for position tracking
                if action == 'entry' and hasattr(self, 'adaptive_signals'):
                    self.adaptive_signals.position_entry_time = live_data['timestamp']
                    self.logger.debug(f"[{self.symbol}] Set position_entry_time on adaptive system to: {live_data['timestamp']}")
                elif action == 'exit' and hasattr(self, 'adaptive_signals'):
                    self.adaptive_signals.position_entry_time = 0
                    self.logger.debug(f"[{self.symbol}] Reset position_entry_time on adaptive system after exit.")
                    
        except Exception as e:
            self.logger.error(f"[{self.symbol}] Error executing order: {e}")
    def _round_price(self, price: float) -> float:
        """Round price to appropriate precision"""
        if price <= 0:
            return 0.0
        
        precision = self.price_precision
        return float(f"{price:.{precision}f}")
    
    def _round_amount(self, amount: float) -> float:
        """Round amount to appropriate precision"""
        if amount <= 0:
            return 0.0
        
        precision = self.amount_precision
        return float(f"{amount:.{precision}f}")
    # def _get_technical_direction(self) -> str:
    #     """FIXED: Fast-response TSI for small crypto momentum"""
    #     direction = 'none'
        
    #     try:
    #         # Get OHLCV data - FIXED: Use 3m for faster signals
    #         ohlcv_data = self.exchange.get_ohlcv(self.symbol, timeframe='3m', limit=100)
            
    #         if not ohlcv_data or len(ohlcv_data) < 50:
    #             self.logger.warning("Insufficient OHLCV data for TSI analysis")
    #             return 'none'
            
    #         # Convert to DataFrame
    #         df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    #         df['close'] = df['close'].astype(float)
            
    #         # FIXED: Faster TSI parameters for small crypto momentum
    #         tsi_result = ta.tsi(df['close'], slow=15, fast=8, signal=6)
            
    #         if tsi_result is None or len(tsi_result.dropna()) < 10:
    #             self.logger.warning("Failed to calculate TSI or insufficient data")
    #             return 'none'
            
    #         # Clean TSI data
    #         tsi_clean = tsi_result.dropna()
            
    #         if len(tsi_clean.columns) >= 2:
    #             tsi_line = tsi_clean.iloc[:, 0]
    #             tsi_signal = tsi_clean.iloc[:, 1]
    #         else:
    #             self.logger.warning("TSI result doesn't have expected columns")
    #             return 'none'
            
    #         if len(tsi_line) < 8:
    #             return 'none'
            
    #         # Get current values
    #         current_price = float(df['close'].iloc[-1])
    #         latest_tsi = tsi_line.iloc[-1]
    #         latest_tsi_signal = tsi_signal.iloc[-1]
    #         prev_tsi = tsi_line.iloc[-2]
            
    #         # FIXED: More aggressive thresholds for small crypto
    #         # Standard TSI levels: +25 overbought, -25 oversold
    #         # Adjusted for small crypto volatility
    #         OVERBOUGHT_LEVEL = 20.0
    #         OVERSOLD_LEVEL = -20.0
            
    #         # FIXED: Simple, fast momentum detection
    #         tsi_momentum = latest_tsi - prev_tsi
    #         tsi_crossover = latest_tsi - latest_tsi_signal
    #         prev_crossover = tsi_line.iloc[-2] - tsi_signal.iloc[-2]
            
    #         # FIXED: Primary signal = crossover (remove double conditions)
    #         bullish_crossover = (latest_tsi > latest_tsi_signal and prev_crossover <= 0)
    #         bearish_crossover = (latest_tsi < latest_tsi_signal and prev_crossover >= 0)
            
    #         # FIXED: Momentum confirmation (not requirement)
    #         momentum_bullish = tsi_momentum > 0
    #         momentum_bearish = tsi_momentum < 0
            
    #         # FIXED: Dynamic strength based on recent volatility (not median)
    #         recent_tsi = tsi_line.tail(10)  # Shorter window for small crypto
    #         tsi_range = recent_tsi.max() - recent_tsi.min()
    #         min_strength = max(2.0, tsi_range * 0.3)  # Minimum strength threshold
            
    #         current_strength = abs(latest_tsi)
    #         has_strength = current_strength >= min_strength
            
    #         # FIXED: Much shorter persistence for small crypto (15-30 seconds max)
    #         last_signal = getattr(self, '_last_signal', 'none')
    #         last_signal_time = getattr(self, '_last_signal_time', 0)
    #         current_time = time.time()
            
    #         # FIXED: Adaptive persistence based on market conditions
    #         if latest_tsi > OVERBOUGHT_LEVEL or latest_tsi < OVERSOLD_LEVEL:
    #             # Near extremes: allow faster reversals
    #             min_persistence = 10  # 10 seconds
    #         else:
    #             # Normal range: slightly longer
    #             min_persistence = 20  # 20 seconds
            
    #         signal_too_recent = (current_time - last_signal_time) < min_persistence
            
    #         # FIXED: Overbought/Oversold Protection
    #         at_overbought = latest_tsi >= OVERBOUGHT_LEVEL
    #         at_oversold = latest_tsi <= OVERSOLD_LEVEL
            
    #         # SIGNAL LOGIC - FIXED: Fast and responsive
            
    #         # 1. STRONG BUY: Oversold bounce with bullish momentum
    #         if (at_oversold and momentum_bullish and latest_tsi > latest_tsi_signal):
    #             if last_signal != 'buy' or not signal_too_recent:
    #                 direction = 'buy'
    #                 self._last_signal = 'buy'
    #                 self._last_signal_time = current_time
    #                 self.logger.info(f"🚀 OVERSOLD BOUNCE BUY: TSI={latest_tsi:.2f} Price=${current_price:.6f}")
                    
    #         # 2. STRONG SELL: Overbought reversal with bearish momentum  
    #         elif (at_overbought and momentum_bearish and latest_tsi < latest_tsi_signal):
    #             if last_signal != 'sell' or not signal_too_recent:
    #                 direction = 'sell'
    #                 self._last_signal = 'sell'
    #                 self._last_signal_time = current_time
    #                 self.logger.info(f"📉 OVERBOUGHT REVERSAL SELL: TSI={latest_tsi:.2f} Price=${current_price:.6f}")
                    
    #         # 3. BULLISH CROSSOVER: Fresh momentum up
    #         elif (bullish_crossover and has_strength and not at_overbought):
    #             if last_signal != 'buy' or not signal_too_recent:
    #                 direction = 'buy'
    #                 self._last_signal = 'buy'
    #                 self._last_signal_time = current_time
    #                 self.logger.info(f"📈 BULLISH CROSS BUY: TSI={latest_tsi:.2f}>{latest_tsi_signal:.2f} Price=${current_price:.6f}")
                    
    #         # 4. BEARISH CROSSOVER: Fresh momentum down
    #         elif (bearish_crossover and has_strength and not at_oversold):
    #             if last_signal != 'sell' or not signal_too_recent:
    #                 direction = 'sell'
    #                 self._last_signal = 'sell'
    #                 self._last_signal_time = current_time
    #                 self.logger.info(f"📉 BEARISH CROSS SELL: TSI={latest_tsi:.2f}<{latest_tsi_signal:.2f} Price=${current_price:.6f}")
                    
    #         # 5. MOMENTUM CONTINUATION: Strong trending
    #         elif (latest_tsi > latest_tsi_signal and momentum_bullish and current_strength > min_strength * 1.5):
    #             if last_signal != 'buy' or not signal_too_recent:
    #                 direction = 'buy'
    #                 self._last_signal = 'buy'
    #                 self._last_signal_time = current_time
    #                 self.logger.info(f"⚡ MOMENTUM BUY: TSI={latest_tsi:.2f} Strength={current_strength:.2f}")
                    
    #         elif (latest_tsi < latest_tsi_signal and momentum_bearish and current_strength > min_strength * 1.5):
    #             if last_signal != 'sell' or not signal_too_recent:
    #                 direction = 'sell'
    #                 self._last_signal = 'sell'
    #                 self._last_signal_time = current_time
    #                 self.logger.info(f"⚡ MOMENTUM SELL: TSI={latest_tsi:.2f} Strength={current_strength:.2f}")
            
    #         # 6. HOLD PREVIOUS SIGNAL: No clear new direction
    #         else:
    #             direction = 'none'
    #             self.logger.debug(f"⚠️ NO CLEAR SIGNAL: TSI={latest_tsi:.2f} Signal={latest_tsi_signal:.2f} Momentum={tsi_momentum:.2f}")
                
    #     except Exception as e:
    #         self.logger.error(f"Error in FIXED TSI analysis: {e}")
    #         direction = 'none'
        
    #     return direction
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
            # self.logger.info(f"🛑 Stopping strategy for {self.symbol}")
            
            # Cancel all orders
            try:
                cancelled = self.exchange.cancel_all_orders(self.symbol)
                #if cancelled:
                    # self.logger.info(f"Cancelled {len(cancelled)} orders for {self.symbol}")
            except Exception as e:
                self.logger.error(f"Error cancelling orders for {self.symbol}: {e}")
            
            # Reset tracking flags
            if hasattr(self, '_position_closed_by_signal'):
                self._position_closed_by_signal = False
            
            # self.logger.info(f"✅ Strategy stopped for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy for {self.symbol}: {e}")
            self.running = False
    
    # Update this method in core/signal_strategy.py

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status with strategy type information."""
        try:
            with self.update_lock:
                pnl_percentage = (self.total_pnl / self.position_size_usd * 100) if self.position_size_usd > 0 else 0.0
                has_position = self._has_active_position()
                effective_running = has_position or self.running
                
                # Get strategy display name (updated with ROC multi-timeframe)
                strategy_names = {
                    'qqe_supertrend_fixed': 'QQE+ST (Current)',
                    'qqe_supertrend_fast': 'QQE+ST (Fast)', 
                    'rsi_macd': 'RSI+MACD',
                    'tsi_vwap': 'TSI+VWAP',
                    'roc_multi_timeframe': 'ROC Multi-TF'
                }
                strategy_display = strategy_names.get(self.strategy_type, self.strategy_type)
                
                return {
                    'strategy_id': self.strategy_id,
                    'symbol': self.symbol,
                    'display_symbol': self.original_symbol,
                    'position_size_usd': self.position_size_usd,
                    'leverage': self.leverage,
                    'strategy_type': self.strategy_type,
                    'strategy_display': strategy_display,  # For UI display
                    'running': effective_running,
                    'has_position': has_position,
                    'trades_count': self.total_trades,
                    'pnl': self.total_pnl,
                    'pnl_percentage': pnl_percentage,
                    'last_update': self.last_update_time,
                    'exit_method': f'{self.strategy_type}_signals'
                }
                
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {
                'strategy_id': self.strategy_id,
                'symbol': self.symbol,
                'strategy_type': getattr(self, 'strategy_type', 'unknown'),
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