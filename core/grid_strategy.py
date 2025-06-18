"""
Simplified Grid Trading Strategy - Places buy/sell orders at grid intervals.
Enhanced with 2% stop-loss orders when positions exist.
"""

import logging
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from collections import deque
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta

from core.exchange import Exchange


class GridStrategy:
    """
    Simplified Grid Trading Strategy with Stop-Loss Orders
    
    Places buy orders below current price and sell orders above current price
    at calculated grid intervals. When orders fill, places counter orders.
    Enhanced with 2% stop-loss orders when positions are detected.
    """
    
    def __init__(self, 
                 exchange: Exchange, 
                 symbol: str, 
                 grid_number: int,
                 investment: float,
                 take_profit_pnl: float,
                 stop_loss_pnl: float,
                 grid_id: str,
                 leverage: float = 20.0,
                 enable_grid_adaptation: bool = True,
                 enable_samig: bool = False):
        """Initialize simplified grid strategy."""
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Exchange and symbol
        self.exchange = exchange
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        self.placed_order_ids: set = set()
        
        # ENHANCEMENT: Track stop-loss orders separately
        self.stop_loss_order_ids: set = set()
        self.stop_loss_percentage = 0.02  # 2% stop-loss
        
        # Fetch market information
        self._fetch_market_info()
        
        # Calculate price range using current price
        try:
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # Use ±10% range around current price
            range_pct = 0.10  # 10%
            self.user_price_lower = self._round_price(current_price * (1 - range_pct))
            self.user_price_upper = self._round_price(current_price * (1 + range_pct))
            
            self.logger.info(f"Price range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f} (±{range_pct*100}% around ${current_price:.6f})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize price range: {e}")
            # Emergency fallback
            self.user_price_lower = 1.0
            self.user_price_upper = 100.0
        
        # User parameters
        self.user_grid_number = int(grid_number)
        self.user_total_investment = float(investment)
        self.user_investment_per_grid = self.user_total_investment / self.user_grid_number
        self.user_leverage = float(leverage)
        
        # Strategy settings
        self.take_profit_pnl = float(take_profit_pnl)
        self.stop_loss_pnl = float(stop_loss_pnl)
        self.grid_id = grid_id
        self.enable_grid_adaptation = enable_grid_adaptation
        
        # FIXED: Tight spacing for ping-pong orders (configurable)
        self.ping_pong_spacing_pct = 0.01  # 1% default spacing

        # Core tracking - simplified
        self.grid_orders: Dict[str, Dict] = {}  # Track grid orders
        self.filled_orders: List[Dict] = []     # Track filled orders for PnL
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # State management
        self.running = False
        self.last_update_time = 0
        
        # Threading
        self.update_lock = threading.Lock()
        
        self.logger.info(f"Simplified Grid initialized: {self.original_symbol}, "
                        f"Range: ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}, "
                        f"Levels: {self.user_grid_number}, Investment: ${self.user_total_investment:.2f}, "
                        f"Stop-Loss: {self.stop_loss_percentage*100}%")
        
    def _fetch_market_info(self):
        """Fetch market precision and limits from exchange"""
        try:
            market_info = self.exchange.get_market_info(self.symbol)
            
            # Extract precision information
            precision_info = market_info.get('precision', {})
            self.price_precision = int(precision_info.get('price', 6))
            self.amount_precision = int(precision_info.get('amount', 6))
            
            # Extract limits
            limits = market_info.get('limits', {})
            amount_limits = limits.get('amount', {})
            cost_limits = limits.get('cost', {})
            
            self.min_amount = float(amount_limits.get('min', 0.0001))
            self.max_amount = float(amount_limits.get('max', 1000000))
            self.min_cost = float(cost_limits.get('min', 1.0))
            self.max_cost = float(cost_limits.get('max', 1000000))
            
        except Exception as e:
            self.logger.error(f"Error fetching market info: {e}")
            # Set reasonable defaults
            self.price_precision = 6
            self.amount_precision = 6
            self.min_amount = 0.0001
            self.max_amount = 1000000
            self.min_cost = 1.0
            self.max_cost = 1000000
    
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
    
    def _calculate_order_amount(self, price: float) -> float:
        """Calculate order amount for given price"""
        try:
            if price <= 0:
                return self.min_amount
            
            # Calculate notional value from investment per grid
            notional_value = self.user_investment_per_grid * self.user_leverage
            
            # Calculate base quantity
            quantity = notional_value / price
            rounded_amount = self._round_amount(quantity)
            
            # Ensure minimum amount
            if rounded_amount < self.min_amount:
                rounded_amount = self.min_amount
            
            # Ensure maximum amount
            if rounded_amount > self.max_amount:
                rounded_amount = self.max_amount
            
            # Check minimum notional value
            order_notional = price * rounded_amount
            if order_notional < self.min_cost:
                rounded_amount = self.min_cost / price
                rounded_amount = self._round_amount(rounded_amount)
            
            return max(self.min_amount, rounded_amount)
            
        except Exception as e:
            self.logger.error(f"Error calculating order amount for price ${price:.6f}: {e}")
            return self.min_amount
    
    def _calculate_grid_levels(self) -> List[float]:
        """Calculate grid price levels"""
        try:
            if self.user_grid_number <= 0:
                return []
            
            price_range = self.user_price_upper - self.user_price_lower
            if price_range <= 0:
                self.logger.error(f"Invalid price range: {self.user_price_lower} - {self.user_price_upper}")
                return []
            
            # Calculate equal intervals
            interval = price_range / self.user_grid_number
            levels = []
            
            for i in range(self.user_grid_number + 1):
                level = self.user_price_lower + (i * interval)
                levels.append(self._round_price(level))
            
            # Remove duplicates and sort
            levels = sorted(list(set(levels)))
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating grid levels: {e}")
            return []
    
    def setup_grid(self):
        """Setup initial grid orders (2 orders: 1 buy, 1 sell)"""
        try:
            # Get current market price
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            self.logger.info(f"🚀 SETTING UP PING-PONG GRID: Current price ${current_price:.6f}")
            self.logger.info(f"📊 Strategy: 2 orders total (1 BUY below, 1 SELL above current price)")
            
            # Cancel existing orders for safety
            try:
                cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                if cancelled_orders:
                    self.logger.info(f"Cancelled {len(cancelled_orders)} existing orders")
                time.sleep(1)  # Wait for cancellations
            except Exception as e:
                self.logger.warning(f"Failed to cancel existing orders: {e}")
            
            # Calculate grid levels
            grid_levels = self._calculate_grid_levels()
            if not grid_levels:
                self.logger.error("Failed to calculate grid levels")
                return False
            
            # Place initial orders (ping-pong strategy)
            orders_placed = self._place_initial_grid_orders(current_price, grid_levels)
            
            # FIXED: Allow grid to start even with 0 orders (for mixed signals)
            # The grid will wait for clear signals and place orders during monitoring
            self.running = True
            
            if orders_placed >= 1:
                self.logger.info(f"✅ PING-PONG GRID SETUP COMPLETE: {orders_placed} orders placed")
                if orders_placed == 2:
                    self.logger.info(f"🎯 Perfect setup: Both BUY and SELL orders active")
                else:
                    self.logger.warning(f"⚠️ Partial setup: Only {orders_placed} order(s) placed due to insufficient margin")
            else:
                # FIXED: Don't treat 0 orders as failure - grid can start and wait for signals
                self.logger.info(f"✅ GRID STARTED IN STANDBY MODE: No initial orders placed")
                self.logger.info(f"📊 Grid will monitor and place orders when clear signals appear")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR in grid setup: {e}")
            return False
    
    def _get_technical_direction(self) -> str:
        """Enhanced momentum signal with multiple confirmation layers"""
        # FIXED: Default to 'none' instead of 'buy'
        direction = 'none'  # Safe default - no trade
        
        try:
            # Get OHLCV data (keep 3m but add more confirmations)
            ohlcv_data = self.exchange.get_ohlcv(self.symbol, timeframe='3m', limit=100)
            
            if not ohlcv_data or len(ohlcv_data) < 30:
                self.logger.warning("Insufficient OHLCV data for technical analysis")
                return 'none'  # FIXED: Return 'none' instead of 'buy'
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close'] = df['close'].astype(float)
            
            # Calculate indicators - OPTIMIZED for small crypto momentum
            jma = ta.jma(df['close'], length=12, phase=50, pow=1.5)  # Faster response, less smoothing
            kama = ta.kama(df['close'], length=12)  # Aligned period, faster adaptation
            rsi = ta.rsi(df['close'], length=14)  # Keep standard RSI
            
            # Add MACD for momentum confirmation - FIXED: Handle both pandas and numpy returns
            try:
                macd_result = ta.macd(df['close'], fast=8, slow=21, signal=9)
                if hasattr(macd_result, 'iloc'):
                    # pandas DataFrame result
                    macd_hist = macd_result.iloc[:, 2]  # Get histogram column
                else:
                    # numpy array result - convert to pandas Series
                    macd_hist = pd.Series(macd_result[2], index=df.index)
            except Exception as e:
                self.logger.warning(f"MACD calculation failed: {e}, using RSI only")
                macd_hist = None
            
            if jma is None or kama is None or rsi is None:
                self.logger.warning("Failed to calculate core technical indicators")
                return 'none'  # FIXED: Return 'none' for failed calculations
            
            # Clean and get latest values
            jma_clean = jma.dropna()
            kama_clean = kama.dropna()
            rsi_clean = rsi.dropna()
            
            # Handle MACD (might be None if calculation failed)
            if macd_hist is not None:
                macd_hist_clean = macd_hist.dropna()
                macd_available = len(macd_hist_clean) >= 3
            else:
                macd_hist_clean = None
                macd_available = False
            
            if len(jma_clean) < 5 or len(kama_clean) < 5 or len(rsi_clean) < 3:
                self.logger.warning("Insufficient cleaned indicator data")
                return 'none'  # FIXED: Return 'none' for insufficient data
            
            current_price = float(df['close'].iloc[-1])
            latest_jma = jma_clean.iloc[-1]
            latest_kama = kama_clean.iloc[-1]
            latest_rsi = rsi_clean.iloc[-1]
            
            # Handle MACD values (if available)
            if macd_available:
                latest_macd_hist = macd_hist_clean.iloc[-1]
                prev_macd_hist = macd_hist_clean.iloc[-2]
            else:
                latest_macd_hist = 0
                prev_macd_hist = 0
            
            # ENHANCEMENT 1: MA Direction Filter - More sensitive for small cryptos
            jma_rising = latest_jma > jma_clean.iloc[-4]  # Check over 4 periods for trend
            kama_rising = latest_kama > kama_clean.iloc[-4]  # More periods for confirmation
            
            # ENHANCEMENT 2: MA Separation Filter - Adjusted for small crypto volatility
            ma_separation = abs(latest_jma - latest_kama) / current_price
            min_separation = 0.0015  # Reduced to 0.15% for small crypto sensitivity
            mas_aligned = ma_separation < min_separation
            
            # ENHANCEMENT 3: Price Position with stronger momentum requirement
            price_above_jma = current_price > latest_jma
            price_above_kama = current_price > latest_kama
            price_strongly_above = current_price > max(latest_jma, latest_kama) * 1.002  # 0.2% above both
            price_strongly_below = current_price < min(latest_jma, latest_kama) * 0.998  # 0.2% below both
            
            # ENHANCEMENT 4: Multi-layer momentum confirmation
            rsi_momentum_up = latest_rsi > 55 and latest_rsi > rsi_clean.iloc[-2]  # Stronger RSI requirement
            rsi_momentum_down = latest_rsi < 45 and latest_rsi < rsi_clean.iloc[-2]
            
            # MACD momentum confirmation (if available)
            if macd_available:
                macd_momentum_up = latest_macd_hist > 0 and latest_macd_hist > prev_macd_hist
                macd_momentum_down = latest_macd_hist < 0 and latest_macd_hist < prev_macd_hist
            else:
                # Fallback to RSI-based momentum if MACD unavailable
                macd_momentum_up = rsi_momentum_up
                macd_momentum_down = rsi_momentum_down
            
            # ENHANCEMENT 5: Adaptive Rate of Change - Based on recent volatility
            price_roc_3 = (current_price - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100
            recent_volatility = df['close'].rolling(10).std().iloc[-1] / current_price * 100
            
            # Dynamic momentum threshold based on volatility
            base_threshold = 0.25
            momentum_threshold = max(base_threshold, recent_volatility * 0.5)  # Adaptive threshold
            
            strong_momentum_up = price_roc_3 > momentum_threshold
            strong_momentum_down = price_roc_3 < -momentum_threshold
            
            # ENHANCEMENT 6: Signal Persistence (prevent immediate flips)
            last_signal = getattr(self, '_last_signal', 'none')
            last_signal_time = getattr(self, '_last_signal_time', 0)
            current_time = time.time()
            
            # Don't flip signals too quickly (minimum 2 minutes)
            signal_too_recent = (current_time - last_signal_time) < 120
            
            # ENHANCED SIGNAL LOGIC - Optimized for sustained momentum moves
            
            # STRONG BULLISH: All primary conditions + momentum confirmation
            strong_bullish_conditions = [
                price_strongly_above,                      # Price clearly above both MAs
                jma_rising and kama_rising,                # Both MAs trending up strongly
                not mas_aligned,                           # MAs have clear separation
                rsi_momentum_up,                           # RSI showing upward momentum
                macd_momentum_up,                          # MACD confirming momentum
                strong_momentum_up,                        # Price has strong momentum
            ]
            
            # MODERATE BULLISH: Relaxed conditions for continuation
            moderate_bullish_conditions = [
                price_above_jma and price_above_kama,      # Price above both MAs
                jma_rising or kama_rising,                 # At least one MA trending up
                rsi_momentum_up or macd_momentum_up,       # At least one momentum indicator
                price_roc_3 > momentum_threshold * 0.6,    # Moderate momentum
            ]
            
            # STRONG BEARISH: All primary conditions + momentum confirmation  
            strong_bearish_conditions = [
                price_strongly_below,                      # Price clearly below both MAs
                not jma_rising and not kama_rising,        # Both MAs trending down
                not mas_aligned,                           # MAs have clear separation
                rsi_momentum_down,                         # RSI showing downward momentum
                macd_momentum_down,                        # MACD confirming momentum
                strong_momentum_down,                      # Price has strong momentum
            ]
            
            # MODERATE BEARISH: Relaxed conditions for continuation
            moderate_bearish_conditions = [
                not price_above_jma and not price_above_kama,  # Price below both MAs
                not jma_rising or not kama_rising,              # At least one MA trending down
                rsi_momentum_down or macd_momentum_down,        # At least one momentum indicator
                price_roc_3 < -momentum_threshold * 0.6,        # Moderate momentum
            ]
            
            # SIGNAL DECISION - Prioritize strong signals, allow moderate for continuation
            if all(strong_bullish_conditions):
                if last_signal != 'buy' or not signal_too_recent:
                    direction = 'buy'
                    self._last_signal = 'buy'
                    self._last_signal_time = current_time
                    self.logger.info(f"📈 STRONG BUY: Price: ${current_price:.6f}, JMA: ${latest_jma:.6f}, KAMA: ${latest_kama:.6f}")
                    if macd_available:
                        self.logger.info(f"📈 Momentum: RSI: {latest_rsi:.1f}, MACD: {latest_macd_hist:.6f}, ROC: {price_roc_3:.2f}%")
                    else:
                        self.logger.info(f"📈 Momentum: RSI: {latest_rsi:.1f}, ROC: {price_roc_3:.2f}% (MACD unavailable)")
                else:
                    direction = last_signal
                    
            elif all(moderate_bullish_conditions) and last_signal == 'buy':
                # Continue bullish if already in position and moderate conditions met
                direction = 'buy'
                self.logger.debug(f"📈 MODERATE BUY (continuation): Keeping bullish bias")
                
            elif all(strong_bearish_conditions):
                if last_signal != 'sell' or not signal_too_recent:
                    direction = 'sell'
                    self._last_signal = 'sell'
                    self._last_signal_time = current_time
                    self.logger.info(f"📉 STRONG SELL: Price: ${current_price:.6f}, JMA: ${latest_jma:.6f}, KAMA: ${latest_kama:.6f}")
                    if macd_available:
                        self.logger.info(f"📉 Momentum: RSI: {latest_rsi:.1f}, MACD: {latest_macd_hist:.6f}, ROC: {price_roc_3:.2f}%")
                    else:
                        self.logger.info(f"📉 Momentum: RSI: {latest_rsi:.1f}, ROC: {price_roc_3:.2f}% (MACD unavailable)")
                else:
                    direction = last_signal
                    
            elif all(moderate_bearish_conditions) and last_signal == 'sell':
                # Continue bearish if already in position and moderate conditions met
                direction = 'sell'
                self.logger.debug(f"📉 MODERATE SELL (continuation): Keeping bearish bias")
                
            else:
                # NO CLEAR SIGNAL - prevents whipsaws in unclear conditions
                direction = 'none'
                
                # Enhanced debug logging
                failed_conditions = []
                if mas_aligned:
                    failed_conditions.append(f"MAs too close ({ma_separation*100:.3f}%)")
                if not strong_momentum_up and not strong_momentum_down:
                    failed_conditions.append(f"Weak momentum (ROC: {price_roc_3:.2f}%, threshold: ±{momentum_threshold:.2f}%)")
                if not (rsi_momentum_up or rsi_momentum_down):
                    failed_conditions.append(f"RSI unclear ({latest_rsi:.1f})")
                if macd_available and not (macd_momentum_up or macd_momentum_down):
                    failed_conditions.append(f"MACD flat ({latest_macd_hist:.6f})")
                elif not macd_available:
                    failed_conditions.append("MACD unavailable")
                    
                if failed_conditions:
                    self.logger.debug(f"⚠️ NO SIGNAL: {', '.join(failed_conditions[:2])}")  # Show first 2 reasons
            
        except Exception as e:
            self.logger.error(f"Error in enhanced technical analysis: {e}")
            direction = 'none'  # FIXED: Return 'none' on exception, not 'buy'
        
        return direction

    def _ensure_ping_pong_orders(self):
        """FIXED: Pass already-detected entry price to stop-loss method"""
        try:
            current_time = time.time()
            last_ensure_time = getattr(self, '_last_ensure_orders_time', 0)
            if current_time - last_ensure_time < 30:
                return
            
            # Get live data from exchange
            live_positions = self.exchange.get_positions(self.symbol)
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # Check current positions
            has_position = False
            position_size = 0.0
            entry_price = current_price
            position_side = None
            
            for pos in live_positions:
                pos_symbol = pos.get('info', {}).get('symbol', '')
                if pos_symbol != self.symbol:
                    continue
                size = float(pos.get('contracts', 0))
                if abs(size) >= 0.001:  # Has meaningful position
                    has_position = True
                    position_size = size
                    position_side = pos.get('side', '').lower()
                    entry_price = float(pos.get('entryPrice', current_price))
                    break
            
            # Count current orders BY TYPE for THIS SYMBOL
            symbol_orders = [o for o in live_orders if o.get('info', {}).get('symbol', '') == self.symbol]
            has_stop_orders = any('STOP' in o.get('type', '').upper() for o in symbol_orders)
            has_tp_orders = any('TAKE_PROFIT' in o.get('type', '').upper() for o in symbol_orders)
            
            if has_position:
                # Get technical analysis for current position
                tech_direction = self._get_technical_direction()
                
                # We have position - ensure BOTH stop-loss AND take-profit orders exist
                self.logger.info(f"📊 {self.symbol} Position detected: {position_side.upper()} {abs(position_size):.6f} @ ${entry_price:.6f}")
                self.logger.info(f"📊 {self.symbol} Technical Analysis: {tech_direction.upper()} signal for current position")

                # Check if technical analysis suggests exiting position
                if self._check_exit_on_opposite_signal(position_side, position_size, tech_direction):
                    return  # Position was closed, exit early
                
                # Place missing stop-loss order - FIXED: Pass entry_price
                # if not has_stop_orders:
                #     self.logger.info(f"🛡️ No stop-loss found for {self.symbol}, placing...")
                #     is_long = position_side == 'long'
                #     is_short = position_side == 'short'
                #     self._ensure_stop_loss_orders(is_long, is_short, entry_price, position_size, live_orders)
                
                # # Place missing take-profit order
                # if not has_tp_orders:
                #     self.logger.info(f"🎯 No take-profit found for {self.symbol}, placing...")
                #     self._ensure_take_profit_orders(position_side, entry_price, position_size)
            
            elif len(symbol_orders) == 0:
                # No position and no orders - place new market order
                tech_direction = self._get_technical_direction()
                if tech_direction != 'none':
                    time_since_last_order = current_time - getattr(self, '_last_market_order_time', 0)
                    if time_since_last_order > 60:  # 1 minute throttle
                        try:
                            amount = self._calculate_order_amount(current_price)
                            if amount >= self.min_amount:
                                order = self.exchange.create_market_order(self.symbol, tech_direction, amount)
                                if order and 'id' in order:
                                    self._last_market_order_time = current_time
                                    self.logger.info(f"✅ New MARKET order: {tech_direction.upper()} {amount:.6f}")
                                    time.sleep(2)  # Let position settle
                        except Exception as e:
                            self.logger.error(f"❌ Error placing new market order: {e}")
            
            self._last_ensure_orders_time = current_time
                    
        except Exception as e:
            self.logger.error(f"❌ Error in simplified ping-pong logic: {e}")

    def _ensure_take_profit_orders(self, position_side: str, entry_price: float, position_size: float):
        """Place take-profit order for existing position"""
        try:
            if not entry_price or entry_price <= 0:
                self.logger.error(f"🎯 Invalid entry price for take-profit: ${entry_price:.6f}")
                return
            
            # Calculate take-profit price (2% profit target)
            profit_percentage = 0.02  # 2% profit
            
            if position_side == 'long':
                # For LONG position: take-profit ABOVE entry price (sell order)
                tp_price = self._round_price(entry_price * (1 + profit_percentage))
                tp_side = 'sell'
            else:  # short
                # For SHORT position: take-profit BELOW entry price (buy order)
                tp_price = self._round_price(entry_price * (1 - profit_percentage))
                tp_side = 'buy'
            
            tp_amount = max(abs(position_size), self.min_amount)
            
            # Use TAKE_PROFIT_MARKET order type for Binance USDM
            try:
                tp_order = self.exchange.create_take_profit_market_order(
                    self.symbol, tp_side, tp_amount, tp_price
                )
                
                if tp_order and 'id' in tp_order:
                    profit_pct = profit_percentage * 100
                    position_type = "LONG" if position_side == 'long' else "SHORT"
                    
                    self.logger.info(f"🎯 TAKE-PROFIT PLACED: {position_type} position")
                    self.logger.info(f"🎯 TP Order: {tp_side.upper()} {tp_amount:.6f} TP @ ${tp_price:.6f} (+{profit_pct:.1f}% profit)")
                    self.logger.info(f"🎯 Order ID: {tp_order['id']}")
                else:
                    self.logger.error(f"❌ Failed to place take-profit order")
                    
            except Exception as e:
                self.logger.error(f"🎯 TAKE_PROFIT_MARKET failed: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ Error ensuring take-profit orders: {e}")

    def _ensure_stop_loss_orders(self, has_long_position: bool, has_short_position: bool, 
                        entry_price: float, position_size: float, live_orders: List[Dict]):
        """FIXED: Use provided entry_price instead of re-fetching from exchange"""
        try:
            # Count existing stop-loss orders
            existing_sl_orders = 0
            for order in live_orders:
                order_symbol = order.get('info', {}).get('symbol', '')
                if order_symbol != self.symbol:
                    continue
                
                order_id = order.get('id', '')
                if order_id in self.stop_loss_order_ids:
                    existing_sl_orders += 1
            
            # If we already have stop-loss orders, don't place more
            if existing_sl_orders > 0:
                self.logger.debug(f"🛡️ Stop-loss already active: {existing_sl_orders} order(s)")
                return
            
            # FIXED: Use provided entry_price directly, no need to re-fetch
            if entry_price is None or entry_price <= 0:
                self.logger.error(f"🛡️ Invalid entry price provided: ${entry_price:.6f}")
                return
            
            self.logger.info(f"🛡️ Using provided entry price: ${entry_price:.6f}")
            
            # Calculate stop-loss price based on PROVIDED ENTRY PRICE
            if has_long_position:
                # For LONG position: stop-loss BELOW entry price (sell order)
                sl_price = self._round_price(entry_price * (1 - self.stop_loss_percentage))
                sl_side = 'sell'
                sl_amount = abs(position_size)
                
            elif has_short_position:
                # For SHORT position: stop-loss ABOVE entry price (buy order) 
                sl_price = self._round_price(entry_price * (1 + self.stop_loss_percentage))
                sl_side = 'buy'
                sl_amount = abs(position_size)
            else:
                return
            
            # Validate stop-loss price is within grid range
            if sl_price < self.user_price_lower or sl_price > self.user_price_upper:
                self.logger.warning(f"🛡️ Stop-loss price ${sl_price:.6f} outside grid range [{self.user_price_lower:.6f}, {self.user_price_upper:.6f}], skipping")
                return
            
            # Ensure minimum amount
            if sl_amount < self.min_amount:
                sl_amount = self.min_amount
            
            # Log the calculation for verification
            loss_pct = self.stop_loss_percentage * 100
            expected_loss_pct = abs((sl_price - entry_price) / entry_price) * 100
            
            self.logger.info(f"🛡️ Stop-loss calculation: Entry price: ${entry_price:.6f} Stop-loss price: ${sl_price:.6f} Expected loss: {expected_loss_pct:.2f}% (target: {loss_pct:.1f}%)")
            # self.logger.info(f"🛡️   Entry price: ${entry_price:.6f}")
            # self.logger.info(f"🛡️   Stop-loss price: ${sl_price:.6f}")
            # self.logger.info(f"🛡️   Expected loss: {expected_loss_pct:.2f}% (target: {loss_pct:.1f}%)")
            
            # Use proper Binance USDM futures stop order
            try:
                sl_order = self.exchange.exchange.create_order(
                    symbol=self.symbol,
                    type='STOP_MARKET',
                    side=sl_side.upper(),
                    amount=sl_amount,
                    price=None,
                    params={
                        'stopPrice': sl_price,
                        'timeInForce': 'GTE_GTC'
                    }
                )
                
                if sl_order and 'id' in sl_order:
                    self.stop_loss_order_ids.add(sl_order['id'])
                    
                    position_type = "LONG" if has_long_position else "SHORT"
                    
                    self.logger.info(f"🛡️ STOP-LOSS PLACED: {position_type} position SL Order: {sl_side.upper()} {sl_amount:.6f} STOP @ ${sl_price:.6f} ({loss_pct}% protection) ")
                    # self.logger.info(f"🛡️ SL Order: {sl_side.upper()} {sl_amount:.6f} STOP @ ${sl_price:.6f} ({loss_pct}% protection)")
                    # self.logger.info(f"🛡️ Order ID: {sl_order['id']}")
                else:
                    self.logger.error(f"❌ Failed to place stop-loss order")
                    
            except Exception as e:
                self.logger.error(f"🛡️ STOP_MARKET failed: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ Error ensuring stop-loss orders: {e}")
    def _check_exit_on_opposite_signal(self, position_side: str, position_size: float, tech_direction: str) -> bool:
        """Enhanced signal-based exit with improved logging and error handling"""
        try:
            # Don't exit on mixed signals - this is correct behavior
            if tech_direction == 'none':
                self.logger.debug(f"📊 Mixed technical signals - keeping {position_side.upper()} position")
                return False
            
            # Check if signal is opposite to position
            should_exit = False
            exit_reason = ""
            
            if position_side == 'long' and tech_direction == 'sell':
                should_exit = True
                exit_reason = "LONG position but SELL signal detected"
            elif position_side == 'short' and tech_direction == 'buy':
                should_exit = True
                exit_reason = "SHORT position but BUY signal detected"
            
            if should_exit:
                self.logger.warning(f"🚨 SIGNAL-BASED EXIT: {exit_reason}")
                self.logger.warning(f"🚨 Closing {position_side.upper()} position {abs(position_size):.6f}")
                
                # Cancel all existing orders first - MORE ROBUST
                try:
                    cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                    if cancelled_orders:
                        self.logger.info(f"🚨 Cancelled {len(cancelled_orders)} orders before signal exit")
                    
                    # Clear ALL tracking (not just regular orders)
                    self.placed_order_ids.clear()
                    self.stop_loss_order_ids.clear()  # Keep for cleanup
                    
                except Exception as e:
                    self.logger.error(f"❌ Error cancelling orders before signal exit: {e}")
                    # Continue with exit even if cancellation fails
                
                # Close position with market order - ENHANCED ERROR HANDLING
                try:
                    close_side = 'sell' if position_side == 'long' else 'buy'
                    close_amount = abs(position_size)
                    
                    exit_order = self.exchange.create_market_order(self.symbol, close_side, close_amount)
                    
                    if exit_order and 'id' in exit_order:
                        # Record the exit trade with enhanced info
                        self.filled_orders.append({
                            'id': exit_order['id'],
                            'side': close_side,
                            'price': float(exit_order.get('average', 0)),
                            'amount': close_amount,
                            'timestamp': time.time(),
                            'type': 'signal_exit',  # Clear identification
                            'exit_reason': exit_reason  # Additional context
                        })
                        
                        self.total_trades += 1
                        
                        self.logger.warning(f"🚨 POSITION CLOSED: {close_side.upper()} {close_amount:.6f} (Signal Exit)")
                        self.logger.info(f"✅ Exit Order ID: {exit_order['id']}")
                        
                        return True  # Position was successfully closed
                    else:
                        self.logger.error(f"❌ Failed to close position - invalid exit order response")
                        return False  # Exit failed
                        
                except Exception as e:
                    self.logger.error(f"❌ Error closing position on signal reversal: {e}")
                    return False  # Exit failed
            else:
                # Signal aligns with position - enhanced logging
                if (position_side == 'long' and tech_direction == 'buy') or (position_side == 'short' and tech_direction == 'sell'):
                    self.logger.debug(f"📊 Signal ALIGNED: {position_side.upper()} position with {tech_direction.upper()} signal")
                
            return False  # Position was not closed
            
        except Exception as e:
            self.logger.error(f"❌ Error in signal-based exit check: {e}")
            return False  # Fail safe - don't exit on errors
    def _place_initial_grid_orders(self, current_price: float, grid_levels: List[float]) -> int:
        """Place single initial MARKET order based on technical analysis"""
        orders_placed = 0
        
        try:
            # Get technical direction for initial market order
            tech_direction = self._get_technical_direction()
            
            if tech_direction == 'none':
                self.logger.info(f"📊 Mixed technical signals - NO ORDER will be placed")
                self.logger.info(f"📊 Waiting for clear technical direction...")
                return 0  # Don't place any order ✅
            
            self.logger.info(f"Placing 1 MARKET order: {tech_direction.upper()} (immediate execution)")
            
            # Calculate amount for market order
            amount = self._calculate_order_amount(current_price)
            if amount < self.min_amount:
                self.logger.error(f"Calculated amount {amount:.6f} below minimum {self.min_amount}")
                return 0
            
            # Place MARKET order for immediate execution
            order = self.exchange.create_market_order(self.symbol, tech_direction, amount)
            if order and 'id' in order:
                # Track as filled order immediately
                fill_price = float(order.get('average', current_price))
                self.filled_orders.append({
                    'id': order['id'],
                    'side': tech_direction,
                    'price': fill_price,
                    'amount': amount,
                    'timestamp': time.time(),
                    'type': 'market'
                })
                
                self.total_trades += 1
                orders_placed = 1
                
                self.logger.info(f"✅ MARKET order executed: {tech_direction.upper()} {amount:.6f} @ ${fill_price:.6f}")
                time.sleep(2)  # Let position settle
                    
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error placing initial market order: {e}")
            return orders_placed
    
    def update_grid(self):
        """Main grid update loop - FIXED: Added logging to show when called"""
        try:
            with self.update_lock:
                if not self.running:
                    return
                
                # Log update activity every 12 calls (1 minute with 5-second intervals)
                update_count = getattr(self, '_update_count', 0) + 1
                self._update_count = update_count
                
                # if update_count % 12 == 1:
                #     self.logger.info(f"🔄 Grid update #{update_count}: Checking orders and fills...")
                
                # Check for filled orders and place new ones
                self._check_filled_orders()
                
                # FIXED: Ensure we always have orders (ping-pong strategy needs 2 orders)
                self._ensure_ping_pong_orders()
                
                # Update PnL
                self._update_pnl()
                
                # Check TP/SL conditions
                self._check_tp_sl()
                
                self.last_update_time = time.time()
                
        except Exception as e:
            self.logger.error(f"❌ Error updating grid: {e}")
    def _place_stop_loss_immediately(self, position_size: float, position_side: str, entry_price: float):
        """Place stop-loss based on current position data"""
        try:
            if not entry_price or entry_price <= 0:
                current_price = float(self.exchange.get_ticker(self.symbol)['last'])
                entry_price = current_price
                self.logger.warning(f"🛡️ Using current price as entry: ${entry_price:.6f}")
            
            # Calculate stop-loss
            if position_side == 'long':
                sl_price = self._round_price(entry_price * (1 - self.stop_loss_percentage))
                sl_side = 'sell'
            else:  # short
                sl_price = self._round_price(entry_price * (1 + self.stop_loss_percentage))
                sl_side = 'buy'
            
            sl_amount = max(abs(position_size), self.min_amount)
            
            # Place stop using existing exchange method
            sl_order = self.exchange.create_stop_order(
                symbol=self.symbol,
                side=sl_side,
                amount=sl_amount,
                stop_price=sl_price,
                order_type='stop_market'
            )
            
            if sl_order and 'id' in sl_order:
                self.logger.info(f"🛡️ STOP-LOSS PLACED: {sl_side.upper()} @ ${sl_price:.6f}")
                self.logger.info(f"🛡️ Protection: {self.stop_loss_percentage*100:.1f}% from ${entry_price:.6f}")
            
        except Exception as e:
            self.logger.error(f"❌ Stop-loss placement failed: {e}")
    def _check_filled_orders(self):
        """Enhanced: Check both regular and stop-loss filled orders - verify actual fill status"""
        try:
            # Get current open orders only
            live_orders = self.exchange.get_open_orders(self.symbol)
            live_order_ids = set()
            
            # Extract order IDs with symbol verification
            for order in live_orders:
                order_symbol = order.get('info', {}).get('symbol', '')
                if order_symbol == self.symbol:
                    live_order_ids.add(order['id'])
            
            # Find orders that are no longer open (could be filled, cancelled, or rejected)
            potentially_filled_order_ids = self.placed_order_ids - live_order_ids
            potentially_filled_sl_order_ids = self.stop_loss_order_ids - live_order_ids
            
            # FIXED: Verify actual fill status instead of assuming
            actually_filled_order_ids = set()
            actually_filled_sl_order_ids = set()
            
            # Check regular orders
            for order_id in potentially_filled_order_ids:
                try:
                    order_status = self.exchange.get_order_status(order_id, self.symbol)
                    status = order_status.get('status', '').lower()
                    
                    if status in ['filled', 'closed']:
                        actually_filled_order_ids.add(order_id)
                        self.logger.debug(f"✅ Confirmed filled order: {order_id[:8]} - {status}")
                    elif status in ['cancelled', 'canceled', 'rejected']:
                        self.logger.info(f"🚫 Order {order_id[:8]} was {status}, removing from tracking")
                        # Remove from tracking since it's not going to fill
                        self.placed_order_ids.discard(order_id)
                    else:
                        self.logger.debug(f"⏳ Order {order_id[:8]} status: {status}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not verify order {order_id[:8]} status: {e}")
                    # Keep in tracking for next check
            
            # Check stop-loss orders
            # for sl_order_id in potentially_filled_sl_order_ids:
            #     try:
            #         order_status = self.exchange.get_order_status(sl_order_id, self.symbol)
            #         status = order_status.get('status', '').lower()
                    
            #         if status in ['filled', 'closed']:
            #             actually_filled_sl_order_ids.add(sl_order_id)
            #             self.logger.debug(f"🛡️ Confirmed filled stop-loss: {sl_order_id[:8]} - {status}")
            #         elif status in ['cancelled', 'canceled', 'rejected']:
            #             self.logger.info(f"🛡️ Stop-loss {sl_order_id[:8]} was {status}, removing from tracking")
            #             # Remove from tracking since it's not going to fill
            #             self.stop_loss_order_ids.discard(sl_order_id)
            #         else:
            #             self.logger.debug(f"🛡️ Stop-loss {sl_order_id[:8]} status: {status}")
                        
            #     except Exception as e:
            #         self.logger.warning(f"🛡️ Could not verify stop-loss {sl_order_id[:8]} status: {e}")
            #         # Keep in tracking for next check
            
            # Process actually filled regular orders
            if actually_filled_order_ids:
                self.logger.info(f"🎯 FOUND {len(actually_filled_order_ids)} FILLED ORDERS")
                
                for order_id in actually_filled_order_ids:
                    self._process_filled_order_live(order_id)
                    # Remove from tracking
                    self.placed_order_ids.discard(order_id)
                
                self.logger.info(f"✅ Processed {len(actually_filled_order_ids)} filled orders")
            
            # Process actually filled stop-loss orders
            if actually_filled_sl_order_ids:
                self.logger.info(f"🛡️ FOUND {len(actually_filled_sl_order_ids)} FILLED STOP-LOSS ORDERS")
                
                for sl_order_id in actually_filled_sl_order_ids:
                    self._process_filled_stop_loss_order(sl_order_id)
                    # Remove from tracking
                    self.stop_loss_order_ids.discard(sl_order_id)
                
                self.logger.info(f"🛡️ Processed {len(actually_filled_sl_order_ids)} filled stop-loss orders")
                    
        except Exception as e:
            self.logger.error(f"❌ Error checking filled orders: {e}")
    def _process_filled_stop_loss_from_exchange_data(self, order: Dict):
        """Process filled stop-loss using exchange order data"""
        try:
            order_id = order.get('id', '')
            fill_side = order.get('side', '').lower()
            fill_price = float(order.get('average', 0))
            fill_amount = float(order.get('filled', 0))
            order_type = order.get('type', '').upper()
            
            self.logger.warning(f"🛡️ STOP/TP EXECUTED: {fill_side.upper()} {fill_amount:.6f} @ ${fill_price:.6f}")
            self.logger.warning(f"🛡️ Order Type: {order_type}")
            
            # Record the stop-loss fill
            self.filled_orders.append({
                'id': order_id,
                'side': fill_side,
                'price': fill_price,
                'amount': fill_amount,
                'timestamp': time.time(),
                'type': 'stop_loss'
            })
            
            self.total_trades += 1
            
            # Cancel remaining orders after stop-loss
            try:
                cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                if cancelled_orders:
                    self.logger.warning(f"🛡️ Cancelled {len(cancelled_orders)} orders after stop/TP")
                
            except Exception as e:
                self.logger.error(f"❌ Error cancelling orders after stop-loss: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing stop-loss from exchange data: {e}")
    def _process_filled_order_from_exchange_data(self, order: Dict):
        """Process filled order using exchange order data"""
        try:
            order_id = order.get('id', '')
            fill_side = order.get('side', '').lower()
            fill_price = float(order.get('average', 0))
            fill_amount = float(order.get('filled', 0))
            order_type = order.get('type', '').upper()
            
            if fill_price <= 0 or fill_amount <= 0:
                self.logger.warning(f"Invalid fill data for order {order_id[:8]}")
                return
            
            # Record the fill
            self.filled_orders.append({
                'id': order_id,
                'side': fill_side,
                'price': fill_price,
                'amount': fill_amount,
                'timestamp': time.time(),
                'type': order_type.lower()
            })
            
            self.total_trades += 1
            
            self.logger.info(f"✅ RECORDED FILL: {fill_side.upper()} {fill_amount:.6f} @ ${fill_price:.6f}")
            
            # For market orders, don't place counter orders (they create positions immediately)
            if 'MARKET' in order_type:
                self.logger.info(f"📊 Market order filled - position created, waiting for take profit")
                return
            
            # For limit orders, place counter take profit order
            self._place_counter_take_profit_order(fill_side, fill_price, fill_amount)
                    
        except Exception as e:
            self.logger.error(f"Error processing filled order from exchange data: {e}")
    def _place_counter_take_profit_order(self, filled_side: str, filled_price: float, filled_amount: float):
        """Place counter take profit order after limit order fill"""
        try:
            # Determine counter side and price
            counter_side = 'sell' if filled_side == 'buy' else 'buy'
            
            # Calculate take profit price (2% profit)
            profit_pct = self.ping_pong_spacing_pct * 2  # 2% profit
            
            if counter_side == 'sell':
                tp_price = self._round_price(filled_price * (1 + profit_pct))
            else:
                tp_price = self._round_price(filled_price * (1 - profit_pct))
            
            # Validate price is within range
            if tp_price < self.user_price_lower or tp_price > self.user_price_upper:
                self.logger.warning(f"Take profit price ${tp_price:.6f} outside grid range")
                return
            
            # Place take profit order
            try:
                tp_order = self.exchange.create_take_profit_market_order(
                    self.symbol, counter_side, filled_amount, tp_price
                )
                if tp_order and 'id' in tp_order:
                    profit_pct_display = profit_pct * 100
                    self.logger.info(f"🔄 COUNTER TP: {filled_side.upper()} @ ${filled_price:.6f} → "
                                f"{counter_side.upper()} TP @ ${tp_price:.6f} (+{profit_pct_display:.1f}%)")
            except Exception as e:
                self.logger.error(f"❌ Error placing counter take profit: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ Error in counter take profit logic: {e}")
    def _process_filled_stop_loss_order(self, sl_order_id: str):
        """FIXED: Simplified stop-loss processing"""
        try:
            # Get stop-loss order details
            order_status = self.exchange.get_order_status(sl_order_id, self.symbol)
            
            if order_status.get('status') not in ['filled', 'closed']:
                return
            
            fill_side = order_status.get('side', '').lower()
            fill_price = float(order_status.get('average', 0))
            fill_amount = float(order_status.get('filled', 0))
            
            self.logger.warning(f"🛡️ STOP-LOSS EXECUTED: {fill_side.upper()} {fill_amount:.6f} @ ${fill_price:.6f}")
            
            # Record the stop-loss fill
            self.filled_orders.append({
                'id': sl_order_id,
                'side': fill_side,
                'price': fill_price,
                'amount': fill_amount,
                'timestamp': time.time(),
                'type': 'stop_loss'
            })
            
            self.total_trades += 1
            
            # Cancel remaining orders after stop-loss
            try:
                cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                if cancelled_orders:
                    self.logger.warning(f"🛡️ Cancelled {len(cancelled_orders)} orders after stop-loss")
                
                # Clear tracking
                self.placed_order_ids.clear()
                self.stop_loss_order_ids.clear()
                
            except Exception as e:
                self.logger.error(f"❌ Error cancelling orders after stop-loss: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing stop-loss order {sl_order_id}: {e}")

    def _process_filled_order_live(self, order_id: str):
        """Process filled order using live exchange data only"""
        try:
            # Get fill details from exchange (live data)
            order_status = self.exchange.get_order_status(order_id, self.symbol)
            
            if order_status.get('status') not in ['filled', 'closed']:
                return  # Not actually filled
            
            fill_side = order_status.get('side', '').lower()
            fill_price = float(order_status.get('average', 0))
            fill_amount = float(order_status.get('filled', 0))
            
            if fill_price <= 0 or fill_amount <= 0:
                self.logger.warning(f"Invalid fill data for order {order_id[:8]}")
                return
            
            # Record the fill
            self.filled_orders.append({
                'id': order_id,
                'side': fill_side,
                'price': fill_price,
                'amount': fill_amount,
                'timestamp': time.time()
            })
            
            self.total_trades += 1
            
            # Get current live orders to check what we have
            live_orders = self.exchange.get_open_orders(self.symbol)
            buy_orders = 0
            sell_orders = 0
            
            for order in live_orders:
                order_symbol = order.get('info', {}).get('symbol', '')
                if order_symbol == self.symbol:
                    side = order.get('side', '').lower()
                    if side == 'buy':
                        buy_orders += 1
                    elif side == 'sell':
                        sell_orders += 1
            
            # Determine counter order
            counter_side = 'sell' if fill_side == 'buy' else 'buy'
            counter_side_count = sell_orders if counter_side == 'sell' else buy_orders
            
            # Only place counter order if we don't have one of that type
            if counter_side_count > 0:
                self.logger.info(f"⚠️ Already have {counter_side_count} {counter_side.upper()} order(s), skipping")
                return
            
            # Place counter order
            tight_spacing = fill_price * self.ping_pong_spacing_pct
            
            if counter_side == 'sell':
                counter_price = self._round_price(fill_price + tight_spacing)
            else:
                counter_price = self._round_price(fill_price - tight_spacing)
            
            if counter_price < self.user_price_lower or counter_price > self.user_price_upper:
                self.logger.info(f"Counter order price ${counter_price:.6f} outside range")
                return
            
            try:
                amount = self._calculate_order_amount(counter_price)
                if amount >= self.min_amount:
                    counter_order = self.exchange.create_limit_order(self.symbol, counter_side, amount, counter_price)
                    if counter_order and 'id' in counter_order:
                        # Track only the order ID
                        self.placed_order_ids.add(counter_order['id'])
                        
                        distance_pct = abs((counter_price - fill_price) / fill_price) * 100
                        self.logger.info(f"🔄 COUNTER ORDER: {fill_side.upper()} @ ${fill_price:.6f} → "
                                    f"{counter_side.upper()} @ ${counter_price:.6f} ({distance_pct:.2f}%)")
                        
                        # Log using live data
                        final_buy = buy_orders + (1 if counter_side == 'buy' else 0)
                        final_sell = sell_orders + (1 if counter_side == 'sell' else 0)
                        self.logger.info(f"📊 Order status: {final_buy} BUY, {final_sell} SELL orders active")
                            
            except Exception as e:
                self.logger.error(f"❌ Error placing counter order: {e}")
                
        except Exception as e:
            self.logger.error(f"Error processing filled order {order_id}: {e}")

    def _update_pnl(self):
        """Update PnL calculations"""
        try:
            if len(self.filled_orders) < 2:
                return
            
            # Simple PnL calculation from completed pairs
            total_pnl = 0.0
            
            # Group fills by pairs (buy then sell, or sell then buy)
            for i in range(len(self.filled_orders) - 1):
                order1 = self.filled_orders[i]
                order2 = self.filled_orders[i + 1]
                
                # Check if this is a pair (different sides)
                if order1['side'] != order2['side']:
                    if order1['side'] == 'buy' and order2['side'] == 'sell':
                        # Buy low, sell high
                        pnl = (order2['price'] - order1['price']) * order1['amount']
                    elif order1['side'] == 'sell' and order2['side'] == 'buy':
                        # Sell high, buy low
                        pnl = (order1['price'] - order2['price']) * order1['amount']
                    else:
                        continue
                    
                    total_pnl += pnl
            
            self.total_pnl = total_pnl
            
        except Exception as e:
            self.logger.error(f"Error updating PnL: {e}")
    
    def _check_tp_sl(self):
        """Check take profit and stop loss conditions"""
        try:
            if self.user_total_investment <= 0:
                return
            
            pnl_percentage = (self.total_pnl / self.user_total_investment) * 100
            
            # Check take profit
            if pnl_percentage >= self.take_profit_pnl:
                self.logger.info(f"Take profit reached: {pnl_percentage:.2f}% >= {self.take_profit_pnl:.2f}%")
                self.stop_grid()
                return
            
            # Check stop loss
            if pnl_percentage <= -self.stop_loss_pnl:
                self.logger.info(f"Stop loss reached: {pnl_percentage:.2f}% <= -{self.stop_loss_pnl:.2f}%")
                self.stop_grid()
                return
                
        except Exception as e:
            self.logger.error(f"Error checking TP/SL: {e}")
    
    def stop_grid(self):
        """Enhanced: Stop grid and clear all tracking including stop-loss orders"""
        try:
            if not self.running:
                return
            
            self.running = False
            
            # Cancel all orders using live data
            cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
            if cancelled_orders:
                self.logger.info(f"Cancelled {len(cancelled_orders)} orders")
            
            # ENHANCEMENT: Clear all tracking including stop-loss orders
            self.placed_order_ids.clear()
            self.stop_loss_order_ids.clear()
            
            # Close positions at market
            positions = self.exchange.get_positions(self.symbol)
            for position in positions:
                pos_symbol = position.get('info', {}).get('symbol', '')
                if pos_symbol != self.symbol:
                    continue
                size = float(position.get('contracts', 0))
                if abs(size) >= 0.001:
                    close_side = 'sell' if size > 0 else 'buy'
                    self.exchange.create_market_order(self.symbol, close_side, abs(size))
                    self.logger.info(f"Closed position: {close_side.upper()} {abs(size):.6f}")
            
            self.logger.info(f"✅ GRID STOPPED")
            
        except Exception as e:
            self.logger.error(f"Error stopping grid: {e}")
            self.running = False
        
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive grid status"""
        try:
            with self.update_lock:
                # Calculate PnL percentage
                pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0.0
                
                # Count active orders (including stop-loss)
                active_orders = len(self.grid_orders)
                active_sl_orders = len(self.stop_loss_order_ids)
                
                return {
                    # Identification
                    'grid_id': self.grid_id,
                    'symbol': self.symbol,
                    'display_symbol': self.original_symbol,
                    
                    # Configuration
                    'price_lower': self.user_price_lower,
                    'price_upper': self.user_price_upper,
                    'grid_number': self.user_grid_number,
                    'investment': self.user_total_investment,
                    'leverage': self.user_leverage,
                    
                    # Strategy settings
                    'take_profit_pnl': self.take_profit_pnl,
                    'stop_loss_pnl': self.stop_loss_pnl,
                    'stop_loss_percentage': self.stop_loss_percentage * 100,  # ENHANCEMENT: Show SL%
                    'enable_grid_adaptation': self.enable_grid_adaptation,
                    'enable_samig': False,  # Always False for simplified version
                    
                    # Current state
                    'running': self.running,
                    'active_positions': 0,  # No position tracking in simplified version
                    'orders_count': active_orders,
                    'stop_loss_orders_count': active_sl_orders,  # ENHANCEMENT: Show SL orders
                    'trades_count': self.total_trades,
                    
                    # Performance
                    'pnl': self.total_pnl,
                    'pnl_percentage': pnl_percentage,
                    'last_update': self.last_update_time,
                }
                
        except Exception as e:
            self.logger.error(f"Error getting grid status: {e}")
            return {
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'running': False,
                'error': str(e)
            }