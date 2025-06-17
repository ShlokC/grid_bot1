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
            
            # Place initial 2 orders (ping-pong strategy)
            orders_placed = self._place_initial_grid_orders(current_price, grid_levels)
            
            if orders_placed >= 1:  # At least 1 order should be placed
                self.running = True
                self.logger.info(f"✅ PING-PONG GRID SETUP COMPLETE: {orders_placed} orders placed")
                if orders_placed == 2:
                    self.logger.info(f"🎯 Perfect setup: Both BUY and SELL orders active")
                else:
                    self.logger.warning(f"⚠️ Partial setup: Only {orders_placed} order(s) placed due to insufficient margin")
                return True
            else:
                self.logger.error("❌ GRID SETUP FAILED: No orders placed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR in grid setup: {e}")
            return False
    
    def _get_technical_direction(self) -> str:
        """Get trading direction using JMA and KAMA technical analysis."""
        direction = 'buy'  # FIXED: Initialize direction first
        
        try:
            # Get OHLCV data for analysis (last 50 candles, 5m timeframe)
            ohlcv_data = self.exchange.get_ohlcv(self.symbol, timeframe='5m', limit=50)
            
            if not ohlcv_data or len(ohlcv_data) < 20:
                self.logger.warning("Insufficient OHLCV data for technical analysis, defaulting to BUY")
                return direction
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close'] = df['close'].astype(float)
            
            # Calculate JMA (Jurik Moving Average) - period 14
            jma = ta.jma(df['close'], length=20, phase=0.5, pow=2.0)
            
            # Calculate KAMA (Kaufman's Adaptive Moving Average) - period 14  
            kama = ta.kama(df['close'], length=10)
            
            if jma is None or kama is None or len(jma) < 2 or len(kama) < 2:
                self.logger.warning("Failed to calculate JMA/KAMA, defaulting to BUY")
                return direction
            
            # Get latest values (skip NaN)
            jma_clean = jma.dropna()
            kama_clean = kama.dropna()
            
            if jma_clean.empty or kama_clean.empty:
                self.logger.warning("JMA/KAMA values are empty, defaulting to BUY")
                return direction
                
            latest_jma = jma_clean.iloc[-1]
            latest_kama = kama_clean.iloc[-1]
            current_price = float(df['close'].iloc[-1])
            
            # Direction logic: Both JMA and KAMA must agree
            jma_bullish = current_price > latest_jma
            kama_bullish = current_price > latest_kama
            
            if jma_bullish and kama_bullish:
                direction = 'buy'
                self.logger.info(f"📈 Technical Analysis: BUY signal (Price: ${current_price:.6f}, JMA: ${latest_jma:.6f}, KAMA: ${latest_kama:.6f})")
            elif not jma_bullish and not kama_bullish:
                direction = 'sell'
                self.logger.info(f"📉 Technical Analysis: SELL signal (Price: ${current_price:.6f}, JMA: ${latest_jma:.6f}, KAMA: ${latest_kama:.6f})")
            else:
                # FIXED: Mixed signals = NO TRADE (don't force arbitrary direction)
                direction = 'none'
                self.logger.info(f"⚠️ Technical Analysis: Mixed signals, NO TRADE (JMA: {jma_bullish}, KAMA: {kama_bullish})")
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            direction = 'buy'  # FIXED: Ensure direction is always set
        
        return direction

    def _ensure_ping_pong_orders(self):
        """FIXED: Simplified position-based order placement using exchange data only"""
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
            
            # Count current orders
            active_orders = len([o for o in live_orders 
                            if o.get('info', {}).get('symbol', '') == self.symbol])
            
            if has_position:
                # We have position - ensure stop-loss and take-profit orders exist
                self.logger.info(f"📊 Position detected: {position_side.upper()} {abs(position_size):.6f}")
                
                # Check if we have stop-loss orders
                has_stop_orders = any('STOP' in o.get('type', '').upper() or 'TAKE_PROFIT' in o.get('type', '').upper()
                                    for o in live_orders 
                                    if o.get('info', {}).get('symbol', '') == self.symbol)
                
                if not has_stop_orders:
                    self.logger.info(f"📊 No stop/TP orders found, placing them...")
                    # Place stop-loss order
                    is_long = position_side == 'long'
                    is_short = position_side == 'short'
                    self._ensure_stop_loss_orders(is_long, is_short, current_price, position_size, live_orders)
            
            elif active_orders == 0:
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

    def _ensure_stop_loss_orders(self, has_long_position: bool, has_short_position: bool, 
                            current_price: float, position_size: float, live_orders: List[Dict]):
        """ENHANCEMENT: Ensure stop-loss orders exist for positions using LIVE EXCHANGE DATA ONLY"""
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
            
            # FIXED: Get entry price ONLY from live exchange position data
            entry_price = None
            position_side = None
            
            try:
                # Get fresh position data from exchange
                live_positions = self.exchange.get_positions(self.symbol)
                self.logger.debug(f"🛡️ Fetched {len(live_positions)} live positions from exchange")
                
                for pos in live_positions:
                    pos_symbol = pos.get('info', {}).get('symbol', '')
                    if pos_symbol != self.symbol:
                        continue
                        
                    size = float(pos.get('contracts', 0))
                    side = pos.get('side', '').lower()
                    
                    # Debug position data
                    self.logger.debug(f"🛡️ Position: size={size}, side={side}")
                    
                    # Check if this matches our detected position
                    if (has_long_position and side == 'long' and size > 0) or \
                    (has_short_position and side == 'short' and size < 0):
                        
                        # Try multiple entry price fields
                        entry_price = float(pos.get('entryPrice', 0))
                        if entry_price <= 0:
                            entry_price = float(pos.get('avgPrice', 0))
                        if entry_price <= 0:
                            entry_price = float(pos.get('average', 0))
                        if entry_price <= 0:
                            # Last resort - check info field
                            info = pos.get('info', {})
                            entry_price = float(info.get('entryPrice', 0))
                            if entry_price <= 0:
                                entry_price = float(info.get('avgPrice', 0))
                        
                        if entry_price > 0:
                            position_side = side
                            self.logger.info(f"🛡️ LIVE ENTRY PRICE: ${entry_price:.6f} from exchange position data")
                            break
                        else:
                            # Log available fields for debugging
                            available_fields = list(pos.keys())
                            info_fields = list(pos.get('info', {}).keys()) if pos.get('info') else []
                            self.logger.warning(f"🛡️ No entry price found in position fields: {available_fields}")
                            self.logger.warning(f"🛡️ Info fields available: {info_fields}")
                            
            except Exception as e:
                self.logger.error(f"🛡️ Error fetching live position data: {e}")
            
            # FIXED: Fail if no live entry price found - don't use fallbacks
            if entry_price is None or entry_price <= 0:
                self.logger.error(f"🛡️ CANNOT PLACE STOP-LOSS: No entry price in live exchange data")
                self.logger.error(f"🛡️ Position detected but exchange doesn't provide entry price")
                self.logger.error(f"🛡️ Manual stop-loss monitoring required")
                return
            
            # Calculate stop-loss price based on LIVE ENTRY PRICE
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
            
            self.logger.info(f"🛡️ Stop-loss calculation (LIVE DATA):")
            self.logger.info(f"🛡️   Live entry price: ${entry_price:.6f}")
            self.logger.info(f"🛡️   Current price: ${current_price:.6f}")
            self.logger.info(f"🛡️   Stop-loss price: ${sl_price:.6f}")
            self.logger.info(f"🛡️   Expected loss: {expected_loss_pct:.2f}% (target: {loss_pct:.1f}%)")
            
            # Use proper Binance USDM futures stop order
            try:
                # For Binance USDM futures, use STOP_MARKET order type
                sl_order = self.exchange.exchange.create_order(
                    symbol=self.symbol,
                    type='STOP_MARKET',  # Correct Binance USDM order type
                    side=sl_side.upper(),  # Binance expects uppercase
                    amount=sl_amount,
                    price=None,  # No price for STOP_MARKET
                    params={
                        'stopPrice': sl_price,  # Trigger price
                        'timeInForce': 'GTC'
                    }
                )
                order_type_used = 'STOP_MARKET'
                self.logger.info(f"🛡️ SUCCESS: Created STOP_MARKET order using live entry price")
                
            except Exception as e:
                self.logger.error(f"🛡️ STOP_MARKET failed: {e}")
                self.logger.error(f"🛡️ SKIPPING stop-loss placement - exchange API issue")
                return  # Don't use fallbacks that might execute incorrectly
            
            if sl_order and 'id' in sl_order:
                self.stop_loss_order_ids.add(sl_order['id'])
                
                position_type = "LONG" if has_long_position else "SHORT"
                
                self.logger.info(f"🛡️ STOP-LOSS PLACED: {position_type} position (LIVE DATA)")
                self.logger.info(f"🛡️ SL Order: {sl_side.upper()} {sl_amount:.6f} STOP @ ${sl_price:.6f} ({loss_pct}% protection)")
                self.logger.info(f"🛡️ Order Type: {order_type_used}")
                self.logger.info(f"🛡️ Order ID: {sl_order['id']}")
            else:
                self.logger.error(f"❌ Failed to place stop-loss order")
                    
        except Exception as e:
            self.logger.error(f"❌ Error ensuring stop-loss orders: {e}")
    def _place_initial_grid_orders(self, current_price: float, grid_levels: List[float]) -> int:
        """Place single initial MARKET order based on technical analysis"""
        orders_placed = 0
        
        try:
            # Get technical direction for initial market order
            tech_direction = self._get_technical_direction()
            
            if tech_direction == 'none':
                tech_direction = 'buy'  # Default to buy if mixed signals
                self.logger.info(f"Mixed technical signals - defaulting to BUY market order")
            
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
                
                if update_count % 12 == 1:
                    self.logger.info(f"🔄 Grid update #{update_count}: Checking orders and fills...")
                
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
            for sl_order_id in potentially_filled_sl_order_ids:
                try:
                    order_status = self.exchange.get_order_status(sl_order_id, self.symbol)
                    status = order_status.get('status', '').lower()
                    
                    if status in ['filled', 'closed']:
                        actually_filled_sl_order_ids.add(sl_order_id)
                        self.logger.debug(f"🛡️ Confirmed filled stop-loss: {sl_order_id[:8]} - {status}")
                    elif status in ['cancelled', 'canceled', 'rejected']:
                        self.logger.info(f"🛡️ Stop-loss {sl_order_id[:8]} was {status}, removing from tracking")
                        # Remove from tracking since it's not going to fill
                        self.stop_loss_order_ids.discard(sl_order_id)
                    else:
                        self.logger.debug(f"🛡️ Stop-loss {sl_order_id[:8]} status: {status}")
                        
                except Exception as e:
                    self.logger.warning(f"🛡️ Could not verify stop-loss {sl_order_id[:8]} status: {e}")
                    # Keep in tracking for next check
            
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