"""
Simplified Grid Trading Strategy - Places buy/sell orders at grid intervals.
No hedge mode, no complex market intelligence, just simple grid trading.
"""

import logging
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from collections import deque

from core.exchange import Exchange


class GridStrategy:
    """
    Simplified Grid Trading Strategy
    
    Places buy orders below current price and sell orders above current price
    at calculated grid intervals. When orders fill, places counter orders.
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
        self.ping_pong_spacing_pct = 0.003  # 0.3% default spacing
        
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
                        f"Levels: {self.user_grid_number}, Investment: ${self.user_total_investment:.2f}")
        
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
    
    def _ensure_ping_pong_orders(self):
        """FIXED: Check positions first - don't place orders against existing positions"""
        try:
            # FIXED: Throttle order placement to avoid spam (max once per 30 seconds)
            current_time = time.time()
            last_ensure_time = getattr(self, '_last_ensure_orders_time', 0)
            if current_time - last_ensure_time < 30:
                return
            
            # FIXED: Check existing positions first
            live_positions = self.exchange.get_positions(self.symbol)
            has_long_position = False
            has_short_position = False
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                side = pos.get('side', '').lower()
                if abs(size) >= 0.001:  # Meaningful position
                    if side == 'long':
                        has_long_position = True
                    elif side == 'short':
                        has_short_position = True
            
            # Get current live orders
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            # Count buy and sell orders
            buy_orders = 0
            sell_orders = 0
            
            for order in live_orders:
                side = order.get('side', '').lower()
                if side == 'buy':
                    buy_orders += 1
                elif side == 'sell':
                    sell_orders += 1
            
            # FIXED: Logic based on positions + orders
            needs_buy_order = False
            needs_sell_order = False
            
            if has_long_position:
                # Have LONG position → need SELL order to take profit, don't need BUY order
                needs_sell_order = (sell_orders == 0)
                needs_buy_order = False
                if needs_sell_order:
                    self.logger.info(f"📊 Have LONG position, need SELL order to take profit")
            elif has_short_position:
                # Have SHORT position → need BUY order to take profit, don't need SELL order  
                needs_buy_order = (buy_orders == 0)
                needs_sell_order = False
                if needs_buy_order:
                    self.logger.info(f"📊 Have SHORT position, need BUY order to take profit")
            else:
                # No positions → need both orders for ping-pong
                needs_buy_order = (buy_orders == 0)
                needs_sell_order = (sell_orders == 0)
                if needs_buy_order or needs_sell_order:
                    self.logger.info(f"📊 No positions, need ping-pong orders (BUY: {needs_buy_order}, SELL: {needs_sell_order})")
            
            # Only proceed if we actually need orders
            if not needs_buy_order and not needs_sell_order:
                return
                
            # Mark that we're placing orders
            self._last_ensure_orders_time = current_time
            
            # Get current price for order placement
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # FIXED: Use much smaller spacing for ping-pong orders (0.2% to 0.5% from current price)
            # Instead of using full grid range, use tight spacing for active trading
            tight_spacing = current_price * self.ping_pong_spacing_pct
            
            self.logger.debug(f"Using tight spacing: {self.ping_pong_spacing_pct*100:.1f}% = ${tight_spacing:.6f}")
            
            # Place BUY order if needed (only when no LONG position)
            if needs_buy_order:
                buy_price = self._round_price(current_price - tight_spacing)
                if buy_price >= self.user_price_lower:
                    try:
                        amount = self._calculate_order_amount(buy_price)
                        if amount >= self.min_amount:
                            order = self.exchange.create_limit_order(self.symbol, 'buy', amount, buy_price)
                            if order and 'id' in order:
                                self.grid_orders[order['id']] = {
                                    'id': order['id'],
                                    'side': 'buy',
                                    'price': buy_price,
                                    'amount': amount,
                                    'timestamp': time.time(),
                                    'type': 'missing_order'
                                }
                                distance_pct = ((current_price - buy_price) / current_price) * 100
                                self.logger.info(f"✅ Placed BUY order: {amount:.6f} @ ${buy_price:.6f} ({distance_pct:.2f}% below current)")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to place BUY order: {e}")
            
            # Place SELL order if needed (only when no SHORT position)
            if needs_sell_order:
                sell_price = self._round_price(current_price + tight_spacing)
                if sell_price <= self.user_price_upper:
                    try:
                        amount = self._calculate_order_amount(sell_price)
                        if amount >= self.min_amount:
                            order = self.exchange.create_limit_order(self.symbol, 'sell', amount, sell_price)
                            if order and 'id' in order:
                                self.grid_orders[order['id']] = {
                                    'id': order['id'],
                                    'side': 'sell',
                                    'price': sell_price,
                                    'amount': amount,
                                    'timestamp': time.time(),
                                    'type': 'missing_order'
                                }
                                distance_pct = ((sell_price - current_price) / current_price) * 100
                                self.logger.info(f"✅ Placed SELL order: {amount:.6f} @ ${sell_price:.6f} ({distance_pct:.2f}% above current)")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to place SELL order: {e}")
                        
        except Exception as e:
            self.logger.error(f"❌ Error ensuring ping-pong orders: {e}")
    
    def _place_initial_grid_orders(self, current_price: float, grid_levels: List[float]) -> int:
        """Place only 2 initial orders: 1 buy below current price, 1 sell above current price"""
        orders_placed = 0
        
        try:
            # Split levels into buy and sell sides based on current price
            buy_levels = [level for level in grid_levels if level < current_price]
            sell_levels = [level for level in grid_levels if level > current_price]
            
            # Take only the closest level to current price for each side
            if buy_levels:
                buy_price = max(buy_levels)  # Highest buy price (closest to current)
            else:
                # FIXED: Use tight spacing for initial orders
                tight_spacing = current_price * self.ping_pong_spacing_pct
                buy_price = self._round_price(current_price - tight_spacing)
            
            if sell_levels:
                sell_price = min(sell_levels)  # Lowest sell price (closest to current)
            else:
                # FIXED: Use tight spacing for initial orders
                tight_spacing = current_price * self.ping_pong_spacing_pct
                sell_price = self._round_price(current_price + tight_spacing)
            
            self.logger.info(f"Placing 1 BUY order and 1 SELL order (ping-pong strategy)")
            
            # Place 1 BUY order (below current price)
            try:
                amount = self._calculate_order_amount(buy_price)
                if amount >= self.min_amount:
                    order = self.exchange.create_limit_order(self.symbol, 'buy', amount, buy_price)
                    if order and 'id' in order:
                        self.grid_orders[order['id']] = {
                            'id': order['id'],
                            'side': 'buy',
                            'price': buy_price,
                            'amount': amount,
                            'timestamp': time.time(),
                            'type': 'grid'
                        }
                        orders_placed += 1
                        distance_pct = ((current_price - buy_price) / current_price) * 100
                        self.logger.info(f"✅ BUY order placed: {amount:.6f} @ ${buy_price:.6f} ({distance_pct:.2f}% below current)")
                    else:
                        self.logger.error(f"❌ Failed to place BUY order @ ${buy_price:.6f}")
            except Exception as e:
                self.logger.error(f"❌ Error placing BUY order @ ${buy_price:.6f}: {e}")
            
            # Small delay between orders
            time.sleep(0.5)
            
            # Place 1 SELL order (above current price)
            try:
                amount = self._calculate_order_amount(sell_price)
                if amount >= self.min_amount:
                    order = self.exchange.create_limit_order(self.symbol, 'sell', amount, sell_price)
                    if order and 'id' in order:
                        self.grid_orders[order['id']] = {
                            'id': order['id'],
                            'side': 'sell',
                            'price': sell_price,
                            'amount': amount,
                            'timestamp': time.time(),
                            'type': 'grid'
                        }
                        orders_placed += 1
                        distance_pct = ((sell_price - current_price) / current_price) * 100
                        self.logger.info(f"✅ SELL order placed: {amount:.6f} @ ${sell_price:.6f} ({distance_pct:.2f}% above current)")
                    else:
                        self.logger.error(f"❌ Failed to place SELL order @ ${sell_price:.6f}")
            except Exception as e:
                self.logger.error(f"❌ Error placing SELL order @ ${sell_price:.6f}: {e}")
            
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error placing initial grid orders: {e}")
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
    
    def _check_filled_orders(self):
        """Check for filled orders and place counter orders - FIXED: Better logging and sync tracking"""
        try:
            # Get live orders
            live_orders = self.exchange.get_open_orders(self.symbol)
            live_order_ids = {order['id'] for order in live_orders}
            
            # FIXED: Sync internal tracking with live orders
            # Remove orders from tracking that no longer exist (cancelled/expired)
            cancelled_orders = []
            for order_id in list(self.grid_orders.keys()):
                if order_id not in live_order_ids:
                    # Check if this order was filled or just cancelled
                    try:
                        order_status = self.exchange.get_order_status(order_id, self.symbol)
                        if order_status.get('status') in ['filled', 'closed']:
                            # This is a filled order
                            continue
                        else:
                            # This order was cancelled/expired, remove from tracking
                            cancelled_orders.append(order_id)
                    except:
                        # If we can't get status, assume it's cancelled
                        cancelled_orders.append(order_id)
            
            # Clean up cancelled orders from tracking
            for order_id in cancelled_orders:
                if order_id in self.grid_orders:
                    self.logger.warning(f"⚠️ Removing cancelled/expired order from tracking: {order_id[:8]}")
                    del self.grid_orders[order_id]
            
            # Find filled orders (orders in our tracking but not in live orders, and not cancelled)
            filled_order_ids = []
            for order_id in list(self.grid_orders.keys()):
                if order_id not in live_order_ids and order_id not in cancelled_orders:
                    filled_order_ids.append(order_id)
            
            # Log order checking activity
            total_tracked = len(self.grid_orders)
            total_live = len(live_orders)
            
            if filled_order_ids:
                self.logger.info(f"🎯 FOUND {len(filled_order_ids)} FILLED ORDERS: {total_tracked} tracked, {total_live} live")
            
            if cancelled_orders:
                self.logger.warning(f"⚠️ Found {len(cancelled_orders)} cancelled/expired orders")
            
            # Process filled orders
            for order_id in filled_order_ids:
                self._process_filled_order(order_id)
            
            # Clean up tracking for filled orders
            for order_id in filled_order_ids:
                if order_id in self.grid_orders:
                    del self.grid_orders[order_id]
            
            if filled_order_ids:
                self.logger.info(f"✅ Processed {len(filled_order_ids)} filled orders")
                
        except Exception as e:
            self.logger.error(f"❌ Error checking filled orders: {e}")
    
    def _process_filled_order(self, order_id: str):
        """Process a filled order and place counter order (maintain max 2 orders)"""
        try:
            if order_id not in self.grid_orders:
                return
            
            filled_order = self.grid_orders[order_id]
            
            # Get actual fill details from exchange
            try:
                order_status = self.exchange.get_order_status(order_id, self.symbol)
                fill_price = float(order_status.get('average', filled_order['price']))
                fill_amount = float(order_status.get('filled', filled_order['amount']))
            except:
                # Fallback to tracked values
                fill_price = filled_order['price']
                fill_amount = filled_order['amount']
            
            # Record the fill
            self.filled_orders.append({
                'id': order_id,
                'side': filled_order['side'],
                'price': fill_price,
                'amount': fill_amount,
                'timestamp': time.time()
            })
            
            self.total_trades += 1
            
            # Check if we already have 2 orders (should have only 1 remaining after this fill)
            current_orders = len([o for o in self.grid_orders.values() if o['id'] != order_id])
            if current_orders >= 2:
                self.logger.warning(f"Already have {current_orders} orders, skipping counter order placement")
                return
            
            # Calculate counter order price and side
            counter_side = 'sell' if filled_order['side'] == 'buy' else 'buy'
            
            # FIXED: Use tight spacing for counter orders
            tight_spacing = fill_price * self.ping_pong_spacing_pct
            
            if counter_side == 'sell':
                # Place sell order above fill price
                counter_price = self._round_price(fill_price + tight_spacing)
            else:
                # Place buy order below fill price
                counter_price = self._round_price(fill_price - tight_spacing)
            
            # Ensure counter price is within grid range
            if counter_price < self.user_price_lower or counter_price > self.user_price_upper:
                self.logger.info(f"Counter order price ${counter_price:.6f} outside grid range, skipping")
                return
            
            # Place counter order to maintain ping-pong trading
            try:
                amount = self._calculate_order_amount(counter_price)
                if amount >= self.min_amount:
                    counter_order = self.exchange.create_limit_order(self.symbol, counter_side, amount, counter_price)
                    if counter_order and 'id' in counter_order:
                        self.grid_orders[counter_order['id']] = {
                            'id': counter_order['id'],
                            'side': counter_side,
                            'price': counter_price,
                            'amount': amount,
                            'timestamp': time.time(),
                            'type': 'counter'
                        }
                        
                        distance_pct = abs((counter_price - fill_price) / fill_price) * 100
                        self.logger.info(f"🔄 COUNTER ORDER: {filled_order['side'].upper()} @ ${fill_price:.6f} → "
                                       f"{counter_side.upper()} @ ${counter_price:.6f} ({distance_pct:.2f}% spacing)")
                        
                        # Log current order count
                        remaining_orders = len([o for o in self.grid_orders.values() if o['id'] != order_id])
                        self.logger.info(f"📊 Order status: {remaining_orders} orders remain active")
                    else:
                        self.logger.error(f"❌ Failed to place counter order")
                        
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
        """Stop the grid strategy and cleanup"""
        try:
            if not self.running:
                return
            
            self.logger.info(f"🛑 STOPPING GRID...")
            
            # Set running to false first
            self.running = False
            
            # Cancel all orders
            try:
                cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                if cancelled_orders:
                    self.logger.info(f"Cancelled {len(cancelled_orders)} orders")
                
                # Clear tracking
                self.grid_orders.clear()
                
            except Exception as e:
                self.logger.error(f"Error cancelling orders during stop: {e}")
            
            # Close any open positions at market
            try:
                positions = self.exchange.get_positions(self.symbol)
                for position in positions:
                    size = float(position.get('contracts', 0))
                    if abs(size) >= 0.001:
                        # Close position with market order
                        close_side = 'sell' if size > 0 else 'buy'
                        self.exchange.create_market_order(self.symbol, close_side, abs(size))
                        self.logger.info(f"Closed position: {close_side.upper()} {abs(size):.6f}")
                        
            except Exception as e:
                self.logger.error(f"Error closing positions during stop: {e}")
            
            # Final PnL calculation
            final_pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0
            
            self.logger.info(f"✅ GRID STOPPED: {self.total_trades} trades, Final PnL: ${self.total_pnl:.2f} ({final_pnl_percentage:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error stopping grid: {e}")
            self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive grid status"""
        try:
            with self.update_lock:
                # Calculate PnL percentage
                pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0.0
                
                # Count active orders
                active_orders = len(self.grid_orders)
                
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
                    'enable_grid_adaptation': self.enable_grid_adaptation,
                    'enable_samig': False,  # Always False for simplified version
                    
                    # Current state
                    'running': self.running,
                    'active_positions': 0,  # No position tracking in simplified version
                    'orders_count': active_orders,
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