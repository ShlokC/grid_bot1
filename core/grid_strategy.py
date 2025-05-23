"""
Enhanced Grid Trading Strategy with Proper Position Management
Maintains constant grid size while intelligently shifting with price movement.
"""
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque
import traceback
from core.exchange import Exchange

# Export classes for external imports
__all__ = ['GridStrategy', 'MarketIntelligence', 'GridPosition', 'GridZone', 'MarketSnapshot']

@dataclass
class GridPosition:
    """Represents a filled position from grid trading"""
    position_id: str
    grid_level: int
    entry_price: float
    quantity: float
    side: str  # 'long' or 'short'
    entry_time: float
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    has_counter_order: bool = False
    counter_order_id: Optional[str] = None

@dataclass
class GridZone:
    """Represents an active grid zone"""
    zone_id: str
    price_lower: float
    price_upper: float
    grid_count: int
    investment_per_grid: float
    active: bool = True
    creation_time: float = 0.0
    positions: Dict[str, GridPosition] = None
    orders: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.orders is None:
            self.orders = {}
        if self.creation_time == 0.0:
            self.creation_time = time.time()

@dataclass
class MarketSnapshot:
    timestamp: float
    price: float
    volume: float
    volatility: float
    momentum: float
    trend_strength: float

class MarketIntelligence:
    """Simplified market intelligence for grid strategy"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_history = deque(maxlen=100)
        self.last_analysis_time = 0
    
    def analyze_market(self, exchange: Exchange) -> MarketSnapshot:
        """Analyze current market conditions"""
        try:
            ticker = exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            volume = float(ticker.get('quoteVolume', 0))
            
            self.price_history.append(current_price)
            
            # Calculate simple indicators
            volatility = self._calculate_volatility()
            momentum = self._calculate_momentum()
            trend_strength = self._calculate_trend_strength()
            
            return MarketSnapshot(
                timestamp=time.time(),
                price=current_price,
                volume=volume,
                volatility=volatility,
                momentum=momentum,
                trend_strength=trend_strength
            )
        except Exception as e:
            logging.error(f"Error in market analysis: {e}")
            return MarketSnapshot(time.time(), 0, 0, 1.0, 0, 0)
    
    def _calculate_volatility(self) -> float:
        """Calculate price volatility"""
        if len(self.price_history) < 10:
            return 1.0
        
        prices = list(self.price_history)[-10:]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return 1.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        return max(0.5, min(3.0, volatility * 100))  # Normalize to 0.5-3.0 range
    
    def _calculate_momentum(self) -> float:
        """Calculate price momentum"""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = list(self.price_history)
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-20:]) / 20
        
        if long_ma > 0:
            momentum = (short_ma - long_ma) / long_ma
            return max(-1.0, min(1.0, momentum * 10))  # Normalize to -1 to 1
        
        return 0.0
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength (0 = ranging, 1 = strong trend)"""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = list(self.price_history)
        
        # Count directional movements
        up_moves = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        down_moves = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
        total_moves = up_moves + down_moves
        
        if total_moves == 0:
            return 0.0
        
        # Strong trend = mostly moves in one direction
        directional_bias = abs(up_moves - down_moves) / total_moves
        return min(1.0, directional_bias * 2)

class GridStrategy:
    def __init__(self, 
                 exchange: Exchange, 
                 symbol: str, 
                 price_lower: float, 
                 price_upper: float,
                 grid_number: int,
                 investment: float,
                 take_profit_pnl: float,
                 stop_loss_pnl: float,
                 grid_id: str,
                 leverage: float = 20.0,
                 enable_grid_adaptation: bool = True,
                 enable_samig: bool = False):
        """Initialize grid strategy with proper position management"""
        
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        
        # IMMUTABLE USER PARAMETERS - Never change these
        self.user_price_lower = float(price_lower)
        self.user_price_upper = float(price_upper)
        self.user_grid_number = int(grid_number)
        self.user_total_investment = float(investment)
        self.user_investment_per_grid = self.user_total_investment / self.user_grid_number
        self.user_leverage = float(leverage)
        
        # Configuration parameters
        self.take_profit_pnl = float(take_profit_pnl)
        self.stop_loss_pnl = float(stop_loss_pnl)
        self.grid_id = grid_id
        self.enable_grid_adaptation = enable_grid_adaptation
        self.enable_samig = enable_samig
        
        # Position and Zone Management
        self.active_zones: Dict[str, GridZone] = {}
        self.all_positions: Dict[str, GridPosition] = {}
        self.pending_orders: Dict[str, Dict] = {}
        
        # Market intelligence
        if self.enable_samig:
            self.market_intel = MarketIntelligence(symbol)
        
        # State tracking
        self.running = False
        self.total_trades = 0
        self.total_pnl = 0.0
        self.last_price_check = 0
        self.last_adaptation_time = 0
        
        # Market info
        self._fetch_market_info()
        
        # Create initial zone
        self._create_initial_zone()
        
        self.logger.info(f"Grid Strategy Initialized:")
        self.logger.info(f"  Symbol: {symbol}")
        self.logger.info(f"  Price Range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
        self.logger.info(f"  Grid Count: {self.user_grid_number}")
        self.logger.info(f"  Investment per Grid: ${self.user_investment_per_grid:.2f}")
        self.logger.info(f"  Total Investment: ${self.user_total_investment:.2f}")
        self.logger.info(f"  Leverage: {self.user_leverage}x")
    
    def _fetch_market_info(self):
        """Fetch market information for the trading pair"""
        try:
            market_info = self.exchange.get_market_info(self.symbol)
            
            # Log the raw market info for debugging
            self.logger.debug(f"Raw market info for {self.symbol}: {market_info}")
            
            # Safely extract precision values
            precision_info = market_info.get('precision', {})
            price_precision = precision_info.get('price', 6)
            amount_precision = precision_info.get('amount', 6)
            
            # Handle different precision formats from exchanges
            if isinstance(price_precision, (int, float)):
                self.price_precision = int(price_precision)
            else:
                self.logger.warning(f"Invalid price precision format: {price_precision}, using default 6")
                self.price_precision = 6
                
            if isinstance(amount_precision, (int, float)):
                self.amount_precision = int(amount_precision)
            else:
                self.logger.warning(f"Invalid amount precision format: {amount_precision}, using default 6")
                self.amount_precision = 6
            
            # Extract limits safely
            limits = market_info.get('limits', {})
            amount_limits = limits.get('amount', {})
            cost_limits = limits.get('cost', {})
            
            self.min_amount = float(amount_limits.get('min', 0.0001))
            self.min_cost = float(cost_limits.get('min', 1.0))
            
            # Log the processed values
            self.logger.info(f"Market info for {self.symbol}:")
            self.logger.info(f"  Price precision: {self.price_precision} decimals")
            self.logger.info(f"  Amount precision: {self.amount_precision} decimals")  
            self.logger.info(f"  Min amount: {self.min_amount}")
            self.logger.info(f"  Min cost: {self.min_cost}")
            
            # Warning if precision seems wrong
            if self.price_precision == 0:
                self.logger.warning(f"Exchange returned 0 price precision for {self.symbol} - will use intelligent rounding")
            if self.amount_precision == 0:
                self.logger.warning(f"Exchange returned 0 amount precision for {self.symbol} - will use intelligent rounding")
            
        except Exception as e:
            self.logger.error(f"Error fetching market info for {self.symbol}: {e}")
            # Safe fallback values
            self.price_precision = 6
            self.amount_precision = 6
            self.min_amount = 0.0001
            self.min_cost = 1.0
            self.logger.info(f"Using fallback market info for {self.symbol}")
    
    def _validate_rounded_values(self, price: float, amount: float) -> Tuple[float, float]:
        """Validate that rounded values meet exchange requirements"""
        try:
            # Ensure price is not zero
            if price <= 0:
                raise ValueError(f"Invalid price after rounding: {price}")
            
            # Ensure amount meets minimum requirements
            if amount < self.min_amount:
                self.logger.warning(f"Amount {amount} below minimum {self.min_amount}, adjusting")
                amount = self.min_amount
            
            # Ensure order value meets minimum cost
            order_value = price * amount
            if order_value < self.min_cost:
                self.logger.warning(f"Order value {order_value} below minimum {self.min_cost}, adjusting amount")
                amount = self.min_cost / price
                amount = self._round_amount(amount)
            
            return price, amount
            
        except Exception as e:
            self.logger.error(f"Error validating rounded values: {e}")
            return price, amount
    
    def _create_initial_zone(self):
        """Create the initial trading zone"""
        zone_id = f"zone_{int(time.time())}"
        initial_zone = GridZone(
            zone_id=zone_id,
            price_lower=self.user_price_lower,
            price_upper=self.user_price_upper,
            grid_count=self.user_grid_number,
            investment_per_grid=self.user_investment_per_grid
        )
        self.active_zones[zone_id] = initial_zone
        
        self.logger.info(f"Created initial zone: {zone_id}")
        self.logger.info(f"  Range: ${initial_zone.price_lower:.6f} - ${initial_zone.price_upper:.6f}")
    
    def _round_price(self, price: float) -> float:
        """Round price with intelligent precision detection"""
        try:
            # Get base precision from exchange
            base_precision = int(self.price_precision) if hasattr(self, 'price_precision') else 0
            
            # Intelligent precision based on price magnitude (crypto-optimized)
            if price < 0.00001:      # Very small coins
                smart_precision = 8
            elif price < 0.0001:     # Small altcoins  
                smart_precision = 7
            elif price < 0.001:      # Micro-priced tokens
                smart_precision = 6
            elif price < 0.01:       # Low-priced coins
                smart_precision = 5
            elif price < 0.1:        # Sub-dollar coins
                smart_precision = 4
            elif price < 1.0:        # Dollar-range coins
                smart_precision = 3
            elif price < 100:        # Normal crypto prices
                smart_precision = 2
            else:                    # High-priced assets
                smart_precision = 1
            
            # Use the maximum of exchange precision and smart precision
            # This prevents losing precision on small values
            final_precision = max(base_precision, smart_precision)
            
            # Ensure minimum precision of 2 for crypto trading
            final_precision = max(final_precision, 2)
            
            # Apply rounding
            rounded_price = float(f"{price:.{final_precision}f}")
            
            self.logger.debug(f"Price rounding: {price} -> {rounded_price} "
                            f"(base_prec: {base_precision}, smart_prec: {smart_precision}, final: {final_precision})")
            
            return rounded_price
            
        except Exception as e:
            self.logger.error(f"Error in _round_price for {price}: {e}")
            # Ultimate fallback: format based on price magnitude
            if price >= 1:
                return float(f"{price:.2f}")
            elif price >= 0.01:
                return float(f"{price:.4f}")
            else:
                return float(f"{price:.6f}")
    
    def _round_amount(self, amount: float) -> float:
        """Round amount with intelligent precision detection"""
        try:
            # Get base precision from exchange
            base_precision = int(self.amount_precision) if hasattr(self, 'amount_precision') else 0
            
            # Intelligent precision for crypto amounts
            if amount < 0.0001:      # Very small amounts
                smart_precision = 8
            elif amount < 0.001:     # Micro amounts
                smart_precision = 7
            elif amount < 0.01:      # Small amounts
                smart_precision = 6
            elif amount < 1:         # Sub-unit amounts
                smart_precision = 5
            elif amount < 100:       # Normal amounts
                smart_precision = 4
            else:                    # Large amounts
                smart_precision = 3
            
            # Use the maximum of exchange precision and smart precision
            final_precision = max(base_precision, smart_precision)
            
            # Ensure minimum precision of 4 for crypto amounts
            final_precision = max(final_precision, 4)
            
            # Apply rounding
            rounded_amount = float(f"{amount:.{final_precision}f}")
            
            self.logger.debug(f"Amount rounding: {amount} -> {rounded_amount} "
                            f"(base_prec: {base_precision}, smart_prec: {smart_precision}, final: {final_precision})")
            
            return rounded_amount
            
        except Exception as e:
            self.logger.error(f"Error in _round_amount for {amount}: {e}")
            # Ultimate fallback
            return float(f"{amount:.6f}")
    
    def _calculate_order_amount(self, price: float, investment_per_grid: float) -> float:
        """Calculate order amount for specific price and investment"""
        try:
            # Round price first
            rounded_price = self._round_price(price)
            
            # Validate price is not zero
            if rounded_price <= 0:
                self.logger.error(f"Price rounded to zero: {price} -> {rounded_price}")
                # Use original price if rounding failed
                rounded_price = price if price > 0 else 0.01
            
            # Calculate notional value
            notional_value = investment_per_grid * self.user_leverage
            
            # Calculate amount
            amount = notional_value / rounded_price
            
            # Round amount
            rounded_amount = self._round_amount(amount)
            
            # Validate both values
            final_price, final_amount = self._validate_rounded_values(rounded_price, rounded_amount)
            
            self.logger.debug(f"Order calculation:")
            self.logger.debug(f"  Investment: ${investment_per_grid:.2f}")
            self.logger.debug(f"  Leverage: {self.user_leverage}x")
            self.logger.debug(f"  Notional: ${notional_value:.2f}")
            self.logger.debug(f"  Price: {price:.8f} -> {final_price:.8f}")
            self.logger.debug(f"  Amount: {amount:.8f} -> {final_amount:.8f}")
            self.logger.debug(f"  Order value: ${final_price * final_amount:.2f}")
            
            return final_amount
            
        except Exception as e:
            self.logger.error(f"Error calculating order amount for price ${price:.8f}: {e}")
            # Emergency fallback
            return max(self.min_amount, 0.0001)
    
    def _get_grid_levels(self, zone: GridZone) -> List[float]:
        """Calculate grid levels for a zone"""
        interval = (zone.price_upper - zone.price_lower) / zone.grid_count
        levels = []
        
        for i in range(zone.grid_count + 1):
            level = zone.price_lower + (i * interval)
            levels.append(self._round_price(level))
        
        return levels
    
    def setup_grid(self, debug_rounding: bool = False):
        """Setup initial grid orders"""
        try:
            if not self.active_zones:
                self.logger.error("No active zones to setup grid")
                return
            
            # Run rounding test for debugging (optional)
            if debug_rounding:
                self.test_rounding_precision()
            
            # Get current price
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            self.logger.info(f"Setting up grid at current price: ${current_price:.6f}")
            
            # Cancel any existing orders
            try:
                self.exchange.cancel_all_orders(self.symbol)
                time.sleep(1)
            except Exception as e:
                self.logger.warning(f"Error cancelling existing orders: {e}")
            
            # Setup orders for each active zone
            total_orders_placed = 0
            
            for zone_id, zone in self.active_zones.items():
                if not zone.active:
                    continue
                
                orders_placed = self._setup_zone_orders(zone, current_price)
                total_orders_placed += orders_placed
                
                self.logger.info(f"Zone {zone_id}: {orders_placed} orders placed")
            
            if total_orders_placed > 0:
                self.running = True
                self.logger.info(f"Grid setup complete: {total_orders_placed} total orders")
            else:
                self.running = False
                self.logger.warning("Grid setup failed: no orders placed")
                
        except Exception as e:
            self.logger.error(f"Error setting up grid: {e}")
            self.running = False
    
    def _setup_zone_orders(self, zone: GridZone, current_price: float) -> int:
        """Setup orders for a specific zone with correct grid logic"""
        try:
            grid_levels = self._get_grid_levels(zone)
            orders_placed = 0
            buy_orders = []
            sell_orders = []
            
            self.logger.info(f"Setting up zone orders:")
            self.logger.info(f"  Range: ${zone.price_lower:.6f} - ${zone.price_upper:.6f}")
            self.logger.info(f"  Current Price: ${current_price:.6f}")
            self.logger.info(f"  Grid Levels: {[f'${l:.6f}' for l in grid_levels]}")
            
            # Place orders at each grid level based on current price
            for i, level_price in enumerate(grid_levels):
                
                # Skip levels that are too close to current price (avoid immediate fills)
                price_diff_pct = abs(level_price - current_price) / current_price
                if price_diff_pct < 0.005:  # Skip if within 0.5% of current price
                    self.logger.debug(f"  Skipping level ${level_price:.6f} - too close to current price ({price_diff_pct:.1%})")
                    continue
                
                # Determine order type based on price relative to current price
                if level_price < current_price:
                    # Place buy order below current price
                    order_type = 'buy'
                    target_price = level_price
                    
                elif level_price > current_price:
                    # Place sell order above current price  
                    order_type = 'sell'
                    target_price = level_price
                    
                else:
                    # Skip if exactly at current price
                    continue
                
                # Calculate order amount
                amount = self._calculate_order_amount(target_price, zone.investment_per_grid)
                
                try:
                    # Place the order
                    order = self.exchange.create_limit_order(self.symbol, order_type, amount, target_price)
                    
                    order_info = {
                        'zone_id': zone.zone_id,
                        'type': order_type,
                        'price': target_price,
                        'amount': amount,
                        'grid_level': i,
                        'target_investment': zone.investment_per_grid,
                        'status': 'open'
                    }
                    
                    self.pending_orders[order['id']] = order_info
                    zone.orders[order['id']] = order_info
                    orders_placed += 1
                    
                    # Track for summary
                    order_summary = {
                        'level': i,
                        'price': target_price,
                        'amount': amount,
                        'value': amount * target_price,
                        'investment': (amount * target_price) / self.user_leverage
                    }
                    
                    if order_type == 'buy':
                        buy_orders.append(order_summary)
                    else:
                        sell_orders.append(order_summary)
                    
                    self.logger.debug(f"  ✅ {order_type.upper()} order: Level {i}, ${target_price:.6f} x {amount:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"  ❌ Failed to place {order_type} order at ${target_price:.6f}: {e}")
            
            # Print summary
            self.logger.info(f"📊 Order Placement Summary:")
            self.logger.info(f"  Total Orders: {orders_placed}")
            self.logger.info(f"  Buy Orders: {len(buy_orders)} (below ${current_price:.6f})")
            for order in buy_orders:
                self.logger.info(f"    Level {order['level']}: ${order['price']:.6f} x {order['amount']:.4f} = ${order['investment']:.2f}")
            
            self.logger.info(f"  Sell Orders: {len(sell_orders)} (above ${current_price:.6f})")
            for order in sell_orders:
                self.logger.info(f"    Level {order['level']}: ${order['price']:.6f} x {order['amount']:.4f} = ${order['investment']:.2f}")
            
            total_investment = sum(o['investment'] for o in buy_orders + sell_orders)
            self.logger.info(f"  Total Investment Allocated: ${total_investment:.2f}")
            self.logger.info(f"  Expected per Grid: ${zone.investment_per_grid:.2f}")
            
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error setting up zone orders: {e}")
            return 0
    
    def update_grid(self):
        """Main grid update logic with intelligent adaptation"""
        try:
            if not self.running:
                return
            
            # Get current market state
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # Update orders and positions
            self._update_orders_and_positions()
            
            # Check for grid adaptation needs
            if self.enable_grid_adaptation:
                self._check_and_adapt_grid(current_price)
            
            # Maintain counter orders for all positions
            self._maintain_counter_orders()
            
            # Update PnL
            self._update_pnl()
            
            # Check take profit / stop loss
            self._check_tp_sl()
            
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
    
    def _update_orders_and_positions(self):
        """Update order status and track new positions"""
        try:
            # Get current open orders from exchange
            open_orders = self.exchange.get_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            # Check each pending order
            filled_orders = []
            
            for order_id in list(self.pending_orders.keys()):
                if order_id not in open_order_ids:
                    # Order was filled or cancelled
                    try:
                        order_status = self.exchange.get_order_status(order_id, self.symbol)
                        
                        if order_status['status'] in ['filled', 'closed']:
                            filled_orders.append(order_id)
                            self._process_filled_order(order_id, order_status)
                        else:
                            # Order was cancelled
                            self._remove_cancelled_order(order_id)
                            
                    except Exception as e:
                        self.logger.error(f"Error checking order {order_id}: {e}")
            
            if filled_orders:
                self.logger.info(f"Processed {len(filled_orders)} filled orders")
                
        except Exception as e:
            self.logger.error(f"Error updating orders and positions: {e}")
    
    def _process_filled_order(self, order_id: str, order_status: Dict):
        """Process a filled order and create position"""
        try:
            order_info = self.pending_orders.get(order_id)
            if not order_info:
                self.logger.warning(f"No info found for filled order {order_id}")
                return
            
            # Create position from filled order
            position_id = str(uuid.uuid4())
            fill_price = float(order_status.get('average', order_info['price']))
            fill_amount = float(order_status.get('filled', order_info['amount']))
            
            position = GridPosition(
                position_id=position_id,
                grid_level=order_info['grid_level'],
                entry_price=fill_price,
                quantity=fill_amount,
                side='long' if order_info['type'] == 'buy' else 'short',
                entry_time=time.time()
            )
            
            # Store position
            self.all_positions[position_id] = position
            
            # Add to zone positions
            zone_id = order_info['zone_id']
            if zone_id in self.active_zones:
                self.active_zones[zone_id].positions[position_id] = position
            
            # Remove from pending orders
            del self.pending_orders[order_id]
            
            # Update trade counter
            self.total_trades += 1
            
            self.logger.info(f"New position created:")
            self.logger.info(f"  ID: {position_id[:8]}")
            self.logger.info(f"  Side: {position.side}")
            self.logger.info(f"  Price: ${fill_price:.6f}")
            self.logger.info(f"  Amount: {fill_amount:.4f}")
            self.logger.info(f"  Investment: ${(fill_amount * fill_price / self.user_leverage):.2f}")
            
        except Exception as e:
            self.logger.error(f"Error processing filled order {order_id}: {e}")
    
    def _remove_cancelled_order(self, order_id: str):
        """Remove cancelled order from tracking"""
        try:
            order_info = self.pending_orders.get(order_id)
            if order_info:
                # Remove from zone orders
                zone_id = order_info['zone_id']
                if zone_id in self.active_zones:
                    zone = self.active_zones[zone_id]
                    if order_id in zone.orders:
                        del zone.orders[order_id]
                
                # Remove from pending orders
                del self.pending_orders[order_id]
                
        except Exception as e:
            self.logger.error(f"Error removing cancelled order {order_id}: {e}")
    
    def _maintain_counter_orders(self):
        """Ensure all positions have corresponding exit orders"""
        try:
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            for position_id, position in self.all_positions.items():
                if position.exit_time is not None:
                    continue  # Position already closed
                
                if not position.has_counter_order:
                    self._create_counter_order(position, current_price)
                else:
                    # Check if counter order still exists
                    if position.counter_order_id:
                        if position.counter_order_id not in self.pending_orders:
                            # Counter order was filled or cancelled, create new one
                            position.has_counter_order = False
                            position.counter_order_id = None
                            self._create_counter_order(position, current_price)
                            
        except Exception as e:
            self.logger.error(f"Error maintaining counter orders: {e}")
    
    def _create_counter_order(self, position: GridPosition, current_price: float):
        """Create counter order for a position"""
        try:
            # Determine counter order side and price
            if position.side == 'long':
                # Long position needs sell order
                counter_side = 'sell'
                # Try to sell at next grid level or current price + small profit
                counter_price = position.entry_price * 1.005  # 0.5% profit minimum
                counter_price = max(counter_price, current_price * 1.002)  # Or 0.2% above current
            else:
                # Short position needs buy order
                counter_side = 'buy'
                counter_price = position.entry_price * 0.995  # 0.5% profit minimum
                counter_price = min(counter_price, current_price * 0.998)  # Or 0.2% below current
            
            counter_price = self._round_price(counter_price)
            
            # Create counter order
            order = self.exchange.create_limit_order(
                self.symbol, 
                counter_side, 
                position.quantity, 
                counter_price
            )
            
            # Track counter order
            order_info = {
                'zone_id': 'counter',
                'type': counter_side,
                'price': counter_price,
                'amount': position.quantity,
                'position_id': position.position_id,
                'status': 'open'
            }
            
            self.pending_orders[order['id']] = order_info
            
            # Update position
            position.has_counter_order = True
            position.counter_order_id = order['id']
            
            self.logger.debug(f"Counter order created for position {position.position_id[:8]}: "
                            f"{counter_side} {position.quantity:.4f} @ ${counter_price:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error creating counter order for position {position.position_id}: {e}")
    
    def _check_and_adapt_grid(self, current_price: float):
        """Check if grid needs adaptation and perform gradual migration"""
        try:
            # Throttle adaptation checks
            if time.time() - self.last_adaptation_time < 60:  # Max once per minute
                return
            
            self.last_adaptation_time = time.time()
            
            # Get market intelligence if enabled
            market_shift_needed = False
            shift_direction = None
            
            if self.enable_samig and hasattr(self, 'market_intel'):
                market_snapshot = self.market_intel.analyze_market(self.exchange)
                market_shift_needed, shift_direction = self._assess_market_shift(market_snapshot, current_price)
            
            # Check if price is significantly outside any active zone
            price_outside_zones = self._is_price_outside_zones(current_price)
            
            if price_outside_zones or market_shift_needed:
                self.logger.info(f"Grid adaptation triggered:")
                self.logger.info(f"  Current Price: ${current_price:.6f}")
                self.logger.info(f"  Price Outside Zones: {price_outside_zones}")
                self.logger.info(f"  Market Shift Needed: {market_shift_needed}")
                
                self._perform_gradual_grid_migration(current_price, shift_direction)
                
        except Exception as e:
            self.logger.error(f"Error in grid adaptation: {e}")
    
    def _assess_market_shift(self, market_snapshot: MarketSnapshot, current_price: float) -> Tuple[bool, Optional[str]]:
        """Assess if market conditions require grid shift"""
        try:
            # Strong trend detection
            if market_snapshot.trend_strength > 0.7:
                if market_snapshot.momentum > 0.3:
                    return True, 'up'
                elif market_snapshot.momentum < -0.3:
                    return True, 'down'
            
            # High volatility with directional bias
            if market_snapshot.volatility > 2.0 and abs(market_snapshot.momentum) > 0.4:
                direction = 'up' if market_snapshot.momentum > 0 else 'down'
                return True, direction
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error assessing market shift: {e}")
            return False, None
    
    def _is_price_outside_zones(self, current_price: float) -> bool:
        """Check if current price is outside all active zones"""
        try:
            for zone in self.active_zones.values():
                if not zone.active:
                    continue
                
                buffer = (zone.price_upper - zone.price_lower) * 0.05  # 5% buffer
                if (zone.price_lower - buffer) <= current_price <= (zone.price_upper + buffer):
                    return False  # Price is within at least one zone
            
            return True  # Price is outside all zones
            
        except Exception as e:
            self.logger.error(f"Error checking if price outside zones: {e}")
            return False
    
    def _perform_gradual_grid_migration(self, current_price: float, shift_direction: Optional[str]):
        """Perform gradual grid migration instead of sudden shift"""
        try:
            # Calculate new zone parameters
            zone_width = self.user_price_upper - self.user_price_lower
            
            if shift_direction == 'up':
                # Shift zone upward
                new_lower = max(self.user_price_lower, current_price - zone_width * 0.3)
                new_upper = new_lower + zone_width
            elif shift_direction == 'down':
                # Shift zone downward
                new_upper = min(self.user_price_upper, current_price + zone_width * 0.3)
                new_lower = new_upper - zone_width
            else:
                # Center around current price
                new_lower = current_price - zone_width * 0.5
                new_upper = current_price + zone_width * 0.5
            
            # Ensure new zone is reasonable
            new_lower = max(new_lower, current_price * 0.7)  # Not more than 30% below
            new_upper = min(new_upper, current_price * 1.3)  # Not more than 30% above
            
            new_lower = self._round_price(new_lower)
            new_upper = self._round_price(new_upper)
            
            # Create new zone
            self._create_migration_zone(new_lower, new_upper, current_price)
            
            # Schedule old zone deactivation (gradual phase-out)
            self._schedule_zone_deactivation()
            
        except Exception as e:
            self.logger.error(f"Error performing grid migration: {e}")
    
    def _create_migration_zone(self, new_lower: float, new_upper: float, current_price: float):
        """Create new zone for grid migration"""
        try:
            zone_id = f"zone_{int(time.time())}"
            
            migration_zone = GridZone(
                zone_id=zone_id,
                price_lower=new_lower,
                price_upper=new_upper,
                grid_count=self.user_grid_number,
                investment_per_grid=self.user_investment_per_grid
            )
            
            self.active_zones[zone_id] = migration_zone
            
            # Setup orders for new zone
            orders_placed = self._setup_zone_orders(migration_zone, current_price)
            
            self.logger.info(f"Created migration zone: {zone_id}")
            self.logger.info(f"  Range: ${new_lower:.6f} - ${new_upper:.6f}")
            self.logger.info(f"  Orders placed: {orders_placed}")
            
        except Exception as e:
            self.logger.error(f"Error creating migration zone: {e}")
    
    def _schedule_zone_deactivation(self):
        """Schedule deactivation of old zones that have no positions"""
        try:
            zones_to_deactivate = []
            
            for zone_id, zone in self.active_zones.items():
                if not zone.active:
                    continue
                
                # Check if zone has active positions
                active_positions = sum(1 for pos in zone.positions.values() 
                                     if pos.exit_time is None)
                
                # Check zone age
                zone_age = time.time() - zone.creation_time
                
                # Deactivate old zones with no positions after 5 minutes
                if active_positions == 0 and zone_age > 300:
                    zones_to_deactivate.append(zone_id)
            
            for zone_id in zones_to_deactivate:
                self._deactivate_zone(zone_id)
                
        except Exception as e:
            self.logger.error(f"Error scheduling zone deactivation: {e}")
    
    def _deactivate_zone(self, zone_id: str):
        """Deactivate a zone and cancel its orders"""
        try:
            if zone_id not in self.active_zones:
                return
            
            zone = self.active_zones[zone_id]
            zone.active = False
            
            # Cancel zone orders
            orders_cancelled = 0
            for order_id in list(zone.orders.keys()):
                try:
                    self.exchange.cancel_order(order_id, self.symbol)
                    orders_cancelled += 1
                    
                    # Remove from tracking
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    del zone.orders[order_id]
                    
                except Exception as e:
                    self.logger.warning(f"Error cancelling order {order_id}: {e}")
            
            self.logger.info(f"Deactivated zone {zone_id}: {orders_cancelled} orders cancelled")
            
        except Exception as e:
            self.logger.error(f"Error deactivating zone {zone_id}: {e}")
    
    def _update_pnl(self):
        """Update total PnL from all positions"""
        try:
            total_unrealized = 0.0
            total_realized = 0.0
            
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            for position in self.all_positions.values():
                if position.exit_time is None:
                    # Unrealized PnL
                    if position.side == 'long':
                        unrealized = (current_price - position.entry_price) * position.quantity
                    else:
                        unrealized = (position.entry_price - current_price) * position.quantity
                    
                    position.unrealized_pnl = unrealized
                    total_unrealized += unrealized
                else:
                    # Realized PnL
                    total_realized += position.realized_pnl
            
            self.total_pnl = total_realized + total_unrealized
            
        except Exception as e:
            self.logger.error(f"Error updating PnL: {e}")
    
    def _check_tp_sl(self):
        """Check take profit and stop loss conditions"""
        try:
            if self.user_total_investment <= 0:
                return
            
            pnl_percentage = (self.total_pnl / self.user_total_investment) * 100
            
            if pnl_percentage >= self.take_profit_pnl:
                self.logger.info(f"Take profit reached: {pnl_percentage:.2f}%")
                self.stop_grid()
            elif pnl_percentage <= -self.stop_loss_pnl:
                self.logger.info(f"Stop loss reached: {pnl_percentage:.2f}%")
                self.stop_grid()
                
        except Exception as e:
            self.logger.error(f"Error checking TP/SL: {e}")
    
    def stop_grid(self):
        """Stop the grid strategy"""
        try:
            if not self.running:
                return
            
            self.logger.info(f"Stopping grid strategy for {self.symbol}")
            
            # Cancel all pending orders
            try:
                self.exchange.cancel_all_orders(self.symbol)
                self.pending_orders.clear()
                
                for zone in self.active_zones.values():
                    zone.orders.clear()
                    zone.active = False
                    
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
            
            # Close all open positions
            try:
                for position in self.all_positions.values():
                    if position.exit_time is None:
                        self._close_position_at_market(position)
                        
            except Exception as e:
                self.logger.error(f"Error closing positions: {e}")
            
            self.running = False
            self.logger.info(f"Grid strategy stopped for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error stopping grid: {e}")
            self.running = False
    
    def _close_position_at_market(self, position: GridPosition):
        """Close position at market price"""
        try:
            side = 'sell' if position.side == 'long' else 'buy'
            
            order = self.exchange.create_market_order(
                self.symbol, 
                side, 
                position.quantity
            )
            
            # Mark position as closed
            position.exit_time = time.time()
            position.exit_price = float(order.get('average', 0))
            
            # Calculate realized PnL
            if position.side == 'long':
                position.realized_pnl = (position.exit_price - position.entry_price) * position.quantity
            else:
                position.realized_pnl = (position.entry_price - position.exit_price) * position.quantity
            
            self.logger.info(f"Closed position {position.position_id[:8]} at market: "
                           f"PnL = ${position.realized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position {position.position_id}: {e}")
    
    def test_rounding_precision(self):
        """Test rounding precision for debugging"""
        try:
            self.logger.info(f"Testing rounding precision for {self.symbol}:")
            self.logger.info(f"Exchange precision - Price: {self.price_precision}, Amount: {self.amount_precision}")
            
            # Test various price points
            test_prices = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            
            for test_price in test_prices:
                rounded_price = self._round_price(test_price)
                self.logger.info(f"  Price: {test_price} -> {rounded_price}")
                
                # Test amount calculation
                amount = self._calculate_order_amount(test_price, self.user_investment_per_grid)
                self.logger.info(f"    Amount: {amount} (for ${self.user_investment_per_grid:.2f} investment)")
                
                # Check if order value makes sense
                order_value = rounded_price * amount
                actual_investment = order_value / self.user_leverage
                self.logger.info(f"    Order value: ${order_value:.2f}, Investment: ${actual_investment:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error in rounding test: {e}")
    
    def get_status(self) -> Dict:
        """Get comprehensive grid status"""
        try:
            # Count active positions and orders
            active_positions = sum(1 for pos in self.all_positions.values() if pos.exit_time is None)
            active_orders = len(self.pending_orders)
            active_zones = sum(1 for zone in self.active_zones.values() if zone.active)
            
            # Calculate position breakdown
            long_positions = sum(1 for pos in self.all_positions.values() 
                               if pos.exit_time is None and pos.side == 'long')
            short_positions = active_positions - long_positions
            
            # Calculate total investment in positions
            total_position_value = 0.0
            for pos in self.all_positions.values():
                if pos.exit_time is None:
                    total_position_value += pos.quantity * pos.entry_price / self.user_leverage
            
            status = {
                # Immutable user parameters
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'display_symbol': self.original_symbol,
                'user_price_lower': self.user_price_lower,
                'user_price_upper': self.user_price_upper,
                'user_grid_number': self.user_grid_number,
                'user_total_investment': self.user_total_investment,
                'user_investment_per_grid': self.user_investment_per_grid,
                
                # Current state
                'price_lower': self.user_price_lower,  # For display compatibility
                'price_upper': self.user_price_upper,  # For display compatibility
                'grid_number': self.user_grid_number,  # For display compatibility
                'investment': self.user_total_investment,
                'investment_per_grid': self.user_investment_per_grid,
                'leverage': self.user_leverage,
                
                # Strategy settings
                'take_profit_pnl': self.take_profit_pnl,
                'stop_loss_pnl': self.stop_loss_pnl,
                'enable_grid_adaptation': self.enable_grid_adaptation,
                'enable_samig': self.enable_samig,
                
                # Current status
                'running': self.running,
                'active_zones': active_zones,
                'active_positions': active_positions,
                'long_positions': long_positions,
                'short_positions': short_positions,
                'orders_count': active_orders,
                'trades_count': self.total_trades,
                'total_position_value': total_position_value,
                
                # PnL
                'pnl': self.total_pnl,
                'pnl_percentage': (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0,
                
                # Zone information
                'zones': [
                    {
                        'zone_id': zone.zone_id,
                        'range': f"${zone.price_lower:.6f} - ${zone.price_upper:.6f}",
                        'active': zone.active,
                        'positions': len([p for p in zone.positions.values() if p.exit_time is None]),
                        'orders': len(zone.orders)
                    }
                    for zone in self.active_zones.values()
                ]
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting grid status: {e}")
            return {}