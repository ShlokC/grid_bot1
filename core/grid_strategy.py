"""
Enhanced Grid Trading Strategy with Strict Investment and Grid Limits
Fixed critical bugs causing excessive investment and grid creation.
"""
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque
import traceback
from core.exchange import Exchange

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
    # KAMA-based directional intelligence
    kama_value: float = 0.0
    kama_direction: str = 'neutral'  # 'up', 'down', 'neutral'
    kama_strength: float = 0.0  # 0-1, strength of directional move
    directional_bias: float = 0.0  # -1 to 1, negative=bearish, positive=bullish

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
        
        return max(0.5, min(3.0, volatility * 100))
    
    def _calculate_momentum(self) -> float:
        """Calculate price momentum"""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = list(self.price_history)
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-20:]) / 20
        
        if long_ma > 0:
            momentum = (short_ma - long_ma) / long_ma
            return max(-1.0, min(1.0, momentum * 10))
        
        return 0.0
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength (0 = ranging, 1 = strong trend)"""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = list(self.price_history)
        
        up_moves = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        down_moves = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
        total_moves = up_moves + down_moves
        
        if total_moves == 0:
            return 0.0
        
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
        """Initialize grid strategy with strict investment and grid limits"""
        
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
        
        # CRITICAL: Investment tracking and limits
        self.total_investment_used = 0.0  # Track actual investment used
        self.max_concurrent_zones = 1  # Limit concurrent zones to prevent multiplication
        self.max_orders_per_zone = self.user_grid_number  # Strict order limit per zone
        
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
        
        self.logger.info(f"Grid Strategy Initialized with STRICT LIMITS:")
        self.logger.info(f"  Symbol: {symbol}")
        self.logger.info(f"  Price Range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
        self.logger.info(f"  Grid Count: {self.user_grid_number} (MAXIMUM)")
        self.logger.info(f"  Investment per Grid: ${self.user_investment_per_grid:.2f}")
        self.logger.info(f"  Total Investment: ${self.user_total_investment:.2f} (MAXIMUM)")
        self.logger.info(f"  Leverage: {self.user_leverage}x")
        self.logger.info(f"  Max Concurrent Zones: {self.max_concurrent_zones}")
    
    def _fetch_market_info(self):
        """Fetch market information for the trading pair"""
        try:
            market_info = self.exchange.get_market_info(self.symbol)
            
            precision_info = market_info.get('precision', {})
            price_precision = precision_info.get('price', 6)
            amount_precision = precision_info.get('amount', 6)
            
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
            
            limits = market_info.get('limits', {})
            amount_limits = limits.get('amount', {})
            cost_limits = limits.get('cost', {})
            
            self.min_amount = float(amount_limits.get('min', 0.0001))
            self.min_cost = float(cost_limits.get('min', 1.0))
            
            self.logger.info(f"Market info for {self.symbol}:")
            self.logger.info(f"  Price precision: {self.price_precision} decimals")
            self.logger.info(f"  Amount precision: {self.amount_precision} decimals")  
            self.logger.info(f"  Min amount: {self.min_amount}")
            self.logger.info(f"  Min cost: {self.min_cost}")
            
        except Exception as e:
            self.logger.error(f"Error fetching market info for {self.symbol}: {e}")
            self.price_precision = 6
            self.amount_precision = 6
            self.min_amount = 0.0001
            self.min_cost = 1.0
            self.logger.info(f"Using fallback market info for {self.symbol}")
    
    def _validate_investment_limits(self) -> bool:
        """Validate that we haven't exceeded investment limits"""
        try:
            if self.total_investment_used >= self.user_total_investment:
                self.logger.warning(f"Investment limit reached: ${self.total_investment_used:.2f} / ${self.user_total_investment:.2f}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating investment limits: {e}")
            return False
    
    def _validate_zone_limits(self) -> bool:
        """Validate that we haven't exceeded zone limits"""
        try:
            active_zone_count = sum(1 for zone in self.active_zones.values() if zone.active)
            if active_zone_count >= self.max_concurrent_zones:
                self.logger.warning(f"Zone limit reached: {active_zone_count} / {self.max_concurrent_zones}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating zone limits: {e}")
            return False
    
    def _create_initial_zone(self):
        """Create the initial trading zone with strict limits"""
        zone_id = f"zone_{int(time.time())}"
        initial_zone = GridZone(
            zone_id=zone_id,
            price_lower=self.user_price_lower,
            price_upper=self.user_price_upper,
            grid_count=self.user_grid_number,  # Exact user specification
            investment_per_grid=self.user_investment_per_grid
        )
        self.active_zones[zone_id] = initial_zone
        
        self.logger.info(f"Created initial zone: {zone_id}")
        self.logger.info(f"  Range: ${initial_zone.price_lower:.6f} - ${initial_zone.price_upper:.6f}")
        self.logger.info(f"  Grid count: {initial_zone.grid_count} (STRICT LIMIT)")
    
    def _round_price(self, price: float) -> float:
        """Round price with intelligent precision detection"""
        try:
            base_precision = int(self.price_precision) if hasattr(self, 'price_precision') else 0
            
            if price < 0.00001:
                smart_precision = 8
            elif price < 0.0001:
                smart_precision = 7
            elif price < 0.001:
                smart_precision = 6
            elif price < 0.01:
                smart_precision = 5
            elif price < 0.1:
                smart_precision = 4
            elif price < 1.0:
                smart_precision = 3
            elif price < 100:
                smart_precision = 2
            else:
                smart_precision = 1
            
            final_precision = max(base_precision, smart_precision, 2)
            rounded_price = float(f"{price:.{final_precision}f}")
            
            return rounded_price
            
        except Exception as e:
            self.logger.error(f"Error in _round_price for {price}: {e}")
            if price >= 1:
                return float(f"{price:.2f}")
            elif price >= 0.01:
                return float(f"{price:.4f}")
            else:
                return float(f"{price:.6f}")
    
    def _round_amount(self, amount: float) -> float:
        """Round amount with intelligent precision detection"""
        try:
            base_precision = int(self.amount_precision) if hasattr(self, 'amount_precision') else 0
            
            if amount < 0.0001:
                smart_precision = 8
            elif amount < 0.001:
                smart_precision = 7
            elif amount < 0.01:
                smart_precision = 6
            elif amount < 1:
                smart_precision = 5
            elif amount < 100:
                smart_precision = 4
            else:
                smart_precision = 3
            
            final_precision = max(base_precision, smart_precision, 4)
            rounded_amount = float(f"{amount:.{final_precision}f}")
            
            return rounded_amount
            
        except Exception as e:
            self.logger.error(f"Error in _round_amount for {amount}: {e}")
            return float(f"{amount:.6f}")
    
    def _calculate_order_amount_fixed(self, price: float, investment_per_grid: float) -> float:
        """CORRECTED: Calculate order amount with proper leverage for exchange requirements"""
        try:
            rounded_price = self._round_price(price)
            
            if rounded_price <= 0:
                self.logger.error(f"Price rounded to zero: {price} -> {rounded_price}")
                rounded_price = price if price > 0 else 0.01
            
            # CORRECTED APPROACH:
            # 1. Use leverage to calculate notional position size (for exchange requirements)
            # 2. Track actual margin used (for budget control)
            
            # Your actual margin/investment per grid
            actual_margin = investment_per_grid  # $2 per grid (your money at risk)
            
            # Notional position size (what exchange sees) 
            notional_value = actual_margin * self.user_leverage  # $2 × 20x = $40
            
            # Calculate quantity based on notional value
            quantity = notional_value / rounded_price
            rounded_amount = self._round_amount(quantity)
            
            # Validate exchange minimum requirements
            if rounded_amount < self.min_amount:
                self.logger.warning(f"Amount {rounded_amount} below minimum {self.min_amount}, adjusting")
                rounded_amount = self.min_amount
            
            # Check notional value meets exchange minimum ($5 for Binance)
            order_notional = rounded_price * rounded_amount
            if order_notional < self.min_cost:
                self.logger.warning(f"Order notional ${order_notional:.2f} below minimum ${self.min_cost:.2f}, adjusting")
                rounded_amount = self.min_cost / rounded_price
                rounded_amount = self._round_amount(rounded_amount)
                order_notional = rounded_price * rounded_amount
            
            # Calculate actual margin required (this is what we track for budget)
            actual_margin_used = order_notional / self.user_leverage
            
            self.logger.debug(f"Order calculation (CORRECTED):")
            self.logger.debug(f"  Target Margin: ${actual_margin:.2f}")
            self.logger.debug(f"  Leverage: {self.user_leverage}x")
            self.logger.debug(f"  Notional Value: ${notional_value:.2f}")
            self.logger.debug(f"  Price: {rounded_price:.8f}")
            self.logger.debug(f"  Quantity: {rounded_amount:.8f}")
            self.logger.debug(f"  Order Notional: ${order_notional:.2f}")
            self.logger.debug(f"  Actual Margin Used: ${actual_margin_used:.2f}")
            
            return rounded_amount
            
        except Exception as e:
            self.logger.error(f"Error calculating order amount for price ${price:.8f}: {e}")
            return max(self.min_amount, 0.0001)
    
    def _get_grid_levels(self, zone: GridZone) -> List[float]:
        """Calculate EXACT grid levels for a zone - no more than specified"""
        try:
            # Use the exact grid count specified by user
            grid_count = min(zone.grid_count, self.user_grid_number)  # Double safety check
            
            if grid_count <= 0:
                self.logger.error(f"Invalid grid count: {grid_count}")
                return []
            
            interval = (zone.price_upper - zone.price_lower) / grid_count
            levels = []
            
            # Generate exactly grid_count + 1 levels (for grid_count intervals)
            for i in range(grid_count + 1):
                level = zone.price_lower + (i * interval)
                levels.append(self._round_price(level))
            
            self.logger.debug(f"Generated {len(levels)} grid levels for {grid_count} grids")
            return levels
            
        except Exception as e:
            self.logger.error(f"Error generating grid levels: {e}")
            return []
    
    def setup_grid(self):
        """Setup initial grid orders with STRICT limits"""
        try:
            if not self.active_zones:
                self.logger.error("No active zones to setup grid")
                return
            
            # Get current price
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            self.logger.info(f"Setting up grid at current price: ${current_price:.6f}")
            self.logger.info(f"STRICT LIMITS: Max Investment=${self.user_total_investment:.2f}, Max Grids={self.user_grid_number}")
            
            # Cancel any existing orders
            try:
                self.exchange.cancel_all_orders(self.symbol)
                time.sleep(1)
            except Exception as e:
                self.logger.warning(f"Error cancelling existing orders: {e}")
            
            # Reset investment tracking
            self.total_investment_used = 0.0
            
            # Setup orders for ONLY the first active zone (prevent multiple zones)
            total_orders_placed = 0
            zones_processed = 0
            
            for zone_id, zone in self.active_zones.items():
                if not zone.active:
                    continue
                
                if zones_processed >= self.max_concurrent_zones:
                    self.logger.warning(f"Reached max concurrent zones limit: {self.max_concurrent_zones}")
                    break
                
                orders_placed = self._setup_zone_orders_strict(zone, current_price)
                total_orders_placed += orders_placed
                zones_processed += 1
                
                self.logger.info(f"Zone {zone_id}: {orders_placed} orders placed")
            
            if total_orders_placed > 0:
                self.running = True
                self.logger.info(f"Grid setup complete: {total_orders_placed} total orders")
                self.logger.info(f"Total investment used: ${self.total_investment_used:.2f} / ${self.user_total_investment:.2f}")
            else:
                self.running = False
                self.logger.warning("Grid setup failed: no orders placed")
                
        except Exception as e:
            self.logger.error(f"Error setting up grid: {e}")
            self.running = False
    
    def _setup_zone_orders_strict(self, zone: GridZone, current_price: float) -> int:
        """Setup orders for a zone with position-aware dynamic market intelligence"""
        try:
            grid_levels = self._get_grid_levels(zone)
            orders_placed = 0
            
            # **CRITICAL: Analyze current exposure first**
            exposure_analysis = self._analyze_current_exposure()
            
            self.logger.info(f"📊 Current Position Analysis:")
            self.logger.info(f"  Open Positions: {exposure_analysis['long_positions']}L / {exposure_analysis['short_positions']}S")
            self.logger.info(f"  Pending Orders: {exposure_analysis['buy_orders']}B / {exposure_analysis['sell_orders']}S")
            self.logger.info(f"  Total Commitment: {exposure_analysis['total_commitment']}/{self.max_orders_per_zone}")
            self.logger.info(f"  Net Exposure Bias: {exposure_analysis['total_exposure_bias']} (+ve=bullish, -ve=bearish)")
            self.logger.info(f"  Unrealized PnL: ${exposure_analysis['unrealized_pnl']:.2f}")
            self.logger.info(f"  Covered Levels: {len(exposure_analysis['covered_levels'])}")
            
            # Check if we have remaining capacity
            if exposure_analysis['total_commitment'] >= self.max_orders_per_zone:
                self.logger.warning(f"Maximum commitment reached: {exposure_analysis['total_commitment']}/{self.max_orders_per_zone}")
                return 0
            
            # Check position limit for safety (20x leverage limit)
            if exposure_analysis['position_count'] >= exposure_analysis['max_positions']:
                self.logger.warning(f"Maximum position limit reached: {exposure_analysis['position_count']}/{exposure_analysis['max_positions']}")
                return 0
            
            # Get market intelligence for dynamic parameters
            directional_bias = 0.0
            volatility_regime = 1.0
            trend_strength = 0.0
            
            if self.enable_samig and hasattr(self, 'market_intel'):
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    directional_bias = market_snapshot.directional_bias
                    volatility_regime = self.market_intel.current_volatility_regime
                    trend_strength = self.market_intel.current_trend_strength
                    
                    kama_info = f"KAMA: {market_snapshot.kama_direction} (bias: {directional_bias:.3f}, strength: {market_snapshot.kama_strength:.2f})"
                    self.logger.info(f"🧠 SAMIG Intelligence: {kama_info}")
                    self.logger.info(f"📈 Market Regime: vol={volatility_regime:.2f}, trend={trend_strength:.2f}")
                except Exception as e:
                    self.logger.warning(f"SAMIG analysis failed: {e}")
            
            self.logger.info(f"Setting up zone orders with position-aware intelligence:")
            self.logger.info(f"  Current Price: ${current_price:.6f}")
            self.logger.info(f"  Zone Range: ${zone.price_lower:.6f} - ${zone.price_upper:.6f}") 
            self.logger.info(f"  Raw Market Bias: {directional_bias:.3f} (-1=bearish, +1=bullish)")
            
            # **CRITICAL: Calculate position-aware distribution**
            buy_orders_target, sell_orders_target = self._calculate_position_aware_distribution(
                grid_levels, current_price, directional_bias, exposure_analysis
            )
            
            if buy_orders_target == 0 and sell_orders_target == 0:
                self.logger.warning("No orders to place after position-aware analysis")
                return 0
            
            self.logger.info(f"  Position-Aware Targets: {buy_orders_target}B / {sell_orders_target}S")
            
            # Calculate dynamic minimum gap based on market conditions and existing positions
            zone_width = zone.price_upper - zone.price_lower
            zone_width_pct = zone_width / current_price
            
            # Base minimum gap
            base_min_gap = 0.0005 if zone_width_pct < 0.02 else 0.001
            
            # Adjust based on volatility and trend (same as before)
            if volatility_regime > 1.5:
                vol_adjustment = 0.7
            elif volatility_regime < 0.7:
                vol_adjustment = 1.3
            else:
                vol_adjustment = 1.0
            
            if trend_strength > 0.7:
                trend_adjustment = 0.8
            elif trend_strength < 0.3:
                trend_adjustment = 1.2
            else:
                trend_adjustment = 1.0
            
            # **NEW: Adjust gap based on existing position density**
            density_adjustment = 1.0
            if len(exposure_analysis['covered_levels']) > self.user_grid_number * 0.6:
                # Many levels already covered, use wider gaps to avoid clustering
                density_adjustment = 1.4
                self.logger.info(f"High position density detected, using wider gaps")
            
            dynamic_min_gap_pct = base_min_gap * vol_adjustment * trend_adjustment * density_adjustment
            dynamic_min_gap_pct = max(0.0002, min(0.01, dynamic_min_gap_pct))
            
            self.logger.info(f"Dynamic gap: {dynamic_min_gap_pct:.4f} (base: {base_min_gap:.4f}, "
                           f"adjustments: vol={vol_adjustment:.2f}, trend={trend_adjustment:.2f}, density={density_adjustment:.2f})")
            
            buy_orders_placed = 0
            sell_orders_placed = 0
            
            # Sort grid levels by distance from current price (prioritize closer levels)
            grid_levels_with_distance = [(level, abs(level - current_price)) for level in grid_levels]
            grid_levels_with_distance.sort(key=lambda x: x[1])
            
            for level_price, distance in grid_levels_with_distance:
                # Check limits
                if orders_placed >= min(buy_orders_target + sell_orders_target, self.max_orders_per_zone):
                    break
                
                if not self._validate_investment_limits():
                    break
                
                # **CRITICAL: Skip if level already covered**
                if round(level_price, 6) in exposure_analysis['covered_levels']:
                    self.logger.debug(f"Skip ${level_price:.6f}: already covered by existing position/order")
                    continue
                
                # Check dynamic minimum gap
                price_diff_pct = abs(level_price - current_price) / current_price
                if price_diff_pct < dynamic_min_gap_pct:
                    continue
                
                # **CRITICAL: Check gap against existing positions/orders**
                too_close_to_existing = False
                min_gap_from_existing = current_price * 0.003  # 0.3% minimum gap from existing
                
                for existing_level in exposure_analysis['covered_levels']:
                    if abs(level_price - existing_level) < min_gap_from_existing:
                        too_close_to_existing = True
                        break
                
                if too_close_to_existing:
                    self.logger.debug(f"Skip ${level_price:.6f}: too close to existing position/order")
                    continue
                
                # Determine order type with position-aware intelligence
                should_place_order = False
                order_type = None
                priority = ""
                
                if level_price < current_price:
                    # Buy order candidate
                    if buy_orders_placed < buy_orders_target:
                        order_type = 'buy'
                        should_place_order = True
                        if directional_bias > 0:
                            priority = f"High (market+position analysis: {buy_orders_placed+1}/{buy_orders_target})"
                        else:
                            priority = f"Medium (position rebalancing: {buy_orders_placed+1}/{buy_orders_target})"
                
                elif level_price > current_price:
                    # Sell order candidate
                    if sell_orders_placed < sell_orders_target:
                        order_type = 'sell'
                        should_place_order = True
                        if directional_bias < 0:
                            priority = f"High (market+position analysis: {sell_orders_placed+1}/{sell_orders_target})"
                        else:
                            priority = f"Medium (position rebalancing: {sell_orders_placed+1}/{sell_orders_target})"
                
                if not should_place_order or not order_type:
                    continue
                
                # Calculate order amount and place order
                amount = self._calculate_order_amount_fixed(level_price, zone.investment_per_grid)
                order_notional = level_price * amount
                order_margin = order_notional / self.user_leverage
                
                # Check investment limit
                if self.total_investment_used + order_margin > self.user_total_investment:
                    remaining = self.user_total_investment - self.total_investment_used
                    if remaining > (self.min_cost / self.user_leverage):
                        remaining_notional = remaining * self.user_leverage
                        amount = remaining_notional / level_price
                        amount = self._round_amount(amount)
                        order_margin = remaining
                    else:
                        break
                
                # Place order
                try:
                    order = self.exchange.create_limit_order(self.symbol, order_type, amount, level_price)
                    
                    order_info = {
                        'zone_id': zone.zone_id,
                        'type': order_type,
                        'price': level_price,
                        'amount': amount,
                        'grid_level': 0,  # Will be updated based on actual level
                        'target_investment': zone.investment_per_grid,
                        'actual_margin_used': order_margin,
                        'notional_value': level_price * amount,
                        'directional_priority': priority,
                        'market_regime': f"vol:{volatility_regime:.2f},trend:{trend_strength:.2f}",
                        'position_aware': True,
                        'existing_exposure_bias': exposure_analysis['total_exposure_bias'],
                        'status': 'open'
                    }
                    
                    self.pending_orders[order['id']] = order_info
                    zone.orders[order['id']] = order_info
                    orders_placed += 1
                    self.total_investment_used += order_margin
                    
                    if order_type == 'buy':
                        buy_orders_placed += 1
                    else:
                        sell_orders_placed += 1
                    
                    self.logger.info(f"✅ {order_type.upper()}: ${level_price:.6f} "
                                   f"Margin: ${order_margin:.2f} Priority: {priority}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed {order_type} at ${level_price:.6f}: {e}")
            
            # Final summary
            final_exposure = exposure_analysis['total_exposure_bias'] + buy_orders_placed - sell_orders_placed
            
            self.logger.info(f"Position-aware zone setup complete:")
            self.logger.info(f"  Orders placed: {orders_placed} ({buy_orders_placed}B / {sell_orders_placed}S)")
            self.logger.info(f"  Exposure change: {exposure_analysis['total_exposure_bias']} → {final_exposure}")
            self.logger.info(f"  Market-adaptive strategy successfully considered existing positions")
            self.logger.info(f"  Investment used: ${self.total_investment_used:.2f} / ${self.user_total_investment:.2f}")
            
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error setting up position-aware zone orders: {e}")
            return 0
    
    def _analyze_current_exposure(self) -> Dict[str, Any]:
        """Analyze current positions and orders to understand exposure"""
        try:
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            # Analyze open positions
            long_positions = []
            short_positions = []
            total_position_value = 0.0
            unrealized_pnl = 0.0
            
            for position in self.all_positions.values():
                if position.exit_time is None:  # Open position
                    position_value = position.quantity * position.entry_price
                    total_position_value += position_value
                    
                    # Calculate unrealized PnL
                    if position.side == 'long':
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                        long_positions.append(position)
                    else:
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                        short_positions.append(position)
                    
                    unrealized_pnl += position.unrealized_pnl
            
            # Analyze pending orders
            buy_orders = []
            sell_orders = []
            pending_buy_value = 0.0
            pending_sell_value = 0.0
            
            for order_info in self.pending_orders.values():
                if order_info['status'] == 'open':
                    order_value = order_info['price'] * order_info['amount']
                    
                    if order_info['type'] == 'buy':
                        buy_orders.append(order_info)
                        pending_buy_value += order_value
                    else:
                        sell_orders.append(order_info)
                        pending_sell_value += order_value
            
            # Calculate exposure metrics
            net_position_bias = len(long_positions) - len(short_positions)
            net_order_bias = len(buy_orders) - len(sell_orders)
            total_exposure_bias = net_position_bias + net_order_bias
            
            # Calculate risk metrics
            position_count = len(long_positions) + len(short_positions)
            order_count = len(buy_orders) + len(sell_orders)
            total_commitment = position_count + order_count
            
            # Calculate price levels already covered
            covered_levels = set()
            for position in self.all_positions.values():
                if position.exit_time is None:
                    covered_levels.add(round(position.entry_price, 6))
            
            for order_info in self.pending_orders.values():
                if order_info['status'] == 'open':
                    covered_levels.add(round(order_info['price'], 6))
            
            exposure_analysis = {
                'current_price': current_price,
                'long_positions': len(long_positions),
                'short_positions': len(short_positions),
                'buy_orders': len(buy_orders),
                'sell_orders': len(sell_orders),
                'net_position_bias': net_position_bias,  # +ve = more long, -ve = more short
                'net_order_bias': net_order_bias,       # +ve = more buy orders, -ve = more sell orders
                'total_exposure_bias': total_exposure_bias,
                'position_count': position_count,
                'order_count': order_count,
                'total_commitment': total_commitment,
                'total_position_value': total_position_value,
                'unrealized_pnl': unrealized_pnl,
                'covered_levels': covered_levels,
                'pending_buy_value': pending_buy_value,
                'pending_sell_value': pending_sell_value,
                'max_positions': 8,  # Safe limit for 20x leverage
                'exposure_ratio': total_commitment / self.user_grid_number if self.user_grid_number > 0 else 0
            }
            
            return exposure_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing current exposure: {e}")
            return {
                'current_price': 0, 'long_positions': 0, 'short_positions': 0,
                'buy_orders': 0, 'sell_orders': 0, 'net_position_bias': 0,
                'net_order_bias': 0, 'total_exposure_bias': 0, 'position_count': 0,
                'order_count': 0, 'total_commitment': 0, 'covered_levels': set(),
                'exposure_ratio': 0
            }
    
    def _calculate_position_aware_distribution(self, grid_levels: List[float], current_price: float, 
                                             directional_bias: float, exposure_analysis: Dict) -> Tuple[int, int]:
        """Calculate order distribution considering existing positions and orders"""
        try:
            # Get market intelligence
            volatility_regime = 1.0
            trend_strength = 0.0
            
            if self.enable_samig and hasattr(self, 'market_intel'):
                try:
                    volatility_regime = self.market_intel.current_volatility_regime
                    trend_strength = self.market_intel.current_trend_strength
                except Exception as e:
                    self.logger.warning(f"Could not get market intelligence: {e}")
            
            # Count available levels (excluding already covered ones)
            available_buy_levels = [level for level in grid_levels 
                                  if level < current_price and round(level, 6) not in exposure_analysis['covered_levels']]
            available_sell_levels = [level for level in grid_levels 
                                   if level > current_price and round(level, 6) not in exposure_analysis['covered_levels']]
            
            total_available_levels = len(available_buy_levels) + len(available_sell_levels)
            
            if total_available_levels == 0:
                self.logger.warning("No available levels - all grid levels already covered")
                return 0, 0
            
            # Calculate remaining capacity
            max_total_commitment = min(self.max_orders_per_zone, exposure_analysis['max_positions'])
            remaining_capacity = max_total_commitment - exposure_analysis['total_commitment']
            
            if remaining_capacity <= 0:
                self.logger.warning(f"No remaining capacity: {exposure_analysis['total_commitment']}/{max_total_commitment}")
                return 0, 0
            
            # Dynamic bias threshold calculation (same as before)
            base_threshold = 0.2
            vol_adjustment = volatility_regime * 0.1
            trend_adjustment = -trend_strength * 0.15
            dynamic_threshold = max(0.1, min(0.5, base_threshold + vol_adjustment + trend_adjustment))
            
            # **CRITICAL: Adjust directional bias based on current exposure**
            current_exposure_bias = exposure_analysis['total_exposure_bias']
            exposure_adjustment = 0.0
            
            # If we're already heavily biased in one direction, reduce that bias
            if current_exposure_bias > 2:  # Too many long positions/orders
                exposure_adjustment = -min(0.4, current_exposure_bias * 0.1)  # Reduce bullish bias
                self.logger.info(f"Heavy long exposure detected ({current_exposure_bias}), reducing bullish bias by {-exposure_adjustment:.2f}")
                
            elif current_exposure_bias < -2:  # Too many short positions/orders
                exposure_adjustment = min(0.4, abs(current_exposure_bias) * 0.1)  # Reduce bearish bias
                self.logger.info(f"Heavy short exposure detected ({current_exposure_bias}), reducing bearish bias by {exposure_adjustment:.2f}")
            
            # Apply exposure adjustment to directional bias
            adjusted_directional_bias = directional_bias + exposure_adjustment
            adjusted_directional_bias = max(-1.0, min(1.0, adjusted_directional_bias))
            
            if abs(exposure_adjustment) > 0.05:
                self.logger.info(f"Bias adjustment: {directional_bias:.3f} → {adjusted_directional_bias:.3f} (exposure correction)")
            
            # Calculate target ratios based on adjusted bias
            abs_bias = abs(adjusted_directional_bias)
            
            if abs_bias > dynamic_threshold * 2:  # Very strong bias
                if adjusted_directional_bias > 0:  # Bullish
                    buy_ratio = 0.7 + min(0.15, (abs_bias - dynamic_threshold * 2) * 0.3)
                else:  # Bearish
                    sell_ratio = 0.7 + min(0.15, (abs_bias - dynamic_threshold * 2) * 0.3)
                    buy_ratio = 1.0 - sell_ratio
                    
            elif abs_bias > dynamic_threshold:  # Moderate bias
                if adjusted_directional_bias > 0:  # Bullish
                    buy_ratio = 0.55 + min(0.15, (abs_bias - dynamic_threshold) * 0.5)
                else:  # Bearish
                    sell_ratio = 0.55 + min(0.15, (abs_bias - dynamic_threshold) * 0.5)
                    buy_ratio = 1.0 - sell_ratio
                    
            else:  # Weak or neutral bias
                buy_ratio = 0.5 + adjusted_directional_bias * 0.1
            
            # Ensure ratios are within bounds
            buy_ratio = max(0.15, min(0.85, buy_ratio))
            sell_ratio = 1.0 - buy_ratio
            
            # **CRITICAL: Further adjust based on P&L and risk**
            pnl_percentage = (exposure_analysis['unrealized_pnl'] / self.user_total_investment * 100) if self.user_total_investment > 0 else 0
            
            # If we're losing money, be more conservative
            if pnl_percentage < -2:  # Losing more than 2%
                conservatism = min(0.3, abs(pnl_percentage) * 0.05)
                buy_ratio = buy_ratio * (1 - conservatism) + 0.5 * conservatism
                sell_ratio = 1.0 - buy_ratio
                self.logger.info(f"Applying loss-based conservatism: {conservatism:.2f} (PnL: {pnl_percentage:.1f}%)")
            
            # Calculate target order counts
            target_buy_orders = min(len(available_buy_levels), max(0, int(remaining_capacity * buy_ratio)))
            target_sell_orders = min(len(available_sell_levels), max(0, int(remaining_capacity * sell_ratio)))
            
            # Ensure we don't exceed remaining capacity
            total_target = target_buy_orders + target_sell_orders
            if total_target > remaining_capacity:
                reduction_factor = remaining_capacity / total_target
                target_buy_orders = int(target_buy_orders * reduction_factor)
                target_sell_orders = remaining_capacity - target_buy_orders
            
            # Ensure minimum grid functionality if we have capacity and levels
            if remaining_capacity >= 2 and total_available_levels >= 2:
                if target_buy_orders == 0 and len(available_buy_levels) > 0:
                    target_buy_orders = 1
                    target_sell_orders = min(target_sell_orders, remaining_capacity - 1)
                if target_sell_orders == 0 and len(available_sell_levels) > 0:
                    target_sell_orders = 1
                    target_buy_orders = min(target_buy_orders, remaining_capacity - 1)
            
            self.logger.info(f"Position-aware distribution:")
            self.logger.info(f"  Current exposure: {current_exposure_bias} (L:{exposure_analysis['long_positions']} S:{exposure_analysis['short_positions']} "
                           f"B:{exposure_analysis['buy_orders']} S:{exposure_analysis['sell_orders']})")
            self.logger.info(f"  Adjusted bias: {directional_bias:.3f} → {adjusted_directional_bias:.3f}")
            self.logger.info(f"  Available levels: {len(available_buy_levels)}B / {len(available_sell_levels)}S")
            self.logger.info(f"  Remaining capacity: {remaining_capacity}")
            self.logger.info(f"  Target orders: {target_buy_orders}B / {target_sell_orders}S")
            
            return target_buy_orders, target_sell_orders
            
        except Exception as e:
            self.logger.error(f"Error calculating position-aware distribution: {e}")
            # Fallback: minimal orders to avoid overexposure
            remaining_capacity = max(0, self.max_orders_per_zone - exposure_analysis.get('total_commitment', 0))
            return min(1, remaining_capacity // 2), max(0, remaining_capacity - min(1, remaining_capacity // 2))
    
    def _calculate_dynamic_grid_range(self, current_price: float) -> Tuple[float, float]:
        """Calculate dynamic grid range centered around current price"""
        try:
            # Use original grid width but center around current price
            original_width = self.user_price_upper - self.user_price_lower
            
            # Get market intelligence for dynamic range adjustment
            volatility_regime = 1.0
            trend_strength = 0.0
            
            if self.enable_samig and hasattr(self, 'market_intel'):
                try:
                    volatility_regime = self.market_intel.current_volatility_regime
                    trend_strength = self.market_intel.current_trend_strength
                except Exception:
                    pass
            
            # Adjust grid width based on market conditions
            width_multiplier = 1.0
            
            # In high volatility, use wider grid to capture more moves
            if volatility_regime > 1.5:
                width_multiplier = 1.2 + min(0.3, (volatility_regime - 1.5) * 0.2)
            # In low volatility, use narrower grid for tighter captures
            elif volatility_regime < 0.7:
                width_multiplier = 0.8 + volatility_regime * 0.3
            
            # In strong trends, bias the grid in trend direction
            dynamic_width = original_width * width_multiplier
            
            if trend_strength > 0.6:
                # Get trend direction from market intelligence
                market_snapshot = self.market_intel.analyze_market(self.exchange) if hasattr(self, 'market_intel') else None
                directional_bias = market_snapshot.directional_bias if market_snapshot else 0.0
                
                # Bias the grid in the trend direction
                if directional_bias > 0.3:  # Strong uptrend
                    # Shift grid higher (more room above current price)
                    lower_pct = 0.3  # 30% below current price
                    upper_pct = 0.7  # 70% above current price
                elif directional_bias < -0.3:  # Strong downtrend
                    # Shift grid lower (more room below current price)
                    lower_pct = 0.7  # 70% below current price
                    upper_pct = 0.3  # 30% above current price
                else:
                    # Balanced grid
                    lower_pct = 0.5
                    upper_pct = 0.5
            else:
                # Weak trend - keep balanced
                lower_pct = 0.5
                upper_pct = 0.5
            
            # Calculate new range
            dynamic_lower = current_price - (dynamic_width * lower_pct)
            dynamic_upper = current_price + (dynamic_width * upper_pct)
            
            # Ensure minimum distance from current price (avoid too tight ranges)
            min_distance = current_price * 0.01  # 1% minimum on each side
            if dynamic_upper - current_price < min_distance:
                dynamic_upper = current_price + min_distance
            if current_price - dynamic_lower < min_distance:
                dynamic_lower = current_price - min_distance
            
            # Round prices
            dynamic_lower = self._round_price(dynamic_lower)
            dynamic_upper = self._round_price(dynamic_upper)
            
            return dynamic_lower, dynamic_upper
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic grid range: {e}")
            # Fallback: use original range shifted to current price
            width = self.user_price_upper - self.user_price_lower
            return (
                self._round_price(current_price - width * 0.5),
                self._round_price(current_price + width * 0.5)
            )
    
    def _analyze_order_relevance(self, current_price: float, dynamic_lower: float, dynamic_upper: float) -> Dict[str, Any]:
        """Analyze which orders are still relevant to current price action"""
        try:
            relevant_orders = []
            irrelevant_orders = []
            
            # Define relevance threshold - orders too far from current price are irrelevant
            max_distance_pct = 0.05  # 5% max distance from current price
            max_distance = current_price * max_distance_pct
            
            # Also check if orders are within dynamic range
            for order_id, order_info in self.pending_orders.items():
                if order_info['status'] != 'open':
                    continue
                
                order_price = order_info['price']
                distance_from_current = abs(order_price - current_price)
                distance_pct = distance_from_current / current_price
                
                # Check multiple relevance criteria
                is_relevant = True
                relevance_reasons = []
                
                # 1. Distance from current price
                if distance_pct > max_distance_pct:
                    is_relevant = False
                    relevance_reasons.append(f"too far from current ({distance_pct:.1%} > {max_distance_pct:.1%})")
                
                # 2. Within dynamic range
                if order_price < dynamic_lower or order_price > dynamic_upper:
                    is_relevant = False
                    relevance_reasons.append("outside dynamic range")
                
                # 3. Logical order direction
                if order_info['type'] == 'buy' and order_price > current_price:
                    is_relevant = False
                    relevance_reasons.append("buy order above current price")
                elif order_info['type'] == 'sell' and order_price < current_price:
                    is_relevant = False
                    relevance_reasons.append("sell order below current price")
                
                order_analysis = {
                    'order_id': order_id,
                    'order_info': order_info,
                    'distance_pct': distance_pct,
                    'within_range': dynamic_lower <= order_price <= dynamic_upper,
                    'relevant': is_relevant,
                    'reasons': relevance_reasons
                }
                
                if is_relevant:
                    relevant_orders.append(order_analysis)
                else:
                    irrelevant_orders.append(order_analysis)
            
            return {
                'relevant_orders': relevant_orders,
                'irrelevant_orders': irrelevant_orders,
                'total_orders': len(self.pending_orders),
                'relevant_count': len(relevant_orders),
                'irrelevant_count': len(irrelevant_orders),
                'cleanup_needed': len(irrelevant_orders) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing order relevance: {e}")
            return {
                'relevant_orders': [], 'irrelevant_orders': [], 'total_orders': 0,
                'relevant_count': 0, 'irrelevant_count': 0, 'cleanup_needed': False
            }
    
    def _cleanup_irrelevant_orders(self, irrelevant_orders: List[Dict]) -> int:
        """Cancel orders that are no longer relevant to current price action"""
        try:
            cancelled_count = 0
            
            for order_analysis in irrelevant_orders:
                order_id = order_analysis['order_id']
                order_info = order_analysis['order_info']
                reasons = order_analysis['reasons']
                
                try:
                    # Cancel the order
                    self.exchange.cancel_order(order_id, self.symbol)
                    
                    # Remove from tracking
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    
                    # Remove from zone orders
                    zone_id = order_info.get('zone_id')
                    if zone_id and zone_id in self.active_zones:
                        zone = self.active_zones[zone_id]
                        if order_id in zone.orders:
                            del zone.orders[order_id]
                    
                    # Update investment tracking
                    margin_used = order_info.get('actual_margin_used', 0)
                    if margin_used > 0:
                        self.total_investment_used -= margin_used
                        self.total_investment_used = max(0, self.total_investment_used)
                    
                    cancelled_count += 1
                    
                    self.logger.info(f"🗑️ Cancelled irrelevant order: {order_info['type'].upper()} ${order_info['price']:.6f} "
                                   f"({', '.join(reasons)})")
                    
                except Exception as e:
                    self.logger.error(f"Error cancelling order {order_id}: {e}")
            
            if cancelled_count > 0:
                self.logger.info(f"Dynamic grid cleanup: {cancelled_count} irrelevant orders cancelled")
            
            return cancelled_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up irrelevant orders: {e}")
            return 0
    
    def _should_rebalance_grid(self, current_price: float) -> bool:
        """Determine if grid needs rebalancing based on price movement"""
        try:
            # Check if current price has moved significantly from original range
            original_center = (self.user_price_lower + self.user_price_upper) / 2
            original_width = self.user_price_upper - self.user_price_lower
            
            # Calculate how far price has moved from original center
            distance_from_center = abs(current_price - original_center)
            distance_as_pct_of_width = distance_from_center / original_width
            
            # Check if price is outside original range
            outside_original_range = current_price < self.user_price_lower or current_price > self.user_price_upper
            
            # Rebalance thresholds based on market conditions
            base_threshold = 0.3  # 30% of grid width
            
            # Get market conditions for dynamic threshold
            if self.enable_samig and hasattr(self, 'market_intel'):
                try:
                    volatility_regime = self.market_intel.current_volatility_regime
                    trend_strength = self.market_intel.current_trend_strength
                    
                    # In high volatility, rebalance more frequently
                    if volatility_regime > 1.5:
                        base_threshold = 0.2  # 20% threshold
                    # In low volatility, rebalance less frequently  
                    elif volatility_regime < 0.7:
                        base_threshold = 0.4  # 40% threshold
                    
                    # In strong trends, rebalance more frequently to follow trend
                    if trend_strength > 0.7:
                        base_threshold *= 0.8  # 20% reduction in threshold
                        
                except Exception:
                    pass
            
            # Decision criteria
            needs_rebalance = (
                outside_original_range or  # Price moved outside original range
                distance_as_pct_of_width > base_threshold  # Price moved significantly from center
            )
            
            if needs_rebalance:
                self.logger.info(f"🔄 Grid rebalance needed:")
                self.logger.info(f"  Current price: ${current_price:.6f}")
                self.logger.info(f"  Original range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
                self.logger.info(f"  Distance from center: {distance_as_pct_of_width:.1%} (threshold: {base_threshold:.1%})")
                self.logger.info(f"  Outside original range: {outside_original_range}")
            
            return needs_rebalance
            
        except Exception as e:
            self.logger.error(f"Error checking grid rebalance need: {e}")
            return False
    
    def _perform_dynamic_grid_rebalance(self, current_price: float) -> bool:
        """Perform dynamic grid rebalancing to keep orders relevant"""
        try:
            self.logger.info(f"🔄 Performing dynamic grid rebalance at ${current_price:.6f}")
            
            # 1. Calculate new dynamic range
            dynamic_lower, dynamic_upper = self._calculate_dynamic_grid_range(current_price)
            
            self.logger.info(f"  New dynamic range: ${dynamic_lower:.6f} - ${dynamic_upper:.6f}")
            
            # 2. Analyze current order relevance
            relevance_analysis = self._analyze_order_relevance(current_price, dynamic_lower, dynamic_upper)
            
            self.logger.info(f"  Order analysis: {relevance_analysis['relevant_count']} relevant, "
                           f"{relevance_analysis['irrelevant_count']} irrelevant")
            
            # 3. Cancel irrelevant orders
            if relevance_analysis['cleanup_needed']:
                cancelled_count = self._cleanup_irrelevant_orders(relevance_analysis['irrelevant_orders'])
                
                # Wait a moment for cancellations to process
                if cancelled_count > 0:
                    time.sleep(1)
            
            # 4. Create temporary zone for new range
            temp_zone_id = f"dynamic_{int(time.time())}"
            dynamic_zone = GridZone(
                zone_id=temp_zone_id,
                price_lower=dynamic_lower,
                price_upper=dynamic_upper,
                grid_count=self.user_grid_number,
                investment_per_grid=self.user_investment_per_grid
            )
            
            # 5. Set up new orders in dynamic range (with position awareness)
            orders_placed = self._setup_zone_orders_strict(dynamic_zone, current_price)
            
            if orders_placed > 0:
                # Add dynamic zone to active zones
                self.active_zones[temp_zone_id] = dynamic_zone
                
                # Deactivate old zones that are no longer relevant
                self._deactivate_old_zones(dynamic_lower, dynamic_upper)
                
                self.logger.info(f"✅ Dynamic grid rebalance complete: {orders_placed} new orders placed")
                return True
            else:
                self.logger.warning("⚠️ Dynamic grid rebalance failed: no orders placed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error performing dynamic grid rebalance: {e}")
            return False
    
    def _deactivate_old_zones(self, new_lower: float, new_upper: float):
        """Deactivate zones that don't overlap with new dynamic range"""
        try:
            zones_to_deactivate = []
            
            for zone_id, zone in self.active_zones.items():
                if not zone.active:
                    continue
                
                # Check if zone overlaps with new range
                zone_overlaps = not (zone.price_upper < new_lower or zone.price_lower > new_upper)
                
                # Deactivate zones that don't overlap and have no active positions
                if not zone_overlaps:
                    active_positions = sum(1 for pos in zone.positions.values() if pos.exit_time is None)
                    if active_positions == 0:
                        zones_to_deactivate.append(zone_id)
            
            for zone_id in zones_to_deactivate:
                self._deactivate_zone(zone_id)
                
        except Exception as e:
            self.logger.error(f"Error deactivating old zones: {e}")
    
    def update_grid(self):
        """Main grid update logic with dynamic grid management"""
        try:
            if not self.running:
                return
            
            # Get current market state
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # Update orders and positions
            self._update_orders_and_positions()
            
            # **NEW: Dynamic Grid Management**
            # Check if grid needs rebalancing based on price movement
            if self.enable_grid_adaptation and self._should_rebalance_grid(current_price):
                # Perform dynamic grid rebalancing
                rebalance_success = self._perform_dynamic_grid_rebalance(current_price)
                if rebalance_success:
                    self.logger.info("Dynamic grid rebalancing completed successfully")
                else:
                    self.logger.warning("Dynamic grid rebalancing failed")
            
            # Maintain counter orders for all positions (with position awareness)
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
            open_orders = self.exchange.get_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            filled_orders = []
            
            for order_id in list(self.pending_orders.keys()):
                if order_id not in open_order_ids:
                    try:
                        order_status = self.exchange.get_order_status(order_id, self.symbol)
                        
                        if order_status['status'] in ['filled', 'closed']:
                            filled_orders.append(order_id)
                            self._process_filled_order(order_id, order_status)
                        else:
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
            
            if 'position_id' in order_info:
                self._close_position_from_counter_order(order_id, order_status)
                return
            
            position_id = str(uuid.uuid4())
            fill_price = float(order_status.get('average', order_info['price']))
            fill_amount = float(order_status.get('filled', order_info['amount']))
            
            position = GridPosition(
                position_id=position_id,
                grid_level=order_info.get('grid_level', 0),
                entry_price=fill_price,
                quantity=fill_amount,
                side='long' if order_info['type'] == 'buy' else 'short',
                entry_time=time.time()
            )
            
            self.all_positions[position_id] = position
            
            zone_id = order_info['zone_id']
            if zone_id in self.active_zones:
                self.active_zones[zone_id].positions[position_id] = position
            
            del self.pending_orders[order_id]
            self.total_trades += 1
            
            # Calculate actual margin used for this position
            actual_margin_used = order_info.get('actual_margin_used', fill_price * fill_amount / self.user_leverage)
            
            self.logger.info(f"New position created:")
            self.logger.info(f"  ID: {position_id[:8]}")
            self.logger.info(f"  Side: {position.side}")
            self.logger.info(f"  Price: ${fill_price:.6f}")
            self.logger.info(f"  Amount: {fill_amount:.4f}")
            self.logger.info(f"  Notional: ${fill_price * fill_amount:.2f}")
            self.logger.info(f"  Margin Used: ${actual_margin_used:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error processing filled order {order_id}: {e}")
    
    def _close_position_from_counter_order(self, order_id: str, order_status: Dict):
        """Close position from counter order fill"""
        try:
            order_info = self.pending_orders[order_id]
            position_id = order_info['position_id']
            
            if position_id in self.all_positions:
                position = self.all_positions[position_id]
                position.exit_time = time.time()
                position.exit_price = float(order_status.get('average', order_info['price']))
                
                if position.side == 'long':
                    position.realized_pnl = (position.exit_price - position.entry_price) * position.quantity
                else:
                    position.realized_pnl = (position.entry_price - position.exit_price) * position.quantity
                
                position.has_counter_order = False
                position.counter_order_id = None
                
                self.logger.info(f"Closed position {position_id[:8]} via counter order: PnL = ${position.realized_pnl:.2f}")
            
            del self.pending_orders[order_id]
            
        except Exception as e:
            self.logger.error(f"Error closing position from counter order {order_id}: {e}")
    
    def _remove_cancelled_order(self, order_id: str):
        """Remove cancelled order from tracking"""
        try:
            order_info = self.pending_orders.get(order_id)
            if order_info:
                zone_id = order_info['zone_id']
                if zone_id in self.active_zones:
                    zone = self.active_zones[zone_id]
                    if order_id in zone.orders:
                        del zone.orders[order_id]
                
                del self.pending_orders[order_id]
                
        except Exception as e:
            self.logger.error(f"Error removing cancelled order {order_id}: {e}")
    
    def _maintain_counter_orders(self):
        """Ensure all positions have corresponding exit orders with position awareness"""
        try:
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            # Get current exposure to avoid over-ordering
            exposure_analysis = self._analyze_current_exposure()
            
            for position_id, position in self.all_positions.items():
                if position.exit_time is not None:
                    continue  # Position already closed
                
                if not position.has_counter_order:
                    # Check if we have capacity for more orders
                    if exposure_analysis['total_commitment'] >= self.max_orders_per_zone:
                        self.logger.warning(f"Cannot create counter order for position {position_id[:8]}: max orders reached")
                        continue
                    
                    self._create_position_aware_counter_order(position, current_price, exposure_analysis)
                else:
                    # Check if counter order still exists
                    if position.counter_order_id:
                        if position.counter_order_id not in self.pending_orders:
                            # Counter order was filled or cancelled, create new one
                            position.has_counter_order = False
                            position.counter_order_id = None
                            
                            # Check capacity before creating new counter order
                            if exposure_analysis['total_commitment'] < self.max_orders_per_zone:
                                self._create_position_aware_counter_order(position, current_price, exposure_analysis)
                            
        except Exception as e:
            self.logger.error(f"Error maintaining position-aware counter orders: {e}")
    
    def _create_position_aware_counter_order(self, position: GridPosition, current_price: float, exposure_analysis: Dict):
        """Create counter order for a position with awareness of existing orders"""
        try:
            # Determine counter order side and target price
            if position.side == 'long':
                counter_side = 'sell'
                # Try to sell at a profitable level
                base_target = position.entry_price * 1.008  # 0.8% minimum profit
                market_target = current_price * 1.004      # Or 0.4% above current
                counter_price = max(base_target, market_target)
            else:
                counter_side = 'buy'
                base_target = position.entry_price * 0.992  # 0.8% minimum profit
                market_target = current_price * 0.996       # Or 0.4% below current
                counter_price = min(base_target, market_target)
            
            counter_price = self._round_price(counter_price)
            
            # **CRITICAL: Check if counter price is too close to existing orders**
            min_gap_from_existing = current_price * 0.004  # 0.4% minimum gap
            
            for existing_level in exposure_analysis['covered_levels']:
                if abs(counter_price - existing_level) < min_gap_from_existing:
                    # Adjust counter price to avoid clustering
                    if counter_side == 'sell':
                        counter_price = existing_level + min_gap_from_existing
                    else:
                        counter_price = existing_level - min_gap_from_existing
                    
                    counter_price = self._round_price(counter_price)
                    break
            
            # Validate the adjusted price still makes sense
            if counter_side == 'sell' and counter_price <= position.entry_price * 1.002:
                # Counter price too low, skip this counter order
                self.logger.warning(f"Cannot create profitable counter order for long position {position.position_id[:8]}: "
                                  f"adjusted price ${counter_price:.6f} too close to entry ${position.entry_price:.6f}")
                return
                
            elif counter_side == 'buy' and counter_price >= position.entry_price * 0.998:
                # Counter price too high, skip this counter order
                self.logger.warning(f"Cannot create profitable counter order for short position {position.position_id[:8]}: "
                                  f"adjusted price ${counter_price:.6f} too close to entry ${position.entry_price:.6f}")
                return
            
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
                'target_profit_pct': abs(counter_price - position.entry_price) / position.entry_price * 100,
                'position_aware': True,
                'status': 'open'
            }
            
            self.pending_orders[order['id']] = order_info
            
            # Update position
            position.has_counter_order = True
            position.counter_order_id = order['id']
            
            self.logger.info(f"Position-aware counter order created for {position.position_id[:8]}: "
                           f"{counter_side} {position.quantity:.4f} @ ${counter_price:.6f} "
                           f"(target profit: {order_info['target_profit_pct']:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error creating position-aware counter order for position {position.position_id}: {e}")
    
    def _update_pnl(self):
        """Update total PnL from all positions"""
        try:
            total_unrealized = 0.0
            total_realized = 0.0
            
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            for position in self.all_positions.values():
                if position.exit_time is None:
                    if position.side == 'long':
                        unrealized = (current_price - position.entry_price) * position.quantity
                    else:
                        unrealized = (position.entry_price - current_price) * position.quantity
                    
                    position.unrealized_pnl = unrealized
                    total_unrealized += unrealized
                else:
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
            
            try:
                self.exchange.cancel_all_orders(self.symbol)
                self.pending_orders.clear()
                
                for zone in self.active_zones.values():
                    zone.orders.clear()
                    zone.active = False
                    
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
            
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
            
            position.exit_time = time.time()
            position.exit_price = float(order.get('average', 0))
            
            if position.side == 'long':
                position.realized_pnl = (position.exit_price - position.entry_price) * position.quantity
            else:
                position.realized_pnl = (position.entry_price - position.exit_price) * position.quantity
            
            self.logger.info(f"Closed position {position.position_id[:8]} at market: "
                           f"PnL = ${position.realized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position {position.position_id}: {e}")
    
    def get_status(self) -> Dict:
        """Get comprehensive grid status with investment tracking"""
        try:
            active_positions = sum(1 for pos in self.all_positions.values() if pos.exit_time is None)
            active_orders = len(self.pending_orders)
            active_zones = sum(1 for zone in self.active_zones.values() if zone.active)
            
            long_positions = sum(1 for pos in self.all_positions.values() 
                               if pos.exit_time is None and pos.side == 'long')
            short_positions = active_positions - long_positions
            
            total_position_value = 0.0
            for pos in self.all_positions.values():
                if pos.exit_time is None:
                    total_position_value += pos.quantity * pos.entry_price
            
            status = {
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'display_symbol': self.original_symbol,
                
                # User parameters (immutable)
                'user_price_lower': self.user_price_lower,
                'user_price_upper': self.user_price_upper,
                'user_grid_number': self.user_grid_number,
                'user_total_investment': self.user_total_investment,
                'user_investment_per_grid': self.user_investment_per_grid,
                
                # For display compatibility
                'price_lower': self.user_price_lower,
                'price_upper': self.user_price_upper,
                'grid_number': self.user_grid_number,
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
                
                # FIXED: Investment tracking
                'total_investment_used': self.total_investment_used,
                'investment_limit_status': f"${self.total_investment_used:.2f} / ${self.user_total_investment:.2f}",
                
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
                        'orders': len(zone.orders),
                        'grid_count': zone.grid_count
                    }
                    for zone in self.active_zones.values()
                ]
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting grid status: {e}")
            return {}