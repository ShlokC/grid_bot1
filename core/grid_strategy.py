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
    kama_direction: str = 'neutral'
    kama_strength: float = 0.0
    directional_bias: float = 0.0  # FIX: Add this missing attribute

class MarketIntelligence:
    """Simplified market intelligence for grid strategy"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_history = deque(maxlen=100)
        self.last_analysis_time = 0
        
        # FIX: Initialize missing attributes
        self.current_volatility_regime = 1.0
        self.current_trend_strength = 0.0
        self.last_volatility_update = 0
        self.last_trend_update = 0
    
    def analyze_market(self, exchange: Exchange) -> 'MarketSnapshot':
        """FIXED: Set KAMA fields using real KAMA calculation"""
        try:
            ticker = exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            volume = float(ticker.get('quoteVolume', 0))
            
            self.price_history.append(current_price)
            
            # Update regime attributes
            self.current_volatility_regime = self._calculate_volatility()
            self.current_trend_strength = self._calculate_trend_strength()
            momentum = self._calculate_momentum()  # Now uses real KAMA
            
            # FIXED: Calculate KAMA values for MarketSnapshot
            kama_value = 0.0
            kama_direction = 'neutral'
            kama_strength = 0.0
            
            if len(self.price_history) >= 20:
                import pandas as pd
                import pandas_ta as ta
                
                prices = list(self.price_history)
                kama = ta.kama(pd.Series(prices))
                
                if kama is not None and not kama.isna().all():
                    kama_value = float(kama.iloc[-1])
                    
                    # KAMA direction from recent slope
                    if len(kama) >= 3:
                        kama_slope = kama.iloc[-1] - kama.iloc[-3]
                        if kama_slope > kama_value * 0.002:
                            kama_direction = 'bullish'
                            kama_strength = min(1.0, abs(kama_slope / kama_value) * 50)
                        elif kama_slope < -kama_value * 0.002:
                            kama_direction = 'bearish'  
                            kama_strength = min(1.0, abs(kama_slope / kama_value) * 50)
            
            self.last_volatility_update = time.time()
            self.last_trend_update = time.time()
            
            return MarketSnapshot(
                timestamp=time.time(),
                price=current_price,
                volume=volume,
                volatility=self.current_volatility_regime,
                momentum=momentum,
                trend_strength=self.current_trend_strength,
                kama_value=kama_value,
                kama_direction=kama_direction,
                kama_strength=kama_strength,
                directional_bias=momentum  # FIXED: Now uses KAMA-based momentum
            )
        except Exception as e:
            logging.error(f"Error in market analysis: {e}")
            return MarketSnapshot(time.time(), 0, 0, 1.0, 0, 0, directional_bias=0.0)
    
    def _calculate_volatility(self) -> float:
        """Calculate price volatility and update regime"""
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
        """FIXED: Use real KAMA instead of fake momentum"""
        if len(self.price_history) < 20:
            return 0.0
        
        # FIXED: Single line KAMA calculation using pandas-ta
        import pandas as pd
        import pandas_ta as ta
        
        prices = list(self.price_history)
        kama = ta.kama(pd.Series(prices))
        
        if kama is None or kama.isna().all():
            return 0.0
        
        # Use KAMA trend direction as momentum
        current_price = prices[-1]
        current_kama = float(kama.iloc[-1])
        
        if current_kama > 0:
            momentum = (current_price - current_kama) / current_kama
            return max(-1.0, min(1.0, momentum * 2))
        
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
    
    def _deactivate_zone(self, zone_id: str) -> bool:
        """Deactivate a zone by cancelling its orders and marking it inactive."""
        try:
            if zone_id not in self.active_zones:
                self.logger.warning(f"Zone {zone_id} not found for deactivation")
                return False
            
            zone = self.active_zones[zone_id]
            
            # Cancel all orders in this zone and free investment
            orders_cancelled = 0
            investment_freed = 0.0
            
            for order_id in list(zone.orders.keys()):
                try:
                    self.exchange.cancel_order(order_id, self.symbol)
                    
                    # Free the investment
                    order_info = zone.orders.get(order_id, {})
                    margin_used = order_info.get('actual_margin_used', 0)
                    
                    if margin_used > 0:
                        investment_freed += margin_used
                        self.total_investment_used -= margin_used
                        self.total_investment_used = max(0, self.total_investment_used)
                    
                    # Remove from tracking
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    del zone.orders[order_id]
                    orders_cancelled += 1
                    
                except Exception as e:
                    self.logger.error(f"Error cancelling order {order_id}: {e}")
            
            # Mark zone as inactive
            zone.active = False
            
            self.logger.info(f"Deactivated zone {zone_id}: "
                        f"{orders_cancelled} orders cancelled, "
                        f"${investment_freed:.2f} freed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deactivating zone {zone_id}: {e}")
            return False
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
        """FIXED: Setup orders using live data for limits, internal tracking for metadata"""
        try:
            grid_levels = self._get_grid_levels(zone)
            orders_placed = 0
            
            # **FIXED: Use live data reconciliation for accurate limits**
            live_data = self._reconcile_internal_with_live_data()
            
            self.logger.info(f"📊 FIXED Live Data Analysis:")
            self.logger.info(f"  Live Orders: {live_data['live_order_count']}")
            self.logger.info(f"  Live Positions: {live_data['live_position_count']}")
            self.logger.info(f"  Total Live Commitment: {live_data['total_live_commitment']}/{self.max_orders_per_zone}")
            self.logger.info(f"  Remaining Capacity: {live_data['remaining_capacity']}")
            self.logger.info(f"  Live Investment: ${live_data['live_investment_used']:.2f}/${self.user_total_investment:.2f}")
            
            # Check remaining capacity based on LIVE data
            if live_data['remaining_capacity'] <= 0:
                self.logger.warning(f"No remaining capacity based on live data: {live_data['total_live_commitment']}/{self.max_orders_per_zone}")
                return 0
            
            # Calculate adaptive gap
            natural_grid_spacing = (zone.price_upper - zone.price_lower) / zone.grid_count
            natural_gap_pct = natural_grid_spacing / current_price
            adaptive_gap_pct = natural_gap_pct * 0.25
            adaptive_gap_pct = max(0.001, min(0.005, adaptive_gap_pct))
            
            # Get market intelligence
            directional_bias = 0.0
            if self.enable_samig and hasattr(self, 'market_intel'):
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    directional_bias = market_snapshot.directional_bias
                except Exception as e:
                    self.logger.warning(f"SAMIG analysis failed: {e}")
            
            # **FIXED: Calculate targets based on LIVE remaining capacity**
            max_orders_to_place = min(live_data['remaining_capacity'], len(grid_levels), zone.grid_count)
            
            if abs(directional_bias) > 0.3:
                if directional_bias > 0:  # Bullish
                    buy_target = int(max_orders_to_place * 0.6)
                else:  # Bearish
                    buy_target = int(max_orders_to_place * 0.4)
            else:  # Balanced
                buy_target = max_orders_to_place // 2
            
            sell_target = max_orders_to_place - buy_target
            
            self.logger.info(f"📋 Targets based on live data: {buy_target}B / {sell_target}S (max: {max_orders_to_place})")
            
            buy_orders_placed = 0
            sell_orders_placed = 0
            
            # Process grid levels
            for level_price in sorted(grid_levels, key=lambda x: abs(x - current_price)):
                if orders_placed >= max_orders_to_place:
                    break
                
                # **FIXED: Check investment limit based on LIVE data**
                if live_data['remaining_investment'] <= self.user_investment_per_grid:
                    self.logger.warning(f"Insufficient remaining investment: ${live_data['remaining_investment']:.2f}")
                    break
                
                # Adaptive gap check
                price_diff_pct = abs(level_price - current_price) / current_price
                if price_diff_pct < adaptive_gap_pct:
                    continue
                
                # **FIXED: Check against LIVE covered levels**
                adaptive_gap_absolute = current_price * adaptive_gap_pct
                too_close_to_existing = any(abs(level_price - existing) < adaptive_gap_absolute 
                                        for existing in live_data['covered_levels'])
                
                if too_close_to_existing:
                    continue
                
                # Determine order type
                order_type = None
                if level_price < current_price and buy_orders_placed < buy_target:
                    order_type = 'buy'
                elif level_price > current_price and sell_orders_placed < sell_target:
                    order_type = 'sell'
                
                if not order_type:
                    continue
                
                # Calculate order
                amount = self._calculate_order_amount_fixed(level_price, zone.investment_per_grid)
                order_notional = level_price * amount
                order_margin = order_notional / self.user_leverage
                
                # Place order
                try:
                    order = self.exchange.create_limit_order(self.symbol, order_type, amount, level_price)
                    
                    # **KEEP: Store metadata in internal tracking**
                    order_info = {
                        'zone_id': zone.zone_id,
                        'type': order_type,
                        'price': level_price,
                        'amount': amount,
                        'grid_level': 0,
                        'target_investment': zone.investment_per_grid,
                        'actual_margin_used': order_margin,
                        'notional_value': order_notional,
                        'live_data_based': True,
                        'status': 'open'
                    }
                    
                    # Store in internal tracking for metadata
                    self.pending_orders[order['id']] = order_info
                    zone.orders[order['id']] = order_info
                    
                    orders_placed += 1
                    if order_type == 'buy':
                        buy_orders_placed += 1
                    else:
                        sell_orders_placed += 1
                    
                    self.logger.info(f"✅ {order_type.upper()}: ${level_price:.6f} Margin: ${order_margin:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed {order_type} at ${level_price:.6f}: {e}")
            
            self.logger.info(f"🎯 FIXED Grid Setup (Live Data Based):")
            self.logger.info(f"  Orders placed: {orders_placed}/{max_orders_to_place} ({buy_orders_placed}B / {sell_orders_placed}S)")
            self.logger.info(f"  Used live data for limits, internal tracking for metadata")
            
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error in FIXED setup with live data: {e}")
            return 0
    
    def _analyze_current_exposure(self) -> Dict[str, Any]:
        """FIXED: Analyze current positions and orders using LIVE exchange data"""
        try:
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            # FIXED: Get LIVE positions from exchange instead of internal tracking
            live_positions = self.exchange.get_positions(self.symbol)
            
            long_positions = []
            short_positions = []
            total_position_value = 0.0
            unrealized_pnl = 0.0
            
            for position in live_positions:
                position_size = float(position.get('contracts', 0))
                if position_size == 0:
                    continue
                    
                entry_price = float(position.get('entryPrice', 0))
                if entry_price <= 0:
                    continue
                    
                position_value = abs(position_size) * entry_price
                total_position_value += position_value
                
                # Calculate unrealized PnL
                if position_size > 0:  # Long position
                    unrealized = (current_price - entry_price) * position_size
                    long_positions.append({
                        'size': position_size,
                        'entry_price': entry_price,
                        'unrealized_pnl': unrealized
                    })
                else:  # Short position
                    unrealized = (entry_price - current_price) * abs(position_size)
                    short_positions.append({
                        'size': abs(position_size),
                        'entry_price': entry_price,
                        'unrealized_pnl': unrealized
                    })
                
                unrealized_pnl += unrealized
            
            # FIXED: Get LIVE orders from exchange instead of internal tracking
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            buy_orders = []
            sell_orders = []
            pending_buy_value = 0.0
            pending_sell_value = 0.0
            covered_levels = set()
            
            for order in live_orders:
                order_side = order.get('side', '').lower()
                order_price = float(order.get('price', 0))
                order_amount = float(order.get('amount', 0))
                
                if order_price <= 0 or order_amount <= 0:
                    continue
                    
                order_value = order_price * order_amount
                covered_levels.add(round(order_price, 6))
                
                if order_side == 'buy':
                    buy_orders.append({
                        'price': order_price,
                        'amount': order_amount,
                        'value': order_value
                    })
                    pending_buy_value += order_value
                elif order_side == 'sell':
                    sell_orders.append({
                        'price': order_price,
                        'amount': order_amount,
                        'value': order_value
                    })
                    pending_sell_value += order_value
            
            # Add position entry prices to covered levels
            for pos in long_positions + short_positions:
                covered_levels.add(round(pos['entry_price'], 6))
            
            # Calculate exposure metrics
            net_position_bias = len(long_positions) - len(short_positions)
            net_order_bias = len(buy_orders) - len(sell_orders)
            total_exposure_bias = net_position_bias + net_order_bias
            
            # Calculate risk metrics
            position_count = len(long_positions) + len(short_positions)
            order_count = len(buy_orders) + len(sell_orders)
            total_commitment = position_count + order_count
            
            exposure_analysis = {
                'current_price': current_price,
                'long_positions': len(long_positions),
                'short_positions': len(short_positions),
                'buy_orders': len(buy_orders),
                'sell_orders': len(sell_orders),
                'net_position_bias': net_position_bias,
                'net_order_bias': net_order_bias,
                'total_exposure_bias': total_exposure_bias,
                'position_count': position_count,
                'order_count': order_count,
                'total_commitment': total_commitment,
                'total_position_value': total_position_value,
                'unrealized_pnl': unrealized_pnl,
                'covered_levels': covered_levels,
                'pending_buy_value': pending_buy_value,
                'pending_sell_value': pending_sell_value,
                'max_positions': 8,
                'exposure_ratio': total_commitment / self.user_grid_number if self.user_grid_number > 0 else 0,
                # FIXED: Add live data indicators
                'live_data_used': True,
                'live_positions_count': len(live_positions),
                'live_orders_count': len(live_orders)
            }
            
            self.logger.info(f"LIVE Exchange Data Analysis:")
            self.logger.info(f"  Live Positions: {len(live_positions)} ({len(long_positions)}L / {len(short_positions)}S)")
            self.logger.info(f"  Live Orders: {len(live_orders)} ({len(buy_orders)}B / {len(sell_orders)}S)")
            self.logger.info(f"  Total Commitment: {total_commitment}")
            self.logger.info(f"  Unrealized PnL: ${unrealized_pnl:.2f}")
            
            return exposure_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing LIVE current exposure: {e}")
            return {
                'current_price': 0, 'long_positions': 0, 'short_positions': 0,
                'buy_orders': 0, 'sell_orders': 0, 'net_position_bias': 0,
                'net_order_bias': 0, 'total_exposure_bias': 0, 'position_count': 0,
                'order_count': 0, 'total_commitment': 0, 'covered_levels': set(),
                'exposure_ratio': 0, 'live_data_used': False
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
        """FIXED: More aggressive relevance analysis for better rebalancing"""
        try:
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            relevant_orders = []
            irrelevant_orders = []
            
            # **FIXED: More aggressive relevance criteria**
            # If price moved significantly, be more aggressive about moving orders
            price_move_pct = abs(current_price - (self.user_price_lower + self.user_price_upper) / 2) / current_price
            
            if price_move_pct > 0.05:  # 5% move - be more aggressive
                relevance_buffer = 0.02  # 2% buffer around new range
            else:
                relevance_buffer = 0.05  # 5% buffer for smaller moves
            
            # Calculate effective range with buffer
            effective_lower = dynamic_lower * (1 - relevance_buffer)
            effective_upper = dynamic_upper * (1 + relevance_buffer)
            
            self.logger.info(f"🔍 FIXED Relevance Analysis:")
            self.logger.info(f"  Price move: {price_move_pct:.1%} from original center")
            self.logger.info(f"  Dynamic range: ${dynamic_lower:.6f} - ${dynamic_upper:.6f}")
            self.logger.info(f"  Effective range (with {relevance_buffer:.1%} buffer): ${effective_lower:.6f} - ${effective_upper:.6f}")
            self.logger.info(f"  Live orders: {len(live_orders)}")
            
            for order in live_orders:
                order_id = order.get('id', '')
                order_side = order.get('side', '').lower()
                order_price = float(order.get('price', 0))
                order_amount = float(order.get('amount', 0))
                
                if order_price <= 0 or order_amount <= 0:
                    continue
                
                # **FIXED: More aggressive irrelevance detection**
                is_relevant = True
                relevance_reasons = []
                
                # 1. Outside effective range = irrelevant
                if order_price < effective_lower or order_price > effective_upper:
                    is_relevant = False
                    distance_from_range = min(abs(order_price - effective_lower), abs(order_price - effective_upper))
                    distance_pct = distance_from_range / current_price
                    relevance_reasons.append(f"outside effective range by {distance_pct:.1%}")
                
                # 2. Far from current price (even if in range) = less relevant for large moves
                elif price_move_pct > 0.1:  # 10% price move
                    distance_from_current = abs(order_price - current_price) / current_price
                    if distance_from_current > 0.08:  # 8% from current price
                        is_relevant = False
                        relevance_reasons.append(f"too far from current price ({distance_from_current:.1%})")
                
                # 3. Wrong side of current price
                elif order_side == 'buy' and order_price > current_price * 1.01:
                    is_relevant = False
                    relevance_reasons.append("buy order too far above current price")
                elif order_side == 'sell' and order_price < current_price * 0.99:
                    is_relevant = False
                    relevance_reasons.append("sell order too far below current price")
                
                order_info = {
                    'order_id': order_id,
                    'type': order_side,
                    'price': order_price,
                    'amount': order_amount,
                    'notional_value': order_price * order_amount,
                    'margin_used': (order_price * order_amount) / self.user_leverage,
                    'live_order': True
                }
                
                order_analysis = {
                    'order_id': order_id,
                    'order_info': order_info,
                    'distance_from_current_pct': abs(order_price - current_price) / current_price,
                    'within_effective_range': effective_lower <= order_price <= effective_upper,
                    'relevant': is_relevant,
                    'reasons': relevance_reasons
                }
                
                if is_relevant:
                    relevant_orders.append(order_analysis)
                    self.logger.info(f"✅ Relevant: {order_side.upper()} ${order_price:.6f}")
                else:
                    irrelevant_orders.append(order_analysis)
                    self.logger.info(f"❌ Irrelevant: {order_side.upper()} ${order_price:.6f} - {', '.join(relevance_reasons)}")
            
            result = {
                'relevant_orders': relevant_orders,
                'irrelevant_orders': irrelevant_orders,
                'total_orders': len(live_orders),
                'relevant_count': len(relevant_orders),
                'irrelevant_count': len(irrelevant_orders),
                'cleanup_needed': len(irrelevant_orders) > 0,
                'price_move_pct': price_move_pct,
                'aggressive_mode': price_move_pct > 0.05,
                'live_data_used': True
            }
            
            self.logger.info(f"📊 FIXED Relevance Summary:")
            self.logger.info(f"  Relevant: {result['relevant_count']}, Irrelevant: {result['irrelevant_count']}")
            self.logger.info(f"  Aggressive mode: {result['aggressive_mode']} (price moved {price_move_pct:.1%})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in FIXED relevance analysis: {e}")
            return {
                'relevant_orders': [], 'irrelevant_orders': [], 'total_orders': 0,
                'relevant_count': 0, 'irrelevant_count': 0, 'cleanup_needed': False,
                'live_data_used': False
            }
    
    def _cleanup_irrelevant_orders(self, irrelevant_orders: List[Dict]) -> int:
        """FIXED: Cancel orders using LIVE order data"""
        try:
            cancelled_count = 0
            
            for order_analysis in irrelevant_orders:
                order_id = order_analysis['order_id']
                order_info = order_analysis['order_info']
                reasons = order_analysis['reasons']
                
                try:
                    # Cancel the order using live order ID
                    self.exchange.cancel_order(order_id, self.symbol)
                    
                    # FIXED: Clean up internal tracking if it exists
                    if order_id in self.pending_orders:
                        # Free investment from internal tracking
                        internal_order = self.pending_orders[order_id]
                        margin_used = internal_order.get('actual_margin_used', 0)
                        if margin_used > 0:
                            self.total_investment_used -= margin_used
                            self.total_investment_used = max(0, self.total_investment_used)
                        
                        del self.pending_orders[order_id]
                    
                    # Remove from zone orders if exists
                    for zone in self.active_zones.values():
                        if order_id in zone.orders:
                            del zone.orders[order_id]
                            break
                    
                    cancelled_count += 1
                    
                    self.logger.info(f"🗑️ Cancelled LIVE order: {order_info['type'].upper()} ${order_info['price']:.6f} "
                                f"({', '.join(reasons)})")
                    
                except Exception as e:
                    self.logger.error(f"Error cancelling LIVE order {order_id}: {e}")
            
            if cancelled_count > 0:
                self.logger.info(f"LIVE order cleanup: {cancelled_count} orders cancelled")
                # Update investment tracking after cleanup
                self.logger.info(f"Updated investment usage: ${self.total_investment_used:.2f} / ${self.user_total_investment:.2f}")
            
            return cancelled_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up LIVE irrelevant orders: {e}")
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
        """FIXED: Use corrected investment calculation in rebalancing"""
        try:
            self.logger.info(f"🔄 FIXED Dynamic Grid Rebalance at ${current_price:.6f}")
            
            # 1. Calculate new dynamic range
            dynamic_lower, dynamic_upper = self._calculate_dynamic_grid_range(current_price)
            self.logger.info(f"  New dynamic range: ${dynamic_lower:.6f} - ${dynamic_upper:.6f}")
            
            # 2. Analyze order relevance
            relevance_analysis = self._analyze_order_relevance(current_price, dynamic_lower, dynamic_upper)
            
            # 3. **FIXED: Use corrected available investment calculation**
            live_data = self._reconcile_internal_with_live_data()
            current_available = live_data['available_investment']
            
            # Calculate potential investment that will be freed
            potential_freed_investment = sum(
                order_analysis['order_info'].get('margin_used', 0) 
                for order_analysis in relevance_analysis['irrelevant_orders']
            )
            
            total_available_after_cleanup = current_available + potential_freed_investment
            
            self.logger.info(f"💰 FIXED Investment Analysis:")
            self.logger.info(f"  Current available: ${current_available:.2f}")
            self.logger.info(f"  Will be freed: ${potential_freed_investment:.2f}")
            self.logger.info(f"  Total after cleanup: ${total_available_after_cleanup:.2f}")
            # **FIXED: Lower minimum threshold for small adjustments**
            min_investment_needed = self.user_investment_per_grid * 1.1  # Was 2.0, now 1.1

            if total_available_after_cleanup < min_investment_needed:
                self.logger.warning(f"⚠️ Insufficient investment even after cleanup: ${total_available_after_cleanup:.2f} < ${min_investment_needed:.2f}")
                return False
            
            # Cancel irrelevant orders and free investment
            investment_actually_freed = 0.0
            if relevance_analysis['cleanup_needed']:
                investment_actually_freed = self._cleanup_irrelevant_orders_with_investment_tracking(
                    relevance_analysis['irrelevant_orders']
                )
                
                if investment_actually_freed > 0:
                    time.sleep(1)
            
            # **FIXED: Re-calculate available investment after cleanup**
            final_live_data = self._reconcile_internal_with_live_data()
            final_available = final_live_data['available_investment']
            
            self.logger.info(f"💎 Final investment check:")
            self.logger.info(f"  Actually freed: ${investment_actually_freed:.2f}")  
            self.logger.info(f"  Now available: ${final_available:.2f}")
            
            if final_available < self.user_investment_per_grid:
                self.logger.warning(f"⚠️ Still insufficient investment after cleanup: ${final_available:.2f}")
                return False
            
            # Create dynamic zone and place new orders
            temp_zone_id = f"dynamic_{int(time.time())}"
            dynamic_zone = GridZone(
                zone_id=temp_zone_id,
                price_lower=dynamic_lower,
                price_upper=dynamic_upper,
                grid_count=self.user_grid_number,
                investment_per_grid=self.user_investment_per_grid
            )
            
            # Setup new orders with corrected available investment
            orders_placed = self._setup_rebalance_orders_with_available_investment(dynamic_zone, current_price, final_available)
            
            if orders_placed > 0:
                self.active_zones[temp_zone_id] = dynamic_zone
                self._deactivate_old_zones(dynamic_lower, dynamic_upper)
                
                self.logger.info(f"✅ FIXED Dynamic rebalance complete: {orders_placed} new orders placed")
                return True
            else:
                self.logger.warning(f"⚠️ Dynamic rebalance failed: no orders placed with ${final_available:.2f} available")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in FIXED dynamic rebalance: {e}")
            return False
    def _setup_rebalance_orders_with_available_investment(self, zone: GridZone, current_price: float, available_investment: float) -> int:
        """FIXED: Setup rebalance orders with known available investment"""
        try:
            self.logger.info(f"🎯 FIXED Rebalance Order Setup:")
            self.logger.info(f"  Available investment: ${available_investment:.2f}")
            self.logger.info(f"  Zone range: ${zone.price_lower:.6f} - ${zone.price_upper:.6f}")
            
            if available_investment < self.user_investment_per_grid:
                self.logger.warning(f"Insufficient investment: ${available_investment:.2f} < ${self.user_investment_per_grid:.2f}")
                return 0
            
            grid_levels = self._get_grid_levels(zone)
            if not grid_levels:
                return 0
            
            # Calculate how many orders we can afford
            max_orders_by_investment = int(available_investment / self.user_investment_per_grid)
            
            # Get current exposure for capacity check
            live_orders = self.exchange.get_open_orders(self.symbol)
            live_positions = self.exchange.get_positions(self.symbol)
            current_commitment = len(live_orders) + len([p for p in live_positions if float(p.get('contracts', 0)) != 0])
            
            max_orders_by_capacity = max(0, self.max_orders_per_zone - current_commitment)
            max_orders = min(max_orders_by_investment, max_orders_by_capacity, len(grid_levels))
            
            self.logger.info(f"  Max orders: {max_orders} (investment: {max_orders_by_investment}, capacity: {max_orders_by_capacity})")
            
            if max_orders <= 0:
                return 0
            
            # Use adaptive gap
            natural_grid_spacing = (zone.price_upper - zone.price_lower) / zone.grid_count
            natural_gap_pct = natural_grid_spacing / current_price
            adaptive_gap_pct = natural_gap_pct * 0.25  # Same as main setup
            adaptive_gap_pct = max(0.001, min(0.005, adaptive_gap_pct))
            
            # Simple balanced distribution
            buy_target = max_orders // 2
            sell_target = max_orders - buy_target
            
            # Place orders
            orders_placed = 0
            buy_orders_placed = 0
            sell_orders_placed = 0
            
            # Get covered levels from current live orders
            covered_levels = set()
            for order in live_orders:
                order_price = float(order.get('price', 0))
                if order_price > 0:
                    covered_levels.add(round(order_price, 6))
            
            for level_price in sorted(grid_levels, key=lambda x: abs(x - current_price)):
                if orders_placed >= max_orders:
                    break
                
                # Gap check
                price_diff_pct = abs(level_price - current_price) / current_price
                if price_diff_pct < adaptive_gap_pct:
                    continue
                
                # Check against existing orders
                adaptive_gap_absolute = current_price * adaptive_gap_pct
                too_close = any(abs(level_price - existing) < adaptive_gap_absolute for existing in covered_levels)
                if too_close:
                    continue
                
                # Determine order type
                order_type = None
                if level_price < current_price and buy_orders_placed < buy_target:
                    order_type = 'buy'
                elif level_price > current_price and sell_orders_placed < sell_target:
                    order_type = 'sell'
                
                if not order_type:
                    continue
                
                # Calculate order
                amount = self._calculate_order_amount_fixed(level_price, zone.investment_per_grid)
                order_notional = level_price * amount
                order_margin = order_notional / self.user_leverage
                
                # Investment check
                if self.total_investment_used + order_margin > self.user_total_investment:
                    self.logger.warning(f"Investment limit hit during rebalance order placement")
                    break
                
                # Place order
                try:
                    order = self.exchange.create_limit_order(self.symbol, order_type, amount, level_price)
                    
                    order_info = {
                        'zone_id': zone.zone_id,
                        'type': order_type,
                        'price': level_price,
                        'amount': amount,
                        'grid_level': 0,
                        'target_investment': zone.investment_per_grid,
                        'actual_margin_used': order_margin,
                        'notional_value': order_notional,
                        'rebalance_order': True,
                        'status': 'open'
                    }
                    
                    self.pending_orders[order['id']] = order_info
                    zone.orders[order['id']] = order_info
                    
                    orders_placed += 1
                    self.total_investment_used += order_margin
                    covered_levels.add(round(level_price, 6))  # Update covered levels
                    
                    if order_type == 'buy':
                        buy_orders_placed += 1
                    else:
                        sell_orders_placed += 1
                    
                    self.logger.info(f"✅ Rebalance {order_type.upper()}: ${level_price:.6f} Margin: ${order_margin:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed rebalance {order_type} at ${level_price:.6f}: {e}")
            
            self.logger.info(f"🎯 FIXED Rebalance orders complete: {orders_placed}/{max_orders} ({buy_orders_placed}B / {sell_orders_placed}S)")
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error in FIXED rebalance order setup: {e}")
            return 0
    def _cleanup_irrelevant_orders_with_investment_tracking(self, irrelevant_orders: List[Dict]) -> float:
        """FIXED: Use consistent margin calculation when freeing investment"""
        try:
            cancelled_count = 0
            investment_freed = 0.0
            
            self.logger.info(f"🧹 FIXED Investment Cleanup: Cancelling {len(irrelevant_orders)} irrelevant orders")
            
            for order_analysis in irrelevant_orders:
                order_id = order_analysis['order_id']
                order_info = order_analysis['order_info']
                reasons = order_analysis['reasons']
                
                try:
                    # **FIXED: Calculate margin to be freed using consistent method**
                    order_notional = order_info['price'] * order_info['amount']
                    margin_to_free = round(order_notional / self.user_leverage, 2)
                    
                    # Cancel the order
                    self.exchange.cancel_order(order_id, self.symbol)
                    
                    # Track freed investment
                    investment_freed += margin_to_free
                    
                    # **FIXED: Don't update internal tracking here - let reconciliation handle it**
                    # This prevents double-accounting issues
                    
                    # Remove from internal tracking
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    
                    # Remove from zone orders
                    for zone in self.active_zones.values():
                        if order_id in zone.orders:
                            del zone.orders[order_id]
                            break
                    
                    cancelled_count += 1
                    
                    self.logger.info(f"🗑️ Cancelled: {order_info['type'].upper()} ${order_info['price']:.6f} "
                                f"(will free: ${margin_to_free:.2f}) - {', '.join(reasons)}")
                    
                except Exception as e:
                    self.logger.error(f"Error cancelling order {order_id}: {e}")
            
            if cancelled_count > 0:
                self.logger.info(f"💰 FIXED Investment Freed: ${investment_freed:.2f} from {cancelled_count} cancelled orders")
            
            return investment_freed
            
        except Exception as e:
            self.logger.error(f"Error in FIXED investment cleanup: {e}")
            return 0.0

    def _reconcile_internal_with_live_data(self) -> Dict[str, Any]:
        """FIXED: Simplified investment calculation without artificial capping"""
        try:
            # Get live data from exchange
            live_positions = self.exchange.get_positions(self.symbol)
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            live_order_ids = {order['id'] for order in live_orders}
            live_position_count = len([pos for pos in live_positions if float(pos.get('contracts', 0)) != 0])
            
            # Clean up stale internal tracking
            stale_orders_removed = 0
            for order_id in list(self.pending_orders.keys()):
                if order_id not in live_order_ids:
                    order_info = self.pending_orders[order_id]
                    zone_id = order_info.get('zone_id')
                    if zone_id and zone_id in self.active_zones:
                        if order_id in self.active_zones[zone_id].orders:
                            del self.active_zones[zone_id].orders[order_id]
                    del self.pending_orders[order_id]
                    stale_orders_removed += 1
            
            if stale_orders_removed > 0:
                self.logger.info(f"🧹 Cleaned up {stale_orders_removed} stale internal order records")
            
            # Calculate live-based metrics
            live_order_count = len(live_orders)
            total_live_commitment = live_position_count + live_order_count
            
            # **FIXED: Simplified investment calculation per user's formula**
            # investment = position margin + order margin + available
            
            position_margin_used = 0.0
            order_margin_used = 0.0
            
            # Calculate position margins
            for position in live_positions:
                position_size = float(position.get('contracts', 0))
                if position_size != 0:
                    entry_price = float(position.get('entryPrice', 0))
                    if entry_price > 0:
                        # FIXED: Use consistent margin calculation with proper rounding
                        position_notional = abs(position_size) * entry_price
                        position_margin = round(position_notional / self.user_leverage, 2)
                        position_margin_used += position_margin
            
            # Calculate order margins
            for order in live_orders:
                order_price = float(order.get('price', 0))
                order_amount = float(order.get('amount', 0))
                if order_price > 0 and order_amount > 0:
                    # FIXED: Use consistent margin calculation with proper rounding
                    order_notional = order_price * order_amount
                    order_margin = round(order_notional / self.user_leverage, 2)
                    order_margin_used += order_margin
            
            # **FIXED: Total used = position + order margins (no artificial capping)**
            total_margin_used = round(position_margin_used + order_margin_used, 2)
            
            # **FIXED: Available = Total - Used (simple subtraction)**
            available_investment = round(self.user_total_investment - total_margin_used, 2)
            
            # **FIXED: Update internal tracking to match live calculation exactly**
            self.total_investment_used = total_margin_used
            
            # Create covered levels set from live data
            covered_levels = set()
            for order in live_orders:
                order_price = float(order.get('price', 0))
                if order_price > 0:
                    covered_levels.add(round(order_price, 6))
            
            for position in live_positions:
                if float(position.get('contracts', 0)) != 0:
                    entry_price = float(position.get('entryPrice', 0))
                    if entry_price > 0:
                        covered_levels.add(round(entry_price, 6))
            
            reconciled_data = {
                'live_order_count': live_order_count,
                'live_position_count': live_position_count,
                'total_live_commitment': total_live_commitment,
                'position_margin_used': position_margin_used,
                'order_margin_used': order_margin_used,
                'total_margin_used': total_margin_used,
                'live_investment_used': total_margin_used,  # FIXED: Key name expected by get_status()
                'available_investment': available_investment,
                'remaining_investment': available_investment,  # FIXED: Alternative key name
                'covered_levels': covered_levels,
                'remaining_capacity': max(0, self.max_orders_per_zone - total_live_commitment),
                'stale_orders_cleaned': stale_orders_removed,
                'internal_tracking_synced': True
            }
            
            self.logger.info(f"📊 FIXED Investment Calculation:")
            self.logger.info(f"  Position margin: ${position_margin_used:.2f}")
            self.logger.info(f"  Order margin: ${order_margin_used:.2f}")
            self.logger.info(f"  Total used: ${total_margin_used:.2f}")
            self.logger.info(f"  Available: ${available_investment:.2f}")
            self.logger.info(f"  Formula: ${self.user_total_investment:.2f} - ${total_margin_used:.2f} = ${available_investment:.2f}")
            
            return reconciled_data
            
        except Exception as e:
            self.logger.error(f"Error in FIXED investment reconciliation: {e}")
            # Fallback to safe defaults
            return {
                'live_order_count': 0, 'live_position_count': 0, 'total_live_commitment': 0,
                'position_margin_used': 0, 'order_margin_used': 0, 'total_margin_used': self.user_total_investment,
                'available_investment': 0, 'covered_levels': set(), 'remaining_capacity': 0, 
                'internal_tracking_synced': False
            }
    def _setup_zone_orders_with_freed_investment(self, zone: GridZone, current_price: float) -> int:
        """FIXED: Setup orders for rebalancing with adaptive gap instead of restrictive filtering"""
        try:
            available_investment = self.user_total_investment - self.total_investment_used
            self.logger.info(f"Setting up rebalance zone:")
            self.logger.info(f"  Available investment: ${available_investment:.2f}")
            
            if available_investment <= 0:
                return 0
            
            grid_levels = self._get_grid_levels(zone)
            if not grid_levels:
                return 0
            
            # Get current exposure
            exposure_analysis = self._analyze_current_exposure()
            
            # Calculate order limits
            max_orders_by_investment = int(available_investment / self.user_investment_per_grid)
            max_orders_by_capacity = self.max_orders_per_zone - exposure_analysis['total_commitment']
            max_orders = min(max_orders_by_investment, max_orders_by_capacity, len(grid_levels))
            
            if max_orders <= 0:
                return 0
            
            # **FIXED: Use same adaptive gap logic as main grid setup**
            natural_grid_spacing = (zone.price_upper - zone.price_lower) / zone.grid_count
            natural_gap_pct = natural_grid_spacing / current_price
            adaptive_gap_pct = natural_gap_pct * 0.3  # 30% for rebalancing (slightly more flexible)
            adaptive_gap_pct = max(0.001, min(0.006, adaptive_gap_pct))
            
            self.logger.info(f"🔧 Rebalance Gap: {adaptive_gap_pct:.3%} (30% of natural {natural_gap_pct:.3%})")
            
            # Simple distribution
            buy_target = max_orders // 2
            sell_target = max_orders - buy_target
            
            # Place orders
            orders_placed = 0
            buy_orders_placed = 0
            sell_orders_placed = 0
            
            for level_price in sorted(grid_levels, key=lambda x: abs(x - current_price)):
                if orders_placed >= max_orders:
                    break
                
                # **SINGLE ADAPTIVE GAP CHECK**
                price_diff_pct = abs(level_price - current_price) / current_price
                if price_diff_pct < adaptive_gap_pct:
                    continue
                
                # Check against existing levels
                adaptive_gap_absolute = current_price * adaptive_gap_pct
                too_close = any(abs(level_price - existing) < adaptive_gap_absolute 
                            for existing in exposure_analysis['covered_levels'])
                
                if too_close:
                    continue
                
                # Determine order type
                order_type = None
                if level_price < current_price and buy_orders_placed < buy_target:
                    order_type = 'buy'
                elif level_price > current_price and sell_orders_placed < sell_target:
                    order_type = 'sell'
                
                if not order_type:
                    continue
                
                # Calculate and place order
                amount = self._calculate_order_amount_fixed(level_price, zone.investment_per_grid)
                order_notional = level_price * amount
                order_margin = order_notional / self.user_leverage
                
                if self.total_investment_used + order_margin > self.user_total_investment:
                    break
                
                try:
                    order = self.exchange.create_limit_order(self.symbol, order_type, amount, level_price)
                    
                    order_info = {
                        'zone_id': zone.zone_id,
                        'type': order_type,
                        'price': level_price,
                        'amount': amount,
                        'grid_level': 0,
                        'target_investment': zone.investment_per_grid,
                        'actual_margin_used': order_margin,
                        'notional_value': order_notional,
                        'rebalance_order': True,
                        'adaptive_gap_used': adaptive_gap_pct,
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
                    
                    self.logger.info(f"✅ Rebalance {order_type.upper()}: ${level_price:.6f} Margin: ${order_margin:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed rebalance {order_type} at ${level_price:.6f}: {e}")
            
            self.logger.info(f"🎯 FIXED Rebalance complete: {orders_placed} orders ({buy_orders_placed}B / {sell_orders_placed}S)")
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error in FIXED rebalance setup: {e}")
            return 0
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
        """FIXED: Main update with live data reconciliation"""
        try:
            if not self.running:
                return
            
            # **ADDED: Regular reconciliation with live data**
            live_data = self._reconcile_internal_with_live_data()
            
            # Get current market state
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # Update orders and positions (now uses reconciled data)
            self._update_orders_and_positions()
            
            # Dynamic grid management (if enabled)
            if self.enable_grid_adaptation and self._should_rebalance_grid(current_price):
                rebalance_success = self._perform_dynamic_grid_rebalance(current_price)
                if rebalance_success:
                    self.logger.info("Dynamic grid rebalancing completed")
            
            # Maintain counter orders (now uses live data limits)
            self._maintain_counter_orders()
            
            # Update PnL
            self._update_pnl()
            
            # Check TP/SL
            self._check_tp_sl()
            
        except Exception as e:
            self.logger.error(f"Error in FIXED grid update: {e}")
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
        """FIXED: Maintain counter orders using live data for limits"""
        try:
            # **FIXED: Get live data for accurate limit checking**
            live_data = self._reconcile_internal_with_live_data()
            
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            # Only process positions that we have internal metadata for
            for position_id, position in self.all_positions.items():
                if position.exit_time is not None:
                    continue  # Position already closed
                
                if not position.has_counter_order:
                    # **FIXED: Check capacity based on live data**
                    if live_data['remaining_capacity'] <= 0:
                        self.logger.debug(f"Cannot create counter order: no remaining capacity ({live_data['total_live_commitment']}/{self.max_orders_per_zone})")
                        continue
                    
                    self._create_counter_order_with_live_limits(position, current_price, live_data)
                    
                else:
                    # Check if counter order still exists in live data
                    if position.counter_order_id and position.counter_order_id not in self.pending_orders:
                        # Counter order was filled or cancelled
                        position.has_counter_order = False
                        position.counter_order_id = None
                        
                        # Try to create new counter order if capacity allows
                        if live_data['remaining_capacity'] > 0:
                            self._create_counter_order_with_live_limits(position, current_price, live_data)
                            
        except Exception as e:
            self.logger.error(f"Error maintaining counter orders with live data: {e}")
    def _create_counter_order_with_live_limits(self, position: GridPosition, current_price: float, live_data: Dict):
        """Create counter order respecting live data limits"""
        try:
            # Determine counter order details
            if position.side == 'long':
                counter_side = 'sell'
                base_target = position.entry_price * 1.008
                market_target = current_price * 1.004
                counter_price = max(base_target, market_target)
            else:
                counter_side = 'buy'
                base_target = position.entry_price * 0.992
                market_target = current_price * 0.996
                counter_price = min(base_target, market_target)
            
            counter_price = self._round_price(counter_price)
            
            # Use adaptive gap based on live data
            if self.active_zones:
                active_zone = next(iter(self.active_zones.values()))
                natural_spacing = (active_zone.price_upper - active_zone.price_lower) / active_zone.grid_count
                natural_gap_pct = natural_spacing / current_price
            else:
                natural_gap_pct = 0.006
            
            adaptive_gap_pct = natural_gap_pct * 0.3
            adaptive_gap_pct = max(0.002, min(0.008, adaptive_gap_pct))
            min_gap_from_existing = current_price * adaptive_gap_pct
            
            # **FIXED: Check against LIVE covered levels**
            for existing_level in live_data['covered_levels']:
                if abs(counter_price - existing_level) < min_gap_from_existing:
                    if counter_side == 'sell':
                        counter_price = existing_level + min_gap_from_existing
                    else:
                        counter_price = existing_level - min_gap_from_existing
                    counter_price = self._round_price(counter_price)
                    break
            
            # Validate profitability
            if counter_side == 'sell' and counter_price <= position.entry_price * 1.002:
                return
            elif counter_side == 'buy' and counter_price >= position.entry_price * 0.998:
                return
            
            # Create counter order
            order = self.exchange.create_limit_order(self.symbol, counter_side, position.quantity, counter_price)
            
            # Store metadata in internal tracking
            order_info = {
                'zone_id': 'counter',
                'type': counter_side,
                'price': counter_price,
                'amount': position.quantity,
                'position_id': position.position_id,
                'target_profit_pct': abs(counter_price - position.entry_price) / position.entry_price * 100,
                'live_data_based': True,
                'status': 'open'
            }
            
            self.pending_orders[order['id']] = order_info
            position.has_counter_order = True
            position.counter_order_id = order['id']
            
            self.logger.info(f"✅ Counter order (Live Data): {counter_side} @ ${counter_price:.6f} "
                        f"(profit: {order_info['target_profit_pct']:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error creating counter order with live limits: {e}")
    def _create_position_aware_counter_order(self, position: GridPosition, current_price: float, exposure_analysis: Dict):
        """FIXED: Create counter order with adaptive gap instead of restrictive 0.4% gap"""
        try:
            # Determine counter order side and target price
            if position.side == 'long':
                counter_side = 'sell'
                base_target = position.entry_price * 1.008  # 0.8% minimum profit
                market_target = current_price * 1.004      # Or 0.4% above current
                counter_price = max(base_target, market_target)
            else:
                counter_side = 'buy'
                base_target = position.entry_price * 0.992  # 0.8% minimum profit
                market_target = current_price * 0.996       # Or 0.4% below current
                counter_price = min(base_target, market_target)
            
            counter_price = self._round_price(counter_price)
            
            # **FIXED: Use adaptive gap instead of fixed 0.4%**
            # Calculate natural grid spacing for current active zone
            natural_gap_pct = 0.006  # Default 0.6%
            if self.active_zones:
                active_zone = next(iter(self.active_zones.values()))
                natural_spacing = (active_zone.price_upper - active_zone.price_lower) / active_zone.grid_count
                natural_gap_pct = natural_spacing / current_price
            
            # Use 30% of natural gap for counter orders (less restrictive than main grid)
            adaptive_gap_pct = natural_gap_pct * 0.3
            adaptive_gap_pct = max(0.002, min(0.008, adaptive_gap_pct))  # 0.2% to 0.8%
            min_gap_from_existing = current_price * adaptive_gap_pct
            
            # **SIMPLIFIED: Check against existing levels with adaptive gap**
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
                self.logger.debug(f"Counter price ${counter_price:.6f} too close to entry ${position.entry_price:.6f}")
                return
                
            elif counter_side == 'buy' and counter_price >= position.entry_price * 0.998:
                self.logger.debug(f"Counter price ${counter_price:.6f} too close to entry ${position.entry_price:.6f}")
                return
            
            # Create counter order
            order = self.exchange.create_limit_order(self.symbol, counter_side, position.quantity, counter_price)
            
            # Track counter order
            order_info = {
                'zone_id': 'counter',
                'type': counter_side,
                'price': counter_price,
                'amount': position.quantity,
                'position_id': position.position_id,
                'target_profit_pct': abs(counter_price - position.entry_price) / position.entry_price * 100,
                'adaptive_gap_used': adaptive_gap_pct,
                'status': 'open'
            }
            
            self.pending_orders[order['id']] = order_info
            
            # Update position
            position.has_counter_order = True
            position.counter_order_id = order['id']
            
            self.logger.info(f"✅ FIXED Counter order: {counter_side} {position.quantity:.4f} @ ${counter_price:.6f} "
                        f"(gap: {adaptive_gap_pct:.3%}, profit: {order_info['target_profit_pct']:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error creating FIXED counter order for position {position.position_id}: {e}")
    
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
        """FIXED: Get status with live data reconciliation"""
        try:
            # Reconcile with live data first
            live_data = self._reconcile_internal_with_live_data()
            
            # Count based on internal metadata (for grid-specific info)
            internal_active_positions = sum(1 for pos in self.all_positions.values() if pos.exit_time is None)
            internal_active_orders = len(self.pending_orders)
            
            status = {
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'display_symbol': self.original_symbol,
                
                # User parameters
                'price_lower': self.user_price_lower,
                'price_upper': self.user_price_upper,
                'grid_number': self.user_grid_number,
                'investment': self.user_total_investment,
                'leverage': self.user_leverage,
                
                # Strategy settings
                'take_profit_pnl': self.take_profit_pnl,
                'stop_loss_pnl': self.stop_loss_pnl,
                'enable_grid_adaptation': self.enable_grid_adaptation,
                'enable_samig': self.enable_samig,
                
                # **FIXED: Status based on live data**
                'running': self.running,
                'active_positions': live_data['live_position_count'],
                'orders_count': live_data['live_order_count'],
                'total_commitment': live_data['total_live_commitment'],
                'trades_count': self.total_trades,
                
                # **FIXED: Investment tracking synced with live data**
                'total_investment_used': live_data['live_investment_used'],
                'remaining_capacity': live_data['remaining_capacity'],
                'live_data_synced': live_data['internal_tracking_synced'],
                
                # PnL (calculated from internal position tracking for accuracy)
                'pnl': self.total_pnl,
                'pnl_percentage': (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0,
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting FIXED status: {e}")
            return {}