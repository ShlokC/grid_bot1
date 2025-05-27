"""
Grid Trading Strategy - Simplified Implementation Without Zones
Author: Grid Trading Bot
Date: 2025-05-26

Complete rewrite removing zone complexity and artificial restrictions.
Uses real KAMA instead of fake momentum, direct order placement within user range.
FIXED: Now properly uses OHLCV data for KAMA calculation.
"""

import logging
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import Enum

# Import exchange interface
from core.exchange import Exchange


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class GridPosition:
    """Represents a filled position from grid trading"""
    position_id: str
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
    
    def is_open(self) -> bool:
        """Check if position is still open"""
        return self.exit_time is None
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for open position"""
        if not self.is_open():
            return 0.0
        
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity


@dataclass
class MarketSnapshot:
    """Market data snapshot with technical indicators"""
    timestamp: float
    price: float
    volume: float
    volatility: float
    momentum: float
    trend_strength: float
    kama_value: float = 0.0
    kama_direction: str = 'neutral'  # 'bullish', 'bearish', 'neutral'
    kama_strength: float = 0.0
    directional_bias: float = 0.0
    
    def is_trending(self, threshold: float = 0.5) -> bool:
        """Check if market is in trending state"""
        return self.trend_strength > threshold
    
    def is_bullish(self, threshold: float = 0.1) -> bool:
        """Check if market sentiment is bullish"""
        return self.directional_bias > threshold
    
    def is_bearish(self, threshold: float = 0.1) -> bool:
        """Check if market sentiment is bearish"""
        return self.directional_bias < -threshold


class MarketIntelligence:
    """Market intelligence using real KAMA indicator"""
    
    def __init__(self, symbol: str, history_length: int = 100):
        self.symbol = symbol
        self.price_history = deque(maxlen=history_length)
        self.volume_history = deque(maxlen=history_length)
        self.last_analysis_time = 0
        
        # OHLCV data cache to avoid excessive API calls
        self.last_ohlcv_fetch = 0
        self.ohlcv_cache_duration = 300  # 5 minutes
        self.ohlcv_data_fetched = False
        
        # Market regime tracking
        self.current_volatility_regime = 1.0
        self.current_trend_strength = 0.0
        self.last_volatility_update = 0
        self.last_trend_update = 0
        
        # KAMA tracking
        self.last_kama_value = 0.0
        self.kama_history = deque(maxlen=50)
        
        self.logger = logging.getLogger(f"{__name__}.MarketIntel")
    
    def analyze_market(self, exchange: Exchange) -> MarketSnapshot:
        """Analyze market using real KAMA and other indicators"""
        try:
            # FIXED: First fetch OHLCV data to populate price_history properly
            current_time = time.time()
            
            # Fetch OHLCV data if cache expired or first fetch
            if (current_time - self.last_ohlcv_fetch > self.ohlcv_cache_duration) or not self.ohlcv_data_fetched:
                self._fetch_historical_data(exchange)
            
            # Get current price from ticker for real-time data
            ticker = exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            volume = float(ticker.get('quoteVolume', 0))
            
            # Add current price to history if we have space
            if len(self.price_history) == 0 or self.price_history[-1] != current_price:
                self.price_history.append(current_price)
                self.volume_history.append(volume)
            
            # Update market regime indicators
            self.current_volatility_regime = self._calculate_volatility()
            self.current_trend_strength = self._calculate_trend_strength()
            
            # Calculate KAMA-based momentum
            momentum = self._calculate_momentum()
            
            # Calculate KAMA values for market snapshot
            kama_value = 0.0
            kama_direction = 'neutral'
            kama_strength = 0.0
            
            if len(self.price_history) >= 20:
                kama_data = self._calculate_kama_indicators()
                kama_value = kama_data['value']
                kama_direction = kama_data['direction']
                kama_strength = kama_data['strength']
            
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
                directional_bias=momentum
            )
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return MarketSnapshot(
                timestamp=time.time(),
                price=0.0, volume=0.0, volatility=1.0, momentum=0.0, trend_strength=0.0,
                directional_bias=0.0
            )
    
    def _fetch_historical_data(self, exchange: Exchange) -> None:
        """FIXED: Fetch historical OHLCV data from exchange"""
        try:
            self.logger.info(f"Fetching OHLCV data for {self.symbol}...")
            
            # Use the existing get_ohlcv method from exchange
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=100)
            
            if not ohlcv_data:
                self.logger.warning(f"No OHLCV data received for {self.symbol}")
                return
            
            # Clear existing data and populate with historical data
            self.price_history.clear()
            self.volume_history.clear()
            
            # Process OHLCV data: [timestamp, open, high, low, close, volume]
            for candle in ohlcv_data:
                if len(candle) >= 6:
                    timestamp, open_price, high, low, close, volume = candle[:6]
                    # Use close price for technical analysis
                    self.price_history.append(float(close))
                    self.volume_history.append(float(volume))
            
            self.last_ohlcv_fetch = time.time()
            self.ohlcv_data_fetched = True
            
            self.logger.info(f"✅ OHLCV data loaded: {len(self.price_history)} price points")
            if len(self.price_history) > 0:
                self.logger.info(f"   Price range: ${min(self.price_history):.6f} - ${max(self.price_history):.6f}")
                self.logger.info(f"   Latest close: ${self.price_history[-1]:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")
            # Don't clear existing data on error
    
    def _calculate_momentum(self) -> float:
        """Calculate momentum using real KAMA instead of fake moving averages"""
        if len(self.price_history) < 20:
            required = 45  # KAMA(40) + buffer
            available = len(self.price_history)
            self.logger.warning(f"⚠️ Insufficient data for KAMA(40): {available}/{required}")
            return 0.0
        
        try:
            import pandas as pd
            import pandas_ta as ta
            
            prices = list(self.price_history)
            price_series = pd.Series(prices)
            
            # Calculate KAMA using pandas-ta
            kama = ta.kama(price_series)
            
            if kama is None or kama.isna().all():
                self.logger.warning("pandas-ta KAMA calculation failed, using fallback")
                return self._fallback_momentum()
            
            # Use KAMA for momentum calculation
            current_price = prices[-1]
            current_kama = float(kama.iloc[-1])
            
            if current_kama > 0:
                # Price position relative to KAMA
                momentum = (current_price - current_kama) / current_kama
                return max(-1.0, min(1.0, momentum * 2))
            
            return 0.0
            
        except ImportError:
            self.logger.warning("pandas-ta not available, using fallback momentum")
            return self._fallback_momentum()
        except Exception as e:
            self.logger.error(f"Error calculating KAMA momentum: {e}")
            return self._fallback_momentum()
    
    def _calculate_kama_indicators(self) -> Dict[str, Any]:
        """Calculate KAMA value, direction, and strength"""
        try:
            import pandas as pd
            import pandas_ta as ta
            
            prices = list(self.price_history)
            price_series = pd.Series(prices)
            
            # Calculate KAMA
            kama = ta.kama(price_series)
            
            if kama is None or kama.isna().all():
                return {'value': 0.0, 'direction': 'neutral', 'strength': 0.0}
            
            current_kama = float(kama.iloc[-1])
            self.kama_history.append(current_kama)
            
            # Determine KAMA direction and strength
            direction = 'neutral'
            strength = 0.0
            
            if len(kama) >= 5:
                # Calculate KAMA slope over recent periods
                recent_kama = kama.iloc[-5:]
                kama_slope = recent_kama.iloc[-1] - recent_kama.iloc[0]
                
                if current_kama > 0:
                    slope_percentage = kama_slope / current_kama
                    
                    if slope_percentage > 0.003:  # 0.3% upward slope
                        direction = 'bullish'
                        strength = min(1.0, abs(slope_percentage) * 100)
                    elif slope_percentage < -0.003:  # 0.3% downward slope
                        direction = 'bearish'
                        strength = min(1.0, abs(slope_percentage) * 100)
            
            return {
                'value': current_kama,
                'direction': direction,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating KAMA indicators: {e}")
            return {'value': 0.0, 'direction': 'neutral', 'strength': 0.0}
    
    def _fallback_momentum(self) -> float:
        """Fallback momentum calculation when KAMA is not available"""
        if len(self.price_history) < 10:
            return 0.0
        
        prices = list(self.price_history)
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-10:]) / 10
        
        if long_ma > 0:
            momentum = (short_ma - long_ma) / long_ma
            return max(-1.0, min(1.0, momentum * 5))
        
        return 0.0
    
    def _calculate_volatility(self) -> float:
        """Calculate price volatility"""
        if len(self.price_history) < 10:
            return 1.0
        
        prices = list(self.price_history)[-10:]
        returns = []
        
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 1.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        # Normalize to reasonable range
        return max(0.5, min(3.0, volatility * 100))
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength based on directional consistency"""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = list(self.price_history)
        up_moves = 0
        down_moves = 0
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                up_moves += 1
            elif prices[i] < prices[i-1]:
                down_moves += 1
        
        total_moves = up_moves + down_moves
        if total_moves == 0:
            return 0.0
        
        # Measure directional bias
        directional_bias = abs(up_moves - down_moves) / total_moves
        return min(1.0, directional_bias * 2)


class GridStrategy:
    """
    Simplified Grid Trading Strategy without Zone complexity
    
    Key Features:
    - Direct order placement within user-defined range
    - Real KAMA-based market intelligence
    - Simple investment tracking (position + order margins)
    - No artificial zone restrictions
    """
    
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
        """
        Initialize simplified grid strategy
        
        Args:
            exchange: Exchange interface
            symbol: Trading symbol
            price_lower: Lower bound of grid range
            price_upper: Upper bound of grid range
            grid_number: Number of grid levels
            investment: Total investment amount
            take_profit_pnl: Take profit PnL amount
            stop_loss_pnl: Stop loss PnL amount
            grid_id: Unique grid identifier
            leverage: Trading leverage
            enable_grid_adaptation: Enable adaptive features
            enable_samig: Enable SAMIG market intelligence
        """
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Exchange and symbol
        self.exchange = exchange
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        
        # User parameters (immutable)
        self.user_price_lower = float(price_lower)
        self.user_price_upper = float(price_upper)
        self.user_grid_number = int(grid_number)
        self.user_total_investment = float(investment)
        self.user_investment_per_grid = self.user_total_investment / self.user_grid_number
        self.user_leverage = float(leverage)
        
        # Strategy settings
        self.take_profit_pnl = float(take_profit_pnl)
        self.stop_loss_pnl = float(stop_loss_pnl)
        self.grid_id = grid_id
        self.enable_grid_adaptation = enable_grid_adaptation
        self.enable_samig = enable_samig
        
        # Core tracking (simplified - no zones)
        self.pending_orders: Dict[str, Dict] = {}
        self.all_positions: Dict[str, GridPosition] = {}
        self.max_total_orders = self.user_grid_number * 2  # Allow some buffer
        
        # Investment tracking
        self.total_investment_used = 0.0
        
        # Market intelligence
        if self.enable_samig:
            self.market_intel = MarketIntelligence(symbol)
        else:
            self.market_intel = None
        
        # State management
        self.running = False
        self.total_trades = 0
        self.total_pnl = 0.0
        self.last_update_time = 0
        
        # Threading
        self.update_lock = threading.Lock()
        
        # Market information
        self._fetch_market_info()
        
        self.logger.info(f"🚀 Simplified Grid Strategy Initialized:")
        self.logger.info(f"  Symbol: {symbol} → {self.symbol}")
        self.logger.info(f"  Range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
        self.logger.info(f"  Grid levels: {self.user_grid_number}")
        self.logger.info(f"  Investment per grid: ${self.user_investment_per_grid:.2f}")
        self.logger.info(f"  Total investment: ${self.user_total_investment:.2f}")
        self.logger.info(f"  Leverage: {self.user_leverage}x")
        self.logger.info(f"  SAMIG enabled: {self.enable_samig}")
    
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
            
            self.logger.info(f"📊 Market Info:")
            self.logger.info(f"  Price precision: {self.price_precision}")
            self.logger.info(f"  Amount precision: {self.amount_precision}")
            self.logger.info(f"  Min amount: {self.min_amount}")
            self.logger.info(f"  Min cost: {self.min_cost}")
            
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
        
        # Dynamic precision based on price magnitude
        if price < 0.01:
            precision = max(8, self.price_precision)
        elif price < 1:
            precision = max(6, self.price_precision)
        elif price < 100:
            precision = max(4, self.price_precision)
        else:
            precision = max(2, self.price_precision)
        
        return float(f"{price:.{precision}f}")
    
    def _round_amount(self, amount: float) -> float:
        """Round amount to appropriate precision"""
        if amount <= 0:
            return 0.0
        
        # Dynamic precision based on amount magnitude
        if amount < 0.01:
            precision = max(8, self.amount_precision)
        elif amount < 1:
            precision = max(6, self.amount_precision)
        else:
            precision = max(4, self.amount_precision)
        
        return float(f"{amount:.{precision}f}")
    
    def _calculate_order_amount(self, price: float) -> float:
        """Calculate order amount for given price and investment per grid"""
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
            
            # Final validation
            final_notional = price * rounded_amount
            if final_notional > self.max_cost:
                rounded_amount = self.max_cost / price
                rounded_amount = self._round_amount(rounded_amount)
            
            return max(self.min_amount, rounded_amount)
            
        except Exception as e:
            self.logger.error(f"Error calculating order amount for price ${price:.6f}: {e}")
            return self.min_amount
    
    def _calculate_grid_levels(self) -> List[float]:
        """Calculate grid levels within user-defined range"""
        try:
            if self.user_grid_number <= 0:
                return []
            
            # Calculate interval between grid levels
            price_range = self.user_price_upper - self.user_price_lower
            if price_range <= 0:
                self.logger.error(f"Invalid price range: {self.user_price_lower} - {self.user_price_upper}")
                return []
            
            interval = price_range / self.user_grid_number
            levels = []
            
            # Generate grid levels
            for i in range(self.user_grid_number + 1):
                level = self.user_price_lower + (i * interval)
                rounded_level = self._round_price(level)
                levels.append(rounded_level)
            
            # Remove duplicates and sort
            levels = sorted(list(set(levels)))
            
            self.logger.debug(f"Generated {len(levels)} grid levels: {levels[:3]}...{levels[-3:] if len(levels) > 6 else levels}")
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating grid levels: {e}")
            return []
    
    def _get_live_market_data(self) -> Dict[str, Any]:
        """Get live exchange data for accurate investment calculation"""
        try:
            # Get live positions and orders
            live_positions = self.exchange.get_positions(self.symbol)
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            # Clean up stale internal tracking
            live_order_ids = {order['id'] for order in live_orders}
            stale_orders = []
            
            for order_id in list(self.pending_orders.keys()):
                if order_id not in live_order_ids:
                    stale_orders.append(order_id)
            
            # Remove stale orders
            for order_id in stale_orders:
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                    self.logger.debug(f"Cleaned stale order: {order_id}")
            
            # Calculate position margin usage
            position_margin = 0.0
            active_positions = 0
            
            for position in live_positions:
                size = float(position.get('contracts', 0))
                if size != 0:
                    entry_price = float(position.get('entryPrice', 0))
                    if entry_price > 0:
                        position_notional = abs(size) * entry_price
                        margin = round(position_notional / self.user_leverage, 2)
                        position_margin += margin
                        active_positions += 1
            
            # Calculate order margin usage
            order_margin = 0.0
            active_orders = 0
            
            for order in live_orders:
                price = float(order.get('price', 0))
                amount = float(order.get('amount', 0))
                if price > 0 and amount > 0:
                    order_notional = price * amount
                    margin = round(order_notional / self.user_leverage, 2)
                    order_margin += margin
                    active_orders += 1
            
            # Calculate totals
            total_margin_used = round(position_margin + order_margin, 2)
            available_investment = round(self.user_total_investment - total_margin_used, 2)
            
            # Update internal tracking
            self.total_investment_used = total_margin_used
            
            # Create price coverage set
            covered_prices = set()
            for order in live_orders:
                price = float(order.get('price', 0))
                if price > 0:
                    covered_prices.add(self._round_price(price))
            
            for position in live_positions:
                if float(position.get('contracts', 0)) != 0:
                    entry_price = float(position.get('entryPrice', 0))
                    if entry_price > 0:
                        covered_prices.add(self._round_price(entry_price))
            
            market_data = {
                'live_orders': live_orders,
                'live_positions': live_positions,
                'position_margin': position_margin,
                'order_margin': order_margin,
                'total_margin_used': total_margin_used,
                'available_investment': available_investment,
                'active_orders': active_orders,
                'active_positions': active_positions,
                'total_commitment': active_orders + active_positions,
                'covered_prices': covered_prices,
                'stale_orders_cleaned': len(stale_orders)
            }
            
            if len(stale_orders) > 0:
                self.logger.info(f"🧹 Cleaned {len(stale_orders)} stale order records")
            
            self.logger.debug(f"💰 Investment Status:")
            self.logger.debug(f"  Position margin: ${position_margin:.2f}")
            self.logger.debug(f"  Order margin: ${order_margin:.2f}")
            self.logger.debug(f"  Total used: ${total_margin_used:.2f}")
            self.logger.debug(f"  Available: ${available_investment:.2f}")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting live market data: {e}")
            # Return safe defaults
            return {
                'live_orders': [], 'live_positions': [], 'position_margin': 0.0, 'order_margin': 0.0,
                'total_margin_used': self.user_total_investment, 'available_investment': 0.0,
                'active_orders': 0, 'active_positions': 0, 'total_commitment': 0,
                'covered_prices': set(), 'stale_orders_cleaned': 0
            }
    
    def setup_grid(self):
        """Setup initial grid orders within user-defined range"""
        try:
            self.logger.info(f"🏗️ Setting up grid strategy...")
            
            # Get current market price
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            self.logger.info(f"Current price: ${current_price:.6f}")
            self.logger.info(f"User range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
            
            # Check if current price is within user range
            if current_price < self.user_price_lower or current_price > self.user_price_upper:
                self.logger.warning(f"⚠️ Current price ${current_price:.6f} is outside user range!")
                self.logger.warning(f"   Grid may not be effective immediately")
            
            # Cancel any existing orders to start fresh
            try:
                cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                if cancelled_orders:
                    self.logger.info(f"🗑️ Cancelled {len(cancelled_orders)} existing orders")
                    time.sleep(1)  # Wait for cancellations to process
            except Exception as e:
                self.logger.warning(f"Error cancelling existing orders: {e}")
            
            # Get current market data
            market_data = self._get_live_market_data()
            available_investment = market_data['available_investment']
            
            self.logger.info(f"💰 Available investment: ${available_investment:.2f}")
            
            # Place initial grid orders
            orders_placed = self._place_initial_grid_orders(current_price, available_investment)
            
            if orders_placed > 0:
                self.running = True
                self.logger.info(f"✅ Grid setup complete: {orders_placed} orders placed")
                self.logger.info(f"🎯 Grid is now active and running")
            else:
                self.logger.error(f"❌ Grid setup failed: no orders placed")
                self.running = False
                
        except Exception as e:
            self.logger.error(f"Error setting up grid: {e}")
            self.running = False
    
    def _place_initial_grid_orders(self, current_price: float, available_investment: float) -> int:
        """Place initial grid orders within user range"""
        try:
            # Get grid levels
            grid_levels = self._calculate_grid_levels()
            if not grid_levels:
                self.logger.error("No grid levels calculated")
                return 0
            
            # Calculate how many orders we can place
            max_orders_by_investment = int(available_investment / self.user_investment_per_grid)
            max_orders_by_capacity = min(self.max_total_orders, len(grid_levels))
            max_orders = min(max_orders_by_investment, max_orders_by_capacity)
            
            if max_orders <= 0:
                self.logger.warning(f"Cannot place orders:")
                self.logger.warning(f"  By investment: {max_orders_by_investment} (need ${self.user_investment_per_grid:.2f} each)")
                self.logger.warning(f"  By capacity: {max_orders_by_capacity}")
                return 0
            
            # Calculate minimum gap to avoid clustering
            price_range = self.user_price_upper - self.user_price_lower
            min_gap = price_range / self.user_grid_number * 0.25  # 25% of natural grid spacing
            
            # Get market intelligence for order distribution
            directional_bias = 0.0
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    directional_bias = market_snapshot.directional_bias
                    
                    self.logger.info(f"📊 Market Intelligence:")
                    self.logger.info(f"  KAMA: {market_snapshot.kama_direction} (value: {market_snapshot.kama_value:.6f})")
                    self.logger.info(f"  Directional bias: {directional_bias:.3f}")
                    self.logger.info(f"  Trend strength: {market_snapshot.trend_strength:.3f}")
                    
                except Exception as e:
                    self.logger.warning(f"Market intelligence failed: {e}")
            
            # Calculate order distribution based on bias
            if abs(directional_bias) > 0.3:
                if directional_bias > 0:  # Bullish bias
                    buy_ratio = 0.6
                else:  # Bearish bias
                    buy_ratio = 0.4
            else:  # Neutral market
                buy_ratio = 0.5
            
            buy_target = int(max_orders * buy_ratio)
            sell_target = max_orders - buy_target
            
            self.logger.info(f"🎯 Order targets: {buy_target} buy + {sell_target} sell = {max_orders} total")
            self.logger.info(f"   Min gap: ${min_gap:.6f} ({min_gap/current_price*100:.2f}% of current price)")
            
            # Sort levels by distance from current price (place closest first)
            sorted_levels = sorted(grid_levels, key=lambda x: abs(x - current_price))
            
            orders_placed = 0
            buy_orders_placed = 0
            sell_orders_placed = 0
            
            for level_price in sorted_levels:
                if orders_placed >= max_orders:
                    break
                
                # Skip levels too close to current price
                if abs(level_price - current_price) < min_gap:
                    self.logger.debug(f"Skipping ${level_price:.6f} - too close to current price")
                    continue
                
                # Determine order type based on position relative to current price
                if level_price < current_price and buy_orders_placed < buy_target:
                    order_type = OrderType.BUY.value
                elif level_price > current_price and sell_orders_placed < sell_target:
                    order_type = OrderType.SELL.value
                else:
                    continue
                
                # Place the order
                if self._place_single_order(level_price, order_type):
                    orders_placed += 1
                    if order_type == OrderType.BUY.value:
                        buy_orders_placed += 1
                    else:
                        sell_orders_placed += 1
                    
                    self.logger.debug(f"Order {orders_placed}/{max_orders}: {order_type.upper()} @ ${level_price:.6f}")
            
            self.logger.info(f"📈 Initial grid orders placed:")
            self.logger.info(f"  Buy orders: {buy_orders_placed}")
            self.logger.info(f"  Sell orders: {sell_orders_placed}")
            self.logger.info(f"  Total: {orders_placed}")
            
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error placing initial grid orders: {e}")
            return 0
    
    def _place_single_order(self, price: float, side: str) -> bool:
        """Place a single limit order with KAMA intelligence check"""
        try:
            # INTELLIGENCE CHECK - Prevent counter-trend orders
            if self.market_intel:
                try:
                    current_price = float(self.exchange.get_ticker(self.symbol)['last'])
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    
                    kama_strength = market_snapshot.kama_strength
                    kama_direction = market_snapshot.kama_direction
                    distance_pct = abs(price - current_price) / current_price
                    
                    # STRONG MOMENTUM RULES (Strength > 0.7)
                    if kama_strength > 0.7:
                        if kama_direction == 'bearish' and side == 'sell':
                            self.logger.info(f"🧠 INTELLIGENCE BLOCK: {side.upper()} @ ${price:.6f}")
                            self.logger.info(f"   Reason: STRONG bearish momentum ({kama_strength:.3f}) - avoid selling into downtrend")
                            return False
                        elif kama_direction == 'bullish' and side == 'buy':
                            self.logger.info(f"🧠 INTELLIGENCE BLOCK: {side.upper()} @ ${price:.6f}")
                            self.logger.info(f"   Reason: STRONG bullish momentum ({kama_strength:.3f}) - avoid buying into uptrend")
                            return False
                    
                    # MEDIUM MOMENTUM RULES (Strength 0.4-0.7)
                    elif kama_strength > 0.4:
                        is_counter_trend = (
                            (kama_direction == 'bearish' and side == 'sell' and distance_pct > 0.02) or
                            (kama_direction == 'bullish' and side == 'buy' and distance_pct > 0.02)
                        )
                        if is_counter_trend:
                            self.logger.info(f"🧠 INTELLIGENCE BLOCK: {side.upper()} @ ${price:.6f}")
                            self.logger.info(f"   Reason: MEDIUM {kama_direction} momentum - avoid counter-trend >2%")
                            return False
                    
                except Exception as e:
                    self.logger.warning(f"Intelligence check failed: {e}")
            
            # Calculate order amount
            amount = self._calculate_order_amount(price)
            
            # Validate order parameters
            if amount < self.min_amount:
                self.logger.warning(f"Order amount {amount:.6f} below minimum {self.min_amount}")
                return False
            
            notional_value = price * amount
            if notional_value < self.min_cost:
                self.logger.warning(f"Order notional ${notional_value:.2f} below minimum ${self.min_cost}")
                return False
            
            # Place the order
            order = self.exchange.create_limit_order(self.symbol, side, amount, price)
            
            if not order or 'id' not in order:
                self.logger.error(f"Failed to place {side} order - invalid response")
                return False
            
            # Calculate margin used
            margin_used = round(notional_value / self.user_leverage, 2)
            
            # Store order information
            order_info = {
                'type': side,
                'price': price,
                'amount': amount,
                'notional_value': notional_value,
                'margin_used': margin_used,
                'timestamp': time.time(),
                'status': 'open'
            }
            
            self.pending_orders[order['id']] = order_info
            
            self.logger.info(f"✅ SMART {side.upper()}: ${price:.6f} x {amount:.6f} (margin: ${margin_used:.2f})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to place {side} order at ${price:.6f}: {e}")
            return False
    
    def update_grid(self):
        """Main grid update loop - simplified without complex rebalancing"""
        try:
            with self.update_lock:
                if not self.running:
                    return
                
                # Get current market price
                ticker = self.exchange.get_ticker(self.symbol)
                current_price = float(ticker['last'])
                
                # Update filled orders and positions
                self._update_orders_and_positions()
                
                # Maintain grid coverage within user range
                self._maintain_grid_coverage(current_price)
                
                # Create counter orders for open positions
                self._maintain_counter_orders(current_price)
                
                # Update PnL calculations
                self._update_pnl(current_price)
                
                # Check take profit and stop loss
                self._check_tp_sl()
                
                self.last_update_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
    
    def _maintain_grid_coverage(self, current_price: float):
        """Ensure grid has adequate coverage within user range"""
        try:
            # Only add orders if price is within user range
            if current_price < self.user_price_lower or current_price > self.user_price_upper:
                return
            
            # Get current market data
            market_data = self._get_live_market_data()
            
            # Check if we have investment and capacity for more orders
            if market_data['available_investment'] < self.user_investment_per_grid:
                return
            
            if market_data['total_commitment'] >= self.max_total_orders:
                return
            
            # Find gaps in grid coverage and fill them
            self._fill_grid_gaps(current_price, market_data)
            
        except Exception as e:
            self.logger.error(f"Error maintaining grid coverage: {e}")
    
    def _fill_grid_gaps(self, current_price: float, market_data: Dict[str, Any]):
        """Fill gaps in grid coverage within user range"""
        try:
            # Get current covered prices
            covered_prices = market_data['covered_prices']
            
            # Get grid levels
            grid_levels = self._calculate_grid_levels()
            if not grid_levels:
                return
            
            # Calculate minimum gap
            price_range = self.user_price_upper - self.user_price_lower
            min_gap = price_range / self.user_grid_number * 0.3
            
            # Find levels that need orders
            levels_needing_orders = []
            
            for level_price in grid_levels:
                # Skip if too close to current price
                if abs(level_price - current_price) < min_gap:
                    continue
                
                # Skip if already covered
                is_covered = any(abs(level_price - covered) < min_gap for covered in covered_prices)
                if is_covered:
                    continue
                
                levels_needing_orders.append(level_price)
            
            if not levels_needing_orders:
                return
            
            # Sort by distance from current price
            levels_needing_orders.sort(key=lambda x: abs(x - current_price))
            
            # Place one order at the most appropriate level
            for level_price in levels_needing_orders[:1]:  # Only place one at a time
                side = OrderType.BUY.value if level_price < current_price else OrderType.SELL.value
                
                if self._place_single_order(level_price, side):
                    self.logger.info(f"🔄 Added grid coverage: {side} @ ${level_price:.6f}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error filling grid gaps: {e}")
    
    def _update_orders_and_positions(self):
        """Check for filled orders and update position tracking"""
        try:
            # Get current open orders
            open_orders = self.exchange.get_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            # Check each pending order
            filled_order_ids = []
            for order_id in list(self.pending_orders.keys()):
                if order_id not in open_order_ids:
                    filled_order_ids.append(order_id)
            
            # Process filled orders
            for order_id in filled_order_ids:
                try:
                    order_status = self.exchange.get_order_status(order_id, self.symbol)
                    if order_status['status'] in ['filled', 'closed']:
                        self._process_filled_order(order_id, order_status)
                    else:
                        # Order was cancelled
                        if order_id in self.pending_orders:
                            order_info = self.pending_orders[order_id]
                            self.logger.info(f"📝 Order cancelled: {order_info['type']} @ ${order_info['price']:.6f}")
                            del self.pending_orders[order_id]
                except Exception as e:
                    self.logger.error(f"Error checking order status {order_id}: {e}")
                    # Remove problematic order
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                        
        except Exception as e:
            self.logger.error(f"Error updating orders and positions: {e}")
    
    def _process_filled_order(self, order_id: str, order_status: Dict):
        """Process a filled order and create position or close existing position"""
        try:
            order_info = self.pending_orders.get(order_id)
            if not order_info:
                self.logger.warning(f"Order {order_id} not found in pending orders")
                return
            
            # Get fill details
            fill_price = float(order_status.get('average', order_info['price']))
            fill_amount = float(order_status.get('filled', order_info['amount']))
            
            # Check if this is a counter order (closing a position)
            if 'position_id' in order_info:
                self._close_position_from_counter_order(order_id, order_status, order_info)
            else:
                # This is a new position
                self._create_position_from_filled_order(order_id, order_status, order_info)
            
            # Remove from pending orders
            del self.pending_orders[order_id]
            
        except Exception as e:
            self.logger.error(f"Error processing filled order {order_id}: {e}")
    
    def _create_position_from_filled_order(self, order_id: str, order_status: Dict, order_info: Dict):
        """Create new position from filled grid order"""
        try:
            fill_price = float(order_status.get('average', order_info['price']))
            fill_amount = float(order_status.get('filled', order_info['amount']))
            
            # Create new position
            position_id = str(uuid.uuid4())
            position_side = PositionSide.LONG.value if order_info['type'] == OrderType.BUY.value else PositionSide.SHORT.value
            
            position = GridPosition(
                position_id=position_id,
                entry_price=fill_price,
                quantity=fill_amount,
                side=position_side,
                entry_time=time.time()
            )
            
            self.all_positions[position_id] = position
            self.total_trades += 1
            
            self.logger.info(f"🎯 New position opened:")
            self.logger.info(f"   Position ID: {position_id[:8]}...")
            self.logger.info(f"   Side: {position_side.upper()}")
            self.logger.info(f"   Entry: ${fill_price:.6f}")
            self.logger.info(f"   Quantity: {fill_amount:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error creating position from filled order: {e}")
    
    def _close_position_from_counter_order(self, order_id: str, order_status: Dict, order_info: Dict):
        """Close position from filled counter order"""
        try:
            position_id = order_info['position_id']
            
            if position_id not in self.all_positions:
                self.logger.warning(f"Position {position_id} not found for counter order {order_id}")
                return
            
            position = self.all_positions[position_id]
            
            # Update position with exit details
            position.exit_time = time.time()
            position.exit_price = float(order_status.get('average', order_info['price']))
            position.has_counter_order = False
            position.counter_order_id = None
            
            # Calculate realized PnL
            if position.side == PositionSide.LONG.value:
                position.realized_pnl = (position.exit_price - position.entry_price) * position.quantity
            else:
                position.realized_pnl = (position.entry_price - position.exit_price) * position.quantity
            
            # Reset unrealized PnL
            position.unrealized_pnl = 0.0
            
            self.logger.info(f"💰 Position closed:")
            self.logger.info(f"   Position ID: {position_id[:8]}...")
            self.logger.info(f"   Entry: ${position.entry_price:.6f} → Exit: ${position.exit_price:.6f}")
            self.logger.info(f"   Realized PnL: ${position.realized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position from counter order: {e}")
    
    def _maintain_counter_orders(self, current_price: float):
        """Create counter orders for open positions"""
        try:
            for position in self.all_positions.values():
                # Skip closed positions or positions that already have counter orders
                if not position.is_open() or position.has_counter_order:
                    continue
                
                # Create counter order
                self._create_counter_order_for_position(position, current_price)
                
        except Exception as e:
            self.logger.error(f"Error maintaining counter orders: {e}")
    
    def _create_counter_order_for_position(self, position: GridPosition, current_price: float):
        """Create counter order for a specific position"""
        try:
            # Calculate counter order parameters
            if position.side == PositionSide.LONG.value:
                counter_side = OrderType.SELL.value
                # Target small profit above entry price
                profit_target = position.entry_price * 1.008  # 0.8% profit
                market_target = current_price * 1.004  # Or 0.4% above current
                counter_price = max(profit_target, market_target)
            else:
                counter_side = OrderType.BUY.value
                # Target small profit below entry price
                profit_target = position.entry_price * 0.992  # 0.8% profit
                market_target = current_price * 0.996  # Or 0.4% below current
                counter_price = min(profit_target, market_target)
            
            counter_price = self._round_price(counter_price)
            
            # Validate counter order makes sense
            if counter_side == OrderType.SELL.value and counter_price <= position.entry_price * 1.002:
                self.logger.debug(f"Counter price too low for profit: ${counter_price:.6f} vs ${position.entry_price:.6f}")
                return
            elif counter_side == OrderType.BUY.value and counter_price >= position.entry_price * 0.998:
                self.logger.debug(f"Counter price too high for profit: ${counter_price:.6f} vs ${position.entry_price:.6f}")
                return
            
            # Place counter order
            order = self.exchange.create_limit_order(self.symbol, counter_side, position.quantity, counter_price)
            
            if not order or 'id' not in order:
                self.logger.error(f"Failed to create counter order")
                return
            
            # Store counter order info
            counter_order_info = {
                'type': counter_side,
                'price': counter_price,
                'amount': position.quantity,
                'position_id': position.position_id,
                'timestamp': time.time(),
                'status': 'open'
            }
            
            self.pending_orders[order['id']] = counter_order_info
            
            # Update position
            position.has_counter_order = True
            position.counter_order_id = order['id']
            
            expected_profit = abs(counter_price - position.entry_price) * position.quantity
            self.logger.info(f"🔄 Counter order created:")
            self.logger.info(f"   Position: {position.side} @ ${position.entry_price:.6f}")
            self.logger.info(f"   Counter: {counter_side} @ ${counter_price:.6f}")
            self.logger.info(f"   Expected profit: ${expected_profit:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error creating counter order for position {position.position_id}: {e}")
    
    def _update_pnl(self, current_price: float):
        """Update unrealized and total PnL"""
        try:
            total_unrealized = 0.0
            total_realized = 0.0
            
            # Calculate PnL for all positions
            for position in self.all_positions.values():
                if position.is_open():
                    # Update unrealized PnL for open positions
                    position.unrealized_pnl = position.calculate_unrealized_pnl(current_price)
                    total_unrealized += position.unrealized_pnl
                else:
                    # Add realized PnL from closed positions
                    total_realized += position.realized_pnl
            
            # Update total PnL
            self.total_pnl = total_realized + total_unrealized
            
            # Log PnL status periodically
            if self.last_update_time == 0 or time.time() - self.last_update_time > 300:  # Every 5 minutes
                pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0
                
                self.logger.info(f"📊 PnL Update:")
                self.logger.info(f"   Realized: ${total_realized:.2f}")
                self.logger.info(f"   Unrealized: ${total_unrealized:.2f}")
                self.logger.info(f"   Total: ${self.total_pnl:.2f} ({pnl_percentage:.2f}%)")
            
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
                self.logger.info(f"🎯 Take profit reached: {pnl_percentage:.2f}% >= {self.take_profit_pnl:.2f}%")
                self.stop_grid()
                return
            
            # Check stop loss
            if pnl_percentage <= -self.stop_loss_pnl:
                self.logger.info(f"🛑 Stop loss reached: {pnl_percentage:.2f}% <= -{self.stop_loss_pnl:.2f}%")
                self.stop_grid()
                return
                
        except Exception as e:
            self.logger.error(f"Error checking TP/SL: {e}")
    
    def stop_grid(self):
        """Stop the grid strategy and cleanup"""
        try:
            if not self.running:
                return
            
            self.logger.info(f"🛑 Stopping grid strategy...")
            
            # Set running to false first
            self.running = False
            
            # Cancel all pending orders
            try:
                cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                if cancelled_orders:
                    self.logger.info(f"🗑️ Cancelled {len(cancelled_orders)} pending orders")
                
                # Clear pending orders tracking
                self.pending_orders.clear()
                
            except Exception as e:
                self.logger.error(f"Error cancelling orders during stop: {e}")
            
            # Close all open positions at market price
            try:
                open_positions = [pos for pos in self.all_positions.values() if pos.is_open()]
                
                for position in open_positions:
                    self._close_position_at_market(position)
                
                if open_positions:
                    self.logger.info(f"💹 Closed {len(open_positions)} positions at market price")
                    
            except Exception as e:
                self.logger.error(f"Error closing positions during stop: {e}")
            
            # Final PnL calculation
            final_pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0
            
            self.logger.info(f"📈 Grid strategy stopped:")
            self.logger.info(f"   Total trades: {self.total_trades}")
            self.logger.info(f"   Final PnL: ${self.total_pnl:.2f} ({final_pnl_percentage:.2f}%)")
            self.logger.info(f"   Strategy duration: {(time.time() - (self.last_update_time or time.time()))/3600:.1f} hours")
            
        except Exception as e:
            self.logger.error(f"Error stopping grid: {e}")
            self.running = False
    
    def _close_position_at_market(self, position: GridPosition):
        """Close position at current market price"""
        try:
            # Determine order side to close position
            close_side = OrderType.SELL.value if position.side == PositionSide.LONG.value else OrderType.BUY.value
            
            # Place market order to close position
            order = self.exchange.create_market_order(self.symbol, close_side, position.quantity)
            
            if order and 'average' in order:
                # Update position with market close details
                position.exit_time = time.time()
                position.exit_price = float(order['average'])
                position.has_counter_order = False
                position.counter_order_id = None
                
                # Calculate final realized PnL
                if position.side == PositionSide.LONG.value:
                    position.realized_pnl = (position.exit_price - position.entry_price) * position.quantity
                else:
                    position.realized_pnl = (position.entry_price - position.exit_price) * position.quantity
                
                position.unrealized_pnl = 0.0
                
                self.logger.info(f"💹 Position closed at market:")
                self.logger.info(f"   {position.side} @ ${position.entry_price:.6f} → ${position.exit_price:.6f}")
                self.logger.info(f"   PnL: ${position.realized_pnl:.2f}")
                
            else:
                self.logger.error(f"Failed to close position {position.position_id} at market")
                
        except Exception as e:
            self.logger.error(f"Error closing position at market: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive grid strategy status"""
        try:
            with self.update_lock:
                # Get current market data
                market_data = self._get_live_market_data()
                
                # Count open positions
                open_positions = [pos for pos in self.all_positions.values() if pos.is_open()]
                closed_positions = [pos for pos in self.all_positions.values() if not pos.is_open()]
                
                # Calculate PnL percentage
                pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0.0
                
                # Basic status
                status = {
                    # Identification
                    'grid_id': self.grid_id,
                    'symbol': self.symbol,
                    'display_symbol': self.original_symbol,
                    
                    # User configuration
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
                    
                    # Current state
                    'running': self.running,
                    'active_positions': len(open_positions),
                    'closed_positions': len(closed_positions),
                    'orders_count': market_data['active_orders'],
                    'trades_count': self.total_trades,
                    
                    # Investment tracking
                    'total_investment_used': market_data['total_margin_used'],
                    'available_investment': market_data['available_investment'],
                    
                    # Performance
                    'pnl': self.total_pnl,
                    'pnl_percentage': pnl_percentage,
                    'last_update': self.last_update_time,
                    
                    # Additional info
                    'max_total_orders': self.max_total_orders,
                    'investment_per_grid': self.user_investment_per_grid,
                }
                
                return status
                
        except Exception as e:
            self.logger.error(f"Error getting grid status: {e}")
            return {
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'running': False,
                'error': str(e)
            }


# Module-level utility functions
def create_grid_strategy(exchange: Exchange, config: Dict[str, Any]) -> GridStrategy:
    """
    Factory function to create a GridStrategy instance
    
    Args:
        exchange: Exchange interface
        config: Configuration dictionary with required parameters
        
    Returns:
        GridStrategy instance
    """
    required_fields = [
        'symbol', 'price_lower', 'price_upper', 'grid_number', 
        'investment', 'take_profit_pnl', 'stop_loss_pnl'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    return GridStrategy(
        exchange=exchange,
        symbol=config['symbol'],
        price_lower=config['price_lower'],
        price_upper=config['price_upper'],
        grid_number=config['grid_number'],
        investment=config['investment'],
        take_profit_pnl=config['take_profit_pnl'],
        stop_loss_pnl=config['stop_loss_pnl'],
        grid_id=config.get('grid_id', str(uuid.uuid4())),
        leverage=config.get('leverage', 20.0),
        enable_grid_adaptation=config.get('enable_grid_adaptation', True),
        enable_samig=config.get('enable_samig', False)
    )


# Export main classes and functions
__all__ = [
    'GridStrategy',
    'GridPosition', 
    'MarketSnapshot',
    'MarketIntelligence',
    'OrderType',
    'PositionSide',
    'create_grid_strategy'
]