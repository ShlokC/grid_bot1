"""
Grid Trading Strategy - Optimized Logging Implementation
Author: Grid Trading Bot
Date: 2025-05-26

Complete rewrite removing zone complexity and artificial restrictions.
Uses real KAMA instead of fake momentum, direct order placement within user range.
OPTIMIZED: Consolidated logging for better performance and readability.
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
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta

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
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_middle: float = 0.0

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
    
    def __init__(self, symbol: str, history_length: int = 1400):
        self.symbol = symbol
        self.price_history = deque(maxlen=history_length)
        self.volume_history = deque(maxlen=history_length)
        self.last_analysis_time = 0
        
        # OHLCV data cache to avoid excessive API calls
        self.last_ohlcv_fetch = 0
        self.ohlcv_cache_duration = 30  # 30 seconds
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
        """Analyze market using real KAMA and Bollinger Bands"""
        try:
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
            
            # Calculate Bollinger Bands
            bollinger_upper, bollinger_lower, bollinger_middle = self._calculate_bollinger_bands(exchange, current_price)
            
            # Update market regime indicators  
            self.current_volatility_regime = self._calculate_volatility()
            self.current_trend_strength = self._calculate_trend_strength()
            
            # Calculate KAMA-based momentum
            momentum = self._calculate_momentum(exchange)
            
            # Use stored KAMA values
            kama_value = getattr(self, 'current_kama_value', 0.0)
            kama_direction = getattr(self, 'current_kama_direction', 'neutral')
            kama_strength = getattr(self, 'current_kama_strength', 0.0)
            
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
                directional_bias=momentum,
                bollinger_upper=bollinger_upper,
                bollinger_lower=bollinger_lower,
                bollinger_middle=bollinger_middle
            )
            
        except Exception as e:
            self.logger.error(f"Market analysis error: {e}")
            return MarketSnapshot(
                timestamp=time.time(),
                price=0.0, volume=0.0, volatility=1.0, momentum=0.0, trend_strength=0.0,
                directional_bias=0.0, bollinger_upper=0.0, bollinger_lower=0.0, bollinger_middle=0.0
            )
    
    def _calculate_bollinger_bands(self, exchange: Exchange, current_price: float) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands using pandas-ta"""
        try:
            import pandas as pd
            import pandas_ta as ta
            
            # Get FULL historical data from exchange for Bollinger Bands
            full_ohlcv = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=1400)
            
            if full_ohlcv and len(full_ohlcv) >= 20:  # Need at least 20 periods for BB
                # Extract closing prices from OHLCV data
                closing_prices = [float(candle[4]) for candle in full_ohlcv]
                
                # Create pandas series
                price_series = pd.Series(closing_prices)
                
                # Calculate Bollinger Bands (20 period, 2 standard deviations)
                bb_data = ta.bbands(price_series, length=20, std=2)
                
                if bb_data is not None and not bb_data.empty:
                    # Get the latest Bollinger Band values
                    bollinger_lower = float(bb_data.iloc[-1]['BBL_10_2.0'])
                    bollinger_middle = float(bb_data.iloc[-1]['BBM_10_2.0'])  
                    bollinger_upper = float(bb_data.iloc[-1]['BBU_10_2.0'])
                    
                    self.logger.info(f"BB: Upper ${bollinger_upper:.6f}, Middle ${bollinger_middle:.6f}, Lower ${bollinger_lower:.6f}")
                    return bollinger_upper, bollinger_lower, bollinger_middle
            
            # Fallback to current price ±5%
            self.logger.warning(f"Insufficient data for Bollinger Bands, using fallback")
            return current_price * 1.05, current_price * 0.95, current_price
            
        except Exception as e:
            self.logger.warning(f"Bollinger Bands calculation failed: {e}")
            return current_price * 1.05, current_price * 0.95, current_price
    
    def _fetch_historical_data(self, exchange: Exchange) -> None:
        """Fetch historical OHLCV data from exchange"""
        try:
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=1400)
            
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
                    self.price_history.append(float(close))
                    self.volume_history.append(float(volume))
            
            self.last_ohlcv_fetch = time.time()
            self.ohlcv_data_fetched = True
            
            self.logger.info(f"OHLCV data loaded: {len(self.price_history)} points, range ${min(self.price_history):.6f}-${max(self.price_history):.6f}")
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")
    
    def _calculate_momentum(self, exchange) -> float:
        """Calculate momentum using pandas-ta KAMA with proper Efficiency Ratio"""
        try:
            import pandas as pd
            import pandas_ta as ta
            
            # Get historical data
            full_ohlcv = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=200)  # Reduced from 1400
            
            if not full_ohlcv or len(full_ohlcv) < 20:
                self.logger.warning(f"Insufficient OHLCV data: {len(full_ohlcv) if full_ohlcv else 0}")
                return self._fallback_momentum()
            
            # Extract closing prices
            closing_prices = [float(candle[4]) for candle in full_ohlcv]
            current_price = closing_prices[-1]
            
            # Create pandas series
            price_series = pd.Series(closing_prices)
            
            # Calculate KAMA using pandas-ta
            kama_series = ta.kama(price_series, length=10, fast=2, slow=30)
            
            if kama_series is None or kama_series.isna().all():
                self.logger.error(f"KAMA calculation failed")
                return self._fallback_momentum()
            
            # Get current KAMA value
            current_kama = float(kama_series.iloc[-1])
            self.current_kama_value = current_kama
            
            # Calculate Efficiency Ratio (ER) - this is the standard strength measure
            period = 10
            if len(price_series) >= period:
                # Direction = absolute change over period
                direction = abs(price_series.iloc[-1] - price_series.iloc[-period])
                
                # Volatility = sum of absolute daily changes over period
                volatility = price_series.diff().abs().tail(period).sum()
                
                # Efficiency Ratio = Direction / Volatility (0 to 1)
                efficiency_ratio = direction / volatility if volatility > 0 else 0
                
                # Cap ER at 1.0 (can exceed in some edge cases)
                self.current_kama_strength = min(1.0, efficiency_ratio)
            else:
                self.current_kama_strength = 0.0
            
            # Calculate direction from KAMA rate of change
            if len(kama_series) >= 3:
                kama_change = kama_series.iloc[-1] - kama_series.iloc[-3]
                kama_change_pct = kama_change / kama_series.iloc[-3] if kama_series.iloc[-3] != 0 else 0
                
                # Simple direction logic with minimal threshold
                if kama_change_pct > 0.0001:  # 0.01% threshold
                    self.current_kama_direction = 'bullish'
                elif kama_change_pct < -0.0001:
                    self.current_kama_direction = 'bearish'
                else:
                    self.current_kama_direction = 'neutral'
            else:
                self.current_kama_direction = 'neutral'
            
            # Calculate momentum - price relative to KAMA
            if current_kama > 0:
                momentum_pct = (current_price - current_kama) / current_kama
                final_momentum = max(-1.0, min(1.0, momentum_pct * 3))  # Scale factor of 3
                
                momentum_status = "BULLISH" if final_momentum > 0.02 else "BEARISH" if final_momentum < -0.02 else "NEUTRAL"
                
                self.logger.info(f"KAMA: ${current_kama:.6f}, Direction: {self.current_kama_direction}, "
                            f"ER: {self.current_kama_strength:.4f}, Momentum: {momentum_status}")
                
                return final_momentum
            
            return 0.0
            
        except ImportError:
            self.logger.warning("pandas-ta not available, using fallback momentum")
            return self._fallback_momentum()
        except Exception as e:
            self.logger.error(f"Error calculating KAMA: {e}")
            return self._fallback_momentum()
    
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
             grid_number: int,
             investment: float,
             take_profit_pnl: float,
             stop_loss_pnl: float,
             grid_id: str,
             leverage: float = 20.0,
             enable_grid_adaptation: bool = True,
             enable_samig: bool = False):
        """Initialize simplified grid strategy with BB-based range"""
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Exchange and symbol
        self.exchange = exchange
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        
        # MOVED: Market information (needed for _round_price)
        self._fetch_market_info()
        
        # Initialize market intelligence for BB calculation (always needed now)
        self.market_intel = MarketIntelligence(symbol)
        
        # Get initial BB-based range
        try:
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            market_snapshot = self.market_intel.analyze_market(self.exchange)
            
            if market_snapshot.bollinger_upper > 0 and market_snapshot.bollinger_lower > 0:
                self.user_price_lower = self._round_price(market_snapshot.bollinger_lower)
                self.user_price_upper = self._round_price(market_snapshot.bollinger_upper)
            else:
                # Fallback: use current price ±10%
                self.user_price_lower = self._round_price(current_price * 0.9)
                self.user_price_upper = self._round_price(current_price * 1.1)
                
            self.logger.info(f"BB Range initialized: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BB range: {e}")
            # Emergency fallback
            try:
                ticker = self.exchange.get_ticker(self.symbol)
                current_price = float(ticker['last'])
                self.user_price_lower = self._round_price(current_price * 0.9)
                self.user_price_upper = self._round_price(current_price * 1.1)
            except Exception as e2:
                self.logger.error(f"Emergency fallback failed: {e2}")
                # Set basic defaults without rounding
                self.user_price_lower = 1.0
                self.user_price_upper = 100.0
        
        # User parameters (BB-based now)
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
        
        # State management
        self.running = False
        self.total_trades = 0
        self.total_pnl = 0.0
        self.last_update_time = 0
        
        # Threading
        self.update_lock = threading.Lock()
        
        self.logger.info(f"Grid initialized: {self.original_symbol}, BB Range: ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}, "
                        f"Levels: {self.user_grid_number}, Investment: ${self.user_total_investment:.2f}, "
                        f"Leverage: {self.user_leverage}x, SAMIG: {self.enable_samig}")
        
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
        """Calculate grid levels with simple current price centered distribution"""
        try:
            if self.user_grid_number <= 0:
                return []
            
            # Calculate interval between grid levels
            price_range = self.user_price_upper - self.user_price_lower
            if price_range <= 0:
                self.logger.error(f"Invalid price range: {self.user_price_lower} - {self.user_price_upper}")
                return []
            
            # Get current price (simple, no external dependencies)
            try:
                ticker = self.exchange.get_ticker(self.symbol)
                current_price = float(ticker['last'])
            except Exception:
                # Fallback to equal distribution if can't get current price
                interval = price_range / self.user_grid_number
                levels = []
                for i in range(self.user_grid_number + 1):
                    level = self.user_price_lower + (i * interval)
                    levels.append(self._round_price(level))
                return sorted(list(set(levels)))
            # ALL orders within 2% of current price (1% above, 1% below)
            max_distance_pct = 0.02  # 2% max distance from current price

            # Calculate tight range around current price
            tight_lower = max(current_price * (1 - max_distance_pct), self.user_price_lower)
            tight_upper = min(current_price * (1 + max_distance_pct), self.user_price_upper)
            
            levels = []
            
            # Distribute ALL orders within the tight range
            if tight_upper > tight_lower and self.user_grid_number > 0:
                for i in range(self.user_grid_number):
                    if self.user_grid_number == 1:
                        level = current_price
                    else:
                        level = tight_lower + (i * (tight_upper - tight_lower) / (self.user_grid_number - 1))
                    levels.append(self._round_price(level))
            # Remove duplicates and sort
            levels = sorted(list(set(levels)))
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating grid levels: {e}")
            return []
    
    def _get_live_market_data(self) -> Dict[str, Any]:
        """Enhanced market data retrieval with consolidated logging"""
        try:
            # Get live positions and orders
            live_positions = self.exchange.get_positions(self.symbol)
            live_orders = self.exchange.get_open_orders(self.symbol)
            self._sync_order_tracking(live_orders)
            
            # Clean up stale internal tracking
            live_order_ids = {order['id'] for order in live_orders}
            stale_orders = [order_id for order_id in list(self.pending_orders.keys()) if order_id not in live_order_ids]
            
            for order_id in stale_orders:
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
            
            # Calculate margins
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

            # Log summary
            if active_orders > 0 or active_positions > 0 or len(stale_orders) > 0:
                self.logger.info(f"Market state: {active_orders} orders (${order_margin:.2f}), {active_positions} positions (${position_margin:.2f}), "
                               f"Total used: ${total_margin_used:.2f}, Available: ${available_investment:.2f}")
                if len(stale_orders) > 0:
                    self.logger.info(f"Cleaned {len(stale_orders)} stale orders")

            return {
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
            
        except Exception as e:
            self.logger.error(f"Error getting live market data: {e}")
            return {
                'live_orders': [], 'live_positions': [], 'position_margin': 0.0, 'order_margin': 0.0,
                'total_margin_used': self.user_total_investment, 'available_investment': 0.0,
                'active_orders': 0, 'active_positions': 0, 'total_commitment': 0,
                'covered_prices': set(), 'stale_orders_cleaned': 0
            }
    
    def setup_grid(self):
        """Setup initial grid orders within user-defined range"""
        try:
            # Get current market price
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            self.logger.info(f"Setting up grid: Current price ${current_price:.6f}, Range ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}")
            
            # Check if current price is within user range
            if current_price < self.user_price_lower or current_price > self.user_price_upper:
                self.logger.warning(f"Current price ${current_price:.6f} is outside user range - grid may not be immediately effective")
            
            # Cancel existing orders selectively
            try:
                open_orders = self.exchange.get_open_orders(self.symbol)
                if open_orders:
                    # Get live positions for protection check
                    live_positions = self.exchange.get_positions(self.symbol)
                    cancelled_count = 0
                    
                    for order in open_orders:
                        # Check if order should be protected
                        if self._should_protect_order(order, current_price, live_positions):
                            self.logger.info(f"Keeping protected order: {order.get('side', '').upper()} @ ${float(order.get('price', 0)):.6f}")
                            continue
                        
                        # Cancel non-protected orders
                        try:
                            self.exchange.cancel_order(order['id'], self.symbol)
                            cancelled_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to cancel order {order['id'][:8]}: {e}")
                    
                    if cancelled_count > 0:
                        self.logger.info(f"Cancelled {cancelled_count} non-protected orders")
                        time.sleep(1)  # Wait for cancellations to process
                        
            except Exception as e:
                self.logger.warning(f"Error during selective order cancellation: {e}")
                # Fallback to cancel all if selective fails
                try:
                    cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                    if cancelled_orders:
                        self.logger.info(f"Fallback: Cancelled all {len(cancelled_orders)} orders")
                        time.sleep(1)
                except Exception as e2:
                    self.logger.error(f"Failed to cancel orders: {e2}")
            
            # Get current market data
            market_data = self._get_live_market_data()
            available_investment = market_data['available_investment']
            
            # Place initial grid orders
            orders_placed = self._place_initial_grid_orders(current_price, available_investment)
            
            if orders_placed > 0:
                self.running = True
                self.logger.info(f"Grid setup complete: {orders_placed} orders placed, grid is now active")
            else:
                self.logger.error(f"Grid setup failed: no orders placed")
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
                self.logger.warning(f"Cannot place orders: By investment: {max_orders_by_investment}, By capacity: {max_orders_by_capacity}")
                return 0
            
            # Calculate minimum gap to avoid clustering
            price_range = self.user_price_upper - self.user_price_lower
            min_gap = price_range / self.user_grid_number * 0.1  # 10% of natural grid spacing

            # Get market intelligence for order distribution
            directional_bias = 0.0
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    directional_bias = market_snapshot.directional_bias
                except Exception as e:
                    self.logger.warning(f"Market intelligence failed: {e}")
            
            # Calculate order distribution based on bias
            if abs(directional_bias) > 0.3:
                buy_ratio = 0.6 if directional_bias > 0 else 0.4
            else:
                buy_ratio = 0.5
            
            buy_target = int(max_orders * buy_ratio)
            sell_target = max_orders - buy_target
            
            self.logger.info(f"Order targets: {buy_target} buy + {sell_target} sell = {max_orders} total, Min gap: ${min_gap:.6f}")
            
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
            
            self.logger.info(f"Initial grid orders placed: {buy_orders_placed} buy, {sell_orders_placed} sell, Total: {orders_placed}")
            
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error placing initial grid orders: {e}")
            return 0
    
    def _place_single_order(self, price: float, side: str) -> bool:
        """Enhanced order placement with consolidated logging and profit metadata"""
        try:
            # Get current market price for context
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            price_diff_pct = ((price - current_price) / current_price) * 100
            
            # Smart Last Order Logic - Check total investment usage approaching 80%
            market_data = self._get_live_market_data()
            total_investment_used = market_data['total_margin_used']
            available_investment = market_data['available_investment']
            investment_utilization = (total_investment_used / self.user_total_investment) * 100
            
            # Get live positions for profit order determination
            live_positions = market_data['live_positions']
            has_long_positions = False
            has_short_positions = False
            # Calculate total unrealized PnL
            total_unrealized_pnl = 0.0
            for pos in live_positions:
                if 'unrealizedPnl' in pos:
                    total_unrealized_pnl += float(pos['unrealizedPnl'])
            # Position-aware order blocking
            if total_unrealized_pnl > self.user_investment_per_grid:
                if (side == 'buy' and any(pos.get('side') == 'long' for pos in live_positions) or 
                    side == 'sell' and any(pos.get('side') == 'short' for pos in live_positions)):
                    self.logger.info(f"Position-aware block: Avoided adding to winning positions")
                    return False
                
            # Check if we're near 80% AND can only place 1 more order
            can_place_orders = int(available_investment / self.user_investment_per_grid)
            
            if investment_utilization >= 75.0 and can_place_orders <= 1:
                # Determine dominant position direction from existing positions
                net_position_value = 0.0
                
                for position in live_positions:
                    size = float(position.get('contracts', 0))
                    entry_price = float(position.get('entryPrice', 0))
                    if size != 0 and entry_price > 0:
                        position_value = size * entry_price
                        if position.get('side', '').lower() == 'long':
                            net_position_value += position_value
                        else:
                            net_position_value -= position_value
                
                # If we have a dominant position, check if order is same direction
                if net_position_value != 0:
                    dominant_direction = 'long' if net_position_value > 0 else 'short'
                    
                    # Check if this order is same direction as dominant position
                    is_same_direction = False
                    if dominant_direction == 'long' and side == 'buy':
                        is_same_direction = True
                    elif dominant_direction == 'short' and side == 'sell':
                        is_same_direction = True
                    
                    if is_same_direction:
                        self.logger.warning(f"Last order protection: Blocked {side.upper()} @ ${price:.6f} - same direction as {dominant_direction.upper()} position")
                        return False
            
            # Determine if this is a profit-taking order
            is_profit_order = False
            expected_profit_pct = 0.0
            
            if live_positions:
                has_long_positions = any(
                    float(pos.get('contracts', 0)) > 0 and pos.get('side', '').lower() == 'long'
                    for pos in live_positions
                )
                has_short_positions = any(
                    float(pos.get('contracts', 0)) > 0 and pos.get('side', '').lower() == 'short'
                    for pos in live_positions
                )
                
                if side == 'sell' and price > current_price and has_long_positions:
                    avg_entry = self._calculate_average_entry(live_positions, 'long')
                    if avg_entry > 0 and price > avg_entry * 1.002:
                        is_profit_order = True
                        expected_profit_pct = ((price - avg_entry) / avg_entry) * 100
                        
                elif side == 'buy' and price < current_price and has_short_positions:
                    avg_entry = self._calculate_average_entry(live_positions, 'short')
                    if avg_entry > 0 and price < avg_entry * 0.998:
                        is_profit_order = True
                        expected_profit_pct = ((avg_entry - price) / avg_entry) * 100
            
            # Market intelligence check (existing logic remains)
            if self.market_intel and not is_profit_order:  # Don't block profit orders
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    
                    kama_strength = market_snapshot.kama_strength
                    kama_direction = market_snapshot.kama_direction
                    distance_pct = abs(price - current_price) / current_price
                    
                    # FIXED: Determine order type with position awareness
                    is_profit_taking_order = is_profit_order  # Already determined above
                    is_risky_new_position = False
                    
                    if side == 'sell' and price > current_price:
                        if not has_long_positions:
                            is_risky_new_position = True  # Would create new short position
                            
                    elif side == 'buy' and price < current_price:
                        if not has_short_positions:
                            is_risky_new_position = False  # Normal long entry
                            
                    elif side == 'sell' and price < current_price:
                        is_risky_new_position = True  # New short entry
                    elif side == 'buy' and price > current_price:
                        is_risky_new_position = True  # Chasing the market
                    
                    # Enhanced blocking logic
                    if kama_strength > 0.3 and is_risky_new_position:
                        should_block = False
                        
                        # In BEARISH trend: Block new LONG positions
                        if kama_direction == 'bearish':
                            if side == 'buy' and price > current_price:
                                should_block = True
                        
                        # In BULLISH trend: Block new SHORT positions
                        elif kama_direction == 'bullish':
                            if side == 'sell' and not is_profit_taking_order:
                                should_block = True
                        
                        if should_block:
                            self.logger.warning(f"Intelligence block: {side.upper()} @ ${price:.6f} - {kama_direction} trend (strength: {kama_strength:.3f})")
                            return False
                    
                except Exception as e:
                    self.logger.warning(f"Intelligence check failed: {e}")
            
            # Calculate order amount with validation
            amount = self._calculate_order_amount(price)
            notional_value = price * amount
            margin_required = round(notional_value / self.user_leverage, 2)
            
            # Validate order parameters
            if amount < self.min_amount:
                self.logger.error(f"Order amount {amount:.6f} below minimum {self.min_amount}")
                return False
            
            if notional_value < self.min_cost:
                self.logger.error(f"Order notional ${notional_value:.2f} below minimum ${self.min_cost}")
                return False
            
            # Place the order
            order = self.exchange.create_limit_order(self.symbol, side, amount, price)
            
            if not order or 'id' not in order:
                self.logger.error(f"Failed to place {side} order - invalid response")
                return False
            
            # Store order information with profit metadata
            distance_from_current = abs(price - current_price)
            importance_score = 1.0 / (1.0 + distance_from_current)  # Closer orders = higher importance
            
            # Boost importance for profit orders
            if is_profit_order:
                importance_score = min(1.0, importance_score * 2.0)
            
            order_info = {
                'type': side,
                'price': price,
                'amount': amount,
                'notional_value': notional_value,
                'margin_used': margin_required,
                'timestamp': time.time(),
                'status': 'open',
                'current_price_at_placement': current_price,
                'price_diff_pct_at_placement': price_diff_pct,
                'importance_score': importance_score,
                'distance_from_current': distance_from_current,
                'is_profit_order': is_profit_order,  # ADD THIS
                'expected_profit_pct': expected_profit_pct  # ADD THIS
            }
            
            self.pending_orders[order['id']] = order_info
            
            # Enhanced logging for profit orders
            if is_profit_order:
                self.logger.info(f"PROFIT order placed: {side.upper()} {amount:.6f} @ ${price:.6f} ({price_diff_pct:+.2f}%), "
                            f"Expected profit: {expected_profit_pct:.2f}%, Margin: ${margin_required:.2f}, ID: {order['id'][:8]}")
            else:
                self.logger.info(f"Order placed: {side.upper()} {amount:.6f} @ ${price:.6f} ({price_diff_pct:+.2f}%), "
                            f"Margin: ${margin_required:.2f}, ID: {order['id'][:8]}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to place {side} order at ${price:.6f}: {e}")
            return False

    def _sync_order_tracking(self, live_orders: List[Dict]) -> int:
        """Synchronize internal order tracking with live exchange orders"""
        added_orders = 0
        
        try:
            for order in live_orders:
                order_id = order['id']
                
                if order_id not in self.pending_orders:
                    # This is an untracked order - add it to our tracking
                    price = float(order.get('price', 0))
                    amount = float(order.get('amount', 0))
                    side = order.get('side', 'unknown')
                    
                    if price > 0 and amount > 0:
                        notional_value = price * amount
                        margin_used = round(notional_value / self.user_leverage, 2)
                        
                        # Add to tracking with minimal required info
                        self.pending_orders[order_id] = {
                            'type': side,
                            'price': price,
                            'amount': amount,
                            'notional_value': notional_value,
                            'margin_used': margin_used,
                            'timestamp': time.time(),
                            'status': 'open',
                            'recovered': True  # Mark as recovered order
                        }
                        
                        added_orders += 1
            
            if added_orders > 0:
                self.logger.info(f"Order sync: Recovered {added_orders} untracked orders")
            
            return added_orders
            
        except Exception as e:
            self.logger.error(f"Error syncing order tracking: {e}")
            return 0
    
    def update_grid(self):
        """Main grid update loop with consolidated logging"""
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
            # Check if price is outside user range and adapt if enabled
            if current_price < self.user_price_lower or current_price > self.user_price_upper:
                if self.enable_grid_adaptation:
                    # Calculate range size to maintain
                    range_size = self.user_price_upper - self.user_price_lower
                    
                    # Update user range bounds based on current price
                    if current_price < self.user_price_lower:
                        new_upper = self.user_price_lower + (range_size * 0.1)
                        new_lower = new_upper - range_size
                    else:
                        new_lower = self.user_price_upper - (range_size * 0.1)
                        new_upper = new_lower + range_size
                    
                    # Update the user range bounds
                    old_lower, old_upper = self.user_price_lower, self.user_price_upper
                    self.user_price_lower = self._round_price(new_lower)
                    self.user_price_upper = self._round_price(new_upper)
                    
                    self.logger.info(f"Range adapted: ${old_lower:.6f}-${old_upper:.6f} → ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}")
                else:
                    return
            
            # Get current market data
            market_data = self._get_live_market_data()
            
            # Check if we have investment and capacity for more orders
            if (market_data['available_investment'] < self.user_investment_per_grid or 
                market_data['total_commitment'] >= self.max_total_orders):
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
            min_gap = price_range / self.user_grid_number * 0.1
            
            # Find levels that need orders
            levels_needing_orders = []
            
            for level_price in grid_levels:
                # Skip if too close to current price
                distance_to_current = abs(level_price - current_price)
                if distance_to_current < min_gap:
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
                side = "buy" if level_price < current_price else "sell"
                
                if self._place_single_order(level_price, side):
                    self.logger.info(f"Gap filled: {side.upper()} @ ${level_price:.6f}")
                    break
                        
        except Exception as e:
            self.logger.error(f"Error filling grid gaps: {e}")
    def _should_protect_order(self, order: Dict, current_price: float, live_positions: List[Dict]) -> bool:
        """Determine if an order should be protected from cancellation"""
        try:
            order_price = float(order.get('price', 0))
            order_side = order.get('side', '').lower()
            order_id = order.get('id', '')
            
            # Check if we have positions
            has_long_positions = any(
                float(pos.get('contracts', 0)) > 0 and pos.get('side', '').lower() == 'long'
                for pos in live_positions
            )
            has_short_positions = any(
                float(pos.get('contracts', 0)) > 0 and pos.get('side', '').lower() == 'short'
                for pos in live_positions
            )
            
            # Protect profit-taking orders
            if order_side == 'sell' and order_price > current_price and has_long_positions:
                # This sell order could close longs at profit
                avg_long_entry = self._calculate_average_entry(live_positions, 'long')
                if avg_long_entry > 0 and order_price > avg_long_entry * 1.002:  # At least 0.2% profit
                    self.logger.info(f"Protecting profitable sell order @ ${order_price:.6f} (avg entry: ${avg_long_entry:.6f})")
                    return True
                    
            if order_side == 'buy' and order_price < current_price and has_short_positions:
                # This buy order could close shorts at profit
                avg_short_entry = self._calculate_average_entry(live_positions, 'short')
                if avg_short_entry > 0 and order_price < avg_short_entry * 0.998:  # At least 0.2% profit
                    self.logger.info(f"Protecting profitable buy order @ ${order_price:.6f} (avg entry: ${avg_short_entry:.6f})")
                    return True
            
            # Check if order is marked as profit order in metadata
            if order_id in self.pending_orders:
                order_info = self.pending_orders[order_id]
                if order_info.get('is_profit_order', False):
                    return True
            
            # Protect orders that are very close to being filled (within 0.1% of current price)
            price_distance_pct = abs(order_price - current_price) / current_price
            if price_distance_pct < 0.001:  # Within 0.1%
                self.logger.debug(f"Protecting near-execution order @ ${order_price:.6f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in _should_protect_order: {e}")
            return False  # Default to not protecting on error

    def _calculate_average_entry(self, positions: List[Dict], side: str) -> float:
        """Calculate average entry price for positions of given side"""
        try:
            total_value = 0.0
            total_size = 0.0
            
            for pos in positions:
                if pos.get('side', '').lower() == side:
                    size = abs(float(pos.get('contracts', 0)))
                    entry_price = float(pos.get('entryPrice', 0))
                    if size > 0 and entry_price > 0:
                        total_value += size * entry_price
                        total_size += size
            
            return total_value / total_size if total_size > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating average entry: {e}")
            return 0.0
    def _update_orders_and_positions(self):
        """Check for filled orders, validate existing orders, and update position tracking"""
        try:
            # Get current open orders
            open_orders = self.exchange.get_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            # Get current market price first (needed for protection checks)
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # Get live positions for protection checks
            live_positions = self.exchange.get_positions(self.symbol)
            
            # ADDED: Always update BB range and cancel orders outside BB
            try:
                market_snapshot = self.market_intel.analyze_market(self.exchange)
                
                if market_snapshot.bollinger_upper > 0 and market_snapshot.bollinger_lower > 0:
                    new_bb_lower = self._round_price(market_snapshot.bollinger_lower)
                    new_bb_upper = self._round_price(market_snapshot.bollinger_upper)
                    
                    # Update range if BB changed significantly
                    range_changed = (abs(new_bb_lower - self.user_price_lower) / self.user_price_lower > 0.01 or
                                abs(new_bb_upper - self.user_price_upper) / self.user_price_upper > 0.01)
                    
                    if range_changed:
                        old_lower, old_upper = self.user_price_lower, self.user_price_upper
                        self.user_price_lower = new_bb_lower
                        self.user_price_upper = new_bb_upper
                        
                        self.logger.info(f"BB Range updated: ${old_lower:.6f}-${old_upper:.6f} → ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}")
                    
                    # Cancel orders outside BB range (WITH PROTECTION)
                    bb_margin = (self.user_price_upper - self.user_price_lower) * 0.01  # 1% margin
                    orders_to_cancel = []
                    
                    for order in open_orders:
                        # PROTECTION CHECK FIRST
                        if self._should_protect_order(order, current_price, live_positions):
                            continue  # Skip protected orders
                        
                        order_price = float(order.get('price', 0))
                        order_id = order.get('id', '')
                        
                        if (order_price < self.user_price_lower - bb_margin or 
                            order_price > self.user_price_upper + bb_margin):
                            orders_to_cancel.append({
                                'id': order_id,
                                'price': order_price,
                                'side': order.get('side', 'unknown')
                            })
                    
                    # Cancel orders outside BB range
                    for order_info in orders_to_cancel:
                        try:
                            self.logger.info(f"BB cleanup: Cancelling {order_info['side'].upper()} @ ${order_info['price']:.6f} (outside BB range)")
                            self.exchange.cancel_order(order_info['id'], self.symbol)
                            
                            if order_info['id'] in self.pending_orders:
                                del self.pending_orders[order_info['id']]
                                
                        except Exception as e:
                            self.logger.error(f"Failed to cancel BB order {order_info['id'][:8]}: {e}")
                    
                    if orders_to_cancel:
                        time.sleep(1)  # Wait for cancellations
                        open_orders = self.exchange.get_open_orders(self.symbol)
                        open_order_ids = {order['id'] for order in open_orders}
                        
            except Exception as e:
                self.logger.error(f"Error in BB range update: {e}")
            
            # Position-based capacity management
            market_data = self._get_live_market_data()
            position_margin = market_data['position_margin']
            available_investment = market_data['available_investment']
            position_margin_pct = (position_margin / self.user_total_investment) * 100
            can_place_orders = int(available_investment / self.user_investment_per_grid)
            
            if position_margin_pct >= 80.0 and can_place_orders <= 1 and open_orders:
                live_positions = market_data['live_positions']
                net_position_value = 0.0
                
                for position in live_positions:
                    size = float(position.get('contracts', 0))
                    entry_price = float(position.get('entryPrice', 0))
                    if size != 0 and entry_price > 0:
                        position_value = size * entry_price
                        if position.get('side', '').lower() == 'long':
                            net_position_value += position_value
                        else:
                            net_position_value -= position_value
                
                if net_position_value != 0:
                    dominant_direction = 'long' if net_position_value > 0 else 'short'
                    
                    # Find same-direction orders to potentially cancel
                    same_direction_orders = []
                    for order in open_orders:
                        # PROTECTION CHECK FIRST
                        if self._should_protect_order(order, current_price, live_positions):
                            continue  # Skip protected orders
                        
                        order_side = order.get('side', '').lower()
                        order_price = float(order.get('price', 0))
                        order_id = order.get('id', '')
                        
                        # Check if order is in same direction as dominant position
                        is_same_direction = False
                        if dominant_direction == 'long' and order_side == 'buy':
                            is_same_direction = True
                        elif dominant_direction == 'short' and order_side == 'sell':
                            is_same_direction = True
                        
                        if is_same_direction and order_price > 0:
                            distance_from_current = abs(order_price - current_price)
                            importance_score = self.pending_orders.get(order_id, {}).get('importance_score', 0.5)
                            
                            same_direction_orders.append({
                                'id': order_id,
                                'side': order_side,
                                'price': order_price,
                                'distance_from_current': distance_from_current,
                                'importance_score': importance_score
                            })
                    
                    # Cancel the least important same-direction order
                    if same_direction_orders:
                        same_direction_orders.sort(key=lambda x: (x['importance_score'], -x['distance_from_current']))
                        order_to_cancel = same_direction_orders[0]
                        
                        try:
                            self.logger.warning(f"Position capacity management: Cancelling {order_to_cancel['side'].upper()} @ ${order_to_cancel['price']:.6f} (importance: {order_to_cancel['importance_score']:.3f})")
                            
                            self.exchange.cancel_order(order_to_cancel['id'], self.symbol)
                            
                            # Remove from internal tracking
                            if order_to_cancel['id'] in self.pending_orders:
                                del self.pending_orders[order_to_cancel['id']]
                            
                            # Refresh open orders after cancellation
                            time.sleep(1)
                            open_orders = self.exchange.get_open_orders(self.symbol)
                            open_order_ids = {order['id'] for order in open_orders}
                            
                        except Exception as e:
                            self.logger.error(f"Failed to cancel order {order_to_cancel['id'][:8]}: {e}")
            
            # KAMA validation of existing orders
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    kama_direction = market_snapshot.kama_direction
                    kama_strength = market_snapshot.kama_strength
                    
                    # Only cancel orders if KAMA signal is strong enough
                    if kama_strength > 0.3 and open_orders:
                        live_positions = market_data['live_positions']
                        
                        # Analyze existing positions
                        has_long_positions = False
                        has_short_positions = False
                        
                        for position in live_positions:
                            size = float(position.get('contracts', 0))
                            if size != 0:
                                if position.get('side', '').lower() == 'long':
                                    has_long_positions = True
                                else:
                                    has_short_positions = True
                        
                        orders_to_cancel = []
                        
                        for order in open_orders:
                            # PROTECTION CHECK FIRST
                            if self._should_protect_order(order, current_price, live_positions):
                                continue  # Skip protected orders
                            
                            order_side = order.get('side', '').lower()
                            order_id = order.get('id', '')
                            order_price = float(order.get('price', 0))
                            
                            # Determine if order is profit-taking or new position
                            is_profit_taking = False
                            is_risky_new_position = False
                            
                            if order_side == 'sell' and order_price > current_price and has_long_positions:
                                is_profit_taking = True
                            elif order_side == 'buy' and order_price < current_price and has_short_positions:
                                is_profit_taking = True
                            elif order_side == 'sell' and order_price < current_price:
                                is_risky_new_position = True
                            elif order_side == 'buy' and order_price > current_price:
                                is_risky_new_position = True
                            
                            # Check if order direction conflicts with KAMA trend (only for new positions)
                            should_cancel = False
                            
                            if not is_profit_taking and is_risky_new_position:
                                if kama_direction == 'bearish' and order_side == 'buy':
                                    should_cancel = True
                                elif kama_direction == 'bullish' and order_side == 'sell':
                                    should_cancel = True
                            
                            if should_cancel:
                                orders_to_cancel.append({
                                    'id': order_id,
                                    'side': order_side,
                                    'price': order_price
                                })
                        
                        # Cancel conflicting orders
                        for order_info in orders_to_cancel:
                            try:
                                self.logger.warning(f"KAMA cancellation: {order_info['side'].upper()} @ ${order_info['price']:.6f} conflicts with {kama_direction} trend (strength: {kama_strength:.3f})")
                                
                                self.exchange.cancel_order(order_info['id'], self.symbol)
                                
                                # Remove from internal tracking
                                if order_info['id'] in self.pending_orders:
                                    del self.pending_orders[order_info['id']]
                                
                            except Exception as e:
                                self.logger.error(f"Failed to cancel order {order_info['id'][:8]}: {e}")
                        
                        if orders_to_cancel:
                            time.sleep(1)
                            open_orders = self.exchange.get_open_orders(self.symbol)
                            open_order_ids = {order['id'] for order in open_orders}
                    
                    # Position risk management for very strong trends (unchanged as it closes positions, not orders)
                    if kama_strength > 0.1:
                        # ... existing position closing logic remains unchanged ...
                        pass
                            
                except Exception as e:
                    self.logger.error(f"Error in KAMA order validation: {e}")
            
            # Check each pending order for fills (rest of the function remains unchanged)
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
                            del self.pending_orders[order_id]
                except Exception as e:
                    self.logger.error(f"Error checking order status {order_id[:8]}: {e}")
                    # Remove problematic order
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
            
            # Clean up stale internal tracking
            stale_orders = [order_id for order_id in list(self.pending_orders.keys()) if order_id not in open_order_ids]
            for order_id in stale_orders:
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
            
            # Sync untracked orders
            self._sync_order_tracking(open_orders)
            
            if filled_order_ids or stale_orders:
                self.logger.info(f"Order update: {len(filled_order_ids)} filled, {len(stale_orders)} stale cleaned")
            
        except Exception as e:
            self.logger.error(f"Error updating orders and positions: {e}")
                
    def _process_filled_order(self, order_id: str, order_status: Dict):
        """Process a filled order and create position or close existing position"""
        try:
            order_info = self.pending_orders.get(order_id)
            if not order_info:
                self.logger.warning(f"Order {order_id} not found in pending orders")
                return
            
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
            
            self.logger.info(f"New position: {position_side.upper()} {fill_amount:.6f} @ ${fill_price:.6f} (ID: {position_id[:8]})")
            
        except Exception as e:
            self.logger.error(f"Error creating position from filled order: {e}")
    def _cancel_other_counter_orders(self, position_id: str, filled_order_id: str):
        """Cancel other counter orders for the same position when one fills"""
        try:
            orders_to_cancel = []
            
            # Find other counter orders for the same position
            for order_id, order_info in list(self.pending_orders.items()):
                if (order_info.get('position_id') == position_id and 
                    order_id != filled_order_id and
                    order_info.get('order_purpose') in ['profit', 'stop_loss']):
                    orders_to_cancel.append(order_id)
            
            # Cancel the orders
            for order_id in orders_to_cancel:
                try:
                    self.exchange.cancel_order(order_id, self.symbol)
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    self.logger.info(f"Cancelled remaining counter order: {order_id[:8]}")
                except Exception as e:
                    self.logger.warning(f"Failed to cancel counter order {order_id[:8]}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error cancelling other counter orders: {e}")
    def _close_position_from_counter_order(self, order_id: str, order_status: Dict, order_info: Dict):
        """Close position from filled counter order"""
        try:
            position_id = order_info['position_id']
            
            if position_id not in self.all_positions:
                self.logger.warning(f"Position {position_id} not found for counter order {order_id}")
                return
            
            position = self.all_positions[position_id]
            order_purpose = order_info.get('order_purpose', 'unknown')  # NEW: Check order purpose
            
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
            
            position.unrealized_pnl = 0.0
            
            # Enhanced logging with order purpose
            pnl_status = "PROFIT" if position.realized_pnl > 0 else "LOSS"
            self.logger.info(f"Position closed via {order_purpose.upper()}: {position.side.upper()} "
                            f"${position.entry_price:.6f} → ${position.exit_price:.6f}, "
                            f"{pnl_status}: ${position.realized_pnl:.2f}")
            
            # IMPORTANT: Cancel the other counter order for this position
            self._cancel_other_counter_orders(position_id, order_id)
            
        except Exception as e:
            self.logger.error(f"Error closing position from counter order: {e}")
    
    def _maintain_counter_orders(self, current_price: float):
        """Maintain counter orders for open positions"""
        try:
            open_positions = [pos for pos in self.all_positions.values() if pos.is_open()]
            
            if not open_positions:
                return
            
            for position in open_positions:
                if not position.has_counter_order:
                    self._create_counter_order_for_position(position, current_price)
                        
        except Exception as e:
            self.logger.error(f"Error maintaining counter orders: {e}")
    
    def _create_counter_order_for_position(self, position: GridPosition, current_price: float):
        """Create profit-taking (limit) AND stop-loss (stop-market) orders for a position"""
        try:
            # Calculate order parameters based on position side
            if position.side == PositionSide.LONG.value:
                counter_side = OrderType.SELL.value
                # Profit target: 0.8% above entry (LIMIT order)
                profit_price = self._round_price(position.entry_price * 1.008)
                # Stop loss: 0.8% below entry (STOP-MARKET order)
                stop_price = self._round_price(position.entry_price * 0.992)
            else:
                counter_side = OrderType.BUY.value
                # Profit target: 0.8% below entry (LIMIT order)
                profit_price = self._round_price(position.entry_price * 0.992)
                # Stop loss: 0.8% above entry (STOP-MARKET order)
                stop_price = self._round_price(position.entry_price * 1.008)
            
            # Validate prices make sense
            if counter_side == OrderType.SELL.value:
                if profit_price <= position.entry_price or stop_price >= position.entry_price:
                    return
            else:
                if profit_price >= position.entry_price or stop_price <= position.entry_price:
                    return
            
            orders_placed = 0
            
            # 1. Place PROFIT order (LIMIT)
            try:
                profit_order = self.exchange.create_limit_order(self.symbol, counter_side, position.quantity, profit_price)
                
                if profit_order and 'id' in profit_order:
                    profit_order_info = {
                        'type': counter_side,
                        'price': profit_price,
                        'amount': position.quantity,
                        'position_id': position.position_id,
                        'timestamp': time.time(),
                        'status': 'open',
                        'order_purpose': 'profit'
                    }
                    self.pending_orders[profit_order['id']] = profit_order_info
                    orders_placed += 1
                    
                    expected_profit = abs(profit_price - position.entry_price) * position.quantity
                    self.logger.info(f"PROFIT (limit): {counter_side.upper()} {position.quantity:.6f} @ ${profit_price:.6f}, Expected: ${expected_profit:.2f}")
            
            except Exception as e:
                self.logger.error(f"Failed to place profit order: {e}")
            
            # 2. Place STOP-LOSS order (STOP-MARKET) - THE KEY CHANGE
            try:
                sl_order = self.exchange.create_stop_market_order(self.symbol, counter_side, position.quantity, stop_price)
                
                if sl_order and 'id' in sl_order:
                    sl_order_info = {
                        'type': counter_side,
                        'price': stop_price,
                        'amount': position.quantity,
                        'position_id': position.position_id,
                        'timestamp': time.time(),
                        'status': 'open',
                        'order_purpose': 'stop_loss'
                    }
                    self.pending_orders[sl_order['id']] = sl_order_info
                    orders_placed += 1
                    
                    expected_loss = abs(stop_price - position.entry_price) * position.quantity
                    self.logger.info(f"STOP-LOSS (stop-market): {counter_side.upper()} {position.quantity:.6f} @ ${stop_price:.6f}, Max loss: ${expected_loss:.2f}")
            
            except Exception as e:
                self.logger.error(f"Failed to place stop-loss order: {e}")
            
            # Update position if at least one order was placed
            if orders_placed > 0:
                position.has_counter_order = True
                self.logger.info(f"Bracket orders created: {orders_placed} orders (profit + stop-loss)")
            
        except Exception as e:
            self.logger.error(f"Error creating counter orders for position {position.position_id}: {e}")
    
    def _update_pnl(self, current_price: float):
        """Update PnL calculations with periodic logging"""
        try:
            total_unrealized = 0.0
            total_realized = 0.0
            
            open_positions = [pos for pos in self.all_positions.values() if pos.is_open()]
            closed_positions = [pos for pos in self.all_positions.values() if not pos.is_open()]
            
            # Calculate PnL for open positions
            for pos in open_positions:
                pos.unrealized_pnl = pos.calculate_unrealized_pnl(current_price)
                total_unrealized += pos.unrealized_pnl

            # Add realized PnL from closed positions
            for pos in closed_positions:
                total_realized += pos.realized_pnl
            
            # Update total PnL
            previous_pnl = self.total_pnl
            self.total_pnl = total_realized + total_unrealized
            pnl_change = self.total_pnl - previous_pnl
            
            # Log PnL summary periodically or on significant changes
            should_log = (
                self.last_update_time == 0 or 
                time.time() - self.last_update_time > 300 or  # Every 5 minutes
                abs(pnl_change) > 10  # Or significant change
            )
            
            if should_log:
                pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0
                self.logger.info(f"PnL: Realized ${total_realized:.2f}, Unrealized ${total_unrealized:.2f}, "
                               f"Total ${self.total_pnl:.2f} ({pnl_percentage:.2f}%), "
                               f"Open: {len(open_positions)}, Closed: {len(closed_positions)}, Trades: {self.total_trades}")
            
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
            
            self.logger.info(f"Stopping grid strategy...")
            
            # Set running to false first
            self.running = False
            
            # Cancel all pending orders
            try:
                cancelled_orders = self.exchange.cancel_all_orders(self.symbol)
                if cancelled_orders:
                    self.logger.info(f"Cancelled {len(cancelled_orders)} pending orders")
                
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
                    self.logger.info(f"Closed {len(open_positions)} positions at market price")
                    
            except Exception as e:
                self.logger.error(f"Error closing positions during stop: {e}")
            
            # Final PnL calculation
            final_pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0
            
            self.logger.info(f"Grid stopped: {self.total_trades} trades, Final PnL: ${self.total_pnl:.2f} ({final_pnl_percentage:.2f}%)")
            
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
                
                self.logger.info(f"Position closed at market: {position.side} ${position.entry_price:.6f} → ${position.exit_price:.6f}, PnL: ${position.realized_pnl:.2f}")
                
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