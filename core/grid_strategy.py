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
                bb_data = ta.bbands(price_series, length=10, std=2.0, mamode='ema')

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
    
    # MINIMAL FIX: Replace _get_live_market_data() with proper None handling

    # MINIMAL FIX: Update _get_live_market_data() to detect TP/SL by order type

    def _get_live_market_data(self) -> Dict[str, Any]:
        """FIXED: Exclude TP/SL orders from margin calculation since they're reduceOnly"""
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
            
            # Calculate position margins
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

            # Calculate order margins (FIXED: Exclude TP/SL orders)
            order_margin = 0.0
            active_orders = 0
            tp_sl_orders = 0
            
            for order in live_orders:
                order_id = order.get('id', '')
                
                # Safe price extraction using helper function
                price = self._get_order_price(order)
                
                # Safe amount extraction with None check
                amount = order.get('amount')
                if amount is None:
                    amount = 0.0
                else:
                    try:
                        amount = float(amount)
                    except (ValueError, TypeError):
                        amount = 0.0
                
                # FIXED: Check if this is a TP/SL order by actual order type
                order_type = order.get('type', '').lower()
                is_tp_sl = order_type in [
                    'take_profit_market', 'stop_market', 
                    'take_profit', 'stop_loss',
                    'take_profit_limit', 'stop_loss_limit'
                ]
                
                if is_tp_sl:
                    tp_sl_orders += 1
                
                # CRITICAL FIX: Only calculate margin for NON-TP/SL orders
                # TP/SL orders are reduceOnly and don't require additional margin
                if price > 0 and amount > 0 and not is_tp_sl:
                    order_notional = price * amount
                    margin = round(order_notional / self.user_leverage, 2)
                    order_margin += margin
                    active_orders += 1
                
                # TP/SL orders don't consume margin but are counted separately
                elif is_tp_sl:
                    # Log TP/SL order for debugging but don't add to margin
                    self.logger.debug(f"TP/SL order (no margin): {order_type} {order.get('side', '').upper()} @ ${price:.6f}")

            # Calculate totals
            total_margin_used = round(position_margin + order_margin, 2)
            available_investment = round(self.user_total_investment - total_margin_used, 2)
            self.total_investment_used = total_margin_used
            
            # Create price coverage set
            covered_prices = set()
            for order in live_orders:
                price = self._get_order_price(order)
                if price > 0:
                    covered_prices.add(self._round_price(price))
            
            for position in live_positions:
                if float(position.get('contracts', 0)) != 0:
                    entry_price = float(position.get('entryPrice', 0))
                    if entry_price > 0:
                        covered_prices.add(self._round_price(entry_price))

            # Enhanced logging with TP/SL margin breakdown
            total_orders = active_orders + tp_sl_orders
            if total_orders > 0 or active_positions > 0 or len(stale_orders) > 0:
                self.logger.info(f"Market state: {active_orders} orders + {tp_sl_orders} TP/SL (${order_margin:.2f}), "
                            f"{active_positions} positions (${position_margin:.2f}), "
                            f"Total used: ${total_margin_used:.2f}, Available: ${available_investment:.2f}")
                
                # Additional debugging for margin calculation
                if tp_sl_orders > 0:
                    self.logger.info(f"TP/SL orders consume $0.00 margin (reduceOnly), Regular orders: ${order_margin:.2f}")

            return {
                'live_orders': live_orders,
                'live_positions': live_positions,
                'position_margin': position_margin,
                'order_margin': order_margin,  # Now excludes TP/SL orders
                'total_margin_used': total_margin_used,  # Now correct
                'available_investment': available_investment,  # Now correct
                'active_orders': active_orders,  # Excludes TP/SL orders
                'active_positions': active_positions,
                'total_commitment': active_orders + active_positions,  # Excludes TP/SL
                'covered_prices': covered_prices,
                'stale_orders_cleaned': len(stale_orders),
                'tp_sl_orders': tp_sl_orders
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live market data: {e}")
            return {
                'live_orders': [], 'live_positions': [], 'position_margin': 0.0, 'order_margin': 0.0,
                'total_margin_used': self.user_total_investment, 'available_investment': 0.0,
                'active_orders': 0, 'active_positions': 0, 'total_commitment': 0,
                'covered_prices': set(), 'stale_orders_cleaned': 0, 'tp_sl_orders': 0
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
        """Place initial grid orders with BB-based distribution strategy"""
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
            
            # BB-BASED MARKET ANALYSIS FOR ORDER DISTRIBUTION
            bb_position = 0.5  # Default to middle
            bb_width_pct = 0.04  # Default width
            is_low_volatility = False
            
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    bb_upper = market_snapshot.bollinger_upper
                    bb_lower = market_snapshot.bollinger_lower
                    bb_middle = market_snapshot.bollinger_middle
                    
                    if bb_upper > 0 and bb_lower > 0 and bb_upper > bb_lower and bb_middle > 0:
                        # Calculate WHERE current price is within BB range
                        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                        bb_position = max(0.0, min(1.0, bb_position))
                        
                        # Calculate BB width as volatility indicator
                        bb_width_pct = (bb_upper - bb_lower) / bb_middle
                        is_low_volatility = bb_width_pct < 0.03
                    
                except Exception as e:
                    self.logger.warning(f"BB analysis failed for order distribution: {e}")
            
            # BB-BASED ORDER DISTRIBUTION
            price_in_upper_zone = bb_position > 0.75
            price_in_lower_zone = bb_position < 0.25
            price_in_middle_zone = 0.35 < bb_position < 0.65
            
            if price_in_upper_zone:
                # Price expensive (near upper BB) - favor sell orders
                buy_ratio = 0.3  # 30% buy, 70% sell
                strategy_info = f"UPPER BB zone ({bb_position:.3f}) - favoring sell orders"
            elif price_in_lower_zone:
                # Price cheap (near lower BB) - favor buy orders
                buy_ratio = 0.7  # 70% buy, 30% sell
                strategy_info = f"LOWER BB zone ({bb_position:.3f}) - favoring buy orders"
            elif is_low_volatility:
                # Low volatility + middle zone - balanced but aggressive grid
                buy_ratio = 0.5  # 50% buy, 50% sell
                strategy_info = f"LOW volatility + MIDDLE zone ({bb_position:.3f}) - balanced grid"
            else:
                # Normal conditions - balanced approach
                buy_ratio = 0.5  # 50% buy, 50% sell
                strategy_info = f"NORMAL conditions ({bb_position:.3f}) - balanced grid"
            
            buy_target = int(max_orders * buy_ratio)
            sell_target = max_orders - buy_target
            
            # SMART ORDER SEQUENCING BASED ON BB POSITION
            if price_in_upper_zone:
                # Price expensive - prioritize sell orders, place them closer to current price
                sell_levels = [level for level in grid_levels if level > current_price]
                buy_levels = [level for level in grid_levels if level < current_price]
                
                # Sort sell levels by proximity (closest first for better fills)
                sell_levels.sort(key=lambda x: x - current_price)
                # Sort buy levels by distance (farthest first to avoid immediate fills)
                buy_levels.sort(key=lambda x: current_price - x, reverse=True)
                
                # Prioritize sell orders
                sorted_levels_with_side = [(level, 'sell') for level in sell_levels[:sell_target]] + \
                                        [(level, 'buy') for level in buy_levels[:buy_target]]
                                        
            elif price_in_lower_zone:
                # Price cheap - prioritize buy orders, place them closer to current price
                buy_levels = [level for level in grid_levels if level < current_price]
                sell_levels = [level for level in grid_levels if level > current_price]
                
                # Sort buy levels by proximity (closest first for better fills)
                buy_levels.sort(key=lambda x: current_price - x)
                # Sort sell levels by distance (farthest first to avoid immediate fills)
                sell_levels.sort(key=lambda x: x - current_price, reverse=True)
                
                # Prioritize buy orders
                sorted_levels_with_side = [(level, 'buy') for level in buy_levels[:buy_target]] + \
                                        [(level, 'sell') for level in sell_levels[:sell_target]]
            else:
                # Middle zone or normal conditions - balanced placement
                sorted_levels = sorted(grid_levels, key=lambda x: abs(x - current_price))
                
                # Alternate between buy and sell orders for balanced placement
                sorted_levels_with_side = []
                buy_count = 0
                sell_count = 0
                
                for level_price in sorted_levels:
                    if buy_count < buy_target and level_price < current_price:
                        sorted_levels_with_side.append((level_price, 'buy'))
                        buy_count += 1
                    elif sell_count < sell_target and level_price > current_price:
                        sorted_levels_with_side.append((level_price, 'sell'))
                        sell_count += 1
                    
                    if buy_count >= buy_target and sell_count >= sell_target:
                        break
            
            self.logger.info(f"BB Strategy: {strategy_info}")
            self.logger.info(f"Order targets: {buy_target} buy + {sell_target} sell = {max_orders} total")
            self.logger.info(f"BB width: {bb_width_pct:.3f} ({'tight' if is_low_volatility else 'normal/wide'} bands)")
            
            # Place orders according to the BB-based sequence
            orders_placed = 0
            buy_orders_placed = 0
            sell_orders_placed = 0
            
            for level_price, order_side in sorted_levels_with_side:
                if orders_placed >= max_orders:
                    break
                
                # Place the order (BB logic will filter inappropriate orders)
                if self._place_single_order(level_price, order_side):
                    orders_placed += 1
                    if order_side == 'buy':
                        buy_orders_placed += 1
                    else:
                        sell_orders_placed += 1
            
            self.logger.info(f"Initial BB-based grid orders placed: {buy_orders_placed} buy, {sell_orders_placed} sell, Total: {orders_placed}")
            
            return orders_placed
            
        except Exception as e:
            self.logger.error(f"Error placing initial grid orders: {e}")
            return 0
    
    def _place_single_order(self, price: float, side: str) -> bool:
        """Enhanced order placement with BB-based value assessment logic"""
        try:
            # Get current market price for context
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            price_diff_pct = ((price - current_price) / current_price) * 100
            
            # Get current market data for fund allocation check
            market_data = self._get_live_market_data()
            available_investment = market_data['available_investment']
            
            # Calculate order parameters
            amount = self._calculate_order_amount(price)
            notional_value = price * amount
            margin_required = round(notional_value / self.user_leverage, 2)
            
            # FUND ALLOCATION CHECK
            if available_investment < margin_required:
                self.logger.info(f"Insufficient funds: Need ${margin_required:.2f}, Available: ${available_investment:.2f}")
                return False
            
            # BB-BASED MARKET ANALYSIS WITH MOMENTUM
            bb_position = 0.5  # Default to middle
            bb_width_pct = 0.04  # Default width
            is_low_volatility = False
            is_high_volatility = False
            momentum = 0.0  # Add momentum tracking
            momentum_direction = 'neutral'  # Add momentum direction
            
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    bb_upper = market_snapshot.bollinger_upper
                    bb_lower = market_snapshot.bollinger_lower
                    bb_middle = market_snapshot.bollinger_middle
                    momentum = market_snapshot.momentum  # Get momentum value
                    momentum_direction = market_snapshot.kama_direction  # Get momentum direction
                    
                    if bb_upper > 0 and bb_lower > 0 and bb_upper > bb_lower:
                        # Calculate WHERE current price is within BB range (0.0 to 1.0)
                        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                        bb_position = max(0.0, min(1.0, bb_position))  # Clamp to 0-1 range
                        
                        # Calculate BB width as volatility indicator
                        bb_width_pct = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.04
                        
                        # Volatility regime detection
                        is_low_volatility = bb_width_pct < 0.03  # Tight bands = sideways market
                        is_high_volatility = bb_width_pct > 0.06  # Wide bands = trending market
                        
                        self.logger.debug(f"BB Analysis: Position {bb_position:.3f} (0=lower, 1=upper), "
                                        f"Width {bb_width_pct:.3f}, Volatility: {'HIGH' if is_high_volatility else 'LOW' if is_low_volatility else 'NORMAL'}, "
                                        f"Momentum: {momentum_direction.upper()}")
                    
                except Exception as e:
                    self.logger.warning(f"BB analysis failed, using defaults: {e}")
            
            # FIXED: REASONABLE GAP CALCULATION ALIGNED WITH GRID STRATEGY
            # Grid distributes orders within 2% of current price, so min_gap should be much smaller
            base_gap_pct = 0.002  # 0.2% base minimum gap
            
            if is_high_volatility:
                # Slightly larger gaps in high volatility but not excessive
                min_gap_pct = min(0.008, bb_width_pct * 0.2)  # Max 0.8% or 20% of BB width
            elif is_low_volatility:
                # Very small gaps in low volatility for tight grid
                min_gap_pct = max(0.001, bb_width_pct * 0.1)  # Min 0.1% or 10% of BB width  
            else:
                # Normal volatility - moderate gap
                min_gap_pct = 0.003  # 0.3% gap
            
            min_gap = current_price * min_gap_pct
            
            # Check minimum gap requirement with debug logging
            distance_from_current = abs(price - current_price)
            if distance_from_current < min_gap:
                gap_pct = (distance_from_current / current_price) * 100
                min_gap_pct_actual = (min_gap / current_price) * 100
                self.logger.info(f"Order rejected - too close: {gap_pct:.3f}% < {min_gap_pct_actual:.3f}% required "
                            f"(${distance_from_current:.6f} < ${min_gap:.6f}), BB width: {bb_width_pct*100:.2f}%")
                return False
            
            # ENHANCED BB-BASED VALUE ASSESSMENT WITH MOMENTUM INTEGRATION
            price_in_upper_zone = bb_position > 0.75  # Upper 25% of BB range
            price_in_lower_zone = bb_position < 0.25  # Lower 25% of BB range
            price_in_middle_zone = 0.35 < bb_position < 0.65  # Middle zone
            
            # Momentum strength thresholds
            strong_bearish = momentum < -0.02 or momentum_direction == 'bearish'
            strong_bullish = momentum > 0.02 or momentum_direction == 'bullish'
            
            should_block = False
            block_reason = ""
            
            if price_in_upper_zone:
                # Price near upper BB = expensive, likely to reverse down
                if side == 'buy':
                    should_block = True
                    block_reason = f"Price in upper BB zone ({bb_position:.3f}) - too expensive to buy"
                    
            elif price_in_lower_zone:
                # ENHANCED: Consider momentum in lower zone
                if side == 'sell':
                    should_block = True
                    block_reason = f"Price in lower BB zone ({bb_position:.3f}) - too cheap to sell"
                elif side == 'buy' and strong_bearish:
                    # NEW: Block buy orders in lower zone during strong bearish momentum
                    should_block = True
                    block_reason = f"Price in lower BB zone ({bb_position:.3f}) with strong bearish momentum ({momentum_direction}) - avoid buying into falling knife"
            
            # Additional protection in extreme zones with momentum consideration
            if bb_position > 0.90:
                if side == 'buy':
                    should_block = True
                    block_reason = f"Price at extreme upper BB ({bb_position:.3f}) - very expensive"
                elif side == 'sell' and strong_bullish:
                    # Don't sell in extreme upper zone during strong bullish momentum
                    should_block = True  
                    block_reason = f"Price at extreme upper BB ({bb_position:.3f}) with bullish momentum - avoid selling too early"
                    
            elif bb_position < 0.10:
                if side == 'sell':
                    should_block = True
                    block_reason = f"Price at extreme lower BB ({bb_position:.3f}) - very cheap"
                elif side == 'buy' and strong_bearish:
                    # Don't buy in extreme lower zone during strong bearish momentum
                    should_block = True
                    block_reason = f"Price at extreme lower BB ({bb_position:.3f}) with bearish momentum - avoid catching falling knife"
            
            if should_block:
                self.logger.info(f"BB+Momentum Block: {block_reason}")
                return False
            
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
            
            # Store order information
            distance_from_current = abs(price - current_price)
            importance_score = 1.0 / (1.0 + distance_from_current)
            
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
                'bb_position_at_placement': bb_position,
                'bb_width_at_placement': bb_width_pct,
                'momentum_at_placement': momentum,
                'momentum_direction_at_placement': momentum_direction
            }
            
            self.pending_orders[order['id']] = order_info
            
            # Enhanced logging with BB and momentum context
            volatility_info = "HIGH-VOL" if is_high_volatility else "LOW-VOL" if is_low_volatility else "NORMAL-VOL"
            bb_zone = "UPPER" if price_in_upper_zone else "LOWER" if price_in_lower_zone else "MIDDLE"
            momentum_info = momentum_direction.upper()
            
            self.logger.info(f"Order placed ({volatility_info}, {bb_zone}, {momentum_info}): {side.upper()} {amount:.6f} @ ${price:.6f} "
                            f"({price_diff_pct:+.2f}%), BB pos: {bb_position:.3f}, Momentum: {momentum:.3f}, Margin: ${margin_required:.2f}, ID: {order['id'][:8]}")
            
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
    def _cancel_orphaned_tp_sl_orders(self):
        """ADDED: Clean up TP/SL orders that have no corresponding positions"""
        try:
            exchange_positions = self.exchange.get_positions(self.symbol)
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            # Create set of existing position keys
            existing_position_keys = set()
            for pos in exchange_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                
                if abs(size) >= 0.001 and entry_price > 0:
                    position_key = f"{side}_{entry_price:.6f}"
                    existing_position_keys.add(position_key)
            
            # Find orphaned TP/SL orders
            orphaned_orders = []
            for order in live_orders:
                order_id = order.get('id', '')
                if order_id in self.pending_orders:
                    order_info = self.pending_orders[order_id]
                    if order_info.get('is_tp_sl_order', False):
                        position_key = order_info.get('position_key', '')
                        if position_key and position_key not in existing_position_keys:
                            orphaned_orders.append({
                                'id': order_id,
                                'price': float(order.get('price', 0)),
                                'side': order.get('side', ''),
                                'position_key': position_key
                            })
            
            # Cancel orphaned orders
            for order_info in orphaned_orders:
                try:
                    self.logger.warning(f"Cancelling orphaned TP/SL: {order_info['side'].upper()} @ ${order_info['price']:.6f} (no position: {order_info['position_key']})")
                    self.exchange.cancel_order(order_info['id'], self.symbol)
                    
                    if order_info['id'] in self.pending_orders:
                        del self.pending_orders[order_info['id']]
                        
                except Exception as e:
                    self.logger.error(f"Failed to cancel orphaned TP/SL {order_info['id'][:8]}: {e}")
            
            if orphaned_orders:
                self.logger.info(f"Cleaned up {len(orphaned_orders)} orphaned TP/SL orders")
                
        except Exception as e:
            self.logger.error(f"Error cleaning orphaned TP/SL orders: {e}")
    def _should_protect_order(self, order: Dict, current_price: float, live_positions: List[Dict]) -> bool:
        """MINIMAL FIX: Only protect orders very close to execution"""
        try:
            order_price = float(order.get('price', 0))
            
            # Only protect orders very close to current price (likely about to execute)
            price_distance_pct = abs(order_price - current_price) / current_price
            if price_distance_pct < 0.005:  # Within 0.5%
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Protection check error: {e}")
            return False

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
        
    def _get_order_price(self, order: Dict) -> float:
        """ADDED: Helper to safely get order price from various fields"""
        # Try different price fields in order of preference
        price_fields = ['price', 'triggerPrice', 'stopPrice', 'takeProfitPrice', 'stopLossPrice']
        
        for field in price_fields:
            value = order.get(field)
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
        
        return 0.0  # Default if no valid price found
    # MINIMAL FIX: Update _update_orders_and_positions() to protect TP/SL orders in BB cleanup

    def _update_orders_and_positions(self):
        """FIXED: Protect TP/SL orders in BB cleanup and use live data only"""
        try:
            # Get LIVE data as primary source of truth
            live_orders = self.exchange.get_open_orders(self.symbol)
            live_positions = self.exchange.get_positions(self.symbol)
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # Clean local tracking to match live data
            live_order_ids = {order['id'] for order in live_orders}
            stale_local_orders = [oid for oid in list(self.pending_orders.keys()) if oid not in live_order_ids]
            
            # Process filled orders before cleaning
            filled_orders = []
            for order_id in stale_local_orders:
                try:
                    order_status = self.exchange.get_order_status(order_id, self.symbol)
                    if order_status['status'] in ['filled', 'closed']:
                        filled_orders.append((order_id, order_status))
                except:
                    pass
            
            # Process fills and clean stale tracking
            for order_id, order_status in filled_orders:
                self._process_filled_order(order_id, order_status)
            
            for order_id in stale_local_orders:
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
            
            # Add minimal tracking for untracked live orders
            for order in live_orders:
                order_id = order['id']
                if order_id not in self.pending_orders:
                    order_price = self._get_order_price(order)
                    
                    self.pending_orders[order_id] = {
                        'type': order.get('side', ''),
                        'price': order_price,
                        'amount': float(order.get('amount', 0)),
                        'timestamp': time.time(),
                        'status': 'open'
                    }
            
            # FIXED: BB cleanup with proper TP/SL protection by order type
            try:
                if self.market_intel:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    if market_snapshot.bollinger_upper > 0 and market_snapshot.bollinger_lower > 0:
                        new_bb_lower = self._round_price(market_snapshot.bollinger_lower)
                        new_bb_upper = self._round_price(market_snapshot.bollinger_upper)
                        
                        # Update range if changed significantly
                        if (abs(new_bb_lower - self.user_price_lower) / self.user_price_lower > 0.01 or
                            abs(new_bb_upper - self.user_price_upper) / self.user_price_upper > 0.01):
                            self.user_price_lower = new_bb_lower
                            self.user_price_upper = new_bb_upper
                        
                        # Cancel orders outside BB range (WITH TP/SL PROTECTION)
                        bb_margin = (self.user_price_upper - self.user_price_lower) * 0.02
                        for order in live_orders:
                            order_price = self._get_order_price(order)
                            order_side = order.get('side', '')
                            order_id = order.get('id', '')
                            order_type = order.get('type', '').lower()
                            
                            # CRITICAL: Skip TP/SL orders (protect by type)
                            if order_type in ['take_profit_market', 'stop_market', 'take_profit', 'stop_loss', 'take_profit_limit', 'stop_loss_limit']:
                                continue  # Don't cancel TP/SL orders in BB cleanup
                            
                            # Skip if no valid price or too close to current price
                            if order_price <= 0:
                                continue
                                
                            price_distance = abs(order_price - current_price) / current_price
                            if price_distance < 0.01:  # Within 1% of current price
                                continue
                                
                            # Cancel if outside BB range (only non-TP/SL orders)
                            if (order_price < self.user_price_lower - bb_margin or 
                                order_price > self.user_price_upper + bb_margin):
                                try:
                                    self.logger.info(f"BB cleanup: {order_side.upper()} @ ${order_price:.6f} (type: {order_type})")
                                    self.exchange.cancel_order(order_id, self.symbol)
                                    if order_id in self.pending_orders:
                                        del self.pending_orders[order_id]
                                except Exception as e:
                                    self.logger.error(f"Cancel failed {order_id[:8]}: {e}")
            except Exception as e:
                self.logger.error(f"BB cleanup error: {e}")
            
            # FIXED: Simplified orphan cleanup - trust GTC orders to auto-cancel, only clean clearly invalid orders
            if not hasattr(self, '_last_cleanup') or time.time() - self._last_cleanup > 30:
                for order in live_orders:
                    order_price = self._get_order_price(order)
                    order_side = order.get('side', '')
                    order_id = order.get('id', '')
                    order_type = order.get('type', '').lower()
                    
                    # CRITICAL: Skip TP/SL orders completely - GTC orders auto-cancel when position closes
                    if order_type in ['take_profit_market', 'stop_market', 'take_profit', 'stop_loss', 'take_profit_limit', 'stop_loss_limit']:
                        continue  # Trust GTC orders to handle themselves
                    
                    # Skip if no valid price
                    if order_price <= 0:
                        continue
                    
                    # Only check grid orders that are VERY far from current price (potential orphans)
                    price_distance = abs(order_price - current_price) / current_price
                    if price_distance > 0.05:  # More than 5% away (very conservative threshold)
                        # Check if any live position justifies this order
                        has_matching_position = False
                        
                        for pos in live_positions:
                            pos_size = float(pos.get('contracts', 0))
                            pos_entry = float(pos.get('entryPrice', 0))
                            
                            if abs(pos_size) >= 0.001 and pos_entry > 0:
                                # For grid orders, check if it's within reasonable range of position
                                price_diff = abs(order_price - pos_entry) / pos_entry
                                if price_diff < 0.10:  # Within 10% of position entry (very generous)
                                    has_matching_position = True
                                    break
                        
                        # Only cancel if CLEARLY orphaned (very conservative approach)
                        if not has_matching_position:
                            try:
                                self.logger.warning(f"Cancelling clearly orphaned grid order: {order_side.upper()} @ ${order_price:.6f} (type: {order_type})")
                                self.exchange.cancel_order(order_id, self.symbol)
                                if order_id in self.pending_orders:
                                    del self.pending_orders[order_id]
                            except Exception as e:
                                self.logger.error(f"Cancel failed {order_id[:8]}: {e}")
                
                self._last_cleanup = time.time()
            
            # Update PnL from live positions only (don't store locally)
            total_unrealized = 0.0
            active_position_count = 0
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                
                if abs(size) >= 0.001 and entry_price > 0:
                    active_position_count += 1
                    if side == 'long':
                        unrealized = (current_price - entry_price) * abs(size)
                    else:
                        unrealized = (entry_price - current_price) * abs(size)
                    total_unrealized += unrealized
            
            # Calculate total PnL (don't store positions locally)
            total_realized = sum(pos.realized_pnl for pos in self.all_positions.values() if not pos.is_open())
            self.total_pnl = total_realized + total_unrealized
            
            # Check TP/SL
            if self.user_total_investment > 0:
                pnl_pct = (self.total_pnl / self.user_total_investment) * 100
                if pnl_pct >= self.take_profit_pnl or pnl_pct <= -self.stop_loss_pnl:
                    self.stop_grid()
            
            if filled_orders or stale_local_orders:
                self.logger.info(f"Update: {len(filled_orders)} filled, {len(stale_local_orders)} cleaned")
                
        except Exception as e:
            self.logger.error(f"Update error: {e}")
                    
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
        """STABLE: Simple TP/SL creation without order churning"""
        try:
            live_positions = self.exchange.get_positions(self.symbol)
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            # Count existing stop orders for safety
            existing_stop_count = len([o for o in live_orders 
                                    if o.get('type', '').lower() in ['stop_market', 'stop_loss', 'stop_loss_limit', 
                                                                    'take_profit_market', 'take_profit', 'take_profit_limit']])
            
            # Safety check - don't create if close to limits
            if existing_stop_count >= 8:
                return
            
            # Find active positions
            active_positions = []
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                
                if abs(size) >= 0.001 and entry_price > 0:
                    # Create simple position identifier
                    pos_id = f"{side}_{entry_price:.6f}"
                    active_positions.append({
                        'id': pos_id,
                        'entry_price': entry_price,
                        'side': side,
                        'quantity': abs(size)
                    })
            
            if not active_positions:
                return
            
            # Check which positions already have TP/SL orders in our tracking
            for pos in active_positions:
                pos_id = pos['id']
                entry_price = pos['entry_price']
                side = pos['side']
                quantity = pos['quantity']
                
                # Check if we already have TP/SL orders for this position in our tracking
                has_tp_in_tracking = False
                has_sl_in_tracking = False
                
                for order_id, order_info in self.pending_orders.items():
                    if (order_info.get('is_tp_sl_order') and 
                        order_info.get('position_id', '').startswith(pos_id)):
                        
                        order_purpose = order_info.get('order_purpose', '')
                        if order_purpose == 'profit':
                            has_tp_in_tracking = True
                        elif order_purpose == 'stop_loss':
                            has_sl_in_tracking = True
                
                # Only create TP/SL if we don't have them in our tracking
                if has_tp_in_tracking and has_sl_in_tracking:
                    continue
                
                # Calculate TP/SL prices
                if side == 'long':
                    tp_price = self._round_price(entry_price * 1.005)
                    sl_price = self._round_price(entry_price * 0.985)
                    counter_side = 'sell'
                else:
                    tp_price = self._round_price(entry_price * 0.995)
                    sl_price = self._round_price(entry_price * 1.015)
                    counter_side = 'buy'
                
                # Create missing orders
                orders_needed = []
                if not has_tp_in_tracking:
                    orders_needed.append(('TP', tp_price, 'TAKE_PROFIT_MARKET', 'profit'))
                if not has_sl_in_tracking:
                    orders_needed.append(('SL', sl_price, 'STOP_MARKET', 'stop_loss'))
                
                if not orders_needed:
                    continue
                    
                # Safety check before creating
                if existing_stop_count + len(orders_needed) > 9:
                    continue
                
                # Create the needed orders
                for order_type, price, binance_type, purpose in orders_needed:
                    try:
                        self.logger.info(f"Creating {order_type} for {pos_id}: {counter_side.upper()} {quantity:.6f} @ ${price:.6f}")
                        
                        symbol_id = self.exchange._get_symbol_id(self.symbol)
                        new_order = self.exchange.exchange.create_order(
                            symbol=symbol_id,
                            type=binance_type,
                            side=counter_side.upper(),
                            amount=quantity,
                            price=None,
                            params={
                                'stopPrice': price,
                                'timeInForce': 'GTE_GTC'
                            }
                        )
                        
                        if new_order and 'id' in new_order:
                            # Track with position linkage
                            self.pending_orders[new_order['id']] = {
                                'type': counter_side,
                                'price': price,
                                'amount': quantity,
                                'timestamp': time.time(),
                                'order_purpose': purpose,
                                'position_id': f"{pos_id}_{new_order['id'][:8]}",  # Unique position link
                                'is_tp_sl_order': True
                            }
                            
                            existing_stop_count += 1
                            self.logger.info(f"{order_type} created: {new_order['id'][:8]}")
                            time.sleep(0.2)
                            
                    except Exception as e:
                        if "-4045" in str(e):
                            self.logger.warning("Hit stop order limit, stopping TP/SL creation")
                            return
                        else:
                            self.logger.error(f"{order_type} creation failed: {e}")
            
            # Clean tracking of orders that no longer exist
            live_order_ids = {o['id'] for o in live_orders}
            stale_tp_sl = [oid for oid, info in self.pending_orders.items() 
                        if info.get('is_tp_sl_order') and oid not in live_order_ids]
            
            for order_id in stale_tp_sl:
                del self.pending_orders[order_id]
            
            if stale_tp_sl:
                self.logger.debug(f"Cleaned {len(stale_tp_sl)} stale TP/SL from tracking")
                
        except Exception as e:
            self.logger.error(f"Error in _maintain_counter_orders: {e}")
    def _create_counter_order_for_position(self, position: GridPosition, current_price: float):
        """Create profit-taking AND stop-loss orders with consistent 0.8% and GTC auto-cancel"""
        try:
            # Calculate order parameters based on position side (CONSISTENT 0.8%)
            if position.side == PositionSide.LONG.value:
                counter_side = OrderType.SELL.value
                # Profit target: 1% above entry
                profit_price = self._round_price(position.entry_price * 1.005)
                # Stop loss: 1% below entry
                stop_price = self._round_price(position.entry_price * 0.985)
            else:
                counter_side = OrderType.BUY.value
                # Profit target: 1% below entry
                profit_price = self._round_price(position.entry_price * 0.995)
                # Stop loss: 1% above entry
                stop_price = self._round_price(position.entry_price * 1.015)

            # Validate prices make sense
            if counter_side == OrderType.SELL.value:
                if profit_price <= position.entry_price or stop_price >= position.entry_price:
                    return
            else:
                if profit_price >= position.entry_price or stop_price <= position.entry_price:
                    return
            
            orders_placed = 0
            
            # 1. Place PROFIT order (TAKE_PROFIT_MARKET with GTC)
            try:
                self.logger.info(f"Creating TP order for position {position.position_id[:8]}")
                
                symbol_id = self.exchange._get_symbol_id(self.symbol)
                profit_order = self.exchange.exchange.create_order(
                    symbol=symbol_id,
                    type='TAKE_PROFIT_MARKET',
                    side=counter_side.upper(),
                    amount=position.quantity,
                    price=None,
                    params={
                        'stopPrice': profit_price, #'reduceOnly': True,
                        
                        'timeInForce': 'GTE_GTC'  # Auto-cancels when position closes
                    }
                )
                
                if profit_order and 'id' in profit_order:
                    profit_order_info = {
                        'type': counter_side,
                        'price': profit_price,
                        'amount': position.quantity,
                        'position_id': position.position_id,
                        'timestamp': time.time(),
                        'status': 'open',
                        'order_purpose': 'profit',
                        'is_tp_sl_order': True  # MARK as TP/SL order
                    }
                    self.pending_orders[profit_order['id']] = profit_order_info
                    orders_placed += 1
                    
                    expected_profit = abs(profit_price - position.entry_price) * position.quantity
                    self.logger.info(f"PROFIT (TP): {counter_side.upper()} {position.quantity:.6f} @ ${profit_price:.6f}, Expected: ${expected_profit:.2f}")
            
            except Exception as e:
                self.logger.error(f"Failed to place profit order: {e}")
            
            # 2. Place STOP-LOSS order (STOP_MARKET with GTC)
            try:
                symbol_id = self.exchange._get_symbol_id(self.symbol)
                sl_order = self.exchange.exchange.create_order(
                    symbol=symbol_id,
                    type='STOP_MARKET',
                    side=counter_side.upper(),
                    amount=position.quantity,
                    price=None,
                    params={
                        'stopPrice': stop_price, #'reduceOnly': True,                        
                        'timeInForce': 'GTE_GTC'  # Auto-cancels when position closes
                    }
                )
            
                if sl_order and 'id' in sl_order:
                    sl_order_info = {
                        'type': counter_side,
                        'price': stop_price,
                        'amount': position.quantity,
                        'position_id': position.position_id,
                        'timestamp': time.time(),
                        'status': 'open',
                        'order_purpose': 'stop_loss',
                        'is_tp_sl_order': True  # MARK as TP/SL order
                    }
                    self.pending_orders[sl_order['id']] = sl_order_info
                    orders_placed += 1
                    
                    expected_loss = abs(stop_price - position.entry_price) * position.quantity
                    self.logger.info(f"STOP-LOSS (SL): {counter_side.upper()} {position.quantity:.6f} @ ${stop_price:.6f}, Max loss: ${expected_loss:.2f}")
            
            except Exception as e:
                self.logger.error(f"Failed to place stop-loss order: {e}")
            
            # Update position if at least one order was placed
            if orders_placed > 0:
                position.has_counter_order = True
                self.logger.info(f"TP/SL orders created: {orders_placed} orders with GTC auto-cancel")
            
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