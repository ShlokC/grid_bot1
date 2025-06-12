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
from typing import Dict, List, Any, Optional, Tuple, Set
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
    # NEW: JMA trend detection fields
    jma_value: float = 0.0
    jma_trend: str = 'neutral'  # 'bullish', 'bearish', 'neutral'
    jma_strength: float = 0.0

    def is_trending(self, threshold: float = 0.5) -> bool:
        """Check if market is in trending state"""
        return self.trend_strength > threshold
    
    def is_bullish(self, threshold: float = 0.1) -> bool:
        """Check if market sentiment is bullish"""
        return self.directional_bias > threshold
    
    def is_bearish(self, threshold: float = 0.1) -> bool:
        """Check if market sentiment is bearish"""
        return self.directional_bias < -threshold
    
    # NEW: JMA trend methods
    def is_jma_bullish(self) -> bool:
        """Check if JMA indicates bullish trend"""
        return self.jma_trend == 'bullish'
    
    def is_jma_bearish(self) -> bool:
        """Check if JMA indicates bearish trend"""
        return self.jma_trend == 'bearish'


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
        """Analyze market using real KAMA, Bollinger Bands, and JMA"""
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
            
            # Calculate JMA trend
            jma_value, jma_trend, jma_strength = self._calculate_jma_trend(exchange, current_price)
            
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
                bollinger_middle=bollinger_middle,
                jma_value=jma_value,
                jma_trend=jma_trend,
                jma_strength=jma_strength
            )
            
        except Exception as e:
            self.logger.error(f"Market analysis error: {e}")
            return MarketSnapshot(
                timestamp=time.time(),
                price=0.0, volume=0.0, volatility=1.0, momentum=0.0, trend_strength=0.0,
                directional_bias=0.0, bollinger_upper=0.0, bollinger_lower=0.0, bollinger_middle=0.0,
                jma_value=0.0, jma_trend='neutral', jma_strength=0.0
            )
    def _calculate_jma_trend(self, exchange: Exchange, current_price: float) -> Tuple[float, str, float]:
        """SIMPLE: Just check if JMA is rising, falling, or flat - no complex math"""
        try:
            import pandas as pd
            import pandas_ta as ta
            import numpy as np
            
            # Get recent OHLCV data
            full_ohlcv = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=100)
            
            if not full_ohlcv or len(full_ohlcv) < 30:
                return current_price, 'neutral', 0.0
            
            # Extract closing prices and calculate JMA
            closing_prices = [float(candle[4]) for candle in full_ohlcv]
            price_series = pd.Series(closing_prices)
            jma_series = ta.jma(price_series, length=20, phase=2)
            
            if jma_series is None or jma_series.isna().all():
                return current_price, 'neutral', 0.0
            
            # Get last 5 JMA values
            recent_jma = jma_series.tail(5).dropna().values
            
            if len(recent_jma) < 5:
                return current_price, 'neutral', 0.0
            
            current_jma = float(recent_jma[-1])
            jma_4_ago = recent_jma[0]  # 4 periods ago
            
            # SIMPLE: Compare current JMA to 4 periods ago
            jma_change_pct = (current_jma - jma_4_ago) / jma_4_ago * 100
            
            # REALISTIC THRESHOLDS for trend detection
            if jma_change_pct > 0.1:    # Rising more than 0.1%
                trend = 'bullish'
                strength = min(0.95, abs(jma_change_pct) / 2.0)  # Normalize to 0-1
            elif jma_change_pct < -0.1:  # Falling more than 0.1%
                trend = 'bearish'  
                strength = min(0.95, abs(jma_change_pct) / 2.0)
            else:                       # Flat within ±0.1%
                trend = 'neutral'
                strength = 0.1
            
            # Enhanced logging with realistic values
            self.logger.info(f"${self.symbol} JMA: ${current_jma:.6f}, Price: ${current_price:.6f}, "
                            f"Change: {jma_change_pct:+.3f}%, Trend: {trend.upper()}, Strength: {strength:.3f}")
            
            return current_jma, trend, strength
            
        except Exception as e:
            self.logger.error(f"JMA calculation failed: {e}")
            return current_price, 'neutral', 0.0
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

                    self.logger.debug(f"BB: Upper ${bollinger_upper:.6f}, Middle ${bollinger_middle:.6f}, Lower ${bollinger_lower:.6f}")
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
            full_ohlcv = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=1400)

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

                self.logger.debug(f"KAMA: ${current_kama:.6f}, Direction: {self.current_kama_direction}, "
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
        self._jma_position_tracking = {}
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
            # ALL orders within 4% of current price (2% above, 2% below)
            max_distance_pct = 0.04  # 4% max distance from current price

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
    def _check_jma_trend_alignment(self, current_price: float):
        """FIXED: Smart JMA trend alignment with simplified position tracking"""
        try:
            # FIX #2: Enhanced SAMIG enablement check with logging
            if not self.enable_samig:
                self.logger.debug("JMA trend check skipped: SAMIG disabled")
                return
                
            if not self.market_intel:
                self.logger.warning("JMA trend check failed: MarketIntel not available")
                return
            
            # Get current market analysis including JMA
            try:
                market_snapshot = self.market_intel.analyze_market(self.exchange)
            except Exception as e:
                self.logger.error(f"JMA market analysis failed: {e}")
                return
            
            # Skip if JMA trend is neutral or invalid
            if market_snapshot.jma_trend == 'neutral' or market_snapshot.jma_value <= 0:
                self.logger.debug(f"JMA trend check skipped: trend={market_snapshot.jma_trend}, value=${market_snapshot.jma_value:.6f}")
                return
            
            # Get live positions
            try:
                live_positions = self.exchange.get_positions(self.symbol)
            except Exception as e:
                self.logger.error(f"Failed to get positions for JMA check: {e}")
                return
            
            # FIX #2: Enhanced logging for debugging
            self.logger.debug(f"JMA EXIT CHECK: SAMIG=True, Trend={market_snapshot.jma_trend.upper()}, "
                            f"JMA=${market_snapshot.jma_value:.6f}, Price=${current_price:.6f}")
            self.logger.debug(f"Found {len(live_positions)} live positions, tracking {len(self._jma_position_tracking)} position keys")
            
            current_time = time.time()
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                position_side = pos.get('positionSide', 'BOTH')
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                
                # FIX #3: Simplified position key generation (remove size, reduce precision)
                position_key = f"{side}_{entry_price:.4f}"
                
                # Check if position is opposite to JMA trend
                is_opposite_to_trend = False
                trend_info = ""
                
                if side == 'long' and market_snapshot.is_jma_bearish():
                    is_opposite_to_trend = True
                    trend_info = f"LONG vs JMA BEARISH"
                elif side == 'short' and market_snapshot.is_jma_bullish():
                    is_opposite_to_trend = True
                    trend_info = f"SHORT vs JMA BULLISH"
                
                if not is_opposite_to_trend:
                    # Position aligns with trend, remove from tracking if exists
                    if position_key in self._jma_position_tracking:
                        self.logger.debug(f"Position now aligns with JMA trend, removing from tracking: {position_key}")
                        del self._jma_position_tracking[position_key]
                    continue
                
                # Position is opposite to trend - start/update tracking
                if position_key not in self._jma_position_tracking:
                    # Start tracking this position
                    self._jma_position_tracking[position_key] = {
                        'first_detected': current_time,
                        'entry_price': entry_price,
                        'side': side,
                        'size': abs(size),
                        'initial_jma_value': market_snapshot.jma_value,
                        'trend_info': trend_info,
                        'position_side': position_side
                    }
                    self.logger.info(f"🚨 JMA Trend Warning: {trend_info} detected - monitoring for exit (key: {position_key})")
                    continue
                
                # Position already being tracked
                tracking_info = self._jma_position_tracking[position_key]
                time_elapsed = current_time - tracking_info['first_detected']
                
                # Smart exit conditions
                min_hold_time = 60  # Minimum 60 seconds before JMA exit
                price_movement_threshold = 0.003  # 0.3% price movement required
                
                # Calculate price movement since trend was first detected
                price_change_pct = abs(current_price - entry_price) / entry_price
                
                # Multiple exit conditions for smarter exits
                should_exit = False
                exit_reason = ""
                
                # Condition 1: Minimum time + significant adverse price movement
                if time_elapsed >= min_hold_time and price_change_pct >= price_movement_threshold:
                    if ((side == 'long' and current_price < entry_price * (1 - price_movement_threshold)) or
                        (side == 'short' and current_price > entry_price * (1 + price_movement_threshold))):
                        should_exit = True
                        exit_reason = f"Time buffer ({time_elapsed:.0f}s) + adverse movement ({price_change_pct*100:.2f}%)"
                
                # Condition 2: Strong JMA divergence (JMA value moved significantly further)
                elif time_elapsed >= min_hold_time:
                    jma_change_pct = abs(market_snapshot.jma_value - tracking_info['initial_jma_value']) / tracking_info['initial_jma_value']
                    if jma_change_pct >= 0.005:  # 0.5% JMA movement
                        should_exit = True
                        exit_reason = f"Strong JMA divergence ({jma_change_pct*100:.2f}%) after {time_elapsed:.0f}s"
                
                # Condition 3: Extended time regardless (safety exit)
                elif time_elapsed >= 300:  # 5 minutes maximum
                    should_exit = True
                    exit_reason = f"Maximum hold time ({time_elapsed:.0f}s) reached"
                
                if should_exit:
                    try:
                        # Determine order side to close position
                        close_side = 'sell' if side == 'long' else 'buy'
                        
                        self.logger.info(f"💡 JMA Smart Exit: {trend_info} - {exit_reason}")
                        
                        # Use hedge-aware limit order for better execution
                        if side == 'long':
                            # For long positions, sell slightly below current price
                            exit_price = current_price * 0.9995  # 0.05% below market
                            close_position_side = 'LONG'
                        else:
                            # For short positions, buy slightly above current price  
                            exit_price = current_price * 1.0005  # 0.05% above market
                            close_position_side = 'SHORT'
                        
                        exit_price = self._round_price(exit_price)
                        
                        # Use hedge-aware limit order method
                        order = self.exchange.create_hedge_limit_order(
                            self.symbol, 
                            close_side, 
                            tracking_info['size'], 
                            exit_price,
                            close_position_side
                        )
                        
                        if order and 'id' in order:
                            # Calculate expected PnL
                            if side == 'long':
                                expected_pnl = (exit_price - entry_price) * tracking_info['size']
                            else:
                                expected_pnl = (entry_price - exit_price) * tracking_info['size']
                            
                            self.logger.info(f"✅ JMA Exit order placed: {close_side.upper()} {tracking_info['size']:.6f} @ ${exit_price:.6f}, "
                                        f"Expected PnL: ${expected_pnl:.2f}, Order ID: {order['id'][:8]}")
                            
                            # Mark order as JMA exit order for tracking
                            if order['id'] not in self.pending_orders:
                                self.pending_orders[order['id']] = {}
                            self.pending_orders[order['id']]['is_jma_exit'] = True
                            self.pending_orders[order['id']]['position_key'] = position_key
                            
                        else:
                            self.logger.error(f"❌ Failed to place JMA exit order")
                            continue
                        
                        # Remove from tracking
                        del self._jma_position_tracking[position_key]
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error executing JMA smart exit: {e}")
            
            # Clean up tracking for positions that no longer exist
            live_position_keys = set()
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                if abs(size) >= 0.001 and entry_price > 0:
                    # FIX #3: Use same simplified key format
                    position_key = f"{side}_{entry_price:.4f}"
                    live_position_keys.add(position_key)
            
            # Remove tracking for positions that no longer exist
            stale_keys = [key for key in self._jma_position_tracking.keys() if key not in live_position_keys]
            for key in stale_keys:
                self.logger.debug(f"Removing stale JMA tracking key: {key}")
                del self._jma_position_tracking[key]
            
            if len(self._jma_position_tracking) > 0:
                self.logger.debug(f"🔍 JMA monitoring {len(self._jma_position_tracking)} position(s) for trend alignment")
                
        except Exception as e:
            self.logger.error(f"❌ Error in JMA trend alignment check: {e}")
    # MINIMAL FIX: Replace _get_live_market_data() with proper None handling

    # MINIMAL FIX: Update _get_live_market_data() to detect TP/SL by order type

    def _get_live_market_data(self) -> Dict[str, Any]:
        """ENHANCED: Handle hedge positions with position-side awareness"""
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
            
            # Calculate position margins with hedge mode support
            position_margin = 0.0
            active_positions = 0
            long_positions = 0
            short_positions = 0
            
            for position in live_positions:
                size = float(position.get('contracts', 0))
                if size != 0:
                    entry_price = float(position.get('entryPrice', 0))
                    side = position.get('side', '').lower()
                    position_side = position.get('positionSide', 'BOTH')
                    
                    if entry_price > 0:
                        position_notional = abs(size) * entry_price
                        margin = round(position_notional / self.user_leverage, 2)
                        position_margin += margin
                        active_positions += 1
                        
                        # Track hedge positions separately
                        if side == 'long':
                            long_positions += 1
                        else:
                            short_positions += 1

            # Calculate order margins (exclude TP/SL orders)
            order_margin = 0.0
            active_orders = 0
            tp_sl_orders = 0
            
            for order in live_orders:
                order_id = order.get('id', '')
                
                # Safe price extraction
                price = self._get_order_price(order)
                
                # Safe amount extraction
                amount = order.get('amount')
                if amount is None:
                    amount = 0.0
                else:
                    try:
                        amount = float(amount)
                    except (ValueError, TypeError):
                        amount = 0.0
                
                # Check if this is a TP/SL order by order type
                order_type = order.get('type', '').lower()
                is_tp_sl = order_type in [
                    'take_profit_market', 'stop_market', 
                    'take_profit', 'stop_loss',
                    'take_profit_limit', 'stop_loss_limit'
                ]
                
                if is_tp_sl:
                    tp_sl_orders += 1
                
                # Only calculate margin for NON-TP/SL orders (same as before)
                if price > 0 and amount > 0 and not is_tp_sl:
                    order_notional = price * amount
                    margin = round(order_notional / self.user_leverage, 2)
                    order_margin += margin
                    active_orders += 1

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

            # Enhanced logging with hedge position info
            total_orders = active_orders + tp_sl_orders
            if total_orders > 0 or active_positions > 0 or len(stale_orders) > 0:
                hedge_info = ""
                if long_positions > 0 and short_positions > 0:
                    hedge_info = f" (HEDGE: {long_positions}L/{short_positions}S)"
                elif long_positions > 0:
                    hedge_info = f" ({long_positions} LONG)"
                elif short_positions > 0:
                    hedge_info = f" ({short_positions} SHORT)"
                
                self.logger.info(f"Market state: {active_orders} orders + {tp_sl_orders} TP/SL (${order_margin:.2f}), "
                            f"{active_positions} positions{hedge_info} (${position_margin:.2f}), "
                            f"Total used: ${total_margin_used:.2f}, Available: ${available_investment:.2f}")

            return {
                'live_orders': live_orders,
                'live_positions': live_positions,
                'position_margin': position_margin,
                'order_margin': order_margin,
                'total_margin_used': total_margin_used,
                'available_investment': available_investment,
                'active_orders': active_orders,
                'active_positions': active_positions,
                'long_positions': long_positions,  # NEW: Track hedge positions
                'short_positions': short_positions,  # NEW: Track hedge positions
                'total_commitment': active_orders + active_positions,
                'covered_prices': covered_prices,
                'stale_orders_cleaned': len(stale_orders),
                'tp_sl_orders': tp_sl_orders
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live market data: {e}")
            return {
                'live_orders': [], 'live_positions': [], 'position_margin': 0.0, 'order_margin': 0.0,
                'total_margin_used': self.user_total_investment, 'available_investment': 0.0,
                'active_orders': 0, 'active_positions': 0, 'long_positions': 0, 'short_positions': 0,
                'total_commitment': 0, 'covered_prices': set(), 'stale_orders_cleaned': 0, 'tp_sl_orders': 0
            }
    
    def setup_grid(self):
        """
        UPDATED: Simplified grid setup that only places initial market order.
        Counter orders will be handled by _maintain_counter_orders() in update loop.
        """
        try:
            # Get current market price
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            self.logger.info(f"Setting up grid: Current price ${current_price:.6f}, Range ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}")
            
            # Check if current price is within user range
            if current_price < self.user_price_lower or current_price > self.user_price_upper:
                self.logger.warning(f"Current price ${current_price:.6f} is outside user range - grid may not be immediately effective")
            
            # Cancel existing orders (keep existing logic for safety)
            try:
                open_orders = self.exchange.get_open_orders(self.symbol)
                if open_orders:
                    live_positions = self.exchange.get_positions(self.symbol)
                    cancelled_count = 0
                    
                    for order in open_orders:
                        if self._should_protect_order(order, current_price, live_positions):
                            continue
                        
                        try:
                            self.exchange.cancel_order(order['id'], self.symbol)
                            cancelled_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to cancel order {order['id'][:8]}: {e}")
                    
                    if cancelled_count > 0:
                        self.logger.info(f"Cancelled {cancelled_count} non-protected orders")
                        time.sleep(1)
                            
            except Exception as e:
                self.logger.warning(f"Error during order cancellation: {e}")
            
            # Place initial market order if no position exists
            live_positions = self.exchange.get_positions(self.symbol)
            has_position = any(abs(float(pos.get('contracts', 0))) >= 0.001 for pos in live_positions)
            
            if not has_position:
                if self._place_single_order(current_price, 'buy'):  # Side ignored for market orders
                    self.running = True
                    self.logger.info("Grid setup complete: Initial market order placed, grid is now active")
                    self._grid_start_time = time.time()
                else:
                    self.logger.error("Grid setup failed: Initial market order not placed")
                    self.running = False
            else:
                # Position already exists, just start the grid
                self.running = True
                self.logger.info("Grid setup complete: Position already exists, grid is now active")
                
        except Exception as e:
            self.logger.error(f"Error setting up grid: {e}")
            self.running = False
    
    def _place_initial_grid_orders(self, current_price: float, available_investment: float) -> int:
        """FIXED: Place ONE initial MARKET order to ensure execution"""
        try:
            # Check if initial order already placed
            if hasattr(self, '_initial_order_placed') and self._initial_order_placed:
                self.logger.info("Initial order already placed, skipping")
                return 0
            
            # Check if we have sufficient funds
            if available_investment < self.user_investment_per_grid:
                self.logger.warning(f"Insufficient funds for initial order: ${available_investment:.2f}")
                return 0
            
            # Check if position already exists
            live_positions = self.exchange.get_positions(self.symbol)
            existing_positions = sum(1 for pos in live_positions if abs(float(pos.get('contracts', 0))) >= 0.001)
            
            if existing_positions > 0:
                self.logger.info(f"Position already exists, skipping initial order")
                self._initial_order_placed = True
                return 1  # Consider as successful since position exists
            
            # Get market conditions to determine order side
            jma_trend = 'neutral'
            
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    jma_trend = market_snapshot.jma_trend
                    self.logger.info(f"Market conditions: JMA {jma_trend.upper()}")
                except Exception as e:
                    self.logger.warning(f"Market analysis failed: {e}")
            
            # Determine order side based on JMA trend
            if jma_trend == 'bearish':
                order_side = 'sell'  # Short in bearish market
            elif jma_trend == 'bullish':
                order_side = 'buy'   # Long in bullish market
            else:
                # Neutral - default to buy (generally safer)
                order_side = 'buy'
            
            # Calculate order amount based on current price
            amount = self._calculate_order_amount(current_price)
            notional_value = current_price * amount
            margin_required = round(notional_value / self.user_leverage, 2)
            
            # Final fund check
            if available_investment < margin_required:
                self.logger.error(f"Insufficient funds for market order: Need ${margin_required:.2f}, Available: ${available_investment:.2f}")
                return 0
            
            # Validate order parameters
            if amount < self.min_amount:
                self.logger.error(f"Order amount {amount:.6f} below minimum {self.min_amount}")
                return 0
            
            # CRITICAL: Place MARKET order for guaranteed execution
            try:
                self.logger.info(f"Placing initial MARKET order: {order_side.upper()} {amount:.6f} @ MARKET")
                
                order = self.exchange.create_market_order(self.symbol, order_side, amount)
                
                if not order or 'id' not in order:
                    self.logger.error("Failed to place initial market order - invalid response")
                    return 0
                
                # Mark initial order as placed
                self._initial_order_placed = True
                
                # Get fill price (average execution price)
                fill_price = float(order.get('average', current_price))
                if fill_price == 0:
                    fill_price = current_price  # Fallback if average not available
                
                # Track the order as filled immediately (market orders execute instantly)
                order_info = {
                    'type': order_side,
                    'price': fill_price,
                    'amount': amount,
                    'notional_value': fill_price * amount,
                    'margin_used': margin_required,
                    'timestamp': time.time(),
                    'status': 'filled',  # Market order is immediately filled
                    'is_initial_market_order': True,
                    'jma_trend_at_placement': jma_trend
                }
                
                # Don't add to pending_orders since it's already filled
                # The position will be tracked when _update_orders_and_positions() runs
                
                self.logger.info(f"✅ Initial MARKET order executed: {order_side.upper()} {amount:.6f} @ ${fill_price:.6f}, "
                            f"Notional: ${fill_price * amount:.2f}, Margin: ${margin_required:.2f}, ID: {order['id'][:8]}")
                
                return 1
                
            except Exception as e:
                self.logger.error(f"Failed to place initial market order: {e}")
                return 0
            
        except Exception as e:
            self.logger.error(f"Error in _place_initial_grid_orders: {e}")
            return 0
        
    def _place_single_order(self, price: float, side: str) -> bool:
        """
        ENHANCED: Place hedge orders (primary + hedge position) when no position exists.
        Places both long and short positions to profit from volatility in either direction.
        Uses EXISTING Exchange methods that already support hedge mode.
        """
        try:
            # Check if we already have any positions - if yes, this method shouldn't be called
            live_positions = self.exchange.get_positions(self.symbol)
            has_position = any(abs(float(pos.get('contracts', 0))) >= 0.001 for pos in live_positions)
            
            if has_position:
                self.logger.debug("Position exists - _place_single_order should not create new positions")
                return False
            
            # Get current market price and funds
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            market_data = self._get_live_market_data()
            available_investment = market_data['available_investment']
            
            # HEDGE STRATEGY: Split investment between primary (70%) and hedge (30%) positions
            primary_investment_pct = 0.50
            hedge_investment_pct = 0.50
            
            # Calculate amounts for both positions
            primary_investment = self.user_investment_per_grid * primary_investment_pct
            hedge_investment = self.user_investment_per_grid * hedge_investment_pct
            
            primary_amount = self._calculate_order_amount_for_investment(current_price, primary_investment)
            hedge_amount = self._calculate_order_amount_for_investment(current_price, hedge_investment)
            
            # Calculate total margin required for both positions
            primary_notional = current_price * primary_amount
            hedge_notional = current_price * hedge_amount
            total_margin_required = round((primary_notional + hedge_notional) / self.user_leverage, 2)
            
            # Check funds
            if available_investment < total_margin_required:
                self.logger.info(f"Insufficient funds for hedge orders: Need ${total_margin_required:.2f}, Available: ${available_investment:.2f}")
                return False
            
            # Determine order sides based on JMA trend and hedge strategy
            primary_side = 'buy'  # Default primary side
            hedge_side = 'sell'   # Opposite side for hedge
            jma_trend = 'neutral'
            
            if self.enable_samig and self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    jma_trend = market_snapshot.jma_trend
                    
                    # Primary position follows JMA trend, hedge is opposite
                    if jma_trend == 'bullish':
                        primary_side = 'buy'
                        hedge_side = 'sell'
                    elif jma_trend == 'bearish':
                        primary_side = 'sell'
                        hedge_side = 'buy'
                    else:
                        # Neutral - default to buy primary, sell hedge
                        primary_side = 'buy'
                        hedge_side = 'sell'
                        
                    self.logger.info(f"JMA trend: {jma_trend.upper()} -> Primary: {primary_side.upper()}, Hedge: {hedge_side.upper()}")
                    
                except Exception as e:
                    self.logger.warning(f"JMA analysis failed, using default hedge strategy: {e}")
            
            # Validate order parameters
            if primary_amount < self.min_amount or hedge_amount < self.min_amount:
                self.logger.error(f"Order amounts below minimum - Primary: {primary_amount:.6f}, Hedge: {hedge_amount:.6f}, Min: {self.min_amount}")
                return False
            
            # Validate notional values
            if primary_notional < self.min_cost or hedge_notional < self.min_cost:
                self.logger.error(f"Order notionals below minimum - Primary: ${primary_notional:.2f}, Hedge: ${hedge_notional:.2f}, Min: ${self.min_cost}")
                return False
            
            # Place PRIMARY position (larger position following trend)
            try:
                self.logger.info(f"Placing PRIMARY hedge order: {primary_side.upper()} {primary_amount:.6f} @ MARKET (70% of investment)")
                
                # Use EXISTING exchange method - it already supports hedge mode
                primary_order = self.exchange.create_market_order(
                    self.symbol, 
                    primary_side, 
                    primary_amount
                )
                
                if not primary_order or 'id' not in primary_order:
                    self.logger.error("Failed to place primary hedge order - invalid response")
                    return False
                
                primary_fill_price = float(primary_order.get('average', current_price))
                if primary_fill_price == 0:
                    primary_fill_price = current_price
                
                self.logger.info(f"✅ PRIMARY hedge order executed: {primary_side.upper()} {primary_amount:.6f} @ ${primary_fill_price:.6f}, "
                                f"Margin: ${round(primary_notional / self.user_leverage, 2):.2f}")
                
            except Exception as e:
                self.logger.error(f"Failed to place primary hedge order: {e}")
                return False
            
            # Small delay between orders
            time.sleep(0.5)
            
            # Place HEDGE position (smaller opposite position)
            try:
                self.logger.info(f"Placing HEDGE order: {hedge_side.upper()} {hedge_amount:.6f} @ MARKET (30% of investment)")
                
                # Use EXISTING exchange method - it already supports hedge mode
                hedge_order = self.exchange.create_market_order(
                    self.symbol, 
                    hedge_side, 
                    hedge_amount
                )
                
                if not hedge_order or 'id' not in hedge_order:
                    self.logger.warning("Failed to place hedge order - continuing with primary position only")
                    # Don't return False here - primary order was successful
                else:
                    hedge_fill_price = float(hedge_order.get('average', current_price))
                    if hedge_fill_price == 0:
                        hedge_fill_price = current_price
                    
                    self.logger.info(f"✅ HEDGE order executed: {hedge_side.upper()} {hedge_amount:.6f} @ ${hedge_fill_price:.6f}, "
                                    f"Margin: ${round(hedge_notional / self.user_leverage, 2):.2f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to place hedge order (continuing with primary only): {e}")
            
            # Mark initial orders as placed
            self._initial_order_placed = True
            
            # Calculate total strategy investment
            total_invested = round((primary_notional + hedge_notional) / self.user_leverage, 2)
            net_exposure = abs(primary_notional - hedge_notional) / self.user_leverage
            
            self.logger.info(f"🎯 HEDGE STRATEGY DEPLOYED:")
            self.logger.info(f"   Total Investment: ${total_invested:.2f}")
            self.logger.info(f"   Net Exposure: ${net_exposure:.2f}")
            self.logger.info(f"   Strategy: {primary_side.upper()} (70%) + {hedge_side.upper()} (30%)")
            self.logger.info(f"   JMA Trend: {jma_trend.upper()}")
            
            return True
                
        except Exception as e:
            self.logger.error(f"Error in hedge order placement: {e}")
            return False
    def _calculate_order_amount_for_investment(self, price: float, investment_amount: float) -> float:
        """Calculate order amount for specific investment amount"""
        try:
            if price <= 0 or investment_amount <= 0:
                return self.min_amount
            
            # Calculate notional value from specific investment amount
            notional_value = investment_amount * self.user_leverage
            
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
            self.logger.error(f"Error calculating order amount for investment ${investment_amount:.2f} at price ${price:.6f}: {e}")
            return self.min_amount
    def _maintain_hedge_counter_orders(self, live_positions: List[Dict], live_orders: List[Dict], current_price: float):
        """
        ENHANCED: Maintain counter orders for hedge positions with position-side awareness
        """
        try:
            # Group positions by side for hedge mode
            long_positions = []
            short_positions = []
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                position_side = pos.get('positionSide', 'BOTH')
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                
                notional_value = abs(size) * entry_price
                if notional_value < 0.98:  # Same dust filtering
                    continue
                
                position_data = {
                    'entry_price': entry_price,
                    'side': side,
                    'quantity': abs(size),
                    'position_side': position_side
                }
                
                if side == 'long':
                    long_positions.append(position_data)
                else:
                    short_positions.append(position_data)
            
            # Handle counter orders for long positions
            for position in long_positions:
                self._maintain_position_counter_orders(position, live_orders, current_price, 'LONG')
            
            # Handle counter orders for short positions  
            for position in short_positions:
                self._maintain_position_counter_orders(position, live_orders, current_price, 'SHORT')
                
        except Exception as e:
            self.logger.error(f"Error maintaining hedge counter orders: {e}")
    def _maintain_position_counter_orders(self, position: dict, live_orders: list, current_price: float):
        """Maintain counter orders for a position using EXISTING exchange methods"""
        try:
            entry_price = position['entry_price']
            side = position['side']
            quantity = position['quantity']
            
            # Determine counter order side
            counter_side = 'sell' if side == 'long' else 'buy'
            
            # Check if we already have counter orders (exclude TP/SL orders)
            existing_counter_orders = 0
            for order in live_orders:
                order_side = order.get('side', '').lower()
                order_type = order.get('type', '').lower()
                
                # Count only limit orders in counter direction (not TP/SL)
                if (order_side == counter_side and 
                    order_type == 'limit'):  # Only limit orders are grid counter orders
                    existing_counter_orders += 1
            
            # If we already have a counter order, don't create more
            if existing_counter_orders >= 1:
                self.logger.debug(f"Counter order already exists ({existing_counter_orders}), skipping creation")
                return
            
            # Calculate SINGLE optimal counter order distance
            price_range = self.user_price_upper - self.user_price_lower
            grid_spacing = price_range / self.user_grid_number
            
            # Use optimal spacing: larger of grid spacing or 1% of entry price
            counter_spacing = max(grid_spacing, entry_price * 0.01)
            
            # Calculate single counter order price
            if side == 'long':
                # For long positions, place ONE sell order above entry
                counter_price = self._round_price(entry_price + counter_spacing)
            else:
                # For short positions, place ONE buy order below entry
                counter_price = self._round_price(entry_price - counter_spacing)
            
            # Validate price is within range
            if not (self.user_price_lower <= counter_price <= self.user_price_upper):
                self.logger.debug(f"Counter price ${counter_price:.6f} outside range, skipping")
                return
            
            # Check funds
            market_data = self._get_live_market_data()
            available_investment = market_data['available_investment']
            
            counter_amount = self._calculate_order_amount(counter_price)
            notional_value = counter_price * counter_amount
            margin_required = round(notional_value / self.user_leverage, 2)
            
            if available_investment < margin_required:
                self.logger.debug(f"Insufficient funds for counter order")
                return
            
            # Apply smart filters
            if not self._should_place_counter_order(counter_price, counter_side, current_price, entry_price):
                return
            
            # Place counter order using EXISTING exchange method
            try:
                self.logger.info(f"Placing counter order: {counter_side.upper()} {counter_amount:.6f} @ ${counter_price:.6f}")
                
                # Use EXISTING create_limit_order method - it already supports hedge mode
                order = self.exchange.create_limit_order(
                    self.symbol, 
                    counter_side, 
                    counter_amount, 
                    counter_price
                )
                
                if order and 'id' in order:
                    self.pending_orders[order['id']] = {
                        'type': counter_side,
                        'price': counter_price,
                        'amount': counter_amount,
                        'timestamp': time.time(),
                        'status': 'open',
                        'is_grid_counter_order': True,
                        'position_entry_price': entry_price
                    }
                    
                    distance_pct = abs(counter_price - entry_price) / entry_price * 100
                    profit_potential = abs(counter_price - entry_price) * counter_amount
                    
                    self.logger.info(f"✅ Single counter order placed: {counter_side.upper()} {counter_amount:.6f} @ ${counter_price:.6f} "
                                f"({distance_pct:.2f}% from entry), Potential profit: ${profit_potential:.2f}, ID: {order['id'][:8]}")
                else:
                    self.logger.error("Failed to place counter order - invalid response")
                    
            except Exception as e:
                self.logger.error(f"Failed to place counter order: {e}")
                
        except Exception as e:
            self.logger.error(f"Error maintaining counter orders: {e}")
    def _validate_tp_sl_price(self, side: str, order_type: str, price: float, current_price: float) -> bool:
        """Validate that TP/SL price won't trigger immediately"""
        try:
            if order_type == 'TP':  # Take Profit validation
                if side == 'long':
                    # LONG TP: must be above current price
                    return price > current_price
                else:  # short
                    # SHORT TP: must be below current price
                    return price < current_price
            else:  # Stop Loss validation
                if side == 'long':
                    # LONG SL: must be below current price
                    return price < current_price
                else:  # short
                    # SHORT SL: must be above current price
                    return price > current_price
        except Exception as e:
            self.logger.error(f"Error validating TP/SL price: {e}")
            return False
    def _maintain_hedge_tp_sl_orders(self, live_positions: List[Dict], live_orders: List[Dict]):
        """FIXED: Maintain TP/SL orders for hedge positions with proper price validation"""
        try:
            # Get current market price for validation
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            for position in live_positions:
                entry_price = float(position.get('entryPrice', 0))
                side = position.get('side', '').lower()
                quantity = abs(float(position.get('contracts', 0)))
                position_side = position.get('positionSide', 'BOTH')
                
                if quantity < 0.001 or entry_price <= 0:
                    continue
                
                # Debug logging for position details
                self.logger.debug(f"Processing {side.upper()} position: Entry ${entry_price:.6f}, "
                                f"Current ${current_price:.6f}, Position side: {position_side}")
                
                # Calculate TP/SL prices based on CURRENT MARKET PRICE (not entry price)
                if side == 'long':
                    # For LONG positions:
                    # TP: Above current price (profit when price goes up)
                    # SL: Below current price (limit loss when price goes down)
                    tp_price = self._round_price(max(current_price * 1.002, entry_price * 1.01))  # At least 0.2% above current or 1% above entry
                    sl_price = self._round_price(min(current_price * 0.995, entry_price * 0.97))  # At least 0.5% below current or 3% below entry
                    expected_order_side = 'sell'
                else:  # short
                    # For SHORT positions:
                    # TP: Below current price (profit when price goes down)
                    # SL: Above current price (limit loss when price goes up)
                    tp_price = self._round_price(min(current_price * 0.998, entry_price * 0.99))  # At least 0.2% below current or 1% below entry
                    sl_price = self._round_price(max(current_price * 1.005, entry_price * 1.03))  # At least 0.5% above current or 3% above entry
                    expected_order_side = 'buy'
                
                # CRITICAL: Validate TP/SL prices won't trigger immediately
                if side == 'long':
                    # For LONG: TP must be above current price, SL must be below current price
                    if tp_price <= current_price:
                        tp_price = self._round_price(current_price * 1.003)  # Set 0.3% above current price
                        self.logger.warning(f"Adjusted LONG TP price to ${tp_price:.6f} (was too low)")
                    
                    if sl_price >= current_price:
                        sl_price = self._round_price(current_price * 0.993)  # Set 0.7% below current price
                        self.logger.warning(f"Adjusted LONG SL price to ${sl_price:.6f} (was too high)")
                        
                else:  # short
                    # For SHORT: TP must be below current price, SL must be above current price
                    if tp_price >= current_price:
                        tp_price = self._round_price(current_price * 0.997)  # Set 0.3% below current price
                        self.logger.warning(f"Adjusted SHORT TP price to ${tp_price:.6f} (was too high)")
                    
                    if sl_price <= current_price:
                        sl_price = self._round_price(current_price * 1.007)  # Set 0.7% above current price
                        self.logger.warning(f"Adjusted SHORT SL price to ${sl_price:.6f} (was too low)")
                
                # Log final calculated prices
                self.logger.debug(f"{side.upper()} position prices: TP ${tp_price:.6f}, SL ${sl_price:.6f}, "
                                f"Market ${current_price:.6f}, Entry ${entry_price:.6f}")
                
                # Final sanity check on calculated prices
                tp_valid = self._validate_tp_sl_price(side, 'TP', tp_price, current_price)
                sl_valid = self._validate_tp_sl_price(side, 'SL', sl_price, current_price)
                
                if not tp_valid:
                    self.logger.error(f"Invalid TP price calculated for {side} position: ${tp_price:.6f} vs market ${current_price:.6f}")
                    continue
                    
                if not sl_valid:
                    self.logger.error(f"Invalid SL price calculated for {side} position: ${sl_price:.6f} vs market ${current_price:.6f}")
                    continue
                
                # Check existing TP/SL orders for this position side
                has_tp_order = False
                has_sl_order = False
                
                for order in live_orders:
                    order_type = order.get('type', '').lower()
                    order_side = order.get('side', '').lower()
                    order_position_side = order.get('info', {}).get('positionSide', 'BOTH')
                    
                    if order_side != expected_order_side or order_position_side != position_side:
                        continue
                        
                    if order_type in ['take_profit_market', 'take_profit', 'take_profit_limit']:
                        has_tp_order = True
                    elif order_type in ['stop_market', 'stop_loss', 'stop_loss_limit']:
                        has_sl_order = True
                
                # Create missing TP/SL orders for this position side
                orders_to_create = []
                if not has_tp_order:
                    orders_to_create.append(('TP', tp_price, 'TAKE_PROFIT_MARKET'))
                if not has_sl_order:
                    orders_to_create.append(('SL', sl_price, 'STOP_MARKET'))
                
                for order_type, price, binance_type in orders_to_create:
                    try:
                        # Final validation before placing order
                        price_diff_pct = abs(price - current_price) / current_price * 100
                        
                        if price_diff_pct < 0.1:  # Less than 0.1% difference
                            self.logger.warning(f"Skipping {order_type} for {position_side}: price ${price:.6f} too close to market ${current_price:.6f}")
                            continue
                        
                        # Validate using helper method
                        if not self._validate_tp_sl_price(side, order_type, price, current_price):
                            self.logger.error(f"Invalid {order_type} price for {side} position: ${price:.6f} vs market ${current_price:.6f}")
                            continue
                        
                        self.logger.info(f"Creating {order_type} for {position_side}: {expected_order_side.upper()} @ ${price:.6f} "
                                       f"(market: ${current_price:.6f}, entry: ${entry_price:.6f})")
                        
                        new_order = self.exchange.create_hedge_stop_order(
                            self.symbol,
                            expected_order_side,
                            quantity,
                            price,
                            position_side,
                            binance_type  # Pass the correct order type
                        )
                        
                        if new_order and 'id' in new_order:
                            self.pending_orders[new_order['id']] = {
                                'type': expected_order_side,
                                'price': price,
                                'amount': quantity,
                                'timestamp': time.time(),
                                'order_purpose': 'profit' if order_type == 'TP' else 'stop_loss',
                                'is_tp_sl_order': True,
                                'position_side': position_side
                            }
                            
                            expected_pnl = 0
                            if order_type == 'TP':
                                if side == 'long':
                                    expected_pnl = (price - entry_price) * quantity
                                else:
                                    expected_pnl = (entry_price - price) * quantity
                            
                            self.logger.info(f"✅ {order_type} created for {position_side}: {new_order['id'][:8]} @ ${price:.6f}, "
                                           f"Expected PnL: ${expected_pnl:.2f}")
                            time.sleep(0.2)
                            
                    except Exception as e:
                        self.logger.error(f"{order_type} creation failed for {position_side}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error maintaining hedge TP/SL orders: {e}")
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
        """
        ENHANCED: Main grid update loop with proper hedge position handling.
        Now validates and maintains both standard and hedge strategies correctly.
        """
        try:
            with self.update_lock:
                if not self.running:
                    return
                
                # Get current market price
                ticker = self.exchange.get_ticker(self.symbol)
                current_price = float(ticker['last'])
                
                # 1. Update filled orders and positions (handles hedge positions properly)
                self._update_orders_and_positions()
                
                # 2. Check JMA trend alignment for hedge positions (when SAMIG enabled)
                # if self.enable_samig:
                #     self._check_jma_trend_alignment(current_price)
                
                # 3. Maintain grid coverage (now handles missing hedge positions)
                self._maintain_grid_coverage(current_price)
                
                # 4. Create counter orders for ALL open positions (long and short separately)
                self._maintain_counter_orders(current_price)
                
                # 5. Update PnL calculations (includes hedge position PnL)
                self._update_pnl(current_price)
                
                # 6. Check take profit and stop loss (considers total hedge PnL)
                self._check_tp_sl()
                
                self.last_update_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
    
    def _maintain_grid_coverage(self, current_price: float):
        """
        UPDATED: Enhanced to support hedge position maintenance.
        Now properly handles both standard and hedge strategies.
        """
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
            
            # Fill gaps with proper hedge position handling
            self._fill_grid_gaps(current_price, market_data)
            
        except Exception as e:
            self.logger.error(f"Error maintaining grid coverage: {e}")

    def _should_place_new_orders(self, intended_side: str = None) -> bool:
        """
        FIXED: Updated to handle hedge positioning properly.
        Now considers both long and short positions separately when SAMIG is enabled.
        """
        try:
            live_positions = self.exchange.get_positions(self.symbol)
            
            # Count real positions by side
            MIN_POSITION_NOTIONAL = 0.98
            long_positions = 0
            short_positions = 0
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                    
                notional_value = abs(size) * entry_price
                if notional_value < MIN_POSITION_NOTIONAL:
                    continue
                
                if side == 'long':
                    long_positions += 1
                elif side == 'short':
                    short_positions += 1
            
            # HEDGE STRATEGY: If SAMIG enabled, allow orders when missing hedge positions
            if self.enable_samig:
                # Allow new orders if either long or short position is missing
                if intended_side:
                    if intended_side == 'long' and long_positions == 0:
                        return True
                    elif intended_side == 'short' and short_positions == 0:
                        return True
                else:
                    # No specific side - allow if any position is missing
                    return long_positions == 0 or short_positions == 0
            
            # STANDARD STRATEGY: Only allow new orders if no positions exist
            return long_positions == 0 and short_positions == 0
            
        except Exception as e:
            self.logger.error(f"Error checking if should place new orders: {e}")
            return False


    def _fill_grid_gaps(self, current_price: float, market_data: Dict[str, Any]):
        """
        FIXED: Handle initial position creation with proper hedge position checking.
        Now creates EQUAL-SIZE hedge positions (50:50 split) for true hedging.
        """
        try:
            # Get live positions and categorize them
            live_positions = self.exchange.get_positions(self.symbol)
            
            # Separate long and short positions (hedge-aware checking)
            has_long_position = False
            has_short_position = False
            
            MIN_POSITION_NOTIONAL = 0.98
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                    
                notional_value = abs(size) * entry_price
                if notional_value < MIN_POSITION_NOTIONAL:
                    continue
                
                # Categorize real positions by side
                if side == 'long':
                    has_long_position = True
                elif side == 'short':
                    has_short_position = True
            
            # Check available investment and order capacity
            if (market_data['available_investment'] < self.user_investment_per_grid or 
                market_data['total_commitment'] >= self.max_total_orders):
                return
            
            # HEDGE STRATEGY: Create missing positions with EQUAL SIZES
            if self.enable_samig:
                # Calculate missing positions
                missing_positions = []
                if not has_long_position:
                    missing_positions.append('long')
                if not has_short_position:
                    missing_positions.append('short')
                
                if missing_positions:
                    # For equal-size hedge positions, each position uses 50% of per-grid investment
                    # Total needed = missing_positions * (per_grid_investment * 0.5)
                    investment_per_missing_position = self.user_investment_per_grid * 0.5
                    total_investment_needed = len(missing_positions) * investment_per_missing_position
                    
                    if market_data['available_investment'] >= total_investment_needed:
                        self._place_missing_hedge_positions(current_price, missing_positions)
                        return
                    else:
                        self.logger.info(f"Insufficient funds for {len(missing_positions)} equal-size hedge position(s): "
                                    f"Need ${total_investment_needed:.2f}, Available: ${market_data['available_investment']:.2f}")
                        return
            
            # STANDARD STRATEGY: Only create position if none exists
            if not has_long_position and not has_short_position:
                # Check if initial order was already placed recently
                if hasattr(self, '_initial_order_placed') and self._initial_order_placed:
                    time_since_start = time.time() - getattr(self, '_grid_start_time', 0)
                    if time_since_start < 60:  # Wait 60 seconds after initial order
                        return
                
                # For equal-size hedge positions, we need full investment (50% + 50% = 100%)
                total_investment_needed = self.user_investment_per_grid
                
                if market_data['available_investment'] >= total_investment_needed:
                    # Place initial equal-size hedge orders
                    if self._place_single_order(current_price, 'buy'):  # Side ignored for market orders
                        self.logger.info("Initial equal-size hedge positions created via market orders")
                        self._grid_start_time = time.time()
                else:
                    self.logger.info(f"Insufficient funds for equal-size hedge positions: "
                                f"Need ${total_investment_needed:.2f}, Available: ${market_data['available_investment']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in _fill_grid_gaps: {e}")
    def _place_missing_hedge_positions(self, current_price: float, missing_sides: List[str]):
        """
        FIXED: Place missing hedge positions with EQUAL SIZE to existing positions.
        Matches the quantity of existing positions for true hedge strategy.
        """
        try:
            # Get existing positions to match their sizes
            live_positions = self.exchange.get_positions(self.symbol)
            existing_position_size = 0.0
            
            # Find the size of existing positions to match
            MIN_POSITION_NOTIONAL = 0.98
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                    
                notional_value = abs(size) * entry_price
                if notional_value < MIN_POSITION_NOTIONAL:
                    continue
                
                # Use the first valid position size as reference
                existing_position_size = abs(size)
                break
            
            # If no existing position found, calculate based on equal investment split
            if existing_position_size == 0:
                # For simultaneous creation: split investment equally (50:50)
                equal_investment = self.user_investment_per_grid * 0.5
                existing_position_size = self._calculate_order_amount_for_investment(current_price, equal_investment)
            
            # Get market conditions for side determination
            jma_trend = 'neutral'
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    jma_trend = market_snapshot.jma_trend
                except Exception as e:
                    self.logger.warning(f"Market analysis failed: {e}")
            
            self.logger.info(f"Creating missing hedge positions: {missing_sides} with size {existing_position_size:.6f} (JMA: {jma_trend.upper()})")
            
            for side in missing_sides:
                try:
                    # Use EQUAL SIZE to existing position
                    amount = existing_position_size
                    
                    # Validate order parameters
                    if amount < self.min_amount:
                        self.logger.error(f"Order amount {amount:.6f} below minimum for {side} hedge position")
                        continue
                    
                    notional_value = current_price * amount
                    margin_required = round(notional_value / self.user_leverage, 2)
                    
                    # Determine order side
                    order_side = 'buy' if side == 'long' else 'sell'
                    
                    self.logger.info(f"Placing equal-size {side.upper()} hedge position: {order_side.upper()} {amount:.6f} @ MARKET")
                    
                    # Use existing exchange method (already supports hedge mode)
                    order = self.exchange.create_market_order(self.symbol, order_side, amount)
                    
                    if order and 'id' in order:
                        fill_price = float(order.get('average', current_price))
                        if fill_price == 0:
                            fill_price = current_price
                        
                        self.logger.info(f"✅ Equal-size {side.upper()} hedge position created: {order_side.upper()} {amount:.6f} @ ${fill_price:.6f}, "
                                    f"Margin: ${margin_required:.2f}, ID: {order['id'][:8]}")
                    else:
                        self.logger.error(f"Failed to create {side} hedge position")
                    
                    # Small delay between orders
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Error creating {side} hedge position: {e}")
            
        except Exception as e:
            self.logger.error(f"Error placing missing hedge positions: {e}")
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
        """ENHANCED: Handle hedge positions and orders with position-side awareness"""
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
                    position_side = order.get('info', {}).get('positionSide', 'BOTH')
                    
                    self.pending_orders[order_id] = {
                        'type': order.get('side', ''),
                        'price': order_price,
                        'amount': float(order.get('amount', 0)),
                        'timestamp': time.time(),
                        'status': 'open',
                        'position_side': position_side  # Track position side for hedge mode
                    }
            
            # BB cleanup with hedge mode support (protect TP/SL orders)
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
                        
                        # Cancel orders outside BB range (protect TP/SL orders)
                        bb_margin = (self.user_price_upper - self.user_price_lower) * 0.02
                        for order in live_orders:
                            order_price = self._get_order_price(order)
                            order_side = order.get('side', '')
                            order_id = order.get('id', '')
                            order_type = order.get('type', '').lower()
                            position_side = order.get('info', {}).get('positionSide', 'BOTH')
                            
                            # PROTECT TP/SL orders (same logic as before)
                            if order_type in ['take_profit_market', 'stop_market', 'take_profit', 'stop_loss', 'take_profit_limit', 'stop_loss_limit']:
                                continue
                            
                            # Skip if no valid price or too close to current price
                            if order_price <= 0:
                                continue
                                
                            price_distance = abs(order_price - current_price) / current_price
                            if price_distance < 0.01:
                                continue
                                
                            # Cancel if outside BB range (only non-TP/SL orders)
                            if (order_price < self.user_price_lower - bb_margin or 
                                order_price > self.user_price_upper + bb_margin):
                                try:
                                    self.logger.info(f"BB cleanup: {order_side.upper()} @ ${order_price:.6f} (pos: {position_side})")
                                    self.exchange.cancel_order(order_id, self.symbol)
                                    if order_id in self.pending_orders:
                                        del self.pending_orders[order_id]
                                except Exception as e:
                                    self.logger.error(f"Cancel failed {order_id[:8]}: {e}")
            except Exception as e:
                self.logger.error(f"BB cleanup error: {e}")
            
            # Calculate PnL with hedge mode support
            total_unrealized = 0.0
            active_position_count = 0
            hedge_pnl_breakdown = {'long': 0.0, 'short': 0.0}
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                position_side = pos.get('positionSide', 'BOTH')
                
                if abs(size) >= 0.001 and entry_price > 0:
                    active_position_count += 1
                    
                    # Calculate unrealized PnL for this position
                    if side == 'long':
                        unrealized = (current_price - entry_price) * abs(size)
                        hedge_pnl_breakdown['long'] += unrealized
                    else:
                        unrealized = (entry_price - current_price) * abs(size)
                        hedge_pnl_breakdown['short'] += unrealized
                    
                    total_unrealized += unrealized
            
            # Calculate total PnL (don't store positions locally)
            total_realized = sum(pos.realized_pnl for pos in self.all_positions.values() if not pos.is_open())
            self.total_pnl = total_realized + total_unrealized
            
            # Enhanced hedge PnL logging
            if hedge_pnl_breakdown['long'] != 0 or hedge_pnl_breakdown['short'] != 0:
                net_hedge_pnl = hedge_pnl_breakdown['long'] + hedge_pnl_breakdown['short']
                self.logger.debug(f"Hedge PnL: Long ${hedge_pnl_breakdown['long']:.2f}, "
                                f"Short ${hedge_pnl_breakdown['short']:.2f}, Net ${net_hedge_pnl:.2f}")
            
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
        """
        Handle counter orders using EXISTING exchange methods with hedge mode support
        """
        try:
            live_positions = self.exchange.get_positions(self.symbol)
            live_orders = self.exchange.get_open_orders(self.symbol)
            
            # Filter real positions (same logic as before)
            MIN_POSITION_NOTIONAL = 0.98
            real_positions = []
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                    
                notional_value = abs(size) * entry_price
                if notional_value < MIN_POSITION_NOTIONAL:
                    continue
                
                real_positions.append(pos)
            
            if not real_positions:
                # No real positions - clean up any orphaned orders
                self._cleanup_orphaned_tp_sl_orders(live_orders)
                return
            
            # 1. Handle TP/SL orders using EXISTING exchange methods
            self._maintain_tp_sl_orders(real_positions, live_orders)
            
            # 2. Handle counter orders using EXISTING exchange methods
            self._maintain_grid_counter_orders(real_positions, live_orders, current_price)
            
        except Exception as e:
            self.logger.error(f"Error in counter orders maintenance: {e}")
    def _maintain_grid_counter_orders(self, live_positions: List[Dict], live_orders: List[Dict], current_price: float):
        """
        Maintain counter orders for hedge positions - Uses EXISTING Exchange methods
        """
        try:
            # Group positions by side for hedge mode
            long_positions = []
            short_positions = []
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                position_side = pos.get('positionSide', 'BOTH')
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                
                notional_value = abs(size) * entry_price
                if notional_value < 0.98:  # Same dust filtering
                    continue
                
                position_data = {
                    'entry_price': entry_price,
                    'side': side,
                    'quantity': abs(size),
                    'position_side': position_side
                }
                
                if side == 'long':
                    long_positions.append(position_data)
                else:
                    short_positions.append(position_data)
            
            # Handle counter orders for long positions
            for position in long_positions:
                self._maintain_position_counter_orders(position, live_orders, current_price)
            
            # Handle counter orders for short positions  
            for position in short_positions:
                self._maintain_position_counter_orders(position, live_orders, current_price)
                
        except Exception as e:
            self.logger.error(f"Error maintaining grid counter orders: {e}")

    def _should_place_counter_order(self, target_price: float, counter_side: str, current_price: float, entry_price: float) -> bool:
        """Apply smart filters for grid counter order placement"""
        try:
            # Basic distance check
            distance_from_current = abs(target_price - current_price) / current_price
            if distance_from_current < 0.005:  # Too close to current price
                return False
            
            # Check if counter order would be profitable
            if counter_side == 'sell':
                # Sell order should be above entry price
                if target_price <= entry_price * 1.002:  # At least 0.2% profit
                    return False
            else:
                # Buy order should be below entry price  
                if target_price >= entry_price * 0.998:  # At least 0.2% profit
                    return False
            
            # JMA trend filter for counter orders (less strict than new positions)
            if self.enable_samig and self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    
                    # Only block counter orders in very strong opposing trends
                    if market_snapshot.jma_strength > 0.7:  # High confidence threshold
                        if (counter_side == 'sell' and market_snapshot.jma_trend == 'bearish' or
                            counter_side == 'buy' and market_snapshot.jma_trend == 'bullish'):
                            self.logger.debug(f"Counter order blocked: Strong {market_snapshot.jma_trend} JMA trend")
                            return False
                            
                except Exception as e:
                    self.logger.debug(f"JMA filter error for counter order: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in counter order filter: {e}")
            return False
    def _maintain_tp_sl_orders(self, live_positions: List[Dict], live_orders: List[Dict]):
        """Maintain TP/SL orders for hedge positions with proper price validation - Uses EXISTING Exchange methods"""
        try:
            # Get current market price for validation
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            for position in live_positions:
                entry_price = float(position.get('entryPrice', 0))
                side = position.get('side', '').lower()
                quantity = abs(float(position.get('contracts', 0)))
                position_side = position.get('positionSide', 'BOTH')
                
                if quantity < 0.001 or entry_price <= 0:
                    continue
                
                # Debug logging for position details
                self.logger.debug(f"Processing {side.upper()} position: Entry ${entry_price:.6f}, "
                                f"Current ${current_price:.6f}, Position side: {position_side}")
                
                # Calculate TP/SL prices based on CURRENT MARKET PRICE (not entry price)
                if side == 'long':
                    # For LONG positions:
                    # TP: Above current price (profit when price goes up)
                    # SL: Below current price (limit loss when price goes down)
                    tp_price = self._round_price(max(current_price * 1.002, entry_price * 1.005))  # At least 0.2% above current or 1% above entry
                    sl_price = self._round_price(min(current_price * 0.995, entry_price * 0.97))  # At least 0.5% below current or 3% below entry
                    expected_order_side = 'sell'
                else:  # short
                    # For SHORT positions:
                    # TP: Below current price (profit when price goes down)
                    # SL: Above current price (limit loss when price goes up)
                    tp_price = self._round_price(min(current_price * 0.998, entry_price * 0.995))  # At least 0.2% below current or 1% below entry
                    sl_price = self._round_price(max(current_price * 1.005, entry_price * 1.03))  # At least 0.5% above current or 3% above entry
                    expected_order_side = 'buy'
                
                # CRITICAL: Validate TP/SL prices won't trigger immediately
                if side == 'long':
                    # For LONG: TP must be above current price, SL must be below current price
                    if tp_price <= current_price:
                        tp_price = self._round_price(current_price * 1.003)  # Set 0.3% above current price
                        self.logger.warning(f"Adjusted LONG TP price to ${tp_price:.6f} (was too low)")
                    
                    if sl_price >= current_price:
                        sl_price = self._round_price(current_price * 0.993)  # Set 0.7% below current price
                        self.logger.warning(f"Adjusted LONG SL price to ${sl_price:.6f} (was too high)")
                        
                else:  # short
                    # For SHORT: TP must be below current price, SL must be above current price
                    if tp_price >= current_price:
                        tp_price = self._round_price(current_price * 0.997)  # Set 0.3% below current price
                        self.logger.warning(f"Adjusted SHORT TP price to ${tp_price:.6f} (was too high)")
                    
                    if sl_price <= current_price:
                        sl_price = self._round_price(current_price * 1.007)  # Set 0.7% above current price
                        self.logger.warning(f"Adjusted SHORT SL price to ${sl_price:.6f} (was too low)")
                
                # Log final calculated prices
                self.logger.debug(f"{side.upper()} position prices: TP ${tp_price:.6f}, SL ${sl_price:.6f}, "
                                f"Market ${current_price:.6f}, Entry ${entry_price:.6f}")
                
                # Final sanity check on calculated prices
                tp_valid = self._validate_tp_sl_price(side, 'TP', tp_price, current_price)
                sl_valid = self._validate_tp_sl_price(side, 'SL', sl_price, current_price)
                
                if not tp_valid:
                    self.logger.error(f"Invalid TP price calculated for {side} position: ${tp_price:.6f} vs market ${current_price:.6f}")
                    continue
                    
                if not sl_valid:
                    self.logger.error(f"Invalid SL price calculated for {side} position: ${sl_price:.6f} vs market ${current_price:.6f}")
                    continue
                
                # Check existing TP/SL orders for this position side
                has_tp_order = False
                has_sl_order = False
                
                for order in live_orders:
                    order_type = order.get('type', '').lower()
                    order_side = order.get('side', '').lower()
                    order_position_side = order.get('info', {}).get('positionSide', 'BOTH')
                    
                    is_tp_order = order_type in ['take_profit_market', 'take_profit', 'take_profit_limit']
                    is_sl_order = order_type in ['stop_market', 'stop_loss', 'stop_loss_limit']
                    
                    if not (is_tp_order or is_sl_order):
                        continue
                    
                        
                    if order_type in ['take_profit_market', 'take_profit', 'take_profit_limit']:
                        has_tp_order = True
                    elif order_type in ['stop_market', 'stop_loss', 'stop_loss_limit']:
                        has_sl_order = True
                
                # Create missing TP/SL orders for this position side
                orders_to_create = []
                if not has_tp_order:
                    orders_to_create.append(('TP', tp_price, 'TAKE_PROFIT_MARKET'))
                if not has_sl_order:
                    orders_to_create.append(('SL', sl_price, 'STOP_MARKET'))
                
                for order_type, price, binance_type in orders_to_create:
                    try:
                        # Final validation before placing order
                        price_diff_pct = abs(price - current_price) / current_price * 100
                        
                        if price_diff_pct < 0.1:  # Less than 0.1% difference
                            self.logger.warning(f"Skipping {order_type} for {position_side}: price ${price:.6f} too close to market ${current_price:.6f}")
                            continue
                        
                        # Validate using helper method
                        if not self._validate_tp_sl_price(side, order_type, price, current_price):
                            self.logger.error(f"Invalid {order_type} price for {side} position: ${price:.6f} vs market ${current_price:.6f}")
                            continue
                        
                        self.logger.info(f"Creating {order_type} for {position_side}: {expected_order_side.upper()} @ ${price:.6f} "
                                    f"(market: ${current_price:.6f}, entry: ${entry_price:.6f})")
                        
                        # Use EXISTING exchange methods - they already support hedge mode
                        try:
                            # Use the exchange's create_order method directly - it's already hedge-aware
                            symbol_id = self.exchange._get_symbol_id(self.symbol)
                            
                            # The exchange methods already include positionSide when hedge_mode_enabled=True
                            if binance_type == 'TAKE_PROFIT_MARKET':
                                # For take profit, use conditional order method
                                new_order = self.exchange.create_conditional_order(
                                    self.symbol,
                                    'take_profit_market',
                                    expected_order_side,
                                    quantity,
                                    price
                                )
                            else:  # STOP_MARKET
                                # For stop loss, use stop market order method
                                new_order = self.exchange.create_stop_market_order(
                                    self.symbol,
                                    expected_order_side,
                                    quantity,
                                    price
                                )
                        except Exception as create_error:
                            self.logger.error(f"TP/SL creation via existing methods failed: {create_error}")
                            continue
                        
                        if new_order and 'id' in new_order:
                            self.pending_orders[new_order['id']] = {
                                'type': expected_order_side,
                                'price': price,
                                'amount': quantity,
                                'timestamp': time.time(),
                                'order_purpose': 'profit' if order_type == 'TP' else 'stop_loss',
                                'is_tp_sl_order': True,
                                'position_side': position_side
                            }
                            
                            expected_pnl = 0
                            if order_type == 'TP':
                                if side == 'long':
                                    expected_pnl = (price - entry_price) * quantity
                                else:
                                    expected_pnl = (entry_price - price) * quantity
                            
                            self.logger.info(f"✅ {order_type} created for {position_side}: {new_order['id'][:8]} @ ${price:.6f}, "
                                        f"Expected PnL: ${expected_pnl:.2f}")
                            time.sleep(0.2)
                            
                    except Exception as e:
                        self.logger.error(f"{order_type} creation failed for {position_side}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error maintaining TP/SL orders: {e}")
    def _cleanup_orphaned_tp_sl_orders(self, live_orders: List[Dict]):
        """Clean up TP/SL orders when no position exists"""
        try:
            for order in live_orders:
                order_type = order.get('type', '').lower()
                order_id = order.get('id', '')
                
                # Check if this is a TP/SL order
                if order_type in ['take_profit_market', 'take_profit', 'take_profit_limit',
                                'stop_market', 'stop_loss', 'stop_loss_limit']:
                    try:
                        self.logger.info(f"Cleaning orphaned TP/SL order: {order_type} {order_id[:8]}")
                        self.exchange.cancel_order(order_id, self.symbol)
                        
                        # Remove from tracking if exists
                        if order_id in self.pending_orders:
                            del self.pending_orders[order_id]
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to cancel orphaned TP/SL {order_id[:8]}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error cleaning orphaned TP/SL orders: {e}")
    # def _create_counter_order_for_position(self, position: GridPosition, current_price: float):
    #     """Create profit-taking AND stop-loss orders with consistent 0.8% and GTC auto-cancel"""
    #     try:
    #         # Calculate order parameters based on position side (CONSISTENT 0.8%)
    #         if position.side == PositionSide.LONG.value:
    #             counter_side = OrderType.SELL.value
    #             # Profit target: 1% above entry
    #             profit_price = self._round_price(position.entry_price * 1.005)
    #             # Stop loss: 1% below entry
    #             stop_price = self._round_price(position.entry_price * 0.985)
    #         else:
    #             counter_side = OrderType.BUY.value
    #             # Profit target: 1% below entry
    #             profit_price = self._round_price(position.entry_price * 0.995)
    #             # Stop loss: 1% above entry
    #             stop_price = self._round_price(position.entry_price * 1.015)

    #         # Validate prices make sense
    #         if counter_side == OrderType.SELL.value:
    #             if profit_price <= position.entry_price or stop_price >= position.entry_price:
    #                 return
    #         else:
    #             if profit_price >= position.entry_price or stop_price <= position.entry_price:
    #                 return
            
    #         orders_placed = 0
            
    #         # 1. Place PROFIT order (TAKE_PROFIT_MARKET with GTC)
    #         try:
    #             self.logger.info(f"Creating TP order for position {position.position_id[:8]}")
                
    #             symbol_id = self.exchange._get_symbol_id(self.symbol)
    #             profit_order = self.exchange.exchange.create_order(
    #                 symbol=symbol_id,
    #                 type='TAKE_PROFIT_MARKET',
    #                 side=counter_side.upper(),
    #                 amount=position.quantity,
    #                 price=None,
    #                 params={
    #                     'stopPrice': profit_price, #'reduceOnly': True,
                        
    #                     'timeInForce': 'GTE_GTC'  # Auto-cancels when position closes
    #                 }
    #             )
                
    #             if profit_order and 'id' in profit_order:
    #                 profit_order_info = {
    #                     'type': counter_side,
    #                     'price': profit_price,
    #                     'amount': position.quantity,
    #                     'position_id': position.position_id,
    #                     'timestamp': time.time(),
    #                     'status': 'open',
    #                     'order_purpose': 'profit',
    #                     'is_tp_sl_order': True  # MARK as TP/SL order
    #                 }
    #                 self.pending_orders[profit_order['id']] = profit_order_info
    #                 orders_placed += 1
                    
    #                 expected_profit = abs(profit_price - position.entry_price) * position.quantity
    #                 self.logger.info(f"PROFIT (TP): {counter_side.upper()} {position.quantity:.6f} @ ${profit_price:.6f}, Expected: ${expected_profit:.2f}")
            
    #         except Exception as e:
    #             self.logger.error(f"Failed to place profit order: {e}")
            
    #         # 2. Place STOP-LOSS order (STOP_MARKET with GTC)
    #         try:
    #             symbol_id = self.exchange._get_symbol_id(self.symbol)
    #             sl_order = self.exchange.exchange.create_order(
    #                 symbol=symbol_id,
    #                 type='STOP_MARKET',
    #                 side=counter_side.upper(),
    #                 amount=position.quantity,
    #                 price=None,
    #                 params={
    #                     'stopPrice': stop_price, #'reduceOnly': True,                        
    #                     'timeInForce': 'GTE_GTC'  # Auto-cancels when position closes
    #                 }
    #             )
            
    #             if sl_order and 'id' in sl_order:
    #                 sl_order_info = {
    #                     'type': counter_side,
    #                     'price': stop_price,
    #                     'amount': position.quantity,
    #                     'position_id': position.position_id,
    #                     'timestamp': time.time(),
    #                     'status': 'open',
    #                     'order_purpose': 'stop_loss',
    #                     'is_tp_sl_order': True  # MARK as TP/SL order
    #                 }
    #                 self.pending_orders[sl_order['id']] = sl_order_info
    #                 orders_placed += 1
                    
    #                 expected_loss = abs(stop_price - position.entry_price) * position.quantity
    #                 self.logger.info(f"STOP-LOSS (SL): {counter_side.upper()} {position.quantity:.6f} @ ${stop_price:.6f}, Max loss: ${expected_loss:.2f}")
            
    #         except Exception as e:
    #             self.logger.error(f"Failed to place stop-loss order: {e}")
            
    #         # Update position if at least one order was placed
    #         if orders_placed > 0:
    #             position.has_counter_order = True
    #             self.logger.info(f"TP/SL orders created: {orders_placed} orders with GTC auto-cancel")
            
    #     except Exception as e:
    #         self.logger.error(f"Error creating counter orders for position {position.position_id}: {e}")
        
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
        """ENHANCED: Get comprehensive status including hedge position info"""
        try:
            with self.update_lock:
                # Get current market data
                market_data = self._get_live_market_data()
                
                # Count open positions (including hedge breakdown)
                open_positions = [pos for pos in self.all_positions.values() if pos.is_open()]
                closed_positions = [pos for pos in self.all_positions.values() if not pos.is_open()]
                
                # Calculate PnL percentage
                pnl_percentage = (self.total_pnl / self.user_total_investment * 100) if self.user_total_investment > 0 else 0.0
                
                # Check if hedge mode is active
                is_hedge_mode = market_data.get('long_positions', 0) > 0 and market_data.get('short_positions', 0) > 0
                
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
                    
                    # Hedge mode info (NEW)
                    'is_hedge_mode': is_hedge_mode,
                    'long_positions': market_data.get('long_positions', 0),
                    'short_positions': market_data.get('short_positions', 0),
                    
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