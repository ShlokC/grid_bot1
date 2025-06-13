"""
Grid Trading Strategy - Counter Orders Removed
Author: Grid Trading Bot
Date: 2025-05-26

Complete rewrite removing zone complexity and artificial restrictions.
Uses real KAMA instead of fake momentum, direct order placement within user range.
UPDATED: Removed all counter order logic while preserving TP/SL functionality.
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
    Simplified Grid Trading Strategy without Zone complexity and Counter Orders
    
    Key Features:
    - Direct order placement within user-defined range
    - Real KAMA-based market intelligence
    - Simple investment tracking (position + order margins)
    - TP/SL orders only (no counter orders)
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
        
        # ADDED: Track order placement timing to avoid stale position data
        self._last_order_placement_time = 0
        
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
        FIXED: Enhanced logging for initial grid setup
        """
        try:
            # Get current market price
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # ENHANCED: Clear setup initiation logging
            self.logger.info(f"🚀 INITIATING GRID SETUP: Current price ${current_price:.6f}, Range ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}")
            
            # Check if current price is within user range
            if current_price < self.user_price_lower or current_price > self.user_price_upper:
                self.logger.warning(f"⚠️ PRICE OUTSIDE RANGE: Current ${current_price:.6f} outside ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}")
            
            # Cancel existing orders (keep existing logic for safety)
            try:
                open_orders = self.exchange.get_open_orders(self.symbol)
                if open_orders:
                    live_positions = self.exchange.get_positions(self.symbol)
                    cancelled_count = 0
                    
                    self.logger.info(f"🧹 CLEANING EXISTING ORDERS: Found {len(open_orders)} open orders")
                    
                    for order in open_orders:
                        if self._should_protect_order(order, current_price, live_positions):
                            continue
                        
                        try:
                            order_id = order.get('id', 'unknown')
                            self.logger.info(f"🗑️ CANCELLING ORDER: {order_id[:8]}")
                            self.exchange.cancel_order(order_id, self.symbol)
                            cancelled_count += 1
                        except Exception as e:
                            self.logger.warning(f"⚠️ CANCEL FAILED: Order {order.get('id', 'unknown')[:8]} - {e}")
                    
                    if cancelled_count > 0:
                        self.logger.info(f"✅ ORDER CLEANUP COMPLETE: Cancelled {cancelled_count} orders")
                        time.sleep(1)
                            
            except Exception as e:
                self.logger.warning(f"⚠️ ORDER CLEANUP ERROR: {e}")
            
            # Place initial market order if no position exists
            live_positions = self.exchange.get_positions(self.symbol)
            has_position = any(abs(float(pos.get('contracts', 0))) >= 0.001 for pos in live_positions)
            
            if not has_position:
                self.logger.info(f"🎯 NO POSITIONS FOUND: Placing initial market order")
                if self._place_single_order(current_price, 'buy'):  # Side ignored for market orders
                    self.running = True
                    self.logger.info("✅ GRID SETUP COMPLETE: Initial market order placed, grid is now active")
                    self._grid_start_time = time.time()
                else:
                    self.logger.error("❌ GRID SETUP FAILED: Initial market order not placed")
                    self.running = False
            else:
                # Position already exists, just start the grid
                self.running = True
                position_info = []
                for pos in live_positions:
                    if abs(float(pos.get('contracts', 0))) >= 0.001:
                        side = pos.get('side', '').upper()
                        size = abs(float(pos.get('contracts', 0)))
                        price = float(pos.get('entryPrice', 0))
                        position_info.append(f"{side} {size:.6f} @ ${price:.6f}")
                
                self.logger.info(f"✅ GRID SETUP COMPLETE: Existing positions found ({', '.join(position_info)}), grid is now active")
                
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR in grid setup: {e}")
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
                time.sleep(1)  # Allow time for order to process
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
        UPDATED: Now calls individual side placement for better control
        """
        try:
            # For compatibility, if this method is called, place both sides
            # but use the new individual placement logic
            
            self.logger.info(f"🚀 PLACING INITIAL HEDGE POSITIONS: Both LONG and SHORT")
            
            current_time = time.time()
            self._last_order_placement_time = current_time
            
            # Place LONG position
            long_success = self._place_individual_side_order(price, 'long')
            
            # Wait between orders
            time.sleep(2.0)
            
            # Place SHORT position  
            short_success = self._place_individual_side_order(price, 'short')
            
            if long_success and short_success:
                self.logger.info(f"🎯 HEDGE STRATEGY DEPLOYED: Both LONG and SHORT positions created")
                return True
            elif long_success or short_success:
                created = "LONG" if long_success else "SHORT"
                failed = "SHORT" if long_success else "LONG"
                self.logger.warning(f"⚠️ PARTIAL HEDGE: {created} created, {failed} failed")
                return True  # Partial success is still success
            else:
                self.logger.error(f"❌ HEDGE STRATEGY FAILED: Both positions failed to create")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in hedge order placement: {e}")
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
        SIMPLIFIED: Main grid update loop without over-engineered gap filling.
        """
        try:
            with self.update_lock:
                if not self.running:
                    return
                
                # Get current market price
                ticker = self.exchange.get_ticker(self.symbol)
                current_price = float(ticker['last'])
                
                # 1. Update filled orders and positions
                self._update_orders_and_positions()
                
                # 2. SIMPLIFIED: Maintain coverage (range adaptation + ensure positions)
                self._maintain_grid_coverage(current_price)
                
                # 3. Maintain TP/SL orders only
                self._maintain_tp_sl_orders_only(current_price)
                
                # 4. Update PnL calculations
                self._update_pnl(current_price)
                
                # 5. Check take profit and stop loss
                self._check_tp_sl()
                
                self.last_update_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
    def _maintain_grid_coverage(self, current_price: float):
        """
        FIXED: Updated to use side-specific entry logic
        """
        try:
            # 1. Range adaptation (keep existing logic)
            if current_price < self.user_price_lower or current_price > self.user_price_upper:
                if self.enable_grid_adaptation:
                    range_size = self.user_price_upper - self.user_price_lower
                    
                    if current_price < self.user_price_lower:
                        new_upper = self.user_price_lower + (range_size * 0.1)
                        new_lower = new_upper - range_size
                    else:
                        new_lower = self.user_price_upper - (range_size * 0.1)
                        new_upper = new_lower + range_size
                    
                    old_lower, old_upper = self.user_price_lower, self.user_price_upper
                    self.user_price_lower = self._round_price(new_lower)
                    self.user_price_upper = self._round_price(new_upper)
                    
                    self.logger.info(f"Range adapted: ${old_lower:.6f}-${old_upper:.6f} → ${self.user_price_lower:.6f}-${self.user_price_upper:.6f}")
            
            # 2. FIXED: Side-specific position management
            self._maybe_place_initial_order(current_price)
            
        except Exception as e:
            self.logger.error(f"Error maintaining grid coverage: {e}")
    def _maybe_place_initial_order(self, current_price: float):
        """
        FIXED: Side-specific position checking and entry logic
        """
        try:
            # Throttle: Don't place orders too frequently  
            current_time = time.time()
            last_order_time = getattr(self, '_last_order_placement_time', 0)
            
            if current_time - last_order_time < 30.0:  # 30 second cooldown
                return
            
            # FIXED: Check for LONG and SHORT positions separately
            live_positions = self.exchange.get_positions(self.symbol)
            
            has_long_position = False
            has_short_position = False
            min_meaningful_notional = 1.0  # $1+ notional = meaningful
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                
                if abs(size) > 0.001 and entry_price > 0:
                    notional = abs(size) * entry_price
                    if notional >= min_meaningful_notional:
                        if side == 'long':
                            has_long_position = True
                        elif side == 'short':
                            has_short_position = True
            
            # FIXED: Enter missing sides individually
            missing_sides = []
            if not has_long_position:
                missing_sides.append('long')
            if not has_short_position:
                missing_sides.append('short')
            
            if not missing_sides:
                self.logger.debug("✅ Both LONG and SHORT positions exist, no action needed")
                return
            
            # Check funds for missing positions
            market_data = self._get_live_market_data()
            investment_per_side = self.user_investment_per_grid * 0.5  # 50% each for hedge
            total_investment_needed = len(missing_sides) * investment_per_side
            
            if market_data['available_investment'] < total_investment_needed:
                missing_types = ["LONG" if side == 'long' else "SHORT" for side in missing_sides]
                self.logger.info(f"💰 Insufficient funds for missing {missing_types}: "
                            f"Need ${total_investment_needed:.2f}, Available: ${market_data['available_investment']:.2f}")
                return
            
            # FIXED: Place orders for missing sides only
            missing_types = ["LONG" if side == 'long' else "SHORT" for side in missing_sides]
            self.logger.info(f"🎯 MISSING POSITIONS DETECTED: {missing_types}, placing individual orders")
            
            self._last_order_placement_time = current_time
            
            for side in missing_sides:
                if self._place_individual_side_order(current_price, side):
                    position_type = "LONG" if side == 'long' else "SHORT"
                    self.logger.info(f"✅ {position_type} position created successfully")
                else:
                    position_type = "LONG" if side == 'long' else "SHORT"
                    self.logger.error(f"❌ Failed to create {position_type} position")
                    
        except Exception as e:
            self.logger.error(f"Error in side-specific position check: {e}")

    def _place_individual_side_order(self, current_price: float, side: str) -> bool:
        """
        FIXED: Place order for individual side (LONG or SHORT) instead of both
        """
        try:
            # Calculate investment and amount for this side
            investment_for_side = self.user_investment_per_grid * 0.5  # 50% for hedge strategy
            amount = self._calculate_order_amount_for_investment(current_price, investment_for_side)
            
            # Determine order side
            order_side = 'buy' if side == 'long' else 'sell'
            position_type = "LONG" if side == 'long' else "SHORT"
            
            # Validate order parameters
            if amount < self.min_amount:
                self.logger.error(f"❌ {position_type} order amount {amount:.6f} below minimum {self.min_amount}")
                return False
            
            notional_value = current_price * amount
            margin_required = round(notional_value / self.user_leverage, 2)
            
            if notional_value < self.min_cost:
                self.logger.error(f"❌ {position_type} order notional ${notional_value:.2f} below minimum ${self.min_cost}")
                return False
            
            # Get market trend for context (optional)
            jma_trend = 'neutral'
            if self.enable_samig and self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    jma_trend = market_snapshot.jma_trend
                except Exception:
                    pass
            
            # FIXED: Place single side order with clear logging
            self.logger.info(f"🚀 PLACING INDIVIDUAL {position_type} ORDER: {amount:.6f} {self.symbol} @ MARKET "
                            f"(${notional_value:.2f} notional, ${margin_required:.2f} margin, JMA: {jma_trend.upper()})")
            
            # Use existing exchange method
            order = self.exchange.create_market_order(self.symbol, order_side, amount)
            
            if not order or 'id' not in order:
                self.logger.error(f"❌ {position_type} ORDER FAILED: Invalid response from exchange")
                return False
            
            fill_price = float(order.get('average', current_price))
            if fill_price == 0:
                fill_price = current_price
            
            # Success logging
            self.logger.info(f"✅ INDIVIDUAL {position_type} ORDER FILLED: {amount:.6f} @ ${fill_price:.6f}, "
                            f"Margin: ${margin_required:.2f}, ID: {order['id'][:8]}")
            
            return True
            
        except Exception as e:
            position_type = "LONG" if side == 'long' else "SHORT"
            self.logger.error(f"❌ Error placing individual {position_type} order: {e}")
            return False
    def _should_place_new_orders(self, intended_side: str = None) -> bool:
        """
        FIXED: Updated to handle hedge positioning with proper position size validation.
        Now considers actual investment thresholds instead of arbitrary small amounts.
        """
        try:
            live_positions = self.exchange.get_positions(self.symbol)
            
            # Calculate expected position thresholds based on actual investment
            expected_investment_per_side = self.user_investment_per_grid * 0.5  # 50% each for hedge
            expected_notional_per_side = expected_investment_per_side * self.user_leverage
            
            # CRITICAL: Use meaningful threshold - 80% of expected notional value
            MIN_MEANINGFUL_NOTIONAL = expected_notional_per_side * 0.8
            
            # Count meaningful positions by side
            meaningful_long_positions = 0
            meaningful_short_positions = 0
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                    
                # Calculate actual notional value
                actual_notional = abs(size) * entry_price
                
                # Only count positions that meet meaningful threshold
                if actual_notional >= MIN_MEANINGFUL_NOTIONAL:
                    if side == 'long':
                        meaningful_long_positions += 1
                    elif side == 'short':
                        meaningful_short_positions += 1
            
            # HEDGE STRATEGY: If SAMIG enabled, allow orders when missing meaningful hedge positions
            if self.enable_samig:
                if intended_side:
                    if intended_side == 'long' and meaningful_long_positions == 0:
                        self.logger.debug(f"Should place LONG: No meaningful LONG positions (threshold: ${MIN_MEANINGFUL_NOTIONAL:.2f})")
                        return True
                    elif intended_side == 'short' and meaningful_short_positions == 0:
                        self.logger.debug(f"Should place SHORT: No meaningful SHORT positions (threshold: ${MIN_MEANINGFUL_NOTIONAL:.2f})")
                        return True
                    else:
                        self.logger.debug(f"Should NOT place {intended_side.upper()}: Meaningful position already exists")
                        return False
                else:
                    # No specific side - allow if any meaningful position is missing
                    missing_sides = []
                    if meaningful_long_positions == 0:
                        missing_sides.append("LONG")
                    if meaningful_short_positions == 0:
                        missing_sides.append("SHORT")
                    
                    if missing_sides:
                        self.logger.debug(f"Should place orders: Missing meaningful {missing_sides} positions")
                        return True
                    else:
                        self.logger.debug(f"Should NOT place orders: Both meaningful LONG and SHORT positions exist")
                        return False
            
            # STANDARD STRATEGY: Only allow new orders if no meaningful positions exist
            has_meaningful_positions = meaningful_long_positions > 0 or meaningful_short_positions > 0
            
            if not has_meaningful_positions:
                self.logger.debug(f"Should place orders: No meaningful positions found (threshold: ${MIN_MEANINGFUL_NOTIONAL:.2f})")
            else:
                self.logger.debug(f"Should NOT place orders: Meaningful positions exist (LONG: {meaningful_long_positions}, SHORT: {meaningful_short_positions})")
            
            return not has_meaningful_positions
            
        except Exception as e:
            self.logger.error(f"❌ Error checking if should place new orders: {e}")
            return False


    def _fill_grid_gaps(self, current_price: float, market_data: Dict[str, Any]):
        """
        FIXED: Handle initial position creation with proper timing controls to prevent race conditions.
        Now waits for order execution and position updates before making new decisions.
        """
        try:
            # CRITICAL: Check if orders were placed recently - wait for position data to update
            current_time = time.time()
            
            # Check multiple timing markers to prevent race conditions
            last_order_time = getattr(self, '_last_order_placement_time', 0)
            grid_start_time = getattr(self, '_grid_start_time', 0)
            initial_order_placed = getattr(self, '_initial_order_placed', False)
            
            # ENHANCED: Wait longer after ANY order placement for position data to update
            time_since_last_order = current_time - last_order_time
            time_since_grid_start = current_time - grid_start_time
            
            # If any order was placed recently, wait for position data refresh
            if last_order_time > 0 and time_since_last_order < 15.0:  # Wait 15 seconds after any order
                self.logger.debug(f"⏰ Waiting for position data after recent order: {time_since_last_order:.1f}s elapsed (need 15s)")
                return
                
            # If grid just started, wait for initial positions to settle
            if initial_order_placed and grid_start_time > 0 and time_since_grid_start < 30.0:
                self.logger.debug(f"⏰ Waiting for initial positions to settle: {time_since_grid_start:.1f}s elapsed (need 30s)")
                return
            
            # CRITICAL: Force fresh position data before making decisions
            try:
                live_positions = self.exchange.get_positions(self.symbol)
                # Small delay to ensure we get the latest data
                time.sleep(0.5)
                live_positions = self.exchange.get_positions(self.symbol)  # Double-check for fresh data
            except Exception as e:
                self.logger.error(f"❌ Failed to get fresh position data: {e}")
                return
            
            # Calculate expected position thresholds based on actual investment
            expected_investment_per_side = self.user_investment_per_grid * 0.5  # 50% each for hedge
            expected_notional_per_side = expected_investment_per_side * self.user_leverage
            
            # CRITICAL: Use meaningful threshold - 80% of expected notional value
            MIN_MEANINGFUL_NOTIONAL = expected_notional_per_side * 0.8
            
            # Validate existing positions against expected sizes
            has_meaningful_long = False
            has_meaningful_short = False
            long_position_info = ""
            short_position_info = ""
            
            self.logger.debug(f"🔍 Position validation (fresh data): Expected notional per side: ${expected_notional_per_side:.2f}, "
                            f"Minimum meaningful: ${MIN_MEANINGFUL_NOTIONAL:.2f}")
            
            for pos in live_positions:
                size = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                side = pos.get('side', '').lower()
                
                if abs(size) < 0.001 or entry_price <= 0:
                    continue
                    
                # Calculate actual notional value
                actual_notional = abs(size) * entry_price
                
                # Check if position meets meaningful threshold
                if actual_notional >= MIN_MEANINGFUL_NOTIONAL:
                    if side == 'long':
                        has_meaningful_long = True
                        long_position_info = f"LONG {size:.6f} @ ${entry_price:.6f} (${actual_notional:.2f} notional)"
                    elif side == 'short':
                        has_meaningful_short = True
                        short_position_info = f"SHORT {abs(size):.6f} @ ${entry_price:.6f} (${actual_notional:.2f} notional)"
                else:
                    # Log dust positions that don't meet threshold
                    dust_info = f"{side.upper()} {abs(size):.6f} @ ${entry_price:.6f} (${actual_notional:.2f} dust)"
                    self.logger.debug(f"💸 Ignoring dust position: {dust_info} (below ${MIN_MEANINGFUL_NOTIONAL:.2f} threshold)")
            
            # Check available investment and order capacity
            if (market_data['available_investment'] < self.user_investment_per_grid or 
                market_data['total_commitment'] >= self.max_total_orders):
                if has_meaningful_long or has_meaningful_short:
                    self.logger.debug(f"✅ Existing meaningful positions found, capacity constraints OK")
                return
            
            # ENHANCED: Log current position status with timing info
            position_status = []
            if has_meaningful_long:
                position_status.append(long_position_info)
            if has_meaningful_short:
                position_status.append(short_position_info)
                
            if position_status:
                self.logger.info(f"📊 EXISTING MEANINGFUL POSITIONS (verified fresh): {' | '.join(position_status)}")
            
            # HEDGE STRATEGY: Create missing positions with proper timing controls
            if self.enable_samig:
                # Calculate missing positions based on meaningful positions
                missing_positions = []
                if not has_meaningful_long:
                    missing_positions.append('long')
                if not has_meaningful_short:
                    missing_positions.append('short')
                
                if missing_positions:
                    # Calculate total investment needed for missing positions
                    investment_per_missing_position = expected_investment_per_side
                    total_investment_needed = len(missing_positions) * investment_per_missing_position
                    
                    if market_data['available_investment'] >= total_investment_needed:
                        missing_types = ["LONG" if side == 'long' else "SHORT" for side in missing_positions]
                        self.logger.info(f"🎯 CREATING MISSING HEDGE POSITIONS: {missing_types} "
                                    f"(${investment_per_missing_position:.2f} each, ${total_investment_needed:.2f} total)")
                        
                        # CRITICAL: Set timing marker BEFORE placing orders
                        self._last_order_placement_time = current_time
                        
                        self._place_missing_hedge_positions(current_price, missing_positions)
                        return
                    else:
                        self.logger.info(f"💰 Insufficient funds for missing hedge positions: "
                                    f"Need ${total_investment_needed:.2f}, Available: ${market_data['available_investment']:.2f}")
                        return
                else:
                    self.logger.debug(f"✅ HEDGE COMPLETE: Both meaningful LONG and SHORT positions exist")
                    return
            
            # STANDARD STRATEGY: Only create position if NO meaningful positions exist
            if not has_meaningful_long and not has_meaningful_short:
                # For hedge positions, we need full investment (50% + 50% = 100%)
                total_investment_needed = self.user_investment_per_grid
                
                if market_data['available_investment'] >= total_investment_needed:
                    self.logger.info(f"🚀 CREATING INITIAL HEDGE POSITIONS: No meaningful positions found, "
                                f"placing initial orders (${total_investment_needed:.2f} total investment)")
                    
                    # CRITICAL: Set timing markers BEFORE placing orders
                    self._last_order_placement_time = current_time
                    self._grid_start_time = current_time
                    
                    # Place initial hedge orders
                    if self._place_single_order(current_price, 'buy'):  # Side ignored for market orders
                        self.logger.info("✅ Initial hedge positions creation initiated")
                else:
                    self.logger.info(f"💰 Insufficient funds for initial hedge positions: "
                                f"Need ${total_investment_needed:.2f}, Available: ${market_data['available_investment']:.2f}")
            else:
                existing_info = []
                if has_meaningful_long:
                    existing_info.append(long_position_info)
                if has_meaningful_short:
                    existing_info.append(short_position_info)
                self.logger.debug(f"✅ Meaningful positions exist (verified): {' | '.join(existing_info)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in _fill_grid_gaps: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    def _place_missing_hedge_positions(self, current_price: float, missing_sides: List[str]):
        """
        FIXED: Place missing hedge positions using predetermined investment allocation
        instead of matching existing position sizes.
        """
        try:
            # CRITICAL: Update timing marker to prevent concurrent execution
            current_time = time.time()
            self._last_order_placement_time = current_time
            
            # Get market conditions for side determination
            jma_trend = 'neutral'
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    jma_trend = market_snapshot.jma_trend
                except Exception as e:
                    self.logger.warning(f"Market analysis failed: {e}")
            
            # FIXED: Use predetermined investment allocation instead of matching position sizes
            if self.enable_samig:
                # For hedge mode: split investment per grid level (50/50)
                investment_per_missing_position = self.user_investment_per_grid * 0.5
            else:
                # For standard mode: use full investment per grid level  
                investment_per_missing_position = self.user_investment_per_grid
            
            # CLEAR HEDGE COMPLETION LOGGING
            missing_position_types = ["LONG" if side == 'long' else "SHORT" for side in missing_sides]
            self.logger.info(f"🔄 COMPLETING HEDGE STRATEGY: Missing {missing_position_types} positions, "
                            f"Investment per position: ${investment_per_missing_position:.2f}, JMA: {jma_trend.upper()}")
            
            placed_orders = []  # Track successfully placed orders
            
            for side in missing_sides:
                try:
                    # Map side to position type for clear logging
                    position_type = "LONG" if side == 'long' else "SHORT"
                    order_side = 'buy' if side == 'long' else 'sell'
                    
                    # FIXED: Calculate amount based on predetermined investment, not existing position size
                    amount = self._calculate_order_amount_for_investment(current_price, investment_per_missing_position)
                    
                    # Validate order parameters
                    if amount < self.min_amount:
                        self.logger.error(f"❌ {position_type} hedge order amount {amount:.6f} below minimum {self.min_amount}")
                        continue
                    
                    notional_value = current_price * amount
                    margin_required = round(notional_value / self.user_leverage, 2)
                    
                    # Verify this matches expected investment allocation
                    expected_notional = investment_per_missing_position * self.user_leverage
                    allocation_match = abs(notional_value - expected_notional) / expected_notional < 0.05  # 5% tolerance
                    
                    if not allocation_match:
                        self.logger.warning(f"⚠️ Order size deviation: Expected ${expected_notional:.2f}, "
                                        f"Calculated ${notional_value:.2f}")
                    
                    # CLEAR MISSING HEDGE ORDER LOGGING
                    self.logger.info(f"🚀 PLACING MISSING {position_type} HEDGE ORDER: {amount:.6f} {self.symbol} "
                                    f"@ MARKET (${notional_value:.2f} notional, ${margin_required:.2f} margin, "
                                    f"${investment_per_missing_position:.2f} investment)")
                    
                    # Use existing exchange method (already supports hedge mode)
                    order = self.exchange.create_market_order(self.symbol, order_side, amount)
                    
                    if order and 'id' in order:
                        fill_price = float(order.get('average', current_price))
                        if fill_price == 0:
                            fill_price = current_price
                        
                        # Track successful order
                        placed_orders.append({
                            'id': order['id'],
                            'side': position_type,
                            'amount': amount,
                            'price': fill_price,
                            'investment_used': investment_per_missing_position
                        })
                        
                        # CLEAR SUCCESS LOGGING WITH INVESTMENT TRACKING
                        self.logger.info(f"✅ MISSING {position_type} HEDGE ORDER FILLED: {amount:.6f} @ ${fill_price:.6f}, "
                                    f"Margin: ${margin_required:.2f}, Investment: ${investment_per_missing_position:.2f}, "
                                    f"ID: {order['id'][:8]} - HEDGE COMPLETE")
                        
                    else:
                        self.logger.error(f"❌ Failed to create missing {position_type} hedge order - invalid response")
                    
                    # CRITICAL: Wait between orders for proper execution
                    if len(missing_sides) > 1:
                        self.logger.debug(f"⏰ Waiting between hedge orders...")
                        time.sleep(2.0)  # Wait between multiple orders
                    
                except Exception as e:
                    self.logger.error(f"❌ Error creating missing {position_type} hedge order: {e}")
            
            # CRITICAL: Wait for all orders to be processed and verify execution
            if placed_orders:
                self.logger.debug(f"⏰ Waiting for {len(placed_orders)} hedge orders to be processed...")
                time.sleep(3.0)  # Wait for position updates
                
                # VERIFY order execution by checking positions
                try:
                    verification_positions = self.exchange.get_positions(self.symbol)
                    current_position_count = sum(1 for pos in verification_positions if abs(float(pos.get('contracts', 0))) >= 0.001)
                    
                    # Calculate total investment used for verification
                    total_investment_used = sum(order['investment_used'] for order in placed_orders)
                    
                    self.logger.info(f"✅ HEDGE COMPLETION VERIFICATION: {current_position_count} total positions, "
                                    f"${total_investment_used:.2f} additional investment used")
                    
                    # Log individual position verification
                    for pos in verification_positions:
                        size = float(pos.get('contracts', 0))
                        if abs(size) >= 0.001:
                            side = pos.get('side', '').lower()
                            entry_price = float(pos.get('entryPrice', 0))
                            position_type = "LONG" if side == 'long' else "SHORT"
                            notional = abs(size) * entry_price
                            margin = notional / self.user_leverage
                            self.logger.debug(f"  📊 Verified: {position_type} {abs(size):.6f} @ ${entry_price:.6f} "
                                            f"(${notional:.2f} notional, ${margin:.2f} margin)")
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not verify hedge completion: {e}")
            
            # FINAL HEDGE STATUS SUMMARY with investment tracking
            self._log_final_hedge_status_with_investment(missing_sides, jma_trend, placed_orders)
            
        except Exception as e:
            self.logger.error(f"❌ Error placing missing hedge positions: {e}")

    def _log_hedge_completion_status(self, live_positions: List[Dict], missing_sides: List[str], completed_side: str):
        """Helper function to log hedge completion status"""
        try:
            completed_position_type = "LONG" if completed_side == 'long' else "SHORT"
            remaining_missing = [side for side in missing_sides if side != completed_side]
            
            if not remaining_missing:
                self.logger.info(f"🎯 HEDGE STRATEGY FULLY DEPLOYED: Both LONG and SHORT positions now active")
            else:
                remaining_types = ["LONG" if side == 'long' else "SHORT" for side in remaining_missing]
                self.logger.info(f"🔄 HEDGE PROGRESS: {completed_position_type} completed, still missing: {remaining_types}")
                
        except Exception as e:
            self.logger.error(f"Error logging hedge completion status: {e}")

    def _log_final_hedge_status(self, missing_sides: List[str], jma_trend: str):
        """Helper function to log final hedge deployment summary"""
        try:
            completed_types = ["LONG" if side == 'long' else "SHORT" for side in missing_sides]
            
            if len(completed_types) == 1:
                self.logger.info(f"🎯 HEDGE DEPLOYMENT COMPLETE: Added {completed_types[0]} position (JMA: {jma_trend.upper()})")
            else:
                self.logger.info(f"🎯 HEDGE DEPLOYMENT COMPLETE: Added {' + '.join(completed_types)} positions (JMA: {jma_trend.upper()})")
                
        except Exception as e:
            self.logger.error(f"Error logging final hedge status: {e}")
    
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
            
        except Exception as e:
            self.logger.error(f"Error closing position from counter order: {e}")
    
    def _maintain_tp_sl_orders_only(self, current_price: float):
        """
        UPDATED: Only maintain TP/SL orders with fresh position data check
        """
        try:
            # CRITICAL: Check if we just placed orders recently to avoid stale data
            current_time = time.time()
            if hasattr(self, '_last_order_placement_time'):
                time_since_last_order = current_time - self._last_order_placement_time
                if time_since_last_order < 5.0:  # Wait 5 seconds after placing orders
                    self.logger.debug(f"Waiting for fresh position data ({time_since_last_order:.1f}s since last order)")
                    return
            
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
            
            # ONLY TP/SL orders using EXISTING exchange methods
            self._maintain_tp_sl_orders(real_positions, live_orders)
            
        except Exception as e:
            self.logger.error(f"Error in TP/SL orders maintenance: {e}")

    def _maintain_tp_sl_orders(self, live_positions: List[Dict], live_orders: List[Dict]):
        """
        FIXED: Enhanced logging for TP/SL order creation
        """
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
                
                # Calculate TP/SL prices based on CURRENT MARKET PRICE (not entry price)
                if side == 'long':
                    tp_price = self._round_price(max(current_price * 1.002, entry_price * 1.005))
                    sl_price = self._round_price(min(current_price * 0.995, entry_price * 0.97))
                    expected_order_side = 'sell'
                else:  # short
                    tp_price = self._round_price(min(current_price * 0.998, entry_price * 0.995))
                    sl_price = self._round_price(max(current_price * 1.005, entry_price * 1.03))
                    expected_order_side = 'buy'
                
                # Validate TP/SL prices won't trigger immediately
                if side == 'long':
                    if tp_price <= current_price:
                        tp_price = self._round_price(current_price * 1.003)
                    if sl_price >= current_price:
                        sl_price = self._round_price(current_price * 0.993)
                else:  # short
                    if tp_price >= current_price:
                        tp_price = self._round_price(current_price * 0.997)
                    if sl_price <= current_price:
                        sl_price = self._round_price(current_price * 1.007)
                
                # Final sanity check on calculated prices
                tp_valid = self._validate_tp_sl_price(side, 'TP', tp_price, current_price)
                sl_valid = self._validate_tp_sl_price(side, 'SL', sl_price, current_price)
                
                if not tp_valid:
                    self.logger.error(f"❌ INVALID TP PRICE: {side.upper()} TP ${tp_price:.6f} invalid vs market ${current_price:.6f}")
                    continue
                    
                if not sl_valid:
                    self.logger.error(f"❌ INVALID SL PRICE: {side.upper()} SL ${sl_price:.6f} invalid vs market ${current_price:.6f}")
                    continue
                
                # Count existing TP/SL orders for THIS specific position
                tp_orders_for_position = 0
                sl_orders_for_position = 0
                
                for order in live_orders:
                    order_type = order.get('type', '').lower()
                    order_side = order.get('side', '').lower()
                    order_price = self._get_order_price(order)
                    
                    if order_side == expected_order_side:
                        is_tp_order = order_type in ['take_profit_market', 'take_profit', 'take_profit_limit']
                        is_sl_order = order_type in ['stop_market', 'stop_loss', 'stop_loss_limit']
                        
                        if is_tp_order and order_price > 0:
                            if side == 'long' and order_price > entry_price:
                                tp_orders_for_position += 1
                            elif side == 'short' and order_price < entry_price:
                                tp_orders_for_position += 1
                        elif is_sl_order and order_price > 0:
                            if side == 'long' and order_price < entry_price:
                                sl_orders_for_position += 1
                            elif side == 'short' and order_price > entry_price:
                                sl_orders_for_position += 1
                
                # Create missing TP/SL orders for this position side
                has_tp_order = tp_orders_for_position > 0
                has_sl_order = sl_orders_for_position > 0
                
                if has_tp_order and has_sl_order:
                    self.logger.debug(f"✅ Position {position_side} ({side.upper()} @ ${entry_price:.6f}): TP/SL orders exist")
                    continue
                
                orders_to_create = []
                if not has_tp_order:
                    orders_to_create.append(('TP', tp_price, 'TAKE_PROFIT_MARKET'))
                if not has_sl_order:
                    orders_to_create.append(('SL', sl_price, 'STOP_MARKET'))
                
                for order_type, price, binance_type in orders_to_create:
                    try:
                        # Final validation before placing order
                        price_diff_pct = abs(price - current_price) / current_price * 100
                        
                        if price_diff_pct < 0.1:
                            self.logger.warning(f"⚠️ BLOCKING {order_type}: Price ${price:.6f} too close to market ${current_price:.6f} ({price_diff_pct:.3f}%)")
                            continue
                        
                        # ENHANCED: Pre-creation logging
                        self.logger.info(f"🚀 PLACING {order_type} ORDER: {expected_order_side.upper()} {quantity:.6f} {self.symbol} @ ${price:.6f} (for {position_side} {side.upper()} position)")
                        
                        # Use EXISTING exchange methods - they already support hedge mode
                        try:
                            if binance_type == 'TAKE_PROFIT_MARKET':
                                new_order = self.exchange.create_conditional_order(
                                    self.symbol,
                                    'take_profit_market',
                                    expected_order_side,
                                    quantity,
                                    price
                                )
                            else:  # STOP_MARKET
                                new_order = self.exchange.create_stop_market_order(
                                    self.symbol,
                                    expected_order_side,
                                    quantity,
                                    price
                                )
                        except Exception as create_error:
                            self.logger.error(f"❌ {order_type} ORDER PLACEMENT FAILED: {create_error}")
                            continue
                        
                        if new_order and 'id' in new_order:
                            # Track TP/SL order placement time
                            self._last_order_placement_time = time.time()
                            
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
                            
                            # ENHANCED: Success confirmation logging
                            self.logger.info(f"✅ {order_type} ORDER CREATED: {expected_order_side.upper()} @ ${price:.6f} for {position_side} {side.upper()}, "
                                        f"Expected PnL: ${expected_pnl:.2f}, ID: {new_order['id'][:8]}")
                            time.sleep(0.2)
                        else:
                            self.logger.error(f"❌ {order_type} ORDER CREATION FAILED: Invalid response for {position_side} {side.upper()} position")
                            
                    except Exception as e:
                        self.logger.error(f"❌ {order_type} ORDER CREATION ERROR for {position_side} {side.upper()}: {e}")
                        
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR in TP/SL orders maintenance: {e}")

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
        """
        FIXED: Enhanced logging for position closure orders
        """
        try:
            # Determine order side to close position
            close_side = OrderType.SELL.value if position.side == PositionSide.LONG.value else OrderType.BUY.value
            
            # ENHANCED: Pre-closure logging
            self.logger.info(f"🚀 PLACING MARKET CLOSURE ORDER: {close_side.upper()} {position.quantity:.6f} {self.symbol} @ MARKET (closing {position.side.upper()} position)")
            
            # Place market order to close position
            order = self.exchange.create_market_order(self.symbol, close_side, position.quantity)
            
            if not order:
                self.logger.error(f"❌ MARKET CLOSURE ORDER FAILED: No response from exchange for {position.side.upper()} position {position.position_id[:8]}")
                return
                
            if 'id' not in order:
                self.logger.error(f"❌ MARKET CLOSURE ORDER FAILED: Invalid response for {position.side.upper()} position {position.position_id[:8]}")
                return
            
            # ENHANCED: Success logging with order details
            order_id = order.get('id', 'unknown')
            
            time.sleep(1)  # Wait for order execution
            
            if order and 'average' in order and order['average']:
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
                
                # ENHANCED: Final closure confirmation logging
                self.logger.info(f"✅ MARKET CLOSURE ORDER FILLED: {position.side.upper()} ${position.entry_price:.6f} → ${position.exit_price:.6f}, "
                            f"PnL: ${position.realized_pnl:.2f}, ID: {order_id[:8]}")
                
            else:
                self.logger.error(f"❌ MARKET CLOSURE ORDER EXECUTION FAILED: No fill price for {position.side.upper()} position {position.position_id[:8]}, ID: {order_id[:8]}")
                
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR closing position {position.position_id[:8]} at market: {e}")
    
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