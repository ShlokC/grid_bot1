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
        """Analyze market using real KAMA and Bollinger Bands"""
        try:
            # FIXED: First fetch OHLCV data to populate price_history properly
            current_time = time.time()
            self.logger.info(f"🧠 STARTING MARKET ANALYSIS:")
            self.logger.info(f"   Symbol: {self.symbol}")
            # self.logger.info(f"   Price history length: {len(self.price_history)}")
            # self.logger.info(f"   Last OHLCV fetch: {(current_time - self.last_ohlcv_fetch):.1f}s ago")
            # self.logger.info(f"   OHLCV data fetched: {self.ohlcv_data_fetched}")
            
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
            
            # ADDED: Calculate Bollinger Bands using pandas-ta
            bollinger_upper = 0.0
            bollinger_lower = 0.0
            bollinger_middle = 0.0
            
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
                        bollinger_lower = float(bb_data.iloc[-1]['BBL_20_2.0'])
                        bollinger_middle = float(bb_data.iloc[-1]['BBM_20_2.0'])  
                        bollinger_upper = float(bb_data.iloc[-1]['BBU_20_2.0'])
                        
                        self.logger.info(f"📊 BOLLINGER BANDS CALCULATED:")
                        self.logger.info(f"   Upper Band: ${bollinger_upper:.6f}")
                        self.logger.info(f"   Middle Band: ${bollinger_middle:.6f}")
                        self.logger.info(f"   Lower Band: ${bollinger_lower:.6f}")
                        self.logger.info(f"   Current Price: ${current_price:.6f}")
                        
                        # Calculate position within bands
                        if bollinger_upper > bollinger_lower:
                            band_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
                            self.logger.info(f"   Band Position: {band_position:.2f} (0=lower, 1=upper)")
                    else:
                        self.logger.warning(f"⚠️ Bollinger Bands calculation failed - using current price ±5%")
                        bollinger_lower = current_price * 0.95
                        bollinger_upper = current_price * 1.05
                        bollinger_middle = current_price
                else:
                    self.logger.warning(f"⚠️ Insufficient data for Bollinger Bands - using current price ±5%")
                    bollinger_lower = current_price * 0.95
                    bollinger_upper = current_price * 1.05
                    bollinger_middle = current_price
                    
            except ImportError:
                self.logger.warning("pandas-ta not available for Bollinger Bands, using current price ±5%")
                bollinger_lower = current_price * 0.95
                bollinger_upper = current_price * 1.05
                bollinger_middle = current_price
            except Exception as e:
                self.logger.error(f"Error calculating Bollinger Bands: {e}")
                bollinger_lower = current_price * 0.95
                bollinger_upper = current_price * 1.05
                bollinger_middle = current_price
            
            # Update market regime indicators  
            self.current_volatility_regime = self._calculate_volatility()
            self.current_trend_strength = self._calculate_trend_strength()
            
            # Calculate KAMA-based momentum (this now also calculates and stores KAMA indicators)
            momentum = self._calculate_momentum(exchange)
            
            # Use stored KAMA values (previously calculated in _calculate_kama_indicators)
            kama_value = getattr(self, 'current_kama_value', 0.0)
            kama_direction = getattr(self, 'current_kama_direction', 'neutral')
            kama_strength = getattr(self, 'current_kama_strength', 0.0)
            
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
                directional_bias=momentum,
                # ADDED: Bollinger Band values
                bollinger_upper=bollinger_upper,
                bollinger_lower=bollinger_lower,
                bollinger_middle=bollinger_middle
            )
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return MarketSnapshot(
                timestamp=time.time(),
                price=0.0, volume=0.0, volatility=1.0, momentum=0.0, trend_strength=0.0,
                directional_bias=0.0, bollinger_upper=0.0, bollinger_lower=0.0, bollinger_middle=0.0
            )
    
    def _fetch_historical_data(self, exchange: Exchange) -> None:
        """FIXED: Fetch historical OHLCV data from exchange"""
        try:
            self.logger.info(f"Fetching OHLCV data for {self.symbol}...")
            self.logger.info(f"📊 FETCHING OHLCV DATA:")
            self.logger.info(f"   Symbol: {self.symbol}")
            self.logger.info(f"   Requesting 100 candles of 5m timeframe")
            # Use the existing get_ohlcv method from exchange
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
    
    def _calculate_momentum(self, exchange) -> float:
        """Calculate momentum using pandas-ta KAMA with full historical data"""
        self.logger.info(f"🔢 CALCULATING KAMA MOMENTUM:")
        self.logger.info(f"   Price history length: {len(self.price_history)}")
        
        # Initialize KAMA indicators
        self.current_kama_value = 0.0
        self.current_kama_direction = 'neutral'
        self.current_kama_strength = 0.0
        
        try:
            import pandas as pd
            import pandas_ta as ta
            
            # Get FULL historical data from exchange, not just deque
            full_ohlcv = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=1400)
            
            if not full_ohlcv or len(full_ohlcv) < 50:
                self.logger.warning(f"⚠️ Insufficient OHLCV data: {len(full_ohlcv) if full_ohlcv else 0}")
                return self._fallback_momentum()
            
            # Extract closing prices from OHLCV data
            closing_prices = [float(candle[4]) for candle in full_ohlcv]  # Index 4 = close price
            current_price = closing_prices[-1]
            
            # self.logger.info(f"📊 USING FULL DATASET:")
            # self.logger.info(f"   Total OHLCV candles: {len(full_ohlcv)}")
            # self.logger.info(f"   Total closing prices: {len(closing_prices)}")
            # self.logger.info(f"   Current price: ${current_price:.6f}")
            
            # Create pandas series with full dataset
            price_series = pd.Series(closing_prices)
            
            # Calculate KAMA with TradingView parameters: period=10, fast=2, slow=30
            kama_series = ta.kama(price_series, length=10, fast=2, slow=30)
            
            if kama_series is None or kama_series.isna().all():
                self.logger.error(f"❌ PANDAS-TA KAMA CALCULATION FAILED")
                return self._fallback_momentum()
            
            # Get the latest KAMA value
            current_kama = float(kama_series.iloc[-1])
            self.current_kama_value = current_kama
            
            # Update price history deque with current price only
            if len(self.price_history) == 0 or self.price_history[-1] != current_price:
                self.price_history.append(current_price)
            
            # Store KAMA in history
            self.kama_history.append(current_kama)
            
            # self.logger.info(f"✅ PANDAS-TA KAMA WITH FULL DATA:")
            # self.logger.info(f"   Current price: ${current_price:.6f}")
            # self.logger.info(f"   Current KAMA: ${current_kama:.6f}")
            
            # Calculate KAMA direction from recent KAMA values
            if len(kama_series) >= 10:
                kama_10_ago = float(kama_series.iloc[-10])
                kama_5_ago = float(kama_series.iloc[-5])
                
                # Calculate slope over 5 periods
                kama_slope_5 = (current_kama - kama_5_ago) / kama_5_ago
                # Calculate slope over 10 periods  
                kama_slope_10 = (current_kama - kama_10_ago) / kama_10_ago
                
                # self.logger.info(f"   KAMA 5-period slope: {kama_slope_5:+.6f}")
                # self.logger.info(f"   KAMA 10-period slope: {kama_slope_10:+.6f}")
                
                # Determine direction based on slopes
                if kama_slope_5 > 0.001 and kama_slope_10 > 0.001:
                    self.current_kama_direction = 'bullish'
                    self.current_kama_strength = min(1.0, max(abs(kama_slope_5), abs(kama_slope_10)) * 100)
                elif kama_slope_5 < -0.001 and kama_slope_10 < -0.001:
                    self.current_kama_direction = 'bearish'
                    self.current_kama_strength = min(1.0, max(abs(kama_slope_5), abs(kama_slope_10)) * 100)
                else:
                    # Mixed signals or flat
                    self.current_kama_direction = 'neutral'
                    self.current_kama_strength = 0.0
            
            # Calculate momentum - price position relative to KAMA
            if current_kama > 0:
                momentum_pct = (current_price - current_kama) / current_kama
                final_momentum = max(-1.0, min(1.0, momentum_pct * 5))  # Amplify for sensitivity
                
                # self.logger.info(f"📈 MOMENTUM CALCULATION:")
                # self.logger.info(f"   Price vs KAMA: {momentum_pct * 100:+.3f}%")
                # self.logger.info(f"   Raw momentum: {momentum_pct:+.6f}")
                # self.logger.info(f"   Final momentum: {final_momentum:+.3f}")
                # self.logger.info(f"   KAMA Direction: {self.current_kama_direction}")
                # self.logger.info(f"   KAMA Strength: {self.current_kama_strength:.3f}")
                
                if final_momentum > 0.025:
                    self.logger.info(f"   → BULLISH MOMENTUM")
                elif final_momentum < -0.025:
                    self.logger.info(f"   → BEARISH MOMENTUM")
                else:
                    self.logger.info(f"   → NEUTRAL MOMENTUM")
                
                return final_momentum
            
            return 0.0
            
        except ImportError:
            self.logger.warning("pandas-ta not available, using fallback momentum")
            return self._fallback_momentum()
        except Exception as e:
            self.logger.error(f"Error calculating KAMA with full data: {e}")
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
            self.logger.info(f"🧠 SAMIG ENABLED: MarketIntelligence initialized for {symbol}")
        else:
            self.market_intel = None
            self.logger.info(f"🚫 SAMIG DISABLED: No market intelligence")
        
        # State management
        self.running = False
        self.total_trades = 0
        self.total_pnl = 0.0
        self.last_update_time = 0
        
        # Threading
        self.update_lock = threading.Lock()
        
        # Market information
        self._fetch_market_info()
        self._log_initialization()
        
    def _log_initialization(self):
            """Enhanced initialization logging"""
            self.logger.info("=" * 80)
            self.logger.info(f"🚀 GRID STRATEGY INITIALIZED - {self.grid_id[:8]}")
            self.logger.info("=" * 80)
            self.logger.info(f"📊 CONFIGURATION:")
            self.logger.info(f"   Symbol: {self.original_symbol} → {self.symbol}")
            self.logger.info(f"   Price Range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
            self.logger.info(f"   Grid Levels: {self.user_grid_number}")
            self.logger.info(f"   Investment per Grid: ${self.user_investment_per_grid:.2f}")
            self.logger.info(f"   Total Investment: ${self.user_total_investment:.2f}")
            self.logger.info(f"   Leverage: {self.user_leverage}x")
            self.logger.info(f"   Take Profit: {self.take_profit_pnl}%")
            self.logger.info(f"   Stop Loss: {self.stop_loss_pnl}%")
            self.logger.info(f"   SAMIG: {'✅ ENABLED' if self.enable_samig else '❌ DISABLED'}")
            self.logger.info(f"   Grid Adaptation: {'✅ ENABLED' if self.enable_grid_adaptation else '❌ DISABLED'}")
            self.logger.info("=" * 80)
    
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
        """Enhanced market data retrieval with detailed logging"""
        try:
            self.logger.debug("🔍 Fetching live market data...")
            
            # Get live positions and orders
            live_positions = self.exchange.get_positions(self.symbol)
            live_orders = self.exchange.get_open_orders(self.symbol)
            # ADD THIS LINE to fix untracked orders:
            self._sync_order_tracking(live_orders)
            
            # ENHANCED: Log current market state
            self.logger.info(f"📊 CURRENT MARKET STATE:")
            self.logger.info(f"   Live Orders: {len(live_orders)}")
            self.logger.info(f"   Live Positions: {len([p for p in live_positions if float(p.get('contracts', 0)) != 0])}")
            
            # Clean up stale internal tracking with detailed logging
            live_order_ids = {order['id'] for order in live_orders}
            stale_orders = []
            
            for order_id in list(self.pending_orders.keys()):
                if order_id not in live_order_ids:
                    stale_orders.append(order_id)
                    order_info = self.pending_orders[order_id]
                    self.logger.warning(f"🧹 STALE ORDER DETECTED: {order_id[:8]} ({order_info.get('type', 'unknown')} @ ${order_info.get('price', 0):.6f})")
            
            # Remove stale orders
            for order_id in stale_orders:
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
            
            # ENHANCED: Detailed position analysis
            position_margin = 0.0
            active_positions = 0
            
            for position in live_positions:
                size = float(position.get('contracts', 0))
                if size != 0:
                    entry_price = float(position.get('entryPrice', 0))
                    side = position.get('side', 'unknown')
                    unrealized_pnl = float(position.get('unrealizedPnl', 0))
                    
                    if entry_price > 0:
                        position_notional = abs(size) * entry_price
                        margin = round(position_notional / self.user_leverage, 2)
                        position_margin += margin
                        active_positions += 1
                        
                        self.logger.info(f"📍 POSITION: {side.upper()} {abs(size):.6f} @ ${entry_price:.6f} (PnL: ${unrealized_pnl:.2f}, Margin: ${margin:.2f})")

            # ENHANCED: Detailed order analysis
            order_margin = 0.0
            active_orders = 0
            
            for order in live_orders:
                price = float(order.get('price', 0))
                amount = float(order.get('amount', 0))
                side = order.get('side', 'unknown')
                order_id = order.get('id', 'unknown')
                
                if price > 0 and amount > 0:
                    order_notional = price * amount
                    margin = round(order_notional / self.user_leverage, 2)
                    order_margin += margin
                    active_orders += 1
                    
                    # Check if this order is in our tracking
                    tracked_status = "✅ TRACKED" if order_id in self.pending_orders else "❌ UNTRACKED"
                    
                    self.logger.info(f"📋 ORDER: {side.upper()} {amount:.6f} @ ${price:.6f} (ID: {order_id[:8]}, Margin: ${margin:.2f}) [{tracked_status}]")

            # Calculate totals
            total_margin_used = round(position_margin + order_margin, 2)
            available_investment = round(self.user_total_investment - total_margin_used, 2)
            
            # Update internal tracking
            self.total_investment_used = total_margin_used
            
            # Create price coverage set with logging
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

            # ENHANCED: Summary logging
            self.logger.info(f"💰 INVESTMENT SUMMARY:")
            self.logger.info(f"   Position Margin: ${position_margin:.2f}")
            self.logger.info(f"   Order Margin: ${order_margin:.2f}")
            self.logger.info(f"   Total Used: ${total_margin_used:.2f}")
            self.logger.info(f"   Available: ${available_investment:.2f}")
            self.logger.info(f"   Utilization: {(total_margin_used/self.user_total_investment*100):.1f}%")
            
            if len(stale_orders) > 0:
                self.logger.warning(f"🧹 Cleaned {len(stale_orders)} stale order records")

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
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting live market data: {e}")
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
        """Enhanced order placement with comprehensive logging and validation"""
        try:
            self.logger.info(f"🎯 ATTEMPTING ORDER PLACEMENT:")
            self.logger.info(f"   Type: {side.upper()}")
            self.logger.info(f"   Price: ${price:.6f}")
            
            # Get current market price for context
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            price_diff_pct = ((price - current_price) / current_price) * 100
            
            self.logger.info(f"   Current Price: ${current_price:.6f}")
            self.logger.info(f"   Price Difference: {price_diff_pct:+.2f}%")
            
            # ADDED: Smart Last Order Logic - Check total investment usage approaching 80%
            market_data = self._get_live_market_data()
            total_investment_used = market_data['total_margin_used']
            available_investment = market_data['available_investment']
            investment_utilization = (total_investment_used / self.user_total_investment) * 100
            
            # Check if we're near 80% AND can only place 1 more order
            can_place_orders = int(available_investment / self.user_investment_per_grid)
            
            if investment_utilization >= 75.0 and can_place_orders <= 1:
                # Determine dominant position direction from existing positions
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
                
                # If no positions yet, allow the order
                if net_position_value == 0:
                    self.logger.info(f"✅ NO DOMINANT POSITION - Order allowed")
                else:
                    # Determine dominant direction
                    dominant_direction = 'long' if net_position_value > 0 else 'short'
                    
                    # Check if this order is same direction as dominant position
                    is_same_direction = False
                    if dominant_direction == 'long' and side == 'buy':
                        is_same_direction = True
                    elif dominant_direction == 'short' and side == 'sell':
                        is_same_direction = True
                    
                    if is_same_direction:
                        self.logger.warning(f"🚫 LAST ORDER PROTECTION:")
                        self.logger.warning(f"   Investment Used: {investment_utilization:.1f}% (approaching 80%)")
                        self.logger.warning(f"   Remaining Capacity: {can_place_orders} order(s)")
                        self.logger.warning(f"   Dominant Position: {dominant_direction.upper()}")
                        self.logger.warning(f"   Blocked Order: {side.upper()} (same direction)")
                        self.logger.warning(f"   Preserving capacity for opposite-direction orders")
                        return False
                    else:
                        self.logger.info(f"✅ LAST ORDER CHECK PASSED:")
                        self.logger.info(f"   Investment Used: {investment_utilization:.1f}%")
                        self.logger.info(f"   Order: {side.upper()} (opposite to {dominant_direction.upper()} position)")
            
            # FIXED: Market intelligence check with CORRECT logic
            if self.market_intel:
                try:
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    
                    self.logger.info(f"🧠 MARKET INTELLIGENCE CHECK:")
                    self.logger.info(f"   KAMA Direction: {market_snapshot.kama_direction}")
                    self.logger.info(f"   KAMA Strength: {market_snapshot.kama_strength:.3f}")
                    # self.logger.info(f"   Directional Bias: {market_snapshot.directional_bias:.3f}")
                    # self.logger.info(f"   Volatility: {market_snapshot.volatility:.2f}")
                    
                    kama_strength = market_snapshot.kama_strength
                    kama_direction = market_snapshot.kama_direction
                    distance_pct = abs(price - current_price) / current_price
                    
                    # SMART LOGIC: Differentiate between profit-taking and position-opening orders
                    is_profit_taking_order = False
                    is_risky_new_position = False
                    
                    if side == 'sell' and price > current_price:
                        # SELL above current price = likely profit-taking for LONG position
                        is_profit_taking_order = True
                        self.logger.info(f"   Order type: PROFIT-TAKING (SELL above current price)")
                    elif side == 'buy' and price < current_price:
                        # BUY below current price = likely profit-taking for SHORT position  
                        is_profit_taking_order = True
                        self.logger.info(f"   Order type: PROFIT-TAKING (BUY below current price)")
                    elif side == 'sell' and price < current_price:
                        # SELL below current price = opening new SHORT position
                        is_risky_new_position = True
                        self.logger.info(f"   Order type: NEW SHORT POSITION (SELL below current price)")
                    elif side == 'buy' and price > current_price:
                        # BUY above current price = opening new LONG position
                        is_risky_new_position = True  
                        self.logger.info(f"   Order type: NEW LONG POSITION (BUY above current price)")
                    else:
                        # Neutral/close to current price
                        self.logger.info(f"   Order type: NEUTRAL (close to current price)")
                    
                    # ENHANCED BLOCKING LOGIC: Only block risky new positions, allow profit-taking
                    if kama_strength > 0.7 and is_risky_new_position:
                        should_block = False
                        block_reason = ""
                        
                        # In BEARISH trend: Block new LONG positions and new SHORT positions above current price
                        if kama_direction == 'bearish':
                            if side == 'buy' and price > current_price:
                                should_block = True
                                block_reason = f"NEW LONG position against BEARISH trend (strength: {kama_strength:.3f})"
                            elif side == 'sell' and price < current_price and distance_pct > 0.02:
                                # Allow small distance sells even in bearish (for grid spacing)
                                should_block = True  
                                block_reason = f"NEW SHORT position too far below current in BEARISH trend"
                        
                        # In BULLISH trend: Block new SHORT positions and new LONG positions below current price
                        elif kama_direction == 'bullish':
                            if side == 'sell' and price < current_price:
                                should_block = True
                                block_reason = f"NEW SHORT position against BULLISH trend (strength: {kama_strength:.3f})"
                            elif side == 'buy' and price > current_price and distance_pct > 0.02:
                                # Allow small distance buys even in bullish (for grid spacing)
                                should_block = True
                                block_reason = f"NEW LONG position too far above current in BULLISH trend"
                        
                        if should_block:
                            self.logger.warning(f"🚫 INTELLIGENCE BLOCK: {side.upper()} @ ${price:.6f}")
                            self.logger.warning(f"   Reason: {block_reason}")
                            return False
                    
                    # ALWAYS ALLOW profit-taking orders regardless of trend
                    if is_profit_taking_order:
                        self.logger.info(f"✅ PROFIT-TAKING ORDER APPROVED")
                        self.logger.info(f"   Allowing {side.upper()} @ ${price:.6f} for profit-taking")
                    
                    # For medium strength trends, apply lighter restrictions
                    elif kama_strength > 0.4:
                        if kama_direction == 'bearish' and side == 'buy' and distance_pct > 0.02:
                            self.logger.warning(f"🚫 INTELLIGENCE BLOCK: {side.upper()} @ ${price:.6f}")
                            self.logger.warning(f"   Reason: MEDIUM bearish momentum - avoid counter-trend >2%")
                            return False
                        elif kama_direction == 'bullish' and side == 'sell' and distance_pct > 0.02:
                            self.logger.warning(f"🚫 INTELLIGENCE BLOCK: {side.upper()} @ ${price:.6f}")
                            self.logger.warning(f"   Reason: MEDIUM bullish momentum - avoid counter-trend >2%")
                            return False
                    
                    self.logger.info(f"✅ INTELLIGENCE CHECK PASSED")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Intelligence check failed: {e}")
            
            # Calculate order amount with validation logging
            amount = self._calculate_order_amount(price)
            notional_value = price * amount
            margin_required = round(notional_value / self.user_leverage, 2)
            
            self.logger.info(f"📊 ORDER CALCULATIONS:")
            self.logger.info(f"   Amount: {amount:.6f}")
            self.logger.info(f"   Notional: ${notional_value:.2f}")
            self.logger.info(f"   Margin Required: ${margin_required:.2f}")
            
            # Validate order parameters
            if amount < self.min_amount:
                self.logger.error(f"❌ Order amount {amount:.6f} below minimum {self.min_amount}")
                return False
            
            if notional_value < self.min_cost:
                self.logger.error(f"❌ Order notional ${notional_value:.2f} below minimum ${self.min_cost}")
                return False
            
            # ENHANCED: Pre-order validation
            self.logger.info(f"✅ ORDER VALIDATION PASSED - Placing order...")
            
            # Place the order
            order = self.exchange.create_limit_order(self.symbol, side, amount, price)
            
            if not order or 'id' not in order:
                self.logger.error(f"❌ Failed to place {side} order - invalid response")
                return False
            
            # Store order information with enhanced tracking (ADDED: importance score)
            distance_from_current = abs(price - current_price)
            importance_score = 1.0 / (1.0 + distance_from_current)  # Closer orders = higher importance
            
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
                'distance_from_current': distance_from_current
            }
            
            self.pending_orders[order['id']] = order_info
            
            self.logger.info(f"✅ ORDER PLACED SUCCESSFULLY:")
            self.logger.info(f"   Order ID: {order['id'][:8]}...")
            self.logger.info(f"   Type: {side.upper()}")
            self.logger.info(f"   Price: ${price:.6f}")
            self.logger.info(f"   Amount: {amount:.6f}")
            self.logger.info(f"   Margin Used: ${margin_required:.2f}")
            self.logger.info(f"   Importance Score: {importance_score:.3f}")
            self.logger.info(f"   Total Pending Orders: {len(self.pending_orders)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ FAILED to place {side} order at ${price:.6f}: {e}")
            return False

    def _sync_order_tracking(self, live_orders: List[Dict]) -> int:
        """
        Synchronize internal order tracking with live exchange orders.
        This fixes the "UNTRACKED" order issue with minimal code changes.
        
        Returns:
            int: Number of orders added to tracking
        """
        added_orders = 0
        
        try:
            live_order_ids = {order['id'] for order in live_orders}
            
            # Add untracked orders to our tracking
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
                        self.logger.info(f"✅ RECOVERED ORDER: {side.upper()} {amount:.6f} @ ${price:.6f} (ID: {order_id[:8]})")
            
            if added_orders > 0:
                self.logger.info(f"🔄 ORDER SYNC: Added {added_orders} untracked orders to internal tracking")
            
            return added_orders
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing order tracking: {e}")
            return 0
    def update_grid(self):
        """Enhanced main grid update loop with comprehensive logging"""
        try:
            with self.update_lock:
                if not self.running:
                    return
                
                update_start_time = time.time()
                self.logger.debug(f"🔄 GRID UPDATE CYCLE START")
                
                # Get current market price
                ticker = self.exchange.get_ticker(self.symbol)
                current_price = float(ticker['last'])
                
                self.logger.debug(f"   Current Price: ${current_price:.6f}")
                self.logger.debug(f"   Grid Range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
                
                # Check if price is within range
                within_range = self.user_price_lower <= current_price <= self.user_price_upper
                self.logger.debug(f"   Within Range: {'✅ YES' if within_range else '❌ NO'}")
                
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
                
                update_duration = time.time() - update_start_time
                self.last_update_time = time.time()
                
                self.logger.debug(f"🔄 GRID UPDATE CYCLE COMPLETE ({update_duration:.2f}s)")
                
        except Exception as e:
            self.logger.error(f"❌ Error updating grid: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _maintain_grid_coverage(self, current_price: float):
        """Ensure grid has adequate coverage within user range"""
        try:
            self.logger.info(f"🔄 CHECKING GRID COVERAGE:")
            self.logger.info(f"   Current Price: ${current_price:.6f}")
            self.logger.info(f"   User Range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
            
            # Check if price is outside user range and adapt if enabled
            if current_price < self.user_price_lower or current_price > self.user_price_upper:
                if self.enable_grid_adaptation:
                    self.logger.info(f"🔄 GRID ADAPTATION ENABLED - Updating range")
                    
                    # Calculate range size to maintain
                    range_size = self.user_price_upper - self.user_price_lower
                    
                    # Update user range bounds based on current price
                    if current_price < self.user_price_lower:
                        # Price below range - shift range down
                        new_upper = self.user_price_lower + (range_size * 0.1)  # Small overlap
                        new_lower = new_upper - range_size
                        self.logger.info(f"   Price below range - shifting down")
                    else:
                        # Price above range - shift range up  
                        new_lower = self.user_price_upper - (range_size * 0.1)  # Small overlap
                        new_upper = new_lower + range_size
                        self.logger.info(f"   Price above range - shifting up")
                    
                    # Update the user range bounds
                    old_lower, old_upper = self.user_price_lower, self.user_price_upper
                    self.user_price_lower = self._round_price(new_lower)
                    self.user_price_upper = self._round_price(new_upper)
                    
                    self.logger.info(f"📊 RANGE ADAPTED:")
                    self.logger.info(f"   Old Range: ${old_lower:.6f} - ${old_upper:.6f}")
                    self.logger.info(f"   New Range: ${self.user_price_lower:.6f} - ${self.user_price_upper:.6f}")
                    self.logger.info(f"   Current Price: ${current_price:.6f} (now within range)")
                    
                else:
                    self.logger.warning(f"⚠️ PRICE OUTSIDE USER RANGE - No new orders will be placed")
                    return
            else:
                self.logger.info(f"✅ PRICE WITHIN RANGE - Checking for gaps")
            
            # Get current market data
            market_data = self._get_live_market_data()
            
            self.logger.info(f"💰 GRID COVERAGE CHECK:")
            self.logger.info(f"   Available Investment: ${market_data['available_investment']:.2f}")
            self.logger.info(f"   Required per Grid: ${self.user_investment_per_grid:.2f}")
            self.logger.info(f"   Total Commitment: {market_data['total_commitment']}/{self.max_total_orders}")
            
            # Check if we have investment and capacity for more orders
            if market_data['available_investment'] < self.user_investment_per_grid:
                self.logger.warning(f"⚠️ INSUFFICIENT INVESTMENT for new orders")
                return
            
            if market_data['total_commitment'] >= self.max_total_orders:
                self.logger.warning(f"⚠️ MAX ORDER CAPACITY REACHED")
                return
            
            self.logger.info(f"✅ CAPACITY AVAILABLE - Looking for grid gaps")
            
            # Find gaps in grid coverage and fill them
            self._fill_grid_gaps(current_price, market_data)
            
        except Exception as e:
            self.logger.error(f"❌ Error maintaining grid coverage: {e}")
    
    def _fill_grid_gaps(self, current_price: float, market_data: Dict[str, Any]):
        """Fill gaps in grid coverage within user range"""
        try:
            self.logger.info(f"🔍 ANALYZING GRID GAPS:")
            
            # Get current covered prices
            covered_prices = market_data['covered_prices']
            self.logger.info(f"   Currently covered prices: {len(covered_prices)} levels")
            for price in sorted(covered_prices):
                self.logger.info(f"     ${price:.6f}")
            
            # Get grid levels
            grid_levels = self._calculate_grid_levels()
            if not grid_levels:
                self.logger.error(f"❌ NO GRID LEVELS CALCULATED")
                return
            
            self.logger.info(f"   Target grid levels: {len(grid_levels)} levels")
            for i, level in enumerate(grid_levels):
                self.logger.info(f"     Level {i+1}: ${level:.6f}")
            
            # Calculate minimum gap
            price_range = self.user_price_upper - self.user_price_lower
            min_gap = price_range / self.user_grid_number * 0.3
            self.logger.info(f"   Minimum gap: ${min_gap:.6f}")
            
            # Find levels that need orders
            levels_needing_orders = []
            
            for level_price in grid_levels:
                # Skip if too close to current price
                distance_to_current = abs(level_price - current_price)
                if distance_to_current < min_gap:
                    self.logger.debug(f"   Skip ${level_price:.6f}: Too close to current (${distance_to_current:.6f} < ${min_gap:.6f})")
                    continue
                
                # Skip if already covered
                is_covered = any(abs(level_price - covered) < min_gap for covered in covered_prices)
                if is_covered:
                    self.logger.debug(f"   Skip ${level_price:.6f}: Already covered")
                    continue
                
                levels_needing_orders.append(level_price)
                side = "BUY" if level_price < current_price else "SELL"
                self.logger.info(f"   GAP FOUND: {side} needed @ ${level_price:.6f}")
            
            if not levels_needing_orders:
                self.logger.info(f"✅ NO GAPS FOUND - Grid coverage is adequate")
                return
            
            self.logger.info(f"🎯 ATTEMPTING TO FILL {len(levels_needing_orders)} GAPS:")
            
            # Sort by distance from current price
            levels_needing_orders.sort(key=lambda x: abs(x - current_price))
            
            # Place one order at the most appropriate level
            for level_price in levels_needing_orders[:1]:  # Only place one at a time
                side = "buy" if level_price < current_price else "sell"
                
                self.logger.info(f"🎯 TRYING TO PLACE: {side.upper()} @ ${level_price:.6f}")
                
                if self._place_single_order(level_price, side):
                    self.logger.info(f"✅ GAP FILLED: {side.upper()} @ ${level_price:.6f}")
                    break
                else:
                    self.logger.warning(f"❌ FAILED TO FILL GAP: {side.upper()} @ ${level_price:.6f}")
                        
        except Exception as e:
            self.logger.error(f"❌ Error filling grid gaps: {e}")
    
    def _update_orders_and_positions(self):
        """Check for filled orders, validate existing orders against KAMA, manage margin limits, and update position tracking"""
        try:
            # Get current open orders
            open_orders = self.exchange.get_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            # ENHANCED: Log current order state before any cancellations
            self.logger.info(f"🔍 ORDER CANCELLATION ANALYSIS:")
            self.logger.info(f"   Current open orders: {len(open_orders)}")
            for i, order in enumerate(open_orders, 1):
                order_id = order.get('id', 'unknown')
                order_side = order.get('side', 'unknown')
                order_price = float(order.get('price', 0))
                order_amount = float(order.get('amount', 0))
                tracked_status = "✅ TRACKED" if order_id in self.pending_orders else "❌ UNTRACKED"
                
                self.logger.info(f"   Order {i}: {order_side.upper()} {order_amount:.6f} @ ${order_price:.6f} (ID: {order_id[:8]}) [{tracked_status}]")

            # FIXED: Position-based capacity management instead of total utilization
            market_data = self._get_live_market_data()
            total_investment_used = market_data['total_margin_used']
            available_investment = market_data['available_investment']
            position_margin = market_data['position_margin']  # Only actual positions, not pending orders
            order_margin = market_data['order_margin']       # Only pending orders
            
            # Calculate percentages
            total_utilization = (total_investment_used / self.user_total_investment) * 100
            position_margin_pct = (position_margin / self.user_total_investment) * 100
            
            # Check if we're near position limit AND can only place 1 more order
            can_place_orders = int(available_investment / self.user_investment_per_grid)
            
            self.logger.info(f"💰 INVESTMENT STATUS:")
            self.logger.info(f"   Position margin: ${position_margin:.2f} ({position_margin_pct:.1f}%)")
            self.logger.info(f"   Order margin: ${order_margin:.2f}")
            self.logger.info(f"   Total utilization: {total_utilization:.1f}%")
            self.logger.info(f"   Can place orders: {can_place_orders}")
            self.logger.info(f"   Position-based capacity threshold: {position_margin_pct >= 80.0 and can_place_orders <= 1}")
            
            # FIXED: Trigger based on POSITION MARGIN, not total utilization
            if position_margin_pct >= 80.0 and can_place_orders <= 1 and open_orders:
                self.logger.warning(f"🚨 POSITION CAPACITY MANAGEMENT TRIGGERED")
                self.logger.warning(f"   Position margin at {position_margin_pct:.1f}% (threshold: 80%)")
                
                # Determine dominant position direction from existing positions
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
                
                self.logger.info(f"   Net position value: ${net_position_value:.2f}")
                
                # Only cancel if we have a dominant position
                if net_position_value != 0:
                    dominant_direction = 'long' if net_position_value > 0 else 'short'
                    self.logger.warning(f"   Dominant direction: {dominant_direction.upper()}")
                    
                    # Get current price for distance calculations
                    ticker = self.exchange.get_ticker(self.symbol)
                    current_price = float(ticker['last'])
                    
                    # Find same-direction orders to potentially cancel
                    same_direction_orders = []
                    for order in open_orders:
                        order_side = order.get('side', '').lower()
                        order_price = float(order.get('price', 0))
                        order_id = order.get('id', '')
                        
                        # Check if order is in same direction as dominant position
                        is_same_direction = False
                        if dominant_direction == 'long' and order_side == 'buy':
                            is_same_direction = True
                        elif dominant_direction == 'short' and order_side == 'sell':
                            is_same_direction = True
                        
                        direction_status = "SAME" if is_same_direction else "OPPOSITE"
                        self.logger.info(f"   Order {order_id[:8]}: {order_side.upper()} @ ${order_price:.6f} - {direction_status} direction")
                        
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
                            
                            self.logger.info(f"     → Candidate for cancellation (importance: {importance_score:.3f}, distance: ${distance_from_current:.6f})")
                    
                    # Cancel the least important same-direction order
                    if same_direction_orders:
                        # Sort by importance (lowest importance first = furthest + oldest)
                        same_direction_orders.sort(key=lambda x: (x['importance_score'], -x['distance_from_current']))
                        
                        # Cancel the least important order
                        order_to_cancel = same_direction_orders[0]
                        
                        try:
                            self.logger.warning(f"🗑️ CANCELLING ORDER DUE TO POSITION CAPACITY MANAGEMENT:")
                            self.logger.warning(f"   Position margin: {position_margin_pct:.1f}% (threshold: 80%)")
                            self.logger.warning(f"   Order ID: {order_to_cancel['id'][:8]}")
                            self.logger.warning(f"   Type: {order_to_cancel['side'].upper()}")
                            self.logger.warning(f"   Price: ${order_to_cancel['price']:.6f}")
                            self.logger.warning(f"   Reason: Same direction as {dominant_direction.upper()} position")
                            self.logger.warning(f"   Importance: {order_to_cancel['importance_score']:.3f}")
                            self.logger.warning(f"   Distance: ${order_to_cancel['distance_from_current']:.6f}")
                            
                            self.exchange.cancel_order(order_to_cancel['id'], self.symbol)
                            
                            # Remove from internal tracking
                            if order_to_cancel['id'] in self.pending_orders:
                                del self.pending_orders[order_to_cancel['id']]
                            
                            self.logger.info(f"✅ Freed capacity for opposite-direction orders: {order_to_cancel['id'][:8]}")
                            
                            # Refresh open orders after cancellation
                            time.sleep(1)
                            open_orders = self.exchange.get_open_orders(self.symbol)
                            open_order_ids = {order['id'] for order in open_orders}
                            
                        except Exception as e:
                            self.logger.error(f"❌ Failed to cancel order {order_to_cancel['id'][:8]}: {e}")
                    else:
                        self.logger.info(f"   No same-direction orders found to cancel")
                else:
                    self.logger.info(f"   No dominant position - no orders cancelled")
            else:
                if position_margin_pct < 80.0:
                    self.logger.info(f"   Position margin still low ({position_margin_pct:.1f}% < 80%) - no capacity management needed")
                elif can_place_orders > 1:
                    self.logger.info(f"   Can still place {can_place_orders} orders - no capacity management needed")
                elif not open_orders:
                    self.logger.info(f"   No open orders to manage")
            
            # ENHANCED: Continuous KAMA validation of existing orders and positions
            if self.market_intel:
                try:
                    # Get current market price for position loss calculations
                    ticker = self.exchange.get_ticker(self.symbol)
                    market_snapshot = self.market_intel.analyze_market(self.exchange)
                    kama_direction = market_snapshot.kama_direction
                    kama_strength = market_snapshot.kama_strength
                    if market_snapshot.bollinger_upper > 0 and market_snapshot.bollinger_lower > 0:  
                        self.user_price_lower = self._round_price(market_snapshot.bollinger_lower)
                        self.user_price_upper = self._round_price(market_snapshot.bollinger_upper)
                        self.logger.info(f"   New Range: ${self.user_price_lower:.6f} - ${self.user_price_upper}")
                        self.logger.info(f"   Middle Band: ${market_snapshot.bollinger_middle}")
                        self.logger.info(f"   Current Price: ${market_snapshot.price}")
                    self.logger.info(f"🧠 KAMA INTELLIGENCE ANALYSIS:")
                    self.logger.info(f"   KAMA Direction: {kama_direction}")
                    self.logger.info(f"   KAMA Strength: {kama_strength:.3f}")
                    self.logger.info(f"   Cancellation threshold: 0.7")
                    self.logger.info(f"   Will cancel orders: {kama_strength > 0.7}")
                    
                    # FIXED: Changed threshold from 0.5 to 0.7 to match placement logic
                    # Only cancel orders if KAMA signal is strong enough and there are orders to check
                    if kama_strength > 0.7 and open_orders:
                        self.logger.warning(f"🚨 KAMA-BASED ORDER CANCELLATION TRIGGERED")
                        self.logger.warning(f"   Strong {kama_direction.upper()} trend detected (strength: {kama_strength:.3f})")
                        
                        # ADDED: Check existing positions to determine profit-taking vs new position orders
                        live_positions = market_data['live_positions']
                        current_price = float(ticker['last'])
                        
                        # Analyze existing positions
                        has_long_positions = False
                        has_short_positions = False
                        total_long_value = 0.0
                        total_short_value = 0.0
                        
                        for position in live_positions:
                            size = float(position.get('contracts', 0))
                            entry_price = float(position.get('entryPrice', 0))
                            if size != 0 and entry_price > 0:
                                position_value = abs(size) * entry_price
                                if position.get('side', '').lower() == 'long':
                                    has_long_positions = True
                                    total_long_value += position_value
                                else:
                                    has_short_positions = True
                                    total_short_value += position_value
                        
                        self.logger.info(f"   Position Analysis:")
                        self.logger.info(f"     LONG positions: ${total_long_value:.2f}")
                        self.logger.info(f"     SHORT positions: ${total_short_value:.2f}")
                        
                        orders_to_cancel = []
                        
                        for order in open_orders:
                            order_side = order.get('side', '').lower()
                            order_id = order.get('id', '')
                            order_price = float(order.get('price', 0))
                            
                            # SMART LOGIC: Determine if order is profit-taking or new position
                            is_profit_taking = False
                            is_risky_new_position = False
                            order_type_description = ""
                            
                            if order_side == 'sell' and order_price > current_price and has_long_positions:
                                # SELL above current price with LONG positions = profit-taking
                                is_profit_taking = True
                                order_type_description = "PROFIT-TAKING (SELL above current with LONG positions)"
                            elif order_side == 'buy' and order_price < current_price and has_short_positions:
                                # BUY below current price with SHORT positions = profit-taking
                                is_profit_taking = True
                                order_type_description = "PROFIT-TAKING (BUY below current with SHORT positions)"
                            elif order_side == 'sell' and order_price < current_price:
                                # SELL below current price = opening new SHORT position
                                is_risky_new_position = True
                                order_type_description = "NEW SHORT POSITION (SELL below current price)"
                            elif order_side == 'buy' and order_price > current_price:
                                # BUY above current price = opening new LONG position
                                is_risky_new_position = True
                                order_type_description = "NEW LONG POSITION (BUY above current price)"
                            else:
                                # Other cases (neutral or market making)
                                order_type_description = "MARKET MAKING (close to current price)"
                            
                            # Check if order direction conflicts with KAMA trend (only for new positions)
                            should_cancel = False
                            cancel_reason = ""
                            
                            if is_profit_taking:
                                # NEVER cancel profit-taking orders regardless of trend
                                should_cancel = False
                                cancel_reason = "PROFIT-TAKING - Always allowed"
                            elif is_risky_new_position:
                                # Only cancel risky new positions that conflict with trend
                                if kama_direction == 'bearish' and order_side == 'buy':
                                    should_cancel = True
                                    cancel_reason = f"NEW LONG position conflicts with BEARISH KAMA trend"
                                elif kama_direction == 'bullish' and order_side == 'sell':
                                    should_cancel = True
                                    cancel_reason = f"NEW SHORT position conflicts with BULLISH KAMA trend"
                            
                            # Log detailed analysis
                            conflict_status = "WILL CANCEL" if should_cancel else "WILL KEEP"
                            self.logger.info(f"   Order {order_id[:8]}: {order_side.upper()} @ ${order_price:.6f}")
                            self.logger.info(f"     Type: {order_type_description}")
                            self.logger.info(f"     Decision: {conflict_status}")
                            if should_cancel:
                                self.logger.info(f"     Reason: {cancel_reason}")
                            
                            if should_cancel:
                                orders_to_cancel.append({
                                    'id': order_id,
                                    'side': order_side,
                                    'price': order_price,
                                    'reason': cancel_reason,
                                    'order_type': order_type_description
                                })
                        
                        # Log cancellation summary
                        self.logger.warning(f"   Analysis Summary:")
                        self.logger.warning(f"     Total orders analyzed: {len(open_orders)}")
                        self.logger.warning(f"     Profit-taking orders found: {sum(1 for order in open_orders if (order.get('side') == 'sell' and float(order.get('price', 0)) > current_price and has_long_positions) or (order.get('side') == 'buy' and float(order.get('price', 0)) < current_price and has_short_positions))}")
                        self.logger.warning(f"     New position orders found: {sum(1 for order in open_orders if (order.get('side') == 'sell' and float(order.get('price', 0)) < current_price) or (order.get('side') == 'buy' and float(order.get('price', 0)) > current_price))}")
                        self.logger.warning(f"     Orders to cancel: {len(orders_to_cancel)}")
                        
                        if len(orders_to_cancel) == 0:
                            self.logger.info(f"   ✅ No orders cancelled - all are either profit-taking or align with trend")
                        
                        # Cancel conflicting orders
                        for order_info in orders_to_cancel:
                            try:
                                self.logger.warning(f"🗑️ CANCELLING ORDER DUE TO KAMA CONFLICT:")
                                self.logger.warning(f"   Order ID: {order_info['id'][:8]}")
                                self.logger.warning(f"   Type: {order_info['side'].upper()}")
                                self.logger.warning(f"   Price: ${order_info['price']:.6f}")
                                self.logger.warning(f"   Order Type: {order_info['order_type']}")
                                self.logger.warning(f"   Reason: {order_info['reason']}")
                                self.logger.warning(f"   KAMA Strength: {kama_strength:.3f}")
                                
                                self.exchange.cancel_order(order_info['id'], self.symbol)
                                
                                # Remove from internal tracking
                                if order_info['id'] in self.pending_orders:
                                    del self.pending_orders[order_info['id']]
                                
                                self.logger.info(f"✅ Order cancelled successfully: {order_info['id'][:8]}")
                                
                            except Exception as e:
                                self.logger.error(f"❌ Failed to cancel order {order_info['id'][:8]}: {e}")
                        
                        if orders_to_cancel:
                            # Refresh open orders after cancellations
                            time.sleep(1)  # Brief pause for cancellations to process
                            open_orders = self.exchange.get_open_orders(self.symbol)
                            open_order_ids = {order['id'] for order in open_orders}
                    else:
                        if kama_strength <= 0.7:
                            self.logger.info(f"   KAMA strength too low for cancellation ({kama_strength:.3f} <= 0.7)")
                        if not open_orders:
                            self.logger.info(f"   No open orders to evaluate for KAMA cancellation")
                    
                    # CONSERVATIVE: Only close positions when trend is VERY strong and against us with no reversal signs  
                    if kama_strength > 0.8:  # Much higher threshold - only very strong trends
                        live_positions = self.exchange.get_positions(self.symbol)
                        positions_to_close = []
                        
                        for position in live_positions:
                            size = float(position.get('contracts', 0))
                            if size != 0:  # Only check non-zero positions
                                position_side = position.get('side', '').lower()
                                entry_price = float(position.get('entryPrice', 0))
                                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                                current_price = float(ticker['last'])
                                
                                # Calculate loss percentage from entry
                                if entry_price > 0:
                                    loss_pct = abs(unrealized_pnl) / (abs(size) * entry_price) * 100
                                else:
                                    loss_pct = 0
                                
                                # Check if position is in significant loss AND trend is very strong against it
                                should_close = False
                                close_reason = ""
                                
                                # LONG position: Close only if VERY strong bearish trend AND significant loss
                                if (position_side == 'long' and kama_direction == 'bearish' and 
                                    unrealized_pnl < 0 and loss_pct > 3.0):  # Only if losing >3%
                                    
                                    # Check for reversal signs - don't close if KAMA strength is weakening
                                    if (not hasattr(self, 'prev_kama_strength') or 
                                        kama_strength >= getattr(self, 'prev_kama_strength', 0)):
                                        should_close = True
                                        close_reason = f"LONG position with VERY STRONG bearish trend (strength: {kama_strength:.3f}) and {loss_pct:.1f}% loss"
                                
                                # SHORT position: Close only if VERY strong bullish trend AND significant loss  
                                elif (position_side == 'short' and kama_direction == 'bullish' and 
                                    unrealized_pnl < 0 and loss_pct > 3.0):  # Only if losing >3%
                                    
                                    # Check for reversal signs - don't close if KAMA strength is weakening
                                    if (not hasattr(self, 'prev_kama_strength') or 
                                        kama_strength >= getattr(self, 'prev_kama_strength', 0)):
                                        should_close = True
                                        close_reason = f"SHORT position with VERY STRONG bullish trend (strength: {kama_strength:.3f}) and {loss_pct:.1f}% loss"
                                
                                if should_close:
                                    positions_to_close.append({
                                        'side': position_side,
                                        'size': abs(size),
                                        'entry_price': entry_price,
                                        'unrealized_pnl': unrealized_pnl,
                                        'loss_pct': loss_pct,
                                        'reason': close_reason
                                    })
                        
                        # Close positions only if criteria are met
                        for pos_info in positions_to_close:
                            try:
                                self.logger.warning(f"🚨 CRITICAL POSITION RISK DETECTED:")
                                self.logger.warning(f"   Position: {pos_info['side'].upper()} {pos_info['size']:.6f} @ ${pos_info['entry_price']:.6f}")
                                self.logger.warning(f"   Loss: ${pos_info['unrealized_pnl']:.2f} ({pos_info['loss_pct']:.1f}%)")
                                self.logger.warning(f"   Reason: {pos_info['reason']}")
                                
                                # Determine close side (opposite of position)
                                close_side = 'sell' if pos_info['side'] == 'long' else 'buy'
                                
                                # Close position with market order
                                close_order = self.exchange.create_market_order(self.symbol, close_side, pos_info['size'])
                                
                                if close_order:
                                    self.logger.info(f"✅ Closed high-risk position: {pos_info['side'].upper()} {pos_info['size']:.6f}")
                                    
                                    # Update internal position tracking if exists
                                    for position_id, grid_pos in self.all_positions.items():
                                        if (grid_pos.is_open() and 
                                            abs(grid_pos.entry_price - pos_info['entry_price']) < 0.000001 and
                                            abs(grid_pos.quantity - pos_info['size']) < 0.000001):
                                            
                                            # Mark position as closed
                                            grid_pos.exit_time = time.time()
                                            grid_pos.exit_price = float(close_order.get('average', pos_info['entry_price']))
                                            grid_pos.realized_pnl = pos_info['unrealized_pnl']
                                            grid_pos.unrealized_pnl = 0.0
                                            grid_pos.has_counter_order = False
                                            grid_pos.counter_order_id = None
                                            
                                            self.logger.info(f"📝 Updated internal position tracking: {position_id[:8]}")
                                            break
                                else:
                                    self.logger.error(f"❌ Failed to close position: {pos_info['side'].upper()}")
                                    
                            except Exception as e:
                                self.logger.error(f"❌ Failed to close position {pos_info['side'].upper()}: {e}")
                    
                    # Store KAMA strength for next comparison (to detect weakening trends)
                    self.prev_kama_strength = kama_strength
                            
                except Exception as e:
                    self.logger.error(f"❌ Error in KAMA order validation: {e}")
            
            # Check each pending order for fills (existing logic)
            filled_order_ids = []
            for order_id in list(self.pending_orders.keys()):
                if order_id not in open_order_ids:
                    filled_order_ids.append(order_id)
            
            self.logger.info(f"📋 FILLED ORDER ANALYSIS:")
            self.logger.info(f"   Pending orders in tracking: {len(self.pending_orders)}")
            self.logger.info(f"   Live orders from exchange: {len(open_orders)}")
            self.logger.info(f"   Orders detected as filled: {len(filled_order_ids)}")
            
            # Process filled orders (existing logic)
            for order_id in filled_order_ids:
                try:
                    self.logger.info(f"   Processing filled order: {order_id[:8]}")
                    order_status = self.exchange.get_order_status(order_id, self.symbol)
                    if order_status['status'] in ['filled', 'closed']:
                        self._process_filled_order(order_id, order_status)
                        self.logger.info(f"   ✅ Processed filled order: {order_id[:8]}")
                    else:
                        # Order was cancelled
                        if order_id in self.pending_orders:
                            order_info = self.pending_orders[order_id]
                            self.logger.info(f"   📝 Order cancelled: {order_info['type']} @ ${order_info['price']:.6f}")
                            del self.pending_orders[order_id]
                except Exception as e:
                    self.logger.error(f"   ❌ Error checking order status {order_id[:8]}: {e}")
                    # Remove problematic order
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
            
            # ENHANCED: Clean up stale internal tracking with detailed logging
            live_order_ids = {order['id'] for order in open_orders}
            stale_orders = []
            
            for order_id in list(self.pending_orders.keys()):
                if order_id not in live_order_ids:
                    stale_orders.append(order_id)
                    order_info = self.pending_orders[order_id]
                    self.logger.warning(f"🧹 STALE ORDER DETECTED: {order_id[:8]} ({order_info.get('type', 'unknown')} @ ${order_info.get('price', 0):.6f})")
            
            # Remove stale orders
            for order_id in stale_orders:
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
            
            if len(stale_orders) > 0:
                self.logger.warning(f"🧹 Cleaned {len(stale_orders)} stale order records")
            
            # ADD THIS LINE to fix untracked orders:
            self._sync_order_tracking(open_orders)
            
            self.logger.info(f"📊 UPDATE CYCLE SUMMARY:")
            self.logger.info(f"   Final pending orders: {len(self.pending_orders)}")
            self.logger.info(f"   Final live orders: {len(open_orders)}")
            self.logger.info(f"   Stale orders cleaned: {len(stale_orders)}")
            
        except Exception as e:
            self.logger.error(f"Error updating orders and positions: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
                
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
        """Enhanced counter order maintenance with detailed logging"""
        try:
            open_positions = [pos for pos in self.all_positions.values() if pos.is_open()]
            
            if not open_positions:
                self.logger.debug("No open positions requiring counter orders")
                return
            
            self.logger.info(f"🔄 COUNTER ORDER MAINTENANCE:")
            self.logger.info(f"   Current Price: ${current_price:.6f}")
            self.logger.info(f"   Open Positions: {len(open_positions)}")
            
            for i, position in enumerate(open_positions, 1):
                if position.has_counter_order:
                    self.logger.debug(f"   Position {i}: Already has counter order")
                    continue
                
                self.logger.info(f"   Position {i}: {position.side.upper()} @ ${position.entry_price:.6f} (Size: {position.quantity:.6f})")
                self.logger.info(f"   -> Creating counter order...")
                
                # Create counter order with logging
                self._create_counter_order_for_position(position, current_price)
                
        except Exception as e:
            self.logger.error(f"❌ Error maintaining counter orders: {e}")
    
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
        """Enhanced PnL calculation with detailed logging"""
        try:
            total_unrealized = 0.0
            total_realized = 0.0
            
            open_positions = [pos for pos in self.all_positions.values() if pos.is_open()]
            closed_positions = [pos for pos in self.all_positions.values() if not pos.is_open()]
            
            # Calculate PnL for open positions
            if open_positions:
                self.logger.debug(f"📊 OPEN POSITIONS PnL @ ${current_price:.6f}:")
                
                for pos in open_positions:
                    pos.unrealized_pnl = pos.calculate_unrealized_pnl(current_price)
                    total_unrealized += pos.unrealized_pnl
                    
                    pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.quantity)) * 100
                    self.logger.debug(f"   {pos.side.upper()} @ ${pos.entry_price:.6f}: ${pos.unrealized_pnl:.2f} ({pnl_pct:+.2f}%)")

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
                
                self.logger.info(f"💰 PnL SUMMARY:")
                self.logger.info(f"   Realized PnL: ${total_realized:.2f}")
                self.logger.info(f"   Unrealized PnL: ${total_unrealized:.2f}")
                self.logger.info(f"   Total PnL: ${self.total_pnl:.2f} ({pnl_percentage:.2f}%)")
                self.logger.info(f"   Open Positions: {len(open_positions)}")
                self.logger.info(f"   Closed Positions: {len(closed_positions)}")
                self.logger.info(f"   Total Trades: {self.total_trades}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating PnL: {e}")
    
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