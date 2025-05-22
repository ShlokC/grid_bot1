"""
Enhanced Grid Trading Strategy with Self-Adaptive Market Intelligence (SAMIG)
Combines traditional grid trading with advanced market analysis and adaptive parameters.
"""
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque
import traceback
from core.exchange import Exchange

@dataclass
class MarketSnapshot:
    """Real-time market condition snapshot"""
    timestamp: float
    price: float
    volume: float
    spread: float
    order_book_imbalance: float
    price_velocity: float
    volatility_regime: float
    momentum: float
    mean_reversion_strength: float

@dataclass
class GridPerformanceMetric:
    """Performance tracking for adaptive learning"""
    timestamp: float
    pnl: float
    win_rate: float
    trades_count: int
    market_conditions: MarketSnapshot
    grid_config: Dict

class AdaptiveParameterManager:
    """Crypto-optimized adaptive parameter management"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=50)  # Shorter history for crypto
        self.learning_rate = 0.2  # Higher learning rate for fast crypto markets
        
        # Crypto-specific parameter bounds
        self.param_bounds = {
            'grid_density': {'min': 0.3, 'max': 2.5, 'current': 1.0},
            'volatility_response': {'min': 0.5, 'max': 3.0, 'current': 1.2},
            'momentum_threshold': {'min': 0.05, 'max': 0.8, 'current': 0.3},
            'mean_reversion_factor': {'min': 0.3, 'max': 2.5, 'current': 1.0},
            'exposure_asymmetry': {'min': 0.6, 'max': 2.0, 'current': 1.0},
            'rapid_adaptation': {'min': 0.1, 'max': 1.0, 'current': 0.5},
            'volume_sensitivity': {'min': 0.2, 'max': 2.0, 'current': 1.0},
            'rsi_bounds_adjustment': {'min': 0.5, 'max': 1.5, 'current': 1.0}
        }
        
        # Market regime detection
        self.current_regime = 'neutral'  # trending, ranging, volatile
        self.regime_stability = 0
    
    def update_parameter(self, param_name: str, performance_score: float, market_conditions: MarketSnapshot):
        """Update parameter based on crypto market performance"""
        if param_name not in self.param_bounds:
            return
        
        bounds = self.param_bounds[param_name]
        current = bounds['current']
        
        # Detect market regime
        self._update_market_regime(market_conditions)
        
        # Regime-specific learning rates
        regime_multipliers = {
            'trending': 1.5,    # Learn faster in trending markets
            'ranging': 0.8,     # Learn slower in ranging markets  
            'volatile': 2.0     # Learn fastest in volatile markets
        }
        
        effective_learning_rate = self.learning_rate * regime_multipliers.get(self.current_regime, 1.0)
        
        # Performance-based adjustment with crypto-specific scaling
        if performance_score > 0:
            adjustment = effective_learning_rate * (performance_score / 50)  # Scale for crypto returns
        else:
            adjustment = -effective_learning_rate * (abs(performance_score) / 50)
        
        # Market condition modulations
        adjustment *= self._get_market_adjustment_factor(param_name, market_conditions)
        
        # Apply bounds with momentum
        momentum = 0.1 * (current - bounds.get('previous', current))
        new_value = current + adjustment + momentum
        
        # Store previous value
        bounds['previous'] = current
        bounds['current'] = max(bounds['min'], min(bounds['max'], new_value))
        
        logging.debug(f"Adaptive parameter {param_name}: {current:.3f} -> {bounds['current']:.3f} "
                     f"(regime: {self.current_regime}, perf: {performance_score:.2f}%)")
    
    def _update_market_regime(self, market_conditions: MarketSnapshot):
        """Detect and update current market regime"""
        # Use volatility and momentum to classify regime
        vol_regime = market_conditions.volatility_regime
        momentum = abs(market_conditions.momentum)
        
        # Regime classification thresholds
        if vol_regime > 1.5 and momentum > 0.3:
            new_regime = 'volatile'
        elif momentum > 0.4:
            new_regime = 'trending'
        else:
            new_regime = 'ranging'
        
        # Update regime with stability counter
        if new_regime == self.current_regime:
            self.regime_stability = min(10, self.regime_stability + 1)
        else:
            if self.regime_stability > 3:  # Require stability before changing
                self.current_regime = new_regime
                self.regime_stability = 0
            else:
                self.regime_stability = 0
    
    def _get_market_adjustment_factor(self, param_name: str, market_conditions: MarketSnapshot) -> float:
        """Get market-specific adjustment factors for parameters"""
        vol_regime = market_conditions.volatility_regime
        momentum = market_conditions.momentum
        mean_reversion = market_conditions.mean_reversion_strength
        
        # Parameter-specific market adjustments
        adjustments = {
            'grid_density': 1 + (vol_regime - 1) * 0.5,  # More grids in volatile markets
            'volatility_response': vol_regime,  # Direct volatility scaling
            'momentum_threshold': 1 - abs(momentum) * 0.3,  # Lower threshold in trending markets
            'mean_reversion_factor': mean_reversion,  # Scale with mean reversion strength
            'exposure_asymmetry': 1 + momentum * 0.5,  # Bias exposure with momentum
            'rapid_adaptation': vol_regime * 0.8,  # Adapt faster in volatile markets
            'volume_sensitivity': 1 + market_conditions.volume / 1000000 * 0.1,  # Volume scaling
            'rsi_bounds_adjustment': 1 + (vol_regime - 1) * 0.3  # Adjust RSI bounds for volatility
        }
        
        return adjustments.get(param_name, 1.0)
    
    def get_parameter(self, param_name: str) -> float:
        """Get current adaptive parameter value"""
        return self.param_bounds.get(param_name, {}).get('current', 1.0)
    
    def get_regime_adjusted_parameters(self) -> Dict[str, float]:
        """Get all parameters adjusted for current market regime"""
        base_params = {name: bounds['current'] for name, bounds in self.param_bounds.items()}
        
        # Apply regime-specific adjustments
        if self.current_regime == 'volatile':
            base_params['grid_density'] *= 0.8  # Fewer grids in chaos
            base_params['rapid_adaptation'] *= 1.5  # Adapt faster
            base_params['volatility_response'] *= 1.2  # More responsive
            
        elif self.current_regime == 'trending':
            base_params['momentum_threshold'] *= 0.7  # Lower momentum threshold
            base_params['exposure_asymmetry'] *= 1.3  # More directional bias
            base_params['mean_reversion_factor'] *= 0.8  # Less mean reversion
            
        elif self.current_regime == 'ranging':
            base_params['grid_density'] *= 1.2  # More grids for range trading
            base_params['mean_reversion_factor'] *= 1.3  # More mean reversion
            base_params['momentum_threshold'] *= 1.3  # Higher momentum threshold
        
        return base_params
    
    def reset_parameters(self):
        """Reset all parameters to defaults (useful after major regime changes)"""
        defaults = {
            'grid_density': 1.0,
            'volatility_response': 1.2,
            'momentum_threshold': 0.3,
            'mean_reversion_factor': 1.0,
            'exposure_asymmetry': 1.0,
            'rapid_adaptation': 0.5,
            'volume_sensitivity': 1.0,
            'rsi_bounds_adjustment': 1.0
        }
        
        for param_name, default_value in defaults.items():
            if param_name in self.param_bounds:
                self.param_bounds[param_name]['current'] = default_value
        
        logging.info("Adaptive parameters reset to defaults due to regime change")
class MarketIntelligenceEngine:
    """Crypto-focused market analysis engine using real-time kline data"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.kline_data = deque(maxlen=100)  # Store recent klines
        self.price_history = deque(maxlen=200)
        self.volume_history = deque(maxlen=200)
        self.rsi_history = deque(maxlen=50)
        self.snapshot_history = deque(maxlen=100)
        self.last_kline_time = 0
        
        # Crypto-specific characteristics
        self.asset_characteristics = {
            'typical_volatility': None,
            'volume_profile': None,
            'momentum_persistence': None,
            'price_precision': None
        }
    
    def analyze_market_conditions(self, exchange: Exchange) -> MarketSnapshot:
        """Analyze crypto market using real-time kline data and crypto-specific indicators"""
        try:
            # Get recent kline data (1-minute candles for real-time analysis)
            klines = self._fetch_recent_klines(exchange)
            
            if not klines or len(klines) < 10:
                return self._create_fallback_snapshot(exchange)
            
            # Extract current market data
            current_kline = klines[-1]
            current_price = float(current_kline[4])  # Close price
            current_volume = float(current_kline[5])  # Volume
            
            # Update histories
            self.price_history.append(current_price)
            self.volume_history.append(current_volume)
            self.kline_data.extend(klines)
            
            # Calculate crypto-specific analytics
            price_velocity = self._calculate_price_velocity(klines)
            momentum = self._calculate_crypto_momentum(klines)
            volatility_regime = self._calculate_crypto_volatility(klines)
            volume_momentum = self._calculate_volume_momentum(klines)
            rsi = self._calculate_rsi(klines)
            mean_reversion_strength = self._calculate_mean_reversion(klines)
            
            # Estimate spread from recent price action
            spread_estimate = self._estimate_spread(klines)
            
            snapshot = MarketSnapshot(
                timestamp=time.time(),
                price=current_price,
                volume=current_volume,
                spread=spread_estimate,
                order_book_imbalance=volume_momentum,  # Use volume momentum as proxy
                price_velocity=price_velocity,
                volatility_regime=volatility_regime,
                momentum=momentum,
                mean_reversion_strength=mean_reversion_strength
            )
            
            self.snapshot_history.append(snapshot)
            self._update_asset_characteristics(klines)
            
            return snapshot
            
        except Exception as e:
            logging.error(f"Error in crypto market analysis: {e}")
            return self._create_fallback_snapshot(exchange)
    
    def _fetch_recent_klines(self, exchange: Exchange) -> List:
        """Fetch recent 1-minute klines for real-time crypto analysis"""
        try:
            # Get last 30 minutes of 1-minute klines for detailed analysis
            klines = exchange.exchange.fetch_ohlcv(
                exchange._get_symbol_id(self.symbol),
                timeframe='1m',
                limit=30
            )
            
            # Convert to Binance kline format: [time, open, high, low, close, volume]
            formatted_klines = []
            for kline in klines:
                formatted_klines.append([
                    int(kline[0]),      # timestamp
                    str(kline[1]),      # open
                    str(kline[2]),      # high  
                    str(kline[3]),      # low
                    str(kline[4]),      # close
                    str(kline[5])       # volume
                ])
            
            return formatted_klines
            
        except Exception as e:
            logging.error(f"Error fetching klines: {e}")
            return []
    
    def _create_fallback_snapshot(self, exchange: Exchange) -> MarketSnapshot:
        """Create fallback snapshot using ticker data only"""
        try:
            ticker = exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            volume = float(ticker.get('quoteVolume', 0))
            
            return MarketSnapshot(
                timestamp=time.time(),
                price=current_price,
                volume=volume,
                spread=0.001,  # Default 0.1% spread estimate
                order_book_imbalance=0,
                price_velocity=0,
                volatility_regime=1.0,
                momentum=0,
                mean_reversion_strength=1.0
            )
        except:
            return MarketSnapshot(time.time(), 0, 0, 0, 0, 0, 1, 0, 1)
    
    def _calculate_price_velocity(self, klines: List) -> float:
        """Calculate crypto price velocity (rate of change per minute)"""
        if len(klines) < 5:
            return 0
        
        # Use last 5 minutes for velocity calculation
        recent_prices = [float(k[4]) for k in klines[-5:]]
        
        if len(recent_prices) < 2:
            return 0
        
        # Calculate percentage change per minute
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        time_span = len(recent_prices) - 1  # minutes
        
        velocity = price_change / time_span if time_span > 0 else 0
        
        # Normalize to [-1, 1] range for crypto (can move 5%+ per minute)
        return max(-1, min(1, velocity * 20))
    
    def _calculate_crypto_momentum(self, klines: List) -> float:
        """Calculate crypto momentum using multiple timeframes"""
        if len(klines) < 15:
            return 0
        
        prices = [float(k[4]) for k in klines]
        
        # Multi-timeframe momentum (crypto-optimized)
        short_momentum = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0  # 3-min
        medium_momentum = (prices[-1] - prices[-7]) / prices[-7] if len(prices) >= 7 else 0  # 7-min  
        long_momentum = (prices[-1] - prices[-15]) / prices[-15] if len(prices) >= 15 else 0  # 15-min
        
        # Weight shorter timeframes more heavily for crypto
        momentum = short_momentum * 0.6 + medium_momentum * 0.3 + long_momentum * 0.1
        
        # Normalize for crypto volatility (can easily move 2-3% in minutes)
        return max(-1, min(1, momentum * 30))
    
    def _calculate_crypto_volatility(self, klines: List) -> float:
        """Calculate volatility regime using high-low ranges"""
        if len(klines) < 10:
            return 1.0
        
        # Calculate minute-by-minute volatility
        recent_volatilities = []
        historical_volatilities = []
        
        for i, kline in enumerate(klines):
            high = float(kline[2])
            low = float(kline[3])
            close = float(kline[4])
            
            # Calculate intra-minute volatility
            if close > 0:
                volatility = (high - low) / close
                
                if i >= len(klines) - 5:  # Recent 5 minutes
                    recent_volatilities.append(volatility)
                historical_volatilities.append(volatility)
        
        if not recent_volatilities or not historical_volatilities:
            return 1.0
        
        recent_vol = sum(recent_volatilities) / len(recent_volatilities)
        hist_vol = sum(historical_volatilities) / len(historical_volatilities)
        
        return recent_vol / hist_vol if hist_vol > 0 else 1.0
    
    def _calculate_volume_momentum(self, klines: List) -> float:
        """Calculate volume momentum as proxy for order flow imbalance"""
        if len(klines) < 10:
            return 0
        
        volumes = [float(k[5]) for k in klines]
        prices = [float(k[4]) for k in klines]
        
        # Calculate volume-weighted price momentum
        recent_volumes = volumes[-5:]
        recent_prices = prices[-5:]
        historical_avg_volume = sum(volumes) / len(volumes)
        
        if historical_avg_volume == 0:
            return 0
        
        # Weight price changes by relative volume
        weighted_momentum = 0
        total_weight = 0
        
        for i in range(1, len(recent_prices)):
            price_change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            volume_weight = recent_volumes[i] / historical_avg_volume
            
            weighted_momentum += price_change * volume_weight
            total_weight += volume_weight
        
        if total_weight == 0:
            return 0
        
        momentum = weighted_momentum / total_weight
        
        # Normalize to [-1, 1] range
        return max(-1, min(1, momentum * 50))
    
    def _calculate_rsi(self, klines: List, period: int = 14) -> float:
        """Calculate RSI for crypto (optimized for short periods)"""
        if len(klines) < period + 1:
            return 50  # Neutral RSI
        
        prices = [float(k[4]) for k in klines]
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [max(0, delta) for delta in deltas]
        losses = [abs(min(0, delta)) for delta in deltas]
        
        # Calculate average gains and losses
        if len(gains) < period:
            return 50
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100  # No losses = overbought
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        self.rsi_history.append(rsi)
        return rsi
    
    def _calculate_mean_reversion(self, klines: List) -> float:
        """Calculate mean reversion strength for crypto"""
        if len(klines) < 20:
            return 1.0
        
        prices = [float(k[4]) for k in klines]
        
        # Calculate returns
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if len(returns) < 10:
            return 1.0
        
        # Calculate first-order autocorrelation
        mean_return = sum(returns) / len(returns)
        
        numerator = sum((returns[i] - mean_return) * (returns[i-1] - mean_return) 
                       for i in range(1, len(returns)))
        denominator = sum((r - mean_return) ** 2 for r in returns)
        
        if denominator == 0:
            return 1.0
        
        autocorr = numerator / denominator
        
        # Convert to mean reversion strength (negative correlation = strong mean reversion)
        mean_reversion = max(0.1, min(2.0, 1 - autocorr))
        
        return mean_reversion
    
    def _estimate_spread(self, klines: List) -> float:
        """Estimate bid-ask spread from recent price action"""
        if len(klines) < 5:
            return 0.001  # Default 0.1% spread
        
        # Use recent high-low ranges as spread proxy
        recent_spreads = []
        
        for kline in klines[-5:]:
            high = float(kline[2])
            low = float(kline[3])
            close = float(kline[4])
            
            if close > 0:
                spread = (high - low) / close
                recent_spreads.append(spread)
        
        if not recent_spreads:
            return 0.001
        
        avg_spread = sum(recent_spreads) / len(recent_spreads)
        
        # Cap spread estimate at reasonable levels
        return min(0.01, max(0.0001, avg_spread))  # 0.01% to 1%
    
    def _update_asset_characteristics(self, klines: List):
        """Update learned crypto asset characteristics"""
        if len(klines) < 30:
            return
        
        prices = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        # Calculate typical volatility
        returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                  for i in range(1, len(prices)) if prices[i-1] > 0]
        
        if returns:
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
            self.asset_characteristics['typical_volatility'] = volatility
        
        # Calculate volume profile
        if volumes:
            avg_volume = sum(volumes) / len(volumes)
            self.asset_characteristics['volume_profile'] = avg_volume
        
        # Calculate momentum persistence
        if len(self.rsi_history) > 10:
            rsi_values = list(self.rsi_history)
            rsi_trend = sum(1 for i in range(1, len(rsi_values)) 
                           if (rsi_values[i] - rsi_values[i-1]) * (rsi_values[i-1] - rsi_values[i-2]) > 0)
            persistence = rsi_trend / max(1, len(rsi_values) - 2)
            self.asset_characteristics['momentum_persistence'] = persistence
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
                 enable_samig: bool = True):
        """Initialize enhanced grid strategy with optional SAMIG features"""
        
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        
        # Core parameters
        self.price_lower = float(price_lower)
        self.price_upper = float(price_upper)
        self.grid_number = int(grid_number)
        self.investment = float(investment)
        self.take_profit_pnl = float(take_profit_pnl)
        self.stop_loss_pnl = float(stop_loss_pnl)
        self.grid_id = grid_id
        self.leverage = float(leverage)
        self.enable_grid_adaptation = enable_grid_adaptation
        self.enable_samig = enable_samig
        
        # Base parameters for SAMIG reference
        self.base_price_lower = self.price_lower
        self.base_price_upper = self.price_upper
        self.base_grid_number = self.grid_number
        
        # Grid calculations
        self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
        self.investment_per_grid = self.investment / self.grid_number
        
        # SAMIG components (optional)
        if self.enable_samig:
            self.market_intelligence = MarketIntelligenceEngine(symbol)
            self.parameter_manager = AdaptiveParameterManager()
            self.performance_tracker = deque(maxlen=50)
            self.current_market_snapshot = None
            self.adaptation_count = 0
        
        # Grid state
        self.grid_orders = {}
        self.pnl = 0.0
        self.initial_investment = investment
        self.trades_count = 0
        self.running = False
        self.grid_adjustments_count = 0
        
        # Market information
        self._fetch_market_info()
    
    def _fetch_market_info(self):
        """Fetch market information for the trading pair"""
        try:
            market_info = self.exchange.get_market_info(self.symbol)
            self.price_precision = market_info['precision']['price']
            self.amount_precision = market_info['precision']['amount']
            self.min_amount = market_info['limits']['amount']['min']
            self.min_cost = market_info.get('limits', {}).get('cost', {}).get('min', 0)
        except Exception as e:
            self.logger.error(f"Error fetching market info for {self.symbol}: {e}")
            self.price_precision = 2
            self.amount_precision = 6
            self.min_amount = 0.0
            self.min_cost = 0.0
    
    def _calculate_dynamic_grid_parameters(self) -> Dict:
        """Calculate dynamic grid parameters using SAMIG if enabled"""
        if not self.enable_samig:
            return {
                'price_lower': self.price_lower,
                'price_upper': self.price_upper,
                'grid_number': self.grid_number,
                'long_exposure_limit': self.investment,
                'short_exposure_limit': self.investment,
                'grid_interval': self.grid_interval,
                'order_size_multiplier': 1.0
            }
        
        try:
            # Get market intelligence
            market_snapshot = self.market_intelligence.analyze_market_conditions(self.exchange)
            self.current_market_snapshot = market_snapshot
            
            # Get adaptive parameters
            grid_density = self.parameter_manager.get_parameter('grid_density')
            trend_sensitivity = self.parameter_manager.get_parameter('trend_sensitivity')
            volatility_response = self.parameter_manager.get_parameter('volatility_response')
            momentum_threshold = self.parameter_manager.get_parameter('momentum_threshold')
            mean_reversion_factor = self.parameter_manager.get_parameter('mean_reversion_factor')
            exposure_asymmetry = self.parameter_manager.get_parameter('exposure_asymmetry')
            
            # Calculate dynamic parameters
            current_price = market_snapshot.price
            price_range = self.base_price_upper - self.base_price_lower
            
            # Volatility adjustment
            vol_adjustment = 1 + (market_snapshot.volatility_regime - 1) * volatility_response
            adjusted_range = price_range * vol_adjustment
            
            # Momentum-based center shift
            momentum_shift = market_snapshot.momentum * trend_sensitivity * price_range * 0.1
            grid_center = current_price + momentum_shift
            
            dynamic_price_lower = grid_center - adjusted_range / 2
            dynamic_price_upper = grid_center + adjusted_range / 2
            
            # Ensure positive prices
            if dynamic_price_lower <= 0:
                dynamic_price_lower = current_price * 0.5
                dynamic_price_upper = dynamic_price_lower + adjusted_range
            
            # Dynamic grid density
            if abs(market_snapshot.momentum) > momentum_threshold:
                dynamic_grid_number = max(3, int(self.base_grid_number * 0.7))
            else:
                dynamic_grid_number = int(self.base_grid_number * grid_density * mean_reversion_factor)
            
            # Asymmetric exposure limits
            base_exposure = self.investment
            
            if market_snapshot.order_book_imbalance > 0:
                long_exposure_limit = base_exposure * exposure_asymmetry
                short_exposure_limit = base_exposure / exposure_asymmetry
            elif market_snapshot.order_book_imbalance < 0:
                long_exposure_limit = base_exposure / exposure_asymmetry
                short_exposure_limit = base_exposure * exposure_asymmetry
            else:
                long_exposure_limit = short_exposure_limit = base_exposure
            
            # Momentum bias
            if market_snapshot.momentum > momentum_threshold:
                long_exposure_limit *= (1 + abs(market_snapshot.momentum))
            elif market_snapshot.momentum < -momentum_threshold:
                short_exposure_limit *= (1 + abs(market_snapshot.momentum))
            
            return {
                'price_lower': dynamic_price_lower,
                'price_upper': dynamic_price_upper,
                'grid_number': dynamic_grid_number,
                'long_exposure_limit': long_exposure_limit,
                'short_exposure_limit': short_exposure_limit,
                'grid_interval': (dynamic_price_upper - dynamic_price_lower) / dynamic_grid_number,
                'order_size_multiplier': 1 + market_snapshot.volatility_regime * 0.2
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating dynamic parameters: {e}")
            # Fallback to static parameters
            return {
                'price_lower': self.price_lower,
                'price_upper': self.price_upper,
                'grid_number': self.grid_number,
                'long_exposure_limit': self.investment,
                'short_exposure_limit': self.investment,
                'grid_interval': self.grid_interval,
                'order_size_multiplier': 1.0
            }
    
    def _round_price(self, price: float) -> float:
        """Round price according to market precision"""
        try:
            if hasattr(self, 'price_precision') and isinstance(self.price_precision, int):
                decimals = self.price_precision
            else:
                if price < 0.1:
                    decimals = 8
                elif price < 10:
                    decimals = 6
                elif price < 1000:
                    decimals = 4
                else:
                    decimals = 2
            
            return round(price, decimals)
        except Exception as e:
            self.logger.error(f"Error rounding price {price}: {e}")
            return float(price)
    
    def _round_amount(self, amount: float) -> float:
        """Round amount according to market precision"""
        try:
            if hasattr(self, 'amount_precision') and isinstance(self.amount_precision, int):
                decimals = self.amount_precision
            else:
                decimals = 6
            
            return round(amount, decimals)
        except Exception as e:
            self.logger.error(f"Error rounding amount {amount}: {e}")
            return float(amount)
    
    def _calculate_grid_levels(self, dynamic_params: Dict = None) -> List[float]:
        """Calculate grid price levels"""
        try:
            if dynamic_params:
                price_lower = dynamic_params['price_lower']
                price_upper = dynamic_params['price_upper']
                grid_number = dynamic_params['grid_number']
            else:
                price_lower = self.price_lower
                price_upper = self.price_upper
                grid_number = self.grid_number
            
            grid_interval = (price_upper - price_lower) / grid_number
            levels = []
            
            for i in range(grid_number + 1):
                level = price_lower + (i * grid_interval)
                levels.append(self._round_price(level))
            
            return levels
        except Exception as e:
            self.logger.error(f"Error calculating grid levels: {e}")
            return []
    
    def _calculate_order_amount(self, dynamic_params: Dict = None) -> float:
        """Calculate order amount per grid level"""
        try:
            multiplier = dynamic_params.get('order_size_multiplier', 1.0) if dynamic_params else 1.0
            
            ticker = self.exchange.get_ticker(self.symbol)
            price = float(ticker['last'])
            
            amount = (self.investment_per_grid * self.leverage * multiplier) / price
            amount = max(amount, self.min_amount)
            
            return self._round_amount(amount)
        except Exception as e:
            self.logger.error(f"Error calculating order amount: {e}")
            return self.min_amount
    
    def _get_directional_exposure(self) -> Dict[str, float]:
        """Get current directional exposure breakdown"""
        try:
            positions = self.exchange.get_positions(self.symbol)
            net_position_value = 0.0
            
            for position in positions:
                initial_margin = float(position.get('initialMargin', 0))
                side = position.get('side', '')
                
                if side == 'long':
                    net_position_value += initial_margin
                elif side == 'short':
                    net_position_value -= initial_margin
            
            open_orders = self.exchange.get_open_orders(self.symbol)
            potential_long_exposure = 0.0
            potential_short_exposure = 0.0
            
            for order in open_orders:
                if order['id'] in self.grid_orders:
                    if order['side'] == 'buy':
                        potential_long_exposure += self.investment_per_grid
                    elif order['side'] == 'sell':
                        potential_short_exposure += self.investment_per_grid
            
            total_potential_long = max(0, net_position_value) + potential_long_exposure
            total_potential_short = abs(min(0, net_position_value)) + potential_short_exposure
            
            return {
                'net_position_value': net_position_value,
                'current_long_value': max(0, net_position_value),
                'current_short_value': abs(min(0, net_position_value)),
                'potential_long_exposure': potential_long_exposure,
                'potential_short_exposure': potential_short_exposure,
                'total_potential_long': total_potential_long,
                'total_potential_short': total_potential_short,
                'remaining_long_budget': max(0, self.investment - total_potential_long),
                'remaining_short_budget': max(0, self.investment - total_potential_short)
            }
        except Exception as e:
            self.logger.error(f"Error getting directional exposure: {e}")
            return {
                'net_position_value': 0, 'current_long_value': 0, 'current_short_value': 0,
                'potential_long_exposure': 0, 'potential_short_exposure': 0,
                'total_potential_long': 0, 'total_potential_short': 0,
                'remaining_long_budget': self.investment, 'remaining_short_budget': self.investment
            }
    
    def _can_place_buy_order(self, dynamic_params: Dict = None) -> bool:
        """Check if we can place a buy order"""
        try:
            exposure = self._get_directional_exposure()
            limit = dynamic_params.get('long_exposure_limit', self.investment) if dynamic_params else self.investment
            return exposure['remaining_long_budget'] >= self.investment_per_grid and exposure['total_potential_long'] < limit
        except Exception as e:
            self.logger.error(f"Error checking buy order capability: {e}")
            return False
    
    def _can_place_sell_order(self, dynamic_params: Dict = None) -> bool:
        """Check if we can place a sell order"""
        try:
            exposure = self._get_directional_exposure()
            limit = dynamic_params.get('short_exposure_limit', self.investment) if dynamic_params else self.investment
            return exposure['remaining_short_budget'] >= self.investment_per_grid and exposure['total_potential_short'] < limit
        except Exception as e:
            self.logger.error(f"Error checking sell order capability: {e}")
            return False
    
    def setup_grid(self) -> None:
        """Setup the grid with enhanced adaptive capabilities"""
        try:
            # Get dynamic parameters
            dynamic_params = self._calculate_dynamic_grid_parameters()
            
            if self.enable_samig and self.current_market_snapshot:
                self.logger.info(f"SAMIG Setup - Market: Vol={self.current_market_snapshot.volatility_regime:.2f}, "
                               f"Momentum={self.current_market_snapshot.momentum:.3f}")
                self.logger.info(f"Dynamic: Grids={dynamic_params['grid_number']}, "
                               f"Range=[{dynamic_params['price_lower']:.6f}, {dynamic_params['price_upper']:.6f}]")
            
            # Update current parameters with dynamic values
            self.price_lower = dynamic_params['price_lower']
            self.price_upper = dynamic_params['price_upper']
            self.grid_number = dynamic_params['grid_number']
            self.grid_interval = dynamic_params['grid_interval']
            
            grid_levels = self._calculate_grid_levels(dynamic_params)
            amount = self._calculate_order_amount(dynamic_params)
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            self.logger.info(f"Setting up enhanced grid for {self.symbol}")
            self.logger.info(f"Price range: {self.price_lower:.6f} - {self.price_upper:.6f}")
            self.logger.info(f"Grid levels: {self.grid_number}, Current price: {current_price}")
            
            # Cancel existing orders
            try:
                self.exchange.cancel_all_orders(self.symbol)
                time.sleep(2)
            except Exception as e:
                self.logger.warning(f"Error cancelling orders: {e}")
            
            self.grid_orders = {}
            orders_placed = 0
            
            # Place initial grid orders
            for i in range(len(grid_levels) - 1):
                buy_price = grid_levels[i]
                sell_price = grid_levels[i + 1]
                
                # Place buy orders below current price
                if buy_price < current_price and self._can_place_buy_order(dynamic_params):
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'buy', amount, buy_price)
                        time.sleep(1)
                        
                        order_status = self.exchange.get_order_status(order['id'], self.symbol)
                        if order_status and order_status['status'] in ['open', 'new', 'partially_filled']:
                            self.grid_orders[order['id']] = {
                                'type': 'buy',
                                'price': buy_price,
                                'amount': amount,
                                'status': 'open',
                                'grid_level': i
                            }
                            orders_placed += 1
                    except Exception as e:
                        self.logger.error(f"Failed to place buy order at {buy_price}: {e}")
                
                # Place sell orders above current price
                if sell_price > current_price and self._can_place_sell_order(dynamic_params):
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'sell', amount, sell_price)
                        time.sleep(1)
                        
                        order_status = self.exchange.get_order_status(order['id'], self.symbol)
                        if order_status and order_status['status'] in ['open', 'new', 'partially_filled']:
                            self.grid_orders[order['id']] = {
                                'type': 'sell',
                                'price': sell_price,
                                'amount': amount,
                                'status': 'open',
                                'grid_level': i + 1
                            }
                            orders_placed += 1
                    except Exception as e:
                        self.logger.error(f"Failed to place sell order at {sell_price}: {e}")
            
            if orders_placed > 0:
                self.running = True
                self.logger.info(f"Grid setup complete with {orders_placed} orders")
            else:
                self.running = False
                self.logger.warning("Grid setup failed - no orders placed")
        
        except Exception as e:
            self.logger.error(f"Error setting up grid: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.running = False
    
    def update_grid(self) -> None:
        """Update grid with controlled SAMIG intelligence"""
        try:
            if not self.running:
                return
            
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            # SAMIG regime change detection with dampening
            if self.enable_samig:
                market_snapshot = self.market_intelligence.analyze_market_conditions(self.exchange)
                
                # Only check for regime changes, don't reconfigure on every update
                if self._detect_regime_change(market_snapshot):
                    self.logger.info("SAMIG: Major regime change detected - reconfiguring grid")
                    self.setup_grid()
                    return
                
                # Update current snapshot for next comparison
                self.current_market_snapshot = market_snapshot
                
                # Performance evaluation with reduced frequency
                self._evaluate_performance_and_adapt()
            
            # Standard grid adaptation (less aggressive than SAMIG)
            elif self.enable_grid_adaptation and self._is_price_outside_grid(current_price):
                self._adapt_grid_to_price(current_price)
                return
            
            # Regular order updates
            self._update_orders()
            self._check_tp_sl()
            
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
    
    def _detect_regime_change(self, new_snapshot: MarketSnapshot) -> bool:
        """Detect significant market regime changes with dampening to prevent over-adaptation"""
        if not self.current_market_snapshot:
            self.current_market_snapshot = new_snapshot
            return False
        
        # Minimum time between regime changes (prevent thrashing)
        min_regime_interval = 300  # 5 minutes minimum between reconfigurations
        time_since_last_adaptation = time.time() - getattr(self, 'last_regime_change_time', 0)
        
        if time_since_last_adaptation < min_regime_interval:
            return False
        
        old = self.current_market_snapshot
        new = new_snapshot
        
        # Calculate relative changes with higher thresholds for grid trading
        vol_change = abs(new.volatility_regime - old.volatility_regime) / (old.volatility_regime + 0.1)
        momentum_change = abs(new.momentum - old.momentum)
        mr_change = abs(new.mean_reversion_strength - old.mean_reversion_strength) / (old.mean_reversion_strength + 0.1)
        
        # Much higher thresholds for grid trading stability
        vol_threshold = 0.8     # 80% volatility change required
        momentum_threshold = 0.5  # 50% momentum change required
        mr_threshold = 0.6      # 60% mean reversion change required
        
        # Require multiple indicators to confirm regime change
        significant_changes = 0
        if vol_change > vol_threshold:
            significant_changes += 1
        if momentum_change > momentum_threshold:
            significant_changes += 1
        if mr_change > mr_threshold:
            significant_changes += 1
        
        # Require at least 2 out of 3 indicators to trigger regime change
        regime_change = significant_changes >= 2
        
        if regime_change:
            # Add regime change history tracking
            if not hasattr(self, 'regime_change_history'):
                self.regime_change_history = deque(maxlen=10)
            
            self.regime_change_history.append(time.time())
            
            # Check if we're changing regimes too frequently (dampening)
            recent_changes = sum(1 for t in self.regime_change_history 
                            if time.time() - t < 1800)  # 30 minutes
            
            if recent_changes > 3:  # Max 3 regime changes per 30 minutes
                self.logger.info("Regime change dampening: Too many recent changes, skipping reconfiguration")
                return False
            
            self.last_regime_change_time = time.time()
            self.logger.info(f"Regime change confirmed: Vol Delta={vol_change:.3f}, Mom Delta={momentum_change:.3f}, "
                            f"MR Delta={mr_change:.3f} (Changes: {significant_changes}/3)")
        
        return regime_change
    
    def _evaluate_performance_and_adapt(self):
        """Evaluate performance and adapt SAMIG parameters with less frequency"""
        if not self.enable_samig or len(self.performance_tracker) < 15:  # Increased minimum samples
            return
        
        try:
            # Only evaluate every 10 updates to reduce noise
            if not hasattr(self, 'evaluation_counter'):
                self.evaluation_counter = 0
            
            self.evaluation_counter += 1
            if self.evaluation_counter % 10 != 0:
                return
            
            # Use longer period for performance evaluation
            recent_pnl = sum([p.pnl for p in list(self.performance_tracker)[-15:]])
            performance_score = recent_pnl / self.investment * 100
            
            # Only adapt if performance score is significant
            if abs(performance_score) > 0.5:  # Only adapt for >0.5% performance changes
                if self.current_market_snapshot:
                    self.parameter_manager.update_parameter('grid_density', performance_score, self.current_market_snapshot)
                    self.parameter_manager.update_parameter('trend_sensitivity', performance_score, self.current_market_snapshot)
                    self.parameter_manager.update_parameter('volatility_response', performance_score, self.current_market_snapshot)
                    
                    self.adaptation_count += 1
                    self.logger.info(f"SAMIG Adaptation #{self.adaptation_count}: Performance={performance_score:.2f}%")
        
        except Exception as e:
            self.logger.error(f"Error in performance evaluation: {e}")
    
    def _is_price_outside_grid(self, current_price: float) -> bool:
        """Check if price is outside grid boundaries with larger buffer"""
        # Increase buffer to 2% to reduce unnecessary adaptations
        buffer_size = (self.price_upper - self.price_lower) * 0.02
        
        outside_range = (current_price < (self.price_lower - buffer_size) or 
                        current_price > (self.price_upper + buffer_size))
        
        if outside_range:
            # Add time-based dampening for regular grid adaptation too
            if not hasattr(self, 'last_adaptation_time'):
                self.last_adaptation_time = 0
            
            time_since_adaptation = time.time() - self.last_adaptation_time
            min_adaptation_interval = 180  # 3 minutes minimum between regular adaptations
            
            if time_since_adaptation < min_adaptation_interval:
                return False
            
            self.last_adaptation_time = time.time()
            self.logger.info(f"Price {current_price} is outside grid range [{self.price_lower:.6f} - {self.price_upper:.6f}]")
        
        return outside_range
    
    def _adapt_grid_to_price(self, current_price: float) -> None:
        """Adapt grid to price movement"""
        try:
            self.logger.info(f"Grid adaptation: Price {current_price} outside range")
            
            grid_size = self.price_upper - self.price_lower
            self.price_lower = current_price - (grid_size / 2)
            self.price_upper = current_price + (grid_size / 2)
            
            if self.price_lower <= 0:
                self.price_lower = 0.00001
                self.price_upper = self.price_lower + grid_size
            
            self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
            
            try:
                self.exchange.cancel_all_orders(self.symbol)
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
            
            self.grid_orders = {}
            self.setup_grid()
            self.grid_adjustments_count += 1
            
        except Exception as e:
            self.logger.error(f"Error in grid adaptation: {e}")
    
    def _update_orders(self):
        """Update order status and handle fills"""
        try:
            open_orders = self.exchange.get_open_orders(self.symbol)
            current_order_ids = {order['id']: order for order in open_orders}
            
            for order_id in list(self.grid_orders.keys()):
                order_info = self.grid_orders[order_id]
                
                if order_id not in current_order_ids and order_info['status'] == 'open':
                    try:
                        order_status = self.exchange.get_order_status(order_id, self.symbol)
                        
                        if order_status['status'] in ['filled', 'closed']:
                            order_info['status'] = 'filled'
                            self.trades_count += 1
                            
                            # Track performance for SAMIG
                            if self.enable_samig:
                                self._track_performance()
                            
                            self._calculate_pnl()
                            self._place_counter_order(order_info)
                        
                        elif order_status['status'] == 'canceled':
                            order_info['status'] = 'cancelled'
                            
                    except Exception as e:
                        self.logger.error(f"Error checking order {order_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error updating orders: {e}")
    
    def _track_performance(self):
        """Track performance metrics for SAMIG learning"""
        if not self.enable_samig or not self.current_market_snapshot:
            return
        
        try:
            metric = GridPerformanceMetric(
                timestamp=time.time(),
                pnl=self.pnl,
                win_rate=self.trades_count / max(1, len(self.grid_orders)),
                trades_count=self.trades_count,
                market_conditions=self.current_market_snapshot,
                grid_config={
                    'price_lower': self.price_lower,
                    'price_upper': self.price_upper,
                    'grid_number': self.grid_number,
                    'leverage': self.leverage
                }
            )
            self.performance_tracker.append(metric)
        
        except Exception as e:
            self.logger.error(f"Error tracking performance: {e}")
    
    def _place_counter_order(self, filled_order: Dict) -> None:
        """Place counter order after fill"""
        try:
            grid_level = filled_order['grid_level']
            price = filled_order['price']
            amount = filled_order['amount']
            
            if filled_order['type'] == 'buy':
                sell_price = self._round_price(price + self.grid_interval)
                if sell_price <= self.price_upper and self._can_place_sell_order():
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'sell', amount, sell_price)
                        self.grid_orders[order['id']] = {
                            'type': 'sell',
                            'price': sell_price,
                            'amount': amount,
                            'status': 'open',
                            'grid_level': grid_level + 1
                        }
                    except Exception as e:
                        self.logger.error(f"Error placing counter sell order: {e}")
            
            elif filled_order['type'] == 'sell':
                buy_price = self._round_price(price - self.grid_interval)
                if buy_price >= self.price_lower and self._can_place_buy_order():
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'buy', amount, buy_price)
                        self.grid_orders[order['id']] = {
                            'type': 'buy',
                            'price': buy_price,
                            'amount': amount,
                            'status': 'open',
                            'grid_level': grid_level - 1
                        }
                    except Exception as e:
                        self.logger.error(f"Error placing counter buy order: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in counter order placement: {e}")
    
    def _calculate_pnl(self) -> None:
        """Calculate PnL from positions"""
        try:
            positions = self.exchange.get_positions(self.symbol)
            unrealized_pnl = sum(float(pos.get('unrealizedPnl', 0)) for pos in positions)
            self.pnl = unrealized_pnl
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
    
    def _check_tp_sl(self) -> None:
        """Check take profit and stop loss conditions"""
        try:
            pnl_percentage = (self.pnl / self.initial_investment) * 100
            
            if pnl_percentage >= self.take_profit_pnl:
                self.logger.info(f"Take profit reached: {pnl_percentage:.2f}%")
                self.stop_grid()
            elif pnl_percentage <= -self.stop_loss_pnl:
                self.logger.info(f"Stop loss reached: {pnl_percentage:.2f}%")
                self.stop_grid()
        except Exception as e:
            self.logger.error(f"Error checking TP/SL: {e}")
    
    def stop_grid(self) -> None:
        """Stop the grid strategy"""
        try:
            if not self.running:
                return
            
            self.logger.info(f"Stopping grid for {self.symbol}")
            
            # Cancel orders
            try:
                self.exchange.cancel_all_orders(self.symbol)
                for order_id in self.grid_orders:
                    if self.grid_orders[order_id]['status'] == 'open':
                        self.grid_orders[order_id]['status'] = 'cancelled'
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
            
            # Close positions
            try:
                positions = self.exchange.get_positions(self.symbol)
                for position in positions:
                    if abs(float(position['contracts'])) > 0:
                        side = 'sell' if position['side'] == 'long' else 'buy'
                        self.exchange.create_market_order(self.symbol, side, abs(float(position['contracts'])))
            except Exception as e:
                self.logger.error(f"Error closing positions: {e}")
            
            self.running = False
            self.logger.info(f"Grid stopped for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error stopping grid: {e}")
            self.running = False
    
    def get_status(self) -> Dict:
        """Get comprehensive grid status including SAMIG metrics"""
        try:
            exposure = self._get_directional_exposure()
            
            status = {
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'display_symbol': self.original_symbol,
                'price_lower': self.price_lower,
                'price_upper': self.price_upper,
                'grid_number': self.grid_number,
                'grid_interval': self.grid_interval,
                'investment': self.investment,
                'investment_per_grid': self.investment_per_grid,
                'current_long_value': exposure['current_long_value'],
                'current_short_value': exposure['current_short_value'],
                'remaining_long_budget': exposure['remaining_long_budget'],
                'remaining_short_budget': exposure['remaining_short_budget'],
                'take_profit_pnl': self.take_profit_pnl,
                'stop_loss_pnl': self.stop_loss_pnl,
                'leverage': self.leverage,
                'enable_grid_adaptation': self.enable_grid_adaptation,
                'enable_samig': self.enable_samig,
                'grid_adjustments_count': self.grid_adjustments_count,
                'pnl': self.pnl,
                'pnl_percentage': (self.pnl / self.initial_investment) * 100 if self.initial_investment else 0,
                'trades_count': self.trades_count,
                'running': self.running,
                'orders_count': len([o for o in self.grid_orders.values() if o.get('status') == 'open'])
            }
            
            # Add SAMIG-specific metrics
            if self.enable_samig and self.current_market_snapshot:
                status.update({
                    'samig_active': True,
                    'adaptation_count': self.adaptation_count,
                    'volatility_regime': self.current_market_snapshot.volatility_regime,
                    'momentum': self.current_market_snapshot.momentum,
                    'mean_reversion_strength': self.current_market_snapshot.mean_reversion_strength,
                    'order_book_imbalance': self.current_market_snapshot.order_book_imbalance,
                    'grid_density': self.parameter_manager.get_parameter('grid_density'),
                    'trend_sensitivity': self.parameter_manager.get_parameter('trend_sensitivity'),
                    'volatility_response': self.parameter_manager.get_parameter('volatility_response')
                })
            else:
                status['samig_active'] = False
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting grid status: {e}")
            return {}