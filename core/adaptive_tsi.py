"""
FIXED: Momentum-Based Adaptive TSI System for Crypto Trading
Root Cause: Overly restrictive crossover detection missing momentum opportunities

ORIGINAL ISSUES:
1. Exact crossover detection only - misses momentum phases
2. 6 validation steps ALL must pass - too restrictive
3. Static thresholds - not adaptive to market conditions
4. No momentum consideration - ignores TSI velocity

FIXES APPLIED:
1. Momentum-based signal detection instead of exact crossovers
2. Simplified to 2 key validations: momentum + context
3. Dynamic thresholds based on recent TSI behavior
4. TSI velocity and acceleration consideration
"""

import logging
import time
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional
from collections import deque
from dataclasses import dataclass

@dataclass
class TSIParameters:
    """Simplified TSI parameters for crypto momentum"""
    slow: int = 25
    fast: int = 13  
    signal: int = 7

class MomentumTSISystem:
    """
    FIXED: Momentum-based TSI system optimized for crypto trading.
    Focuses on TSI direction and velocity rather than exact crossovers.
    """
    
    def __init__(self, symbol: str):
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        self.symbol = symbol
        
        # Fixed parameters optimized for crypto
        self.params = TSIParameters()
        
        # Higher thresholds for accuracy
        self.momentum_threshold = 2.0   # Increased from 0.5
        self.strength_threshold = 5.0   # Increased from 2.0
        
        # Signal persistence (prevent spam)
        self.last_signal = 'none'
        self.last_signal_time = 0
        self.signal_cooldown = 30  # 30 seconds between signals
        
        # Recent data for dynamic thresholds
        self.recent_tsi_data = deque(maxlen=50)
        
        self.logger.info(f"🎯 FIXED Momentum TSI System initialized for {symbol}")

    def get_technical_direction(self, exchange) -> str:
        """
        FIXED: Momentum-based signal detection
        Returns: 'buy', 'sell', or 'none'
        """
        try:
            # Prevent signal spam
            if time.time() - self.last_signal_time < self.signal_cooldown:
                return 'none'
            
            # Get OHLCV data
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=100)
            if not ohlcv_data or len(ohlcv_data) < 50:
                return 'none'
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close'] = df['close'].astype(float)
            
            # Calculate TSI
            tsi_result = ta.tsi(df['close'], slow=self.params.slow, fast=self.params.fast, signal=self.params.signal)
            if tsi_result is None or len(tsi_result.dropna()) < 20:
                return 'none'
            
            tsi_clean = tsi_result.dropna()
            if len(tsi_clean.columns) < 2:
                return 'none'
            
            tsi_line = tsi_clean.iloc[:, 0]
            tsi_signal_line = tsi_clean.iloc[:, 1]
            
            # Update dynamic thresholds
            self._update_dynamic_thresholds(tsi_line)
            
            # Generate momentum-based signal
            signal = self._generate_momentum_signal(tsi_line, tsi_signal_line, df)
            
            # Update state if signal generated
            if signal != 'none':
                self.last_signal = signal
                self.last_signal_time = time.time()
                self.logger.info(f"🎯 MOMENTUM Signal: {signal.upper()}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error in momentum TSI: {e}")
            return 'none'

    def _update_dynamic_thresholds(self, tsi_line: pd.Series):
        """Update dynamic thresholds based on recent TSI behavior"""
        try:
            if len(tsi_line) < 10:
                return
                
            # Store recent TSI values
            recent_values = tsi_line.tail(10).tolist()
            self.recent_tsi_data.extend(recent_values)
            
            # Calculate dynamic thresholds
            if len(self.recent_tsi_data) >= 20:
                tsi_array = np.array(list(self.recent_tsi_data))
                tsi_volatility = np.std(tsi_array)
                tsi_range = np.max(tsi_array) - np.min(tsi_array)
                
                # Adaptive thresholds based on recent volatility
                self.momentum_threshold = max(0.3, min(2.0, tsi_volatility * 0.5))
                self.strength_threshold = max(1.0, min(5.0, tsi_range * 0.2))
                
        except Exception:
            pass  # Use default thresholds on error

    def _generate_momentum_signal(self, tsi_line: pd.Series, tsi_signal_line: pd.Series, df: pd.DataFrame) -> str:
        """
        FIXED: Generate signals based on TSI momentum rather than crossovers
        """
        try:
            if len(tsi_line) < 5:
                return 'none'
            
            # Current values
            current_tsi = tsi_line.iloc[-1]
            current_signal = tsi_signal_line.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # TSI momentum (velocity)
            tsi_momentum = tsi_line.iloc[-1] - tsi_line.iloc[-3]  # 2-period momentum
            signal_momentum = tsi_signal_line.iloc[-1] - tsi_signal_line.iloc[-3]
            
            # TSI acceleration (change in momentum)
            prev_momentum = tsi_line.iloc[-2] - tsi_line.iloc[-4]
            tsi_acceleration = tsi_momentum - prev_momentum
            
            # Price momentum for confirmation
            price_momentum = (current_price - df['close'].iloc[-3]) / df['close'].iloc[-3] * 100
            
            self.logger.info(f"🔍 TSI: {current_tsi:.2f} | Signal: {current_signal:.2f} | Momentum: {tsi_momentum:.2f}")
            
            # SIGNAL LOGIC - Higher accuracy requirements
            
            # Calculate signal strength first
            signal_strength = abs(tsi_momentum) + abs(current_tsi - current_signal) * 0.5
            tsi_separation = abs(current_tsi - current_signal)
            
            # 1. STRONG BULLISH MOMENTUM (stricter requirements)
            if (tsi_momentum > self.momentum_threshold and 
                current_tsi > current_signal and
                tsi_separation > 5.0 and     # Must have significant separation
                tsi_acceleration > 0.5 and   # Strong acceleration
                price_momentum > -0.5 and    # Price not declining
                signal_strength > self.strength_threshold):
                return 'buy'
            
            # 2. STRONG BEARISH MOMENTUM (stricter requirements)
            elif (tsi_momentum < -self.momentum_threshold and 
                  current_tsi < current_signal and
                  tsi_separation > 5.0 and     # Must have significant separation
                  tsi_acceleration < -0.5 and  # Strong deceleration
                  price_momentum < 0.5 and     # Price not rising
                  signal_strength > self.strength_threshold):
                return 'sell'
            
            # 3. MOMENTUM REVERSAL SIGNALS
            elif (current_tsi < -20 and  # Oversold region
                  tsi_momentum > self.momentum_threshold * 0.7 and  # Building upward momentum
                  tsi_acceleration > 0):  # Accelerating recovery
                return 'buy'
                
            elif (current_tsi > 20 and   # Overbought region
                  tsi_momentum < -self.momentum_threshold * 0.7 and  # Building downward momentum
                  tsi_acceleration < 0):  # Accelerating decline
                return 'sell'
            
            # 4. SIGNAL LINE MOMENTUM SIGNALS
            elif (abs(current_tsi - current_signal) > 3.0 and  # Significant separation
                  signal_momentum * tsi_momentum > 0 and      # Both moving same direction
                  abs(tsi_momentum) > self.momentum_threshold * 0.5):
                
                if tsi_momentum > 0 and current_tsi > current_signal:
                    return 'buy'
                elif tsi_momentum < 0 and current_tsi < current_signal:
                    return 'sell'
            
            return 'none'
            
        except Exception as e:
            self.logger.error(f"❌ Error generating momentum signal: {e}")
            return 'none'

    def get_system_status(self) -> Dict:
        """Get current system status"""
        try:
            return {
                'system_type': 'momentum_based',
                'momentum_threshold': self.momentum_threshold,
                'strength_threshold': self.strength_threshold,
                'signal_cooldown': self.signal_cooldown,
                'last_signal': self.last_signal,
                'parameters': self.params.__dict__
            }
        except Exception:
            return {}


def integrate_momentum_tsi(grid_strategy_instance):
    """
    FIXED: Integration function for momentum-based TSI
    """
    # Initialize momentum TSI system
    momentum_tsi = MomentumTSISystem(symbol=grid_strategy_instance.symbol)
    
    # Replace the method
    def momentum_get_technical_direction():
        return momentum_tsi.get_technical_direction(grid_strategy_instance.exchange)
    
    # Monkey patch the method
    grid_strategy_instance._get_technical_direction = momentum_get_technical_direction
    grid_strategy_instance._momentum_tsi_system = momentum_tsi
    
    grid_strategy_instance.logger.info("🔧 FIXED Momentum TSI System integrated!")
    
    return momentum_tsi