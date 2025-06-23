

"""
Adaptive Crypto Signal System - Multi-Indicator Strategy
Replaces slow TSI with fast crypto-optimized indicators: QQE, Supertrend, Fisher Transform, VWAP

ENTRY SIGNALS ONLY - No exit logic, relies on existing SL/TP orders
"""

import logging
import time
import os
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import threading
import json

@dataclass
class SignalParameters:
    """Parameters for multi-indicator crypto signals"""
    qqe_length: int = 14
    qqe_smooth: int = 5
    supertrend_period: int = 7
    supertrend_multiplier: float = 3.0
    fisher_period: int = 9
    vwap_bands_std: float = 2.0 # This parameter is not part of the optimization loop in this version
    
    # Performance tracking
    accuracy: float = 0.0
    total_signals: int = 0
    winning_signals: int = 0
    last_used: float = 0.0

class AdaptiveCryptoSignals:
    """
    Fast multi-indicator system optimized for volatile crypto markets.
    Combines QQE, Supertrend, Fisher Transform, and VWAP for high accuracy.
    
    ENTRY SIGNALS ONLY - No exit logic, relies on existing SL/TP orders
    """
    
    def __init__(self, symbol: str, config_file: str = "data/crypto_signal_configs.json"):
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        self.symbol = symbol
        self.config_file = config_file
        
        # Load or initialize parameters
        self.params = self._load_symbol_config()
        
        # Signal tracking
        self.signal_performance = self._load_signal_history()
        self.last_signal = 'none'
        self.last_signal_time = 0
        self.signal_cooldown = 15  # 15 seconds between signals
        self.signal_stability_period = 30  # 30 seconds before allowing opposite signal
        
        # Market state
        self.current_trend = 'neutral'
        self.volatility_level = 'normal'
        self.last_optimization = time.time()
        
        # Optimization settings
        self.min_signals_for_optimization = 5 # Min signals evaluated before opt can run
        self.optimization_interval = 300  # 5 minutes, if enough signals evaluated
        
        self.logger.info(f"🚀 Adaptive Crypto Signals (ENTRY ONLY) initialized for {symbol} "
                        f"[QQE:{self.params.qqe_length}, ST:{self.params.supertrend_period}, "
                        f"Fisher:{self.params.fisher_period}] "
                        f"[Accuracy: {self.params.accuracy:.1f}%]")

    def get_technical_direction(self, exchange) -> str:
        """Main entry point - generates fast, accurate signals for crypto"""
        try:
            # Rate limiting - allow testing without cooldown
            current_time = time.time()
            if hasattr(self, '_force_signal') and self._force_signal:
                pass  # Skip cooldown for testing
            elif current_time - self.last_signal_time < self.signal_cooldown:
                return 'none'  # FIXED: Return 'none' instead of last signal to avoid conflicts
            
            # Get OHLCV data - we need volume for VWAP
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=1400)
            if not ohlcv_data or len(ohlcv_data) < 30: # Min data for basic indicators
                self.logger.debug(f"Insufficient data for {self.symbol}: {len(ohlcv_data) if ohlcv_data else 0} candles")
                return 'none'
            
            # Convert to DataFrame for pandas_ta
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # FIXED: Ensure datetime ordering for VWAP
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            # df.set_index('timestamp', inplace=True) # Keep timestamp as column for _quick_backtest simplicity
            
            # Convert to float
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Generate multi-indicator signal
            # For signal generation, set index for pandas_ta compatibility with existing _calculate methods
            df_indexed = df.set_index('timestamp')
            signal = self._generate_composite_signal(df_indexed) # Pass indexed df
            
            if signal != 'none':
                self._track_signal(signal, float(df['close'].iloc[-1])) # Use original df for price
                self.last_signal = signal
                self.last_signal_time = current_time
                
                self.logger.info(f"⚡ ENTRY SIGNAL: {signal.upper()} @ ${float(df['close'].iloc[-1]):.6f} "
                               f"[Trend: {self.current_trend}] [Vol: {self.volatility_level}] "
                               f"[Accuracy: {self.params.accuracy:.0f}%]")
            
            # Check if optimization needed (pass original non-indexed df for backtesting)
            if self._should_optimize():
                self._optimize_parameters(df.copy()) # Pass a copy of the non-indexed df
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 'none'

    def _generate_composite_signal(self, df: pd.DataFrame) -> str:
        """
        Generate signal using multiple indicators for confirmation.
        MODIFIED FOR CONSERVATISM.
        """
        try:
            # 1. Calculate QQE (momentum oscillator)
            qqe_result = self._calculate_qqe(df)
            if qqe_result is None:
                self.logger.debug(f"QQE calculation failed for {self.symbol}")
                return 'none'
            
            qqe_value = qqe_result['qqe']
            # qqe_signal_line = qqe_result['signal'] # Renamed for clarity
            qqe_direction = qqe_result['direction']
            
            # 2. Calculate Supertrend (trend direction)
            supertrend_result = self._calculate_supertrend(df)
            if supertrend_result is None:
                self.logger.debug(f"Supertrend calculation failed for {self.symbol}")
                return 'none'
            
            trend_direction = supertrend_result['direction']
            # trend_changed = supertrend_result['changed'] # Trend change signals commented out for conservatism
            
            # 3. Calculate Fisher Transform (reversal detection)
            fisher_result = self._calculate_fisher(df)
            if fisher_result is None:
                self.logger.debug(f"Fisher calculation failed for {self.symbol}")
                return 'none'
            
            fisher_value = fisher_result['fisher']
            # fisher_cross = fisher_result.get('cross', None)
            fisher_extreme = fisher_result['extreme']
            
            # 4. Calculate VWAP and bands (support/resistance)
            vwap_result = self._calculate_vwap_bands(df)
            if vwap_result is None:
                self.logger.debug(f"VWAP calculation failed for {self.symbol}")
                return 'none'
            
            price_position = vwap_result['position']
            # near_band = vwap_result['near_band']
            
            # Update market state
            self.current_trend = trend_direction
            self._update_volatility_level(df)
            
            # Store indicators for debugging/logging
            self.last_indicators = {
                'qqe': qqe_result,
                'supertrend': supertrend_result,
                'fisher': fisher_result,
                'vwap': vwap_result
            }
            
            # Debug logging
            self.logger.debug(f"{self.symbol} Indicators - QQE: {qqe_direction}, Trend: {trend_direction}, "
                            f"Fisher: {fisher_value:.2f} ({fisher_extreme or 'normal'}), "
                            f"VWAP: {price_position}, Vol: {self.volatility_level}")
            
            # SIGNAL GENERATION LOGIC (MODIFIED FOR CONSERVATISM)
            signal = 'none'
            
            time_since_last = time.time() - self.last_signal_time
            allow_opposite = time_since_last >= self.signal_stability_period
            
            # 1. Strong Momentum Signals (Stricter Fisher)
            if trend_direction == 'up' and qqe_direction == 'bullish' and \
               (fisher_value < -1.0 or (price_position == 'below_lower' and fisher_value < 0)): # Fisher more oversold or at VWAP support with confirming Fisher
                if not allow_opposite and self.last_signal == 'sell':
                    self.logger.debug(f"Skipping momentum BUY too soon after SELL ({time_since_last:.1f}s)")
                    return 'none'
                signal = 'buy'
                self.logger.debug("🟢 MOMENTUM BUY: Trend up + QQE bullish + (Fisher oversold OR VWAP support with Fisher confirm)")
                
            elif trend_direction == 'down' and qqe_direction == 'bearish' and \
                 (fisher_value > 1.0 or (price_position == 'above_upper' and fisher_value > 0)): # Fisher more overbought or at VWAP resistance with confirming Fisher
                if not allow_opposite and self.last_signal == 'buy':
                    self.logger.debug(f"Skipping momentum SELL too soon after BUY ({time_since_last:.1f}s)")
                    return 'none'
                signal = 'sell'
                self.logger.debug("🔴 MOMENTUM SELL: Trend down + QQE bearish + (Fisher overbought OR VWAP resistance with Fisher confirm)")
            
            # 2. Extreme Reversal Signals (Stricter: QQE must align if counter-Supertrend)
            elif fisher_extreme == 'oversold':
                # If counter-trend (Supertrend is down), QQE MUST be bullish. If trend is up, QQE non-bearish is enough.
                if (trend_direction == 'down' and qqe_direction == 'bullish') or \
                   (trend_direction == 'up' and qqe_direction != 'bearish'):
                    if not allow_opposite and self.last_signal == 'sell':
                        self.logger.debug(f"Skipping reversal BUY too soon after SELL ({time_since_last:.1f}s)")
                        return 'none'
                    signal = 'buy'
                    self.logger.debug("🟢 REVERSAL BUY: Fisher extreme oversold with QQE/Trend confirmation")
                
            elif fisher_extreme == 'overbought':
                 # If counter-trend (Supertrend is up), QQE MUST be bearish. If trend is down, QQE non-bullish is enough.
                if (trend_direction == 'up' and qqe_direction == 'bearish') or \
                   (trend_direction == 'down' and qqe_direction != 'bullish'):
                    if not allow_opposite and self.last_signal == 'buy':
                        self.logger.debug(f"Skipping reversal SELL too soon after BUY ({time_since_last:.1f}s)")
                        return 'none'
                    signal = 'sell'
                    self.logger.debug("🔴 REVERSAL SELL: Fisher extreme overbought with QQE/Trend confirmation")
            
            # 3. VWAP Band Bounce (Stricter: QQE must align, not just 'not opposite')
            # Only consider if volatility is not low, as low vol can lead to fakeouts near bands
            elif self.volatility_level != 'low':
                if price_position == 'below_lower' and qqe_direction == 'bullish': # QQE must be bullish
                    if not allow_opposite and self.last_signal == 'sell':
                        self.logger.debug(f"Skipping VWAP BUY too soon after SELL ({time_since_last:.1f}s)")
                        return 'none'
                    signal = 'buy'
                    self.logger.debug("🟢 VWAP BUY: Price below lower band with QQE bullish")
                elif price_position == 'above_upper' and qqe_direction == 'bearish': # QQE must be bearish
                    if not allow_opposite and self.last_signal == 'buy':
                        self.logger.debug(f"Skipping VWAP SELL too soon after BUY ({time_since_last:.1f}s)")
                        return 'none'
                    signal = 'sell'
                    self.logger.debug("🔴 VWAP SELL: Price above upper band with QQE bearish")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error in composite signal generation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 'none'

    def evaluate_exit_conditions(self, position_side: str, entry_price: float, current_price: float) -> Dict:
        """
        Evaluate exit conditions and return decision data for _make_order_decision()
        
        FIXED to prevent 2-second exits:
        - Added 60-second minimum position time before non-emergency exits
        - Increased stop loss threshold from -2% to -3%
        - Increased momentum thresholds for exits
        - Only exit on actual Supertrend changes, not just direction
        - Increased profit targets from 1% to 2%
        - Added profitability check for Fisher reversals
        
        Returns dict with:
        - should_exit: bool
        - exit_reason: str
        - exit_urgency: str ('immediate', 'normal', 'none')
        """
        try:
            result = {
                'should_exit': False,
                'exit_reason': '',
                'exit_urgency': 'none'
            }
            
            # Debug log entry
            self.logger.debug(f"Evaluating exit for {position_side} position: Entry ${entry_price:.6f}, Current ${current_price:.6f}")
            
            if not hasattr(self, 'last_indicators') or not self.last_indicators:
                self.logger.debug("No indicators available for exit evaluation")
                return result
            
            # Calculate position PnL
            if position_side == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # short
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # FIXED: Add minimum position time before allowing exits (except stop loss)
            # If position_entry_time is 0 or not set, assume position is old enough
            entry_time = getattr(self, 'position_entry_time', 0)
            if entry_time > 0:
                position_time = time.time() - entry_time
            else:
                position_time = 999  # Assume old position if entry time not tracked
                
            min_position_time = 60  # 60 seconds minimum before non-emergency exits
            
            # Debug logging for position time
            if position_time < min_position_time:
                self.logger.debug(f"Position age: {position_time:.1f}s (min: {min_position_time}s)")
            
            # Get current indicators
            qqe = self.last_indicators.get('qqe', {})
            supertrend = self.last_indicators.get('supertrend', {})
            fisher = self.last_indicators.get('fisher', {})
            vwap = self.last_indicators.get('vwap', {})
            
            # 1. EMERGENCY STOP LOSS (Always active)
            if pnl_pct < -3.0:  # FIXED: Increased from -2.0% to -3.0%
                result['should_exit'] = True
                result['exit_reason'] = f"STOP LOSS triggered (PnL: {pnl_pct:.2f}%)"
                result['exit_urgency'] = 'immediate'
                return result
            
            # FIXED: Skip other exit checks if position is too new
            if position_time < min_position_time:
                self.logger.debug(f"Position too new for exit evaluation: {position_time:.1f}s < {min_position_time}s")
                return result
            
            # 2. TREND REVERSAL (Only after minimum time)
            if position_side == 'long' and supertrend.get('direction') == 'down':
                # FIXED: Check if trend just changed or has been down for a while
                if supertrend.get('changed', False):
                    result['should_exit'] = True
                    result['exit_reason'] = f"Supertrend flipped DOWN (PnL: {pnl_pct:.2f}%)"
                    result['exit_urgency'] = 'normal'  # FIXED: Changed from 'immediate'
                    return result
                    
            elif position_side == 'short' and supertrend.get('direction') == 'up':
                if supertrend.get('changed', False):
                    result['should_exit'] = True
                    result['exit_reason'] = f"Supertrend flipped UP (PnL: {pnl_pct:.2f}%)"
                    result['exit_urgency'] = 'normal'
                    return result
            
            # 3. LOSING POSITION + STRONG MOMENTUM REVERSAL
            # FIXED: Increased loss threshold and added stronger momentum check
            if pnl_pct < -2.0:  # FIXED: Increased from -1.0% to -2.0%
                if (position_side == 'long' and qqe.get('direction') == 'bearish' and 
                    qqe.get('momentum', 0) < -1.0):  # FIXED: Stronger momentum required
                    result['should_exit'] = True
                    result['exit_reason'] = f"Strong bearish momentum + losing (PnL: {pnl_pct:.2f}%)"
                    result['exit_urgency'] = 'normal'
                    return result
                elif (position_side == 'short' and qqe.get('direction') == 'bullish' and 
                      qqe.get('momentum', 0) > 1.0):
                    result['should_exit'] = True
                    result['exit_reason'] = f"Strong bullish momentum + losing (PnL: {pnl_pct:.2f}%)"
                    result['exit_urgency'] = 'normal'
                    return result
            
            # 4. TAKE PROFIT AT VWAP BANDS
            # FIXED: Increased profit threshold
            if pnl_pct > 2.0:  # FIXED: Increased from 1.0% to 2.0%
                if (position_side == 'long' and vwap.get('position') == 'above_upper'):
                    result['should_exit'] = True
                    result['exit_reason'] = f"Take profit at VWAP resistance (PnL: {pnl_pct:.2f}%)"
                    result['exit_urgency'] = 'normal'
                    return result
                elif (position_side == 'short' and vwap.get('position') == 'below_lower'):
                    result['should_exit'] = True
                    result['exit_reason'] = f"Take profit at VWAP support (PnL: {pnl_pct:.2f}%)"
                    result['exit_urgency'] = 'normal'
                    return result
            
            # 5. FISHER EXTREME REVERSALS (Only with strong momentum)
            # FIXED: Added profit check and stronger momentum requirement
            if position_time > 120:  # Only after 2 minutes
                if position_side == 'long' and fisher.get('extreme') == 'overbought':
                    if fisher.get('momentum', 0) < -1.0 and pnl_pct > 0:  # FIXED: Only exit if profitable
                        result['should_exit'] = True
                        result['exit_reason'] = f"Fisher overbought reversal (PnL: {pnl_pct:.2f}%)"
                        result['exit_urgency'] = 'normal'
                        return result
                elif position_side == 'short' and fisher.get('extreme') == 'oversold':
                    if fisher.get('momentum', 0) > 1.0 and pnl_pct > 0:
                        result['should_exit'] = True
                        result['exit_reason'] = f"Fisher oversold reversal (PnL: {pnl_pct:.2f}%)"
                        result['exit_urgency'] = 'normal'
                        return result
            
            # 6. TIME-BASED EXIT for stagnant positions
            # FIXED: Increased time and added movement threshold
            if position_time > 900:  # 15 minutes
                if abs(pnl_pct) < 0.3:  # Less than 0.3% movement
                    result['should_exit'] = True
                    result['exit_reason'] = f"Position stagnant for 15 mins (PnL: {pnl_pct:.2f}%)"
                    result['exit_urgency'] = 'normal'
                    return result
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit conditions: {e}")
            return {'should_exit': False, 'exit_reason': '', 'exit_urgency': 'none'}

    def should_exit_on_opposite_signal(self, position_side: str, new_signal: str) -> bool:
        """
        Quick check if position should exit based on opposite signal
        Used for compatibility with existing code
        
        Args:
            position_side: 'long' or 'short'
            new_signal: 'buy' or 'sell'
            
        Returns:
            bool: True if opposite signal detected
        """
        if position_side == 'long' and new_signal == 'sell':
            return True
        elif position_side == 'short' and new_signal == 'buy':
            return True
        return False

    def _calculate_qqe(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate QQE indicator with proper error handling"""
        try:
            # Ensure df has a DatetimeIndex for pandas_ta if not already set
            # df_indexed = df if isinstance(df.index, pd.DatetimeIndex) else df.set_index('timestamp')
            
            qqe = ta.qqe(df['close'], length=self.params.qqe_length, smooth=self.params.qqe_smooth)
            
            if qqe is None or qqe.empty:
                return None
            
            qqe_cols = qqe.columns.tolist()
            if len(qqe_cols) < 2: # QQE typically returns at least 2 columns
                self.logger.debug(f"QQE result has fewer than 2 columns: {qqe_cols}")
                return None
            
            qqe_line = qqe.iloc[:, 0].dropna()
            qqe_signal_line = qqe.iloc[:, 1].dropna() # Renamed for clarity
            
            if len(qqe_line) < 2 or len(qqe_signal_line) < 2 :
                return None
            
            current_qqe = float(qqe_line.iloc[-1])
            current_signal_val = float(qqe_signal_line.iloc[-1]) # Renamed variable
            prev_qqe = float(qqe_line.iloc[-2])
            
            # Determine direction
            direction = 'neutral'
            if current_qqe > current_signal_val and current_qqe > prev_qqe:
                direction = 'bullish'
            elif current_qqe < current_signal_val and current_qqe < prev_qqe:
                direction = 'bearish'
            
            return {
                'qqe': current_qqe,
                'signal': current_signal_val, # Use renamed var
                'direction': direction,
                'momentum': current_qqe - prev_qqe
            }
            
        except Exception as e:
            self.logger.debug(f"QQE calculation error: {e}")
            return None

    def _calculate_supertrend(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Supertrend indicator"""
        try:
            # df_indexed = df if isinstance(df.index, pd.DatetimeIndex) else df.set_index('timestamp')
            st = ta.supertrend(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                length=self.params.supertrend_period,
                multiplier=self.params.supertrend_multiplier
            )
            
            if st is None or st.empty:
                return None
            
            direction_col = None
            for col in st.columns:
                if 'SUPERTd' in col: # Standard column name for direction in pandas_ta
                    direction_col = col
                    break
            
            if direction_col is None:
                self.logger.debug(f"Supertrend direction column not found in {st.columns}")
                return None
            
            directions = st[direction_col].dropna()
            if len(directions) < 2:
                return None
            
            current_dir = int(directions.iloc[-1])
            prev_dir = int(directions.iloc[-2])
            
            trend_changed = False
            if current_dir != prev_dir:
                if len(directions) >= 3:
                    prev_prev_dir = int(directions.iloc[-3])
                    if prev_dir == prev_prev_dir and current_dir != prev_dir:
                        trend_changed = True
                else: # Not enough history to confirm, but it did change from the immediate previous
                    trend_changed = True 
            
            trend_direction = 'up' if current_dir == 1 else 'down'
            
            return {
                'direction': trend_direction,
                'changed': trend_changed,
                'strength': abs(current_dir) # Not really strength, just direction as 1 or -1
            }
            
        except Exception as e:
            self.logger.debug(f"Supertrend calculation error: {e}")
            return None

    def _calculate_fisher(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Fisher Transform for reversal detection"""
        try:
            # df_indexed = df if isinstance(df.index, pd.DatetimeIndex) else df.set_index('timestamp')
            fisher = ta.fisher(
                high=df['high'],
                low=df['low'],
                length=self.params.fisher_period
            )
            
            if fisher is None or fisher.empty:
                return None
            
            fisher_cols = fisher.columns.tolist()
            if len(fisher_cols) < 2: # Fisher typically returns 2 columns
                self.logger.debug(f"Fisher result has fewer than 2 columns: {fisher_cols}")
                return None

            fisher_line = fisher.iloc[:, 0].dropna()
            fisher_signal_line = fisher.iloc[:, 1].dropna() # Renamed for clarity
            
            if len(fisher_line) < 2 or len(fisher_signal_line) < 2:
                return None
            
            current_fisher = float(fisher_line.iloc[-1])
            current_signal_val = float(fisher_signal_line.iloc[-1]) # Renamed variable
            prev_fisher = float(fisher_line.iloc[-2])
            prev_signal_val = float(fisher_signal_line.iloc[-2]) # Renamed variable
            
            extreme = None
            if current_fisher > 1.5:
                extreme = 'overbought'
            elif current_fisher < -1.5:
                extreme = 'oversold'
            
            cross_signal = None
            if prev_fisher <= prev_signal_val and current_fisher > current_signal_val:
                cross_signal = 'bullish_cross'
            elif prev_fisher >= prev_signal_val and current_fisher < current_signal_val:
                cross_signal = 'bearish_cross'
            
            return {
                'fisher': current_fisher,
                'signal': current_signal_val, # Use renamed var
                'extreme': extreme,
                'cross': cross_signal,
                'momentum': current_fisher - prev_fisher
            }
            
        except Exception as e:
            self.logger.debug(f"Fisher Transform calculation error: {e}")
            return None

    def _calculate_vwap_bands(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate VWAP with bands for dynamic support/resistance"""
        try:
            # df_indexed = df if isinstance(df.index, pd.DatetimeIndex) else df.set_index('timestamp')
            vwap = ta.vwap(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            
            if vwap is None or vwap.empty:
                return None
            
            vwap_values = vwap.dropna()
            if len(vwap_values) < 5: # Need some values for VWAP to be meaningful
                return None
            
            current_vwap = float(vwap_values.iloc[-1])
            current_price = float(df['close'].iloc[-1]) # df is already indexed or has 'close'
            
            # Calculate standard deviation for bands using recent close prices from df
            # Ensure enough data points for std calculation
            recent_closes_count = 20
            if len(df['close']) < recent_closes_count:
                return None # Not enough data for std calculation

            recent_closes = df['close'].iloc[-recent_closes_count:].values 
            std_dev = np.std(recent_closes)
            
            upper_band = current_vwap + (self.params.vwap_bands_std * std_dev)
            lower_band = current_vwap - (self.params.vwap_bands_std * std_dev)
            
            position = 'neutral'
            if current_price > upper_band:
                position = 'above_upper'
            elif current_price < lower_band:
                position = 'below_lower'
            elif current_price > current_vwap:
                position = 'above_vwap'
            else:
                position = 'below_vwap'
            
            band_distance = std_dev * 0.3
            near_band = None
            if abs(current_price - upper_band) < band_distance:
                near_band = 'upper'
            elif abs(current_price - lower_band) < band_distance:
                near_band = 'lower'
            
            return {
                'vwap': current_vwap,
                'upper': upper_band,
                'lower': lower_band,
                'position': position,
                'near_band': near_band,
                'price_pct_from_vwap': ((current_price - current_vwap) / current_vwap) * 100 if current_vwap != 0 else 0
            }
            
        except Exception as e:
            self.logger.debug(f"VWAP calculation error: {e}")
            return None

    def _update_volatility_level(self, df: pd.DataFrame):
        """Update current volatility level"""
        try:
            # df_indexed = df if isinstance(df.index, pd.DatetimeIndex) else df.set_index('timestamp')
            atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
            if atr is not None and len(atr.dropna()) > 0:
                current_atr = float(atr.dropna().iloc[-1])
                avg_price = float(df['close'].iloc[-1]) # df is already indexed or has 'close'
                
                if avg_price == 0: # Avoid division by zero
                    self.volatility_level = 'normal'
                    return

                atr_pct = (current_atr / avg_price) * 100
                
                if atr_pct > 2.0:
                    self.volatility_level = 'high'
                elif atr_pct < 0.5:
                    self.volatility_level = 'low'
                else:
                    self.volatility_level = 'normal'
            else: # Default if ATR calculation fails
                self.volatility_level = 'normal'      
        except Exception:
            self.volatility_level = 'normal'

    def _track_signal(self, signal: str, price: float):
        """Track signal for performance evaluation"""
        try:
            signal_data = {
                'signal': signal,
                'price': price,
                'timestamp': time.time(),
                'params': { # Store params used for this signal
                    'qqe_length': self.params.qqe_length,
                    'qqe_smooth': self.params.qqe_smooth,
                    'supertrend_period': self.params.supertrend_period,
                    'supertrend_multiplier': self.params.supertrend_multiplier,
                    'fisher_period': self.params.fisher_period
                },
                'market_state': {
                    'trend': self.current_trend,
                    'volatility': self.volatility_level
                }
            }
            
            self.signal_performance.append(signal_data)
            
            # Keep only last 100 signals (already handled by deque maxlen)
            
            # Evaluate performance (evaluates older signals)
            self._evaluate_signals() # This will now use the refined logic
            
            # Save periodically
            if len(self.signal_performance) % 5 == 0: # Save every 5 new signals
                self._save_config()
                
        except Exception as e:
            self.logger.error(f"Error tracking signal: {e}")

    def _evaluate_signals(self):
        """
        Evaluate signal performance.
        MODIFIED FOR MORE PROFITABILITY-ALIGNED EVALUATION.
        """
        try:
            current_time = time.time()
            evaluated_in_this_run = 0 # To track how many signals are processed in this call
            
            # --- MODIFIED EVALUATION PARAMETERS ---
            eval_period_seconds = 30 * 60  # Evaluate after 30 minutes
            min_price_move_pct = 0.5       # Expect at least 0.5% move in signal direction
            # --- END MODIFIED EVALUATION PARAMETERS ---

            # Iterate through a copy for safe modification if needed, though not modifying here
            for signal_data in list(self.signal_performance): 
                if 'evaluated' in signal_data and signal_data['evaluated']:
                    continue # Already evaluated
                
                # Wait for 'eval_period_seconds' before attempting evaluation
                if current_time - signal_data['timestamp'] < eval_period_seconds:
                    continue # Not old enough to evaluate
                
                # Find the first price point in self.signal_performance that occurred
                # AT LEAST eval_period_seconds AFTER signal_data['timestamp'].
                # This still relies on other signals being generated for future price points.
                # This is a limitation of not having full historical data access here for evaluation.
                future_price_point = None
                future_timestamp = float('inf')

                for future_signal_candidate in self.signal_performance:
                    candidate_ts = future_signal_candidate['timestamp']
                    # Check if candidate is after the original signal + eval_period
                    # and is earlier than any previously found valid future_signal_candidate
                    if candidate_ts >= (signal_data['timestamp'] + eval_period_seconds) and candidate_ts < future_timestamp:
                        future_price_point = future_signal_candidate['price']
                        future_timestamp = candidate_ts # Update to find the earliest possible one

                if future_price_point is not None:
                    future_price = future_price_point
                    price_change_pct = ((future_price - signal_data['price']) / signal_data['price']) * 100

                    signal_data['correct'] = False # Default to incorrect
                    if signal_data['signal'] == 'buy' and price_change_pct > min_price_move_pct:
                        signal_data['correct'] = True
                    elif signal_data['signal'] == 'sell' and price_change_pct < -min_price_move_pct: # price_change_pct is negative
                        signal_data['correct'] = True
                    
                    signal_data['evaluated'] = True
                    signal_data['price_change_eval'] = price_change_pct # Store the evaluated price change
                    evaluated_in_this_run += 1
                # If no suitable future_price_point found yet, it will be tried again in the next call.

            # Update overall accuracy statistics based on ALL evaluated signals in history
            total_evaluated_signals = sum(1 for s in self.signal_performance if s.get('evaluated', False))
            total_correct_signals = sum(1 for s in self.signal_performance if s.get('correct', False))
            
            if total_evaluated_signals > 0:
                self.params.accuracy = (total_correct_signals / total_evaluated_signals) * 100
                self.params.total_signals = total_evaluated_signals # Total signals ever evaluated
                self.params.winning_signals = total_correct_signals
            else: # Reset if no signals evaluated yet
                self.params.accuracy = 0.0
                self.params.total_signals = 0
                self.params.winning_signals = 0

            if evaluated_in_this_run > 0:
                 self.logger.debug(f"Evaluated {evaluated_in_this_run} signals. Current accuracy: {self.params.accuracy:.2f}%")
                
        except Exception as e:
            self.logger.error(f"Error evaluating signals: {e}")

    def _should_optimize(self) -> bool:
        """Check if parameter optimization is needed"""
        current_time = time.time()
        
        # Time-based check (don't optimize too frequently)
        if current_time - self.last_optimization < self.optimization_interval:
            return False
        
        # Performance-based check (ensure enough signals have been evaluated)
        # self.params.total_signals is the count of *evaluated* signals
        if self.params.total_signals >= self.min_signals_for_optimization:
            # Consider optimizing if accuracy is low (e.g., < 40%) or to explore
            if self.params.accuracy < 45.0:  # Example threshold
                return True
        
        return False

    def _optimize_parameters(self, df: pd.DataFrame):
        """
        Optimize indicator parameters based on recent performance using _quick_backtest.
        `df` should be the raw DataFrame with 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
        """
        try:
            self.logger.info(f"🔧 Optimizing parameters for {self.symbol}...")
            
            # Parameter ranges to test
            param_ranges = {
                'qqe_length': [10, 14, 20],
                'qqe_smooth': [3, 5, 7], # Keeping this test range simple
                'supertrend_period': [7, 10, 14], # Adjusted ST periods
                'supertrend_multiplier': [2.0, 3.0], # ST multiplier
                'fisher_period': [7, 9, 12, 15] # Fisher periods
            }
            
            # Use current accuracy as baseline, but quick_backtest returns win rate
            # We want to maximize the win rate from _quick_backtest
            best_score = -1.0 # Initialize with a low score
            best_params_dict = None # Store as dict first
            
            current_params_for_backtest = SignalParameters(
                qqe_length=self.params.qqe_length,
                qqe_smooth=self.params.qqe_smooth,
                supertrend_period=self.params.supertrend_period,
                supertrend_multiplier=self.params.supertrend_multiplier,
                fisher_period=self.params.fisher_period
            )
            # Evaluate current parameters using the same backtest logic for a fair comparison
            # Note: The 'accuracy' field in SignalParameters is from live evaluation, not backtest score.
            # We are trying to find params that perform well in backtest.
            
            # Iterate through combinations (simplified grid search)
            # This can be computationally intensive if ranges are large.
            num_combinations = 0
            for qqe_len in param_ranges['qqe_length']:
                for qqe_sm in param_ranges['qqe_smooth']:
                    for st_period in param_ranges['supertrend_period']:
                        for st_mult in param_ranges['supertrend_multiplier']:
                            for fisher_p in param_ranges['fisher_period']:
                                num_combinations +=1
                                test_signal_params = SignalParameters(
                                    qqe_length=qqe_len,
                                    qqe_smooth=qqe_sm,
                                    supertrend_period=st_period,
                                    supertrend_multiplier=st_mult,
                                    fisher_period=fisher_p
                                )
                                
                                # Perform quick backtest
                                # df here is the full historical data passed for optimization
                                score = self._quick_backtest(df.copy(), test_signal_params) 
                                self.logger.debug(f"Test params: QQE({qqe_len},{qqe_sm}), ST({st_period},{st_mult}), F({fisher_p}) -> Score: {score:.2f}%")

                                if score > best_score:
                                    best_score = score
                                    best_params_dict = {
                                        'qqe_length': qqe_len, 'qqe_smooth': qqe_sm,
                                        'supertrend_period': st_period, 'supertrend_multiplier': st_mult,
                                        'fisher_period': fisher_p
                                    }
            
            self.logger.info(f"Optimization: Tested {num_combinations} combinations. Best backtest score: {best_score:.2f}%")

            # Update parameters if a better set is found
            # (e.g. if new backtest score is significantly better than what might be expected from current params)
            # For simplicity, if any positive score is found and better than current, update.
            # A more robust check might compare new best_score to backtest_score of current live params.
            if best_params_dict and best_score > 0: # Check if best_score is an improvement
                self.logger.info(f"📈 New best parameters found with backtest score {best_score:.2f}%. Updating live parameters.")
                self.params.qqe_length = best_params_dict['qqe_length']
                self.params.qqe_smooth = best_params_dict['qqe_smooth']
                self.params.supertrend_period = best_params_dict['supertrend_period']
                self.params.supertrend_multiplier = best_params_dict['supertrend_multiplier']
                self.params.fisher_period = best_params_dict['fisher_period']
                
                # Reset accuracy tracking as parameters have changed
                self.params.accuracy = 0.0
                self.params.total_signals = 0
                self.params.winning_signals = 0
                self.signal_performance.clear() # Clear old signal history as params changed

                self._save_config() # Save new parameters
            
            self.last_optimization = time.time() # Update optimization time
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _quick_backtest(self, df_hist: pd.DataFrame, params_to_test: SignalParameters) -> float:
        """
        Quick backtest for parameter optimization.
        `df_hist` is a DataFrame with 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
        Returns a score (e.g., win rate percentage).
        REWRITTEN TO PERFORM ACTUAL BACKTESTING.
        """
        try:
            # 1. Calculate all necessary indicators on the df_hist using params_to_test
            # Ensure df_hist is indexed by timestamp for pandas_ta
            # df_indexed = df_hist.set_index('timestamp') if not isinstance(df_hist.index, pd.DatetimeIndex) else df_hist.copy()
            # pandas_ta functions can often take Series directly, so indexing might not be strictly needed for all.
            # However, for consistency and to match how _calculate_ methods might expect it:
            df_for_indicators = df_hist.set_index('timestamp')


            # QQE
            qqe_df = ta.qqe(df_for_indicators['close'], length=params_to_test.qqe_length, smooth=params_to_test.qqe_smooth)
            if qqe_df is None or qqe_df.empty or len(qqe_df.columns) < 2: return 0.0 # Cannot backtest
            df_for_indicators['qqe_line'] = qqe_df.iloc[:, 0]
            df_for_indicators['qqe_signal_line'] = qqe_df.iloc[:, 1]

            # Supertrend
            st_df = ta.supertrend(df_for_indicators['high'], df_for_indicators['low'], df_for_indicators['close'], 
                                  length=params_to_test.supertrend_period, multiplier=params_to_test.supertrend_multiplier)
            if st_df is None or st_df.empty: return 0.0
            st_dir_col_name = [col for col in st_df.columns if 'SUPERTd' in col]
            if not st_dir_col_name: return 0.0
            df_for_indicators['st_dir'] = st_df[st_dir_col_name[0]]

            # Fisher Transform
            fisher_df = ta.fisher(df_for_indicators['high'], df_for_indicators['low'], length=params_to_test.fisher_period)
            if fisher_df is None or fisher_df.empty or len(fisher_df.columns) < 2: return 0.0
            df_for_indicators['fisher_line'] = fisher_df.iloc[:, 0]
            # df_for_indicators['fisher_signal_line'] = fisher_df.iloc[:, 1] # Fisher signal line not directly used in simplified logic

            # VWAP (bands are not used in this simplified backtest signal logic)
            vwap_s = ta.vwap(df_for_indicators['high'], df_for_indicators['low'], df_for_indicators['close'], df_for_indicators['volume'])
            if vwap_s is None or vwap_s.empty: return 0.0
            df_for_indicators['vwap'] = vwap_s
            
            # Drop NaN rows created by indicator calculations to align data
            df_for_indicators.dropna(inplace=True)
            if len(df_for_indicators) < 50: # Need enough data points post-NaN drop
                return 0.0

            # 2. Initialize backtest state
            trades = 0
            wins = 0
            position = None  # None, 'long', 'short'
            entry_price = 0.0
            
            # Simple TP/SL percentages for backtesting simulation
            # These are for the backtest simulation, not live trading
            tp_pct_backtest = 1.0  # 1.0% take profit
            sl_pct_backtest = 0.5  # 0.5% stop loss

            # 3. Iterate through the prepared DataFrame
            for i in range(1, len(df_for_indicators)): # Start from 1 for prev_ values
                current_price = df_for_indicators['close'].iloc[i]
                
                # Manage open position
                if position:
                    if position == 'long':
                        if current_price >= entry_price * (1 + tp_pct_backtest / 100):
                            wins += 1; trades += 1; position = None
                        elif current_price <= entry_price * (1 - sl_pct_backtest / 100):
                            trades += 1; position = None
                    elif position == 'short':
                        if current_price <= entry_price * (1 - tp_pct_backtest / 100):
                            wins += 1; trades += 1; position = None
                        elif current_price >= entry_price * (1 + sl_pct_backtest / 100):
                            trades += 1; position = None
                
                # Attempt to enter new position if none active
                if not position:
                    # Get indicator values for current candle i
                    qqe_val = df_for_indicators['qqe_line'].iloc[i]
                    qqe_sig_val = df_for_indicators['qqe_signal_line'].iloc[i]
                    prev_qqe_val = df_for_indicators['qqe_line'].iloc[i-1] # QQE momentum check
                    
                    bt_qqe_direction = 'neutral'
                    if qqe_val > qqe_sig_val and qqe_val > prev_qqe_val: bt_qqe_direction = 'bullish'
                    elif qqe_val < qqe_sig_val and qqe_val < prev_qqe_val: bt_qqe_direction = 'bearish'

                    bt_trend_direction = 'up' if df_for_indicators['st_dir'].iloc[i] == 1 else 'down'
                    
                    bt_fisher_val = df_for_indicators['fisher_line'].iloc[i]
                    bt_fisher_extreme = None
                    if bt_fisher_val > 1.5: bt_fisher_extreme = 'overbought'
                    elif bt_fisher_val < -1.5: bt_fisher_extreme = 'oversold'

                    # Simplified signal logic for backtest (mirroring conservative _generate_composite_signal)
                    bt_signal = 'none'
                    # Momentum
                    if bt_trend_direction == 'up' and bt_qqe_direction == 'bullish' and bt_fisher_val < -1.0:
                        bt_signal = 'buy'
                    elif bt_trend_direction == 'down' and bt_qqe_direction == 'bearish' and bt_fisher_val > 1.0:
                        bt_signal = 'sell'
                    # Reversal
                    elif bt_fisher_extreme == 'oversold' and \
                         ((bt_trend_direction == 'down' and bt_qqe_direction == 'bullish') or \
                          (bt_trend_direction == 'up' and bt_qqe_direction != 'bearish')):
                        bt_signal = 'buy'
                    elif bt_fisher_extreme == 'overbought' and \
                         ((bt_trend_direction == 'up' and bt_qqe_direction == 'bearish') or \
                          (bt_trend_direction == 'down' and bt_qqe_direction != 'bullish')):
                        bt_signal = 'sell'

                    if bt_signal == 'buy':
                        position = 'long'; entry_price = current_price
                    elif bt_signal == 'sell':
                        position = 'short'; entry_price = current_price
            
            # 4. Calculate and return score (win rate percentage)
            if trades == 0:
                return 0.0 
            win_rate = (wins / trades) * 100.0
            return win_rate

        except Exception as e:
            self.logger.error(f"Error in _quick_backtest: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0 # Return a neutral score on error

    def _load_symbol_config(self) -> SignalParameters:
        """Load saved configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
                    
                if self.symbol in configs:
                    config = configs[self.symbol]
                    # Ensure all fields from dataclass are present, use defaults if not
                    return SignalParameters(
                        qqe_length=config.get('qqe_length', SignalParameters.qqe_length),
                        qqe_smooth=config.get('qqe_smooth', SignalParameters.qqe_smooth),
                        supertrend_period=config.get('supertrend_period', SignalParameters.supertrend_period),
                        supertrend_multiplier=config.get('supertrend_multiplier', SignalParameters.supertrend_multiplier),
                        fisher_period=config.get('fisher_period', SignalParameters.fisher_period),
                        vwap_bands_std=config.get('vwap_bands_std',SignalParameters.vwap_bands_std),
                        accuracy=config.get('accuracy', SignalParameters.accuracy),
                        total_signals=config.get('total_signals', SignalParameters.total_signals),
                        winning_signals=config.get('winning_signals', SignalParameters.winning_signals)
                    )
        except Exception as e:
            self.logger.error(f"Error loading config for {self.symbol}: {e}. Using default parameters.")
        
        return SignalParameters() # Return default if any issue

    def _load_signal_history(self) -> deque:
        """Load signal history"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
                    
                if self.symbol in configs:
                    history = configs[self.symbol].get('signal_history', [])
                    # Ensure items in history are dicts, filter out Nones if any corruption
                    valid_history = [item for item in history if isinstance(item, dict)]
                    return deque(valid_history[-100:], maxlen=100)
        except Exception as e:
            self.logger.error(f"Error loading signal history for {self.symbol}: {e}. Starting fresh history.")
        
        return deque(maxlen=100)

    def _save_config(self):
        """Save configuration and signal history"""
        try:
            configs = {}
            if os.path.exists(self.config_file):
                try: # Protect against corrupted existing file
                    with open(self.config_file, 'r') as f:
                        configs = json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning(f"Config file {self.config_file} is corrupted. Will overwrite with new data.")
                    configs = {} # Start fresh if corrupted
            
            # Ensure signal_performance items are serializable (they should be dicts)
            serializable_history = [dict(item) for item in self.signal_performance]

            configs[self.symbol] = {
                'qqe_length': self.params.qqe_length,
                'qqe_smooth': self.params.qqe_smooth,
                'supertrend_period': self.params.supertrend_period,
                'supertrend_multiplier': self.params.supertrend_multiplier,
                'fisher_period': self.params.fisher_period,
                'vwap_bands_std': self.params.vwap_bands_std,
                'accuracy': self.params.accuracy,
                'total_signals': self.params.total_signals,
                'winning_signals': self.params.winning_signals,
                'signal_history': serializable_history[-100:], # Save last 100
                'last_updated': time.time()
            }
            
            # Ensure directory exists
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir): # Check if config_dir is not empty string
                os.makedirs(config_dir, exist_ok=True)
            
            # Atomic write: write to temp file then replace
            temp_file = self.config_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(configs, f, indent=2)
            os.replace(temp_file, self.config_file)
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'system_type': 'multi_indicator_crypto_entry_only',
            'indicators': {
                'qqe': {'length': self.params.qqe_length, 'smooth': self.params.qqe_smooth},
                'supertrend': {'period': self.params.supertrend_period, 'multiplier': self.params.supertrend_multiplier},
                'fisher': {'period': self.params.fisher_period},
                'vwap': {'bands_std': self.params.vwap_bands_std}
            },
            'performance': {
                'accuracy': self.params.accuracy,
                'total_signals': self.params.total_signals, # Evaluated signals
                'winning_signals': self.params.winning_signals
            },
            'market_state': {
                'trend': self.current_trend,
                'volatility': self.volatility_level
            },
            'last_signal_info': {
                'last_signal': self.last_signal,
                'last_signal_time': self.last_signal_time
            },
            'exit_logic': 'Using original SL/TP orders only (external to this module)'
        }


def integrate_adaptive_crypto_signals(strategy_instance, config_file: str = None):
    """Integration function to replace TSI with multi-indicator crypto signals - ENTRY ONLY"""
    if config_file is None:
        # Default config file path if not provided
        config_file = os.path.join(os.getcwd(), "data", "crypto_signal_configs.json") # More robust default path
    
    strategy_instance.logger.info(f"🔧 Integrating Multi-Indicator Crypto Signals (ENTRY ONLY) using config: {config_file}")
    
    # Create new signal system
    # Ensure strategy_instance.symbol is the base symbol if exchange-specific IDs are used elsewhere
    base_symbol = getattr(strategy_instance, 'original_symbol', strategy_instance.symbol)
    crypto_signals = AdaptiveCryptoSignals(symbol=base_symbol, config_file=config_file)
    
    # Replace ONLY the TSI method with new system
    def crypto_get_technical_direction():
        return crypto_signals.get_technical_direction(strategy_instance.exchange)
    
    def get_signal_status(): # Renamed from get_system_status to avoid potential clashes
        return crypto_signals.get_system_status()
    
    # Ensure test_indicators is bound to crypto_signals instance
    def test_crypto_indicators_wrapper():
        """Wrapper to call test_indicators on the crypto_signals instance."""
        return crypto_signals.test_indicators(strategy_instance.exchange)

    # Replace methods - NO EXIT LOGIC MODIFICATIONS
    strategy_instance._get_technical_direction = crypto_get_technical_direction
    strategy_instance.get_signal_status = get_signal_status # Use the new name
    strategy_instance.test_crypto_indicators = test_crypto_indicators_wrapper # Use the wrapper
    strategy_instance._crypto_signal_system = crypto_signals # Store instance for direct access if needed
    
    # Test indicators on first run
    try:
        # Initial call to get_technical_direction can also serve as a test
        # as it will log indicator calculations if debug level is set.
        # Explicitly calling test_crypto_indicators might be redundant if get_technical_direction runs soon.
        strategy_instance.logger.info(f"Initial signal check for {base_symbol} to warm up indicators...")
        # initial_signal = crypto_get_technical_direction()
        # strategy_instance.logger.info(f"Initial signal for {base_symbol}: {initial_signal}")
        
        # Or, call the test function directly:
        test_result = strategy_instance.test_crypto_indicators()
        strategy_instance.logger.info(f"📊 Indicator Test for {base_symbol}:")
        # Log results carefully, they can be nested dicts
        for key, value in test_result.items():
            if isinstance(value, dict):
                strategy_instance.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    strategy_instance.logger.info(f"    {sub_key}: {sub_value}")
            else:
                strategy_instance.logger.info(f"  {key}: {value}")

    except Exception as e:
        strategy_instance.logger.error(f"Failed to test indicators on integration: {e}")
        import traceback
        strategy_instance.logger.error(traceback.format_exc())

    
    strategy_instance.logger.info("⚡ Multi-Indicator Crypto Signals integrated! (QQE + Supertrend + Fisher + VWAP)")
    strategy_instance.logger.info("📌 Using original SL/TP logic for exits (external to this signal module)")
    
    return crypto_signals


# Add the test_indicators method to AdaptiveCryptoSignals class
# This method was outside the class in the original snippet, placing it inside.
def test_indicators(self, exchange) -> Dict:
        """Test all indicators and return their values for debugging"""
        try:
            # Get OHLCV data
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=1400) # Use self.symbol
            if not ohlcv_data or len(ohlcv_data) < 30: # Min data for basic indicators
                return {'error': f'Insufficient data, got {len(ohlcv_data) if ohlcv_data else 0} candles'}
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            df_indexed = df.set_index('timestamp') # Use indexed df for calculations
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_indexed[col] = df_indexed[col].astype(float)
            
            results = {
                'symbol': self.symbol, # Use self.symbol
                'current_price': float(df_indexed['close'].iloc[-1]),
                'data_points': len(df_indexed)
            }
            
            # Test QQE
            qqe_result = self._calculate_qqe(df_indexed)
            results['qqe'] = qqe_result if qqe_result else 'Failed or Insufficient Data'
            
            # Test Supertrend
            st_result = self._calculate_supertrend(df_indexed)
            results['supertrend'] = st_result if st_result else 'Failed or Insufficient Data'
            
            # Test Fisher
            fisher_result = self._calculate_fisher(df_indexed)
            results['fisher'] = fisher_result if fisher_result else 'Failed or Insufficient Data'
            
            # Test VWAP
            vwap_result = self._calculate_vwap_bands(df_indexed)
            results['vwap'] = vwap_result if vwap_result else 'Failed or Insufficient Data'
            
            # Get composite signal based on these test calculations
            composite_signal = self._generate_composite_signal(df_indexed)
            results['composite_signal'] = composite_signal
            results['current_market_trend_assessment'] = self.current_trend
            results['current_volatility_assessment'] = self.volatility_level
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in test_indicators for {self.symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}

# Bind the method to the class prototype
AdaptiveCryptoSignals.test_indicators = test_indicators


# Alias for easy migration (if used elsewhere)
integrate_crypto_signals = integrate_adaptive_crypto_signals

