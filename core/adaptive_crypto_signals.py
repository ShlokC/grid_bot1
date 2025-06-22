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
    vwap_bands_std: float = 2.0
    
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
        self.min_signals_for_optimization = 5
        self.optimization_interval = 300  # 5 minutes
        
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
            if not ohlcv_data or len(ohlcv_data) < 30:
                self.logger.debug(f"Insufficient data for {self.symbol}: {len(ohlcv_data) if ohlcv_data else 0} candles")
                return 'none'
            
            # Convert to DataFrame for pandas_ta
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # FIXED: Ensure datetime ordering for VWAP
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            df.set_index('timestamp', inplace=True)
            
            # Convert to float
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Generate multi-indicator signal
            signal = self._generate_composite_signal(df)
            
            if signal != 'none':
                self._track_signal(signal, float(df['close'].iloc[-1]))
                self.last_signal = signal
                self.last_signal_time = current_time
                
                self.logger.info(f"⚡ ENTRY SIGNAL: {signal.upper()} @ ${float(df['close'].iloc[-1]):.6f} "
                               f"[Trend: {self.current_trend}] [Vol: {self.volatility_level}] "
                               f"[Accuracy: {self.params.accuracy:.0f}%]")
            
            # Check if optimization needed
            if self._should_optimize():
                self._optimize_parameters(df)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 'none'
            
            # Generate multi-indicator signal
            signal = self._generate_composite_signal(df)
            
            if signal != 'none':
                self._track_signal(signal, float(df['close'].iloc[-1]))
                self.last_signal = signal
                self.last_signal_time = current_time
                
                self.logger.info(f"⚡ SIGNAL: {signal.upper()} @ ${float(df['close'].iloc[-1]):.6f} "
                               f"[Trend: {self.current_trend}] [Vol: {self.volatility_level}] "
                               f"[Accuracy: {self.params.accuracy:.0f}%]")
            
            # Check if optimization needed
            if self._should_optimize():
                self._optimize_parameters(df)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal: {e}")
            return 'none'

    def _generate_composite_signal(self, df: pd.DataFrame) -> str:
        """Generate signal using multiple indicators for confirmation"""
        try:
            # 1. Calculate QQE (momentum oscillator)
            qqe_result = self._calculate_qqe(df)
            if qqe_result is None:
                self.logger.debug(f"QQE calculation failed for {self.symbol}")
                return 'none'
            
            qqe_value = qqe_result['qqe']
            qqe_signal = qqe_result['signal']
            qqe_direction = qqe_result['direction']
            
            # 2. Calculate Supertrend (trend direction)
            supertrend_result = self._calculate_supertrend(df)
            if supertrend_result is None:
                self.logger.debug(f"Supertrend calculation failed for {self.symbol}")
                return 'none'
            
            trend_direction = supertrend_result['direction']
            trend_changed = supertrend_result['changed']
            
            # 3. Calculate Fisher Transform (reversal detection)
            fisher_result = self._calculate_fisher(df)
            if fisher_result is None:
                self.logger.debug(f"Fisher calculation failed for {self.symbol}")
                return 'none'
            
            fisher_value = fisher_result['fisher']
            fisher_cross = fisher_result.get('cross', None)  # Updated from 'signal'
            fisher_extreme = fisher_result['extreme']
            
            # 4. Calculate VWAP and bands (support/resistance)
            vwap_result = self._calculate_vwap_bands(df)
            if vwap_result is None:
                self.logger.debug(f"VWAP calculation failed for {self.symbol}")
                return 'none'
            
            price_position = vwap_result['position']
            near_band = vwap_result['near_band']
            
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
            
            # SIGNAL GENERATION LOGIC
            signal = 'none'
            signal_reason = 'No conditions met'
            
            # FIXED: Add signal stability check to prevent immediate opposite signals
            # This prevents entering a position and immediately getting exit signal
            time_since_last = time.time() - self.last_signal_time
            allow_opposite = time_since_last >= self.signal_stability_period  # 30 seconds minimum
            
            # Less restrictive conditions for volatile crypto
            
            # 1. Trend Change Signals (HIGHEST PRIORITY)
            # if trend_changed:
            #     if trend_direction == 'up':
            #         # Check if opposite signal too soon
            #         if not allow_opposite and self.last_signal == 'sell':
            #             self.logger.debug(f"Skipping BUY signal too soon after SELL ({time_since_last:.1f}s)")
            #             return 'none'
            #         signal = 'buy'
            #         self.logger.debug("🟢 TREND BUY: Supertrend flipped UP")
            #     elif trend_direction == 'down':
            #         # Check if opposite signal too soon
            #         if not allow_opposite and self.last_signal == 'buy':
            #             self.logger.debug(f"Skipping SELL signal too soon after BUY ({time_since_last:.1f}s)")
            #             return 'none'
            #         signal = 'sell'
            #         self.logger.debug("🔴 TREND SELL: Supertrend flipped DOWN")
            #     return signal
            
            # 2. Strong Momentum Signals
            if (trend_direction == 'up' and qqe_direction == 'bullish' and 
                (fisher_value < -1.5 or price_position == 'below_lower')):
                if not allow_opposite and self.last_signal == 'sell':
                    self.logger.debug(f"Skipping momentum BUY too soon after SELL ({time_since_last:.1f}s)")
                    return 'none'
                signal = 'buy'
                self.logger.debug("🟢 MOMENTUM BUY: Trend up + QQE bullish + oversold/support")
                
            elif (trend_direction == 'down' and qqe_direction == 'bearish' and 
                  (fisher_value > 1.5 or price_position == 'above_upper')):
                if not allow_opposite and self.last_signal == 'buy':
                    self.logger.debug(f"Skipping momentum SELL too soon after BUY ({time_since_last:.1f}s)")
                    return 'none'
                signal = 'sell'
                self.logger.debug("🔴 MOMENTUM SELL: Trend down + QQE bearish + overbought/resistance")
            
            # 3. Extreme Reversal Signals (relaxed conditions)
            elif fisher_extreme == 'oversold' and (qqe_direction != 'bearish' or trend_direction == 'up'):
                if not allow_opposite and self.last_signal == 'sell':
                    self.logger.debug(f"Skipping reversal BUY too soon after SELL ({time_since_last:.1f}s)")
                    return 'none'
                signal = 'buy'
                self.logger.debug("🟢 REVERSAL BUY: Fisher extreme oversold")
                
            elif fisher_extreme == 'overbought' and (qqe_direction != 'bullish' or trend_direction == 'down'):
                if not allow_opposite and self.last_signal == 'buy':
                    self.logger.debug(f"Skipping reversal SELL too soon after BUY ({time_since_last:.1f}s)")
                    return 'none'
                signal = 'sell'
                self.logger.debug("🔴 REVERSAL SELL: Fisher extreme overbought")
            
            # 4. VWAP Band Bounce (new condition)
            elif self.volatility_level != 'low':
                if price_position == 'below_lower' and qqe_direction != 'bearish':
                    if not allow_opposite and self.last_signal == 'sell':
                        self.logger.debug(f"Skipping VWAP BUY too soon after SELL ({time_since_last:.1f}s)")
                        return 'none'
                    signal = 'buy'
                    self.logger.debug("🟢 VWAP BUY: Price below lower band")
                elif price_position == 'above_upper' and qqe_direction != 'bullish':
                    if not allow_opposite and self.last_signal == 'buy':
                        self.logger.debug(f"Skipping VWAP SELL too soon after BUY ({time_since_last:.1f}s)")
                        return 'none'
                    signal = 'sell'
                    self.logger.debug("🔴 VWAP SELL: Price above upper band")
            
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
        """
        Enhanced exit signal generation for open positions
        
        Args:
            position_side: 'long' or 'short'
            entry_price: Position entry price
            current_price: Current market price
            
        Returns:
            bool: True if should exit position, False otherwise
        """
        try:
            if not hasattr(self, 'last_indicators') or not self.last_indicators:
                return False
            
            # Calculate position PnL
            if position_side == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # short
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Get current indicators
            qqe = self.last_indicators.get('qqe', {})
            supertrend = self.last_indicators.get('supertrend', {})
            fisher = self.last_indicators.get('fisher', {})
            vwap = self.last_indicators.get('vwap', {})
            
            # EXIT CONDITIONS
            
            # 1. Trend Reversal Exit (HIGHEST PRIORITY)
            if position_side == 'long' and supertrend.get('direction') == 'down':
                self.logger.warning(f"🚨 EXIT LONG: Supertrend turned DOWN (PnL: {pnl_pct:.2f}%)")
                return True
            elif position_side == 'short' and supertrend.get('direction') == 'up':
                self.logger.warning(f"🚨 EXIT SHORT: Supertrend turned UP (PnL: {pnl_pct:.2f}%)")
                return True
            
            # 2. QQE Momentum Reversal
            if position_side == 'long' and qqe.get('direction') == 'bearish':
                if pnl_pct < -1.0:  # If losing, exit quickly
                    self.logger.warning(f"🚨 EXIT LONG: QQE bearish + losing position (PnL: {pnl_pct:.2f}%)")
                    return True
                elif pnl_pct > 2.0 and qqe.get('momentum', 0) < -0.5:  # If winning, wait for strong reversal
                    self.logger.warning(f"🚨 EXIT LONG: QQE strong bearish momentum (PnL: {pnl_pct:.2f}%)")
                    return True
                    
            elif position_side == 'short' and qqe.get('direction') == 'bullish':
                if pnl_pct < -1.0:  # If losing, exit quickly
                    self.logger.warning(f"🚨 EXIT SHORT: QQE bullish + losing position (PnL: {pnl_pct:.2f}%)")
                    return True
                elif pnl_pct > 2.0 and qqe.get('momentum', 0) > 0.5:  # If winning, wait for strong reversal
                    self.logger.warning(f"🚨 EXIT SHORT: QQE strong bullish momentum (PnL: {pnl_pct:.2f}%)")
                    return True
            
            # 3. Fisher Extreme Reversal
            if position_side == 'long' and fisher.get('extreme') == 'overbought':
                if fisher.get('momentum', 0) < -0.5:  # Turning down from overbought
                    self.logger.warning(f"🚨 EXIT LONG: Fisher overbought reversal (PnL: {pnl_pct:.2f}%)")
                    return True
            elif position_side == 'short' and fisher.get('extreme') == 'oversold':
                if fisher.get('momentum', 0) > 0.5:  # Turning up from oversold
                    self.logger.warning(f"🚨 EXIT SHORT: Fisher oversold reversal (PnL: {pnl_pct:.2f}%)")
                    return True
            
            # 4. VWAP Band Touch (Take Profit)
            if position_side == 'long' and vwap.get('position') == 'above_upper':
                if pnl_pct > 1.0:  # Only exit if profitable
                    self.logger.warning(f"🚨 EXIT LONG: Hit upper VWAP band - take profit (PnL: {pnl_pct:.2f}%)")
                    return True
            elif position_side == 'short' and vwap.get('position') == 'below_lower':
                if pnl_pct > 1.0:  # Only exit if profitable
                    self.logger.warning(f"🚨 EXIT SHORT: Hit lower VWAP band - take profit (PnL: {pnl_pct:.2f}%)")
                    return True
            
            # 5. Quick Stop Loss for High Volatility
            if self.volatility_level == 'high' and pnl_pct < -1.5:
                self.logger.warning(f"🚨 STOP LOSS: High volatility + losing position (PnL: {pnl_pct:.2f}%)")
                return True
            
            # 6. Time-based Exit for Stagnant Positions
            if hasattr(self, 'position_entry_time'):
                position_duration = time.time() - self.position_entry_time
                if position_duration > 1800 and abs(pnl_pct) < 0.5:  # 30 mins with no movement
                    self.logger.warning(f"🚨 TIME EXIT: Position stagnant for 30 mins (PnL: {pnl_pct:.2f}%)")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in exit signal generation: {e}")
            return False

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
            qqe = ta.qqe(df['close'], length=self.params.qqe_length, smooth=self.params.qqe_smooth)
            
            if qqe is None or qqe.empty:
                return None
            
            # QQE returns: QQE_{length}_{smooth}, QQE_{length}_{smooth}_RSI_MA
            qqe_cols = qqe.columns.tolist()
            if len(qqe_cols) < 2:
                return None
            
            qqe_line = qqe.iloc[:, 0].dropna()
            qqe_signal = qqe.iloc[:, 1].dropna()
            
            if len(qqe_line) < 2:
                return None
            
            current_qqe = float(qqe_line.iloc[-1])
            current_signal = float(qqe_signal.iloc[-1])
            prev_qqe = float(qqe_line.iloc[-2])
            
            # Determine direction
            direction = 'neutral'
            if current_qqe > current_signal and current_qqe > prev_qqe:
                direction = 'bullish'
            elif current_qqe < current_signal and current_qqe < prev_qqe:
                direction = 'bearish'
            
            return {
                'qqe': current_qqe,
                'signal': current_signal,
                'direction': direction,
                'momentum': current_qqe - prev_qqe
            }
            
        except Exception as e:
            self.logger.debug(f"QQE calculation error: {e}")
            return None

    def _calculate_supertrend(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Supertrend indicator"""
        try:
            st = ta.supertrend(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                length=self.params.supertrend_period,
                multiplier=self.params.supertrend_multiplier
            )
            
            if st is None or st.empty:
                return None
            
            # Supertrend returns: SUPERT_{length}_{multiplier}, SUPERTd_{length}_{multiplier}, SUPERTl_{length}_{multiplier}, SUPERTs_{length}_{multiplier}
            # We need the direction column (SUPERTd)
            direction_col = None
            for col in st.columns:
                if 'SUPERTd' in col:
                    direction_col = col
                    break
            
            if direction_col is None:
                return None
            
            directions = st[direction_col].dropna()
            if len(directions) < 2:
                return None
            
            current_dir = int(directions.iloc[-1])
            prev_dir = int(directions.iloc[-2])
            
            # FIXED: Check multiple candles back to ensure trend change is real
            trend_changed = False
            if current_dir != prev_dir:
                # Verify it's not just noise by checking one more candle back
                if len(directions) >= 3:
                    prev_prev_dir = int(directions.iloc[-3])
                    # Only consider it changed if previous two were same and now different
                    if prev_dir == prev_prev_dir and current_dir != prev_dir:
                        trend_changed = True
                else:
                    trend_changed = True
            
            # Supertrend direction: 1 = up, -1 = down
            trend_direction = 'up' if current_dir == 1 else 'down'
            
            return {
                'direction': trend_direction,
                'changed': trend_changed,
                'strength': abs(current_dir)
            }
            
        except Exception as e:
            self.logger.debug(f"Supertrend calculation error: {e}")
            return None

    def _calculate_fisher(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Fisher Transform for reversal detection"""
        try:
            fisher = ta.fisher(
                high=df['high'],
                low=df['low'],
                length=self.params.fisher_period
            )
            
            if fisher is None or fisher.empty:
                return None
            
            # Fisher returns: FISHERT_{length}, FISHERTs_{length}
            fisher_cols = fisher.columns.tolist()
            if len(fisher_cols) < 2:
                return None
            
            fisher_line = fisher.iloc[:, 0].dropna()
            fisher_signal = fisher.iloc[:, 1].dropna()
            
            if len(fisher_line) < 2:
                return None
            
            current_fisher = float(fisher_line.iloc[-1])
            current_signal = float(fisher_signal.iloc[-1])
            prev_fisher = float(fisher_line.iloc[-2])
            prev_signal = float(fisher_signal.iloc[-2])
            
            # REDUCED thresholds for crypto volatility
            extreme = None
            if current_fisher > 1.5:  # Reduced from 2.5
                extreme = 'overbought'
            elif current_fisher < -1.5:  # Reduced from -2.5
                extreme = 'oversold'
            
            # Detect crossovers
            cross_signal = None
            if prev_fisher <= prev_signal and current_fisher > current_signal:
                cross_signal = 'bullish_cross'
            elif prev_fisher >= prev_signal and current_fisher < current_signal:
                cross_signal = 'bearish_cross'
            
            return {
                'fisher': current_fisher,
                'signal': current_signal,
                'extreme': extreme,
                'cross': cross_signal,  # Renamed from 'signal' to avoid duplicate
                'momentum': current_fisher - prev_fisher
            }
            
        except Exception as e:
            self.logger.debug(f"Fisher Transform calculation error: {e}")
            return None

    def _calculate_vwap_bands(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate VWAP with bands for dynamic support/resistance"""
        try:
            # Calculate VWAP
            vwap = ta.vwap(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            
            if vwap is None or vwap.empty:
                return None
            
            vwap_values = vwap.dropna()
            if len(vwap_values) < 5:
                return None
            
            current_vwap = float(vwap_values.iloc[-1])
            current_price = float(df['close'].iloc[-1])
            
            # Calculate standard deviation for bands
            recent_closes = df['close'].iloc[-20:].values
            std_dev = np.std(recent_closes)
            
            upper_band = current_vwap + (self.params.vwap_bands_std * std_dev)
            lower_band = current_vwap - (self.params.vwap_bands_std * std_dev)
            
            # Determine price position
            position = 'neutral'
            if current_price > upper_band:
                position = 'above_upper'
            elif current_price < lower_band:
                position = 'below_lower'
            elif current_price > current_vwap:
                position = 'above_vwap'
            else:
                position = 'below_vwap'
            
            # Check if near bands
            band_distance = std_dev * 0.3  # 30% of std as "near"
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
                'price_pct_from_vwap': ((current_price - current_vwap) / current_vwap) * 100
            }
            
        except Exception as e:
            self.logger.debug(f"VWAP calculation error: {e}")
            return None

    def _update_volatility_level(self, df: pd.DataFrame):
        """Update current volatility level"""
        try:
            # Calculate ATR for volatility
            atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
            if atr is not None and len(atr) > 0:
                current_atr = float(atr.iloc[-1])
                avg_price = float(df['close'].iloc[-1])
                atr_pct = (current_atr / avg_price) * 100
                
                # Adjusted thresholds for crypto
                if atr_pct > 2.0:  # Reduced from 3.0
                    self.volatility_level = 'high'
                elif atr_pct < 0.5:  # Reduced from 1.0
                    self.volatility_level = 'low'
                else:
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
                'params': {
                    'qqe': self.params.qqe_length,
                    'st': self.params.supertrend_period,
                    'fisher': self.params.fisher_period
                },
                'market_state': {
                    'trend': self.current_trend,
                    'volatility': self.volatility_level
                }
            }
            
            self.signal_performance.append(signal_data)
            
            # Keep only last 100 signals
            if len(self.signal_performance) > 100:
                self.signal_performance.popleft()
            
            # Evaluate performance after 5 minutes
            self._evaluate_signals()
            
            # Save periodically
            if len(self.signal_performance) % 5 == 0:
                self._save_config()
                
        except Exception as e:
            self.logger.error(f"Error tracking signal: {e}")

    def _evaluate_signals(self):
        """Evaluate signal performance"""
        try:
            current_time = time.time()
            evaluated = 0
            correct = 0
            
            for signal_data in self.signal_performance:
                if 'evaluated' in signal_data:
                    continue
                
                # Wait 5 minutes before evaluation
                if current_time - signal_data['timestamp'] < 300:
                    continue
                
                # Find price 5 minutes later
                for future_signal in self.signal_performance:
                    if future_signal['timestamp'] > signal_data['timestamp'] + 300:
                        future_price = future_signal['price']
                        price_change = ((future_price - signal_data['price']) / signal_data['price']) * 100
                        
                        # Evaluate success
                        if signal_data['signal'] == 'buy' and price_change > 0.1:
                            signal_data['correct'] = True
                            correct += 1
                        elif signal_data['signal'] == 'sell' and price_change < -0.1:
                            signal_data['correct'] = True
                            correct += 1
                        else:
                            signal_data['correct'] = False
                        
                        signal_data['evaluated'] = True
                        signal_data['price_change'] = price_change
                        evaluated += 1
                        break
            
            # Update accuracy
            total_evaluated = sum(1 for s in self.signal_performance if 'evaluated' in s)
            total_correct = sum(1 for s in self.signal_performance if s.get('correct', False))
            
            if total_evaluated > 0:
                self.params.accuracy = (total_correct / total_evaluated) * 100
                self.params.total_signals = total_evaluated
                self.params.winning_signals = total_correct
                
        except Exception as e:
            self.logger.error(f"Error evaluating signals: {e}")

    def _should_optimize(self) -> bool:
        """Check if parameter optimization is needed"""
        current_time = time.time()
        
        # Time-based check
        if current_time - self.last_optimization < self.optimization_interval:
            return False
        
        # Performance-based check
        if self.params.total_signals >= self.min_signals_for_optimization:
            if self.params.accuracy < 40:  # Below 40% accuracy
                return True
        
        return False

    def _optimize_parameters(self, df: pd.DataFrame):
        """Optimize indicator parameters based on recent performance"""
        try:
            self.logger.info("🔧 Optimizing parameters...")
            
            # Parameter ranges to test
            param_ranges = {
                'qqe_length': [10, 14, 20],
                'qqe_smooth': [3, 5, 7],
                'supertrend_period': [5, 7, 10],
                'supertrend_multiplier': [2.0, 3.0, 4.0],
                'fisher_period': [7, 9, 12]
            }
            
            best_score = self.params.accuracy
            best_params = None
            
            # Quick optimization - test a few combinations
            for qqe_len in param_ranges['qqe_length']:
                for st_period in param_ranges['supertrend_period']:
                    for fisher_period in param_ranges['fisher_period']:
                        # Test these parameters
                        test_params = SignalParameters(
                            qqe_length=qqe_len,
                            qqe_smooth=5,  # Keep smooth constant
                            supertrend_period=st_period,
                            supertrend_multiplier=3.0,  # Keep multiplier constant
                            fisher_period=fisher_period
                        )
                        
                        # Quick backtest on recent data
                        score = self._quick_backtest(df, test_params)
                        
                        if score > best_score:
                            best_score = score
                            best_params = test_params
            
            # Update parameters if improvement found
            if best_params and best_score > self.params.accuracy + 5:
                self.logger.info(f"📈 Found better parameters: Accuracy {self.params.accuracy:.1f}% → {best_score:.1f}%")
                self.params = best_params
                self.params.accuracy = best_score
                self._save_config()
            
            self.last_optimization = time.time()
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")

    def _quick_backtest(self, df: pd.DataFrame, params: SignalParameters) -> float:
        """Quick backtest for parameter optimization"""
        # Simplified backtest - just return random score for now
        # In production, this would test the parameters on historical data
        return np.random.uniform(30, 70)

    def _load_symbol_config(self) -> SignalParameters:
        """Load saved configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
                    
                if self.symbol in configs:
                    config = configs[self.symbol]
                    return SignalParameters(
                        qqe_length=config.get('qqe_length', 14),
                        qqe_smooth=config.get('qqe_smooth', 5),
                        supertrend_period=config.get('supertrend_period', 7),
                        supertrend_multiplier=config.get('supertrend_multiplier', 3.0),
                        fisher_period=config.get('fisher_period', 9),
                        accuracy=config.get('accuracy', 0.0),
                        total_signals=config.get('total_signals', 0),
                        winning_signals=config.get('winning_signals', 0)
                    )
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
        
        return SignalParameters()

    def _load_signal_history(self) -> deque:
        """Load signal history"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
                    
                if self.symbol in configs:
                    history = configs[self.symbol].get('signal_history', [])
                    return deque(history[-100:], maxlen=100)  # Keep last 100
        except Exception:
            pass
        
        return deque(maxlen=100)

    def _save_config(self):
        """Save configuration and signal history"""
        try:
            configs = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
            
            configs[self.symbol] = {
                'qqe_length': self.params.qqe_length,
                'qqe_smooth': self.params.qqe_smooth,
                'supertrend_period': self.params.supertrend_period,
                'supertrend_multiplier': self.params.supertrend_multiplier,
                'fisher_period': self.params.fisher_period,
                'accuracy': self.params.accuracy,
                'total_signals': self.params.total_signals,
                'winning_signals': self.params.winning_signals,
                'signal_history': list(self.signal_performance)[-100:],
                'last_updated': time.time()
            }
            
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file + '.tmp', 'w') as f:
                json.dump(configs, f, indent=2)
            
            os.replace(self.config_file + '.tmp', self.config_file)
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

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
                'total_signals': self.params.total_signals,
                'winning_signals': self.params.winning_signals
            },
            'market_state': {
                'trend': self.current_trend,
                'volatility': self.volatility_level
            },
            'exit_logic': 'Using original SL/TP orders only'
        }


def integrate_adaptive_crypto_signals(strategy_instance, config_file: str = None):
    """Integration function to replace TSI with multi-indicator crypto signals - ENTRY ONLY"""
    if config_file is None:
        config_file = os.path.join("data", "crypto_signal_configs.json")
    
    strategy_instance.logger.info(f"🔧 Integrating Multi-Indicator Crypto Signals (ENTRY ONLY)")
    
    # Create new signal system
    crypto_signals = AdaptiveCryptoSignals(symbol=strategy_instance.symbol, config_file=config_file)
    
    # Replace ONLY the TSI method with new system
    def crypto_get_technical_direction():
        return crypto_signals.get_technical_direction(strategy_instance.exchange)
    
    def get_signal_status():
        return crypto_signals.get_system_status()
    
    def test_crypto_indicators():
        """Test function to debug indicator values"""
        return crypto_signals.test_indicators(strategy_instance.exchange)
    
    # Replace methods - NO EXIT LOGIC MODIFICATIONS
    strategy_instance._get_technical_direction = crypto_get_technical_direction
    strategy_instance.get_signal_status = get_signal_status
    strategy_instance.test_crypto_indicators = test_crypto_indicators
    strategy_instance._crypto_signal_system = crypto_signals
    
    # Test indicators on first run
    try:
        test_result = test_crypto_indicators()
        strategy_instance.logger.info(f"📊 Indicator Test for {strategy_instance.symbol}:")
        for key, value in test_result.items():
            strategy_instance.logger.info(f"  {key}: {value}")
    except Exception as e:
        strategy_instance.logger.error(f"Failed to test indicators: {e}")
    
    strategy_instance.logger.info("⚡ Multi-Indicator Crypto Signals integrated! (QQE + Supertrend + Fisher + VWAP)")
    strategy_instance.logger.info("📌 Using original SL/TP logic for exits")
    
    return crypto_signals


def test_indicators(self, exchange) -> Dict:
        """Test all indicators and return their values for debugging"""
        try:
            # Get OHLCV data
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=1400)
            if not ohlcv_data or len(ohlcv_data) < 30:
                return {'error': 'Insufficient data'}
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            df.set_index('timestamp', inplace=True)
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Test each indicator
            results = {
                'symbol': self.symbol,
                'current_price': float(df['close'].iloc[-1]),
                'data_points': len(df)
            }
            
            # Test QQE
            qqe_result = self._calculate_qqe(df)
            if qqe_result:
                results['qqe'] = {
                    'value': qqe_result['qqe'],
                    'signal': qqe_result['signal'],
                    'direction': qqe_result['direction']
                }
            else:
                results['qqe'] = 'Failed'
            
            # Test Supertrend
            st_result = self._calculate_supertrend(df)
            if st_result:
                results['supertrend'] = {
                    'direction': st_result['direction'],
                    'changed': st_result['changed']
                }
            else:
                results['supertrend'] = 'Failed'
            
            # Test Fisher
            fisher_result = self._calculate_fisher(df)
            if fisher_result:
                results['fisher'] = {
                    'value': fisher_result['fisher'],
                    'extreme': fisher_result.get('extreme', 'normal')
                }
            else:
                results['fisher'] = 'Failed'
            
            # Test VWAP
            vwap_result = self._calculate_vwap_bands(df)
            if vwap_result:
                results['vwap'] = {
                    'vwap': vwap_result['vwap'],
                    'position': vwap_result['position'],
                    'price_pct_from_vwap': vwap_result['price_pct_from_vwap']
                }
            else:
                results['vwap'] = 'Failed'
            
            # Get signal
            signal = self._generate_composite_signal(df)
            results['signal'] = signal
            
            return results
            
        except Exception as e:
            return {'error': str(e)}


# Alias for easy migration
integrate_crypto_signals = integrate_adaptive_crypto_signals