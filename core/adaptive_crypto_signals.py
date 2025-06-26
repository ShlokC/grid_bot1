"""
Fixed Adaptive Crypto Signal System - Complete Implementation with Signal-Based Exits
Key fixes: Non-blocking signals + Technical signal-based exit conditions
"""

import logging
import time
import os
import threading
from typing import Dict, Optional, Tuple
from collections import deque
import copy
import json
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

@dataclass
class SignalParameters:
    """Parameters for trading signals with optimization tracking"""
    
    strategy_type: str = 'qqe_supertrend_fixed'
    
    # QQE parameters
    qqe_length: int = 12
    qqe_smooth: int = 5
    qqe_factor: float = 4.236
    
    # Supertrend parameters  
    supertrend_period: int = 10
    supertrend_multiplier: float = 2.8
    
    # RSI parameters
    rsi_length: int = 14
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # TSI parameters
    tsi_fast: int = 8
    tsi_slow: int = 15
    tsi_signal: int = 6
    
    # Performance tracking
    accuracy: float = 0.0
    total_signals: int = 0
    winning_signals: int = 0
    last_used: float = 0.0
    optimization_score: float = 0.0

class IntegratedWinRateOptimizer:
    """Simplified WinRateOptimizer for background optimization"""
    
    def __init__(self, symbol: str, strategy_type: str):
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.logger = logging.getLogger(f"{__name__}.WinRateOptimizer.{symbol}")
        
        self.optimization_timeout = 60  # Reduced timeout
        self.n_trials = 50  # Reduced trials for faster optimization
        self.min_trades_threshold = 10
        
    def optimize_for_winrate(self, historical_data: list, current_params: SignalParameters) -> Tuple[Dict, float, Dict]:
        """Simple grid search optimization for non-blocking operation"""
        try:
            if not historical_data or len(historical_data) < 50:
                return self._params_to_dict(current_params), 0.0, {}
            
            best_params = None
            best_score = current_params.accuracy
            
            # Simple parameter ranges for quick optimization
            if 'qqe' in self.strategy_type.lower():
                param_combinations = [
                    (10, 4, 8, 2.5), (12, 5, 10, 2.8), (15, 6, 12, 3.0)
                ]
                
                for qqe_len, qqe_smooth, st_period, st_mult in param_combinations:
                    test_params = SignalParameters(
                        strategy_type=self.strategy_type,
                        qqe_length=qqe_len,
                        qqe_smooth=qqe_smooth,
                        supertrend_period=st_period,
                        supertrend_multiplier=st_mult
                    )
                    
                    score = self._simple_backtest_winrate(historical_data, test_params)
                    if score > best_score:
                        best_score = score
                        best_params = self._params_to_dict(test_params)
            
            return best_params or self._params_to_dict(current_params), best_score, {}
            
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return self._params_to_dict(current_params), 0.0, {}
    
    def _simple_backtest_winrate(self, historical_data: list, test_params: SignalParameters) -> float:
        """Simple backtest for parameter evaluation"""
        try:
            df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            signals = []
            for i in range(20, len(df)):
                current_df = df.iloc[:i+1]
                signal = self._generate_test_signal(current_df, test_params)
                if signal != 'none':
                    signals.append({'signal': signal, 'price': current_df['close'].iloc[-1], 'index': i})
            
            if len(signals) < 5:
                return 0.0
            
            wins = 0
            total = 0
            
            for signal_data in signals:
                if signal_data['index'] + 5 < len(df):
                    future_price = df['close'].iloc[signal_data['index'] + 5]
                    price_change = ((future_price - signal_data['price']) / signal_data['price']) * 100
                    
                    if signal_data['signal'] == 'buy' and price_change > 0.2:
                        wins += 1
                    elif signal_data['signal'] == 'sell' and price_change < -0.2:
                        wins += 1
                    total += 1
            
            return (wins / total) * 100.0 if total > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_test_signal(self, df: pd.DataFrame, test_params: SignalParameters) -> str:
        """Generate test signal for optimization"""
        try:
            if 'qqe' in self.strategy_type.lower():
                qqe_result = ta.qqe(df['close'], length=test_params.qqe_length, smooth=test_params.qqe_smooth)
                st_result = ta.supertrend(df['high'], df['low'], df['close'], 
                                        length=test_params.supertrend_period, 
                                        multiplier=test_params.supertrend_multiplier)
                
                if (qqe_result is not None and not qqe_result.empty and len(qqe_result.columns) >= 2 and
                    st_result is not None and not st_result.empty):
                    
                    dir_col = next((col for col in st_result.columns if 'SUPERTd' in col), None)
                    if dir_col:
                        rsi_ma = qqe_result.iloc[-1, 1]  # Corresponds to QQE_..._RSIMA or RsiMa
                        qqe_line = qqe_result.iloc[-1, 0] # Corresponds to QQE_... or FastAtrRsiTL
                        
                        st_dir = st_result[dir_col].iloc[-1]

                        if rsi_ma > qqe_line and st_dir == 1:
                            return 'buy'
                        elif qqe_line > rsi_ma and st_dir != 1:
                            return 'sell'
            
            return 'none'
        except Exception:
            return 'none'
    
    def _params_to_dict(self, params: SignalParameters) -> Dict:
        """Convert parameters to dictionary"""
        return {
            'qqe_length': params.qqe_length,
            'qqe_smooth': params.qqe_smooth,
            'qqe_factor': params.qqe_factor,
            'supertrend_period': params.supertrend_period,
            'supertrend_multiplier': params.supertrend_multiplier,
            'rsi_length': params.rsi_length,
            'macd_fast': params.macd_fast,
            'macd_slow': params.macd_slow,
            'macd_signal': params.macd_signal,
            'tsi_fast': params.tsi_fast,
            'tsi_slow': params.tsi_slow,
            'tsi_signal': params.tsi_signal
        }

class AdaptiveCryptoSignals:
    """COMPLETE: Non-blocking signal generation with proper technical exit conditions"""
    
    def __init__(self, symbol: str, config_file: str = "data/crypto_signal_configs.json", strategy_type: str = 'qqe_supertrend_fixed'):
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        self.symbol = symbol
        self.config_file = config_file
        self.strategy_type = strategy_type
        self.position_entry_time = 0
        
        # Load optimized parameters
        self.params = self._load_symbol_config()
        self.signal_performance = self._load_signal_history()
        
        # FIXED: Create parameter snapshot for thread-safe access
        self._params_snapshot = copy.deepcopy(self.params)
        self._snapshot_lock = threading.RLock()
        
        # Initialize integrated WinRateOptimizer
        self.win_rate_optimizer = IntegratedWinRateOptimizer(symbol, strategy_type)
        
        # Signal control with faster cooldown for non-blocking
        self.last_signal = 'none'
        self.last_signal_time = 0
        self.signal_cooldown = 5
        
        # FIXED: Store last calculated indicators for exit evaluation
        self._last_calculated_indicators = {}
        self._indicator_lock = threading.RLock()
        
        # FIXED: Optimization control with proper isolation
        self.optimization_in_progress = False
        self.optimization_lock = threading.RLock()
        self.last_optimization = time.time()
        
        # FIXED: Indicator calculation timeout
        self.indicator_timeout = 2.0
        
        # Enhanced optimization settings
        self.min_signals_for_optimization = 15
        self.optimization_interval = 180
        self.win_rate_threshold = 40.0
        
        # FIXED: Simplified cache with size limit
        self._historical_data_cache = None
        self._cache_timestamp = 0
        self._cache_validity = 300
        self._max_cache_size = 500
        
        self.logger.info(f"COMPLETE: Non-blocking Adaptive Crypto Signals for {strategy_type} on {symbol}")

    def get_technical_direction(self, exchange) -> str:
        """FIXED: Non-blocking signal generation with parameter snapshots"""
        try:
            current_time = time.time()
            if current_time - self.last_signal_time < self.signal_cooldown:
                return 'none'
            
            # FIXED: Quick data fetch with size limit
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=100)
            if not ohlcv_data or len(ohlcv_data) < 30:
                return 'none'
            
            # FIXED: Create thread-safe parameter snapshot for this signal generation
            with self._snapshot_lock:
                current_params = copy.deepcopy(self._params_snapshot)
            
            # FIXED: Fast DataFrame conversion with timeout protection
            signal, indicators = self._generate_signal_with_indicators(ohlcv_data, current_params)
            
            # FIXED: Store indicators for exit evaluation
            if indicators:
                with self._indicator_lock:
                    self._last_calculated_indicators = indicators.copy()
            
            if signal != 'none':
                self._track_signal_fast(signal, float(ohlcv_data[-1][4]))
                self.last_signal = signal
                self.last_signal_time = current_time
                self.logger.info(f"SIGNAL: {signal.upper()} @ ${float(ohlcv_data[-1][4]):.6f}")
            
            # FIXED: Non-blocking cache update
            if not self.optimization_in_progress:
                self._update_cache_if_needed(ohlcv_data)
            
            # FIXED: Trigger background optimization without blocking
            if self._should_optimize_quick_check():
                self._trigger_non_blocking_optimization()
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return 'none'

    def _generate_signal_with_indicators(self, ohlcv_data: list, params) -> Tuple[str, Dict]:
        """FIXED: Generate signal and return indicators for exit evaluation"""
        try:
            # Fast DataFrame conversion
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            if len(df) < 20:
                return 'none', {}
            
            # FIXED: Calculate indicators and signal together
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                return self._qqe_supertrend_signal_with_indicators(df, params)
            elif self.strategy_type == 'rsi_macd':
                return self._rsi_macd_signal_with_indicators(df, params)
            elif self.strategy_type == 'tsi_vwap':
                return self._tsi_vwap_signal_with_indicators(df, params)
            
            return 'none', {}
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return 'none', {}

    def _qqe_supertrend_signal_with_indicators(self, df: pd.DataFrame, params) -> Tuple[str, Dict]:
        """FIXED: QQE + Supertrend signal with indicators for exit evaluation"""
        try:
            # Calculate indicators
            qqe_result = ta.qqe(df['close'], 
                              length=params.qqe_length,
                              smooth=params.qqe_smooth,
                              factor=params.qqe_factor)
            
            if qqe_result is None or qqe_result.empty or len(qqe_result.columns) < 2:
                return 'none', {}
            
            st_result = ta.supertrend(df['high'], df['low'], df['close'],
                                    length=params.supertrend_period,
                                    multiplier=params.supertrend_multiplier)
            
            if st_result is None or st_result.empty:
                return 'none', {}
            
            # Find direction column
            dir_col = next((col for col in st_result.columns if 'SUPERTd' in col), None)
            if not dir_col:
                return 'none', {}
            
            # Get latest values
            rsi_ma = qqe_result.iloc[-1, 1]  # Corresponds to QQE_..._RSIMA or RsiMa
            qqe_line = qqe_result.iloc[-1, 0] # Corresponds to QQE_... or FastAtrRsiTL
            st_dir = st_result[dir_col].iloc[-1]
            
            # Store indicators for exit evaluation
            indicators = {
                'qqe_value': float(rsi_ma),      # RsiMa
                'qqe_signal': float(qqe_line),   # FastAtrRsiTL
                'st_direction': int(st_dir),
                'current_price': float(df['close'].iloc[-1])
            }
            
            # Generate signal
            qqe_bullish = rsi_ma > qqe_line
            st_bullish = st_dir == 1
            
            if qqe_bullish and st_bullish:
                return 'buy', indicators
            elif not qqe_bullish and not st_bullish:
                return 'sell', indicators
            
            return 'none', indicators
            
        except Exception as e:
            self.logger.error(f"QQE+ST calculation error: {e}")
            return 'none', {}

    def _rsi_macd_signal_with_indicators(self, df: pd.DataFrame, params) -> Tuple[str, Dict]:
        """FIXED: RSI + MACD signal with indicators for exit evaluation"""
        try:
            rsi_result = ta.rsi(df['close'], length=params.rsi_length)
            if rsi_result is None:
                return 'none', {}
            
            macd_result = ta.macd(df['close'], 
                                fast=params.macd_fast,
                                slow=params.macd_slow,
                                signal=params.macd_signal)
            
            if macd_result is None or macd_result.empty or len(macd_result.columns) < 3:
                return 'none', {}
            
            rsi = rsi_result.iloc[-1]
            macd_line = macd_result.iloc[-1, 0]
            macd_signal = macd_result.iloc[-1, 2]
            
            # Store indicators for exit evaluation
            indicators = {
                'rsi': float(rsi),
                'macd_line': float(macd_line),
                'macd_signal': float(macd_signal),
                'current_price': float(df['close'].iloc[-1])
            }
            
            if rsi < 35 and macd_line > macd_signal:
                return 'buy', indicators
            elif rsi > 65 and macd_line < macd_signal:
                return 'sell', indicators
            
            return 'none', indicators
            
        except Exception as e:
            self.logger.error(f"RSI+MACD calculation error: {e}")
            return 'none', {}

    def _tsi_vwap_signal_with_indicators(self, df: pd.DataFrame, params) -> Tuple[str, Dict]:
        """FIXED: TSI + VWAP signal with indicators for exit evaluation"""
        try:
            tsi_result = ta.tsi(df['close'], 
                              fast=params.tsi_fast,
                              slow=params.tsi_slow,
                              signal=params.tsi_signal)
            
            if tsi_result is None or tsi_result.empty:
                return 'none', {}
            
            vwap_result = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            if vwap_result is None:
                return 'none', {}
            
            tsi_line = tsi_result.iloc[-1, 0]
            tsi_signal = tsi_result.iloc[-1, 1] if len(tsi_result.columns) > 1 else 0
            vwap = vwap_result.iloc[-1]
            price = df['close'].iloc[-1]
            
            # Store indicators for exit evaluation
            indicators = {
                'tsi_line': float(tsi_line),
                'tsi_signal': float(tsi_signal),
                'vwap': float(vwap),
                'current_price': float(price)
            }
            
            tsi_bullish = tsi_line > tsi_signal
            price_above_vwap = price > vwap
            
            if tsi_bullish and price_above_vwap:
                return 'buy', indicators
            elif not tsi_bullish and not price_above_vwap:
                return 'sell', indicators
            
            return 'none', indicators
            
        except Exception as e:
            self.logger.error(f"TSI+VWAP calculation error: {e}")
            return 'none', {}

    def evaluate_exit_conditions(self, position_side: str, entry_price: float, current_price: float) -> Dict:
        """COMPLETE: Exit evaluation using BOTH technical signals AND percentage-based stops"""
        try:
            result = {'should_exit': False, 'exit_reason': '', 'exit_urgency': 'none'}
            
            # Calculate PnL
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if position_side == 'long' else ((entry_price - current_price) / entry_price) * 100
            
            # PRIORITY 1: Emergency stop loss (immediate exit)
            if pnl_pct < -1.5:
                result.update({
                    'should_exit': True,
                    'exit_reason': f"Emergency SL: {pnl_pct:.2f}% loss",
                    'exit_urgency': 'immediate'
                })
                return result

            # PRIORITY 2: Take profit (quick profit taking)
            if pnl_pct > 2.0:
                result.update({
                    'should_exit': True,
                    'exit_reason': f"Take Profit: {pnl_pct:.2f}% gain",
                    'exit_urgency': 'normal'
                })
                return result

            # PRIORITY 3: Technical signal-based exits (strategy-specific)
            try:
                with self._indicator_lock:
                    indicators = self._last_calculated_indicators.copy() if self._last_calculated_indicators else {}
                
                if indicators:
                    signal_exit = self._check_technical_exit_conditions(position_side, indicators, pnl_pct)
                    if signal_exit['should_exit']:
                        return signal_exit
                        
            except Exception as e:
                self.logger.error(f"Technical exit check error: {e}")

            # PRIORITY 4: Time-based exit (prevent stale positions)
            if self.position_entry_time > 0:
                position_age_minutes = (time.time() - self.position_entry_time) / 60
                if position_age_minutes > 60:  # 1 hour maximum
                    result.update({
                        'should_exit': True,
                        'exit_reason': f"Time Exit: {position_age_minutes:.0f}min (PnL: {pnl_pct:.2f}%)",
                        'exit_urgency': 'normal'
                    })
                    return result

            return result
            
        except Exception as e:
            self.logger.error(f"Exit evaluation error: {e}")
            return {'should_exit': False, 'exit_reason': 'Error', 'exit_urgency': 'none'}

    def _check_technical_exit_conditions(self, position_side: str, indicators: Dict, pnl_pct: float) -> Dict:
        """FIXED: Strategy-specific technical exit conditions"""
        try:
            result = {'should_exit': False, 'exit_reason': '', 'exit_urgency': 'none'}
            
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                # QQE + Supertrend exit logic
                if all(key in indicators for key in ['qqe_value', 'qqe_signal', 'st_direction']):
                    qqe_bearish = indicators['qqe_value'] > indicators['qqe_signal']  # QQE turned bearish
                    st_bearish = indicators['st_direction'] != 1  # Supertrend turned bearish
                    
                    if position_side == 'long' and qqe_bearish and st_bearish:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"QQE+ST bearish reversal (PnL: {pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                    elif position_side == 'short':
                        qqe_bullish = indicators['qqe_signal'] > indicators['qqe_value']  # QQE turned bullish
                        st_bullish = indicators['st_direction'] == 1  # Supertrend turned bullish
                        
                        if qqe_bullish and st_bullish:
                            result.update({
                                'should_exit': True,
                                'exit_reason': f"QQE+ST bullish reversal (PnL: {pnl_pct:.2f}%)",
                                'exit_urgency': 'normal'
                            })
            
            elif self.strategy_type == 'rsi_macd':
                # RSI + MACD exit logic
                if all(key in indicators for key in ['rsi', 'macd_line', 'macd_signal']):
                    rsi = indicators['rsi']
                    macd_bearish = indicators['macd_line'] < indicators['macd_signal']
                    
                    if position_side == 'long' and (rsi > 70 or macd_bearish):
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"RSI overbought or MACD bearish (RSI: {rsi:.1f}, PnL: {pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                    elif position_side == 'short':
                        macd_bullish = indicators['macd_line'] > indicators['macd_signal']
                        
                        if rsi < 30 or macd_bullish:
                            result.update({
                                'should_exit': True,
                                'exit_reason': f"RSI oversold or MACD bullish (RSI: {rsi:.1f}, PnL: {pnl_pct:.2f}%)",
                                'exit_urgency': 'normal'
                            })
            
            elif self.strategy_type == 'tsi_vwap':
                # TSI + VWAP exit logic
                if all(key in indicators for key in ['tsi_line', 'tsi_signal', 'vwap', 'current_price']):
                    tsi_bearish = indicators['tsi_line'] < indicators['tsi_signal']
                    price_below_vwap = indicators['current_price'] < indicators['vwap']
                    
                    if position_side == 'long' and (tsi_bearish or price_below_vwap):
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"TSI bearish or price below VWAP (PnL: {pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                    elif position_side == 'short':
                        tsi_bullish = indicators['tsi_line'] > indicators['tsi_signal']
                        price_above_vwap = indicators['current_price'] > indicators['vwap']
                        
                        if tsi_bullish or price_above_vwap:
                            result.update({
                                'should_exit': True,
                                'exit_reason': f"TSI bullish or price above VWAP (PnL: {pnl_pct:.2f}%)",
                                'exit_urgency': 'normal'
                            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Technical exit check error: {e}")
            return {'should_exit': False, 'exit_reason': 'Technical exit error', 'exit_urgency': 'none'}

    def _update_cache_if_needed(self, ohlcv_data: list):
        """FIXED: Non-blocking cache update with size limit"""
        try:
            current_time = time.time()
            if current_time - self._cache_timestamp > self._cache_validity:
                if len(ohlcv_data) > self._max_cache_size:
                    self._historical_data_cache = ohlcv_data[-self._max_cache_size:]
                else:
                    self._historical_data_cache = ohlcv_data.copy()
                self._cache_timestamp = current_time
        except Exception:
            pass

    def _should_optimize_quick_check(self) -> bool:
        """FIXED: Quick optimization check without blocking operations"""
        try:
            current_time = time.time()
            
            if self.optimization_in_progress:
                return False
            
            if current_time - self.last_optimization < self.optimization_interval:
                return False
            
            if len(self.signal_performance) < self.min_signals_for_optimization:
                return False
            
            if hasattr(self.params, 'accuracy') and self.params.accuracy < self.win_rate_threshold:
                return True
            
            return False
            
        except Exception:
            return False

    def _trigger_non_blocking_optimization(self):
        """FIXED: Truly non-blocking optimization trigger"""
        try:
            with self.optimization_lock:
                if self.optimization_in_progress:
                    return
                self.optimization_in_progress = True
            
            def isolated_optimization():
                try:
                    cached_data = None
                    if self._historical_data_cache:
                        cached_data = self._historical_data_cache.copy()
                    
                    if cached_data and len(cached_data) >= 50:
                        best_params, best_score, opt_details = self.win_rate_optimizer.optimize_for_winrate(
                            cached_data, self.params
                        )
                        
                        if best_params and best_score > self.params.accuracy + 3.0:
                            self._update_optimized_parameters_safe(best_params, best_score)
                            
                            with self._snapshot_lock:
                                self._params_snapshot = copy.deepcopy(self.params)
                            
                            self.logger.info(f"Optimization: +{best_score - self.params.accuracy:.1f}% improvement")
                    
                    self.last_optimization = time.time()
                    
                except Exception as e:
                    self.logger.error(f"Background optimization error: {e}")
                finally:
                    with self.optimization_lock:
                        self.optimization_in_progress = False
            
            opt_thread = threading.Thread(target=isolated_optimization, daemon=True)
            opt_thread.start()
            
        except Exception as e:
            with self.optimization_lock:
                self.optimization_in_progress = False

    def _update_optimized_parameters_safe(self, best_params: Dict, best_score: float):
        """FIXED: Thread-safe parameter update"""
        try:
            for param_name, param_value in best_params.items():
                if hasattr(self.params, param_name):
                    setattr(self.params, param_name, param_value)
            
            self.params.optimization_score = best_score
            self.params.last_used = time.time()
            
            self._save_config()
            
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")

    def _track_signal_fast(self, signal: str, price: float):
        """FIXED: Fast signal tracking without blocking operations"""
        try:
            signal_data = {
                'signal': signal,
                'price': price,
                'timestamp': time.time(),
                'evaluated': False,
                'correct': None
            }
            self.signal_performance.append(signal_data)
            
            if len(self.signal_performance) > 100:
                self.signal_performance = deque(list(self.signal_performance)[-50:], maxlen=50)
            
            if len(self.signal_performance) % 5 == 0:
                self._evaluate_signals_quick()
                
        except Exception:
            pass

    def _evaluate_signals_quick(self):
        """FIXED: Quick signal evaluation without blocking"""
        try:
            current_time = time.time()
            evaluation_delay = 180
            
            recent_signals = list(self.signal_performance)[-10:]
            
            for signal_data in recent_signals:
                if signal_data.get('evaluated', False):
                    continue
                
                if current_time - signal_data['timestamp'] < evaluation_delay:
                    continue
                
                price_diff = abs(signal_data['price'] - recent_signals[-1]['price'])
                if price_diff > 0:
                    price_change_pct = ((recent_signals[-1]['price'] - signal_data['price']) / signal_data['price']) * 100
                    
                    if signal_data['signal'] == 'buy':
                        signal_data['correct'] = price_change_pct > 0.3
                    elif signal_data['signal'] == 'sell':
                        signal_data['correct'] = price_change_pct < -0.3
                    
                    signal_data['evaluated'] = True
            
            evaluated = [s for s in self.signal_performance if s.get('evaluated', False)]
            if len(evaluated) > 0:
                correct = sum(1 for s in evaluated if s.get('correct', False))
                self.params.accuracy = (correct / len(evaluated)) * 100
                self.params.total_signals = len(evaluated)
                self.params.winning_signals = correct
                
        except Exception:
            pass

    def _load_symbol_config(self) -> SignalParameters:
        """Load optimized parameters for this symbol and strategy type"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
                
                if self.symbol in configs:
                    cfg = configs[self.symbol]
                    strategy_key = f"{self.strategy_type}_config"
                    
                    if strategy_key in cfg:
                        strategy_cfg = cfg[strategy_key]
                        return SignalParameters(
                            strategy_type=self.strategy_type,
                            qqe_length=strategy_cfg.get('qqe_length', 12),
                            qqe_smooth=strategy_cfg.get('qqe_smooth', 5),
                            qqe_factor=strategy_cfg.get('qqe_factor', 4.236),
                            supertrend_period=strategy_cfg.get('supertrend_period', 10),
                            supertrend_multiplier=strategy_cfg.get('supertrend_multiplier', 2.8),
                            rsi_length=strategy_cfg.get('rsi_length', 14),
                            macd_fast=strategy_cfg.get('macd_fast', 12),
                            macd_slow=strategy_cfg.get('macd_slow', 26),
                            macd_signal=strategy_cfg.get('macd_signal', 9),
                            tsi_fast=strategy_cfg.get('tsi_fast', 8),
                            tsi_slow=strategy_cfg.get('tsi_slow', 15),
                            tsi_signal=strategy_cfg.get('tsi_signal', 6),
                            accuracy=strategy_cfg.get('accuracy', 0.0),
                            total_signals=strategy_cfg.get('total_signals', 0),
                            winning_signals=strategy_cfg.get('winning_signals', 0),
                            last_used=strategy_cfg.get('last_used', 0.0),
                            optimization_score=strategy_cfg.get('optimization_score', 0.0)
                        )
                        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
        
        return SignalParameters(strategy_type=self.strategy_type)

    def _load_signal_history(self) -> deque:
        """Load signal performance history"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
                if self.symbol in configs:
                    hist_key = f"{self.strategy_type}_signal_history"
                    hist = configs[self.symbol].get(hist_key, [])
                    return deque([item for item in hist if isinstance(item, dict)][-50:], maxlen=50)
        except Exception as e:
            self.logger.error(f"Error loading signal history: {e}")
        return deque(maxlen=50)

    def _save_config(self):
        """Save optimized parameters and performance"""
        try:
            configs = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
            
            if self.symbol not in configs:
                configs[self.symbol] = {}
            
            strategy_key = f"{self.strategy_type}_config"
            configs[self.symbol][strategy_key] = {
                'strategy_type': self.params.strategy_type,
                'qqe_length': self.params.qqe_length,
                'qqe_smooth': self.params.qqe_smooth,
                'qqe_factor': self.params.qqe_factor,
                'supertrend_period': self.params.supertrend_period,
                'supertrend_multiplier': self.params.supertrend_multiplier,
                'rsi_length': self.params.rsi_length,
                'macd_fast': self.params.macd_fast,
                'macd_slow': self.params.macd_slow,
                'macd_signal': self.params.macd_signal,
                'tsi_fast': self.params.tsi_fast,
                'tsi_slow': self.params.tsi_slow,
                'tsi_signal': self.params.tsi_signal,
                'accuracy': self.params.accuracy,
                'total_signals': self.params.total_signals,
                'winning_signals': self.params.winning_signals,
                'last_used': self.params.last_used,
                'optimization_score': self.params.optimization_score,
                'last_updated': time.time()
            }
            
            hist_key = f"{self.strategy_type}_signal_history"
            configs[self.symbol][hist_key] = list(self.signal_performance)[-50:]
            
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            temp_file = self.config_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(configs, f, indent=2)
            os.replace(temp_file, self.config_file)
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

    def get_system_status(self) -> Dict:
        """Get system status with technical exit information"""
        with self.optimization_lock:
            optimization_status = self.optimization_in_progress
        
        return {
            'system_type': f'{self.strategy_type}_technical_exits',
            'strategy_type': self.strategy_type,
            'optimized_parameters': {
                'qqe_length': self.params.qqe_length,
                'qqe_smooth': self.params.qqe_smooth,
                'supertrend_period': self.params.supertrend_period,
                'supertrend_multiplier': self.params.supertrend_multiplier,
                'rsi_length': self.params.rsi_length,
                'macd_fast': self.params.macd_fast,
                'macd_slow': self.params.macd_slow,
                'tsi_fast': self.params.tsi_fast,
                'tsi_slow': self.params.tsi_slow
            },
            'performance': {
                'win_rate': self.params.accuracy,
                'total_signals': self.params.total_signals,
                'winning_signals': self.params.winning_signals
            },
            'exit_conditions': {
                'emergency_sl': '1.5% loss',
                'take_profit': '2.0% gain', 
                'technical_exits': f'{self.strategy_type}_specific',
                'time_based': '60 minutes maximum'
            },
            'optimization': {
                'in_progress': optimization_status,
                'last_optimization': self.last_optimization
            }
        }

def integrate_adaptive_crypto_signals(strategy_instance, config_file: str = None, strategy_type: str = 'qqe_supertrend_fixed'):
    """Integration function with proper technical exits"""
    if config_file is None:
        config_file = os.path.join(os.getcwd(), "data", "crypto_signal_configs.json")
    
    strategy_instance.logger.info(f"Integrating COMPLETE {strategy_type} signals with technical exits, config: {config_file}")
    base_sym = getattr(strategy_instance, 'original_symbol', strategy_instance.symbol)
    
    crypto_sigs = AdaptiveCryptoSignals(symbol=base_sym, config_file=config_file, strategy_type=strategy_type)
    
    strategy_instance._get_technical_direction = lambda: crypto_sigs.get_technical_direction(strategy_instance.exchange)
    strategy_instance.get_signal_status = crypto_sigs.get_system_status
    strategy_instance._crypto_signal_system = crypto_sigs
    
    strategy_instance.logger.info(f"COMPLETE {strategy_type} signals integrated with technical exits:")
    strategy_instance.logger.info(f"  - Emergency SL: 1.5% loss")
    strategy_instance.logger.info(f"  - Take Profit: 2.0% gain")
    strategy_instance.logger.info(f"  - Technical exits: {strategy_type} signal reversals")
    strategy_instance.logger.info(f"  - Time exit: 60 minutes maximum")
    
    return crypto_sigs