"""
Fixed Adaptive Crypto Signal System - Current Regime Optimization with Multi-timeframe ROC
UPDATED: Added multi-timeframe ROC strategy (3m and 15m) following qqe_supertrend_fast pattern
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
    
    # ROC Multi-timeframe parameters
    roc_3m_length: int = 10
    roc_15m_length: int = 10
    roc_3m_threshold: float = 1.0
    roc_15m_threshold: float = 2.0
    roc_alignment_factor: float = 0.5
    
    # Performance tracking
    accuracy: float = 0.0
    total_signals: int = 0
    winning_signals: int = 0
    last_used: float = 0.0
    optimization_score: float = 0.0

class IntegratedWinRateOptimizer:
    """FIXED: Current regime optimizer for real-time parameter optimization"""
    
    def __init__(self, symbol: str, strategy_type: str):
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.logger = logging.getLogger(f"{__name__}.CurrentRegimeOptimizer.{symbol}")
        
        # FIXED: Current regime optimization settings
        self.optimization_timeout = 30
        self.n_trials = 20
        self.min_trades_threshold = 3
        
        # FIXED: Current regime parameter ranges (focused like test_strategies.py)
        self.current_regime_params = {
            'qqe_supertrend_fixed': {
                'qqe_length': [8, 10, 12],
                'qqe_smooth': [3, 5],
                'supertrend_period': [6, 10, 12],
                'supertrend_multiplier': [2.2, 2.6, 3.0]
            },
            'qqe_supertrend_fast': {
                'qqe_length': [6, 8, 10],
                'qqe_smooth': [2, 3],
                'supertrend_period': [4, 6, 8],
                'supertrend_multiplier': [2.0, 2.4, 2.8]
            },
            'rsi_macd': {
                'rsi_length': [12, 14, 16],
                'macd_fast': [10, 12],
                'macd_slow': [24, 26],
                'macd_signal': [8, 9]
            },
            'tsi_vwap': {
                'tsi_fast': [6, 8],
                'tsi_slow': [15, 21],
                'tsi_signal': [4, 6]
            },
            'roc_multi_timeframe': {
                'roc_3m_length': [8, 10, 12],
                'roc_15m_length': [8, 10, 12],
                'roc_3m_threshold': [0.8, 1.0, 1.5],
                'roc_15m_threshold': [1.5, 2.0, 2.5],
                'roc_alignment_factor': [0.3, 0.5, 0.7]
            }
        }
        
    def optimize_for_winrate(self, historical_data: list, current_params: SignalParameters) -> Tuple[Dict, float, Dict]:
        """FIXED: Current regime optimization using recent data only"""
        try:
            if not historical_data or len(historical_data) < 50:
                return self._params_to_dict(current_params), 0.0, {}
            
            # FIXED: Use only recent data for current regime optimization
            recent_data = historical_data[-150:] if len(historical_data) > 150 else historical_data
            
            self.logger.info(f"Current regime optimization: {len(recent_data)} recent candles")
            
            best_params = None
            best_score = current_params.accuracy
            
            # FIXED: Use focused parameter combinations for current regime
            param_ranges = self.current_regime_params.get(self.strategy_type, {})
            if not param_ranges:
                return self._params_to_dict(current_params), 0.0, {}
            
            # FIXED: Current regime parameter testing
            if 'qqe' in self.strategy_type.lower():
                param_combinations = [
                    (ql, qs, sp, sm) for ql in param_ranges.get('qqe_length', [12])
                    for qs in param_ranges.get('qqe_smooth', [5])
                    for sp in param_ranges.get('supertrend_period', [10])
                    for sm in param_ranges.get('supertrend_multiplier', [2.8])
                ]
                
                for ql, qs, sp, sm in param_combinations:
                    test_params = SignalParameters(
                        strategy_type=self.strategy_type,
                        qqe_length=ql,
                        qqe_smooth=qs,
                        supertrend_period=sp,
                        supertrend_multiplier=sm
                    )
                    
                    score = self._current_regime_backtest(recent_data, test_params)
                    if score > best_score:
                        best_score = score
                        best_params = self._params_to_dict(test_params)
                        
            elif self.strategy_type == 'rsi_macd':
                param_combinations = [
                    (rl, mf, ms, msig) for rl in param_ranges.get('rsi_length', [14])
                    for mf in param_ranges.get('macd_fast', [12])
                    for ms in param_ranges.get('macd_slow', [26])
                    for msig in param_ranges.get('macd_signal', [9])
                    if mf < ms
                ]
                
                for rl, mf, ms, msig in param_combinations:
                    test_params = SignalParameters(
                        strategy_type=self.strategy_type,
                        rsi_length=rl,
                        macd_fast=mf,
                        macd_slow=ms,
                        macd_signal=msig
                    )
                    
                    score = self._current_regime_backtest(recent_data, test_params)
                    if score > best_score:
                        best_score = score
                        best_params = self._params_to_dict(test_params)
                        
            elif self.strategy_type == 'tsi_vwap':
                param_combinations = [
                    (tf, ts, tsig) for tf in param_ranges.get('tsi_fast', [8])
                    for ts in param_ranges.get('tsi_slow', [15])
                    for tsig in param_ranges.get('tsi_signal', [6])
                    if tf < ts
                ]
                
                for tf, ts, tsig in param_combinations:
                    test_params = SignalParameters(
                        strategy_type=self.strategy_type,
                        tsi_fast=tf,
                        tsi_slow=ts,
                        tsi_signal=tsig
                    )
                    
                    score = self._current_regime_backtest(recent_data, test_params)
                    if score > best_score:
                        best_score = score
                        best_params = self._params_to_dict(test_params)
                        
            elif self.strategy_type == 'roc_multi_timeframe':
                param_combinations = [
                    (r3l, r15l, r3t, r15t, raf) 
                    for r3l in param_ranges.get('roc_3m_length', [10])
                    for r15l in param_ranges.get('roc_15m_length', [10])
                    for r3t in param_ranges.get('roc_3m_threshold', [1.0])
                    for r15t in param_ranges.get('roc_15m_threshold', [2.0])
                    for raf in param_ranges.get('roc_alignment_factor', [0.5])
                ]
                
                for r3l, r15l, r3t, r15t, raf in param_combinations:
                    test_params = SignalParameters(
                        strategy_type=self.strategy_type,
                        roc_3m_length=r3l,
                        roc_15m_length=r15l,
                        roc_3m_threshold=r3t,
                        roc_15m_threshold=r15t,
                        roc_alignment_factor=raf
                    )
                    
                    score = self._current_regime_backtest(recent_data, test_params)
                    if score > best_score:
                        best_score = score
                        best_params = self._params_to_dict(test_params)
            
            return best_params or self._params_to_dict(current_params), best_score, {}
            
        except Exception as e:
            self.logger.error(f"Current regime optimization error: {e}")
            return self._params_to_dict(current_params), 0.0, {}
    
    def _current_regime_backtest(self, recent_data: list, test_params: SignalParameters) -> float:
        """FIXED: Current regime backtesting using recent momentum patterns"""
        try:
            df = pd.DataFrame(recent_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            if len(df) < 30:
                return 0.0
            
            signals = []
            start_idx = max(20, len(df) - 100)
            
            for i in range(start_idx, len(df)):
                current_df = df.iloc[:i+1]
                signal = self._generate_current_regime_signal(current_df, test_params)
                if signal != 'none':
                    signals.append({'signal': signal, 'price': current_df['close'].iloc[-1], 'index': i})
            
            if len(signals) < self.min_trades_threshold:
                return 0.0
            
            wins = 0
            total = 0
            
            for signal_data in signals:
                if signal_data['index'] + 3 < len(df):
                    future_price = df['close'].iloc[signal_data['index'] + 3]
                    price_change = ((future_price - signal_data['price']) / signal_data['price']) * 100
                    
                    if signal_data['signal'] == 'buy' and price_change > 0.15:
                        wins += 1
                    elif signal_data['signal'] == 'sell' and price_change < -0.15:
                        wins += 1
                    total += 1
            
            if total == 0:
                return 0.0
            
            win_rate = (wins / total) * 100.0
            
            regime_bonus = 0
            if len(signals) >= 5:
                regime_bonus = 5.0
            if win_rate > 40:
                regime_bonus += 10.0
                
            return min(win_rate + regime_bonus, 100.0)
            
        except Exception:
            return 0.0
    
    def _generate_current_regime_signal(self, df: pd.DataFrame, test_params: SignalParameters) -> str:
        """FIXED: Generate signals using current regime parameters"""
        try:
            if 'qqe' in self.strategy_type.lower():
                qqe_result = ta.qqe(df['close'], 
                                  length=test_params.qqe_length, 
                                  smooth=test_params.qqe_smooth)
                st_result = ta.supertrend(df['high'], df['low'], df['close'], 
                                        length=test_params.supertrend_period, 
                                        multiplier=test_params.supertrend_multiplier)
                
                if (qqe_result is not None and not qqe_result.empty and len(qqe_result.columns) >= 2 and
                    st_result is not None and not st_result.empty):
                    
                    dir_col = next((col for col in st_result.columns if 'SUPERTd' in col), None)
                    if dir_col:
                        rsi_ma = qqe_result.iloc[-1, 1]
                        qqe_line = qqe_result.iloc[-1, 0]
                        st_dir = st_result[dir_col].iloc[-1]

                        if rsi_ma > qqe_line and st_dir == 1:
                            return 'buy'
                        elif qqe_line > rsi_ma and st_dir != 1:
                            return 'sell'
                            
            elif self.strategy_type == 'rsi_macd':
                rsi = ta.rsi(df['close'], length=test_params.rsi_length)
                macd = ta.macd(df['close'], 
                             fast=test_params.macd_fast,
                             slow=test_params.macd_slow,
                             signal=test_params.macd_signal)
                
                if rsi is not None and macd is not None and not macd.empty and len(macd.columns) >= 3:
                    rsi_val = rsi.iloc[-1]
                    macd_line = macd.iloc[-1, 0]
                    macd_signal = macd.iloc[-1, 2]
                    
                    if rsi_val < 30 and macd_line > macd_signal:
                        return 'buy'
                    elif rsi_val > 70 and macd_line < macd_signal:
                        return 'sell'
                        
            elif self.strategy_type == 'tsi_vwap':
                tsi = ta.tsi(df['close'], 
                           fast=test_params.tsi_fast,
                           slow=test_params.tsi_slow,
                           signal=test_params.tsi_signal)
                vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                
                if tsi is not None and not tsi.empty and vwap is not None:
                    tsi_line = tsi.iloc[-1, 0]
                    tsi_signal = tsi.iloc[-1, 1] if len(tsi.columns) > 1 else 0
                    price = df['close'].iloc[-1]
                    vwap_val = vwap.iloc[-1]
                    
                    if tsi_line > tsi_signal and price > vwap_val:
                        return 'buy'
                    elif tsi_line < tsi_signal and price < vwap_val:
                        return 'sell'
                        
            elif self.strategy_type == 'roc_multi_timeframe':
                if len(df) < max(test_params.roc_3m_length, test_params.roc_15m_length) + 5:
                    return 'none'
                
                # Calculate 3m ROC (current timeframe)
                roc_3m = ta.roc(df['close'], length=test_params.roc_3m_length)
                if roc_3m is None or len(roc_3m.dropna()) < 5:
                    return 'none'
                
                # Simulate 15m timeframe by taking every 5th candle
                df_15m = df.iloc[::5].copy()
                if len(df_15m) < test_params.roc_15m_length + 2:
                    return 'none'
                
                roc_15m = ta.roc(df_15m['close'], length=test_params.roc_15m_length)
                if roc_15m is None or len(roc_15m.dropna()) < 2:
                    return 'none'
                
                # Get current ROC values
                current_roc_3m = roc_3m.iloc[-1]
                current_roc_15m = roc_15m.iloc[-1]
                
                if pd.isna(current_roc_3m) or pd.isna(current_roc_15m):
                    return 'none'
                
                # Multi-timeframe alignment logic
                roc_3m_bullish = current_roc_3m > test_params.roc_3m_threshold
                roc_15m_bullish = current_roc_15m > test_params.roc_15m_threshold
                roc_3m_bearish = current_roc_3m < -test_params.roc_3m_threshold
                roc_15m_bearish = current_roc_15m < -test_params.roc_15m_threshold
                
                # Alignment factor for timeframe confluence
                alignment_strength = abs(current_roc_3m * current_roc_15m) * test_params.roc_alignment_factor
                
                # Strong alignment signals
                if roc_3m_bullish and roc_15m_bullish and alignment_strength > 1.0:
                    return 'buy'
                elif roc_3m_bearish and roc_15m_bearish and alignment_strength > 1.0:
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
            'tsi_signal': params.tsi_signal,
            'roc_3m_length': params.roc_3m_length,
            'roc_15m_length': params.roc_15m_length,
            'roc_3m_threshold': params.roc_3m_threshold,
            'roc_15m_threshold': params.roc_15m_threshold,
            'roc_alignment_factor': params.roc_alignment_factor
        }

class AdaptiveCryptoSignals:
    """FIXED: Current regime adaptive signals with pre-order optimization"""
    
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
        
        # FIXED: Initialize current regime optimizer
        self.win_rate_optimizer = IntegratedWinRateOptimizer(symbol, strategy_type)
        
        # FIXED: Faster signal control for current regime
        self.last_signal = 'none'
        self.last_signal_time = 0
        self.signal_cooldown = 3
        
        # FIXED: Store last calculated indicators for exit evaluation
        self._last_calculated_indicators = {}
        self._indicator_lock = threading.RLock()
        
        # FIXED: Current regime optimization control
        self.optimization_in_progress = False
        self.optimization_lock = threading.RLock()
        self.last_optimization = time.time()
        
        # FIXED: Current regime settings
        self.indicator_timeout = 1.5
        
        # FIXED: Current regime optimization triggers
        self.min_signals_for_optimization = 8
        self.optimization_interval = 120
        self.win_rate_threshold = 35.0
        
        # FIXED: Current data cache settings
        self._historical_data_cache = None
        self._cache_timestamp = 0
        self._cache_validity = 180
        self._max_cache_size = 200
        
        self.logger.info(f"CURRENT REGIME: Adaptive Crypto Signals for {strategy_type} on {symbol}")

    def get_technical_direction(self, exchange) -> str:
        """FIXED: TRUE pre-order optimization - parameters optimized BEFORE signal generation"""
        try:
            current_time = time.time()
            if current_time - self.last_signal_time < self.signal_cooldown:
                return 'none'
            
            # Get data based on strategy type
            if self.strategy_type == 'roc_multi_timeframe':
                # FIXED: Get both timeframes separately
                ohlcv_3m = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=200)
                ohlcv_15m = exchange.get_ohlcv(self.symbol, timeframe='15m', limit=50)
                
                if not ohlcv_3m or len(ohlcv_3m) < 50 or not ohlcv_15m or len(ohlcv_15m) < 20:
                    return 'none'
                    
                # Pass both datasets
                ohlcv_data = {'3m': ohlcv_3m, '15m': ohlcv_15m}
            else:
                ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=200)
                
            if (self.strategy_type != 'roc_multi_timeframe' and 
                (not ohlcv_data or len(ohlcv_data) < 50)):
                return 'none'
            
            # FIXED: TRUE PRE-ORDER OPTIMIZATION
            if self._should_optimize_before_signal(ohlcv_data):
                self.logger.info(f"PRE-ORDER OPTIMIZATION: Optimizing parameters before signal generation")
                optimized = self._execute_pre_order_optimization(ohlcv_data)
                if optimized:
                    self.logger.info(f"PRE-ORDER OPTIMIZATION: Parameters updated before signal")
                else:
                    self.logger.warning(f"PRE-ORDER OPTIMIZATION: Using existing parameters")
            
            # FIXED: Use current parameters
            with self._snapshot_lock:
                current_params = copy.deepcopy(self._params_snapshot)
            
            # FIXED: Generate signal with guaranteed current regime parameters
            signal, indicators = self._generate_signal_with_indicators(ohlcv_data, current_params)
            
            # FIXED: Store indicators for exit evaluation
            if indicators:
                with self._indicator_lock:
                    self._last_calculated_indicators = indicators.copy()
            
            if signal != 'none':
                self._track_signal_fast(signal, float(ohlcv_data[-1][4]))
                self.last_signal = signal
                self.last_signal_time = current_time
                self.logger.info(f"OPTIMIZED SIGNAL: {signal.upper()} @ ${float(ohlcv_data[-1][4]):.6f}")
            
            # FIXED: Update cache for current regime
            self._update_cache_if_needed(ohlcv_data)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in current regime signal generation: {e}")
            return 'none'

    def _should_optimize_before_signal(self, ohlcv_data: list) -> bool:
        """FIXED: Smart pre-order optimization triggers"""
        try:
            current_time = time.time()
            
            if self.optimization_in_progress:
                return False
            
            if current_time - self.last_optimization < 180:
                return False
            
            if len(self.signal_performance) < self.min_signals_for_optimization:
                return False
            
            # TRIGGER 1: Low accuracy
            if hasattr(self.params, 'accuracy') and self.params.accuracy < self.win_rate_threshold:
                self.logger.info(f"PRE-ORDER TRIGGER: Low accuracy {self.params.accuracy:.1f}% < {self.win_rate_threshold}%")
                return True
            
            # TRIGGER 2: No optimization yet but have signals
            if self.last_optimization == 0 and len(self.signal_performance) >= 10:
                self.logger.info(f"PRE-ORDER TRIGGER: Initial optimization needed ({len(self.signal_performance)} signals)")
                return True
            
            # TRIGGER 3: Long time since last optimization with poor recent performance
            time_since_opt = current_time - self.last_optimization
            if time_since_opt > 300:
                recent_signals = list(self.signal_performance)[-5:]
                if len(recent_signals) >= 3:
                    recent_correct = sum(1 for s in recent_signals if s.get('evaluated') and s.get('correct'))
                    recent_total = sum(1 for s in recent_signals if s.get('evaluated'))
                    if recent_total >= 2:
                        recent_accuracy = (recent_correct / recent_total) * 100
                        if recent_accuracy < 30:
                            self.logger.info(f"PRE-ORDER TRIGGER: Poor recent performance {recent_accuracy:.1f}%")
                            return True
            
            # TRIGGER 4: Time-based
            if len(ohlcv_data) >= 150 and time_since_opt > 600:
                self.logger.info(f"PRE-ORDER TRIGGER: Stale parameters ({time_since_opt/60:.1f} min old)")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in pre-order optimization check: {e}")
            return False

    def _execute_pre_order_optimization(self, ohlcv_data: list) -> bool:
        """FIXED: SYNCHRONOUS pre-order optimization"""
        try:
            with self.optimization_lock:
                if self.optimization_in_progress:
                    self.logger.info(f"PRE-ORDER OPTIMIZATION: Already in progress, using existing parameters")
                    return False
                self.optimization_in_progress = True
            
            try:
                start_time = time.time()
                self.logger.info(f"PRE-ORDER OPTIMIZATION: Starting synchronous optimization for {self.symbol}")
                
                current_data = ohlcv_data.copy()
                
                if current_data and len(current_data) >= 100:
                    best_params, best_score, opt_details = self.win_rate_optimizer.optimize_for_winrate(
                        current_data, self.params
                    )
                    
                    if best_params and best_score > self.params.accuracy + 2.0:
                        old_accuracy = self.params.accuracy
                        self._update_optimized_parameters_safe(best_params, best_score)
                        
                        with self._snapshot_lock:
                            self._params_snapshot = copy.deepcopy(self.params)
                        
                        optimization_time = time.time() - start_time
                        self.logger.info(f"PRE-ORDER OPTIMIZATION: Parameters updated in {optimization_time:.2f}s")
                        self.logger.info(f"PRE-ORDER OPTIMIZATION: Accuracy {old_accuracy:.1f}% -> {best_score:.1f}%")
                        
                        self.last_optimization = time.time()
                        return True
                    else:
                        optimization_time = time.time() - start_time
                        self.logger.info(f"PRE-ORDER OPTIMIZATION: No improvement found in {optimization_time:.2f}s")
                        self.last_optimization = time.time()
                        return False
                else:
                    self.logger.warning(f"PRE-ORDER OPTIMIZATION: Insufficient data ({len(current_data)} candles)")
                    return False
                    
            except Exception as e:
                self.logger.error(f"PRE-ORDER OPTIMIZATION: Error during optimization: {e}")
                return False
            finally:
                with self.optimization_lock:
                    self.optimization_in_progress = False
                    
        except Exception as e:
            self.logger.error(f"PRE-ORDER OPTIMIZATION: Critical error: {e}")
            with self.optimization_lock:
                self.optimization_in_progress = False
            return False

    def _generate_signal_with_indicators(self, ohlcv_data: list, params) -> Tuple[str, Dict]:
        """FIXED: Generate signal using current regime optimized parameters"""
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            if len(df) < 30:
                return 'none', {}
            
            # FIXED: Use current regime parameters for signal generation
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                return self._qqe_supertrend_signal_with_indicators(df, params)
            elif self.strategy_type == 'rsi_macd':
                return self._rsi_macd_signal_with_indicators(df, params)
            elif self.strategy_type == 'tsi_vwap':
                return self._tsi_vwap_signal_with_indicators(df, params)
            elif self.strategy_type == 'roc_multi_timeframe':
                return self._roc_multi_timeframe_signal_with_indicators(df, params)
            
            return 'none', {}
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return 'none', {}

    def _qqe_supertrend_signal_with_indicators(self, df: pd.DataFrame, params) -> Tuple[str, Dict]:
        """QQE + Supertrend signal with indicators for exit evaluation"""
        try:
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
            
            dir_col = next((col for col in st_result.columns if 'SUPERTd' in col), None)
            if not dir_col:
                return 'none', {}
            
            lookback_period = min(5, len(qqe_result) - 1)
            if lookback_period < 1:
                return 'none', {}
            
            rsi_ma_current = qqe_result.iloc[-1, 1]
            qqe_line_current = qqe_result.iloc[-1, 0]
            st_dir_current = st_result[dir_col].iloc[-1]
            
            indicators = {
                'qqe_value': float(rsi_ma_current),
                'qqe_signal': float(qqe_line_current),
                'st_direction': int(st_dir_current),
                'current_price': float(df['close'].iloc[-1])
            }
            
            qqe_currently_bullish = rsi_ma_current > qqe_line_current
            qqe_currently_bearish = rsi_ma_current <= qqe_line_current
            st_currently_positive = st_dir_current == 1
            st_currently_negative = st_dir_current == -1
            
            qqe_recently_turned_bullish = False
            for i in range(1, lookback_period + 1):
                rsi_ma_past = qqe_result.iloc[-1-i, 1]
                qqe_line_past = qqe_result.iloc[-1-i, 0]
                if rsi_ma_past <= qqe_line_past and qqe_currently_bullish:
                    qqe_recently_turned_bullish = True
                    break
            
            qqe_recently_turned_bearish = False
            for i in range(1, lookback_period + 1):
                rsi_ma_past = qqe_result.iloc[-1-i, 1]
                qqe_line_past = qqe_result.iloc[-1-i, 0]
                if rsi_ma_past > qqe_line_past and qqe_currently_bearish:
                    qqe_recently_turned_bearish = True
                    break
            
            st_recently_turned_positive = False
            for i in range(1, lookback_period + 1):
                st_dir_past = st_result[dir_col].iloc[-1-i]
                if st_dir_past != 1 and st_currently_positive:
                    st_recently_turned_positive = True
                    break
            
            st_recently_turned_negative = False
            for i in range(1, lookback_period + 1):
                st_dir_past = st_result[dir_col].iloc[-1-i]
                if st_dir_past != -1 and st_currently_negative:
                    st_recently_turned_negative = True
                    break
            
            if st_currently_positive and (qqe_recently_turned_bullish or (qqe_currently_bullish and st_recently_turned_positive)):
                return 'buy', indicators
            elif st_currently_negative and (qqe_recently_turned_bearish or (qqe_currently_bearish and st_recently_turned_negative)):
                return 'sell', indicators
            
            return 'none', indicators
            
        except Exception as e:
            self.logger.error(f"QQE+ST calculation error: {e}")
            return 'none', {}

    def _rsi_macd_signal_with_indicators(self, df: pd.DataFrame, params) -> Tuple[str, Dict]:
        """RSI + MACD with current regime parameters"""
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
        """TSI + VWAP with current regime parameters"""
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

    def _roc_multi_timeframe_signal_with_indicators(self, data_input, params) -> Tuple[str, Dict]:
        """FIXED: Multi-timeframe ROC signal with clear logic like TSI+VWAP"""
        try:
            if isinstance(data_input, dict):
                # New format with separate timeframes
                df_3m = pd.DataFrame(data_input['3m'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_15m = pd.DataFrame(data_input['15m'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_3m[col] = df_3m[col].astype(float)
                    df_15m[col] = df_15m[col].astype(float)
                    
                df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'], unit='ms')
                df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'], unit='ms')
                df_3m = df_3m.set_index('timestamp')
                df_15m = df_15m.set_index('timestamp')
            else:
                # Fallback to old method for compatibility
                df_3m = data_input.copy()
                # Create 15m from 3m by proper resampling
                if isinstance(df_3m.index, pd.DatetimeIndex):
                    df_15m = df_3m.resample('15min').agg({
                        'open': 'first',
                        'high': 'max', 
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                else:
                    return 'none', {}
            
            # Ensure we have enough data
            if len(df_3m) < max(params.roc_3m_length, 10) or len(df_15m) < max(params.roc_15m_length, 5):
                return 'none', {}
            
            # Calculate ROC for both timeframes
            roc_3m = ta.roc(df_3m['close'], length=params.roc_3m_length)
            roc_15m = ta.roc(df_15m['close'], length=params.roc_15m_length)
            
            if roc_3m is None or len(roc_3m.dropna()) < 5 or roc_15m is None or len(roc_15m.dropna()) < 3:
                return 'none', {}
            
            # Get current values
            current_roc_3m = roc_3m.iloc[-1]
            current_roc_15m = roc_15m.iloc[-1]
            
            if pd.isna(current_roc_3m) or pd.isna(current_roc_15m):
                return 'none', {}
            
            # Store indicators for exit evaluation
            indicators = {
                'roc_3m': float(current_roc_3m),
                'roc_15m': float(current_roc_15m),
                'roc_3m_threshold': params.roc_3m_threshold,
                'roc_15m_threshold': params.roc_15m_threshold,
                'alignment_factor': params.roc_alignment_factor,
                'current_price': float(df_3m['close'].iloc[-1])
            }
            
            # FIXED: Simple bullish/bearish logic like TSI+VWAP
            roc_3m_bullish = current_roc_3m > params.roc_3m_threshold
            roc_15m_bullish = current_roc_15m > params.roc_15m_threshold
            roc_3m_bearish = current_roc_3m < -params.roc_3m_threshold
            roc_15m_bearish = current_roc_15m < -params.roc_15m_threshold
            
            # FIXED: Clear entry signals - both timeframes must agree
            if roc_3m_bullish and roc_15m_bullish:
                return 'buy', indicators
            elif roc_3m_bearish and roc_15m_bearish:
                return 'sell', indicators
            
            return 'none', indicators
            
        except Exception as e:
            self.logger.error(f"ROC Multi-timeframe calculation error: {e}")
            return 'none', {}

    def evaluate_exit_conditions(self, position_side: str, entry_price: float, current_price: float) -> Dict:
        """Current regime exit evaluation using optimized parameters"""
        try:
            result = {'should_exit': False, 'exit_reason': '', 'exit_urgency': 'none'}
            
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if position_side == 'long' else ((entry_price - current_price) / entry_price) * 100
            
            # PRIORITY 1: Emergency stop loss
            if pnl_pct < -1.5:
                result.update({
                    'should_exit': True,
                    'exit_reason': f"Emergency SL: {pnl_pct:.2f}% loss",
                    'exit_urgency': 'immediate'
                })
                return result

            # PRIORITY 2: Take profit
            if pnl_pct > 2.0:
                result.update({
                    'should_exit': True,
                    'exit_reason': f"Take Profit: {pnl_pct:.2f}% gain",
                    'exit_urgency': 'normal'
                })
                return result

            # PRIORITY 3: Technical signal-based exits
            try:
                with self._indicator_lock:
                    indicators = self._last_calculated_indicators.copy() if self._last_calculated_indicators else {}
                
                if indicators:
                    signal_exit = self._check_technical_exit_conditions(position_side, indicators, pnl_pct)
                    if signal_exit['should_exit']:
                        return signal_exit
                        
            except Exception as e:
                self.logger.error(f"Technical exit check error: {e}")

            # PRIORITY 4: Time-based exit
            if self.position_entry_time > 0:
                position_age_minutes = (time.time() - self.position_entry_time) / 60
                if position_age_minutes > 60:
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
        """Strategy-specific technical exit conditions using current regime indicators"""
        try:
            result = {'should_exit': False, 'exit_reason': '', 'exit_urgency': 'none'}
            
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                if all(key in indicators for key in ['qqe_value', 'qqe_signal', 'st_direction']):
                    qqe_bearish = indicators['qqe_signal'] > indicators['qqe_value'] 
                    st_bearish = indicators['st_direction'] != 1
                    
                    if position_side == 'long' and (qqe_bearish or st_bearish):
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"QQE+ST bearish reversal (PnL: {pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                    elif position_side == 'short':
                        qqe_bullish = indicators['qqe_value'] > indicators['qqe_signal']
                        st_bullish = indicators['st_direction'] == 1
                        
                        if qqe_bullish or st_bullish:
                            result.update({
                                'should_exit': True,
                                'exit_reason': f"QQE+ST bullish reversal (PnL: {pnl_pct:.2f}%)",
                                'exit_urgency': 'normal'
                            })
            
            elif self.strategy_type == 'rsi_macd':
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
                            
            elif self.strategy_type == 'roc_multi_timeframe':
                # FIXED: Clear ROC exit logic
                if all(key in indicators for key in ['roc_3m', 'roc_15m', 'roc_3m_threshold', 'roc_15m_threshold']):
                    roc_3m = indicators['roc_3m']
                    roc_15m = indicators['roc_15m']
                    threshold_3m = indicators['roc_3m_threshold']
                    threshold_15m = indicators['roc_15m_threshold']
                    
                    # FIXED: Exit on either timeframe turning opposite
                    if position_side == 'long':
                        roc_3m_bearish = roc_3m < -threshold_3m
                        roc_15m_bearish = roc_15m < -threshold_15m
                        
                        if roc_3m_bearish or roc_15m_bearish:
                            result.update({
                                'should_exit': True,
                                'exit_reason': f"ROC bearish: 3m={roc_3m:.2f}, 15m={roc_15m:.2f} (PnL: {pnl_pct:.2f}%)",
                                'exit_urgency': 'normal'
                            })
                    elif position_side == 'short':
                        roc_3m_bullish = roc_3m > threshold_3m
                        roc_15m_bullish = roc_15m > threshold_15m
                        
                        if roc_3m_bullish or roc_15m_bullish:
                            result.update({
                                'should_exit': True,
                                'exit_reason': f"ROC bullish: 3m={roc_3m:.2f}, 15m={roc_15m:.2f} (PnL: {pnl_pct:.2f}%)",
                                'exit_urgency': 'normal'
                            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Technical exit check error: {e}")
            return {'should_exit': False, 'exit_reason': 'Technical exit error', 'exit_urgency': 'none'}

    def _update_cache_if_needed(self, ohlcv_data: list):
        """Current regime cache update with recent data focus"""
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

    def _update_optimized_parameters_safe(self, best_params: Dict, best_score: float):
        """Thread-safe parameter update"""
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
        """Fast signal tracking for current regime"""
        try:
            signal_data = {
                'signal': signal,
                'price': price,
                'timestamp': time.time(),
                'evaluated': False,
                'correct': None
            }
            self.signal_performance.append(signal_data)
            
            if len(self.signal_performance) > 50:
                self.signal_performance = deque(list(self.signal_performance)[-30:], maxlen=50)
            
            if len(self.signal_performance) % 3 == 0:
                self._evaluate_signals_quick()
                
        except Exception:
            pass

    def _evaluate_signals_quick(self):
        """Quick current regime signal evaluation"""
        try:
            current_time = time.time()
            evaluation_delay = 120
            
            recent_signals = list(self.signal_performance)[-8:]
            
            for signal_data in recent_signals:
                if signal_data.get('evaluated', False):
                    continue
                
                if current_time - signal_data['timestamp'] < evaluation_delay:
                    continue
                
                price_diff = abs(signal_data['price'] - recent_signals[-1]['price'])
                if price_diff > 0:
                    price_change_pct = ((recent_signals[-1]['price'] - signal_data['price']) / signal_data['price']) * 100
                    
                    if signal_data['signal'] == 'buy':
                        signal_data['correct'] = price_change_pct > 0.2
                    elif signal_data['signal'] == 'sell':
                        signal_data['correct'] = price_change_pct < -0.2
                    
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
                            roc_3m_length=strategy_cfg.get('roc_3m_length', 10),
                            roc_15m_length=strategy_cfg.get('roc_15m_length', 10),
                            roc_3m_threshold=strategy_cfg.get('roc_3m_threshold', 1.0),
                            roc_15m_threshold=strategy_cfg.get('roc_15m_threshold', 2.0),
                            roc_alignment_factor=strategy_cfg.get('roc_alignment_factor', 0.5),
                            accuracy=strategy_cfg.get('accuracy', 0.0),
                            total_signals=strategy_cfg.get('total_signals', 0),
                            winning_signals=strategy_cfg.get('winning_signals', 0),
                            last_used=strategy_cfg.get('last_used', 0.0),
                            optimization_score=strategy_cfg.get('optimization_score', 0.0)
                        )
                        
        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupted JSON config file: {e}")
            self._handle_corrupted_config_file()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
        
        return SignalParameters(strategy_type=self.strategy_type)
        
    def _handle_corrupted_config_file(self):
        """Handle corrupted config file by backing up and recreating"""
        try:
            if os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.corrupted.{int(time.time())}"
                os.rename(self.config_file, backup_file)
                self.logger.warning(f"Moved corrupted config to: {backup_file}")
            
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump({}, f, indent=2)
            
            self.logger.info(f"Created new clean config file: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error handling corrupted config file: {e}")
            
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
        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupted JSON config file: {e}")
            self._handle_corrupted_config_file()
        except Exception as e:
            self.logger.error(f"Error loading signal history: {e}")
        return deque(maxlen=50)

    def _save_config(self):
        """Save current regime optimized parameters"""
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
                'roc_3m_length': self.params.roc_3m_length,
                'roc_15m_length': self.params.roc_15m_length,
                'roc_3m_threshold': self.params.roc_3m_threshold,
                'roc_15m_threshold': self.params.roc_15m_threshold,
                'roc_alignment_factor': self.params.roc_alignment_factor,
                'accuracy': self.params.accuracy,
                'total_signals': self.params.total_signals,
                'winning_signals': self.params.winning_signals,
                'last_used': self.params.last_used,
                'optimization_score': self.params.optimization_score,
                'last_updated': time.time()
            }
            
            hist_key = f"{self.strategy_type}_signal_history"
            configs[self.symbol][hist_key] = list(self.signal_performance)[-30:]
            
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
        """Get current regime system status"""
        with self.optimization_lock:
            optimization_status = self.optimization_in_progress
        
        status = {
            'system_type': f'{self.strategy_type}_current_regime',
            'strategy_type': self.strategy_type,
            'current_regime_parameters': {
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
                'technical_exits': f'{self.strategy_type}_current_regime',
                'time_based': '60 minutes maximum'
            },
            'current_regime_optimization': {
                'in_progress': optimization_status,
                'last_optimization': self.last_optimization,
                'pre_order_optimization': 'enabled'
            }
        }
        
        # Add ROC-specific parameters if applicable
        if self.strategy_type == 'roc_multi_timeframe':
            status['current_regime_parameters'].update({
                'roc_3m_length': self.params.roc_3m_length,
                'roc_15m_length': self.params.roc_15m_length,
                'roc_3m_threshold': self.params.roc_3m_threshold,
                'roc_15m_threshold': self.params.roc_15m_threshold,
                'roc_alignment_factor': self.params.roc_alignment_factor
            })
        
        return status

def integrate_adaptive_crypto_signals(strategy_instance, config_file: str = None, strategy_type: str = 'qqe_supertrend_fixed'):
    """Integration with current regime optimization"""
    if config_file is None:
        config_file = os.path.join(os.getcwd(), "data", "crypto_signal_configs.json")
    
    strategy_instance.logger.info(f"Integrating CURRENT REGIME {strategy_type} signals with pre-order optimization")
    base_sym = getattr(strategy_instance, 'original_symbol', strategy_instance.symbol)
    
    crypto_sigs = AdaptiveCryptoSignals(symbol=base_sym, config_file=config_file, strategy_type=strategy_type)
    
    strategy_instance._get_technical_direction = lambda: crypto_sigs.get_technical_direction(strategy_instance.exchange)
    strategy_instance.get_signal_status = crypto_sigs.get_system_status
    strategy_instance._crypto_signal_system = crypto_sigs
    
    strategy_instance.logger.info(f"CURRENT REGIME {strategy_type} signals integrated:")
    strategy_instance.logger.info(f"  - Pre-order optimization: enabled")
    strategy_instance.logger.info(f"  - Current regime parameters: focused ranges")
    strategy_instance.logger.info(f"  - Emergency SL: 1.5% loss")
    strategy_instance.logger.info(f"  - Take Profit: 2.0% gain")
    strategy_instance.logger.info(f"  - Technical exits: {strategy_type} signal reversals")
    
    return crypto_sigs