"""
Fixed Adaptive Crypto Signal System - Current Regime Optimization with Multi-timeframe ROC
UPDATED: Added multi-timeframe ROC strategy (3m and 15m) following qqe_supertrend_fast pattern
"""

import logging
import time
import os
import threading
import ollama
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
# from llm_signal_fusion import get_llm_fusion, LLMConfig
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

@dataclass
class SignalParameters:
    """Parameters for trading signals with optimization tracking"""
    
    strategy_type: str = 'roc_multi_timeframe'
    
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
    roc_3m_length: int = 5
    roc_15m_length: int = 20
    roc_3m_threshold: float = 0.5
    roc_15m_threshold: float = 0.5
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
    
    def __init__(self, symbol: str, config_file: str = "data/crypto_signal_configs.json", strategy_type: str = 'qqe_supertrend_fixed', enable_llm: bool = True):
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
        # Initialize LLM fusion
        self.enable_llm = enable_llm
        self._llm_config = self._load_llm_config()
        self._llm_enabled = self._llm_config.get('enabled', False)
        
        # self.logger.info(f"CURRENT REGIME: Adaptive Crypto Signals for {strategy_type} on {symbol}")
    def _load_llm_config(self) -> dict:
        """Load LLM configuration from main config.json"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                return config.get('llm', {})
        except Exception as e:
            self.logger.warning(f"Could not load LLM config: {e}")
            return {'enabled': False}
    def get_technical_direction(self, exchange) -> str:
        """FIXED: Multi-timeframe signal generation for ROC strategy"""
        try:
            current_time = time.time()
            
            # Prevent signal spam
            if current_time - self.last_signal_time < self.signal_cooldown:
                return 'none'
            
            # FIXED: Handle multi-timeframe data properly for ROC strategy
            if self.strategy_type == 'roc_multi_timeframe':
                # Get both 3m and 15m data for ROC strategy
                ohlcv_3m = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=100)
                ohlcv_15m = exchange.get_ohlcv(self.symbol, timeframe='15m', limit=50)
                
                if not ohlcv_3m or len(ohlcv_3m) < 50 or not ohlcv_15m or len(ohlcv_15m) < 20:
                    return 'none'
                
                # Structure multi-timeframe data
                ohlcv_data = {
                    '3m': ohlcv_3m,
                    '15m': ohlcv_15m
                }
            else:
                # For other strategies, use single timeframe
                ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=200)
                if not ohlcv_data or len(ohlcv_data) < 50:
                    return 'none'
            
            # Use thread-safe parameter snapshot
            with self._snapshot_lock:
                current_params = copy.deepcopy(self._params_snapshot)
            
            # Generate signal with proper multi-timeframe handling
            signal, indicators = self._generate_signal_with_indicators(ohlcv_data, current_params)
            
            # Store indicators for exit evaluation
            if indicators:
                with self._indicator_lock:
                    self._last_calculated_indicators = indicators.copy()
            
            # FIXED: Track signal with proper price extraction
            if signal != 'none':
                # Extract current price based on data structure
                if isinstance(ohlcv_data, dict) and '3m' in ohlcv_data:
                    current_price = float(ohlcv_data['3m'][-1][4])
                else:
                    current_price = float(ohlcv_data[-1][4])
                
                self._track_signal_fast(signal, current_price)
                self.last_signal = signal
                self.last_signal_time = current_time
                self.logger.info(f"OPTIMIZED SIGNAL: {signal.upper()} @ ${current_price:.6f}")
            
            # Update cache for current regime
            self._update_cache_if_needed(ohlcv_data)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in get_technical_direction: {e}")
            return 'none'

    def _should_optimize_before_signal(self, ohlcv_data) -> bool:
        """FIXED: Smart pre-order optimization triggers with proper data handling"""
        try:
            current_time = time.time()
            
            if self.optimization_in_progress:
                return False
            
            if current_time - self.last_optimization < 180:
                return False
            
            if len(self.signal_performance) < self.min_signals_for_optimization:
                return False
            
            # Extract data length based on structure
            if isinstance(ohlcv_data, dict) and '3m' in ohlcv_data:
                data_length = len(ohlcv_data['3m'])
            else:
                data_length = len(ohlcv_data) if ohlcv_data else 0
            
            # TRIGGER 1: Low accuracy
            if hasattr(self.params, 'accuracy') and self.params.accuracy < 50.0:
                return True
            
            # TRIGGER 2: Sufficient data available
            if data_length >= 100:
                return True
            
            return False
            
        except Exception:
            return False

    def _execute_pre_order_optimization(self, ohlcv_data: list) -> bool:
        """FIXED: SYNCHRONOUS pre-order optimization"""
        try:
            with self.optimization_lock:
                if self.optimization_in_progress:
                    # self.logger.info(f"PRE-ORDER OPTIMIZATION: Already in progress, using existing parameters")
                    return False
                self.optimization_in_progress = True
            
            try:
                start_time = time.time()
                # self.logger.info(f"PRE-ORDER OPTIMIZATION: Starting synchronous optimization for {self.symbol}")
                
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
                        # self.logger.info(f"PRE-ORDER OPTIMIZATION: Parameters updated in {optimization_time:.2f}s")
                        # self.logger.info(f"PRE-ORDER OPTIMIZATION: Accuracy {old_accuracy:.1f}% -> {best_score:.1f}%")
                        
                        self.last_optimization = time.time()
                        return True
                    else:
                        optimization_time = time.time() - start_time
                        # self.logger.info(f"PRE-ORDER OPTIMIZATION: No improvement found in {optimization_time:.2f}s")
                        self.last_optimization = time.time()
                        return False
                else:
                    # self.logger.warning(f"PRE-ORDER OPTIMIZATION: Insufficient data ({len(current_data)} candles)")
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
    
    def _generate_signal_with_indicators(self, ohlcv_data, params) -> Tuple[str, Dict]:
        """ENHANCED: Generate signal with Chain-of-Thought LLM analysis - Let LLM do the analysis"""
        try:
            # STEP 1: Get traditional signal using existing logic (UNCHANGED)
            if isinstance(ohlcv_data, dict) and '3m' in ohlcv_data and '15m' in ohlcv_data:
                if self.strategy_type == 'roc_multi_timeframe':
                    traditional_signal, indicators = self._roc_multi_timeframe_signal_with_indicators(ohlcv_data, params)
                else:
                    df = pd.DataFrame(ohlcv_data['3m'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    traditional_signal, indicators = self._get_strategy_signal(df, params)
            else:
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                traditional_signal, indicators = self._get_strategy_signal(df, params)
            
            # STEP 2: LLM Chain-of-Thought Analysis with RAW DATA
            if self._llm_enabled and traditional_signal in ['buy', 'sell']:
                try:
                    # Get raw OHLCV data for LLM analysis
                    if isinstance(ohlcv_data, dict) and '3m' in ohlcv_data:
                        raw_data = ohlcv_data['3m']
                    else:
                        raw_data = ohlcv_data
                    
                    # Use last 50 candles for analysis (sufficient for patterns, support/resistance)
                    analysis_candles = raw_data[-50:] if len(raw_data) >= 50 else raw_data
                    
                    if len(analysis_candles) < 20:
                        return traditional_signal, indicators
                    
                    # Format raw OHLCV data for LLM
                    ohlcv_text = self._format_ohlcv_for_llm(analysis_candles)
                    current_price = float(analysis_candles[-1][4])
                    
                    # Simple, direct prompt - let LLM do ALL the analysis
                    cot_prompt = f"""You are a crypto trading expert analyzing small cap tokens for manipulation and genuine signals.

TRADING SCENARIO:
Token: {self.symbol}
Current Price: ${current_price:.6f}
Traditional Signal: {traditional_signal.upper()}
RSI: {indicators.get('rsi', 'N/A')}
Market Trend: {"BULLISH" if indicators.get('st_direction', 0) == 1 else "BEARISH"}

RECENT OHLCV DATA (last 60 3m candles):
{self._format_ohlcv_for_llm(analysis_candles[-60:])}

ANALYSIS TASK:
Determine if the {traditional_signal.upper()} signal should be confirmed, changed, or rejected based on volume patterns, price action, and small cap manipulation risks.

Please structure your response using this format:

<think>
Step 1 - Volume Analysis:
- Examine volume trends in recent candles
- Check for volume spikes that indicate manipulation vs genuine interest
- Volume confirmation: Does volume support the price movement?

Step 2 - Price Action Assessment:
- Identify support/resistance levels from the OHLCV data
- Look for pump/dump patterns typical in small caps
- Assess momentum: Is this sustainable or likely reversal?

Step 3 - Small Cap Risk Evaluation:
- Check for manipulation signs: sudden spikes, thin volume, erratic moves
- Evaluate if this is genuine market movement or whale activity
- Risk assessment: Low/Medium/High manipulation probability

Step 4 - Signal Confidence Calculation:
Rate each factor (1-10):
- Volume confirmation strength: _/10
- Price pattern reliability: _/10
- Technical signal quality: _/10
- Manipulation risk (subtract): _/10

Final confidence = (Volume + Pattern + Technical - Manipulation) / 30

Step 5 - Decision Logic:
Based on analysis:
- Should I CONFIRM the {traditional_signal} signal?
- Should I CHANGE to opposite signal?
- Should I WAIT (use 'none') due to uncertainty?
</think>

Based on my systematic analysis, here is my trading recommendation:

Signal: [buy/sell/none]
Confidence: [0.XX calculated above]
Reasoning: [Brief summary of key factors from analysis]

Response format: {{"signal": "buy/sell/none", "confidence": 0.XX, "reasoning": "Volume shows [X], price action indicates [Y], manipulation risk is [Z], therefore [decision]"}}"""
                    start_time = time.time()
                    
                    # Call LLM with Phi4 thinking mode
                    response = ollama.chat(
                        model=self._llm_config.get('model', 'phi4-mini-reasoning:3.8b'),
                        messages=[{'role': 'user', 'content': cot_prompt}],
                        format='json',
                        options={
                            'temperature': 0.6,
                            'top_p': 0.95, 
                            'top_k': 40,
                            'num_predict': 512
                        }
                    )
                    
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Parse LLM response
                    try:
                        import json
                        result = json.loads(response['message']['content'])
                        llm_signal = result.get('signal', 'none').lower()
                        llm_confidence = float(result.get('confidence', 0.0))
                        llm_reasoning = result.get('reasoning', '')
                        
                        if llm_signal in ['buy', 'sell', 'none'] and 0.0 <= llm_confidence <= 1.0:
                            if llm_signal != traditional_signal:
                                
                                
                                indicators.update({
                                    'llm_enhanced': True,
                                    'llm_signal': llm_signal,
                                    'llm_confidence': llm_confidence,
                                    'llm_reasoning': llm_reasoning,
                                    'original_signal': traditional_signal
                                })
                                self.logger.info(f"LLM Analysis: {traditional_signal} -> {llm_signal} "
                                            f"(confidence: {llm_confidence:.2f}, {inference_time:.0f}ms)")
                                self.logger.info(f"LLM Reasoning: {llm_reasoning}")
                                return llm_signal, indicators
                            else:
                                self.logger.info(f"LLM confirmed {traditional_signal} "
                                            f"(confidence: {llm_confidence:.2f}, {inference_time:.0f}ms)")
                                indicators['llm_confidence'] = llm_confidence
                                indicators['llm_reasoning'] = llm_reasoning
                    
                    except json.JSONDecodeError:
                        self.logger.warning("LLM JSON parse failed")
                        
                except Exception as e:
                    self.logger.debug(f"LLM analysis failed: {e}")
            
            # STEP 3: Return traditional signal
            indicators['llm_enhanced'] = False
            return traditional_signal, indicators
            
        except Exception as e:
            self.logger.error(f"Error in _generate_signal_with_indicators: {e}")
            return 'none', {}

    def _format_ohlcv_for_llm(self, candles) -> str:
        """Format OHLCV data for LLM analysis - simple and clean"""
        try:
            # Show last 60 candles to LLM for pattern recognition
            recent_candles = candles[-60:] if len(candles) >= 60 else candles

            formatted_lines = []
            for i, candle in enumerate(recent_candles):
                timestamp, open_p, high, low, close, volume = candle
                formatted_lines.append(
                    f"{i+1:2d}: [{float(open_p):8.6f}, {float(high):8.6f}, {float(low):8.6f}, {float(close):8.6f}, {int(float(volume)):8d}]"
                )
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            return f"Error formatting OHLCV: {e}"
    # Add this new method to collect all traditional signals
    def _get_all_traditional_signals(self, ohlcv_data, params) -> Dict:
        """Collect signals from all available strategies and indicators"""
        
        # Handle multi-timeframe data for ROC strategy
        if isinstance(ohlcv_data, dict) and '3m' in ohlcv_data and '15m' in ohlcv_data:
            if self.strategy_type == 'roc_multi_timeframe':
                primary_signal, primary_indicators = self._roc_multi_timeframe_signal_with_indicators(ohlcv_data, params)
            else:
                df = pd.DataFrame(ohlcv_data['3m'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                primary_signal, primary_indicators = self._get_strategy_signal(df, params)
        else:
            # Single timeframe data
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            primary_signal, primary_indicators = self._get_strategy_signal(df, params)
        
        # Collect all strategy signals
        all_signals = primary_indicators.copy()
        
        # Add primary strategy signal
        all_signals[f'{self.strategy_type}_signal'] = primary_signal
        
        # Get additional strategy signals for fusion
        try:
            if self.strategy_type != 'qqe_supertrend_fixed':
                qqe_signal, qqe_indicators = self._qqe_supertrend_signal_with_indicators(df, params)
                all_signals['qqe_supertrend_signal'] = qqe_signal
                all_signals.update(qqe_indicators)
            
            if self.strategy_type != 'rsi_macd':
                rsi_macd_signal, rsi_macd_indicators = self._rsi_macd_signal_with_indicators(df, params)
                all_signals['rsi_macd_signal'] = rsi_macd_signal
                all_signals.update(rsi_macd_indicators)
            
            if self.strategy_type != 'tsi_vwap':
                tsi_vwap_signal, tsi_vwap_indicators = self._tsi_vwap_signal_with_indicators(df, params)
                all_signals['tsi_vwap_signal'] = tsi_vwap_signal
                all_signals.update(tsi_vwap_indicators)
                
        except Exception as e:
            self.logger.debug(f"Some additional signals unavailable: {e}")
        
        return all_signals

    def get_signal_performance_summary(self) -> Dict:
        """Get performance summary for the signal system"""
        try:
            # Calculate performance metrics from signal history
            total_signals = len(self.signal_performance)
            
            if total_signals == 0:
                return {
                    'total_signals': 0,
                    'accuracy': 0.0,
                    'winning_signals': 0,
                    'losing_signals': 0,
                    'average_return': 0.0,
                    'last_signal': 'none',
                    'last_signal_time': 0,
                    'strategy_type': self.strategy_type,
                    'symbol': self.symbol
                }
            
            # Count evaluated signals
            evaluated_signals = [s for s in self.signal_performance if s.get('evaluated', False)]
            winning_signals = sum(1 for s in evaluated_signals if s.get('correct', False))
            losing_signals = len(evaluated_signals) - winning_signals
            
            # Calculate accuracy
            accuracy = (winning_signals / len(evaluated_signals) * 100) if evaluated_signals else 0.0
            
            # Calculate average return if available
            returns = [s.get('price_change_pct', 0) for s in evaluated_signals if s.get('price_change_pct') is not None]
            average_return = sum(returns) / len(returns) if returns else 0.0
            
            return {
                'total_signals': total_signals,
                'evaluated_signals': len(evaluated_signals),
                'accuracy': round(accuracy, 2),
                'winning_signals': winning_signals,
                'losing_signals': losing_signals,
                'average_return': round(average_return, 4),
                'last_signal': self.last_signal,
                'last_signal_time': self.last_signal_time,
                'strategy_type': self.strategy_type,
                'symbol': self.symbol,
                'current_params': {
                    'qqe_length': getattr(self.params, 'qqe_length', 12),
                    'supertrend_period': getattr(self.params, 'supertrend_period', 10),
                    'rsi_length': getattr(self.params, 'rsi_length', 14),
                    'strategy_type': self.strategy_type
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_signal_performance_summary: {e}")
            return {
                'total_signals': 0,
                'accuracy': 0.0,
                'winning_signals': 0,
                'losing_signals': 0,
                'average_return': 0.0,
                'last_signal': 'none',
                'last_signal_time': 0,
                'strategy_type': self.strategy_type,
                'symbol': self.symbol,
                'error': str(e)
            }
    # Add this new method to get strategy signal based on type
    def _get_strategy_signal(self, df: pd.DataFrame, params) -> Tuple[str, Dict]:
        """Get signal based on strategy type"""
        if self.strategy_type == 'qqe_supertrend_fixed':
            return self._qqe_supertrend_signal_with_indicators(df, params)
        elif self.strategy_type == 'qqe_supertrend_fast':
            return self._qqe_supertrend_fast_signal_with_indicators(df, params)
        elif self.strategy_type == 'rsi_macd':
            return self._rsi_macd_signal_with_indicators(df, params)
        elif self.strategy_type == 'tsi_vwap':
            return self._tsi_vwap_signal_with_indicators(df, params)
        else:
            self.logger.warning(f"Unknown strategy type: {self.strategy_type}")
            return 'none', {}


    # Add this new method to prepare market data for LLM
    def _prepare_market_data(self, ohlcv_data) -> Dict:
        """Prepare market data for LLM analysis"""
        
        try:
            # Handle multi-timeframe data
            if isinstance(ohlcv_data, dict) and '3m' in ohlcv_data:
                df_3m = pd.DataFrame(ohlcv_data['3m'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_3m['close'] = df_3m['close'].astype(float)
                df_3m['volume'] = df_3m['volume'].astype(float)
                
                current_price = float(df_3m.iloc[-1]['close'])
                
                # Calculate 24h change (assuming 3m candles, 480 candles = 24h)
                if len(df_3m) >= 480:
                    price_24h_ago = float(df_3m.iloc[-480]['close'])
                    price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
                else:
                    price_change_24h = 0.0
                
                # Volume trend analysis
                recent_volume = df_3m['volume'].tail(10).mean()
                historical_volume = df_3m['volume'].iloc[-50:-10].mean() if len(df_3m) >= 50 else recent_volume
                
                if recent_volume > historical_volume * 1.2:
                    volume_trend = 'increasing'
                elif recent_volume < historical_volume * 0.8:
                    volume_trend = 'decreasing'
                else:
                    volume_trend = 'neutral'
                    
            else:
                # Single timeframe data
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                current_price = float(df.iloc[-1]['close'])
                
                if len(df) >= 480:
                    price_24h_ago = float(df.iloc[-480]['close'])
                    price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
                else:
                    price_change_24h = 0.0
                
                recent_volume = df['volume'].tail(10).mean()
                historical_volume = df['volume'].iloc[-50:-10].mean() if len(df) >= 50 else recent_volume
                
                if recent_volume > historical_volume * 1.2:
                    volume_trend = 'increasing'
                elif recent_volume < historical_volume * 0.8:
                    volume_trend = 'decreasing'
                else:
                    volume_trend = 'neutral'
            
            return {
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'volume_trend': volume_trend,
                'symbol': self.symbol,
                'strategy_type': self.strategy_type
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing market data: {e}")
            return {
                'current_price': 0.0,
                'price_change_24h': 0.0,
                'volume_trend': 'neutral',
                'symbol': self.symbol,
                'strategy_type': self.strategy_type
            }


    # Add this method to get primary strategy signal (fallback)
    def _get_primary_strategy_signal(self, ohlcv_data, params) -> Tuple[str, Dict]:
        """Get signal from primary strategy (original behavior)"""
        
        # FIXED: Handle multi-timeframe data for ROC strategy
        if isinstance(ohlcv_data, dict) and '3m' in ohlcv_data and '15m' in ohlcv_data:
            # For ROC multi-timeframe, pass the dict directly
            if self.strategy_type == 'roc_multi_timeframe':
                return self._roc_multi_timeframe_signal_with_indicators(ohlcv_data, params)
            else:
                # For other strategies, use 3m data
                df = pd.DataFrame(ohlcv_data['3m'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        else:
            # Single timeframe data
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert data types for single timeframe strategies
        if not self.strategy_type == 'roc_multi_timeframe':
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Route to appropriate strategy method
            if self.strategy_type == 'qqe_supertrend_fixed':
                return self._qqe_supertrend_signal_with_indicators(df, params)
            elif self.strategy_type == 'qqe_supertrend_fast':
                return self._qqe_supertrend_fast_signal_with_indicators(df, params)
            elif self.strategy_type == 'rsi_macd':
                return self._rsi_macd_signal_with_indicators(df, params)
            elif self.strategy_type == 'tsi_vwap':
                return self._tsi_vwap_signal_with_indicators(df, params)
            else:
                self.logger.warning(f"Unknown strategy type: {self.strategy_type}")
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
        """FIXED: Multi-timeframe ROC with DIVERGENCE logic for reversal detection after spikes + Candle Color Validation"""
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
            min_3m_required = max(params.roc_3m_length, 5)
            min_15m_required = max(params.roc_15m_length, 5)
            
            if len(df_3m) < min_3m_required or len(df_15m) < min_15m_required:
                return 'none', {}
            
            # Calculate ROC for both timeframes with error handling
            roc_3m = ta.roc(df_3m['close'], length=params.roc_3m_length)
            roc_15m = ta.roc(df_15m['close'], length=params.roc_15m_length)
            
            if (roc_3m is None or len(roc_3m.dropna()) < 5 or 
                roc_15m is None or len(roc_15m.dropna()) < 3):
                return 'none', {}
            
            # Get current values with NaN checks
            current_roc_3m = roc_3m.iloc[-1]
            current_roc_15m = roc_15m.iloc[-1]
            
            if pd.isna(current_roc_3m) or pd.isna(current_roc_15m):
                return 'none', {}
            
            # CANDLE COLOR CHECK: Get current 3m candle open/close prices
            current_open = df_3m['open'].iloc[-1]
            current_close = df_3m['close'].iloc[-1]
            
            # Validate candle data
            if pd.isna(current_open) or pd.isna(current_close) or current_open <= 0 or current_close <= 0:
                return 'none', {}
                
            # Determine candle color
            is_green_candle = current_close > current_open  # Green = bullish
            is_red_candle = current_close < current_open    # Red = bearish
            
            # Need at least 6 points for 5-candle lookback
            lookback_period = min(6, len(roc_3m) - 1)
            if lookback_period < 1:
                return 'none', {}
            
            # Store indicators for exit evaluation (including candle color info)
            indicators = {
                'roc_3m': float(current_roc_3m),
                'roc_15m': float(current_roc_15m),
                'roc_3m_threshold': float(params.roc_3m_threshold),
                'roc_15m_threshold': float(params.roc_15m_threshold),
                'current_price': float(df_3m['close'].iloc[-1]),
                'current_open': float(current_open),
                'current_close': float(current_close),
                'is_green_candle': is_green_candle,
                'is_red_candle': is_red_candle
            }
            
            # FIXED: DIVERGENCE LOGIC with 5-candle lookback like QQE pattern
            
            # Current states
            roc_3m_currently_bullish = current_roc_3m > params.roc_3m_threshold
            roc_3m_currently_bearish = current_roc_3m < -params.roc_3m_threshold
            roc_15m_currently_bullish = current_roc_15m > params.roc_15m_threshold
            roc_15m_currently_bearish = current_roc_15m < -params.roc_15m_threshold
            
            # Check if 3m RECENTLY turned bullish (current bullish + any of last 5 was bearish)
            roc_3m_recently_turned_bullish = False
            if roc_3m_currently_bullish:
                for i in range(1, lookback_period + 1):
                    roc_3m_past = roc_3m.iloc[-1-i]
                    if not pd.isna(roc_3m_past) and roc_3m_past < -params.roc_3m_threshold:
                        roc_3m_recently_turned_bullish = True
                        break
            
            # Check if 3m RECENTLY turned bearish (current bearish + any of last 5 was bullish)
            roc_3m_recently_turned_bearish = False
            if roc_3m_currently_bearish:
                for i in range(1, lookback_period + 1):
                    roc_3m_past = roc_3m.iloc[-1-i]
                    if not pd.isna(roc_3m_past) and roc_3m_past > params.roc_3m_threshold:
                        roc_3m_recently_turned_bearish = True
                        break
            
            indicators['roc_3m_recently_turned_bullish'] = roc_3m_recently_turned_bullish
            indicators['roc_3m_recently_turned_bearish'] = roc_3m_recently_turned_bearish
            indicators['roc_15m_currently_bullish'] = roc_15m_currently_bullish
            indicators['roc_15m_currently_bearish'] = roc_15m_currently_bearish
            
            # DIVERGENCE SIGNALS FOR REVERSAL DETECTION (with 5-candle lookback):
            # BUY: 3m recently turned bearish while 15m is bullish (catch bottom) + GREEN CANDLE
            divergence_buy_signal = (roc_3m_currently_bullish and is_green_candle)
            
            # SELL: 3m recently turned bullish while 15m is bearish (catch top) + RED CANDLE  
            divergence_sell_signal = (roc_3m_currently_bearish and is_red_candle)
            
            indicators['divergence_buy_signal'] = divergence_buy_signal
            indicators['divergence_sell_signal'] = divergence_sell_signal
            
            # Return divergence signals with candle color validation
            if divergence_buy_signal:
                self.logger.debug(f"ROC DIVERGENCE BUY: 3m recently turned bearish ({current_roc_3m:.2f}) while 15m bullish ({current_roc_15m:.2f}) + GREEN candle (O:{current_open:.6f} C:{current_close:.6f})")
                return 'buy', indicators
            elif divergence_sell_signal:
                self.logger.debug(f"ROC DIVERGENCE SELL: 3m recently turned bullish ({current_roc_3m:.2f}) while 15m bearish ({current_roc_15m:.2f}) + RED candle (O:{current_open:.6f} C:{current_close:.6f})")
                return 'sell', indicators
            
            # Log when signals are blocked by candle color
            if roc_3m_recently_turned_bearish and roc_15m_currently_bullish and not is_green_candle:
                self.logger.debug(f"ROC BUY signal blocked: RED candle (O:{current_open:.6f} C:{current_close:.6f}) - need GREEN candle")
            elif roc_3m_recently_turned_bullish and roc_15m_currently_bearish and not is_red_candle:
                self.logger.debug(f"ROC SELL signal blocked: GREEN candle (O:{current_open:.6f} C:{current_close:.6f}) - need RED candle")
            
            return 'none', indicators
            
        except Exception as e:
            self.logger.error(f"ROC Multi-timeframe calculation error: {e}")
            return 'none', {}

    def evaluate_exit_conditions(self, position_side: str, entry_price: float, current_price: float) -> Dict:
        """UPDATED: Exit evaluation with fresh indicator support - signature unchanged."""
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
            # if pnl_pct > 2.0:
            #     result.update({
            #         'should_exit': True,
            #         'exit_reason': f"Take Profit: {pnl_pct:.2f}% gain",
            #         'exit_urgency': 'normal'
            #     })
            #     return result

            # PRIORITY 3: Technical signal-based exits (now with fresh indicators)
            try:
                # NOTE: _check_technical_exit_conditions now calculates fresh indicators internally
                # Pass empty dict - fresh indicators will be calculated inside the method
                signal_exit = self._check_technical_exit_conditions(position_side, {}, pnl_pct)
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

    def _get_fresh_exit_indicators(self) -> Dict:
        """Get fresh indicators specifically for exit evaluation - bypasses entry signal cooldown."""
        try:
            # Access exchange through the integration pattern
            if not hasattr(self, '_exchange_ref'):
                self.logger.debug("No exchange reference available for fresh exit indicators")
                return {}
            
            exchange = self._exchange_ref
            
            # Get fresh OHLCV data based on strategy type
            if self.strategy_type == 'roc_multi_timeframe':
                # Multi-timeframe data for ROC strategy
                ohlcv_3m = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=100)
                ohlcv_15m = exchange.get_ohlcv(self.symbol, timeframe='15m', limit=50)
                
                if not ohlcv_3m or len(ohlcv_3m) < 50 or not ohlcv_15m or len(ohlcv_15m) < 20:
                    return {}
                
                ohlcv_data = {
                    '3m': ohlcv_3m,
                    '15m': ohlcv_15m
                }
            else:
                # Single timeframe for other strategies
                ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=200)
                if not ohlcv_data or len(ohlcv_data) < 50:
                    return {}
            
            # Use thread-safe parameter snapshot
            with self._snapshot_lock:
                current_params = copy.deepcopy(self._params_snapshot)
            
            # Generate fresh indicators using existing method (bypasses cooldown)
            signal, fresh_indicators = self._generate_signal_with_indicators(ohlcv_data, current_params)
            
            if fresh_indicators:
                self.logger.debug(f"Fresh exit indicators calculated for {self.strategy_type}")
                return fresh_indicators
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error calculating fresh exit indicators: {e}")
            return {}

    def _check_technical_exit_conditions(self, position_side: str, indicators: Dict, pnl_pct: float) -> Dict:
        """FIXED: Technical exit conditions with fresh indicators for all strategy types."""
        try:
            result = {'should_exit': False, 'exit_reason': '', 'exit_urgency': 'none'}
            
            # CRITICAL FIX: Get fresh indicators for exit evaluation
            fresh_indicators = self._get_fresh_exit_indicators()
            if fresh_indicators:
                indicators = fresh_indicators  # Use fresh instead of stale
                self.logger.debug(f"Using fresh indicators for {self.strategy_type} exit evaluation")
            else:
                self.logger.debug(f"Using fallback indicators for {self.strategy_type} exit evaluation")
            
            # Strategy-specific exit logic with fresh indicators
            if self.strategy_type == 'roc_multi_timeframe':
                # ROC Multi-timeframe exits with fresh indicators
                roc_3m = indicators.get('roc_3m', 0)
                roc_15m = indicators.get('roc_15m', 0)
                threshold_3m = indicators.get('roc_3m_threshold', 1.0)
                threshold_15m = indicators.get('roc_15m_threshold', 2.0)
                
                # Check for divergence exit signals using fresh "recently turned" flags
                roc_3m_recently_turned_bearish = indicators.get('roc_3m_recently_turned_bearish', False)
                roc_3m_recently_turned_bullish = indicators.get('roc_3m_recently_turned_bullish', False)
                
                if position_side == 'long':
                    # Exit long when 3m recently turned bearish (fast exit on reversal)
                    # OR when both timeframes become bearish (strong reversal)
                    roc_3m_bearish = roc_3m < -threshold_3m
                    roc_15m_bearish = roc_15m < -threshold_15m
                    both_bearish = roc_3m_bearish and roc_15m_bearish
                    
                    if roc_3m_bearish or both_bearish:
                        exit_reason = "3m recently turned bearish" if roc_3m_recently_turned_bearish else "both timeframes bearish"
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"ROC divergence exit: {exit_reason} (3m:{roc_3m:.2f}, 15m:{roc_15m:.2f}, PnL:{pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                        
                elif position_side == 'short':
                    # Exit short when 3m recently turned bullish (fast exit on reversal)
                    # OR when both timeframes become bullish (strong reversal)
                    roc_3m_bullish = roc_3m > threshold_3m
                    roc_15m_bullish = roc_15m > threshold_15m
                    both_bullish = roc_3m_bullish and roc_15m_bullish
                    
                    if roc_3m_bullish or both_bullish:
                        exit_reason = "3m recently turned bullish" if roc_3m_recently_turned_bullish else "both timeframes bullish"
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"ROC divergence exit: {exit_reason} (3m:{roc_3m:.2f}, 15m:{roc_15m:.2f}, PnL:{pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
            
            elif self.strategy_type == 'tsi_vwap':
                # TSI + VWAP exits with fresh indicators
                tsi_line = indicators.get('tsi_line', 0)
                tsi_signal = indicators.get('tsi_signal', 0)
                current_price = indicators.get('current_price', 0)
                vwap = indicators.get('vwap', 0)
                
                if position_side == 'long':
                    if tsi_line < tsi_signal or current_price < vwap:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"TSI reversal or price below VWAP (TSI:{tsi_line:.2f}<{tsi_signal:.2f}, Price:${current_price:.6f}<VWAP:${vwap:.6f}, PnL:{pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                elif position_side == 'short':
                    if tsi_line > tsi_signal or current_price > vwap:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"TSI reversal or price above VWAP (TSI:{tsi_line:.2f}>{tsi_signal:.2f}, Price:${current_price:.6f}>VWAP:${vwap:.6f}, PnL:{pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
            
            elif self.strategy_type == 'rsi_macd':
                # RSI + MACD exits with fresh indicators
                rsi = indicators.get('rsi', 50)
                macd_line = indicators.get('macd_line', 0)
                macd_signal = indicators.get('macd_signal', 0)
                
                if position_side == 'long':
                    # Exit long if RSI overbought or MACD turns bearish
                    if rsi > 70 or macd_line < macd_signal:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"RSI overbought or MACD bearish (RSI:{rsi:.1f}, MACD:{macd_line:.4f}<{macd_signal:.4f}, PnL:{pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                elif position_side == 'short':
                    # Exit short if RSI oversold or MACD turns bullish
                    if rsi < 30 or macd_line > macd_signal:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"RSI oversold or MACD bullish (RSI:{rsi:.1f}, MACD:{macd_line:.4f}>{macd_signal:.4f}, PnL:{pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
            
            elif self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                # QQE + Supertrend exits with fresh indicators
                qqe_value = indicators.get('qqe_value', 0)
                qqe_signal = indicators.get('qqe_signal', 0)
                st_direction = indicators.get('st_direction', 0)
                
                if position_side == 'long':
                    # Exit long if QQE turns bearish or Supertrend turns negative
                    if qqe_value <= qqe_signal or st_direction == -1:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"QQE bearish or ST negative (QQE:{qqe_value:.2f}<={qqe_signal:.2f}, ST:{st_direction}, PnL:{pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                elif position_side == 'short':
                    # Exit short if QQE turns bullish or Supertrend turns positive
                    if qqe_value > qqe_signal or st_direction == 1:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"QQE bullish or ST positive (QQE:{qqe_value:.2f}>{qqe_signal:.2f}, ST:{st_direction}, PnL:{pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Technical exit check error: {e}")
            return {'should_exit': False, 'exit_reason': 'Technical exit error', 'exit_urgency': 'none'}

    def _update_cache_if_needed(self, ohlcv_data):
        """FIXED: Current regime cache update with proper multi-timeframe handling"""
        try:
            current_time = time.time()
            if current_time - self._cache_timestamp > self._cache_validity:
                # Handle different data structures
                if isinstance(ohlcv_data, dict) and '3m' in ohlcv_data:
                    cache_data = ohlcv_data['3m']  # Use 3m data for cache
                else:
                    cache_data = ohlcv_data
                
                if len(cache_data) > self._max_cache_size:
                    self._historical_data_cache = cache_data[-self._max_cache_size:]
                else:
                    self._historical_data_cache = cache_data.copy()
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
        """FIXED: Fast signal tracking for current regime with proper error handling"""
        try:
            current_time = time.time()
            
            # Create signal data with all required fields
            signal_data = {
                'signal': signal,
                'price': float(price),  # Ensure price is float
                'timestamp': current_time,
                'evaluated': False,
                'correct': None
            }
            
            # Add to performance tracking
            self.signal_performance.append(signal_data)
            
            # Maintain reasonable signal history size (keep last 50 signals)
            if len(self.signal_performance) > 50:
                self.signal_performance = deque(list(self.signal_performance)[-30:], maxlen=50)
            
            # Evaluate older signals periodically (every 3 signals)
            if len(self.signal_performance) % 3 == 0:
                self._evaluate_signals_quick()
                
            # Update parameters accuracy after evaluation
            self._update_params_accuracy()
            
        except Exception as e:
            self.logger.error(f"Error in _track_signal_fast: {e}")
    def _update_params_accuracy(self):
        """Update parameter accuracy based on evaluated signals"""
        try:
            evaluated = [s for s in self.signal_performance if s.get('evaluated', False)]
            if len(evaluated) > 0:
                correct = sum(1 for s in evaluated if s.get('correct', False))
                self.params.accuracy = (correct / len(evaluated)) * 100
                self.params.total_signals = len(evaluated)
                self.params.winning_signals = correct
        except Exception:
            pass
    def _evaluate_signals_quick(self):
        """FIXED: Quick current regime signal evaluation with proper error handling"""
        try:
            current_time = time.time()
            evaluation_delay = 120  # 2 minutes for faster crypto evaluation
            
            # Get recent signals for evaluation
            recent_signals = list(self.signal_performance)[-8:]
            if len(recent_signals) < 2:
                return
            
            # Evaluate signals that have had enough time to mature
            for signal_data in recent_signals[:-1]:  # Don't evaluate the most recent signal
                if signal_data.get('evaluated', False):
                    continue
                
                if current_time - signal_data['timestamp'] < evaluation_delay:
                    continue
                
                # Use the most recent signal's price for comparison
                recent_price = recent_signals[-1]['price']
                signal_price = signal_data['price']
                
                if signal_price > 0:  # Avoid division by zero
                    price_change_pct = ((recent_price - signal_price) / signal_price) * 100
                    
                    # Evaluate signal correctness based on price movement
                    if signal_data['signal'] == 'buy':
                        signal_data['correct'] = price_change_pct > 0.2  # 0.2% threshold for buy
                    elif signal_data['signal'] == 'sell':
                        signal_data['correct'] = price_change_pct < -0.2  # -0.2% threshold for sell
                    else:
                        signal_data['correct'] = False
                    
                    signal_data['evaluated'] = True
                    signal_data['price_change_pct'] = price_change_pct
            
        except Exception as e:
            self.logger.error(f"Error in _evaluate_signals_quick: {e}")

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

def integrate_adaptive_crypto_signals(strategy_instance, strategy_type: str = 'roc_multi_timeframe', enable_llm: bool = True):
    """
    Integrate adaptive crypto signals into a strategy instance with LLM support
    FIXED: Ensure consistent naming for all code paths
    """
    
    # Create adaptive signals instance with LLM setting
    signals = AdaptiveCryptoSignals(
        symbol=strategy_instance.symbol, 
        strategy_type=strategy_type,
        enable_llm=enable_llm  # Pass LLM setting
    )
    
    # Set exchange reference for real-time data
    signals._exchange_ref = strategy_instance.exchange
    
    # Add to strategy instance with CONSISTENT naming
    strategy_instance.adaptive_signals = signals
    strategy_instance._crypto_signal_system = signals  # FIXED: Add this for consistency
    strategy_instance.get_technical_direction = signals.get_technical_direction
    strategy_instance.evaluate_exit_conditions = signals.evaluate_exit_conditions
    strategy_instance.get_signal_performance_summary = signals.get_signal_performance_summary
    
    # logging.getLogger(__name__).info(
    #     f"✅ Adaptive signals integrated: {strategy_type} {'with LLM' if enable_llm else 'traditional'}"
    # )