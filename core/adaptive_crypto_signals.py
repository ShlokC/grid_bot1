"""
Adaptive Crypto Signal System - Dynamic Parameter Optimization for Win Rate
ENTRY SIGNALS ONLY - Uses optimized parameters instead of fixed ones
"""

import logging
import time
import os
import csv
import datetime
import threading
from typing import List
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional
from collections import deque
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Install with: pip install optuna")
from dataclasses import dataclass
import json

@dataclass
class SignalParameters:
    """Parameters for trading signals with optimization tracking"""
    
    strategy_type: str = 'tsi_vwap'
    
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
    
class AdaptiveCryptoSignals:
    """
    Adaptive signal system with dynamic parameter optimization focused on win rate
    """
    
    def __init__(self, symbol: str, config_file: str = "data/crypto_signal_configs.json", strategy_type: str = 'tsi_vwap'):
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        self.symbol = symbol
        self.config_file = config_file
        self.strategy_type = strategy_type
        self.position_entry_time = 0
        
        # Load optimized parameters
        self.params = self._load_symbol_config()
        self.signal_performance = self._load_signal_history()
        
        # Signal control
        self.last_signal = 'none'
        self.last_signal_time = 0
        self.signal_cooldown = 10
        
        # Market state
        self.current_trend = 'neutral'
        self.volatility_level = 'normal'
        self.last_optimization = time.time()
        
        # CRITICAL: Threading for non-blocking optimization
        self.optimization_in_progress = False
        self.optimization_lock = threading.Lock()
        
        # Optimization settings - more aggressive for win rate focus
        self.min_signals_for_optimization = 5  # Reduced from 10
        self.optimization_interval = 300  # Reduced from 600 (5 minutes)
        self.win_rate_threshold = 40.0  # Target win rate
        
        # Cache for historical data
        self._historical_data_cache = None
        self._cache_timestamp = 0
        self._cache_validity = 300
        
        self.last_indicators: Dict[str, Optional[Dict]] = {
            'qqe': None, 'supertrend': None, 'rsi': None, 
            'macd': None, 'tsi': None, 'vwap': None
        }
        
        self.logger.info(f"Adaptive Crypto Signals with {strategy_type} for {symbol}")
        self.logger.info(f"Current win rate: {self.params.accuracy:.1f}%")

    def get_technical_direction(self, exchange) -> str:
        """Main signal generation using optimized parameters - NON-BLOCKING"""
        try:
            current_time = time.time()
            if current_time - self.last_signal_time < self.signal_cooldown:
                return 'none'
            
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='3m', limit=200)
            if not ohlcv_data or len(ohlcv_data) < 50:
                return 'none'
            
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df_indexed = df.set_index('timestamp')
            
            # Update cache for optimization (non-blocking)
            self._update_historical_cache(ohlcv_data)
            
            # Generate signal using optimized parameters (FAST)
            signal = self._generate_optimized_signal(df_indexed)
            
            if signal != 'none':
                self._track_signal(signal, float(df['close'].iloc[-1]))
                self.last_signal = signal
                self.last_signal_time = current_time
                self.logger.info(f"SIGNAL: {signal.upper()} @ ${float(df['close'].iloc[-1]):.6f}")
            
            # CRITICAL: Non-blocking optimization check
            if self._should_optimize_for_winrate():
                self._trigger_background_optimization()
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return 'none'

    def _generate_optimized_signal(self, df: pd.DataFrame) -> str:
        """Generate signal using optimized parameters instead of fixed ones"""
        try:
            # Calculate indicators using optimized parameters
            indicators = self._calculate_optimized_indicators(df)
            if not indicators:
                return 'none'
            
            # Generate signal based on strategy type
            signal = self._generate_strategy_signal(indicators, df)
            return signal
                
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return 'none'

    def _calculate_optimized_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate indicators using optimized parameters from self.params"""
        try:
            indicators = {}
            
            # Use optimized parameters instead of fixed config
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                # QQE with optimized parameters
                qqe_result = ta.qqe(df['close'], 
                                length=self.params.qqe_length,  # Use optimized
                                smooth=self.params.qqe_smooth,  # Use optimized
                                factor=self.params.qqe_factor)
                if qqe_result is not None and not qqe_result.empty and len(qqe_result.columns) >= 2:
                    indicators['qqe_value'] = float(qqe_result.iloc[-1, 0])
                    indicators['qqe_signal'] = float(qqe_result.iloc[-1, 1])
                
                # Supertrend with optimized parameters
                st_result = ta.supertrend(df['high'], df['low'], df['close'],
                                        length=self.params.supertrend_period,  # Use optimized
                                        multiplier=self.params.supertrend_multiplier)  # Use optimized
                if st_result is not None and not st_result.empty:
                    dir_col = next((col for col in st_result.columns if 'SUPERTd' in col), None)
                    if dir_col:
                        indicators['st_direction'] = int(st_result[dir_col].iloc[-1])
            
            elif self.strategy_type == 'rsi_macd':
                # RSI with optimized parameters
                rsi_result = ta.rsi(df['close'], length=self.params.rsi_length)  # Use optimized
                if rsi_result is not None:
                    indicators['rsi'] = float(rsi_result.iloc[-1])
                
                # MACD with optimized parameters
                macd_result = ta.macd(df['close'], 
                                    fast=self.params.macd_fast,  # Use optimized
                                    slow=self.params.macd_slow,  # Use optimized
                                    signal=self.params.macd_signal)  # Use optimized
                if macd_result is not None and not macd_result.empty and len(macd_result.columns) >= 3:
                    indicators['macd_line'] = float(macd_result.iloc[-1, 0])
                    indicators['macd_histogram'] = float(macd_result.iloc[-1, 1])
                    indicators['macd_signal'] = float(macd_result.iloc[-1, 2])
            
            elif self.strategy_type == 'tsi_vwap':
                # TSI with optimized parameters
                tsi_result = ta.tsi(df['close'], 
                                fast=self.params.tsi_fast,  # Use optimized
                                slow=self.params.tsi_slow,  # Use optimized
                                signal=self.params.tsi_signal)  # Use optimized
                if tsi_result is not None and not tsi_result.empty:
                    indicators['tsi_line'] = float(tsi_result.iloc[-1, 0])
                    if len(tsi_result.columns) > 1:
                        indicators['tsi_signal'] = float(tsi_result.iloc[-1, 1])
                
                # VWAP
                vwap_result = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                if vwap_result is not None:
                    indicators['vwap'] = float(vwap_result.iloc[-1])
                    indicators['price'] = float(df['close'].iloc[-1])
            
            # Store for exit evaluation
            self._last_calculated_indicators = indicators.copy()
            return indicators if indicators else None
            
        except Exception as e:
            self.logger.error(f"Error calculating optimized indicators: {e}")
            return None

    def _generate_strategy_signal(self, indicators: Dict, df: pd.DataFrame) -> str:
        """Generate signal based on strategy-specific logic"""
        try:
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                return self._qqe_supertrend_signal(indicators)
            elif self.strategy_type == 'rsi_macd':
                return self._rsi_macd_signal(indicators)
            elif self.strategy_type == 'tsi_vwap':
                return self._tsi_vwap_signal(indicators)
            else:
                return 'none'
                
        except Exception as e:
            self.logger.error(f"Error generating strategy signal: {e}")
            return 'none'

    def _qqe_supertrend_signal(self, indicators: Dict) -> str:
        """QQE + Supertrend signal logic"""
        if not all(key in indicators for key in ['qqe_value', 'qqe_signal', 'st_direction']):
            return 'none'
        
        qqe_bullish = indicators['qqe_signal'] > indicators['qqe_value']
        st_bullish = indicators['st_direction'] == 1
        
        if qqe_bullish and st_bullish:
            return 'buy'
        elif not qqe_bullish and not st_bullish:
            return 'sell'
        else:
            return 'none'

    def _rsi_macd_signal(self, indicators: Dict) -> str:
        """RSI + MACD signal logic"""
        if not all(key in indicators for key in ['rsi', 'macd_line', 'macd_signal']):
            return 'none'
        
        rsi = indicators['rsi']
        macd_bullish = indicators['macd_line'] > indicators['macd_signal']
        
        if rsi < 35 and macd_bullish:
            return 'buy'
        elif rsi > 65 and not macd_bullish:
            return 'sell'
        else:
            return 'none'

    def _tsi_vwap_signal(self, indicators: Dict) -> str:
        """TSI + VWAP signal logic"""
        if not all(key in indicators for key in ['tsi_line', 'vwap', 'price']):
            return 'none'
        
        tsi_signal = indicators.get('tsi_signal', 0)
        tsi_bullish = indicators['tsi_line'] > tsi_signal
        price_above_vwap = indicators['price'] > indicators['vwap']
        
        if tsi_bullish and price_above_vwap:
            return 'buy'
        elif not tsi_bullish and not price_above_vwap:
            return 'sell'
        else:
            return 'none'

    def _trigger_background_optimization(self):
        """Start optimization in background thread - NON-BLOCKING"""
        try:
            with self.optimization_lock:
                if self.optimization_in_progress:
                    self.logger.debug("Optimization already in progress, skipping")
                    return
                
                self.optimization_in_progress = True
            
            def background_optimization():
                try:
                    self.logger.info("Starting background parameter optimization")
                    start_time = time.time()
                    
                    if OPTUNA_AVAILABLE and self._historical_data_cache:
                        best_params, best_score = self._optimize_with_optuna()
                    else:
                        best_params, best_score = self._optimize_with_grid_search()
                    
                    # Update if improvement found
                    if best_params and best_score > self.params.accuracy + 2:
                        self._update_optimized_parameters(best_params, best_score)
                        optimization_time = time.time() - start_time
                        self.logger.info(f"Background optimization completed in {optimization_time:.1f}s: +{best_score - self.params.accuracy:.1f}% win rate")
                    else:
                        self.logger.debug("Background optimization found no significant improvement")
                    
                    self.last_optimization = time.time()
                    
                except Exception as e:
                    self.logger.error(f"Background optimization error: {e}")
                finally:
                    with self.optimization_lock:
                        self.optimization_in_progress = False
            
            # Start background thread
            optimization_thread = threading.Thread(target=background_optimization, daemon=True)
            optimization_thread.start()
            self.logger.debug("Background optimization thread started")
            
        except Exception as e:
            self.logger.error(f"Error starting background optimization: {e}")
            with self.optimization_lock:
                self.optimization_in_progress = False

    def _should_optimize_for_winrate(self) -> bool:
        """Check if optimization needed - with threading safety"""
        current_time = time.time()
        
        # Skip if optimization already running
        with self.optimization_lock:
            if self.optimization_in_progress:
                return False
        
        # Don't optimize too frequently
        if current_time - self.last_optimization < self.optimization_interval:
            return False
        
        # Need minimum signals
        if len(self.signal_performance) < self.min_signals_for_optimization:
            return False
        
        # Check win rate performance
        evaluated_signals = [s for s in self.signal_performance if s.get('evaluated', False)]
        if len(evaluated_signals) < 3:
            return False
        
        # Trigger optimization if:
        # 1. Win rate below threshold
        # 2. No optimization done yet
        # 3. Recent performance degradation
        
        if self.params.accuracy < self.win_rate_threshold:
            self.logger.debug(f"Optimization trigger: Low win rate ({self.params.accuracy:.1f}%)")
            return True
        
        if self.params.optimization_score == 0.0:
            self.logger.debug("Optimization trigger: Initial optimization")
            return True
        
        # Check recent performance
        recent_signals = list(self.signal_performance)[-5:]
        recent_evaluated = [s for s in recent_signals if s.get('evaluated', False)]
        if len(recent_evaluated) >= 3:
            recent_wins = sum(1 for s in recent_evaluated if s.get('correct', False))
            recent_accuracy = (recent_wins / len(recent_evaluated)) * 100
            if recent_accuracy < self.params.accuracy - 15:  # 15% degradation
                self.logger.debug(f"Optimization trigger: Performance degradation ({recent_accuracy:.1f}%)")
                return True
        
        return False

    def _optimize_with_optuna(self) -> tuple:
        """Optimize using Optuna with focus on win rate - BACKGROUND SAFE"""
        try:
            study = optuna.create_study(direction='maximize')
            
            # CRITICAL: Shorter timeout for background processing
            study.optimize(
                self._winrate_objective,
                n_trials=15,  # Reduced from 25
                timeout=15,   # Reduced from 30 seconds
                show_progress_bar=False
            )
            
            if study.best_trial:
                return study.best_params, study.best_value
            return None, 0.0
            
        except Exception as e:
            self.logger.debug(f"Background Optuna optimization failed: {e}")
            return None, 0.0

    def _winrate_objective(self, trial) -> float:
        """Optuna objective function focused on win rate"""
        try:
            # Define parameter ranges based on strategy type
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                qqe_length = trial.suggest_int('qqe_length', 8, 20)
                qqe_smooth = trial.suggest_int('qqe_smooth', 3, 8)
                supertrend_period = trial.suggest_int('supertrend_period', 6, 15)
                supertrend_multiplier = trial.suggest_float('supertrend_multiplier', 2.0, 3.5)
                
                test_params = SignalParameters(
                    strategy_type=self.strategy_type,
                    qqe_length=qqe_length,
                    qqe_smooth=qqe_smooth,
                    supertrend_period=supertrend_period,
                    supertrend_multiplier=supertrend_multiplier
                )
                
            elif self.strategy_type == 'rsi_macd':
                rsi_length = trial.suggest_int('rsi_length', 10, 20)
                macd_fast = trial.suggest_int('macd_fast', 8, 16)
                macd_slow = trial.suggest_int('macd_slow', 20, 35)
                macd_signal = trial.suggest_int('macd_signal', 6, 12)
                
                test_params = SignalParameters(
                    strategy_type=self.strategy_type,
                    rsi_length=rsi_length,
                    macd_fast=macd_fast,
                    macd_slow=macd_slow,
                    macd_signal=macd_signal
                )
                
            elif self.strategy_type == 'tsi_vwap':
                tsi_fast = trial.suggest_int('tsi_fast', 5, 12)
                tsi_slow = trial.suggest_int('tsi_slow', 12, 25)
                tsi_signal = trial.suggest_int('tsi_signal', 4, 8)
                
                test_params = SignalParameters(
                    strategy_type=self.strategy_type,
                    tsi_fast=tsi_fast,
                    tsi_slow=tsi_slow,
                    tsi_signal=tsi_signal
                )
            else:
                return 0.0
            
            # Backtest and return win rate
            win_rate = self._backtest_winrate(test_params)
            return win_rate
            
        except Exception:
            return 0.0

    def _optimize_with_grid_search(self) -> tuple:
        """Fallback grid search optimization - BACKGROUND SAFE"""
        best_params = None
        best_winrate = 0.0
        
        try:
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                # CRITICAL: Reduced search space for background processing
                for qqe_len in [10, 12, 15]:  # Reduced from [8, 10, 12, 15]
                    for qqe_smooth in [4, 5, 6]:  # Reduced from [3, 4, 5, 6]
                        for st_period in [8, 10, 12]:  # Reduced from [6, 8, 10, 12]
                            for st_mult in [2.5, 3.0]:  # Reduced from [2.0, 2.5, 3.0]
                                test_params = SignalParameters(
                                    strategy_type=self.strategy_type,
                                    qqe_length=qqe_len,
                                    qqe_smooth=qqe_smooth,
                                    supertrend_period=st_period,
                                    supertrend_multiplier=st_mult
                                )
                                
                                winrate = self._backtest_winrate(test_params)
                                if winrate > best_winrate:
                                    best_winrate = winrate
                                    best_params = {
                                        'qqe_length': qqe_len,
                                        'qqe_smooth': qqe_smooth,
                                        'supertrend_period': st_period,
                                        'supertrend_multiplier': st_mult
                                    }
            
            return best_params, best_winrate
            
        except Exception as e:
            self.logger.error(f"Background grid search optimization failed: {e}")
            return None, 0.0

    def _backtest_winrate(self, test_params: SignalParameters) -> float:
        """Backtest parameters and return win rate"""
        try:
            if not self._historical_data_cache:
                return 0.0
            
            df = pd.DataFrame(self._historical_data_cache, 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_calc = df.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_calc[col] = df_calc[col].astype(float)
            
            # Calculate indicators with test parameters
            indicators_history = []
            
            for i in range(30, len(df_calc)):  # Skip first 30 for indicator warmup
                current_df = df_calc.iloc[:i+1]
                
                if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                    # Test QQE + Supertrend
                    qqe_result = ta.qqe(current_df['close'], 
                                      length=test_params.qqe_length,
                                      smooth=test_params.qqe_smooth,
                                      factor=test_params.qqe_factor)
                    
                    st_result = ta.supertrend(current_df['high'], current_df['low'], current_df['close'],
                                            length=test_params.supertrend_period,
                                            multiplier=test_params.supertrend_multiplier)
                    
                    if (qqe_result is not None and not qqe_result.empty and len(qqe_result.columns) >= 2 and
                        st_result is not None and not st_result.empty):
                        
                        dir_col = next((col for col in st_result.columns if 'SUPERTd' in col), None)
                        if dir_col:
                            qqe_val = qqe_result.iloc[-1, 0]
                            qqe_sig = qqe_result.iloc[-1, 1]
                            st_dir = st_result[dir_col].iloc[-1]
                            
                            # Generate signal
                            signal = 'none'
                            if qqe_sig > qqe_val and st_dir == 1:
                                signal = 'buy'
                            elif qqe_sig <= qqe_val and st_dir != 1:
                                signal = 'sell'
                            
                            if signal != 'none':
                                indicators_history.append({
                                    'signal': signal,
                                    'price': current_df['close'].iloc[-1],
                                    'index': i
                                })
            
            # Evaluate signals
            if len(indicators_history) == 0:
                return 0.0
            
            wins = 0
            total = 0
            
            for signal_data in indicators_history:
                signal = signal_data['signal']
                entry_price = signal_data['price']
                entry_idx = signal_data['index']
                
                # Look ahead 10 candles to evaluate
                if entry_idx + 10 < len(df_calc):
                    future_price = df_calc['close'].iloc[entry_idx + 10]
                    price_change_pct = ((future_price - entry_price) / entry_price) * 100
                    
                    is_correct = False
                    if signal == 'buy' and price_change_pct > 0.5:  # 0.5% minimum
                        is_correct = True
                    elif signal == 'sell' and price_change_pct < -0.5:  # 0.5% minimum
                        is_correct = True
                    
                    if is_correct:
                        wins += 1
                    total += 1
            
            return (wins / total) * 100.0 if total > 0 else 0.0
            
        except Exception as e:
            self.logger.debug(f"Backtest error: {e}")
            return 0.0

    def _update_optimized_parameters(self, best_params: Dict, best_score: float):
        """Update parameters with optimization results"""
        try:
            old_accuracy = self.params.accuracy
            
            # Update parameters
            for param_name, param_value in best_params.items():
                if hasattr(self.params, param_name):
                    setattr(self.params, param_name, param_value)
            
            self.params.optimization_score = best_score
            self.params.last_used = time.time()
            
            # Reset performance tracking for new parameters
            self.params.accuracy = 0.0
            self.params.total_signals = 0
            self.params.winning_signals = 0
            self.signal_performance.clear()
            
            self.logger.info(f"Parameters updated: {old_accuracy:.1f}% -> {best_score:.1f}% target win rate")
            self._save_config()
            
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")

    def _update_historical_cache(self, ohlcv_data):
        """Cache historical data for optimization"""
        current_time = time.time()
        if current_time - self._cache_timestamp > self._cache_validity:
            self._historical_data_cache = ohlcv_data.copy()
            self._cache_timestamp = current_time

    def _track_signal(self, signal: str, price: float):
        """Track signal performance for optimization"""
        try:
            signal_data = {
                'signal': signal,
                'price': price,
                'timestamp': time.time(),
                'evaluated': False,
                'correct': None
            }
            self.signal_performance.append(signal_data)
            
            # Evaluate older signals
            self._evaluate_signals()
            
            # Save periodically
            if len(self.signal_performance) % 3 == 0:
                self._save_config()
                
        except Exception as e:
            self.logger.error(f"Error tracking signal: {e}")

    def _evaluate_signals(self):
        """Evaluate signal outcomes for accuracy calculation"""
        try:
            current_time = time.time()
            evaluation_delay = 300  # 5 minutes
            
            for signal_data in self.signal_performance:
                if signal_data.get('evaluated', False):
                    continue
                
                if current_time - signal_data['timestamp'] < evaluation_delay:
                    continue
                
                # Find future price for evaluation
                future_price = None
                for other_signal in self.signal_performance:
                    if (other_signal['timestamp'] > signal_data['timestamp'] + evaluation_delay and
                        other_signal['timestamp'] <= signal_data['timestamp'] + evaluation_delay + 300):
                        future_price = other_signal['price']
                        break
                
                if future_price is not None:
                    price_change_pct = ((future_price - signal_data['price']) / signal_data['price']) * 100
                    
                    if signal_data['signal'] == 'buy':
                        signal_data['correct'] = price_change_pct > 0.5
                    elif signal_data['signal'] == 'sell':
                        signal_data['correct'] = price_change_pct < -0.5
                    
                    signal_data['evaluated'] = True
            
            # Update accuracy
            evaluated = [s for s in self.signal_performance if s.get('evaluated', False)]
            if len(evaluated) > 0:
                correct = sum(1 for s in evaluated if s.get('correct', False))
                self.params.accuracy = (correct / len(evaluated)) * 100
                self.params.total_signals = len(evaluated)
                self.params.winning_signals = correct
                
        except Exception as e:
            self.logger.error(f"Error evaluating signals: {e}")

    def evaluate_exit_conditions(self, position_side: str, entry_price: float, current_price: float) -> Dict:
        """Evaluate exit conditions using current optimized indicators"""
        try:
            result = {'should_exit': False, 'exit_reason': '', 'exit_urgency': 'none'}
            
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if position_side == 'long' else ((entry_price - current_price) / entry_price) * 100
            
            # Emergency stop loss
            if pnl_pct < -2.0:
                result.update({
                    'should_exit': True,
                    'exit_reason': f"Emergency SL: {pnl_pct:.2f}% loss",
                    'exit_urgency': 'immediate'
                })
                return result

            # Strategy-specific exit logic using optimized indicators
            last_indicators = getattr(self, '_last_calculated_indicators', {})
            
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                if all(key in last_indicators for key in ['qqe_value', 'qqe_signal', 'st_direction']):
                    qqe_bearish = last_indicators['qqe_value'] < last_indicators['qqe_signal']
                    st_down = last_indicators['st_direction'] != 1
                    
                    if position_side == 'long' and qqe_bearish and st_down:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"QQE+ST bearish exit (PnL: {pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
                    elif position_side == 'short' and not qqe_bearish and not st_down:
                        result.update({
                            'should_exit': True,
                            'exit_reason': f"QQE+ST bullish exit (PnL: {pnl_pct:.2f}%)",
                            'exit_urgency': 'normal'
                        })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Exit evaluation error: {e}")
            return {'should_exit': False, 'exit_reason': 'Error', 'exit_urgency': 'none'}

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
        
        # Return default parameters
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
            
            # Save strategy-specific optimized config
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
            
            # Save signal history
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
        """Get system status with optimization info - THREAD SAFE"""
        with self.optimization_lock:
            optimization_status = self.optimization_in_progress
        
        return {
            'system_type': f'{self.strategy_type}_optimized',
            'strategy_type': self.strategy_type,
            'optimized_parameters': {
                'qqe_length': self.params.qqe_length,
                'qqe_smooth': self.params.qqe_smooth,
                'supertrend_period': self.params.supertrend_period,
                'supertrend_multiplier': self.params.supertrend_multiplier,
                'rsi_length': self.params.rsi_length,
                'macd_fast': self.params.macd_fast,
                'macd_slow': self.params.macd_slow,
                'macd_signal': self.params.macd_signal,
                'tsi_fast': self.params.tsi_fast,
                'tsi_slow': self.params.tsi_slow,
                'tsi_signal': self.params.tsi_signal
            },
            'performance': {
                'win_rate': self.params.accuracy,
                'total_signals': self.params.total_signals,
                'winning_signals': self.params.winning_signals,
                'optimization_score': self.params.optimization_score
            },
            'optimization': {
                'in_progress': optimization_status,
                'last_optimization': self.last_optimization,
                'target_win_rate': self.win_rate_threshold,
                'interval_seconds': self.optimization_interval
            }
        }

def integrate_adaptive_crypto_signals(strategy_instance, config_file: str = None, strategy_type: str = 'tsi_vwap'):
    """Integration function with NON-BLOCKING dynamic parameter optimization"""
    if config_file is None:
        config_file = os.path.join(os.getcwd(), "data", "crypto_signal_configs.json")
    
    strategy_instance.logger.info(f"Integrating NON-BLOCKING optimized {strategy_type} signals, config: {config_file}")
    base_sym = getattr(strategy_instance, 'original_symbol', strategy_instance.symbol)
    
    crypto_sigs = AdaptiveCryptoSignals(symbol=base_sym, config_file=config_file, strategy_type=strategy_type)
    
    strategy_instance._get_technical_direction = lambda: crypto_sigs.get_technical_direction(strategy_instance.exchange)
    strategy_instance.get_signal_status = crypto_sigs.get_system_status
    strategy_instance._crypto_signal_system = crypto_sigs
    
    strategy_instance.logger.info(f"NON-BLOCKING optimized {strategy_type} signals integrated with win rate target: {crypto_sigs.win_rate_threshold}%")
    strategy_instance.logger.info("Signal generation will NOT be blocked by optimization (runs in background)")
    return crypto_sigs