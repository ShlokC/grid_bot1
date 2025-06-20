"""
Fast Self-Improving Adaptive TSI System - FIXED
Fixed accuracy calculation, signal tracking, and parameter optimization.
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

@dataclass
class TSIParameterSet:
    """Lightweight TSI parameters with performance tracking"""
    slow: int
    fast: int 
    signal: int
    accuracy: float = 0.0
    total_signals: int = 0
    winning_signals: int = 0
    last_used: float = 0.0

class FastAdaptiveTSISystem:
    """
    FIXED: Fast self-improving TSI with proper accuracy tracking and optimization.
    """
    
    def __init__(self, symbol: str, config_file: str = "data/tsi_configs.json"):
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        self.symbol = symbol
        self.config_file = config_file
        
        # Reduced parameter space for faster optimization
        self.param_sets = [
            # Fast responsive (good for volatile markets)
            (15, 8, 4), (18, 9, 5), (20, 10, 5), (22, 11, 6),
            # Medium responsive (balanced)
            (25, 13, 7), (28, 14, 8), (30, 15, 9), (32, 16, 10),
            # Slower (good for trending markets)  
            (35, 18, 11), (40, 20, 12), (45, 23, 13), (50, 25, 15),
            # Very fast (scalping)
            (12, 6, 3), (14, 7, 4), (16, 8, 4), (18, 10, 5)
        ]
        
        # Load saved parameters or use defaults
        self.current_params, self.fallback_params = self._load_symbol_config()
        
        # FIXED: Load persistent signal performance data
        self.signal_performance = self._load_signal_history()
        self.market_volatility = deque(maxlen=30)
        
        # FIXED: Reduced thresholds for more frequent optimization
        self.min_signals_for_optimization = 3      # Reduced from 5
        self.accuracy_threshold = 30.0             # Reduced from 35.0
        self.volatility_change_threshold = 0.3
        self.max_optimization_time = 0.2
        
        # State tracking
        self.last_optimization = self.current_params.last_used if hasattr(self.current_params, 'last_used') else 0
        self.optimization_in_progress = False
        self.last_volatility_regime = 0.0
        
        # Signal control
        self.last_signal = 'none'
        self.last_signal_time = 0
        self.signal_cooldown = 20
        
        # FIXED: Load thresholds from config or use defaults
        self.momentum_threshold = getattr(self.current_params, 'momentum_threshold', 1.0)
        self.strength_threshold = getattr(self.current_params, 'strength_threshold', 3.0)
        
        self.logger.info(f"⚡ FIXED Adaptive TSI initialized for {symbol} "
                        f"[{self.current_params.slow}/{self.current_params.fast}/{self.current_params.signal}] "
                        f"[Acc: {self.current_params.accuracy:.1f}%] [Signals: {len(self.signal_performance)}]")

    def _load_signal_history(self) -> deque:
        """FIXED: Load persistent signal history with proper field reconstruction"""
        try:
            import json
            abs_config_file = os.path.abspath(self.config_file)
            
            if os.path.exists(abs_config_file):
                with open(abs_config_file, 'r') as f:
                    configs = json.load(f)
                
                symbol_config = configs.get(self.symbol, {})
                signal_history = symbol_config.get('signal_history', [])
                
                if signal_history:
                    # Convert to deque with max 50 signals, ensuring all fields are present
                    history_deque = deque(maxlen=50)
                    for signal_data in signal_history[-50:]:  # Keep only last 50
                        # FIXED: Ensure all required fields are present with defaults
                        reconstructed_signal = {
                            'signal': signal_data.get('signal', ''),
                            'price': float(signal_data.get('price', 0)),
                            'timestamp': float(signal_data.get('timestamp', 0)),
                            'outcome_checked': bool(signal_data.get('outcome_checked', False)),
                            'correct': signal_data.get('correct'),  # Can be None, True, or False
                            'price_change_pct': signal_data.get('price_change_pct')  # Can be None or float
                        }
                        history_deque.append(reconstructed_signal)
                    
                    self.logger.info(f"📂 Loaded {len(history_deque)} signal history for {self.symbol}")
                    
                    # FIXED: Verify accuracy calculation matches loaded data
                    evaluated_signals = [s for s in history_deque if s.get('outcome_checked', False)]
                    if len(evaluated_signals) > 0:
                        correct_count = sum(1 for s in evaluated_signals if s.get('correct', False))
                        loaded_accuracy = (correct_count / len(evaluated_signals)) * 100
                        config_accuracy = symbol_config.get('accuracy', 0.0)
                        
                        if abs(loaded_accuracy - config_accuracy) > 1.0:  # More than 1% difference
                            self.logger.warning(f"📊 Accuracy mismatch for {self.symbol}: "
                                              f"calculated {loaded_accuracy:.1f}% vs config {config_accuracy:.1f}%")
                    
                    return history_deque
                else:
                    self.logger.info(f"📂 No signal history found for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading signal history: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        return deque(maxlen=50)

    def _ensure_signal_history_sync(self):
        """FIXED: Ensure signal history is properly synced with accuracy metrics"""
        try:
            # Count evaluated signals for accuracy
            evaluated_signals = [s for s in self.signal_performance if s.get('outcome_checked', False)]
            
            if len(evaluated_signals) > 0:
                correct_signals = sum(1 for s in evaluated_signals if s.get('correct', False))
                
                # Update current params to match actual signal history
                self.current_params.total_signals = len(evaluated_signals)
                self.current_params.winning_signals = correct_signals
                self.current_params.accuracy = (correct_signals / len(evaluated_signals)) * 100
                
                self.logger.debug(f"📊 Synced {self.symbol}: {len(self.signal_performance)} total, "
                                f"{len(evaluated_signals)} evaluated, {correct_signals} correct")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing signal history: {e}")

    def _load_symbol_config(self) -> Tuple[TSIParameterSet, TSIParameterSet]:
        """Load saved TSI configuration for this symbol"""
        try:
            import os
            import json
            
            abs_config_file = os.path.abspath(self.config_file)
            config_dir = os.path.dirname(abs_config_file)
            
            if not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            if os.path.exists(abs_config_file):
                with open(abs_config_file, 'r') as f:
                    configs = json.load(f)
                
                symbol_config = configs.get(self.symbol, {})
                if symbol_config:
                    current_params = TSIParameterSet(
                        slow=symbol_config.get('slow', 25),
                        fast=symbol_config.get('fast', 13),
                        signal=symbol_config.get('signal', 7),
                        accuracy=symbol_config.get('accuracy', 0.0),
                        total_signals=symbol_config.get('total_signals', 0),
                        winning_signals=symbol_config.get('winning_signals', 0),
                        last_used=symbol_config.get('last_used', 0.0)
                    )
                    
                    # Add custom thresholds if saved
                    current_params.momentum_threshold = symbol_config.get('momentum_threshold', 1.0)
                    current_params.strength_threshold = symbol_config.get('strength_threshold', 3.0)
                    
                    self.logger.info(f"📂 Loaded config for {self.symbol}: "
                                   f"{current_params.slow}/{current_params.fast}/{current_params.signal} "
                                   f"(Acc: {current_params.accuracy:.1f}%)")
                    
                    return current_params, current_params
            
        except Exception as e:
            self.logger.error(f"❌ Error loading config for {self.symbol}: {e}")
        
        # Return defaults if loading failed
        default_params = TSIParameterSet(slow=25, fast=13, signal=7)
        self.logger.info(f"📂 Using default config for {self.symbol}: 25/13/7")
        return default_params, default_params

    def _save_symbol_config(self, force: bool = False):
        """FIXED: Unified config save with proper signal history sync"""
        try:
            import os
            import json
            
            # FIXED: Sync signal history before saving
            self._ensure_signal_history_sync()
            
            abs_config_file = os.path.abspath(self.config_file)
            config_dir = os.path.dirname(abs_config_file)
            
            if not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            # Load existing configs
            configs = {}
            if os.path.exists(abs_config_file):
                try:
                    with open(abs_config_file, 'r') as f:
                        configs = json.load(f)
                except Exception as e:
                    self.logger.warning(f"⚠️ Corrupted config file: {e}, creating new one")
                    configs = {}
            
            # FIXED: Ensure signal_performance is properly converted to serializable format
            signal_history_list = []
            for signal_data in self.signal_performance:
                # Convert signal data to serializable dict
                serializable_signal = {
                    'signal': signal_data.get('signal', ''),
                    'price': float(signal_data.get('price', 0)),
                    'timestamp': float(signal_data.get('timestamp', 0)),
                    'outcome_checked': bool(signal_data.get('outcome_checked', False)),
                    'correct': signal_data.get('correct'),
                    'price_change_pct': signal_data.get('price_change_pct')
                }
                signal_history_list.append(serializable_signal)
            
            # Update this symbol's config with verified data
            configs[self.symbol] = {
                'slow': self.current_params.slow,
                'fast': self.current_params.fast,
                'signal': self.current_params.signal,
                'accuracy': self.current_params.accuracy,
                'total_signals': self.current_params.total_signals,
                'winning_signals': self.current_params.winning_signals,
                'last_used': time.time(),
                'momentum_threshold': self.momentum_threshold,
                'strength_threshold': self.strength_threshold,
                'last_optimized': self.last_optimization,
                'signal_history': signal_history_list  # FIXED: Properly serialized signal history
            }
            
            # Atomic write to prevent corruption
            temp_file = abs_config_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(configs, f, indent=2)
            
            os.replace(temp_file, abs_config_file)
            
            self.logger.info(f"💾 SAVED config for {self.symbol}: "
                           f"[{self.current_params.slow}/{self.current_params.fast}/{self.current_params.signal}] "
                           f"[Acc: {self.current_params.accuracy:.1f}%] "
                           f"[History: {len(signal_history_list)} signals] "
                           f"[Total: {self.current_params.total_signals}, Wins: {self.current_params.winning_signals}]")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving config for {self.symbol}: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def get_technical_direction(self, exchange) -> str:
        """Fast signal generation with FIXED tracking and optimization"""
        try:
            # Prevent signal spam
            if time.time() - self.last_signal_time < self.signal_cooldown:
                return 'none'
            
            # Get OHLCV data
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=60)
            if not ohlcv_data or len(ohlcv_data) < 30:
                return 'none'
            
            # Fast DataFrame conversion
            closes = [float(candle[4]) for candle in ohlcv_data]
            
            # Update market state
            self._fast_update_market_state(closes)
            
            # FIXED: Check optimization with lower thresholds
            if self._should_optimize_fast():
                self._trigger_background_optimization(closes)
            
            # Generate signal
            signal = self._generate_fast_signal(closes)
            
            # FIXED: Track signal with proper outcome evaluation
            if signal != 'none':
                self._track_signal_improved(signal, closes[-1])
                self.last_signal = signal
                self.last_signal_time = time.time()
                
                params_str = f"{self.current_params.slow}/{self.current_params.fast}/{self.current_params.signal}"
                acc = self.current_params.accuracy
                signals_count = len(self.signal_performance)
                self.logger.info(f"⚡ Signal: {signal.upper()} [{params_str}] [Acc: {acc:.0f}%] [Count: {signals_count}]")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error in TSI: {e}")
            return 'none'

    def _track_signal_improved(self, signal: str, price: float):
        """FIXED: Improved signal tracking with unified saving"""
        try:
            current_time = time.time()
            
            # Add new signal
            signal_data = {
                'signal': signal,
                'price': price,
                'timestamp': current_time,
                'outcome_checked': False,
                'correct': None
            }
            self.signal_performance.append(signal_data)
            
            # FIXED: Evaluate older signals that haven't been checked
            self._evaluate_signal_outcomes()
            
            # Update accuracy after evaluation
            self._update_accuracy_improved()
            
            # FIXED: Save config every 3 signals instead of separate signal history
            if len(self.signal_performance) % 3 == 0:
                self._save_symbol_config(force=True)
                
        except Exception as e:
            self.logger.error(f"❌ Error tracking signal: {e}")

    def _evaluate_signal_outcomes(self):
        """FIXED: Evaluate signal outcomes after sufficient time has passed"""
        try:
            current_time = time.time()
            evaluation_delay = 300  # 5 minutes to evaluate signal outcome
            
            for signal_data in self.signal_performance:
                # Skip if already evaluated
                if signal_data.get('outcome_checked', False):
                    continue
                
                # Skip if not enough time has passed
                if current_time - signal_data['timestamp'] < evaluation_delay:
                    continue
                
                # Find a recent price to compare against
                recent_price = None
                for other_signal in reversed(self.signal_performance):
                    if (other_signal['timestamp'] > signal_data['timestamp'] + evaluation_delay and 
                        other_signal['timestamp'] <= signal_data['timestamp'] + evaluation_delay + 300):  # Within 5-10 minutes
                        recent_price = other_signal['price']
                        break
                
                if recent_price is None:
                    continue  # Can't evaluate yet
                
                # Evaluate signal correctness
                price_change = recent_price - signal_data['price']
                price_change_pct = (price_change / signal_data['price']) * 100
                
                # FIXED: Proper signal evaluation logic
                if signal_data['signal'] == 'buy':
                    signal_data['correct'] = price_change_pct > 0.1  # At least 0.1% gain
                elif signal_data['signal'] == 'sell':
                    signal_data['correct'] = price_change_pct < -0.1  # At least 0.1% drop
                
                signal_data['outcome_checked'] = True
                signal_data['price_change_pct'] = price_change_pct
                
        except Exception as e:
            self.logger.error(f"❌ Error evaluating signal outcomes: {e}")

    def _update_accuracy_improved(self):
        """FIXED: Improved accuracy calculation with proper sync"""
        try:
            # Only consider evaluated signals
            evaluated_signals = [s for s in self.signal_performance if s.get('outcome_checked', False)]
            
            if len(evaluated_signals) == 0:
                return
            
            # Calculate accuracy
            correct_signals = sum(1 for s in evaluated_signals if s.get('correct', False))
            total_evaluated = len(evaluated_signals)
            
            old_accuracy = self.current_params.accuracy
            new_accuracy = (correct_signals / total_evaluated) * 100
            
            # Update parameters
            self.current_params.accuracy = new_accuracy
            self.current_params.total_signals = total_evaluated
            self.current_params.winning_signals = correct_signals
            
            # FIXED: Log accuracy updates for debugging
            if abs(new_accuracy - old_accuracy) > 0.1:  # Any change
                self.logger.info(f"📊 Accuracy updated for {self.symbol}: "
                               f"{old_accuracy:.1f}% → {new_accuracy:.1f}% "
                               f"({correct_signals}/{total_evaluated}) "
                               f"[History: {len(self.signal_performance)} total]")
                
                # Save config immediately on significant accuracy changes
                if abs(new_accuracy - old_accuracy) > 5.0:  # 5% change threshold
                    self._save_symbol_config(force=True)
                
        except Exception as e:
            self.logger.error(f"❌ Error updating accuracy: {e}")

    def _should_optimize_fast(self) -> bool:
        """FIXED: More aggressive optimization triggers"""
        try:
            current_time = time.time()
            
            # Don't optimize too frequently (minimum 2 minutes)
            if current_time - self.last_optimization < 120:
                return False
            
            if self.optimization_in_progress:
                return False
            
            # FIXED: Lower signal requirement
            if len(self.signal_performance) < self.min_signals_for_optimization:
                return False
            
            # FIXED: Trigger 1: Low accuracy (lowered threshold)
            if self.current_params.accuracy < self.accuracy_threshold:
                self.logger.warning(f"⚡ Low accuracy trigger: {self.current_params.accuracy:.1f}% < {self.accuracy_threshold}%")
                return True
            
            # FIXED: Trigger 2: No optimization yet but have signals
            if self.last_optimization == 0 and len(self.signal_performance) >= 5:
                self.logger.info(f"⚡ Initial optimization trigger: {len(self.signal_performance)} signals ready")
                return True
            
            # Trigger 3: Market regime change
            if len(self.market_volatility) >= 20:
                old_vol = np.mean(list(self.market_volatility)[:10])
                new_vol = np.mean(list(self.market_volatility)[-10:])
                
                if old_vol > 0 and abs(new_vol - old_vol) / old_vol > self.volatility_change_threshold:
                    self.logger.info(f"⚡ Volatility regime change: {old_vol:.4f} → {new_vol:.4f}")
                    return True
            
            # FIXED: Trigger 4: Time-based (reduced to 5 minutes)
            if current_time - self.last_optimization > 300:
                self.logger.info(f"⚡ Time-based optimization trigger")
                return True
            
            return False
            
        except Exception:
            return False

    def _fast_optimize_parameters(self, closes: list):
        """FIXED: Fast parameter optimization with immediate saving"""
        try:
            start_time = time.time()
            
            if len(closes) < 40:
                return
            
            best_params = None
            best_score = -1.0
            current_score = self.current_params.accuracy
            
            # Test parameter sets
            for slow, fast, signal_period in self.param_sets[:8]:  # Test only 8 sets for speed
                try:
                    if time.time() - start_time > self.max_optimization_time:
                        break
                    
                    score = self._quick_backtest(closes, slow, fast, signal_period)
                    if score > best_score:
                        best_score = score
                        best_params = TSIParameterSet(
                            slow=slow, fast=fast, signal=signal_period,
                            accuracy=score, last_used=time.time()
                        )
                        
                except Exception:
                    continue
            
            # FIXED: Update parameters if better found (reduced improvement threshold)
            if best_params and best_score > current_score + 2:  # Only 2% improvement needed
                old_params = f"{self.current_params.slow}/{self.current_params.fast}/{self.current_params.signal}"
                new_params = f"{best_params.slow}/{best_params.fast}/{best_params.signal}"
                
                # Keep fallback
                self.fallback_params = self.current_params
                
                # Update parameters but preserve tracking data
                best_params.total_signals = self.current_params.total_signals
                best_params.winning_signals = self.current_params.winning_signals
                self.current_params = best_params
                
                # FIXED: Save immediately
                self._save_symbol_config(force=True)
                
                self.logger.info(f"⚡ OPTIMIZED & SAVED: {old_params} → {new_params} "
                               f"(Score: +{best_score - current_score:.1f}%)")
            
            self.last_optimization = time.time()
            optimization_time = (time.time() - start_time) * 1000
            self.logger.debug(f"⚡ Optimization completed in {optimization_time:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"❌ Error in optimization: {e}")

    def _fast_update_market_state(self, closes: list):
        """Ultra-fast market state update"""
        try:
            if len(closes) < 10:
                return
            
            # Calculate simple volatility
            recent_returns = []
            for i in range(len(closes)-5, len(closes)):
                if i > 0:
                    ret = abs(closes[i] - closes[i-1]) / closes[i-1]
                    recent_returns.append(ret)
            
            current_volatility = np.mean(recent_returns) if recent_returns else 0.0
            self.market_volatility.append(current_volatility)
            
        except Exception:
            pass

    def _trigger_background_optimization(self, closes: list):
        """Start background optimization without blocking signal generation"""
        try:
            if self.optimization_in_progress:
                return
            
            self.optimization_in_progress = True
            
            def background_optimize():
                try:
                    self._fast_optimize_parameters(closes)
                finally:
                    self.optimization_in_progress = False
            
            optimization_thread = threading.Thread(target=background_optimize, daemon=True)
            optimization_thread.start()
            
        except Exception as e:
            self.optimization_in_progress = False
            self.logger.error(f"❌ Error starting background optimization: {e}")

    def _quick_backtest(self, closes: list, slow: int, fast: int, signal: int) -> float:
        """Ultra-fast backtesting"""
        try:
            price_series = pd.Series(closes)
            
            tsi_result = ta.tsi(price_series, slow=slow, fast=fast, signal=signal)
            if tsi_result is None or len(tsi_result.dropna()) < 15:
                return 0.0
            
            tsi_clean = tsi_result.dropna()
            if len(tsi_clean.columns) < 2:
                return 0.0
            
            tsi_line = tsi_clean.iloc[:, 0]
            tsi_signal_line = tsi_clean.iloc[:, 1]
            
            correct_signals = 0
            total_signals = 0

            test_length = min(40, len(tsi_line) - 5)
            
            for i in range(test_length):
                idx = len(tsi_line) - test_length + i
                if idx < 2:
                    continue
                
                momentum = tsi_line.iloc[idx] - tsi_line.iloc[idx-2]
                current_tsi = tsi_line.iloc[idx]
                current_signal = tsi_signal_line.iloc[idx]
                
                signal_generated = 'none'
                if momentum > 1.0 and current_tsi > current_signal:
                    signal_generated = 'buy'
                elif momentum < -1.0 and current_tsi < current_signal:
                    signal_generated = 'sell'
                
                if signal_generated != 'none':
                    total_signals += 1
                    
                    if idx < len(closes) - 3:
                        price_change = closes[idx + 2] - closes[idx]
                        if (signal_generated == 'buy' and price_change > 0) or \
                           (signal_generated == 'sell' and price_change < 0):
                            correct_signals += 1
            
            return (correct_signals / total_signals * 100) if total_signals > 0 else 0.0
            
        except Exception:
            return 0.0

    def _generate_fast_signal(self, closes: list) -> str:
        """Fast signal generation using current parameters"""
        try:
            if len(closes) < 30:
                return 'none'
            
            price_series = pd.Series(closes)
            tsi_result = ta.tsi(price_series, 
                              slow=self.current_params.slow,
                              fast=self.current_params.fast,
                              signal=self.current_params.signal)
            
            if tsi_result is None or len(tsi_result.dropna()) < 10:
                if self.fallback_params.slow != self.current_params.slow:
                    tsi_result = ta.tsi(price_series,
                                      slow=self.fallback_params.slow,
                                      fast=self.fallback_params.fast,
                                      signal=self.fallback_params.signal)
                    if tsi_result is None:
                        return 'none'
                else:
                    return 'none'
            
            tsi_clean = tsi_result.dropna()
            if len(tsi_clean.columns) < 2 or len(tsi_clean) < 5:
                return 'none'
            
            tsi_line = tsi_clean.iloc[:, 0]
            tsi_signal_line = tsi_clean.iloc[:, 1]
            
            current_tsi = tsi_line.iloc[-1]
            current_signal = tsi_signal_line.iloc[-1]
            
            lookback = max(2, self.current_params.fast // 5)
            if len(tsi_line) > lookback:
                momentum = tsi_line.iloc[-1] - tsi_line.iloc[-1-lookback]
            else:
                momentum = tsi_line.iloc[-1] - tsi_line.iloc[-2]
            
            separation = abs(current_tsi - current_signal)
            
            if (momentum > self.momentum_threshold and 
                current_tsi > current_signal and
                separation > self.strength_threshold):
                return 'buy'
            elif (momentum < -self.momentum_threshold and 
                  current_tsi < current_signal and
                  separation > self.strength_threshold):
                return 'sell'
            elif current_tsi < -25 and momentum > self.momentum_threshold * 0.7:
                return 'buy'
            elif current_tsi > 25 and momentum < -self.momentum_threshold * 0.7:
                return 'sell'
            
            return 'none'
            
        except Exception as e:
            self.logger.error(f"❌ Error in signal generation: {e}")
            return 'none'

    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'system_type': 'fixed_adaptive',
            'current_params': {
                'slow': self.current_params.slow,
                'fast': self.current_params.fast,
                'signal': self.current_params.signal
            },
            'performance': {
                'accuracy': self.current_params.accuracy,
                'total_signals': self.current_params.total_signals,
                'winning_signals': self.current_params.winning_signals,
                'tracked_signals': len(self.signal_performance)
            },
            'optimization': {
                'in_progress': self.optimization_in_progress,
                'last_optimization': self.last_optimization
            },
            'thresholds': {
                'momentum': self.momentum_threshold,
                'strength': self.strength_threshold
            }
        }


def integrate_adaptive_tsi(strategy_instance, config_file: str = None):
    """Integration function with FIXED persistent configuration storage"""
    if config_file is None:
        config_file = os.path.join("data", "tsi_configs.json")
    
    strategy_instance.logger.info(f"🔧 Integrating FIXED TSI with config: {os.path.abspath(config_file)}")
    
    adaptive_tsi = FastAdaptiveTSISystem(symbol=strategy_instance.symbol, config_file=config_file)
    
    def adaptive_get_technical_direction():
        return adaptive_tsi.get_technical_direction(strategy_instance.exchange)
    
    def get_tsi_status():
        return adaptive_tsi.get_system_status()
    
    def save_tsi_config():
        """Force save current configuration"""
        adaptive_tsi._save_symbol_config(force=True)
        return True
    
    def test_tsi_config():
        """Test configuration save/load"""
        try:
            strategy_instance.logger.info("🧪 Testing FIXED TSI config...")
            
            adaptive_tsi._save_symbol_config(force=True)
            
            if os.path.exists(adaptive_tsi.config_file):
                with open(adaptive_tsi.config_file, 'r') as f:
                    import json
                    data = json.load(f)
                    if strategy_instance.symbol in data:
                        symbol_config = data[strategy_instance.symbol]
                        strategy_instance.logger.info(f"✅ FIXED config for {strategy_instance.symbol}: {symbol_config}")
                        return True
                    else:
                        strategy_instance.logger.error(f"❌ No config found for {strategy_instance.symbol}")
                        return False
            else:
                strategy_instance.logger.error(f"❌ Config file not found: {adaptive_tsi.config_file}")
                return False
                
        except Exception as e:
            strategy_instance.logger.error(f"❌ Test failed: {e}")
            return False
    
    strategy_instance._get_technical_direction = adaptive_get_technical_direction
    strategy_instance.get_tsi_status = get_tsi_status
    strategy_instance.save_tsi_config = save_tsi_config
    strategy_instance.test_tsi_config = test_tsi_config
    strategy_instance._adaptive_tsi_system = adaptive_tsi
    
    # Test config immediately
    strategy_instance.test_tsi_config()
    
    strategy_instance.logger.info("⚡ FIXED Adaptive TSI with persistent storage integrated!")
    
    return adaptive_tsi


def integrate_momentum_tsi(strategy_instance, config_file: str = None):
    """Backward compatibility wrapper with FIXED persistent storage"""
    return integrate_adaptive_tsi(strategy_instance, config_file)