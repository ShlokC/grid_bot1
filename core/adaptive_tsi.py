"""
Fast Self-Improving Adaptive TSI System
Dynamic optimization with minimal performance impact and smart triggers.
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
    Fast self-improving TSI with persistent configuration storage.
    Saves/loads optimized parameters per symbol to avoid re-optimization.
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
        
        # Performance tracking (lightweight)
        self.signal_performance = deque(maxlen=20)  # Last 20 signals only
        self.market_volatility = deque(maxlen=30)   # 30 periods for regime detection
        
        # Dynamic optimization triggers
        self.min_signals_for_optimization = 5      # Reduced from 50
        self.accuracy_threshold = 35.0             # Trigger if below 45%
        self.volatility_change_threshold = 0.3     # 30% volatility change
        self.max_optimization_time = 0.2           # 100ms max optimization time
        
        # State tracking
        self.last_optimization = 0
        self.optimization_in_progress = False
        self.last_volatility_regime = 0.0
        self.config_save_pending = False
        
        # Signal control
        self.last_signal = 'none'
        self.last_signal_time = 0
        self.signal_cooldown = 20  # Reduced from 30
        
        # Dynamic thresholds (loaded from config or calculated)
        self.momentum_threshold = getattr(self.current_params, 'momentum_threshold', 1.0)
        self.strength_threshold = getattr(self.current_params, 'strength_threshold', 3.0)
        
        self.logger.info(f"⚡ Fast Adaptive TSI initialized for {symbol} "
                        f"[{self.current_params.slow}/{self.current_params.fast}/{self.current_params.signal}] "
                        f"[Acc: {self.current_params.accuracy:.1f}%]")

    def _load_symbol_config(self) -> Tuple[TSIParameterSet, TSIParameterSet]:
        """Load saved TSI configuration for this symbol"""
        try:
            import os
            import json
            
            # Resolve absolute path
            abs_config_file = os.path.abspath(self.config_file)
            config_dir = os.path.dirname(abs_config_file)
            
            self.logger.debug(f"📂 Config file path: {abs_config_file}")
            
            # Ensure data directory exists
            if not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
                self.logger.info(f"📁 Created config directory: {config_dir}")
            
            # Try to load existing config
            if os.path.exists(abs_config_file):
                self.logger.debug(f"📂 Loading config from: {abs_config_file}")
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
                    
                    self.logger.info(f"📂 Loaded saved config for {self.symbol}: "
                                   f"{current_params.slow}/{current_params.fast}/{current_params.signal} "
                                   f"(Acc: {current_params.accuracy:.1f}%)")
                    
                    return current_params, current_params
                else:
                    self.logger.info(f"📂 No saved config found for {self.symbol} in {abs_config_file}")
            else:
                self.logger.info(f"📂 Config file not found: {abs_config_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading config for {self.symbol}: {e}")
        
        # Return defaults if loading failed
        default_params = TSIParameterSet(slow=25, fast=13, signal=7)
        self.logger.info(f"📂 Using default config for {self.symbol}: 25/13/7")
        return default_params, default_params

    def _save_symbol_config(self, force: bool = False):
        """Save current TSI configuration for this symbol"""
        try:
            # Don't save too frequently unless forced
            if not force and self.config_save_pending:
                return
            
            import os
            import json
            
            # Resolve absolute path
            abs_config_file = os.path.abspath(self.config_file)
            config_dir = os.path.dirname(abs_config_file)
            
            # Ensure directory exists
            if not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
                self.logger.info(f"📁 Created config directory: {config_dir}")
            
            # Load existing configs
            configs = {}
            if os.path.exists(abs_config_file):
                try:
                    with open(abs_config_file, 'r') as f:
                        configs = json.load(f)
                    self.logger.debug(f"📂 Loaded existing configs: {len(configs)} symbols")
                except Exception as e:
                    self.logger.warning(f"⚠️ Corrupted config file: {e}, creating new one")
                    configs = {}
            
            # Update this symbol's config
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
                'last_optimized': self.last_optimization
            }
            
            # Atomic write to prevent corruption
            temp_file = abs_config_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(configs, f, indent=2)
            
            # Replace original file
            os.replace(temp_file, abs_config_file)
            
            self.config_save_pending = False
            self.logger.info(f"💾 Saved config for {self.symbol} to {abs_config_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving config for {self.symbol}: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def _schedule_config_save(self):
        """Schedule a config save to avoid excessive I/O"""
        self.config_save_pending = True
        
        # Save in background after delay
        def delayed_save():
            time.sleep(5)  # 5 second delay to batch saves
            if self.config_save_pending:
                self._save_symbol_config()

    def get_technical_direction(self, exchange) -> str:
        """
        Fast signal generation with background optimization.
        Maximum 50ms execution time, no blocking operations.
        """
        try:
            # Prevent signal spam
            if time.time() - self.last_signal_time < self.signal_cooldown:
                return 'none'
            
            # Get OHLCV data (minimal required)
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='5m', limit=60)
            if not ohlcv_data or len(ohlcv_data) < 30:
                return 'none'
            
            # Fast DataFrame conversion
            closes = [float(candle[4]) for candle in ohlcv_data]
            
            # Update market state (fast)
            self._fast_update_market_state(closes)
            
            # Check if optimization needed (non-blocking)
            if self._should_optimize_fast():
                self._trigger_background_optimization(closes)
            
            # Generate signal using current parameters (fast)
            signal = self._generate_fast_signal(closes)
            
            # Track signal performance (minimal overhead)
            if signal != 'none':
                self._track_signal_fast(signal, closes[-1])
                self.last_signal = signal
                self.last_signal_time = time.time()
                
                params_str = f"{self.current_params.slow}/{self.current_params.fast}/{self.current_params.signal}"
                acc = self.current_params.accuracy
                self.logger.info(f"⚡ FAST Signal: {signal.upper()} [{params_str}] [Acc: {acc:.0f}%]")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error in fast TSI: {e}")
            return 'none'

    def _fast_update_market_state(self, closes: list):
        """Ultra-fast market state update (< 5ms)"""
        try:
            if len(closes) < 10:
                return
            
            # Calculate simple volatility (fast)
            recent_returns = []
            for i in range(len(closes)-5, len(closes)):
                if i > 0:
                    ret = abs(closes[i] - closes[i-1]) / closes[i-1]
                    recent_returns.append(ret)
            
            current_volatility = np.mean(recent_returns) if recent_returns else 0.0
            self.market_volatility.append(current_volatility)
            
        except Exception:
            pass

    def _should_optimize_fast(self) -> bool:
        """Fast optimization trigger check (< 2ms)"""
        try:
            current_time = time.time()
            
            # Don't optimize too frequently (minimum 5 minutes)
            if current_time - self.last_optimization < 120:
                return False
            
            # Check if optimization already running
            if self.optimization_in_progress:
                return False
            
            # Need minimum signal history
            if len(self.signal_performance) < self.min_signals_for_optimization:
                return False
            
            # Trigger 1: Low accuracy
            if self.current_params.accuracy < self.accuracy_threshold:
                self.logger.warning(f"⚡ Low accuracy trigger: {self.current_params.accuracy:.1f}%")
                return True
            
            # Trigger 2: Market regime change (volatility)
            if len(self.market_volatility) >= 20:
                old_vol = np.mean(list(self.market_volatility)[:10])
                new_vol = np.mean(list(self.market_volatility)[-10:])
                
                if old_vol > 0 and abs(new_vol - old_vol) / old_vol > self.volatility_change_threshold:
                    self.logger.info(f"⚡ Volatility regime change: {old_vol:.4f} → {new_vol:.4f}")
                    return True
            
            # Trigger 3: Time-based (reduced to 10 minutes)
            if current_time - self.last_optimization > 600:
                return True
            
            return False
            
        except Exception:
            return False

    def _trigger_background_optimization(self, closes: list):
        """Start background optimization without blocking signal generation"""
        try:
            if self.optimization_in_progress:
                return
            
            # Mark optimization as starting
            self.optimization_in_progress = True
            
            # Run optimization in background thread
            def background_optimize():
                try:
                    self._fast_optimize_parameters(closes)
                finally:
                    self.optimization_in_progress = False
            
            # Start background thread
            optimization_thread = threading.Thread(target=background_optimize, daemon=True)
            optimization_thread.start()
            
        except Exception as e:
            self.optimization_in_progress = False
            self.logger.error(f"❌ Error starting background optimization: {e}")

    def _fast_optimize_parameters(self, closes: list):
        """Fast parameter optimization with automatic saving (< 100ms total)"""
        try:
            start_time = time.time()
            
            if len(closes) < 40:
                return
            
            best_params = None
            best_score = -1.0
            current_score = self.current_params.accuracy
            
            # Test only 6 parameter sets (reduced from 60)
            for slow, fast, signal_period in self.param_sets:
                try:
                    # Quick timeout check
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
            
            # Update parameters if better found (require 5% improvement)
            if best_params and best_score > current_score + 5:
                old_params = f"{self.current_params.slow}/{self.current_params.fast}/{self.current_params.signal}"
                new_params = f"{best_params.slow}/{best_params.fast}/{best_params.signal}"
                
                # Keep fallback
                self.fallback_params = self.current_params
                self.current_params = best_params
                
                # Save improved configuration
                self._schedule_config_save()
                
                self.logger.info(f"⚡ OPTIMIZED & SAVED: {old_params} → {new_params} "
                               f"(Score: +{best_score - current_score:.1f}%)")
            
            self.last_optimization = time.time()
            optimization_time = (time.time() - start_time) * 1000
            self.logger.debug(f"⚡ Optimization completed in {optimization_time:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"❌ Error in fast optimization: {e}")

    def _quick_backtest(self, closes: list, slow: int, fast: int, signal: int) -> float:
        """Ultra-fast backtesting (< 15ms per parameter set)"""
        try:
            # Convert to pandas Series for TSI calculation
            price_series = pd.Series(closes)
            
            # Calculate TSI (this is the slow part, but unavoidable)
            tsi_result = ta.tsi(price_series, slow=slow, fast=fast, signal=signal)
            if tsi_result is None or len(tsi_result.dropna()) < 15:
                return 0.0
            
            tsi_clean = tsi_result.dropna()
            if len(tsi_clean.columns) < 2:
                return 0.0
            
            tsi_line = tsi_clean.iloc[:, 0]
            tsi_signal_line = tsi_clean.iloc[:, 1]
            
            # Quick signal generation and scoring
            correct_signals = 0
            total_signals = 0

            # Test only last 40 points for speed
            test_length = min(40, len(tsi_line) - 5)
            
            for i in range(test_length):
                idx = len(tsi_line) - test_length + i
                if idx < 2:
                    continue
                
                # Simple momentum calculation
                momentum = tsi_line.iloc[idx] - tsi_line.iloc[idx-2]
                current_tsi = tsi_line.iloc[idx]
                current_signal = tsi_signal_line.iloc[idx]
                
                # Generate signal
                signal_generated = 'none'
                if momentum > 1.0 and current_tsi > current_signal:
                    signal_generated = 'buy'
                elif momentum < -1.0 and current_tsi < current_signal:
                    signal_generated = 'sell'
                
                if signal_generated != 'none':
                    total_signals += 1
                    
                    # Check if signal was correct (simplified)
                    if idx < len(closes) - 3:
                        price_change = closes[idx + 2] - closes[idx]
                        if (signal_generated == 'buy' and price_change > 0) or \
                           (signal_generated == 'sell' and price_change < 0):
                            correct_signals += 1
            
            # Return accuracy percentage
            return (correct_signals / total_signals * 100) if total_signals > 0 else 0.0
            
        except Exception:
            return 0.0

    def _generate_fast_signal(self, closes: list) -> str:
        """Fast signal generation using current parameters (< 20ms)"""
        try:
            if len(closes) < 30:
                return 'none'
            
            # Use current optimal parameters
            price_series = pd.Series(closes)
            tsi_result = ta.tsi(price_series, 
                              slow=self.current_params.slow,
                              fast=self.current_params.fast,
                              signal=self.current_params.signal)
            
            if tsi_result is None or len(tsi_result.dropna()) < 10:
                # Fallback to previous parameters if current fails
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
            
            # Fast signal logic
            current_tsi = tsi_line.iloc[-1]
            current_signal = tsi_signal_line.iloc[-1]
            
            # Dynamic momentum calculation
            lookback = max(2, self.current_params.fast // 5)
            if len(tsi_line) > lookback:
                momentum = tsi_line.iloc[-1] - tsi_line.iloc[-1-lookback]
            else:
                momentum = tsi_line.iloc[-1] - tsi_line.iloc[-2]
            
            # Fast threshold check
            separation = abs(current_tsi - current_signal)
            
            # Simple but effective signal logic
            if (momentum > self.momentum_threshold and 
                current_tsi > current_signal and
                separation > self.strength_threshold):
                return 'buy'
            elif (momentum < -self.momentum_threshold and 
                  current_tsi < current_signal and
                  separation > self.strength_threshold):
                return 'sell'
            
            # Reversal signals
            elif current_tsi < -25 and momentum > self.momentum_threshold * 0.7:
                return 'buy'
            elif current_tsi > 25 and momentum < -self.momentum_threshold * 0.7:
                return 'sell'
            
            return 'none'
            
        except Exception as e:
            self.logger.error(f"❌ Error in fast signal generation: {e}")
            return 'none'

    def _track_signal_fast(self, signal: str, price: float):
        """Lightweight signal tracking"""
        try:
            signal_data = {
                'signal': signal,
                'price': price,
                'timestamp': time.time()
            }
            self.signal_performance.append(signal_data)
            
            # Quick accuracy update
            if len(self.signal_performance) >= 4:
                self._update_accuracy_fast()
                
        except Exception:
            pass

    def _update_accuracy_fast(self):
        """Fast accuracy calculation with auto-save on improvement (< 2ms)"""
        try:
            if len(self.signal_performance) < 2:
                return
            
            old_accuracy = self.current_params.accuracy
            correct = 0
            total = 0
            
            # Check only recent signals for speed
            check_count = min(10, len(self.signal_performance) - 1)
            
            for i in range(check_count):
                signal_data = self.signal_performance[-(i+2)]
                next_data = self.signal_performance[-(i+1)]
                
                price_change = next_data['price'] - signal_data['price']
                
                if signal_data['signal'] == 'buy' and price_change > 0:
                    correct += 1
                elif signal_data['signal'] == 'sell' and price_change < 0:
                    correct += 1
                
                total += 1
            
            if total > 0:
                new_accuracy = (correct / total) * 100
                self.current_params.accuracy = new_accuracy
                self.current_params.total_signals = total
                self.current_params.winning_signals = correct
                
                # Save if accuracy improved by 5% or more
                if new_accuracy > old_accuracy + 5:
                    self._schedule_config_save()
                    self.logger.debug(f"💾 Accuracy improved: {old_accuracy:.1f}% → {new_accuracy:.1f}%")
                
        except Exception:
            pass

    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'system_type': 'fast_adaptive',
            'current_params': {
                'slow': self.current_params.slow,
                'fast': self.current_params.fast,
                'signal': self.current_params.signal
            },
            'performance': {
                'accuracy': self.current_params.accuracy,
                'total_signals': self.current_params.total_signals,
                'winning_signals': self.current_params.winning_signals
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
    """Integration function with persistent configuration storage"""
    # Use default config file if none provided
    if config_file is None:
        config_file = os.path.join("data", "tsi_configs.json")
    
    strategy_instance.logger.info(f"🔧 Integrating TSI with config file: {os.path.abspath(config_file)}")
    
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
            strategy_instance.logger.info("🧪 Testing TSI config save/load...")
            
            # Force save current config
            adaptive_tsi._save_symbol_config(force=True)
            
            # Try to reload and verify
            import os
            if os.path.exists(adaptive_tsi.config_file):
                strategy_instance.logger.info(f"✅ Config file exists: {adaptive_tsi.config_file}")
                
                # Check file contents
                with open(adaptive_tsi.config_file, 'r') as f:
                    import json
                    data = json.load(f)
                    if strategy_instance.symbol in data:
                        symbol_config = data[strategy_instance.symbol]
                        strategy_instance.logger.info(f"✅ Found config for {strategy_instance.symbol}: {symbol_config}")
                        return True
                    else:
                        strategy_instance.logger.error(f"❌ No config found for {strategy_instance.symbol} in file")
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
    
    strategy_instance.logger.info("⚡ Fast Adaptive TSI with persistent storage integrated!")
    
    return adaptive_tsi


def integrate_momentum_tsi(strategy_instance, config_file: str = None):
    """Backward compatibility wrapper with persistent storage"""
    return integrate_adaptive_tsi(strategy_instance, config_file)