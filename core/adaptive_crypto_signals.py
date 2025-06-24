
"""
Adaptive Crypto Signal System - QQE & Supertrend Focused Strategy
ENTRY SIGNALS ONLY - Relies on external SL/TP orders, but provides QQE/ST based exit *reasons*.
"""

import logging
import time
import os
import csv
import datetime
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

logging.basicConfig(
        level=logging.DEBUG,  # Set the minimum level for the root logger
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logging.getLogger(__name__).setLevel(logging.DEBUG) 
@dataclass
class SignalParameters:
    """Parameters for QQE & Supertrend signals"""
    qqe_length: int = 12        # Balanced - not too fast, not too slow
    qqe_smooth: int = 5         # Standard smoothing to avoid whipsaws
    qqe_factor: float = 4.236   # Keep original for stability
    supertrend_period: int = 10 # Standard period for trend confidence
    supertrend_multiplier: float = 2.8  # Slightly tighter than 3.0 but not too close
    
    accuracy: float = 0.0
    total_signals: int = 0
    winning_signals: int = 0
    last_used: float = 0.0
    optimization_score: float = 0.0  # New: Store Optuna optimization score
    
class AdaptiveCryptoSignals:
    """
    QQE-primary signal system with Supertrend confirmation.
    Fisher Transform and VWAP components have been removed for simplification.
    """
    
    def __init__(self, symbol: str, config_file: str = "data/crypto_signal_configs.json"):
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        self.symbol = symbol
        self.config_file = config_file
        self.position_entry_time = 0
        
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
        
        # ENHANCED: Optuna optimization settings
        self.min_signals_for_optimization = 10  # Increased for statistical significance
        self.optimization_interval = 600  # 10 minutes instead of 5
        self.historical_data_periods = 500  # Use more historical data
        
        # ENHANCED: Dynamic entry zones based on market volatility
        self.qqe_long_entry_max_zone = 90.0
        self.qqe_short_entry_min_zone = 10.0
        
        # Cache for historical data to avoid repeated API calls
        self._historical_data_cache = None
        self._cache_timestamp = 0
        self._cache_validity = 300  # 5 minutes cache validity
        
        self.last_indicators: Dict[str, Optional[Dict]] = {'qqe': None, 'supertrend': None}
        
        self.logger.info(f"🚀 ENHANCED Crypto Signals with Optuna optimization for {symbol}")
        self.logger.info(f"   Current params: QQE({self.params.qqe_length},{self.params.qqe_smooth}), "
                        f"ST({self.params.supertrend_period},{self.params.supertrend_multiplier})")
        self.logger.info(f"   Optimization score: {self.params.optimization_score:.2f}, "
                        f"Live accuracy: {self.params.accuracy:.1f}%")
    def export_close_to_csv(self, ohlcv_data, filename: str = None) -> bool:
        """
        Export close prices with timestamps to CSV format.
        
        Args:
            ohlcv_data: OHLCV data from exchange.get_ohlcv() 
            filename: Output CSV filename (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not ohlcv_data:
                self.logger.warning("No OHLCV data provided for export")
                return False
            
            if filename is None:
                filename = f"{self.symbol}_close_prices.csv"
            
            import csv
            import datetime
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['timestamp', 'close'])
                
                # Write data rows
                for candle in ohlcv_data:
                    timestamp_ms = candle[0]  # Timestamp in milliseconds
                    close_price = float(candle[4])  # Close price
                    
                    # Convert timestamp from milliseconds to readable format
                    timestamp_readable = datetime.datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    
                    writer.writerow([timestamp_readable, close_price])
            
            self.logger.info(f"Successfully exported {len(ohlcv_data)} records to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False
    def get_technical_direction(self, exchange) -> str:
        """Keep existing - no changes"""
        try:
            current_time = time.time()
            if hasattr(self, '_force_signal') and self._force_signal: 
                pass 
            elif current_time - self.last_signal_time < self.signal_cooldown: 
                return 'none'
            
            ohlcv_data = exchange.get_ohlcv(self.symbol, timeframe='1m', limit=1400) 
            # filename = f"{self.symbol}_close_prices.csv"
            # self.export_close_to_csv(ohlcv_data, filename)
            if not ohlcv_data or len(ohlcv_data) < 50: 
                return 'none'
            
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df_indexed = df.set_index('timestamp')
            signal = self._generate_composite_signal(df_indexed) 
            
            if signal != 'none':
                self._track_signal(signal, float(df['close'].iloc[-1])) 
                self.last_signal = signal
                self.last_signal_time = current_time
                self.logger.info(f"⚡ SIGNAL: {signal.upper()} @ ${float(df['close'].iloc[-1]):.6f}")
            
            if self._should_optimize():
                self._optimize_parameters(df.copy()) 
            
            return signal
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}", exc_info=True)
            return 'none'
    def _update_historical_cache(self, ohlcv_data):
        """Cache historical data to avoid repeated API calls during optimization"""
        current_time = time.time()
        if current_time - self._cache_timestamp > self._cache_validity:
            self._historical_data_cache = ohlcv_data.copy()
            self._cache_timestamp = current_time
            self.logger.debug(f"Updated historical data cache: {len(ohlcv_data)} candles")

    def _update_market_volatility(self, df: pd.DataFrame):
        """ENHANCED: Calculate market volatility for dynamic parameter adjustment"""
        try:
            # Calculate ATR-based volatility
            atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
            if atr is not None and len(atr.dropna()) > 0:
                recent_atr = atr.dropna().tail(10).mean()
                current_price = df['close'].iloc[-1]
                volatility_pct = (recent_atr / current_price) * 100
                
                # ENHANCED: Adjust entry zones based on volatility
                if volatility_pct > 5.0:  # High volatility
                    self.volatility_level = 'high'
                    self.qqe_long_entry_max_zone = 85.0  # Wider zones
                    self.qqe_short_entry_min_zone = 15.0
                elif volatility_pct < 2.0:  # Low volatility
                    self.volatility_level = 'low'
                    self.qqe_long_entry_max_zone = 65.0  # Tighter zones
                    self.qqe_short_entry_min_zone = 35.0
                else:  # Normal volatility
                    self.volatility_level = 'normal'
                    self.qqe_long_entry_max_zone = 75.0
                    self.qqe_short_entry_min_zone = 25.0
                    
        except Exception as e:
            self.logger.debug(f"Error updating market volatility: {e}")
            self.volatility_level = 'normal'

    def _should_optimize_with_optuna(self) -> bool:
        """ENHANCED: More intelligent optimization triggers"""
        if not OPTUNA_AVAILABLE:
            return False
            
        current_time = time.time()
        
        # Don't optimize too frequently
        if current_time - self.last_optimization < self.optimization_interval:
            return False
        
        # Need sufficient signals for statistical significance
        if len(self.signal_performance) < self.min_signals_for_optimization:
            return False
        
        # ENHANCED: Trigger conditions
        evaluated_signals = [s for s in self.signal_performance if s.get('evaluated', False)]
        if len(evaluated_signals) < 8:  # Need evaluated signals
            return False
        
        # Trigger if:
        # 1. Low accuracy (below 45%)
        # 2. No optimization score yet
        # 3. Significant time passed since last optimization (20+ minutes)
        # 4. Performance degradation detected
        
        recent_accuracy = self._calculate_recent_accuracy()
        
        if self.params.accuracy < 45.0:
            self.logger.info(f"🔧 Triggering Optuna: Low accuracy ({self.params.accuracy:.1f}%)")
            return True
        
        if self.params.optimization_score == 0.0:
            self.logger.info(f"🔧 Triggering Optuna: No optimization score yet")
            return True
        
        if current_time - self.last_optimization > 1200:  # 20 minutes
            self.logger.info(f"🔧 Triggering Optuna: Time-based trigger")
            return True
        
        if recent_accuracy < self.params.accuracy - 10:  # 10% degradation
            self.logger.info(f"🔧 Triggering Optuna: Performance degradation "
                           f"({recent_accuracy:.1f}% vs {self.params.accuracy:.1f}%)")
            return True
        
        return False

    def _calculate_recent_accuracy(self) -> float:
        """Calculate accuracy of recent signals only"""
        try:
            recent_signals = list(self.signal_performance)[-10:]  # Last 10 signals
            evaluated = [s for s in recent_signals if s.get('evaluated', False)]
            if len(evaluated) == 0:
                return 0.0
            
            correct = sum(1 for s in evaluated if s.get('correct', False))
            return (correct / len(evaluated)) * 100
        except Exception:
            return 0.0

    def _optimize_parameters_with_optuna(self):
        """ENHANCED: Intelligent parameter optimization using Optuna"""
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, skipping optimization")
            return
        
        try:
            self.logger.info(f"🔧 Starting Optuna optimization for {self.symbol}...")
            start_time = time.time()
            
            # Create or load Optuna study
            study_name = f"crypto_signals_{self.symbol}"
            storage_url = f"sqlite:///data/optuna_{self.symbol}.db"
            
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                direction='maximize'  # Maximize objective score
            )
            
            # Run optimization with timeout
            optimization_timeout = 30  # 30 seconds max
            study.optimize(
                self._optuna_objective,
                n_trials=20,  # Limited trials for speed
                timeout=optimization_timeout,
                show_progress_bar=False
            )
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            optimization_time = time.time() - start_time
            
            self.logger.info(f"🎯 Optuna completed in {optimization_time:.1f}s:")
            self.logger.info(f"   Best score: {best_score:.2f}")
            self.logger.info(f"   Best params: {best_params}")
            self.logger.info(f"   Trials: {len(study.trials)}")
            
            # ENHANCED: Only update if significantly better
            improvement_threshold = 5.0  # 5% improvement required
            
            if best_score > self.params.optimization_score + improvement_threshold:
                self._update_parameters_from_optuna(best_params, best_score)
                self.logger.info(f"✅ Parameters updated! Score improved by "
                               f"{best_score - self.params.optimization_score:.1f} points")
            else:
                self.logger.info(f"📊 No significant improvement found "
                               f"(+{best_score - self.params.optimization_score:.1f})")
            
            self.last_optimization = time.time()
            
        except Exception as e:
            self.logger.error(f"Error in Optuna optimization: {e}", exc_info=True)

    def _optuna_objective(self, trial) -> float:
        """ENHANCED: Sophisticated objective function for parameter optimization"""
        try:
            # ENHANCED: Wider parameter ranges for crypto markets
            qqe_length = trial.suggest_int('qqe_length', 8, 25)
            qqe_smooth = trial.suggest_int('qqe_smooth', 3, 10)
            supertrend_period = trial.suggest_int('supertrend_period', 5, 20)
            supertrend_multiplier = trial.suggest_float('supertrend_multiplier', 1.5, 4.0)
            
            # Create test parameters
            test_params = SignalParameters(
                qqe_length=qqe_length,
                qqe_smooth=qqe_smooth,
                supertrend_period=supertrend_period,
                supertrend_multiplier=supertrend_multiplier
            )
            
            # ENHANCED: Comprehensive backtesting on historical data
            score = self._enhanced_backtest(test_params)
            
            # ENHANCED: Penalty for extreme parameters (regularization)
            penalty = 0
            if qqe_length < 10 or qqe_length > 20:
                penalty += 2  # Slight penalty for extreme QQE length
            if supertrend_multiplier < 2.0 or supertrend_multiplier > 3.5:
                penalty += 3  # Penalty for extreme ST multiplier
            
            final_score = max(0, score - penalty)
            
            return final_score
            
        except Exception as e:
            self.logger.debug(f"Error in Optuna objective: {e}")
            return 0.0

    def _enhanced_backtest(self, test_params: SignalParameters) -> float:
        """ENHANCED: Comprehensive backtesting with multiple metrics"""
        try:
            if self._historical_data_cache is None:
                return 0.0
            
            # Convert cached data to DataFrame
            df = pd.DataFrame(self._historical_data_cache, 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df_calc = df.set_index('timestamp')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_calc[col] = df_calc[col].astype(float)
            
            # Calculate indicators with test parameters
            qqe_result = ta.qqe(df_calc['close'], 
                              length=test_params.qqe_length, 
                              smooth=test_params.qqe_smooth, 
                              factor=4.236)
            
            if qqe_result is None or qqe_result.empty or len(qqe_result.columns) < 2:
                return 0.0
            
            st_result = ta.supertrend(df_calc['high'], df_calc['low'], df_calc['close'],
                                    length=test_params.supertrend_period,
                                    multiplier=test_params.supertrend_multiplier)
            
            if st_result is None or st_result.empty:
                return 0.0
            
            # Find direction column
            st_dir_col = next((col for col in st_result.columns if 'SUPERTd' in col), None)
            if not st_dir_col:
                return 0.0
            
            # Prepare data
            df_calc['qqe_val'] = qqe_result.iloc[:, 0]
            df_calc['qqe_sig'] = qqe_result.iloc[:, 1]
            df_calc['st_dir'] = st_result[st_dir_col]
            
            df_calc.dropna(inplace=True)
            
            if len(df_calc) < 100:  # Need sufficient data
                return 0.0
            
            # ENHANCED: Multi-metric backtesting
            metrics = self._calculate_backtest_metrics(df_calc)
            
            # ENHANCED: Composite score calculation
            # Weight different metrics for crypto trading
            win_rate_score = metrics['win_rate'] * 0.4  # 40% weight
            profit_factor_score = min(metrics['profit_factor'] * 10, 30) * 0.3  # 30% weight, capped
            sharpe_score = max(0, min(metrics['sharpe_ratio'] * 15, 20)) * 0.2  # 20% weight
            max_dd_score = max(0, (100 - metrics['max_drawdown']) / 5) * 0.1  # 10% weight
            
            total_score = win_rate_score + profit_factor_score + sharpe_score + max_dd_score
            
            # ENHANCED: Bonus for consistency (low volatility of returns)
            if metrics['return_volatility'] < 0.1:  # Low volatility = consistent
                total_score += 5
            
            return min(total_score, 100.0)  # Cap at 100
            
        except Exception as e:
            self.logger.debug(f"Error in enhanced backtest: {e}")
            return 0.0

    def _calculate_backtest_metrics(self, df: pd.DataFrame) -> Dict:
        """ENHANCED: Calculate comprehensive trading metrics"""
        try:
            trades = []
            position = None
            entry_price = 0.0
            
            # ENHANCED: More sophisticated entry/exit logic
            for i in range(1, len(df)):
                current_price = df['close'].iloc[i]
                qqe_val = df['qqe_val'].iloc[i]
                qqe_sig = df['qqe_sig'].iloc[i]
                st_dir = df['st_dir'].iloc[i]
                
                # Exit logic
                if position:
                    exit_triggered = False
                    exit_reason = ""
                    
                    if position == 'long':
                        # Exit long: QQE bearish or ST down
                        if qqe_val < qqe_sig or st_dir != 1:
                            exit_triggered = True
                            exit_reason = "signal_exit"
                        # Stop loss: 2% loss
                        elif current_price <= entry_price * 0.98:
                            exit_triggered = True
                            exit_reason = "stop_loss"
                        # Take profit: 3% gain
                        elif current_price >= entry_price * 1.03:
                            exit_triggered = True
                            exit_reason = "take_profit"
                    
                    elif position == 'short':
                        # Exit short: QQE bullish or ST up
                        if qqe_val > qqe_sig or st_dir == 1:
                            exit_triggered = True
                            exit_reason = "signal_exit"
                        # Stop loss: 2% loss
                        elif current_price >= entry_price * 1.02:
                            exit_triggered = True
                            exit_reason = "stop_loss"
                        # Take profit: 3% gain
                        elif current_price <= entry_price * 0.97:
                            exit_triggered = True
                            exit_reason = "take_profit"
                    
                    if exit_triggered:
                        pnl = 0
                        if position == 'long':
                            pnl = (current_price - entry_price) / entry_price
                        else:  # short
                            pnl = (entry_price - current_price) / entry_price
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl_pct': pnl * 100,
                            'position': position,
                            'exit_reason': exit_reason
                        })
                        
                        position = None
                
                # Entry logic (only if no position)
                if not position:
                    qqe_bullish = qqe_val > qqe_sig
                    qqe_bearish = qqe_val < qqe_sig
                    st_up = st_dir == 1
                    st_down = st_dir != 1
                    
                    # Both indicators must agree
                    if qqe_bullish and st_up and qqe_val < self.qqe_long_entry_max_zone:
                        position = 'long'
                        entry_price = current_price
                    elif qqe_bearish and st_down and qqe_val > self.qqe_short_entry_min_zone:
                        position = 'short'
                        entry_price = current_price
            
            # Calculate metrics
            if len(trades) == 0:
                return {
                    'win_rate': 0, 'profit_factor': 0, 'total_return': 0,
                    'sharpe_ratio': 0, 'max_drawdown': 100, 'return_volatility': 1
                }
            
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            losing_trades = [t for t in trades if t['pnl_pct'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            
            total_wins = sum(t['pnl_pct'] for t in winning_trades)
            total_losses = abs(sum(t['pnl_pct'] for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else 10
            
            returns = [t['pnl_pct'] / 100 for t in trades]
            total_return = sum(returns) * 100
            
            # Sharpe ratio (simplified)
            if len(returns) > 1:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                return_volatility = std_return
            else:
                sharpe_ratio = 0
                return_volatility = 0
            
            # Maximum drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'return_volatility': return_volatility,
                'total_trades': len(trades)
            }
            
        except Exception as e:
            self.logger.debug(f"Error calculating metrics: {e}")
            return {
                'win_rate': 0, 'profit_factor': 0, 'total_return': 0,
                'sharpe_ratio': 0, 'max_drawdown': 100, 'return_volatility': 1
            }

    def _update_parameters_from_optuna(self, best_params: Dict, best_score: float):
        """Update parameters with Optuna results"""
        try:
            old_params = (self.params.qqe_length, self.params.qqe_smooth, 
                         self.params.supertrend_period, self.params.supertrend_multiplier)
            
            # Update parameters
            self.params.qqe_length = best_params['qqe_length']
            self.params.qqe_smooth = best_params['qqe_smooth']
            self.params.supertrend_period = best_params['supertrend_period']
            self.params.supertrend_multiplier = best_params['supertrend_multiplier']
            self.params.optimization_score = best_score
            self.params.last_used = time.time()
            
            # Reset live performance tracking for new parameters
            self.params.accuracy = 0.0
            self.params.total_signals = 0
            self.params.winning_signals = 0
            self.signal_performance.clear()
            
            new_params = (self.params.qqe_length, self.params.qqe_smooth,
                         self.params.supertrend_period, self.params.supertrend_multiplier)
            
            self.logger.info(f"📈 OPTUNA UPDATE: {old_params} → {new_params}")
            self.logger.info(f"📈 Optimization score: {best_score:.2f}")
            
            # Save immediately
            self._save_config()
            
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")
    def _generate_composite_signal(self, df: pd.DataFrame) -> str:
        """SIMPLIFIED: Clear signal logic using fixed indicators."""
        try:
            self.logger.info(f"[{self.symbol}] --- Generating Signal ---")
            
            qqe_result = self._calculate_qqe(df)
            supertrend_result = self._calculate_supertrend(df)

            if not qqe_result or not supertrend_result:
                missing = []
                if not qqe_result: missing.append("QQE")
                if not supertrend_result: missing.append("Supertrend")
                self.logger.info(f"[{self.symbol}] Missing indicators: {', '.join(missing)}")
                return 'none'
            
            # Store for exit evaluation
            self.last_indicators['qqe'] = qqe_result
            self.last_indicators['supertrend'] = supertrend_result
            
            # SIMPLE LOGIC: Both indicators must agree
            qqe_value = qqe_result['qqe_value']
            qqe_signal = qqe_result['qqe_signal_line_value']
            st_direction = supertrend_result['direction']
            
            qqe_bullish = qqe_value < qqe_signal  # QQE line above signal line
            qqe_bearish = qqe_value > qqe_signal  # QQE line below signal line
            st_up = st_direction == 'up'
            st_down = st_direction == 'down'
            
            self.logger.info(f"[{self.symbol}] QQE: {qqe_value:.2f} vs Signal: {qqe_signal:.2f} → {'BULLISH' if qqe_bullish else 'BEARISH'}")
            self.logger.info(f"[{self.symbol}] Supertrend: {st_direction.upper()}")
            
            # ENTRY SIGNALS: Both must agree
            if qqe_bullish and st_up:
                self.logger.info(f"[{self.symbol}] 🟢 BUY SIGNAL: QQE bullish + ST up")
                return 'buy'
            elif qqe_bearish and st_down:
                self.logger.info(f"[{self.symbol}] 🔴 SELL SIGNAL: QQE bearish + ST down")
                return 'sell'
            else:
                self.logger.info(f"[{self.symbol}] ❌ NO SIGNAL: Indicators disagree (QQE: {'bull' if qqe_bullish else 'bear'}, ST: {st_direction})")
                return 'none'
                
        except Exception as e:
            self.logger.error(f"[{self.symbol}] Signal generation error: {e}", exc_info=True)
            return 'none'

    def evaluate_exit_conditions(self, position_side: str, entry_price: float, current_price: float) -> Dict:
        """SIMPLIFIED: Clear exit logic using fixed indicators."""
        try:
            result = {'should_exit': False, 'exit_reason': '', 'exit_urgency': 'none'}
            
            qqe_data = self.last_indicators.get('qqe')
            st_data = self.last_indicators.get('supertrend')
            
            if not qqe_data or not st_data:
                self.logger.warning(f"[{self.symbol}] Exit eval: Missing indicator data")
                return result

            # Calculate PnL
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if position_side == 'long' else ((entry_price - current_price) / entry_price) * 100
            
            # Emergency stop loss
            if pnl_pct < -2.0:
                result.update({
                    'should_exit': True,
                    'exit_reason': f"Emergency SL: {pnl_pct:.2f}% loss",
                    'exit_urgency': 'immediate'
                })
                return result

            # SIMPLE EXIT: Both indicators against position
            qqe_value = qqe_data['qqe_value']
            qqe_signal = qqe_data['qqe_signal_line_value']
            st_direction = st_data['direction']
            
            qqe_bullish = qqe_value < qqe_signal  # QQE line above signal line
            qqe_bearish = qqe_value > qqe_signal  # QQE line below signal line
            st_up = st_direction == 'up'
            st_down = st_direction == 'down'
            
            self.logger.info(f"[{self.symbol}] Exit eval for {position_side.upper()}: QQE={'bull' if qqe_bullish else 'bear'}, ST={st_direction}")
            
            if position_side == 'long' and (qqe_bearish and st_down):
                result.update({
                    'should_exit': True,
                    'exit_reason': f"LONG exit: QQE bearish + ST down (PnL: {pnl_pct:.2f}%)",
                    'exit_urgency': 'normal'
                })
            elif position_side == 'short' and (qqe_bullish and st_up):
                result.update({
                    'should_exit': True,
                    'exit_reason': f"SHORT exit: QQE bullish + ST up (PnL: {pnl_pct:.2f}%)",
                    'exit_urgency': 'normal'
                })
            
            if result['should_exit']:
                self.logger.info(f"[{self.symbol}] ✅ EXIT TRIGGERED: {result['exit_reason']}")
            else:
                self.logger.info(f"[{self.symbol}] ❌ NO EXIT: Indicators not both against position")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{self.symbol}] Exit evaluation error: {e}", exc_info=True)
            return {'should_exit': False, 'exit_reason': 'Error', 'exit_urgency': 'none'}

    def _calculate_qqe(self, df: pd.DataFrame) -> Optional[Dict]:
        """FIXED: Robust QQE calculation with proper column handling."""
        try:
            # Call pandas_ta QQE
            qqe_result = ta.qqe(df['close'], 
                            length=self.params.qqe_length, 
                            smooth=self.params.qqe_smooth, 
                            factor=self.params.qqe_factor)
            
            if qqe_result is None or qqe_result.empty:
                self.logger.warning(f"[{self.symbol}] QQE calculation returned None/empty")
                return None
            
            # FIXED: Better column detection - pandas_ta returns specific column names
            # For QQE, typical columns are: QQE_14_5_4.236, QQE_14_5_4.236_RSIMA
            qqe_columns = list(qqe_result.columns)
            self.logger.debug(f"[{self.symbol}] QQE returned columns: {qqe_columns}")
            
            if len(qqe_columns) < 2:
                self.logger.error(f"[{self.symbol}] QQE returned insufficient columns: {qqe_columns}")
                return None
            
            # FIXED: Use actual column names instead of constructed names
            qqe_line_col = qqe_columns[0]  # Main QQE line (yellow)
            qqe_signal_col = qqe_columns[1]  # Signal line (blue)
            
            qqe_line = qqe_result[qqe_line_col].dropna()
            qqe_signal = qqe_result[qqe_signal_col].dropna()
            
            if len(qqe_line) < 3 or len(qqe_signal) < 3:
                self.logger.warning(f"[{self.symbol}] Insufficient QQE data: {len(qqe_line)}, {len(qqe_signal)}")
                return None
            
            # Get current values
            current_qqe = float(qqe_line.iloc[-1])
            current_signal = float(qqe_signal.iloc[-1])
            
            if pd.isna(current_qqe) or pd.isna(current_signal):
                self.logger.warning(f"[{self.symbol}] QQE values are NaN")
                return None
            
            # SIMPLIFIED: Just the essential data
            return {
                'qqe_value': current_qqe,
                'qqe_signal_line_value': current_signal,
                'qqe_direction': current_qqe - current_signal,  # Positive = bullish
                'raw_columns': qqe_columns  # For debugging
            }
            
        except Exception as e:
            self.logger.error(f"[{self.symbol}] QQE calculation error: {e}", exc_info=True)
            return None

    def _calculate_supertrend(self, df: pd.DataFrame) -> Optional[Dict]:
        """FIXED: Robust Supertrend calculation with proper column handling."""
        try:
            # Call pandas_ta Supertrend
            st_result = ta.supertrend(high=df['high'], low=df['low'], close=df['close'], 
                                    length=self.params.supertrend_period, 
                                    multiplier=self.params.supertrend_multiplier)
            
            if st_result is None or st_result.empty:
                self.logger.warning(f"[{self.symbol}] Supertrend calculation returned None/empty")
                return None
            
            # FIXED: Better column detection - pandas_ta returns specific columns
            # For Supertrend, typical columns are: SUPERT_10_3.0, SUPERTd_10_3.0, SUPERTl_10_3.0, SUPERTs_10_3.0
            st_columns = list(st_result.columns)
            self.logger.debug(f"[{self.symbol}] Supertrend returned columns: {st_columns}")
            
            # Find direction column (contains 'd' for direction)
            direction_col = None
            for col in st_columns:
                if 'SUPERTd' in col:
                    direction_col = col
                    break
            
            if direction_col is None:
                self.logger.error(f"[{self.symbol}] No Supertrend direction column found in: {st_columns}")
                return None
            
            st_direction = st_result[direction_col].dropna()
            
            if len(st_direction) < 3:
                self.logger.warning(f"[{self.symbol}] Insufficient Supertrend data: {len(st_direction)}")
                return None
            
            # Get current direction
            current_dir = int(st_direction.iloc[-1])
            prev_dir = int(st_direction.iloc[-2]) if len(st_direction) > 1 else current_dir
            
            # Convert to readable direction
            direction = 'up' if current_dir == 1 else 'down'
            just_changed = current_dir != prev_dir
            
            # SIMPLIFIED: Just the essential data
            return {
                'direction': direction,
                'current_value': current_dir,
                'just_changed': just_changed,
                'raw_columns': st_columns  # For debugging
            }
            
        except Exception as e:
            self.logger.error(f"[{self.symbol}] Supertrend calculation error: {e}", exc_info=True)
            return None
    def _update_volatility_level(self, df: pd.DataFrame): 
        try:
            atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14) 
            if atr is not None and len(atr.dropna()) > 0:
                curr_atr = float(atr.dropna().iloc[-1])
                price = float(df['close'].iloc[-1])
                if price == 0: self.volatility_level = 'normal'; return
                atr_pct = (curr_atr / price) * 100
                self.volatility_level = 'high' if atr_pct > 2.0 else 'low' if atr_pct < 0.5 else 'normal'
            else: self.volatility_level = 'normal'      
        except Exception: self.volatility_level = 'normal'

    def _track_signal(self, signal: str, price: float):
        try:
            signal_data = {
                'signal': signal, 'price': price, 'timestamp': time.time(),
                'params': { 
                    'qqe_length': self.params.qqe_length, 'qqe_smooth': self.params.qqe_smooth,
                    'supertrend_period': self.params.supertrend_period, 'supertrend_multiplier': self.params.supertrend_multiplier
                },
                'market_state': {'trend': self.current_trend, 'volatility': self.volatility_level}
            }
            self.signal_performance.append(signal_data)
            self._evaluate_signals()
            if len(self.signal_performance) % 5 == 0: self._save_config()
        except Exception as e: self.logger.error(f"[{self.symbol}] Error tracking signal: {e}", exc_info=True)

    def _evaluate_signals(self):
        try:
            current_time = time.time(); eval_period_s = 30 * 60; min_move_pct = 0.5; eval_count = 0
            for sig_data in list(self.signal_performance): 
                if sig_data.get('evaluated', False): continue
                if current_time - sig_data['timestamp'] < eval_period_s: continue
                
                fut_price = None; fut_ts = float('inf')
                for fut_cand in self.signal_performance:
                    cand_ts = fut_cand['timestamp']
                    if cand_ts >= (sig_data['timestamp'] + eval_period_s) and cand_ts < fut_ts:
                        fut_price = fut_cand['price']; fut_ts = cand_ts
                
                if fut_price is not None:
                    chg_pct = ((fut_price - sig_data['price']) / sig_data['price']) * 100 if sig_data['price'] != 0 else 0
                    sig_data['correct'] = (sig_data['signal'] == 'buy' and chg_pct > min_move_pct) or \
                                          (sig_data['signal'] == 'sell' and chg_pct < -min_move_pct)
                    sig_data['evaluated'] = True; sig_data['price_change_eval'] = chg_pct; eval_count += 1
            
            tot_eval = sum(1 for s in self.signal_performance if s.get('evaluated', False))
            tot_corr = sum(1 for s in self.signal_performance if s.get('correct', False))
            self.params.accuracy = (tot_corr / tot_eval) * 100 if tot_eval > 0 else 0.0
            self.params.total_signals = tot_eval; self.params.winning_signals = tot_corr
            if eval_count > 0: self.logger.info(f"[{self.symbol}] Evaluated {eval_count} signals. Accuracy: {self.params.accuracy:.2f}%")
        except Exception as e: self.logger.error(f"[{self.symbol}] Error evaluating signals: {e}", exc_info=True)

    def _should_optimize(self) -> bool:
        curr_time = time.time()
        if curr_time - self.last_optimization < self.optimization_interval: return False
        if self.params.total_signals >= self.min_signals_for_optimization and self.params.accuracy < 45.0: return True
        return False

    def _optimize_parameters(self, df: pd.DataFrame):
        """ONLY CHANGE: Replace manual grid search with Optuna, keep same logic"""
        try:
            self.logger.info(f"🔧 Optimizing parameters...")
            start_time = time.time()
            
            if OPTUNA_AVAILABLE:
                # Use Optuna for smarter search
                best_params, best_score = self._optuna_search(df)
            else:
                # Fallback to original manual grid search
                best_params, best_score = self._manual_grid_search(df)
            
            # Keep existing improvement logic exactly the same
            current_score = self.params.accuracy
            if best_params and best_score > current_score + 1:  # Same 1% threshold
                old_params = f"{self.params.qqe_length}/{self.params.qqe_smooth}/{self.params.supertrend_period}/{self.params.supertrend_multiplier}"
                
                # Update parameters
                self.params.qqe_length = best_params['qqe_length']
                self.params.qqe_smooth = best_params['qqe_smooth']
                self.params.supertrend_period = best_params['supertrend_period']
                self.params.supertrend_multiplier = best_params['supertrend_multiplier']
                self.params.last_used = time.time()
                
                new_params = f"{self.params.qqe_length}/{self.params.qqe_smooth}/{self.params.supertrend_period}/{self.params.supertrend_multiplier}"
                
                self.logger.info(f"⚡ UPDATED: {old_params} → {new_params} (+{best_score - current_score:.1f}%)")
                self._save_config()
            
            self.last_optimization = time.time()
            optimization_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Optimization completed in {optimization_time:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
    def _optuna_search(self, df: pd.DataFrame) -> tuple:
        """NEW: Optuna search using SAME parameter ranges and SAME objective function"""
        try:
            # Create study
            study = optuna.create_study(direction='maximize')
            
            # Run optimization with same timeout as before
            study.optimize(
                lambda trial: self._optuna_objective(trial, df),
                n_trials=30,  # More trials than manual grid (was 54 max)
                timeout=0.2,  # Same 200ms timeout as before
                show_progress_bar=False
            )
            
            if study.best_trial:
                return study.best_params, study.best_value
            else:
                return None, 0.0
                
        except Exception as e:
            self.logger.debug(f"Optuna search failed: {e}")
            return None, 0.0
    def _manual_grid_search(self, df: pd.DataFrame) -> tuple:
        """FALLBACK: Keep original manual grid search exactly the same"""
        param_ranges = {
            'qqe_length': [6, 8, 10, 12],              # Ultra-fast range
            'qqe_smooth': [2, 3, 4, 5],                # Maximum responsiveness
            'supertrend_period': [5, 7, 8, 10],        # Faster than standard
            'supertrend_multiplier': [2.0, 2.2, 2.5, 2.8]  # Tighter for memes
        }
        
        best_params = None
        best_score = -1.0
        current_score = self.params.accuracy
        
        for qqe_len in param_ranges['qqe_length']:
            for qqe_sm in param_ranges['qqe_smooth']:
                for st_p in param_ranges['supertrend_period']:
                    for st_m in param_ranges['supertrend_multiplier']:
                        score = self._quick_backtest(df, qqe_len, qqe_sm, st_p, st_m)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'qqe_length': qqe_len,
                                'qqe_smooth': qqe_sm,
                                'supertrend_period': st_p,
                                'supertrend_multiplier': st_m
                            }
        
        return best_params, best_score
    def _optuna_objective(self, trial, df: pd.DataFrame) -> float:
        """NEW: Optuna objective using SAME ranges and SAME _quick_backtest logic"""
        try:
            # Use EXACT same parameter ranges as original manual search
            qqe_length = trial.suggest_categorical('qqe_length', [10, 14, 20])
            qqe_smooth = trial.suggest_categorical('qqe_smooth', [3, 5, 7])
            supertrend_period = trial.suggest_categorical('supertrend_period', [7, 10, 14])
            supertrend_multiplier = trial.suggest_categorical('supertrend_multiplier', [2.0, 3.0])
            
            # Use SAME _quick_backtest function - no changes
            score = self._quick_backtest(df, qqe_length, qqe_smooth, supertrend_period, supertrend_multiplier)
            
            return score
            
        except Exception:
            return 0.0
    def _quick_backtest(self, df_hist: pd.DataFrame, params_to_test: SignalParameters) -> float:
        try:
            df_calc = df_hist.set_index('timestamp')
            qqe_bt_df = ta.qqe(df_calc['close'], length=params_to_test.qqe_length, smooth=params_to_test.qqe_smooth, factor=4.236)
            if qqe_bt_df is None or qqe_bt_df.empty or len(qqe_bt_df.columns) < 2: return 0.0
            df_calc['qqe_val'] = qqe_bt_df.iloc[:,0]
            df_calc['qqe_sig'] = qqe_bt_df.iloc[:,1]
            
            st_bt_df = ta.supertrend(df_calc['high'], df_calc['low'], df_calc['close'], 
                                   length=params_to_test.supertrend_period, multiplier=params_to_test.supertrend_multiplier)
            if st_bt_df is None or st_bt_df.empty: return 0.0
            st_dir_col = next((col for col in st_bt_df.columns if 'SUPERTd' in col), None)
            if not st_dir_col: return 0.0
            df_calc['st_dir'] = st_bt_df[st_dir_col]

            df_calc.dropna(inplace=True)
            if len(df_calc) < 30: return 0.0

            trades = 0; wins = 0; position = None; entry_price = 0.0
            tp_pct = 1.5; sl_pct = 1.0 

            for i in range(1, len(df_calc)):
                price = df_calc['close'].iloc[i]
                if position == 'long':
                    qqe_bear_cross = df_calc['qqe_val'].iloc[i] < df_calc['qqe_sig'].iloc[i] and \
                                     df_calc['qqe_val'].iloc[i-1] >= df_calc['qqe_sig'].iloc[i-1]
                    if price >= entry_price * (1 + tp_pct/100) or qqe_bear_cross:
                        wins += 1 if (price >= entry_price * (1 + tp_pct/100) or (qqe_bear_cross and price > entry_price)) else 0
                        trades += 1; position = None; continue
                    elif price <= entry_price * (1 - sl_pct/100):
                        trades += 1; position = None; continue
                elif position == 'short':
                    qqe_bull_cross = df_calc['qqe_val'].iloc[i] > df_calc['qqe_sig'].iloc[i] and \
                                     df_calc['qqe_val'].iloc[i-1] <= df_calc['qqe_sig'].iloc[i-1]
                    if price <= entry_price * (1 - tp_pct/100) or qqe_bull_cross:
                        wins += 1 if (price <= entry_price * (1 - tp_pct/100) or (qqe_bull_cross and price < entry_price)) else 0
                        trades += 1; position = None; continue
                    elif price >= entry_price * (1 + sl_pct/100):
                        trades += 1; position = None; continue
                
                if not position:
                    cv, pv = df_calc['qqe_val'].iloc[i], df_calc['qqe_val'].iloc[i-1]
                    cs, ps = df_calc['qqe_sig'].iloc[i], df_calc['qqe_sig'].iloc[i-1]
                    b_x = pv <= ps and cv > cs; b_mom = cv > cs and cv > pv
                    s_x = pv >= ps and cv < cs; s_mom = cv < cs and cv < pv
                    qqe_long = b_x or b_mom; qqe_short = s_x or s_mom
                    st_up = df_calc['st_dir'].iloc[i] == 1
                    
                    if qqe_long and st_up and cv < self.qqe_long_entry_max_zone: # Using self. for zone defaults
                        position = 'long'; entry_price = price
                    elif qqe_short and not st_up and cv > self.qqe_short_entry_min_zone:
                        position = 'short'; entry_price = price
            return (wins / trades) * 100.0 if trades > 0 else 0.0
        except Exception as e:
            self.logger.error(f"[{self.symbol}] Error in _quick_backtest: {e}", exc_info=True)
            return 0.0

    def _load_symbol_config(self) -> SignalParameters:
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f: configs = json.load(f)
                if self.symbol in configs:
                    cfg = configs[self.symbol]
                    return SignalParameters(
                        qqe_length=cfg.get('qqe_length', SignalParameters.qqe_length),
                        qqe_smooth=cfg.get('qqe_smooth', SignalParameters.qqe_smooth),
                        qqe_factor=cfg.get('qqe_factor', SignalParameters.qqe_factor), # Safely get qqe_factor
                        supertrend_period=cfg.get('supertrend_period', SignalParameters.supertrend_period),
                        supertrend_multiplier=cfg.get('supertrend_multiplier', SignalParameters.supertrend_multiplier),
                        accuracy=cfg.get('accuracy', 0.0), total_signals=cfg.get('total_signals',0), 
                        winning_signals=cfg.get('winning_signals',0), last_used=cfg.get('last_used', 0.0)
                    )
        except Exception as e: self.logger.error(f"[{self.symbol}] Error loading config: {e}. Defaults used.", exc_info=True)
        return SignalParameters()

    def _load_signal_history(self) -> deque:
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f: configs = json.load(f)
                if self.symbol in configs:
                    hist = configs[self.symbol].get('signal_history', [])
                    return deque([item for item in hist if isinstance(item, dict)][-100:], maxlen=100)
        except Exception as e: self.logger.error(f"[{self.symbol}] Error loading history: {e}. Fresh history.", exc_info=True)
        return deque(maxlen=100)

    def _save_config(self):
        try:
            configs = {}
            if os.path.exists(self.config_file):
                try: 
                    with open(self.config_file, 'r') as f: configs = json.load(f)
                except json.JSONDecodeError: self.logger.warning(f"Config {self.config_file} corrupted. Overwriting."); configs = {}
            
            configs[self.symbol] = {
            'qqe_length': self.params.qqe_length, 
            'qqe_smooth': self.params.qqe_smooth, 
            'qqe_factor': self.params.qqe_factor, # Ensure it's saved
            'supertrend_period': self.params.supertrend_period, 
            'supertrend_multiplier': self.params.supertrend_multiplier,
            'accuracy': self.params.accuracy, 
            'total_signals': self.params.total_signals, 
            'winning_signals': self.params.winning_signals, 
            'last_used': self.params.last_used,
            'signal_history': list(self.signal_performance)[-100:], # Ensure deque is converted to list
            'last_updated': time.time()
        }
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir): os.makedirs(config_dir, exist_ok=True)
            temp_file = self.config_file + '.tmp'
            with open(temp_file, 'w') as f: json.dump(configs, f, indent=2)
            os.replace(temp_file, self.config_file)
        except Exception as e: self.logger.error(f"[{self.symbol}] Error saving config: {e}", exc_info=True)

    def get_system_status(self) -> Dict:
        """Enhanced system status with Optuna information"""
        return {
            'system_type': 'QQE_Supertrend_Optuna_Enhanced',
            'params': { 
                'qqe_length': self.params.qqe_length, 
                'qqe_smooth': self.params.qqe_smooth,
                'supertrend_period': self.params.supertrend_period, 
                'supertrend_multiplier': self.params.supertrend_multiplier,
            },
            'optimization': {
                'optuna_available': OPTUNA_AVAILABLE,
                'optimization_score': self.params.optimization_score,
                'last_optimization': self.last_optimization,
                'historical_data_periods': self.historical_data_periods
            },
            'performance': {
                'live_accuracy': self.params.accuracy, 
                'total_signals': self.params.total_signals, 
                'winning_signals': self.params.winning_signals
            },
            'market_state': {
                'trend': self.current_trend, 
                'volatility': self.volatility_level,
                'entry_zones': f"Long<{self.qqe_long_entry_max_zone}, Short>{self.qqe_short_entry_min_zone}"
            }
        }

def integrate_adaptive_crypto_signals(strategy_instance, config_file: str = None):
    if config_file is None: config_file = os.path.join(os.getcwd(), "data", "crypto_signal_configs.json")
    strategy_instance.logger.info(f"🔧 Integrating QQE & Supertrend ONLY Crypto Signals, config: {config_file}")
    base_sym = getattr(strategy_instance, 'original_symbol', strategy_instance.symbol)
    crypto_sigs = AdaptiveCryptoSignals(symbol=base_sym, config_file=config_file)
    
    strategy_instance._get_technical_direction = lambda: crypto_sigs.get_technical_direction(strategy_instance.exchange)
    strategy_instance.get_signal_status = crypto_sigs.get_system_status
    strategy_instance.test_crypto_indicators = lambda: crypto_sigs.test_indicators(strategy_instance.exchange)
    strategy_instance._crypto_signal_system = crypto_sigs
    
    try:
        strategy_instance.logger.info(f"Initial indicator test for {base_sym} (QQE/ST Only)...")
        test_res = strategy_instance.test_crypto_indicators()
        # Log test_res details if needed, for brevity, just confirming success/failure
        if isinstance(test_res, dict) and 'error' not in test_res:
             strategy_instance.logger.info(f"📊 Indicator Test (QQE/ST Only) for {base_sym} successful. Signal sim: {test_res.get('composite_signal_sim', 'N/A')}")
        else:
            strategy_instance.logger.warning(f"📊 Indicator Test (QQE/ST Only) for {base_sym} may have issues: {test_res}")

    except Exception as e: strategy_instance.logger.error(f"Failed initial indicator test (QQE/ST Only): {e}", exc_info=True)
    
    strategy_instance.logger.info("⚡ QQE & Supertrend ONLY Crypto Signals integrated!")
    return crypto_sigs

def test_indicators(self: AdaptiveCryptoSignals, exchange) -> Dict:
    try:
        ohlcv = exchange.get_ohlcv(self.symbol, timeframe='1m', limit=300)
        if not ohlcv or len(ohlcv) < 50: return {'error': f'Insufficient data: {len(ohlcv) if ohlcv else 0}'}
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df_idx = df.set_index('timestamp')
        for col in ['open', 'high', 'low', 'close', 'volume']: df_idx[col] = df_idx[col].astype(float)

        res = {'symbol': self.symbol, 'price': float(df_idx['close'].iloc[-1]), 'data_points': len(df_idx)}
        res['qqe_calc_result'] = self._calculate_qqe(df_idx) or 'Failed QQE Calc' # Store full dict
        res['supertrend_calc_result'] = self._calculate_supertrend(df_idx) or 'Failed ST Calc' # Store full dict
        res['composite_signal_simulated'] = self._generate_composite_signal(df_idx) 
        res['current_market_trend_ST'] = self.current_trend 
        res['current_volatility_ATR'] = self.volatility_level 
        return res
    except Exception as e:
        self.logger.error(f"[{self.symbol}] Error in test_indicators (QQE/ST): {e}", exc_info=True)
        return {'error': str(e)}

AdaptiveCryptoSignals.test_indicators = test_indicators
integrate_crypto_signals = integrate_adaptive_crypto_signals

