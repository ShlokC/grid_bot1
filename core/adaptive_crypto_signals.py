"""
Adaptive Crypto Signal System - Integrated with WinRateOptimizer from test_strategies.py
Dynamic Parameter Optimization for Win Rate with comprehensive evaluation
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
from typing import Dict, Optional, Tuple
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
    """Integrated WinRateOptimizer from test_strategies.py with exact same implementation"""
    
    def __init__(self, symbol: str, strategy_type: str):
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.logger = logging.getLogger(f"{__name__}.WinRateOptimizer.{symbol}")
        
        # EXACT SAME settings as test_strategies.py WinRateOptimizer
        self.optimization_timeout = 180  # 3 minutes per strategy
        self.n_trials = 100  # More trials for better exploration
        self.min_trades_threshold = 15  # Minimum trades for reliable win rate
        
    def optimize_for_winrate(self, historical_data: List, current_params: SignalParameters) -> Tuple[Dict, float, Dict]:
        """
        EXACT SAME METHOD as test_strategies.py optimize_for_winrate
        Aggressively optimize for win rate with expanded parameter ranges
        """
        if not OPTUNA_AVAILABLE:
            self.logger.warning(f"Optuna not available, using base parameters for {self.strategy_type}")
            return self._params_to_dict(current_params), 0.0, {}
        
        try:
            self.logger.info(f"Optimizing {self.strategy_type} for WIN RATE on {self.symbol}...")
            start_time = time.time()
            
            # Store historical data for optimization
            self.historical_data = historical_data
            
            # Create study with win rate focus
            study_name = f"winrate_{self.strategy_type}_{self.symbol}_{int(time.time())}"
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                storage=None
            )
            
            # Multi-stage optimization - EXACT SAME as test_strategies.py
            best_indicators, best_score, opt_details = self._multi_stage_optimization(
                study, current_params
            )
            
            optimization_time = time.time() - start_time
            
            if best_score > 0:
                opt_details.update({
                    'optimization_time': optimization_time,
                    'trials_completed': len(study.trials),
                    'best_winrate': best_score
                })
                
                self.logger.info(f"Optimization completed in {optimization_time:.1f}s")
                self.logger.info(f"Best win rate: {best_score:.2f}%")
                self.logger.info(f"Trials: {len(study.trials)}")
                
                return best_indicators, best_score, opt_details
            else:
                return self._params_to_dict(current_params), 0.0, {}
                
        except Exception as e:
            self.logger.error(f"Optimization error for {self.strategy_type}: {e}")
            return self._params_to_dict(current_params), 0.0, {'error': str(e)}
    
    def _multi_stage_optimization(self, study, current_params: SignalParameters) -> Tuple[Dict, float, Dict]:
        """
        EXACT SAME METHOD as test_strategies.py _multi_stage_optimization
        Multi-stage optimization: broad search then refinement
        """
        # Stage 1: Broad parameter exploration
        self.logger.info("Stage 1: Broad parameter exploration...")
        objective_func = self._get_broad_objective_function()
        
        study.optimize(
            objective_func,
            n_trials=int(self.n_trials * 0.7),  # 70% of trials for broad search
            timeout=int(self.optimization_timeout * 0.7),
            show_progress_bar=False
        )
        
        stage1_best = study.best_value if study.best_trial else 0
        
        # Stage 2: Refined search around best parameters
        if study.best_trial and stage1_best > 0:
            self.logger.info("Stage 2: Refined parameter search...")
            
            # Create refined ranges around best parameters
            refined_objective = self._get_refined_objective_function(study.best_params)
            
            study.optimize(
                refined_objective,
                n_trials=int(self.n_trials * 0.3),  # 30% for refinement
                timeout=int(self.optimization_timeout * 0.3),
                show_progress_bar=False
            )
        
        if study.best_trial:
            best_params = study.best_params
            best_score = study.best_value
            best_indicators = self._params_to_indicators(best_params)
            
            return best_indicators, best_score, {
                'best_params': best_params,
                'stage1_best': stage1_best,
                'final_best': best_score,
                'improvement_stages': best_score - stage1_best
            }
        else:
            return self._params_to_dict(current_params), 0.0, {}
    
    def _get_broad_objective_function(self):
        """EXACT SAME METHOD as test_strategies.py _get_broad_objective_function"""
        if 'qqe' in self.strategy_type.lower():
            return lambda trial: self._qqe_broad_objective(trial)
        elif 'rsi' in self.strategy_type.lower():
            return lambda trial: self._rsi_broad_objective(trial)
        elif 'tsi' in self.strategy_type.lower():
            return lambda trial: self._tsi_broad_objective(trial)
        else:
            return lambda trial: 0.0
    
    def _get_refined_objective_function(self, best_params: Dict):
        """EXACT SAME METHOD as test_strategies.py _get_refined_objective_function"""
        if 'qqe' in self.strategy_type.lower():
            return lambda trial: self._qqe_refined_objective(trial, best_params)
        elif 'rsi' in self.strategy_type.lower():
            return lambda trial: self._rsi_refined_objective(trial, best_params)
        elif 'tsi' in self.strategy_type.lower():
            return lambda trial: self._tsi_refined_objective(trial, best_params)
        else:
            return lambda trial: 0.0
    
    def _qqe_broad_objective(self, trial) -> float:
        """EXACT SAME METHOD as test_strategies.py _qqe_broad_objective"""
        try:
            # Significantly expanded ranges for aggressive optimization
            qqe_length = trial.suggest_int('qqe_length', 5, 30)  # Expanded from 8-20
            qqe_smooth = trial.suggest_int('qqe_smooth', 2, 12)  # Expanded from 3-8
            qqe_factor = trial.suggest_float('qqe_factor', 2.0, 8.0)  # Expanded from 3-6
            st_length = trial.suggest_int('st_length', 4, 25)  # Expanded from 6-15
            st_multiplier = trial.suggest_float('st_multiplier', 1.5, 5.0)  # Expanded from 2-4
            
            test_params = SignalParameters(
                strategy_type=self.strategy_type,
                qqe_length=qqe_length,
                qqe_smooth=qqe_smooth,
                qqe_factor=qqe_factor,
                supertrend_period=st_length,
                supertrend_multiplier=st_multiplier
            )
            
            return self._evaluate_winrate_focused(test_params)
            
        except Exception:
            return 0.0
    
    def _qqe_refined_objective(self, trial, best_params: Dict) -> float:
        """EXACT SAME METHOD as test_strategies.py _qqe_refined_objective"""
        try:
            # Create ranges +/- 20% around best parameters
            base_qqe_length = best_params.get('qqe_length', 12)
            base_qqe_smooth = best_params.get('qqe_smooth', 5)
            base_qqe_factor = best_params.get('qqe_factor', 4.236)
            base_st_length = best_params.get('st_length', 10)
            base_st_multiplier = best_params.get('st_multiplier', 2.8)
            
            qqe_length = trial.suggest_int('qqe_length', 
                                         max(3, int(base_qqe_length * 0.8)), 
                                         int(base_qqe_length * 1.2))
            qqe_smooth = trial.suggest_int('qqe_smooth',
                                         max(2, int(base_qqe_smooth * 0.8)),
                                         int(base_qqe_smooth * 1.2))
            qqe_factor = trial.suggest_float('qqe_factor',
                                           base_qqe_factor * 0.8,
                                           base_qqe_factor * 1.2)
            st_length = trial.suggest_int('st_length',
                                        max(3, int(base_st_length * 0.8)),
                                        int(base_st_length * 1.2))
            st_multiplier = trial.suggest_float('st_multiplier',
                                              base_st_multiplier * 0.8,
                                              base_st_multiplier * 1.2)
            
            test_params = SignalParameters(
                strategy_type=self.strategy_type,
                qqe_length=qqe_length,
                qqe_smooth=qqe_smooth,
                qqe_factor=qqe_factor,
                supertrend_period=st_length,
                supertrend_multiplier=st_multiplier
            )
            
            return self._evaluate_winrate_focused(test_params)
            
        except Exception:
            return 0.0
    
    def _rsi_broad_objective(self, trial) -> float:
        """EXACT SAME METHOD as test_strategies.py _rsi_broad_objective"""
        try:
            # Expanded ranges for RSI+MACD
            rsi_length = trial.suggest_int('rsi_length', 8, 28)  # Expanded from 10-21
            macd_fast = trial.suggest_int('macd_fast', 6, 20)  # Expanded from 8-16
            macd_slow = trial.suggest_int('macd_slow', 15, 40)  # Expanded from 20-35
            macd_signal = trial.suggest_int('macd_signal', 4, 15)  # Expanded from 6-12
            
            # Ensure MACD fast < slow
            if macd_fast >= macd_slow:
                return 0.0
            
            test_params = SignalParameters(
                strategy_type=self.strategy_type,
                rsi_length=rsi_length,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal
            )
            
            return self._evaluate_winrate_focused(test_params)
            
        except Exception:
            return 0.0
    
    def _rsi_refined_objective(self, trial, best_params: Dict) -> float:
        """EXACT SAME METHOD as test_strategies.py _rsi_refined_objective"""
        try:
            base_rsi = best_params.get('rsi_length', 14)
            base_macd_fast = best_params.get('macd_fast', 12)
            base_macd_slow = best_params.get('macd_slow', 26)
            base_macd_signal = best_params.get('macd_signal', 9)
            
            rsi_length = trial.suggest_int('rsi_length',
                                         max(8, int(base_rsi * 0.8)),
                                         int(base_rsi * 1.2))
            macd_fast = trial.suggest_int('macd_fast',
                                        max(6, int(base_macd_fast * 0.8)),
                                        int(base_macd_fast * 1.2))
            macd_slow = trial.suggest_int('macd_slow',
                                        max(15, int(base_macd_slow * 0.8)),
                                        int(base_macd_slow * 1.2))
            macd_signal = trial.suggest_int('macd_signal',
                                          max(4, int(base_macd_signal * 0.8)),
                                          int(base_macd_signal * 1.2))
            
            if macd_fast >= macd_slow:
                return 0.0
            
            test_params = SignalParameters(
                strategy_type=self.strategy_type,
                rsi_length=rsi_length,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal
            )
            
            return self._evaluate_winrate_focused(test_params)
            
        except Exception:
            return 0.0
    
    def _tsi_broad_objective(self, trial) -> float:
        """EXACT SAME METHOD as test_strategies.py _tsi_broad_objective"""
        try:
            # Expanded ranges for TSI
            tsi_fast = trial.suggest_int('tsi_fast', 4, 18)  # Expanded from 6-12
            tsi_slow = trial.suggest_int('tsi_slow', 8, 35)  # Expanded from 12-25
            tsi_signal = trial.suggest_int('tsi_signal', 3, 15)  # Expanded from 4-10
            
            # Ensure TSI fast < slow
            if tsi_fast >= tsi_slow:
                return 0.0
            
            test_params = SignalParameters(
                strategy_type=self.strategy_type,
                tsi_fast=tsi_fast,
                tsi_slow=tsi_slow,
                tsi_signal=tsi_signal
            )
            
            return self._evaluate_winrate_focused(test_params)
            
        except Exception:
            return 0.0
    
    def _tsi_refined_objective(self, trial, best_params: Dict) -> float:
        """EXACT SAME METHOD as test_strategies.py _tsi_refined_objective"""
        try:
            base_tsi_fast = best_params.get('tsi_fast', 8)
            base_tsi_slow = best_params.get('tsi_slow', 15)
            base_tsi_signal = best_params.get('tsi_signal', 6)
            
            tsi_fast = trial.suggest_int('tsi_fast',
                                       max(4, int(base_tsi_fast * 0.8)),
                                       int(base_tsi_fast * 1.2))
            tsi_slow = trial.suggest_int('tsi_slow',
                                       max(8, int(base_tsi_slow * 0.8)),
                                       int(base_tsi_slow * 1.2))
            tsi_signal = trial.suggest_int('tsi_signal',
                                         max(3, int(base_tsi_signal * 0.8)),
                                         int(base_tsi_signal * 1.2))
            
            if tsi_fast >= tsi_slow:
                return 0.0
            
            test_params = SignalParameters(
                strategy_type=self.strategy_type,
                tsi_fast=tsi_fast,
                tsi_slow=tsi_slow,
                tsi_signal=tsi_signal
            )
            
            return self._evaluate_winrate_focused(test_params)
            
        except Exception:
            return 0.0
    
    def _evaluate_winrate_focused(self, test_params: SignalParameters) -> float:
        """
        EXACT SAME METHOD as test_strategies.py _evaluate_winrate_focused
        Pure win rate optimization with trade count constraints
        """
        try:
            # Use the backtesting method to evaluate parameters
            win_rate, total_trades, profit_factor, max_drawdown = self._backtest_parameters(test_params)
            
            # Strict minimum trade requirement for reliable win rate
            if total_trades < self.min_trades_threshold:
                return 0.0
            
            # Win rate is the primary score (80% weight) - EXACT SAME as test_strategies.py
            win_rate_score = win_rate * 0.8
            
            # Small bonuses for supporting metrics (20% weight total) - EXACT SAME as test_strategies.py
            trade_count_bonus = min(total_trades / 50.0 * 5, 5)  # Max 5% bonus for trade count
            profit_factor_bonus = min(profit_factor * 2, 8) if profit_factor > 1 else 0  # Max 8% bonus
            drawdown_penalty = min(max_drawdown * 0.1, 7)  # Max 7% penalty
            
            # Final score heavily weighted toward win rate - EXACT SAME as test_strategies.py
            final_score = win_rate_score + trade_count_bonus + profit_factor_bonus - drawdown_penalty
            
            return max(0, final_score)
            
        except Exception:
            return 0.0
    
    def _backtest_parameters(self, test_params: SignalParameters) -> Tuple[float, int, float, float]:
        """Enhanced backtesting with comprehensive metrics like test_strategies.py"""
        try:
            if not self.historical_data:
                return 0.0, 0, 0.0, 100.0
            
            df = pd.DataFrame(self.historical_data, 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Generate signals using test parameters
            signals = []
            
            for i in range(50, len(df)):  # Skip first 50 for indicator warmup
                current_df = df.iloc[:i+1]
                signal = self._generate_test_signal(current_df, test_params)
                
                if signal != 'none':
                    signals.append({
                        'signal': signal,
                        'price': current_df['close'].iloc[-1],
                        'index': i,
                        'timestamp': current_df.index[-1]
                    })
            
            if len(signals) == 0:
                return 0.0, 0, 0.0, 100.0
            
            # Evaluate signals with comprehensive metrics
            wins = 0
            losses = 0
            total_pnl = 0.0
            equity_curve = [1.0]  # Starting with $1
            peak_equity = 1.0
            max_drawdown = 0.0
            
            for signal_data in signals:
                signal = signal_data['signal']
                entry_price = signal_data['price']
                entry_idx = signal_data['index']
                
                # Look ahead for exit (TP: 2%, SL: 1.5%, timeout: 90 candles)
                exit_found = False
                exit_pnl = 0.0
                
                for j in range(1, min(91, len(df) - entry_idx)):  # Max 90 candles ahead
                    future_idx = entry_idx + j
                    future_price = df['close'].iloc[future_idx]
                    
                    if signal == 'buy':
                        pnl_pct = ((future_price - entry_price) / entry_price) * 100
                        if pnl_pct >= 2.0:  # TP hit
                            exit_pnl = 0.02  # 2% gain
                            exit_found = True
                            break
                        elif pnl_pct <= -1.5:  # SL hit
                            exit_pnl = -0.015  # 1.5% loss
                            exit_found = True
                            break
                    else:  # sell
                        pnl_pct = ((entry_price - future_price) / entry_price) * 100
                        if pnl_pct >= 2.0:  # TP hit
                            exit_pnl = 0.02  # 2% gain
                            exit_found = True
                            break
                        elif pnl_pct <= -1.5:  # SL hit
                            exit_pnl = -0.015  # 1.5% loss
                            exit_found = True
                            break
                
                # If no TP/SL hit, use final price
                if not exit_found and entry_idx + 90 < len(df):
                    final_price = df['close'].iloc[entry_idx + 90]
                    if signal == 'buy':
                        exit_pnl = ((final_price - entry_price) / entry_price)
                    else:
                        exit_pnl = ((entry_price - final_price) / entry_price)
                    exit_pnl = max(-0.02, min(0.02, exit_pnl))  # Cap at +/-2%
                
                # Apply leverage (20x) and record results
                leveraged_pnl = exit_pnl * 20
                total_pnl += leveraged_pnl
                
                if leveraged_pnl > 0:
                    wins += 1
                else:
                    losses += 1
                
                # Update equity curve
                new_equity = equity_curve[-1] * (1 + leveraged_pnl)
                equity_curve.append(new_equity)
                
                # Track drawdown
                if new_equity > peak_equity:
                    peak_equity = new_equity
                else:
                    drawdown = (peak_equity - new_equity) / peak_equity * 100
                    max_drawdown = max(max_drawdown, drawdown)
            
            total_trades = wins + losses
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
            
            # Calculate profit factor
            gross_profit = sum(pnl for pnl in [leveraged_pnl for leveraged_pnl in [exit_pnl * 20 for exit_pnl in [0.02, -0.015]] if leveraged_pnl > 0])
            gross_loss = abs(sum(pnl for pnl in [leveraged_pnl for leveraged_pnl in [exit_pnl * 20 for exit_pnl in [0.02, -0.015]] if leveraged_pnl <= 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            return win_rate, total_trades, profit_factor, max_drawdown
            
        except Exception as e:
            self.logger.debug(f"Backtest error: {e}")
            return 0.0, 0, 0.0, 100.0
    
    def _generate_test_signal(self, df: pd.DataFrame, test_params: SignalParameters) -> str:
        """Generate signal for testing using specified parameters"""
        try:
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                # QQE + Supertrend
                qqe_result = ta.qqe(df['close'], 
                                  length=test_params.qqe_length,
                                  smooth=test_params.qqe_smooth,
                                  factor=test_params.qqe_factor)
                
                st_result = ta.supertrend(df['high'], df['low'], df['close'],
                                        length=test_params.supertrend_period,
                                        multiplier=test_params.supertrend_multiplier)
                
                if (qqe_result is not None and not qqe_result.empty and len(qqe_result.columns) >= 2 and
                    st_result is not None and not st_result.empty):
                    
                    dir_col = next((col for col in st_result.columns if 'SUPERTd' in col), None)
                    if dir_col:
                        qqe_val = qqe_result.iloc[-1, 0]
                        qqe_sig = qqe_result.iloc[-1, 1]
                        st_dir = st_result[dir_col].iloc[-1]
                        
                        qqe_bullish = qqe_sig > qqe_val
                        st_bullish = st_dir == 1
                        
                        if qqe_bullish and st_bullish:
                            return 'buy'
                        elif not qqe_bullish and not st_bullish:
                            return 'sell'
            
            elif self.strategy_type == 'rsi_macd':
                # RSI + MACD
                rsi_result = ta.rsi(df['close'], length=test_params.rsi_length)
                macd_result = ta.macd(df['close'], 
                                    fast=test_params.macd_fast,
                                    slow=test_params.macd_slow,
                                    signal=test_params.macd_signal)
                
                if (rsi_result is not None and macd_result is not None and 
                    not macd_result.empty and len(macd_result.columns) >= 3):
                    
                    rsi = rsi_result.iloc[-1]
                    macd_line = macd_result.iloc[-1, 0]
                    macd_signal = macd_result.iloc[-1, 2]
                    
                    if rsi < 35 and macd_line > macd_signal:
                        return 'buy'
                    elif rsi > 65 and macd_line < macd_signal:
                        return 'sell'
            
            elif self.strategy_type == 'tsi_vwap':
                # TSI + VWAP
                tsi_result = ta.tsi(df['close'], 
                                fast=test_params.tsi_fast,
                                slow=test_params.tsi_slow,
                                signal=test_params.tsi_signal)
                vwap_result = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                
                if (tsi_result is not None and not tsi_result.empty and
                    vwap_result is not None):
                    
                    tsi_line = tsi_result.iloc[-1, 0]
                    tsi_signal = tsi_result.iloc[-1, 1] if len(tsi_result.columns) > 1 else 0
                    vwap = vwap_result.iloc[-1]
                    price = df['close'].iloc[-1]
                    
                    tsi_bullish = tsi_line > tsi_signal
                    price_above_vwap = price > vwap
                    
                    if tsi_bullish and price_above_vwap:
                        return 'buy'
                    elif not tsi_bullish and not price_above_vwap:
                        return 'sell'
            
            return 'none'
            
        except Exception:
            return 'none'
    
    def _params_to_indicators(self, params: Dict) -> Dict:
        """EXACT SAME METHOD as test_strategies.py _params_to_indicators"""
        if 'qqe' in self.strategy_type.lower():
            return {
                'qqe_length': params['qqe_length'],
                'qqe_smooth': params['qqe_smooth'],
                'qqe_factor': params.get('qqe_factor', 4.236),
                'supertrend_period': params['st_length'],
                'supertrend_multiplier': params['st_multiplier']
            }
        elif 'rsi' in self.strategy_type.lower():
            return {
                'rsi_length': params['rsi_length'],
                'macd_fast': params['macd_fast'],
                'macd_slow': params['macd_slow'],
                'macd_signal': params['macd_signal']
            }
        elif 'tsi' in self.strategy_type.lower():
            return {
                'tsi_fast': params['tsi_fast'],
                'tsi_slow': params['tsi_slow'],
                'tsi_signal': params['tsi_signal']
            }
        else:
            return {}
    
    def _params_to_dict(self, params: SignalParameters) -> Dict:
        """Convert SignalParameters to dictionary"""
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
    """
    Enhanced Adaptive signal system with integrated WinRateOptimizer
    """
    
    def __init__(self, symbol: str, config_file: str = "data/crypto_signal_configs.json", strategy_type: str = 'qqe_supertrend_fixed'):
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        self.symbol = symbol
        self.config_file = config_file
        self.strategy_type = strategy_type
        self.position_entry_time = 0
        
        # Load optimized parameters
        self.params = self._load_symbol_config()
        self.signal_performance = self._load_signal_history()
        
        # Initialize integrated WinRateOptimizer
        self.win_rate_optimizer = IntegratedWinRateOptimizer(symbol, strategy_type)
        
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
        
        # Enhanced optimization settings using WinRateOptimizer values
        self.min_signals_for_optimization = 15  # Increased to match WinRateOptimizer
        self.optimization_interval = 180  # Reduced to 3 minutes to match WinRateOptimizer timeout
        self.win_rate_threshold = 40.0  # Target win rate
        
        # Cache for historical data
        self._historical_data_cache = None
        self._cache_timestamp = 0
        self._cache_validity = 300
        
        self.last_indicators: Dict[str, Optional[Dict]] = {
            'qqe': None, 'supertrend': None, 'rsi': None, 
            'macd': None, 'tsi': None, 'vwap': None
        }
        
        self.logger.info(f"Enhanced Adaptive Crypto Signals with WinRateOptimizer for {strategy_type} on {symbol}")
        self.logger.info(f"Current win rate: {self.params.accuracy:.1f}% | Optimization trials: {self.win_rate_optimizer.n_trials}")

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
            
            # CRITICAL: Non-blocking optimization check using WinRateOptimizer
            if self._should_optimize_for_winrate():
                self._trigger_enhanced_background_optimization()
            
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

    def _trigger_enhanced_background_optimization(self):
        """Enhanced optimization using integrated WinRateOptimizer - NON-BLOCKING"""
        try:
            with self.optimization_lock:
                if self.optimization_in_progress:
                    self.logger.debug("WinRate optimization already in progress, skipping")
                    return
                
                self.optimization_in_progress = True
            
            def enhanced_background_optimization():
                try:
                    self.logger.info("Starting enhanced WinRateOptimizer background optimization")
                    start_time = time.time()
                    
                    if OPTUNA_AVAILABLE and self._historical_data_cache:
                        # Use the integrated WinRateOptimizer
                        best_params, best_score, opt_details = self.win_rate_optimizer.optimize_for_winrate(
                            self._historical_data_cache, self.params
                        )
                    else:
                        # Fallback to simple optimization
                        best_params, best_score = self._optimize_with_grid_search()
                        opt_details = {}
                    
                    # Update if significant improvement found
                    improvement_threshold = 5.0  # 5% improvement required
                    if best_params and best_score > self.params.accuracy + improvement_threshold:
                        self._update_optimized_parameters(best_params, best_score, opt_details)
                        optimization_time = time.time() - start_time
                        self.logger.info(f"WinRateOptimizer completed in {optimization_time:.1f}s: +{best_score - self.params.accuracy:.1f}% win rate")
                        self.logger.info(f"Optimization details: {opt_details}")
                    else:
                        self.logger.debug("WinRateOptimizer found no significant improvement")
                    
                    self.last_optimization = time.time()
                    
                except Exception as e:
                    self.logger.error(f"Enhanced background optimization error: {e}")
                finally:
                    with self.optimization_lock:
                        self.optimization_in_progress = False
            
            # Start background thread
            optimization_thread = threading.Thread(target=enhanced_background_optimization, daemon=True)
            optimization_thread.start()
            self.logger.debug("Enhanced WinRateOptimizer thread started")
            
        except Exception as e:
            self.logger.error(f"Error starting enhanced optimization: {e}")
            with self.optimization_lock:
                self.optimization_in_progress = False

    def _should_optimize_for_winrate(self) -> bool:
        """Enhanced optimization trigger logic using WinRateOptimizer thresholds"""
        current_time = time.time()
        
        # Skip if optimization already running
        with self.optimization_lock:
            if self.optimization_in_progress:
                return False
        
        # Don't optimize too frequently
        if current_time - self.last_optimization < self.optimization_interval:
            return False
        
        # Need minimum signals (using WinRateOptimizer threshold)
        if len(self.signal_performance) < self.min_signals_for_optimization:
            return False
        
        # Check win rate performance
        evaluated_signals = [s for s in self.signal_performance if s.get('evaluated', False)]
        if len(evaluated_signals) < self.min_signals_for_optimization:
            return False
        
        # Trigger optimization if:
        # 1. Win rate below threshold
        # 2. No optimization done yet with sufficient signals
        # 3. Recent performance degradation
        
        if self.params.accuracy < self.win_rate_threshold:
            self.logger.debug(f"WinRate optimization trigger: Low win rate ({self.params.accuracy:.1f}%)")
            return True
        
        if self.params.optimization_score == 0.0 and len(evaluated_signals) >= self.min_signals_for_optimization:
            self.logger.debug("WinRate optimization trigger: Initial comprehensive optimization")
            return True
        
        # Check recent performance degradation
        recent_signals = list(self.signal_performance)[-10:]  # Last 10 signals
        recent_evaluated = [s for s in recent_signals if s.get('evaluated', False)]
        if len(recent_evaluated) >= 5:
            recent_wins = sum(1 for s in recent_evaluated if s.get('correct', False))
            recent_accuracy = (recent_wins / len(recent_evaluated)) * 100
            if recent_accuracy < self.params.accuracy - 20:  # 20% degradation threshold
                self.logger.debug(f"WinRate optimization trigger: Performance degradation ({recent_accuracy:.1f}%)")
                return True
        
        return False

    def _optimize_with_grid_search(self) -> tuple:
        """Fallback grid search optimization - BACKGROUND SAFE"""
        best_params = None
        best_winrate = 0.0
        
        try:
            # Simplified grid search for fallback
            if self.strategy_type in ['qqe_supertrend_fixed', 'qqe_supertrend_fast']:
                for qqe_len in [10, 12, 15]:
                    for qqe_smooth in [4, 5, 6]:
                        for st_period in [8, 10, 12]:
                            for st_mult in [2.5, 3.0]:
                                test_params = SignalParameters(
                                    strategy_type=self.strategy_type,
                                    qqe_length=qqe_len,
                                    qqe_smooth=qqe_smooth,
                                    supertrend_period=st_period,
                                    supertrend_multiplier=st_mult
                                )
                                
                                winrate = self._simple_backtest_winrate(test_params)
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
            self.logger.error(f"Grid search optimization failed: {e}")
            return None, 0.0

    def _simple_backtest_winrate(self, test_params: SignalParameters) -> float:
        """Simple backtest for fallback optimization"""
        try:
            if not self._historical_data_cache:
                return 0.0
            
            df = pd.DataFrame(self._historical_data_cache, 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_calc = df.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_calc[col] = df_calc[col].astype(float)
            
            signals = []
            
            for i in range(30, len(df_calc)):
                current_df = df_calc.iloc[:i+1]
                signal = self.win_rate_optimizer._generate_test_signal(current_df, test_params)
                
                if signal != 'none':
                    signals.append({
                        'signal': signal,
                        'price': current_df['close'].iloc[-1],
                        'index': i
                    })
            
            if len(signals) == 0:
                return 0.0
            
            wins = 0
            total = 0
            
            for signal_data in signals:
                signal = signal_data['signal']
                entry_price = signal_data['price']
                entry_idx = signal_data['index']
                
                if entry_idx + 10 < len(df_calc):
                    future_price = df_calc['close'].iloc[entry_idx + 10]
                    price_change_pct = ((future_price - entry_price) / entry_price) * 100
                    
                    is_correct = False
                    if signal == 'buy' and price_change_pct > 0.5:
                        is_correct = True
                    elif signal == 'sell' and price_change_pct < -0.5:
                        is_correct = True
                    
                    if is_correct:
                        wins += 1
                    total += 1
            
            return (wins / total) * 100.0 if total > 0 else 0.0
            
        except Exception:
            return 0.0

    def _update_optimized_parameters(self, best_params: Dict, best_score: float, opt_details: Dict = None):
        """Update parameters with WinRateOptimizer results"""
        try:
            old_accuracy = self.params.accuracy
            
            # Update parameters from WinRateOptimizer results
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
            
            self.logger.info(f"WinRateOptimizer updated parameters: {old_accuracy:.1f}% -> {best_score:.1f}% target")
            if opt_details:
                self.logger.info(f"Optimization stages: {opt_details}")
            self._save_config()
            
        except Exception as e:
            self.logger.error(f"Error updating optimized parameters: {e}")

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
                    
                    # EXACT SAME evaluation logic as WinRateOptimizer
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
            
            # Emergency stop loss (same as WinRateOptimizer SL threshold)
            if pnl_pct < -1.5:  # Using same 1.5% SL as WinRateOptimizer
                result.update({
                    'should_exit': True,
                    'exit_reason': f"Emergency SL: {pnl_pct:.2f}% loss",
                    'exit_urgency': 'immediate'
                })
                return result

            # Take profit (same as WinRateOptimizer TP threshold)
            if pnl_pct > 2.0:  # Using same 2% TP as WinRateOptimizer
                result.update({
                    'should_exit': True,
                    'exit_reason': f"Take Profit: {pnl_pct:.2f}% gain",
                    'exit_urgency': 'normal'
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
                'last_updated': time.time(),
                'winrate_optimizer_trials': self.win_rate_optimizer.n_trials,
                'winrate_optimizer_timeout': self.win_rate_optimizer.optimization_timeout
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
        """Get system status with WinRateOptimizer integration info"""
        with self.optimization_lock:
            optimization_status = self.optimization_in_progress
        
        return {
            'system_type': f'{self.strategy_type}_winrate_optimized',
            'strategy_type': self.strategy_type,
            'winrate_optimizer': {
                'n_trials': self.win_rate_optimizer.n_trials,
                'timeout_seconds': self.win_rate_optimizer.optimization_timeout,
                'min_trades_threshold': self.win_rate_optimizer.min_trades_threshold,
                'multi_stage_optimization': True
            },
            'optimized_parameters': {
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
                'interval_seconds': self.optimization_interval,
                'comprehensive_evaluation': True
            }
        }

def integrate_adaptive_crypto_signals(strategy_instance, config_file: str = None, strategy_type: str = 'qqe_supertrend_fixed'):
    """Integration function with WinRateOptimizer for comprehensive parameter optimization"""
    if config_file is None:
        config_file = os.path.join(os.getcwd(), "data", "crypto_signal_configs.json")
    
    strategy_instance.logger.info(f"Integrating WinRateOptimizer-enhanced {strategy_type} signals, config: {config_file}")
    base_sym = getattr(strategy_instance, 'original_symbol', strategy_instance.symbol)
    
    crypto_sigs = AdaptiveCryptoSignals(symbol=base_sym, config_file=config_file, strategy_type=strategy_type)
    
    strategy_instance._get_technical_direction = lambda: crypto_sigs.get_technical_direction(strategy_instance.exchange)
    strategy_instance.get_signal_status = crypto_sigs.get_system_status
    strategy_instance._crypto_signal_system = crypto_sigs
    
    strategy_instance.logger.info(f"WinRateOptimizer-enhanced {strategy_type} signals integrated:")
    strategy_instance.logger.info(f"  - Multi-stage optimization: {crypto_sigs.win_rate_optimizer.n_trials} trials")
    strategy_instance.logger.info(f"  - Comprehensive evaluation with TP/SL simulation")
    strategy_instance.logger.info(f"  - Target win rate: {crypto_sigs.win_rate_threshold}%")
    strategy_instance.logger.info(f"  - Background optimization: NON-BLOCKING")
    
    return crypto_sigs