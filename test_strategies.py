"""
Win Rate Focused Strategy Optimizer
Prioritizes win rate improvement over other metrics
"""

import os
import sys
import logging
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.backtester import IntegratedBacktester, BacktestConfig
from core.backtester import qqe_supertrend_signal_fixed, rsi_macd_signal, tsi_vwap_signal

try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("Optuna available for parameter optimization")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

def setup_logging():
    """Setup logging for strategy testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/winrate_optimization.log'),
            logging.StreamHandler()
        ]
    )
    os.makedirs('logs', exist_ok=True)

def get_active_symbols(backtester: IntegratedBacktester, limit: int = 3) -> List[str]:
    """Get top active symbols from exchange"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Fetching top {limit} active symbols from exchange...")
        
        active_symbols_data = backtester.exchange.get_top_active_symbols(limit=limit)
        
        if active_symbols_data:
            symbols = [item['symbol'] for item in active_symbols_data]
            logger.info(f"Found active symbols: {symbols}")
            return symbols
        else:
            fallback_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'][:limit]
            logger.warning(f"Using fallback symbols: {fallback_symbols}")
            return fallback_symbols
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error fetching active symbols: {e}")
        return ['BTCUSDT', 'ETHUSDT'][:limit]

class WinRateOptimizer:
    """Aggressive win rate optimizer with expanded parameter ranges"""
    
    def __init__(self, backtester: IntegratedBacktester, backtest_config: BacktestConfig):
        self.backtester = backtester
        self.backtest_config = backtest_config
        self.logger = logging.getLogger(__name__)
        
        # Aggressive optimization settings for win rate
        self.optimization_timeout = 180  # 3 minutes per strategy
        self.n_trials = 100  # More trials for better exploration
        self.min_trades_threshold = 15  # Minimum trades for reliable win rate
        
    def optimize_for_winrate(self, symbol: str, strategy_name: str, 
                            signal_func, base_indicators: Dict) -> Tuple[Dict, float, Dict]:
        """
        Aggressively optimize for win rate with expanded parameter ranges
        """
        if not OPTUNA_AVAILABLE:
            self.logger.warning(f"Optuna not available, using base parameters for {strategy_name}")
            return base_indicators, 0.0, {}
        
        try:
            self.logger.info(f"Optimizing {strategy_name} for WIN RATE on {symbol}...")
            start_time = time.time()
            
            # Create study with win rate focus
            study_name = f"winrate_{strategy_name}_{symbol}_{int(time.time())}"
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                storage=None
            )
            
            # Multi-stage optimization
            best_indicators, best_score, opt_details = self._multi_stage_optimization(
                study, symbol, strategy_name, signal_func, base_indicators
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
                return base_indicators, 0.0, {}
                
        except Exception as e:
            self.logger.error(f"Optimization error for {strategy_name}: {e}")
            return base_indicators, 0.0, {'error': str(e)}
    
    def _multi_stage_optimization(self, study, symbol: str, strategy_name: str, 
                                 signal_func, base_indicators: Dict) -> Tuple[Dict, float, Dict]:
        """
        Multi-stage optimization: broad search then refinement
        """
        # Stage 1: Broad parameter exploration
        self.logger.info("Stage 1: Broad parameter exploration...")
        objective_func = self._get_broad_objective_function(symbol, strategy_name, signal_func)
        
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
            refined_objective = self._get_refined_objective_function(
                symbol, strategy_name, signal_func, study.best_params
            )
            
            study.optimize(
                refined_objective,
                n_trials=int(self.n_trials * 0.3),  # 30% for refinement
                timeout=int(self.optimization_timeout * 0.3),
                show_progress_bar=False
            )
        
        if study.best_trial:
            best_params = study.best_params
            best_score = study.best_value
            best_indicators = self._params_to_indicators(best_params, strategy_name)
            
            return best_indicators, best_score, {
                'best_params': best_params,
                'stage1_best': stage1_best,
                'final_best': best_score,
                'improvement_stages': best_score - stage1_best
            }
        else:
            return base_indicators, 0.0, {}
    
    def _get_broad_objective_function(self, symbol: str, strategy_name: str, signal_func):
        """Broad parameter search objective function"""
        if 'qqe' in strategy_name.lower():
            return lambda trial: self._qqe_broad_objective(trial, symbol, signal_func)
        elif 'rsi' in strategy_name.lower():
            return lambda trial: self._rsi_broad_objective(trial, symbol, signal_func)
        elif 'tsi' in strategy_name.lower():
            return lambda trial: self._tsi_broad_objective(trial, symbol, signal_func)
        else:
            return lambda trial: 0.0
    
    def _get_refined_objective_function(self, symbol: str, strategy_name: str, 
                                       signal_func, best_params: Dict):
        """Refined parameter search around best found parameters"""
        if 'qqe' in strategy_name.lower():
            return lambda trial: self._qqe_refined_objective(trial, symbol, signal_func, best_params)
        elif 'rsi' in strategy_name.lower():
            return lambda trial: self._rsi_refined_objective(trial, symbol, signal_func, best_params)
        elif 'tsi' in strategy_name.lower():
            return lambda trial: self._tsi_refined_objective(trial, symbol, signal_func, best_params)
        else:
            return lambda trial: 0.0
    
    def _qqe_broad_objective(self, trial, symbol: str, signal_func) -> float:
        """Broad QQE parameter search with expanded ranges"""
        try:
            # Significantly expanded ranges for aggressive optimization
            qqe_length = trial.suggest_int('qqe_length', 5, 30)  # Expanded from 8-20
            qqe_smooth = trial.suggest_int('qqe_smooth', 2, 12)  # Expanded from 3-8
            qqe_factor = trial.suggest_float('qqe_factor', 2.0, 8.0)  # Expanded from 3-6
            st_length = trial.suggest_int('st_length', 4, 25)  # Expanded from 6-15
            st_multiplier = trial.suggest_float('st_multiplier', 1.5, 5.0)  # Expanded from 2-4
            
            indicator_config = {
                'qqe': {
                    'length': qqe_length,
                    'smooth': qqe_smooth, 
                    'factor': qqe_factor
                },
                'supertrend': {
                    'length': st_length,
                    'multiplier': st_multiplier
                }
            }
            
            return self._evaluate_winrate_focused(symbol, signal_func, indicator_config)
            
        except Exception:
            return 0.0
    
    def _qqe_refined_objective(self, trial, symbol: str, signal_func, best_params: Dict) -> float:
        """Refined QQE search around best parameters"""
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
            
            indicator_config = {
                'qqe': {
                    'length': qqe_length,
                    'smooth': qqe_smooth, 
                    'factor': qqe_factor
                },
                'supertrend': {
                    'length': st_length,
                    'multiplier': st_multiplier
                }
            }
            
            return self._evaluate_winrate_focused(symbol, signal_func, indicator_config)
            
        except Exception:
            return 0.0
    
    def _rsi_broad_objective(self, trial, symbol: str, signal_func) -> float:
        """Broad RSI+MACD parameter search"""
        try:
            # Expanded ranges for RSI+MACD
            rsi_length = trial.suggest_int('rsi_length', 8, 28)  # Expanded from 10-21
            macd_fast = trial.suggest_int('macd_fast', 6, 20)  # Expanded from 8-16
            macd_slow = trial.suggest_int('macd_slow', 15, 40)  # Expanded from 20-35
            macd_signal = trial.suggest_int('macd_signal', 4, 15)  # Expanded from 6-12
            
            # Ensure MACD fast < slow
            if macd_fast >= macd_slow:
                return 0.0
            
            indicator_config = {
                'rsi': {'length': rsi_length},
                'macd': {
                    'fast': macd_fast,
                    'slow': macd_slow,
                    'signal': macd_signal
                }
            }
            
            return self._evaluate_winrate_focused(symbol, signal_func, indicator_config)
            
        except Exception:
            return 0.0
    
    def _rsi_refined_objective(self, trial, symbol: str, signal_func, best_params: Dict) -> float:
        """Refined RSI+MACD search"""
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
            
            indicator_config = {
                'rsi': {'length': rsi_length},
                'macd': {
                    'fast': macd_fast,
                    'slow': macd_slow,
                    'signal': macd_signal
                }
            }
            
            return self._evaluate_winrate_focused(symbol, signal_func, indicator_config)
            
        except Exception:
            return 0.0
    
    def _tsi_broad_objective(self, trial, symbol: str, signal_func) -> float:
        """Broad TSI+VWAP parameter search"""
        try:
            # Expanded ranges for TSI
            tsi_fast = trial.suggest_int('tsi_fast', 4, 18)  # Expanded from 6-12
            tsi_slow = trial.suggest_int('tsi_slow', 8, 35)  # Expanded from 12-25
            tsi_signal = trial.suggest_int('tsi_signal', 3, 15)  # Expanded from 4-10
            
            # Ensure TSI fast < slow
            if tsi_fast >= tsi_slow:
                return 0.0
            
            indicator_config = {
                'tsi': {
                    'fast': tsi_fast,
                    'slow': tsi_slow,
                    'signal': tsi_signal
                },
                'vwap': {}
            }
            
            return self._evaluate_winrate_focused(symbol, signal_func, indicator_config)
            
        except Exception:
            return 0.0
    
    def _tsi_refined_objective(self, trial, symbol: str, signal_func, best_params: Dict) -> float:
        """Refined TSI+VWAP search"""
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
            
            indicator_config = {
                'tsi': {
                    'fast': tsi_fast,
                    'slow': tsi_slow,
                    'signal': tsi_signal
                },
                'vwap': {}
            }
            
            return self._evaluate_winrate_focused(symbol, signal_func, indicator_config)
            
        except Exception:
            return 0.0
    
    def _evaluate_winrate_focused(self, symbol: str, signal_func, indicator_config: Dict) -> float:
        """
        Pure win rate optimization with trade count constraints
        """
        try:
            result = self.backtester.run_backtest(
                symbol=symbol,
                signal_func=signal_func,
                indicator_config=indicator_config,
                backtest_config=self.backtest_config
            )
            
            if 'error' in result:
                return 0.0
            
            metrics = result['metrics']
            
            win_rate = metrics.get('win_rate', 0)
            total_trades = metrics.get('total_trades', 0)
            profit_factor = metrics.get('profit_factor', 0)
            max_drawdown = metrics.get('max_drawdown_pct', 100)
            
            # Strict minimum trade requirement for reliable win rate
            if total_trades < self.min_trades_threshold:
                return 0.0
            
            # Win rate is the primary score (80% weight)
            win_rate_score = win_rate * 0.8
            
            # Small bonuses for supporting metrics (20% weight total)
            trade_count_bonus = min(total_trades / 50.0 * 5, 5)  # Max 5% bonus for trade count
            profit_factor_bonus = min(profit_factor * 2, 8) if profit_factor > 1 else 0  # Max 8% bonus
            drawdown_penalty = min(max_drawdown * 0.1, 7)  # Max 7% penalty
            
            # Final score heavily weighted toward win rate
            final_score = win_rate_score + trade_count_bonus + profit_factor_bonus - drawdown_penalty
            
            return max(0, final_score)
            
        except Exception:
            return 0.0
    
    def _params_to_indicators(self, params: Dict, strategy_name: str) -> Dict:
        """Convert Optuna parameters back to indicator configuration"""
        if 'qqe' in strategy_name.lower():
            return {
                'qqe': {
                    'length': params['qqe_length'],
                    'smooth': params['qqe_smooth'],
                    'factor': params['qqe_factor']
                },
                'supertrend': {
                    'length': params['st_length'],
                    'multiplier': params['st_multiplier']
                }
            }
        elif 'rsi' in strategy_name.lower():
            return {
                'rsi': {'length': params['rsi_length']},
                'macd': {
                    'fast': params['macd_fast'],
                    'slow': params['macd_slow'],
                    'signal': params['macd_signal']
                }
            }
        elif 'tsi' in strategy_name.lower():
            return {
                'tsi': {
                    'fast': params['tsi_fast'],
                    'slow': params['tsi_slow'],
                    'signal': params['tsi_signal']
                },
                'vwap': {}
            }
        else:
            return {}

def run_winrate_focused_optimization():
    """Run win rate focused optimization"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting WIN RATE FOCUSED optimization")
    
    try:
        backtester = IntegratedBacktester('config.json')
        
        # Optimized config for win rate
        config = BacktestConfig(
            initial_capital=1.0,
            leverage=20.0,
            commission_pct=0.075,
            take_profit_pct=2.0,  # Slightly higher TP
            stop_loss_pct=1.5,    # Slightly higher SL
            timeframe='3m',
            limit=1400,
            max_open_time_minutes=90  # Longer hold time
        )
        
        optimizer = WinRateOptimizer(backtester, config)
        
        # Strategy definitions
        strategies = {
            'QQE_Supertrend': {
                'indicators': {
                    'qqe': {'length': 12, 'smooth': 5, 'factor': 4.236},
                    'supertrend': {'length': 10, 'multiplier': 2.8}
                },
                'signal_func': qqe_supertrend_signal_fixed,
                'description': 'QQE + Supertrend'
            },
            'RSI_MACD': {
                'indicators': {
                    'rsi': {'length': 14},
                    'macd': {'fast': 12, 'slow': 26, 'signal': 9}
                },
                'signal_func': rsi_macd_signal,
                'description': 'RSI + MACD'
            },
            'TSI_VWAP': {
                'indicators': {
                    'tsi': {'fast': 8, 'slow': 15, 'signal': 6},
                    'vwap': {}
                },
                'signal_func': tsi_vwap_signal,
                'description': 'TSI + VWAP'
            }
        }
        
        test_symbols = get_active_symbols(backtester, limit=3)
        all_results = []
        
        logger.info(f"Testing on symbols: {test_symbols}")
        
        for symbol in test_symbols:
            logger.info(f"\nTesting symbol: {symbol}")
            
            for strategy_name, strategy_config in strategies.items():
                logger.info(f"Processing: {strategy_name}")
                
                # Test base strategy
                base_result = backtester.run_backtest(
                    symbol=symbol,
                    signal_func=strategy_config['signal_func'],
                    indicator_config=strategy_config['indicators'],
                    backtest_config=config
                )
                
                if 'error' not in base_result:
                    base_metrics = base_result['metrics']
                    
                    all_results.append({
                        'symbol': symbol,
                        'strategy': f"{strategy_name}_Base",
                        'type': 'base',
                        'description': strategy_config['description'],
                        **base_metrics,
                        'parameters': strategy_config['indicators']
                    })
                    
                    logger.info(f"Base - Trades: {base_metrics['total_trades']}, "
                              f"Win Rate: {base_metrics['win_rate']:.2f}%")
                    
                    # Run aggressive optimization
                    if OPTUNA_AVAILABLE:
                        opt_indicators, opt_winrate, opt_details = optimizer.optimize_for_winrate(
                            symbol, strategy_name, strategy_config['signal_func'], 
                            strategy_config['indicators']
                        )
                        
                        if opt_winrate > 0:
                            opt_result = backtester.run_backtest(
                                symbol=symbol,
                                signal_func=strategy_config['signal_func'],
                                indicator_config=opt_indicators,
                                backtest_config=config
                            )
                            
                            if 'error' not in opt_result:
                                opt_metrics = opt_result['metrics']
                                
                                all_results.append({
                                    'symbol': symbol,
                                    'strategy': f"{strategy_name}_WinRate_Optimized",
                                    'type': 'optimized',
                                    'description': f"{strategy_config['description']} (Win Rate Optimized)",
                                    **opt_metrics,
                                    'parameters': opt_indicators,
                                    'optimization_details': opt_details
                                })
                                
                                wr_improvement = opt_metrics['win_rate'] - base_metrics['win_rate']
                                
                                logger.info(f"Optimized - Trades: {opt_metrics['total_trades']}, "
                                          f"Win Rate: {opt_metrics['win_rate']:.2f}% "
                                          f"(+{wr_improvement:.2f}%)")
        
        # Export results
        if all_results:
            os.makedirs('winrate_optimization_results', exist_ok=True)
            
            df = pd.DataFrame(all_results)
            df.to_csv('winrate_optimization_results/winrate_focused_results.csv', index=False)
            
            # Summary statistics
            logger.info(f"\nWIN RATE OPTIMIZATION SUMMARY:")
            logger.info("=" * 60)
            
            base_results = [r for r in all_results if r['type'] == 'base']
            opt_results = [r for r in all_results if r['type'] == 'optimized']
            
            if base_results and opt_results:
                avg_base_wr = sum(r['win_rate'] for r in base_results) / len(base_results)
                avg_opt_wr = sum(r['win_rate'] for r in opt_results) / len(opt_results)
                avg_improvement = avg_opt_wr - avg_base_wr
                
                logger.info(f"Average Base Win Rate: {avg_base_wr:.2f}%")
                logger.info(f"Average Optimized Win Rate: {avg_opt_wr:.2f}%")
                logger.info(f"Average Improvement: +{avg_improvement:.2f}%")
                
                # Best results
                best_opt = max(opt_results, key=lambda x: x['win_rate'])
                logger.info(f"\nBest Optimized Strategy:")
                logger.info(f"  {best_opt['strategy']} on {best_opt['symbol']}")
                logger.info(f"  Win Rate: {best_opt['win_rate']:.2f}%")
                logger.info(f"  Trades: {best_opt['total_trades']}")
                logger.info(f"  Parameters: {best_opt['parameters']}")
            
            logger.info(f"\nResults exported to: winrate_optimization_results/winrate_focused_results.csv")
        
    except Exception as e:
        logger.error(f"Critical error in win rate optimization: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    """Main function"""
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Win Rate Focused Strategy Optimizer")
    
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is required for win rate optimization")
        logger.error("Install with: pip install optuna")
        return
    
    run_winrate_focused_optimization()

if __name__ == "__main__":
    main()