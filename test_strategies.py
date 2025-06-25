"""
Walk-Forward Strategy Optimizer with Risk-Adjusted Metrics
Implements proper walk-forward analysis to prevent overfitting and uses Sharpe ratio optimization.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import time
from dataclasses import dataclass

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

@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    symbol: str
    strategy: str
    in_sample_performance: Dict
    out_of_sample_performance: Dict
    walk_forward_efficiency: float
    overall_sharpe: float
    overall_win_rate: float
    total_trades: int
    max_drawdown: float
    parameters: Dict

def setup_logging():
    """Setup logging for strategy testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/walk_forward_optimization.log'),
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

class WalkForwardOptimizer:
    """Walk-forward optimizer with risk-adjusted metrics"""
    
    def __init__(self, backtester: IntegratedBacktester, backtest_config: BacktestConfig):
        self.backtester = backtester
        self.backtest_config = backtest_config
        self.logger = logging.getLogger(__name__)
        
        # Walk-forward settings
        self.in_sample_pct = 0.70  # 70% for optimization
        self.out_sample_pct = 0.30  # 30% for validation
        self.num_walks = 3  # Reduced walks for smaller datasets
        self.min_trades_threshold = 5  # Reduced minimum trades
        self.min_data_points = 100  # Reduced minimum data requirement
        
        # Optimization settings - reduced for smaller datasets
        self.optimization_timeout = 20  # 20 seconds per optimization
        self.n_trials = 15  # Trials per walk
        
    def run_walk_forward_analysis(self, symbol: str, strategy_name: str, 
                                 signal_func, base_indicators: Dict) -> WalkForwardResult:
        """
        Run complete walk-forward analysis with proper in-sample/out-of-sample testing
        """
        try:
            self.logger.info(f"Running walk-forward analysis for {strategy_name} on {symbol}")
            
            # Get historical data
            df = self.backtester.fetch_ohlcv_data(symbol, 
                                                 self.backtest_config.timeframe, 
                                                 self.backtest_config.limit)
            
            # Debug information
            self.logger.debug(f"Fetched DataFrame shape: {df.shape}")
            self.logger.debug(f"DataFrame columns: {list(df.columns)}")
            self.logger.debug(f"DataFrame index type: {type(df.index)}")
            
            if len(df) < self.min_data_points:
                self.logger.warning(f"Insufficient data for walk-forward analysis: {len(df)} candles (minimum: {self.min_data_points})")
                return None
            
            # Calculate walk windows
            walk_windows = self._calculate_walk_windows(len(df))
            
            walk_results = []
            out_of_sample_trades = []
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(walk_windows):
                self.logger.info(f"Walk {i+1}/{len(walk_windows)}: Train[{train_start}:{train_end}] Test[{test_start}:{test_end}]")
                
                # Split data
                train_df = df.iloc[train_start:train_end].copy()
                test_df = df.iloc[test_start:test_end].copy()
                
                # Debug sliced DataFrames
                self.logger.debug(f"Train DataFrame columns: {list(train_df.columns)}")
                self.logger.debug(f"Train DataFrame index type: {type(train_df.index)}")
                self.logger.debug(f"Test DataFrame columns: {list(test_df.columns)}")
                self.logger.debug(f"Test DataFrame index type: {type(test_df.index)}")
                
                # Optimize on training data
                optimized_indicators, in_sample_metrics = self._optimize_on_training_data(
                    train_df, signal_func, base_indicators, strategy_name
                )
                
                # Test on out-of-sample data
                out_sample_metrics = self._test_on_validation_data(
                    test_df, signal_func, optimized_indicators
                )
                
                if out_sample_metrics and out_sample_metrics.get('total_trades', 0) >= self.min_trades_threshold:
                    walk_results.append({
                        'walk': i + 1,
                        'in_sample': in_sample_metrics,
                        'out_sample': out_sample_metrics,
                        'parameters': optimized_indicators,
                        'efficiency': self._calculate_walk_efficiency(in_sample_metrics, out_sample_metrics)
                    })
                    
                    # Collect out-of-sample trades for overall analysis
                    out_of_sample_trades.extend(self._extract_trade_data(test_df, signal_func, optimized_indicators))
            
            if not walk_results:
                self.logger.warning(f"No valid walk-forward results for {strategy_name}")
                return None
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(walk_results, out_of_sample_trades)
            
            return WalkForwardResult(
                symbol=symbol,
                strategy=strategy_name,
                in_sample_performance=self._aggregate_in_sample_metrics(walk_results),
                out_of_sample_performance=overall_metrics,
                walk_forward_efficiency=np.mean([w['efficiency'] for w in walk_results]),
                overall_sharpe=overall_metrics.get('sharpe_ratio', 0),
                overall_win_rate=overall_metrics.get('win_rate', 0),
                total_trades=overall_metrics.get('total_trades', 0),
                max_drawdown=overall_metrics.get('max_drawdown_pct', 0),
                parameters=self._get_best_parameters(walk_results)
            )
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed for {strategy_name}: {e}")
            return None
    
    def _calculate_walk_windows(self, total_length: int) -> List[Tuple[int, int, int, int]]:
        """Calculate walk-forward windows with adaptive sizing"""
        windows = []
        
        # Adaptive number of walks based on data size
        if total_length < 150:
            num_walks = 2
            min_train_size = 60
            min_test_size = 15
        elif total_length < 300:
            num_walks = 3
            min_train_size = 80
            min_test_size = 20
        else:
            num_walks = self.num_walks
            min_train_size = 100
            min_test_size = 30
        
        # Calculate window sizes
        test_size = max(min_test_size, int(total_length * self.out_sample_pct / num_walks))
        initial_train_size = max(min_train_size, int(total_length * 0.5))  # Start with 50% for training
        
        for i in range(num_walks):
            train_start = max(0, i * test_size)
            train_end = train_start + initial_train_size
            test_start = train_end
            test_end = min(total_length, test_start + test_size)
            
            # Ensure we don't exceed data bounds
            if test_end > total_length:
                break
                
            # Ensure minimum requirements
            train_length = train_end - train_start
            test_length = test_end - test_start
            
            if train_length < min_train_size or test_length < min_test_size:
                continue
                
            windows.append((train_start, train_end, test_start, test_end))
        
        return windows
    
    def _ensure_indexed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has proper structure for backtesting"""
        try:
            df_copy = df.copy()
            
            # Check if we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = list(df_copy.columns)
            
            self.logger.debug(f"DataFrame shape: {df_copy.shape}")
            self.logger.debug(f"Available columns: {available_cols}")
            self.logger.debug(f"Index type: {type(df_copy.index)}")
            
            # If DataFrame comes from backtester, it should already be properly indexed
            # Just verify we have required columns
            missing_cols = [col for col in required_cols if col not in available_cols]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"DataFrame missing required columns: {missing_cols}")
            
            # Ensure index is datetime
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                self.logger.debug("Converting index to DatetimeIndex")
                try:
                    if 'timestamp' in df_copy.columns:
                        # If timestamp is still a column, use it as index
                        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ms')
                        df_copy = df_copy.set_index('timestamp')
                    else:
                        # Try to convert existing index
                        df_copy.index = pd.to_datetime(df_copy.index)
                except Exception as idx_error:
                    self.logger.warning(f"Could not convert index to datetime: {idx_error}")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error in _ensure_indexed_dataframe: {e}")
            self.logger.error(f"DataFrame info: shape={df.shape}, columns={list(df.columns)}, index_type={type(df.index)}")
            raise

    def _optimize_on_training_data(self, train_df: pd.DataFrame, signal_func, 
                                  base_indicators: Dict, strategy_name: str) -> Tuple[Dict, Dict]:
        """Optimize parameters on training data using Sharpe ratio objective"""
        
        if not OPTUNA_AVAILABLE:
            # Run base strategy
            train_df_copy = train_df.copy()
            train_df_indexed = self._ensure_indexed_dataframe(train_df_copy)
            train_df_with_indicators = self.backtester.add_indicators(train_df_indexed, base_indicators)
            
            metrics = self._backtest_on_dataframe(train_df_with_indicators, signal_func)
            return base_indicators, metrics
        
        try:
            study = optuna.create_study(direction='maximize')
            
            def objective(trial):
                return self._sharpe_objective(trial, train_df, signal_func, strategy_name)
            
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.optimization_timeout,
                show_progress_bar=False
            )
            
            if study.best_trial:
                best_params = self._params_to_indicators(study.best_params, strategy_name)
                
                # Get detailed metrics for best parameters
                train_df_indexed = self._ensure_indexed_dataframe(train_df)
                train_df_with_indicators = self.backtester.add_indicators(train_df_indexed, best_params)
                metrics = self._backtest_on_dataframe(train_df_with_indicators, signal_func)
                
                return best_params, metrics
            else:
                # Fallback to base parameters
                train_df_indexed = self._ensure_indexed_dataframe(train_df)
                train_df_with_indicators = self.backtester.add_indicators(train_df_indexed, base_indicators)
                metrics = self._backtest_on_dataframe(train_df_with_indicators, signal_func)
                return base_indicators, metrics
                
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            # Fallback to base parameters
            train_df_indexed = self._ensure_indexed_dataframe(train_df)
            train_df_with_indicators = self.backtester.add_indicators(train_df_indexed, base_indicators)
            metrics = self._backtest_on_dataframe(train_df_with_indicators, signal_func)
            return base_indicators, metrics
    
    def _test_on_validation_data(self, test_df: pd.DataFrame, signal_func, indicators: Dict) -> Dict:
        """Test optimized parameters on out-of-sample validation data"""
        try:
            test_df_indexed = self._ensure_indexed_dataframe(test_df)
            test_df_with_indicators = self.backtester.add_indicators(test_df_indexed, indicators)
            
            return self._backtest_on_dataframe(test_df_with_indicators, signal_func)
            
        except Exception as e:
            self.logger.error(f"Validation testing failed: {e}")
            return {}
    
    def _sharpe_objective(self, trial, df: pd.DataFrame, signal_func, strategy_name: str) -> float:
        """Sharpe ratio focused objective function"""
        try:
            # Generate parameters based on strategy
            if 'qqe' in strategy_name.lower():
                qqe_length = trial.suggest_int('qqe_length', 8, 20)
                qqe_smooth = trial.suggest_int('qqe_smooth', 3, 8)
                qqe_factor = trial.suggest_float('qqe_factor', 3.0, 6.0)
                st_length = trial.suggest_int('st_length', 6, 15)
                st_multiplier = trial.suggest_float('st_multiplier', 2.0, 4.0)
                
                indicator_config = {
                    'qqe': {'length': qqe_length, 'smooth': qqe_smooth, 'factor': qqe_factor},
                    'supertrend': {'length': st_length, 'multiplier': st_multiplier}
                }
                
            elif 'rsi' in strategy_name.lower():
                rsi_length = trial.suggest_int('rsi_length', 10, 21)
                macd_fast = trial.suggest_int('macd_fast', 8, 16)
                macd_slow = trial.suggest_int('macd_slow', 20, 35)
                macd_signal = trial.suggest_int('macd_signal', 6, 12)
                
                if macd_fast >= macd_slow:
                    return -10.0  # Invalid configuration
                
                indicator_config = {
                    'rsi': {'length': rsi_length},
                    'macd': {'fast': macd_fast, 'slow': macd_slow, 'signal': macd_signal}
                }
                
            elif 'tsi' in strategy_name.lower():
                tsi_fast = trial.suggest_int('tsi_fast', 6, 12)
                tsi_slow = trial.suggest_int('tsi_slow', 12, 25)
                tsi_signal = trial.suggest_int('tsi_signal', 4, 10)
                
                if tsi_fast >= tsi_slow:
                    return -10.0  # Invalid configuration
                
                indicator_config = {
                    'tsi': {'fast': tsi_fast, 'slow': tsi_slow, 'signal': tsi_signal},
                    'vwap': {}
                }
            else:
                return 0.0
            
            # Backtest with these parameters
            df_indexed = self._ensure_indexed_dataframe(df)
            df_with_indicators = self.backtester.add_indicators(df_indexed, indicator_config)
            
            metrics = self._backtest_on_dataframe(df_with_indicators, signal_func)
            
            # Return Sharpe ratio (primary metric) with minimum trade filter
            if metrics.get('total_trades', 0) < self.min_trades_threshold:
                return -10.0
            
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            
            # Bonus for good win rate and reasonable drawdown
            win_rate = metrics.get('win_rate', 0)
            max_dd = metrics.get('max_drawdown_pct', 100)
            
            # Combined score: 70% Sharpe, 20% win rate, 10% drawdown penalty
            score = (sharpe_ratio * 0.7 + 
                    (win_rate / 100.0) * 0.2 - 
                    (max_dd / 100.0) * 0.1)
            
            return score
            
        except Exception:
            return -10.0
    
    def _backtest_on_dataframe(self, df: pd.DataFrame, signal_func) -> Dict:
        """Run backtest on prepared dataframe and return metrics"""
        try:
            if len(df) < 50:
                return {}
            
            # Reset backtest state
            trades = []
            position = None
            entry_idx = 0
            current_capital = self.backtest_config.initial_capital
            
            # Run backtest
            for i in range(30, len(df)):  # Skip first 30 for indicator warmup
                signal = signal_func(df, i)
                current_price = df['close'].iloc[i]
                
                # Exit logic
                if position:
                    exit_reason = self._check_exit_conditions(df, i, position, entry_idx)
                    if exit_reason:
                        trade = self._close_position(position, entry_idx, i, df)
                        if trade:
                            trades.append(trade)
                            current_capital += trade['pnl_abs']
                        position = None
                
                # Entry logic
                if not position and signal in ['buy', 'sell']:
                    position = {
                        'side': 'long' if signal == 'buy' else 'short',
                        'entry_price': current_price,
                        'entry_time': df.index[i],
                        'quantity': self._calculate_position_size(current_price)
                    }
                    entry_idx = i
            
            # Calculate metrics
            if len(trades) == 0:
                return {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0}
            
            return self._calculate_trade_metrics(trades, current_capital)
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return {}
    
    def _check_exit_conditions(self, df: pd.DataFrame, current_idx: int, position: Dict, entry_idx: int) -> str:
        """Check if position should be exited"""
        current_price = df['close'].iloc[current_idx]
        entry_price = position['entry_price']
        side = position['side']
        
        # Calculate PnL
        if side == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Take profit
        if pnl_pct >= self.backtest_config.take_profit_pct:
            return 'tp'
        
        # Stop loss
        if pnl_pct <= -self.backtest_config.stop_loss_pct:
            return 'sl'
        
        # Time exit
        if current_idx - entry_idx >= self.backtest_config.max_open_time_minutes:
            return 'timeout'
        
        return None
    
    def _close_position(self, position: Dict, entry_idx: int, exit_idx: int, df: pd.DataFrame) -> Dict:
        """Close position and calculate trade metrics"""
        try:
            exit_price = df['close'].iloc[exit_idx]
            side = position['side']
            quantity = position['quantity']
            entry_price = position['entry_price']
            
            # Calculate PnL
            if side == 'long':
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            
            # Apply leverage
            leveraged_pnl_pct = pnl_pct * self.backtest_config.leverage
            
            # Calculate absolute PnL
            position_value = self.backtest_config.initial_capital * (self.backtest_config.position_size_pct / 100)
            pnl_abs = position_value * (leveraged_pnl_pct / 100)
            
            # Apply costs
            commission = position_value * (self.backtest_config.commission_pct / 100) * 2
            net_pnl = pnl_abs - commission
            
            return {
                'entry_time': df.index[entry_idx],
                'exit_time': df.index[exit_idx],
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_leveraged_pct': leveraged_pnl_pct,
                'pnl_abs': net_pnl,
                'duration': exit_idx - entry_idx
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size"""
        position_value = self.backtest_config.initial_capital * (self.backtest_config.position_size_pct / 100)
        return (position_value * self.backtest_config.leverage) / price
    
    def _calculate_trade_metrics(self, trades: List[Dict], final_capital: float) -> Dict:
        """Calculate comprehensive trade metrics"""
        if not trades:
            return {}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl_abs'] > 0]
        win_rate = len(winning_trades) / total_trades * 100
        
        # PnL metrics
        total_pnl = sum(t['pnl_abs'] for t in trades)
        total_return_pct = (total_pnl / self.backtest_config.initial_capital) * 100
        
        # Risk metrics
        returns = [t['pnl_abs'] / self.backtest_config.initial_capital for t in trades]
        
        if len(returns) > 1:
            # Sharpe ratio calculation
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            
            # Annualized Sharpe (assuming daily returns)
            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Drawdown calculation
        equity_curve = [self.backtest_config.initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['pnl_abs'])
        
        peak = self.backtest_config.initial_capital
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Profit factor
        gross_profit = sum(t['pnl_abs'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_duration': np.mean([t['duration'] for t in trades])
        }
    
    def _calculate_walk_efficiency(self, in_sample: Dict, out_sample: Dict) -> float:
        """Calculate walk-forward efficiency"""
        in_sample_return = in_sample.get('total_return_pct', 0)
        out_sample_return = out_sample.get('total_return_pct', 0)
        
        if in_sample_return <= 0:
            return 0.0
        
        efficiency = (out_sample_return / in_sample_return) * 100
        return max(0, min(100, efficiency))  # Cap between 0-100%
    
    def _extract_trade_data(self, df: pd.DataFrame, signal_func, indicators: Dict) -> List[Dict]:
        """Extract trade data for overall analysis"""
        df_indexed = self._ensure_indexed_dataframe(df)
        df_with_indicators = self.backtester.add_indicators(df_indexed, indicators)
        
        trades = []
        position = None
        entry_idx = 0
        
        for i in range(30, len(df_with_indicators)):
            signal = signal_func(df_with_indicators, i)
            current_price = df_with_indicators['close'].iloc[i]
            
            if position:
                exit_reason = self._check_exit_conditions(df_with_indicators, i, position, entry_idx)
                if exit_reason:
                    trade = self._close_position(position, entry_idx, i, df_with_indicators)
                    if trade:
                        trades.append(trade)
                    position = None
            
            if not position and signal in ['buy', 'sell']:
                position = {
                    'side': 'long' if signal == 'buy' else 'short',
                    'entry_price': current_price,
                    'entry_time': df_with_indicators.index[i],
                    'quantity': self._calculate_position_size(current_price)
                }
                entry_idx = i
        
        return trades
    
    def _calculate_overall_metrics(self, walk_results: List[Dict], all_trades: List[Dict]) -> Dict:
        """Calculate overall out-of-sample metrics"""
        if not all_trades:
            return {}
        
        return self._calculate_trade_metrics(all_trades, 
                                           self.backtest_config.initial_capital + sum(t['pnl_abs'] for t in all_trades))
    
    def _aggregate_in_sample_metrics(self, walk_results: List[Dict]) -> Dict:
        """Aggregate in-sample metrics across all walks"""
        if not walk_results:
            return {}
        
        metrics = ['total_trades', 'win_rate', 'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct']
        aggregated = {}
        
        for metric in metrics:
            values = [walk['in_sample'].get(metric, 0) for walk in walk_results if walk['in_sample'].get(metric) is not None]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        return aggregated
    
    def _get_best_parameters(self, walk_results: List[Dict]) -> Dict:
        """Get the best performing parameters"""
        if not walk_results:
            return {}
        
        # Find walk with best out-of-sample Sharpe ratio
        best_walk = max(walk_results, key=lambda x: x['out_sample'].get('sharpe_ratio', 0))
        return best_walk['parameters']
    
    def _params_to_indicators(self, params: Dict, strategy_name: str) -> Dict:
        """Convert Optuna parameters to indicator configuration"""
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

def run_walk_forward_optimization():
    """Run walk-forward optimization with risk-adjusted metrics"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting WALK-FORWARD optimization with risk-adjusted metrics")
    
    try:
        backtester = IntegratedBacktester('config.json')
        
        # Optimized config for robust testing
        config = BacktestConfig(
            initial_capital=1.0,
            leverage=20.0,
            commission_pct=0.075,
            take_profit_pct=1.5,  # More conservative
            stop_loss_pct=1.0,    # Tighter stop loss
            timeframe='3m',
            limit=2000,  # Increased data fetch
            max_open_time_minutes=60
        )
        
        optimizer = WalkForwardOptimizer(backtester, config)
        
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
            logger.info(f"\n{'='*60}")
            logger.info(f"WALK-FORWARD ANALYSIS: {symbol}")
            logger.info(f"{'='*60}")
            
            for strategy_name, strategy_config in strategies.items():
                logger.info(f"\nProcessing: {strategy_name}")
                
                # Run walk-forward analysis
                wf_result = optimizer.run_walk_forward_analysis(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    signal_func=strategy_config['signal_func'],
                    base_indicators=strategy_config['indicators']
                )
                
                if wf_result:
                    # Create result entry
                    result_entry = {
                        'symbol': symbol,
                        'strategy': f"{strategy_name}_WalkForward",
                        'type': 'walk_forward',
                        'description': f"{strategy_config['description']} (Walk-Forward Optimized)",
                        'total_trades': wf_result.total_trades,
                        'win_rate': wf_result.overall_win_rate,
                        'sharpe_ratio': wf_result.overall_sharpe,
                        'walk_forward_efficiency': wf_result.walk_forward_efficiency,
                        'max_drawdown_pct': wf_result.max_drawdown,
                        'parameters': wf_result.parameters,
                        'in_sample_avg_sharpe': wf_result.in_sample_performance.get('avg_sharpe_ratio', 0),
                        'out_sample_sharpe': wf_result.out_of_sample_performance.get('sharpe_ratio', 0)
                    }
                    
                    all_results.append(result_entry)
                    
                    logger.info(f"Walk-Forward Results:")
                    logger.info(f"  Out-of-Sample Trades: {wf_result.total_trades}")
                    logger.info(f"  Out-of-Sample Win Rate: {wf_result.overall_win_rate:.2f}%")
                    logger.info(f"  Out-of-Sample Sharpe: {wf_result.overall_sharpe:.3f}")
                    logger.info(f"  Walk-Forward Efficiency: {wf_result.walk_forward_efficiency:.2f}%")
                    logger.info(f"  Max Drawdown: {wf_result.max_drawdown:.2f}%")
                else:
                    logger.warning(f"Walk-forward analysis failed for {strategy_name}")
        
        # Export results
        if all_results:
            os.makedirs('walk_forward_results', exist_ok=True)
            
            df = pd.DataFrame(all_results)
            df.to_csv('walk_forward_results/walk_forward_analysis.csv', index=False)
            
            # Summary statistics
            logger.info(f"\n{'='*80}")
            logger.info("WALK-FORWARD OPTIMIZATION SUMMARY")
            logger.info(f"{'='*80}")
            
            if all_results:
                # Filter valid results (efficiency > 50%, Sharpe > 0.5)
                valid_results = [r for r in all_results 
                               if r['walk_forward_efficiency'] > 50 and r['sharpe_ratio'] > 0.5]
                
                logger.info(f"Total Strategies Tested: {len(all_results)}")
                logger.info(f"Strategies Passing Validation: {len(valid_results)}")
                
                if valid_results:
                    avg_efficiency = np.mean([r['walk_forward_efficiency'] for r in valid_results])
                    avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid_results])
                    avg_win_rate = np.mean([r['win_rate'] for r in valid_results])
                    
                    logger.info(f"Average Walk-Forward Efficiency: {avg_efficiency:.2f}%")
                    logger.info(f"Average Out-of-Sample Sharpe: {avg_sharpe:.3f}")
                    logger.info(f"Average Out-of-Sample Win Rate: {avg_win_rate:.2f}%")
                    
                    # Best strategy
                    best_strategy = max(valid_results, key=lambda x: x['sharpe_ratio'])
                    logger.info(f"\nBest Strategy (by Sharpe Ratio):")
                    logger.info(f"  {best_strategy['strategy']} on {best_strategy['symbol']}")
                    logger.info(f"  Sharpe Ratio: {best_strategy['sharpe_ratio']:.3f}")
                    logger.info(f"  Win Rate: {best_strategy['win_rate']:.2f}%")
                    logger.info(f"  Walk-Forward Efficiency: {best_strategy['walk_forward_efficiency']:.2f}%")
                    logger.info(f"  Parameters: {best_strategy['parameters']}")
                else:
                    logger.warning("No strategies passed validation criteria!")
                    logger.warning("Consider relaxing criteria: efficiency > 50%, Sharpe > 0.5")
            
            logger.info(f"\nResults exported to: walk_forward_results/walk_forward_analysis.csv")
        
    except Exception as e:
        logger.error(f"Critical error in walk-forward optimization: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    """Main function"""
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Walk-Forward Strategy Optimizer with Risk-Adjusted Metrics")
    
    run_walk_forward_optimization()

if __name__ == "__main__":
    main()