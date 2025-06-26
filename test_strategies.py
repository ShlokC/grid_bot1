"""
VectorBT-based Strategy Testing with Intelligent Data Validation
Properly architected to pre-filter symbols and leverage VectorBT's capabilities
"""

import os
import sys

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import logging
import json
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
import time
from typing import Dict, List, Tuple, Optional
from itertools import product

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
    print("VectorBT available for advanced backtesting")
except ImportError:
    VECTORBT_AVAILABLE = False
    print("VectorBT not available. Install with: pip install vectorbt pandas-ta")
    sys.exit(1)

from core.exchange import Exchange

def setup_logging():
    """Setup logging for strategy testing with proper Unicode handling"""
    os.makedirs('logs', exist_ok=True)
    
    file_handler = logging.FileHandler('logs/vectorbt_intelligent_optimization.log', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def get_symbol_data_info(exchange: Exchange, symbol: str, timeframe: str = '3m', limit: int = 1400) -> Dict:
    """Get comprehensive data availability info for a symbol"""
    try:
        ohlcv_data = exchange.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        if not ohlcv_data:
            return {'symbol': symbol, 'available': False, 'candles': 0, 'error': 'No data'}
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df_clean = df.dropna()
        
        return {
            'symbol': symbol,
            'available': True,
            'candles': len(df_clean),
            'raw_candles': len(df),
            'data_quality': len(df_clean) / len(df) if len(df) > 0 else 0,
            'price_range': (df_clean['close'].min(), df_clean['close'].max()),
            'date_range': (df_clean.index.min(), df_clean.index.max()),
            'data': df_clean
        }
        
    except Exception as e:
        return {'symbol': symbol, 'available': False, 'candles': 0, 'error': str(e)}

def calculate_strategy_data_requirements() -> Dict[str, int]:
    """Calculate minimum data requirements for each strategy"""
    return {
        'qqe_supertrend': 50,  # QQE(20) + ST(19) + buffer(11) = 50
        'rsi_macd': 75,        # RSI(24) + MACD(39+14) + buffer(22) = 75  
        'tsi_vwap': 65,        # TSI(29+11) + VWAP(10) + buffer(15) = 65
        'minimum_for_all': 75  # Use the highest requirement
    }

def filter_viable_symbols(exchange: Exchange, candidate_symbols: List[str]) -> Tuple[List[str], Dict]:
    """Pre-filter symbols based on data availability for all strategies"""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating data availability for {len(candidate_symbols)} symbols...")
    
    requirements = calculate_strategy_data_requirements()
    min_required = requirements['minimum_for_all']
    
    viable_symbols = []
    symbol_info = {}
    
    for symbol in candidate_symbols:
        info = get_symbol_data_info(exchange, symbol)
        symbol_info[symbol] = info
        
        if info['available'] and info['candles'] >= min_required:
            viable_symbols.append(symbol)
            logger.info(f"✓ {symbol}: {info['candles']} candles (viable)")
        else:
            reason = info.get('error', f"only {info['candles']} candles")
            logger.warning(f"✗ {symbol}: {reason} (need {min_required})")
    
    logger.info(f"Symbol filtering complete: {len(viable_symbols)}/{len(candidate_symbols)} symbols viable")
    return viable_symbols, symbol_info

class IntelligentVectorBTTester:
    """Intelligent VectorBT tester with proper data validation and optimization"""
    
    def __init__(self, exchange: Exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
        
        # VectorBT configuration
        self.initial_cash = 1.0
        self.commission = 0.00075
        self.leverage = 20.0
        
        # Optimized parameter ranges for faster execution
        self.strategies = {
            'qqe_supertrend': {
                'name': 'QQE + Supertrend',
                'params': {
                    'qqe_length': [8, 10, 12, 15, 18],
                    'qqe_smooth': [3, 4, 5, 6, 7],
                    'st_length': [6, 8, 10, 12, 15],
                    'st_multiplier': [2.0, 2.3, 2.6, 2.9, 3.2]
                }
            },
            'rsi_macd': {
                'name': 'RSI + MACD',
                'params': {
                    'rsi_length': [10, 12, 14, 16, 18, 20],
                    'macd_fast': [8, 10, 12, 14],
                    'macd_slow': [20, 24, 26, 30],
                    'macd_signal': [6, 8, 9, 10, 12]
                }
            },
            'tsi_vwap': {
                'name': 'TSI + VWAP',
                'params': {
                    'tsi_fast': [4, 6, 8, 10],
                    'tsi_slow': [12, 15, 18, 21, 25],
                    'tsi_signal': [3, 4, 5, 6, 8]
                }
            }
        }
    
    def optimize_strategy_efficiently(self, data: pd.DataFrame, symbol: str, strategy_name: str) -> List[Dict]:
        """Efficiently optimize strategy using VectorBT's capabilities"""
        try:
            self.logger.info(f"Optimizing {strategy_name} on {symbol} ({len(data)} candles)")
            
            strategy_config = self.strategies[strategy_name]
            results = []
            
            if strategy_name == 'qqe_supertrend':
                results = self._optimize_qqe_supertrend(data, symbol, strategy_config)
            elif strategy_name == 'rsi_macd':
                results = self._optimize_rsi_macd(data, symbol, strategy_config)
            elif strategy_name == 'tsi_vwap':
                results = self._optimize_tsi_vwap(data, symbol, strategy_config)
            
            # Sort by optimization score
            if results:
                results.sort(key=lambda x: x['score'], reverse=True)
                self.logger.info(f"Generated {len(results)} valid results for {strategy_name}")
            else:
                self.logger.warning(f"No valid results for {strategy_name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error optimizing {strategy_name} on {symbol}: {e}")
            return []
    
    def _optimize_qqe_supertrend(self, data: pd.DataFrame, symbol: str, config: Dict) -> List[Dict]:
        """Optimize QQE + Supertrend strategy"""
        results = []
        
        # Generate parameter combinations efficiently
        param_combinations = list(product(
            config['params']['qqe_length'],
            config['params']['qqe_smooth'],
            config['params']['st_length'],
            config['params']['st_multiplier']
        ))
        
        # Process in batches for memory efficiency
        batch_size = 50
        for i in range(0, len(param_combinations), batch_size):
            batch = param_combinations[i:i+batch_size]
            batch_results = self._process_qqe_supertrend_batch(data, symbol, batch)
            results.extend(batch_results)
        
        return results
    
    def _process_qqe_supertrend_batch(self, data: pd.DataFrame, symbol: str, param_batch: List) -> List[Dict]:
        """Process a batch of QQE + Supertrend parameters"""
        batch_results = []
        
        for ql, qs, sl, sm in param_batch:
            try:
                # Calculate indicators
                qqe = ta.qqe(data['close'], length=ql, smooth=qs)
                if qqe is None or qqe.empty or len(qqe.columns) < 2:
                    continue
                
                st = ta.supertrend(data['high'], data['low'], data['close'], length=sl, multiplier=sm)
                if st is None or st.empty:
                    continue
                
                dir_cols = [col for col in st.columns if 'SUPERTd' in col]
                if not dir_cols:
                    continue
                
                # Generate signals
                qqe_line = qqe.iloc[:, 0]
                qqe_signal = qqe.iloc[:, 1]
                st_direction = st[dir_cols[0]]
                
                # Explicit bullish/bearish conditions
                qqe_bullish = qqe_line > qqe_signal
                qqe_bearish = qqe_line < qqe_signal
                st_bullish = st_direction == 1
                st_bearish = st_direction == -1
                
                # Generate entry/exit signals
                long_entries = qqe_bullish & st_bullish
                short_entries = qqe_bearish & st_bearish
                long_exits = qqe_bearish | st_bearish
                short_exits = qqe_bullish | st_bullish
                
                # Run VectorBT simulation
                pf = vbt.Portfolio.from_signals(
                    data['close'],
                    long_entries, long_exits,
                    short_entries, short_exits,
                    init_cash=self.initial_cash,
                    fees=self.commission,
                    freq='3min'
                )
                
                # Extract statistics safely
                stats = pf.stats()
                result = self._extract_portfolio_stats(stats, symbol, 'qqe_supertrend', 
                                                     f"QQE({ql},{qs}), ST({sl},{sm})",
                                                     {'qqe_length': ql, 'qqe_smooth': qs, 
                                                      'st_length': sl, 'st_multiplier': sm})
                
                if result and result.get('total_trades', 0) >= 5:  # Minimum trade requirement
                    batch_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Error processing QQE+ST params ({ql},{qs},{sl},{sm}): {e}")
                continue
        
        return batch_results
    
    def _optimize_rsi_macd(self, data: pd.DataFrame, symbol: str, config: Dict) -> List[Dict]:
        """Optimize RSI + MACD strategy"""
        results = []
        
        param_combinations = [
            (rl, mf, ms, msig) for rl in config['params']['rsi_length']
            for mf in config['params']['macd_fast']
            for ms in config['params']['macd_slow']
            for msig in config['params']['macd_signal']
            if mf < ms  # Valid MACD constraint
        ]
        
        # Process in batches
        batch_size = 40
        for i in range(0, len(param_combinations), batch_size):
            batch = param_combinations[i:i+batch_size]
            batch_results = self._process_rsi_macd_batch(data, symbol, batch)
            results.extend(batch_results)
        
        return results
    
    def _process_rsi_macd_batch(self, data: pd.DataFrame, symbol: str, param_batch: List) -> List[Dict]:
        """Process a batch of RSI + MACD parameters"""
        batch_results = []
        
        for rl, mf, ms, msig in param_batch:
            try:
                # Calculate indicators
                rsi = ta.rsi(data['close'], length=rl)
                if rsi is None:
                    continue
                
                macd = ta.macd(data['close'], fast=mf, slow=ms, signal=msig)
                if macd is None or macd.empty or len(macd.columns) < 3:
                    continue
                
                # Generate signals
                macd_line = macd.iloc[:, 0]
                macd_signal_line = macd.iloc[:, 2]
                
                macd_bullish = macd_line > macd_signal_line
                macd_bearish = macd_line < macd_signal_line
                
                long_entries = (rsi < 30) & macd_bullish
                short_entries = (rsi > 70) & macd_bearish
                long_exits = (rsi > 50) | macd_bearish
                short_exits = (rsi < 50) | macd_bullish
                
                # Run VectorBT simulation
                pf = vbt.Portfolio.from_signals(
                    data['close'],
                    long_entries, long_exits,
                    short_entries, short_exits,
                    init_cash=self.initial_cash,
                    fees=self.commission,
                    freq='3min'
                )
                
                # Extract statistics
                stats = pf.stats()
                result = self._extract_portfolio_stats(stats, symbol, 'rsi_macd',
                                                     f"RSI({rl}), MACD({mf},{ms},{msig})",
                                                     {'rsi_length': rl, 'macd_fast': mf,
                                                      'macd_slow': ms, 'macd_signal': msig})
                
                if result and result.get('total_trades', 0) >= 5:
                    batch_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Error processing RSI+MACD params ({rl},{mf},{ms},{msig}): {e}")
                continue
        
        return batch_results
    
    def _optimize_tsi_vwap(self, data: pd.DataFrame, symbol: str, config: Dict) -> List[Dict]:
        """Optimize TSI + VWAP strategy"""
        results = []
        
        # Pre-calculate VWAP once (same for all parameter combinations)
        vwap = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
        if vwap is None:
            return results
        
        param_combinations = [
            (tf, ts, tsig) for tf in config['params']['tsi_fast']
            for ts in config['params']['tsi_slow']
            for tsig in config['params']['tsi_signal']
            if tf < ts  # Valid TSI constraint
        ]
        
        # Process in batches
        batch_size = 30
        for i in range(0, len(param_combinations), batch_size):
            batch = param_combinations[i:i+batch_size]
            batch_results = self._process_tsi_vwap_batch(data, symbol, batch, vwap)
            results.extend(batch_results)
        
        return results
    
    def _process_tsi_vwap_batch(self, data: pd.DataFrame, symbol: str, param_batch: List, vwap: pd.Series) -> List[Dict]:
        """Process a batch of TSI + VWAP parameters"""
        batch_results = []
        
        for tf, ts, tsig in param_batch:
            try:
                # Calculate TSI
                tsi = ta.tsi(data['close'], fast=tf, slow=ts, signal=tsig)
                if tsi is None or tsi.empty:
                    continue
                
                # Generate signals
                tsi_line = tsi.iloc[:, 0]
                tsi_signal_line = tsi.iloc[:, 1] if len(tsi.columns) > 1 else pd.Series(0, index=tsi_line.index)
                
                tsi_bullish = tsi_line > tsi_signal_line
                tsi_bearish = tsi_line < tsi_signal_line
                price_above_vwap = data['close'] > vwap
                price_below_vwap = data['close'] < vwap
                
                long_entries = tsi_bullish & price_above_vwap
                short_entries = tsi_bearish & price_below_vwap
                long_exits = tsi_bearish | price_below_vwap
                short_exits = tsi_bullish | price_above_vwap
                
                # Run VectorBT simulation
                pf = vbt.Portfolio.from_signals(
                    data['close'],
                    long_entries, long_exits,
                    short_entries, short_exits,
                    init_cash=self.initial_cash,
                    fees=self.commission,
                    freq='3min'
                )
                
                # Extract statistics
                stats = pf.stats()
                result = self._extract_portfolio_stats(stats, symbol, 'tsi_vwap',
                                                     f"TSI({tf},{ts},{tsig}), VWAP",
                                                     {'tsi_fast': tf, 'tsi_slow': ts, 'tsi_signal': tsig})
                
                if result and result.get('total_trades', 0) >= 5:
                    batch_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Error processing TSI+VWAP params ({tf},{ts},{tsig}): {e}")
                continue
        
        return batch_results
    
    def _extract_portfolio_stats(self, stats, symbol: str, strategy: str, params_str: str, parameters: Dict) -> Optional[Dict]:
        """Extract comprehensive portfolio statistics"""
        try:
            # Handle different VectorBT stat formats
            def safe_get_stat(stat_name: str, alternatives: List[str] = None, default=0.0):
                if alternatives is None:
                    alternatives = []
                
                all_names = [stat_name] + alternatives
                for name in all_names:
                    if hasattr(stats, 'get'):
                        val = stats.get(name)
                        if val is not None:
                            return float(val)
                    elif hasattr(stats, name):
                        val = getattr(stats, name)
                        if val is not None:
                            return float(val)
                    elif name in stats.index:
                        return float(stats[name])
                return default
            
            # Extract key statistics
            win_rate = safe_get_stat('Win Rate [%]', ['Win Rate', 'win_rate'])
            total_trades = int(safe_get_stat('Total Trades', ['Trades', '# Trades', 'total_trades']))
            total_return = safe_get_stat('Total Return [%]', ['Total Return', 'total_return'])
            max_drawdown = safe_get_stat('Max Drawdown [%]', ['Max Drawdown', 'max_drawdown'])
            profit_factor = safe_get_stat('Profit Factor', ['profit_factor'], 1.0)
            sharpe_ratio = safe_get_stat('Sharpe Ratio', ['sharpe_ratio'])
            calmar_ratio = safe_get_stat('Calmar Ratio', ['calmar_ratio'])
            
            # Calculate optimization score
            score = self._calculate_optimization_score(win_rate, total_return, max_drawdown, profit_factor, total_trades)
            
            return {
                'symbol': symbol,
                'strategy': strategy,
                'params': params_str,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'total_return': total_return,
                'max_drawdown': abs(max_drawdown),  # Ensure positive
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'score': score,
                'parameters': parameters
            }
            
        except Exception as e:
            self.logger.debug(f"Error extracting stats: {e}")
            return None
    
    def _calculate_optimization_score(self, win_rate: float, total_return: float, 
                                    max_drawdown: float, profit_factor: float, trade_count: int) -> float:
        """Calculate comprehensive optimization score"""
        try:
            # Weighted scoring system (0-100 scale)
            win_rate_score = min(win_rate, 100.0) * 0.25      # 25% weight
            return_score = max(total_return, 0) * 0.25         # 25% weight  
            drawdown_score = max(0, (50 - abs(max_drawdown))) * 0.20  # 20% weight
            profit_factor_score = min(max(profit_factor - 1, 0) * 20, 30) * 0.20  # 20% weight
            trade_count_score = min(trade_count / 20.0, 1.0) * 10 * 0.10  # 10% weight
            
            total_score = (win_rate_score + return_score + drawdown_score + 
                         profit_factor_score + trade_count_score)
            
            return round(total_score, 2)
        except:
            return 0.0

def get_viable_symbols_for_testing(exchange: Exchange, limit: int = 5) -> List[str]:
    """Get symbols that are both active and have sufficient data"""
    logger = logging.getLogger(__name__)
    
    try:
        # Get active symbols from exchange
        active_symbols_data = exchange.get_top_active_symbols(limit=limit*3)  # Get more candidates
        
        if active_symbols_data:
            candidate_symbols = [item['symbol'] for item in active_symbols_data]
        else:
            # Fallback to known stable symbols
            candidate_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
        
        # Filter based on data availability
        viable_symbols, symbol_info = filter_viable_symbols(exchange, candidate_symbols)
        
        if not viable_symbols:
            logger.warning("No viable symbols found in active list, using fallback symbols")
            fallback_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            viable_symbols, _ = filter_viable_symbols(exchange, fallback_symbols)
        
        return viable_symbols[:limit]  # Return only requested number
        
    except Exception as e:
        logger.error(f"Error getting viable symbols: {e}")
        return ['BTCUSDT', 'ETHUSDT'][:limit]

def run_intelligent_vectorbt_optimization():
    """Run intelligent VectorBT optimization with proper data validation"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Intelligent VectorBT Optimization with Data Validation")
    
    try:
        # Load exchange configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        exchange = Exchange(config['api_key'], config['api_secret'])
        tester = IntelligentVectorBTTester(exchange)
        
        # Get viable symbols with data validation
        viable_symbols = get_viable_symbols_for_testing(exchange, limit=3)
        
        if not viable_symbols:
            logger.error("No viable symbols found for testing")
            return
        
        logger.info(f"Testing strategies on viable symbols: {viable_symbols}")
        
        all_results = []
        
        # Process each viable symbol
        for symbol in viable_symbols:
            logger.info(f"\nProcessing symbol: {symbol}")
            
            # Get symbol data once
            symbol_info = get_symbol_data_info(exchange, symbol)
            if not symbol_info['available']:
                logger.error(f"Unexpected error: {symbol} data not available")
                continue
            
            symbol_data = symbol_info['data']
            symbol_results_count = 0
            
            # Test all strategies on this symbol
            for strategy_name in tester.strategies.keys():
                start_time = time.time()
                
                try:
                    results = tester.optimize_strategy_efficiently(symbol_data, symbol, strategy_name)
                    optimization_time = time.time() - start_time
                    
                    if results:
                        all_results.extend(results)
                        symbol_results_count += len(results)
                        
                        # Show top result
                        best_result = results[0]
                        logger.info(f"✓ {strategy_name}: {len(results)} combinations, "
                                  f"best score: {best_result['score']:.2f} "
                                  f"({optimization_time:.2f}s)")
                    else:
                        logger.warning(f"✗ {strategy_name}: No valid results ({optimization_time:.2f}s)")
                        
                except Exception as e:
                    optimization_time = time.time() - start_time
                    logger.error(f"✗ {strategy_name}: Error - {str(e)} ({optimization_time:.2f}s)")
            
            logger.info(f"Symbol {symbol} complete: {symbol_results_count} total valid combinations")
        
        # Export and summarize results
        if all_results:
            os.makedirs('vectorbt_results', exist_ok=True)
            
            df = pd.DataFrame(all_results)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = f'vectorbt_results/intelligent_optimization_{timestamp}.csv'
            df.to_csv(results_file, index=False)
            
            # Comprehensive summary
            logger.info(f"\nIntelligent VectorBT Optimization Results:")
            logger.info("=" * 60)
            
            # Overall statistics
            best_overall = df.loc[df['score'].idxmax()]
            strategy_summary = df.groupby('strategy').agg({
                'score': ['count', 'mean', 'max'],
                'win_rate': 'mean',
                'total_return': 'mean'
            }).round(2)
            
            logger.info(f"Total valid combinations: {len(all_results)}")
            logger.info(f"Symbols successfully processed: {df['symbol'].nunique()}")
            logger.info(f"Average optimization score: {df['score'].mean():.2f}")
            
            logger.info(f"\nBest Overall Result (Score: {best_overall['score']:.2f}):")
            logger.info(f"  Strategy: {best_overall['strategy']} on {best_overall['symbol']}")
            logger.info(f"  Parameters: {best_overall['params']}")
            logger.info(f"  Win Rate: {best_overall['win_rate']:.2f}%")
            logger.info(f"  Return: {best_overall['total_return']:.2f}%")
            logger.info(f"  Max Drawdown: {best_overall['max_drawdown']:.2f}%")
            logger.info(f"  Profit Factor: {best_overall['profit_factor']:.2f}")
            
            logger.info(f"\nStrategy Performance Summary:")
            logger.info(strategy_summary)
            
            logger.info(f"\nResults exported to: {results_file}")
        else:
            logger.error("No valid optimization results generated")
        
    except Exception as e:
        logger.error(f"Critical error in intelligent optimization: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    """Main function"""
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Intelligent VectorBT Strategy Optimizer")
    
    if not VECTORBT_AVAILABLE:
        logger.error("VectorBT is required for strategy optimization")
        logger.error("Install with: pip install vectorbt pandas-ta")
        return
    
    try:
        logger.info(f"VectorBT version: {vbt.__version__}")
    except AttributeError:
        logger.warning("Could not determine VectorBT version")
    
    run_intelligent_vectorbt_optimization()

if __name__ == "__main__":
    main()