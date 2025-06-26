"""
VectorBT-based Strategy Testing for Win Rate Optimization
Simplified implementation replacing custom backtester
"""

import os
import sys
import logging
import json
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
import time
from typing import Dict, List

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

try:
    import vectorbt as vbt
    import pandas_ta as ta
    VECTORBT_AVAILABLE = True
    print("VectorBT available for backtesting")
except ImportError:
    VECTORBT_AVAILABLE = False
    print("VectorBT not available. Install with: pip install vectorbt pandas-ta")
    sys.exit(1)

from core.exchange import Exchange

def setup_logging():
    """Setup logging for strategy testing"""
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG to see stat keys
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/vectorbt_optimization.log'),
            logging.StreamHandler()
        ]
    )
    os.makedirs('logs', exist_ok=True)

def get_active_symbols(exchange: Exchange, limit: int = 3) -> List[str]:
    """Get top active symbols from exchange"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Fetching top {limit} active symbols...")
        
        active_symbols_data = exchange.get_top_active_symbols(limit=limit)
        
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

def fetch_data_for_symbol(exchange: Exchange, symbol: str, limit: int = 1400) -> pd.DataFrame:
    """Fetch OHLCV data and convert to VectorBT format"""
    try:
        ohlcv_data = exchange.get_ohlcv(symbol, timeframe='3m', limit=limit)
        
        if not ohlcv_data or len(ohlcv_data) < 50:
            raise ValueError(f"Insufficient data: {len(ohlcv_data) if ohlcv_data else 0} candles")
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df.dropna()
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        raise

class VectorBTStrategyTester:
    """Simplified VectorBT-based strategy testing focused on win rate"""
    
    def __init__(self, exchange: Exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
        
        # VectorBT configuration
        self.initial_cash = 1.0
        self.commission = 0.00075  # 0.075%
        self.leverage = 20.0
        
        # Strategy configurations
        self.strategies = {
            'qqe_supertrend': {
                'name': 'QQE + Supertrend',
                'params': {
                    'qqe_length': range(8, 21),
                    'qqe_smooth': range(3, 9), 
                    'st_length': range(6, 16),
                    'st_multiplier': np.arange(2.0, 3.6, 0.2)
                }
            },
            'rsi_macd': {
                'name': 'RSI + MACD', 
                'params': {
                    'rsi_length': range(10, 21),
                    'macd_fast': range(8, 17),
                    'macd_slow': range(20, 36),
                    'macd_signal': range(6, 13)
                }
            },
            'tsi_vwap': {
                'name': 'TSI + VWAP',
                'params': {
                    'tsi_fast': range(5, 13),
                    'tsi_slow': range(12, 26),
                    'tsi_signal': range(4, 9)
                }
            }
        }
    
    def generate_qqe_supertrend_signals(self, data: pd.DataFrame, qqe_length: int, qqe_smooth: int, 
                                       st_length: int, st_multiplier: float) -> tuple:
        """Generate QQE + Supertrend signals using pandas_ta"""
        try:
            # QQE calculation using pandas_ta
            qqe = ta.qqe(data['close'], length=qqe_length, smooth=qqe_smooth)
            if qqe is None or qqe.empty or len(qqe.columns) < 2:
                return None, None
            
            qqe_line = qqe.iloc[:, 0]
            qqe_signal = qqe.iloc[:, 1]
            
            # Supertrend calculation using pandas_ta
            st = ta.supertrend(data['high'], data['low'], data['close'], 
                              length=st_length, multiplier=st_multiplier)
            if st is None or st.empty:
                return None, None
            
            # Find direction column
            dir_cols = [col for col in st.columns if 'SUPERTd' in col]
            if not dir_cols:
                return None, None
            
            st_direction = st[dir_cols[0]]
            
            # Generate signals
            qqe_bullish = qqe_signal > qqe_line
            st_bullish = st_direction == 1
            
            entries = qqe_bullish & st_bullish
            exits = (~qqe_bullish) & (~st_bullish)
            
            return entries, exits
            
        except Exception:
            return None, None
    
    def generate_rsi_macd_signals(self, data: pd.DataFrame, rsi_length: int, 
                                 macd_fast: int, macd_slow: int, macd_signal: int) -> tuple:
        """Generate RSI + MACD signals using pandas_ta"""
        try:
            if macd_fast >= macd_slow:
                return None, None
            
            # RSI calculation using pandas_ta
            rsi = ta.rsi(data['close'], length=rsi_length)
            if rsi is None:
                return None, None
            
            # MACD calculation using pandas_ta
            macd = ta.macd(data['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd is None or macd.empty or len(macd.columns) < 3:
                return None, None
            
            macd_line = macd.iloc[:, 0]
            macd_signal_line = macd.iloc[:, 2]
            
            # Generate signals
            entries = (rsi < 35) & (macd_line > macd_signal_line)
            exits = (rsi > 65) & (macd_line < macd_signal_line)
            
            return entries, exits
            
        except Exception:
            return None, None
    
    def generate_tsi_vwap_signals(self, data: pd.DataFrame, tsi_fast: int, 
                                 tsi_slow: int, tsi_signal: int) -> tuple:
        """Generate TSI + VWAP signals using pandas_ta"""
        try:
            if tsi_fast >= tsi_slow:
                return None, None
            
            # TSI calculation using pandas_ta
            tsi = ta.tsi(data['close'], fast=tsi_fast, slow=tsi_slow, signal=tsi_signal)
            if tsi is None or tsi.empty:
                return None, None
            
            tsi_line = tsi.iloc[:, 0]
            tsi_signal_line = tsi.iloc[:, 1] if len(tsi.columns) > 1 else 0
            
            # VWAP calculation using pandas_ta
            vwap = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
            if vwap is None:
                return None, None
            
            # Generate signals
            tsi_bullish = tsi_line > tsi_signal_line
            price_above_vwap = data['close'] > vwap
            
            entries = tsi_bullish & price_above_vwap
            exits = (~tsi_bullish) & (~price_above_vwap)
            
            return entries, exits
            
        except Exception:
            return None, None
    
    def test_strategy_on_symbol(self, symbol: str, strategy_name: str, max_combinations: int = 100) -> List[Dict]:
        """Test strategy on symbol using VectorBT optimization"""
        try:
            self.logger.info(f"Testing {strategy_name} on {symbol}")
            
            # Fetch data
            data = fetch_data_for_symbol(self.exchange, symbol)
            strategy_config = self.strategies[strategy_name]
            
            results = []
            tested_combinations = 0
            
            # Generate parameter combinations (limit to avoid memory issues)
            if strategy_name == 'qqe_supertrend':
                param_combinations = [
                    (ql, qs, sl, sm) 
                    for ql in list(strategy_config['params']['qqe_length'])[::2]  # Sample every 2nd
                    for qs in list(strategy_config['params']['qqe_smooth'])[::2]
                    for sl in list(strategy_config['params']['st_length'])[::2] 
                    for sm in list(strategy_config['params']['st_multiplier'])[::2]
                ][:max_combinations]
                
                for qqe_length, qqe_smooth, st_length, st_multiplier in param_combinations:
                    entries, exits = self.generate_qqe_supertrend_signals(
                        data, qqe_length, qqe_smooth, st_length, st_multiplier)
                    
                    if entries is not None and exits is not None:
                        pf = vbt.Portfolio.from_signals(
                            data['close'], entries, exits,
                            init_cash=self.initial_cash,
                            fees=self.commission,
                            freq='3min'
                        )
                        
                        stats = pf.stats()
                        
                        # Debug: Log available stat keys
                        self.logger.debug(f"Available VectorBT stats: {list(stats.index)}")
                        
                        # Handle different possible stat key names
                        trades_key = None
                        for key in ['Total Trades', 'Trades', '# Trades', 'total_trades']:
                            if key in stats.index:
                                trades_key = key
                                break
                        
                        win_rate_key = None  
                        for key in ['Win Rate [%]', 'Win Rate', 'win_rate']:
                            if key in stats.index:
                                win_rate_key = key
                                break
                        
                        return_key = None
                        for key in ['Total Return [%]', 'Total Return', 'total_return']:
                            if key in stats.index:
                                return_key = key
                                break
                        
                        drawdown_key = None
                        for key in ['Max Drawdown [%]', 'Max Drawdown', 'max_drawdown']:
                            if key in stats.index:
                                drawdown_key = key
                                break
                        
                        # Skip if essential stats are missing
                        if not all([trades_key, win_rate_key, return_key]):
                            self.logger.warning(f"Missing essential stats - trades: {trades_key}, win_rate: {win_rate_key}, return: {return_key}")
                            continue
                        
                        results.append({
                            'symbol': symbol,
                            'strategy': strategy_name,
                            'params': f"QQE({qqe_length},{qqe_smooth}), ST({st_length},{st_multiplier})",
                            'win_rate': float(stats[win_rate_key]) if win_rate_key else 0.0,
                            'total_trades': int(stats[trades_key]) if trades_key else 0,
                            'total_return': float(stats[return_key]) if return_key else 0.0,
                            'max_drawdown': float(stats[drawdown_key]) if drawdown_key else 0.0,
                            'profit_factor': float(stats.get('Profit Factor', 0)),
                            'parameters': {
                                'qqe_length': qqe_length,
                                'qqe_smooth': qqe_smooth, 
                                'st_length': st_length,
                                'st_multiplier': st_multiplier
                            }
                        })
                        tested_combinations += 1
            
            elif strategy_name == 'rsi_macd':
                param_combinations = [
                    (rl, mf, ms, msig)
                    for rl in list(strategy_config['params']['rsi_length'])[::2]
                    for mf in list(strategy_config['params']['macd_fast'])[::2]
                    for ms in list(strategy_config['params']['macd_slow'])[::2]
                    for msig in list(strategy_config['params']['macd_signal'])[::2]
                    if mf < ms  # Ensure fast < slow
                ][:max_combinations]
                
                for rsi_length, macd_fast, macd_slow, macd_signal in param_combinations:
                    entries, exits = self.generate_rsi_macd_signals(
                        data, rsi_length, macd_fast, macd_slow, macd_signal)
                    
                    if entries is not None and exits is not None:
                        pf = vbt.Portfolio.from_signals(
                            data['close'], entries, exits,
                            init_cash=self.initial_cash,
                            fees=self.commission,
                            freq='3min'
                        )
                        
                        stats = pf.stats()
                        
                        # Handle different possible stat key names
                        trades_key = None
                        for key in ['Total Trades', 'Trades', '# Trades', 'total_trades']:
                            if key in stats.index:
                                trades_key = key
                                break
                        
                        win_rate_key = None  
                        for key in ['Win Rate [%]', 'Win Rate', 'win_rate']:
                            if key in stats.index:
                                win_rate_key = key
                                break
                        
                        return_key = None
                        for key in ['Total Return [%]', 'Total Return', 'total_return']:
                            if key in stats.index:
                                return_key = key
                                break
                        
                        drawdown_key = None
                        for key in ['Max Drawdown [%]', 'Max Drawdown', 'max_drawdown']:
                            if key in stats.index:
                                drawdown_key = key
                                break
                        
                        # Skip if essential stats are missing
                        if not all([trades_key, win_rate_key, return_key]):
                            continue
                        
                        results.append({
                            'symbol': symbol,
                            'strategy': strategy_name,
                            'params': f"RSI({rsi_length}), MACD({macd_fast},{macd_slow},{macd_signal})",
                            'win_rate': float(stats[win_rate_key]) if win_rate_key else 0.0,
                            'total_trades': int(stats[trades_key]) if trades_key else 0,
                            'total_return': float(stats[return_key]) if return_key else 0.0,
                            'max_drawdown': float(stats[drawdown_key]) if drawdown_key else 0.0,
                            'profit_factor': float(stats.get('Profit Factor', 0)),
                            'parameters': {
                                'rsi_length': rsi_length,
                                'macd_fast': macd_fast,
                                'macd_slow': macd_slow, 
                                'macd_signal': macd_signal
                            }
                        })
                        tested_combinations += 1
            
            elif strategy_name == 'tsi_vwap':
                param_combinations = [
                    (tf, ts, tsig)
                    for tf in list(strategy_config['params']['tsi_fast'])[::2]
                    for ts in list(strategy_config['params']['tsi_slow'])[::2]
                    for tsig in list(strategy_config['params']['tsi_signal'])[::2]
                    if tf < ts  # Ensure fast < slow
                ][:max_combinations]
                
                for tsi_fast, tsi_slow, tsi_signal in param_combinations:
                    entries, exits = self.generate_tsi_vwap_signals(
                        data, tsi_fast, tsi_slow, tsi_signal)
                    
                    if entries is not None and exits is not None:
                        pf = vbt.Portfolio.from_signals(
                            data['close'], entries, exits,
                            init_cash=self.initial_cash,
                            fees=self.commission,
                            freq='3min'
                        )
                        
                        stats = pf.stats()
                        
                        # Handle different possible stat key names
                        trades_key = None
                        for key in ['Total Trades', 'Trades', '# Trades', 'total_trades']:
                            if key in stats.index:
                                trades_key = key
                                break
                        
                        win_rate_key = None  
                        for key in ['Win Rate [%]', 'Win Rate', 'win_rate']:
                            if key in stats.index:
                                win_rate_key = key
                                break
                        
                        return_key = None
                        for key in ['Total Return [%]', 'Total Return', 'total_return']:
                            if key in stats.index:
                                return_key = key
                                break
                        
                        drawdown_key = None
                        for key in ['Max Drawdown [%]', 'Max Drawdown', 'max_drawdown']:
                            if key in stats.index:
                                drawdown_key = key
                                break
                        
                        # Skip if essential stats are missing
                        if not all([trades_key, win_rate_key, return_key]):
                            continue
                        
                        results.append({
                            'symbol': symbol,
                            'strategy': strategy_name,
                            'params': f"TSI({tsi_fast},{tsi_slow},{tsi_signal}), VWAP",
                            'win_rate': float(stats[win_rate_key]) if win_rate_key else 0.0,
                            'total_trades': int(stats[trades_key]) if trades_key else 0,
                            'total_return': float(stats[return_key]) if return_key else 0.0,
                            'max_drawdown': float(stats[drawdown_key]) if drawdown_key else 0.0,
                            'profit_factor': float(stats.get('Profit Factor', 0)),
                            'parameters': {
                                'tsi_fast': tsi_fast,
                                'tsi_slow': tsi_slow,
                                'tsi_signal': tsi_signal
                            }
                        })
                        tested_combinations += 1
            
            self.logger.info(f"Tested {tested_combinations} combinations for {strategy_name} on {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing {strategy_name} on {symbol}: {e}")
            return []

def run_vectorbt_optimization():
    """Run VectorBT-based strategy optimization"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting VectorBT Win Rate Optimization")
    
    try:
        # Load exchange configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        exchange = Exchange(config['api_key'], config['api_secret'])
        tester = VectorBTStrategyTester(exchange)
        
        # Get test symbols
        test_symbols = get_active_symbols(exchange, limit=3)
        all_results = []
        
        logger.info(f"Testing strategies on symbols: {test_symbols}")
        
        # Test each strategy on each symbol
        for symbol in test_symbols:
            logger.info(f"\nTesting symbol: {symbol}")
            
            for strategy_name in tester.strategies.keys():
                logger.info(f"Processing: {strategy_name}")
                
                results = tester.test_strategy_on_symbol(symbol, strategy_name, max_combinations=50)
                all_results.extend(results)
                
                if results:
                    # Show best result for this combination
                    best_result = max(results, key=lambda x: x['win_rate'])
                    logger.info(f"Best {strategy_name} on {symbol}: "
                              f"Win Rate: {best_result['win_rate']:.2f}%, "
                              f"Trades: {best_result['total_trades']}, "
                              f"Return: {best_result['total_return']:.2f}%")
        
        # Export results
        if all_results:
            os.makedirs('vectorbt_results', exist_ok=True)
            
            df = pd.DataFrame(all_results)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = f'vectorbt_results/strategy_optimization_{timestamp}.csv'
            df.to_csv(results_file, index=False)
            
            # Summary statistics
            logger.info(f"\nVectorBT Optimization Summary:")
            logger.info("=" * 50)
            
            if len(all_results) > 0:
                avg_win_rate = df['win_rate'].mean()
                best_overall = df.loc[df['win_rate'].idxmax()]
                
                logger.info(f"Total combinations tested: {len(all_results)}")
                logger.info(f"Average win rate: {avg_win_rate:.2f}%")
                logger.info(f"Best overall result:")
                logger.info(f"  Strategy: {best_overall['strategy']} on {best_overall['symbol']}")
                logger.info(f"  Win Rate: {best_overall['win_rate']:.2f}%")
                logger.info(f"  Trades: {best_overall['total_trades']}")
                logger.info(f"  Return: {best_overall['total_return']:.2f}%")
                logger.info(f"  Parameters: {best_overall['params']}")
            
            logger.info(f"\nResults exported to: {results_file}")
        
    except Exception as e:
        logger.error(f"Critical error in VectorBT optimization: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    """Main function"""
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("VectorBT Strategy Optimizer")
    
    if not VECTORBT_AVAILABLE:
        logger.error("VectorBT is required for strategy optimization")
        logger.error("Install with: pip install vectorbt pandas-ta")
        return
    
    # Check VectorBT version compatibility
    try:
        logger.info(f"VectorBT version: {vbt.__version__}")
    except AttributeError:
        logger.warning("Could not determine VectorBT version")
    
    run_vectorbt_optimization()

if __name__ == "__main__":
    main()