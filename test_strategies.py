"""
Strategy Tester - Integration with existing folder structure
File: test_strategies.py (place in root directory)

This script integrates with your existing:
- core/exchange.py
- config.json
- Current folder structure
"""

import os
import sys
import logging
import json
from typing import Dict, List
import pandas as pd

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.backtester import IntegratedBacktester, BacktestConfig
from core.backtester import qqe_supertrend_signal_fixed, rsi_macd_signal, tsi_vwap_signal

def setup_logging():
    """Setup logging for strategy testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/strategy_testing.log'),
            logging.StreamHandler()
        ]
    )
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

def get_active_symbols(backtester: IntegratedBacktester, limit: int = 5) -> List[str]:
    """Get top active symbols from your exchange"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(" Fetching top active symbols from exchange...")
        
        # Use your exchange's method to get active symbols
        active_symbols_data = backtester.exchange.get_top_active_symbols(limit=limit)
        
        if active_symbols_data:
            symbols = [item['symbol'] for item in active_symbols_data]
            logger.info(f" Found active symbols: {symbols}")
            return symbols
        else:
            # Fallback to common crypto pairs
            fallback_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            logger.warning(f"⚠️ Using fallback symbols: {fallback_symbols}")
            return fallback_symbols
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error fetching active symbols: {e}")
        # Return some common symbols as fallback
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

def run_strategy_comparison():
    """Run comprehensive strategy comparison using live exchange data"""
    
    logger = logging.getLogger(__name__)
    logger.info("tarting Strategy Comparison with Live Exchange Data")
    
    try:
        # Initialize backtester with your config.json
        backtester = IntegratedBacktester('config.json')
        
        # Test configuration
        config = BacktestConfig(
            initial_capital=1.0,        # $1 per trade (matches your current system)
            leverage=20.0,              # 20x leverage
            commission_pct=0.075,       # Binance futures commission
            take_profit_pct=1.5,        # 1.5% TP
            stop_loss_pct=1.0,          # 1.0% SL
            timeframe='3m',             # 1-minute timeframe
            limit=1400,                 # 1400 candles as requested
            max_open_time_minutes=60    # Max 1 hour per trade
        )
        
        # Strategy definitions
        strategies = {
            'Current_QQE_ST_Fixed': {
                'indicators': {
                    'qqe': {'length': 12, 'smooth': 5, 'factor': 4.236},
                    'supertrend': {'length': 10, 'multiplier': 2.8}
                },
                'signal_func': qqe_supertrend_signal_fixed,
                'description': 'Current QQE + Supertrend with FIXED logic'
            },
            
            'Fast_QQE_ST': {
                'indicators': {
                    'qqe': {'length': 8, 'smooth': 3, 'factor': 4.236},
                    'supertrend': {'length': 6, 'multiplier': 2.2}
                },
                'signal_func': qqe_supertrend_signal_fixed,
                'description': 'Faster QQE + Supertrend for 3m crypto'
            },
            
            'RSI_MACD_Combo': {
                'indicators': {
                    'rsi': {'length': 14},
                    'macd': {'fast': 12, 'slow': 26, 'signal': 9}
                },
                'signal_func': rsi_macd_signal,
                'description': 'RSI Oversold/Overbought + MACD Confirmation'
            },
            
            'TSI_VWAP_Trend': {
                'indicators': {
                    'tsi': {'fast': 8, 'slow': 15, 'signal': 6},
                    'vwap': {}
                },
                'signal_func': tsi_vwap_signal,
                'description': 'Fast TSI + VWAP Trend Following'
            }
        }
        
        # Get symbols to test (using your exchange connection)
        test_symbols = get_active_symbols(backtester, limit=3)
        
        # Results storage
        all_results = []
        summary_data = []
        
        logger.info(f"Testing {len(strategies)} strategies on {len(test_symbols)} symbols")
        logger.info(f"Symbols: {test_symbols}")
        logger.info(f"Data: {config.limit} {config.timeframe} candles per symbol")
        
        # Run tests
        for symbol in test_symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f" TESTING SYMBOL: {symbol}")
            logger.info(f"{'='*60}")
            
            symbol_results = []
            
            for strategy_name, strategy_config in strategies.items():
                logger.info(f"\n   Running: {strategy_name}")
                logger.info(f"     {strategy_config['description']}")
                
                try:
                    # Run backtest with live data
                    result = backtester.run_backtest(
                        symbol=symbol,
                        signal_func=strategy_config['signal_func'],
                        indicator_config=strategy_config['indicators'],
                        backtest_config=config
                    )
                    
                    if 'error' not in result:
                        metrics = result['metrics']
                        
                        # Store results
                        detailed_result = {
                            'symbol': symbol,
                            'strategy': strategy_name,
                            'description': strategy_config['description'],
                            **metrics,
                            'data_points': result['data_points']
                        }
                        all_results.append(detailed_result)
                        symbol_results.append(detailed_result)
                        
                        # Export individual results
                        trades_file = backtester.export_results(symbol, strategy_name)
                        
                        # Log results
                        logger.info(f"      Trades: {metrics['total_trades']}")
                        logger.info(f"      Win Rate: {metrics['win_rate']}%")
                        logger.info(f"      Return: {metrics['total_return_pct']}%")
                        logger.info(f"     Profit Factor: {metrics['profit_factor']}")
                        logger.info(f"      Max DD: {metrics['max_drawdown_pct']}%")
                        if trades_file:
                            logger.info(f"      Exported: {trades_file}")
                    
                    else:
                        logger.error(f"     Error: {result['error']}")
                
                except Exception as e:
                    logger.error(f"     Exception: {e}")
            
            # Symbol summary
            if symbol_results:
                best_strategy = max(symbol_results, key=lambda x: x['total_return_pct'])
                logger.info(f"\n   Best for {symbol}: {best_strategy['strategy']}")
                logger.info(f"      Return: {best_strategy['total_return_pct']}%")
                logger.info(f"      Win Rate: {best_strategy['win_rate']}%")
        
        # Create final summary
        if all_results:
            logger.info(f"\n{'='*80}")
            logger.info("FINAL STRATEGY COMPARISON SUMMARY")
            logger.info(f"{'='*80}")
            
            # Group by strategy
            strategy_totals = {}
            for result in all_results:
                strategy = result['strategy']
                if strategy not in strategy_totals:
                    strategy_totals[strategy] = {
                        'total_return': 0,
                        'avg_win_rate': 0,
                        'avg_profit_factor': 0,
                        'max_drawdown': 0,
                        'total_trades': 0,
                        'symbols_tested': 0
                    }
                
                totals = strategy_totals[strategy]
                totals['total_return'] += result['total_return_pct']
                totals['avg_win_rate'] += result['win_rate']
                totals['avg_profit_factor'] += result['profit_factor']
                totals['max_drawdown'] = max(totals['max_drawdown'], result['max_drawdown_pct'])
                totals['total_trades'] += result['total_trades']
                totals['symbols_tested'] += 1
            
            # Calculate averages
            for strategy, totals in strategy_totals.items():
                count = totals['symbols_tested']
                totals['avg_win_rate'] = round(totals['avg_win_rate'] / count, 1)
                totals['avg_profit_factor'] = round(totals['avg_profit_factor'] / count, 2)
                totals['total_return'] = round(totals['total_return'], 2)
            
            # Print summary table
            print(f"\n{'Strategy':<25} {'Total Return':<12} {'Avg Win%':<10} {'Avg PF':<8} {'Max DD%':<8} {'Trades':<8}")
            print("-" * 85)
            
            # Sort by total return
            sorted_strategies = sorted(strategy_totals.items(), 
                                     key=lambda x: x[1]['total_return'], 
                                     reverse=True)
            
            for strategy_name, totals in sorted_strategies:
                print(f"{strategy_name:<25} {totals['total_return']:>10}% "
                      f"{totals['avg_win_rate']:>8}% "
                      f"{totals['avg_profit_factor']:>6} "
                      f"{totals['max_drawdown']:>6}% "
                      f"{totals['total_trades']:>6}")
            
            # Export summary
            os.makedirs('backtest_results', exist_ok=True)
            
            # Detailed results
            detailed_df = pd.DataFrame(all_results)
            detailed_df.to_csv('backtest_results/live_data_detailed_results.csv', index=False)
            
            # Summary
            summary_df = pd.DataFrame([
                {
                    'strategy': name,
                    'description': strategies[name]['description'],
                    **totals
                }
                for name, totals in sorted_strategies
            ])
            summary_df.to_csv('backtest_results/live_data_strategy_summary.csv', index=False)
            
            # Winner announcement
            if sorted_strategies:
                winner = sorted_strategies[0]
                logger.info(f"\ OVERALL WINNER: {winner[0]}")
                logger.info(f"    Total Return: {winner[1]['total_return']}%")
                logger.info(f"    Avg Win Rate: {winner[1]['avg_win_rate']}%")
                logger.info(f"   Avg Profit Factor: {winner[1]['avg_profit_factor']}")
                logger.info(f"    Max Drawdown: {winner[1]['max_drawdown']}%")
                logger.info(f"    Total Trades: {winner[1]['total_trades']}")
                logger.info(f"    Description: {strategies[winner[0]]['description']}")
            
            logger.info(f"\ Results exported to:")
            logger.info(f"   Summary: backtest_results/live_data_strategy_summary.csv")
            logger.info(f"    Detailed: backtest_results/live_data_detailed_results.csv")
            logger.info(f"    Individual trades: backtest_results/*_trades_*.csv")
        
        else:
            logger.warning("No successful backtests completed")
    
    except Exception as e:
        logger.error(f"Critical error in strategy comparison: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def test_single_symbol_all_strategies(symbol: str = None):
    """Test all strategies on a single symbol for detailed analysis"""
    
    logger = logging.getLogger(__name__)
    
    try:
        backtester = IntegratedBacktester('config.json')
        
        # Use provided symbol or get one active symbol
        if symbol is None:
            active_symbols = get_active_symbols(backtester, limit=1)
            symbol = active_symbols[0] if active_symbols else 'BTCUSDT'
        
        logger.info(f" DETAILED TESTING ON: {symbol}")
        
        config = BacktestConfig(
            initial_capital=1.0,
            leverage=20.0,
            commission_pct=0.075,
            take_profit_pct=1.5,
            stop_loss_pct=1.0,
            timeframe='3m',
            limit=1400
        )
        
        strategies = {
            'Current_Fixed': {
                'indicators': {'qqe': {'length': 12, 'smooth': 5, 'factor': 4.236}, 'supertrend': {'length': 10, 'multiplier': 2.8}},
                'signal_func': qqe_supertrend_signal_fixed
            },
            'Fast_Version': {
                'indicators': {'qqe': {'length': 8, 'smooth': 3, 'factor': 4.236}, 'supertrend': {'length': 6, 'multiplier': 2.2}},
                'signal_func': qqe_supertrend_signal_fixed
            },
            'RSI_MACD': {
                'indicators': {'rsi': {'length': 14}, 'macd': {'fast': 12, 'slow': 26, 'signal': 9}},
                'signal_func': rsi_macd_signal
            }
        }
        
        results = []
        
        for strategy_name, strategy_config in strategies.items():
            logger.info(f"\n Testing {strategy_name}...")
            
            result = backtester.run_backtest(
                symbol=symbol,
                signal_func=strategy_config['signal_func'],
                indicator_config=strategy_config['indicators'],
                backtest_config=config
            )
            
            if 'error' not in result:
                metrics = result['metrics']
                results.append((strategy_name, metrics))
                
                backtester.export_results(symbol, strategy_name)
                
                logger.info(f"   Trades: {metrics['total_trades']}, Win Rate: {metrics['win_rate']}%, Return: {metrics['total_return_pct']}%")
        
        # Compare results
        if results:
            logger.info(f"\nCOMPARISON FOR {symbol}:")
            for strategy_name, metrics in sorted(results, key=lambda x: x[1]['total_return_pct'], reverse=True):
                logger.info(f"   {strategy_name}: {metrics['total_return_pct']}% return, {metrics['win_rate']}% win rate")
    
    except Exception as e:
        logger.error(f"Error in single symbol test: {e}")

def main():
    """Main function to run strategy testing"""
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("tarting Integrated Strategy Testing")
    
    import argparse
    parser = argparse.ArgumentParser(description='Test trading strategies with live exchange data')
    parser.add_argument('--mode', choices=['full', 'single'], default='full',
                       help='Run full comparison or single symbol test')
    parser.add_argument('--symbol', type=str, help='Symbol for single mode')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_strategy_comparison()
    else:
        test_single_symbol_all_strategies(args.symbol)

if __name__ == "__main__":
    main()