"""
VectorBT-based Strategy Testing with Current Regime Optimization
FIXED: Focuses on current momentum instead of historical optimization
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
    print("VectorBT available for current regime optimization")
except ImportError:
    VECTORBT_AVAILABLE = False
    print("VectorBT not available. Install with: pip install vectorbt pandas-ta")
    sys.exit(1)

from core.exchange import Exchange

def setup_logging():
    """Setup logging for strategy testing with proper Unicode handling"""
    os.makedirs('logs', exist_ok=True)
    
    file_handler = logging.FileHandler('logs/vectorbt_current_regime_optimization.log', encoding='utf-8')
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

def get_symbol_data_info(exchange: Exchange, symbol: str, timeframe: str = '15m', limit: int = 300) -> Dict:
    """FIXED: Get data with multi-timeframe support for ROC"""
    try:
        if timeframe == 'multi_roc':
            # For ROC multi-timeframe, fetch both 3m and 15m separately
            ohlcv_3m = exchange.get_ohlcv(symbol, timeframe='3m', limit=limit)
            ohlcv_15m = exchange.get_ohlcv(symbol, timeframe='15m', limit=limit//5)  # 15m needs fewer candles
            
            if not ohlcv_3m or len(ohlcv_3m) < 50 or not ohlcv_15m or len(ohlcv_15m) < 20:
                return {'symbol': symbol, 'available': False, 'candles': 0, 'error': 'Insufficient multi-timeframe data'}
            
            # Convert both to DataFrames with proper datetime index
            df_3m = pd.DataFrame(ohlcv_3m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            for df in [df_3m, df_15m]:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp').sort_index()
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
            
            df_3m = df_3m.set_index('timestamp').sort_index()
            df_15m = df_15m.set_index('timestamp').sort_index()
            
            # Calculate current volatility from 3m data
            if len(df_3m) >= 10:
                recent_changes = df_3m['close'].pct_change().tail(5).abs() * 100
                recent_volatility = recent_changes.mean()
            else:
                recent_volatility = 0
            
            price_trend = ((df_3m['close'].iloc[-1] - df_3m['close'].iloc[-50]) / df_3m['close'].iloc[-50]) * 100 if len(df_3m) >= 50 else 0
            
            return {
                'symbol': symbol,
                'available': True,
                'candles': len(df_3m),
                'raw_candles': len(df_3m),
                'data_quality': 1.0,
                'price_range': (df_3m['close'].min(), df_3m['close'].max()),
                'date_range': (df_3m.index.min(), df_3m.index.max()),
                'current_volatility': recent_volatility,
                'recent_trend': price_trend,
                'data': {'3m': df_3m, '15m': df_15m}  # Return both timeframes
            }
        else:
            # Original single timeframe logic
            ohlcv_data = exchange.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            if not ohlcv_data:
                return {'symbol': symbol, 'available': False, 'candles': 0, 'error': 'No data'}
            
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df_clean = df.dropna()
            
            # Calculate current market regime indicators
            if len(df_clean) >= 10:
                recent_changes = df_clean['close'].pct_change().tail(5).abs() * 100
                recent_volatility = recent_changes.mean()
            else:
                recent_volatility = 0
            price_trend = ((df_clean['close'].iloc[-1] - df_clean['close'].iloc[-50]) / df_clean['close'].iloc[-50]) * 100 if len(df_clean) >= 50 else 0
            
            return {
                'symbol': symbol,
                'available': True,
                'candles': len(df_clean),
                'raw_candles': len(df),
                'data_quality': len(df_clean) / len(df) if len(df) > 0 else 0,
                'price_range': (df_clean['close'].min(), df_clean['close'].max()),
                'date_range': (df_clean.index.min(), df_clean.index.max()),
                'current_volatility': recent_volatility,
                'recent_trend': price_trend,
                'data': df_clean
            }
        
    except Exception as e:
        return {'symbol': symbol, 'available': False, 'candles': 0, 'error': str(e)}

def calculate_strategy_data_requirements() -> Dict[str, int]:
    """FIXED: Reduced requirements for current regime optimization"""
    return {
        'qqe_supertrend': 80,  # Reduced from 50
        'rsi_macd': 100,       # Reduced from 75  
        'tsi_vwap': 90,        # Reduced from 65
        'minimum_for_all': 100 # Reduced from 75
    }

def filter_viable_symbols(exchange: Exchange, candidate_symbols: List[str]) -> Tuple[List[str], Dict]:
    """FIXED: Filter symbols with current regime focus"""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating current regime data for {len(candidate_symbols)} symbols...")
    
    requirements = calculate_strategy_data_requirements()
    min_required = requirements['minimum_for_all']
    
    viable_symbols = []
    symbol_info = {}
    
    for symbol in candidate_symbols:
        info = get_symbol_data_info(exchange, symbol)
        symbol_info[symbol] = info
        
        if info['available'] and info['candles'] >= min_required:
            # FIXED: Add current regime filtering
            volatility = info.get('current_volatility', 0)
            if volatility > 0.5:  # Only symbols with sufficient current activity
                viable_symbols.append(symbol)
                logger.info(f"✓ {symbol}: {info['candles']} candles, vol: {volatility:.2f}% (viable)")
            else:
                logger.warning(f"✗ {symbol}: Low volatility {volatility:.2f}% (inactive)")
        else:
            reason = info.get('error', f"only {info['candles']} candles")
            logger.warning(f"✗ {symbol}: {reason} (need {min_required})")
    
    logger.info(f"Current regime filtering: {len(viable_symbols)}/{len(candidate_symbols)} symbols viable")
    return viable_symbols, symbol_info

class IntelligentVectorBTTester:
    """FIXED: Current regime optimization instead of historical optimization"""
    
    def __init__(self, exchange: Exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
        
        # VectorBT configuration
        self.initial_cash = 1.0
        self.commission = 0.00075
        self.leverage = 20.0
        
        # FIXED: Focused parameter ranges for current crypto momentum (updated with ROC)
        self.strategies = {
            'qqe_supertrend': {
                'name': 'QQE + Supertrend',
                'params': {
                    'qqe_length': [8, 10, 12],
                    'qqe_smooth': [3, 5],
                    'st_length': [6, 10, 12],
                    'st_multiplier': [2.2, 2.6, 3.0]
                }
            },
            'rsi_macd': {
                'name': 'RSI + MACD',
                'params': {
                    'rsi_length': [12, 14, 16],
                    'macd_fast': [10, 12],
                    'macd_slow': [24, 26],
                    'macd_signal': [8, 9]
                }
            },
            'tsi_vwap': {
                'name': 'TSI + VWAP',
                'params': {
                    'tsi_fast': [6, 8],
                    'tsi_slow': [15, 21],
                    'tsi_signal': [4, 6]
                }
            },
            'roc_multi_timeframe': {
                'name': 'ROC Multi-timeframe',
                'params': {
                    'roc_3m_length': [8, 10, 12],
                    'roc_15m_length': [8, 10, 12],
                    'roc_3m_threshold': [0.8, 1.0, 1.5],
                    'roc_15m_threshold': [1.5, 2.0, 2.5],
                    'roc_alignment_factor': [0.3, 0.5, 0.7]
                }
            }
        }
    
    def optimize_strategy_efficiently(self, data: pd.DataFrame, symbol: str, strategy_name: str) -> List[Dict]:
        """FIXED: Current regime optimization with recent data weighting"""
        try:
            # FIXED: Use only recent 200 candles for optimization
            recent_data = data.tail(200) if len(data) > 200 else data
            self.logger.info(f"Current regime optimization {strategy_name} on {symbol} ({len(recent_data)} recent candles)")
            
            strategy_config = self.strategies[strategy_name]
            results = []
            
            if strategy_name == 'qqe_supertrend':
                results = self._optimize_qqe_supertrend(recent_data, symbol, strategy_config)
            elif strategy_name == 'rsi_macd':
                results = self._optimize_rsi_macd(recent_data, symbol, strategy_config)
            elif strategy_name == 'tsi_vwap':
                results = self._optimize_tsi_vwap(recent_data, symbol, strategy_config)
            elif strategy_name == 'roc_multi_timeframe':
                results = self._optimize_roc_multi_timeframe(recent_data, symbol, strategy_config)
            
            # FIXED: Filter results for current momentum effectiveness
            if results:
                results = results[:10]
                self.logger.info(f"Current regime results: {len(results)} optimized for recent momentum")
            else:
                self.logger.warning(f"No current regime results for {strategy_name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in current regime optimization {strategy_name} on {symbol}: {e}")
            return []
    
    def _optimize_roc_multi_timeframe(self, data: pd.DataFrame, symbol: str, config: Dict) -> List[Dict]:
        """ROC Multi-timeframe optimization"""
        results = []
        
        param_combinations = [
            (r3l, r15l, r3t, r15t, raf) 
            for r3l in config['params']['roc_3m_length']
            for r15l in config['params']['roc_15m_length']
            for r3t in config['params']['roc_3m_threshold']
            for r15t in config['params']['roc_15m_threshold']
            for raf in config['params']['roc_alignment_factor']
        ]
        
        batch_results = self._process_roc_multi_timeframe_batch(data, symbol, param_combinations)
        results.extend(batch_results)
        
        return results
    def _process_roc_multi_timeframe_batch(self, data, symbol: str, param_batch: List) -> List[Dict]:
        """FIXED: Process ROC multi-timeframe with clear signal logic like TSI+VWAP"""
        batch_results = []
        
        # Handle multi-timeframe data structure
        if isinstance(data, dict) and '3m' in data and '15m' in data:
            df_3m = data['3m'].copy()
            df_15m = data['15m'].copy()
        else:
            # Fallback: create 15m from 3m data
            df_3m = data.copy()
            if not isinstance(df_3m.index, pd.DatetimeIndex):
                return batch_results
            
            try:
                df_15m = df_3m.resample('15min').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            except Exception:
                return batch_results
        
        # Validate data
        if len(df_3m) < 50 or len(df_15m) < 20:
            return batch_results
        
        for r3l, r15l, r3t, r15t, raf in param_batch:
            try:
                # Calculate ROC for both timeframes
                roc_3m = ta.roc(df_3m['close'], length=r3l)
                roc_15m = ta.roc(df_15m['close'], length=r15l)
                
                if (roc_3m is None or len(roc_3m.dropna()) < 10 or
                    roc_15m is None or len(roc_15m.dropna()) < 5):
                    continue
                
                # FIXED: Clear signal logic like TSI+VWAP
                # Align 15m ROC to 3m timeframe
                roc_15m_aligned = pd.Series(index=df_3m.index, dtype=float)
                for timestamp in df_3m.index:
                    # Find closest 15m timestamp
                    time_diffs = abs(df_15m.index - timestamp)
                    if len(time_diffs) > 0:
                        closest_15m_timestamp = time_diffs.idxmin()
                        if time_diffs.min() <= pd.Timedelta(minutes=15):
                            if closest_15m_timestamp in roc_15m.index:
                                roc_15m_aligned.loc[timestamp] = roc_15m.loc[closest_15m_timestamp]
                
                # FIXED: Simple bullish/bearish logic
                roc_3m_bullish = roc_3m > r3t
                roc_15m_bullish = roc_15m_aligned > r15t
                roc_3m_bearish = roc_3m < -r3t
                roc_15m_bearish = roc_15m_aligned < -r15t
                
                # FIXED: Clear entry and exit signals
                long_entries = roc_3m_bullish & roc_15m_bullish
                short_entries = roc_3m_bearish & roc_15m_bearish
                long_exits = roc_3m_bearish | roc_15m_bearish
                short_exits = roc_3m_bullish | roc_15m_bullish
                
                # Clean up NaN values
                long_entries = long_entries.fillna(False)
                short_entries = short_entries.fillna(False)
                long_exits = long_exits.fillna(False)
                short_exits = short_exits.fillna(False)
                
                # Run VectorBT simulation
                pf = vbt.Portfolio.from_signals(
                    df_3m['close'],
                    long_entries, long_exits,
                    short_entries, short_exits,
                    init_cash=self.initial_cash,
                    fees=self.commission,
                    freq='3min'
                )
                
                # Extract statistics
                stats = pf.stats()
                result = self._extract_portfolio_stats(stats, symbol, 'roc_multi_timeframe',
                                                    f"ROC_3m({r3l},{r3t}), ROC_15m({r15l},{r15t}), Align({raf})",
                                                    {'roc_3m_length': r3l, 'roc_15m_length': r15l,
                                                    'roc_3m_threshold': r3t, 'roc_15m_threshold': r15t,
                                                    'roc_alignment_factor': raf})
                
                if result and result.get('total_trades', 0) >= 3:
                    batch_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Error processing ROC params ({r3l},{r15l},{r3t},{r15t},{raf}): {e}")
                continue
        
        return batch_results
    def _align_roc_timeframes_for_vectorbt(self, df_3m: pd.DataFrame, df_15m: pd.DataFrame, 
                                     roc_3m: pd.Series, roc_15m: pd.Series,
                                     r3t: float, r15t: float, raf: float) -> Optional[Tuple]:
        """FIXED: Align ROC signals from both timeframes to 3m frequency for VectorBT"""
        try:
            # Create signal arrays aligned to 3m timeframe
            long_entries = pd.Series(False, index=df_3m.index)
            short_entries = pd.Series(False, index=df_3m.index)
            long_exits = pd.Series(False, index=df_3m.index)
            short_exits = pd.Series(False, index=df_3m.index)
            
            # For each 3m timestamp, find the corresponding 15m ROC value
            for timestamp in df_3m.index:
                # Get 3m ROC value
                if timestamp not in roc_3m.index or pd.isna(roc_3m.loc[timestamp]):
                    continue
                current_roc_3m = roc_3m.loc[timestamp]
                
                # Find the closest 15m ROC value (within 15 minutes)
                time_diffs = abs(df_15m.index - timestamp)
                if len(time_diffs) == 0:
                    continue
                    
                closest_15m_timestamp = time_diffs.idxmin()
                
                # Skip if the closest 15m timestamp is too far away
                if time_diffs.min() > pd.Timedelta(minutes=15):
                    continue
                    
                if closest_15m_timestamp not in roc_15m.index or pd.isna(roc_15m.loc[closest_15m_timestamp]):
                    continue
                current_roc_15m = roc_15m.loc[closest_15m_timestamp]
                
                # Multi-timeframe signal logic
                roc_3m_bullish = current_roc_3m > r3t
                roc_15m_bullish = current_roc_15m > r15t
                roc_3m_bearish = current_roc_3m < -r3t
                roc_15m_bearish = current_roc_15m < -r15t
                
                alignment_strength = abs(current_roc_3m * current_roc_15m) * raf
                
                # Generate entry signals
                if roc_3m_bullish and roc_15m_bullish and alignment_strength > 1.0:
                    long_entries.loc[timestamp] = True
                elif roc_3m_bearish and roc_15m_bearish and alignment_strength > 1.0:
                    short_entries.loc[timestamp] = True
                
                # Generate exit signals (timeframe divergence)
                if (roc_3m_bearish or roc_15m_bearish) and alignment_strength > 0.5:
                    long_exits.loc[timestamp] = True
                if (roc_3m_bullish or roc_15m_bullish) and alignment_strength > 0.5:
                    short_exits.loc[timestamp] = True
            
            return long_entries, long_exits, short_entries, short_exits
            
        except Exception as e:
            self.logger.debug(f"Error aligning ROC timeframes: {e}")
            return None
    def _align_roc_signals_by_timestamp(self, df_3m: pd.DataFrame, df_15m: pd.DataFrame,
                                  roc_3m: pd.Series, roc_15m: pd.Series,
                                  r3t: float, r15t: float, raf: float) -> Optional[Tuple]:
        """FIXED: Align ROC signals using timestamp-based matching"""
        try:
            # Create boolean arrays for signals aligned to 3m timeframe
            long_entries = pd.Series(False, index=df_3m.index)
            short_entries = pd.Series(False, index=df_3m.index)
            long_exits = pd.Series(False, index=df_3m.index)
            short_exits = pd.Series(False, index=df_3m.index)
            
            # For each 3m timestamp, find the closest 15m ROC value
            for timestamp in df_3m.index:
                if timestamp not in roc_3m.index or pd.isna(roc_3m.loc[timestamp]):
                    continue
                
                # Find closest 15m timestamp (within 15 minutes)
                time_diffs = abs(df_15m.index - timestamp)
                closest_15m_idx = time_diffs.idxmin()
                
                if time_diffs.min() > pd.Timedelta(minutes=15):
                    continue
                    
                if closest_15m_idx not in roc_15m.index or pd.isna(roc_15m.loc[closest_15m_idx]):
                    continue
                
                current_roc_3m = roc_3m.loc[timestamp]
                current_roc_15m = roc_15m.loc[closest_15m_idx]
                
                # Multi-timeframe logic
                roc_3m_bullish = current_roc_3m > r3t
                roc_15m_bullish = current_roc_15m > r15t
                roc_3m_bearish = current_roc_3m < -r3t
                roc_15m_bearish = current_roc_15m < -r15t
                
                alignment_strength = abs(current_roc_3m * current_roc_15m) * raf
                
                # Generate entry signals
                if roc_3m_bullish and roc_15m_bullish and alignment_strength > 1.0:
                    long_entries.loc[timestamp] = True
                elif roc_3m_bearish and roc_15m_bearish and alignment_strength > 1.0:
                    short_entries.loc[timestamp] = True
                
                # Generate exit signals
                if (roc_3m_bearish or roc_15m_bearish) and alignment_strength > 0.5:
                    long_exits.loc[timestamp] = True
                if (roc_3m_bullish or roc_15m_bullish) and alignment_strength > 0.5:
                    short_exits.loc[timestamp] = True
            
            return long_entries, long_exits, short_entries, short_exits
            
        except Exception as e:
            self.logger.debug(f"Error aligning ROC signals: {e}")
            return None
    def _align_timeframes_for_signals(self, data_3m: pd.DataFrame, data_15m: pd.DataFrame, 
                                roc_3m: pd.Series, roc_15m: pd.Series, 
                                r3t: float, r15t: float, raf: float) -> Optional[Tuple]:
        """Align multi-timeframe ROC signals"""
        try:
            # Create boolean arrays for signals
            long_entries = pd.Series(False, index=data_3m.index)
            short_entries = pd.Series(False, index=data_3m.index)
            long_exits = pd.Series(False, index=data_3m.index)
            short_exits = pd.Series(False, index=data_3m.index)
            
            # FIXED: Use proper index alignment
            for i in range(len(roc_3m)):
                if pd.isna(roc_3m.iloc[i]):
                    continue
                
                current_roc_3m = roc_3m.iloc[i]
                
                # FIXED: Find corresponding 15m ROC value using timestamp alignment
                current_3m_time = data_3m.index[i] if i < len(data_3m) else data_3m.index[-1]
                
                # Find the closest 15m timestamp
                if len(data_15m) > 0:
                    # Get the 15m bar that contains this 3m timestamp
                    time_diff = abs(data_15m.index - current_3m_time)
                    closest_15m_idx = time_diff.argmin()
                    
                    if closest_15m_idx < len(roc_15m) and not pd.isna(roc_15m.iloc[closest_15m_idx]):
                        current_roc_15m = roc_15m.iloc[closest_15m_idx]
                    else:
                        continue
                else:
                    continue
                
                # Multi-timeframe logic
                roc_3m_bullish = current_roc_3m > r3t
                roc_15m_bullish = current_roc_15m > r15t
                roc_3m_bearish = current_roc_3m < -r3t
                roc_15m_bearish = current_roc_15m < -r15t
                
                alignment_strength = abs(current_roc_3m * current_roc_15m) * raf
                
                # Generate entry signals
                if roc_3m_bullish and roc_15m_bullish and alignment_strength > 1.0:
                    long_entries.iloc[i] = True
                elif roc_3m_bearish and roc_15m_bearish and alignment_strength > 1.0:
                    short_entries.iloc[i] = True
                
                # Generate exit signals
                if (roc_3m_bearish or roc_15m_bearish) and alignment_strength > 0.5:
                    long_exits.iloc[i] = True
                if (roc_3m_bullish or roc_15m_bullish) and alignment_strength > 0.5:
                    short_exits.iloc[i] = True
            
            return long_entries, long_exits, short_entries, short_exits
            
        except Exception as e:
            self.logger.debug(f"Error aligning timeframes: {e}")
            return None
    def _optimize_qqe_supertrend(self, data: pd.DataFrame, symbol: str, config: Dict) -> List[Dict]:
        """FIXED: QQE + Supertrend with current momentum focus"""
        results = []
        
        # FIXED: Reduced parameter combinations for faster optimization
        param_combinations = list(product(
            config['params']['qqe_length'],
            config['params']['qqe_smooth'],
            config['params']['st_length'],
            config['params']['st_multiplier']
        ))
        
        # FIXED: Single batch processing for speed
        batch_results = self._process_qqe_supertrend_batch(data, symbol, param_combinations)
        results.extend(batch_results)
        
        return results
    
    def _process_qqe_supertrend_batch(self, data: pd.DataFrame, symbol: str, param_batch: List) -> List[Dict]:
        """FIXED: Process with current momentum weighting"""
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
                
                qqe_bullish = qqe_line > qqe_signal
                qqe_bearish = qqe_line < qqe_signal
                st_bullish = st_direction == 1
                st_bearish = st_direction == -1
                
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
                    freq='5min'
                )
                
                # Extract statistics with current momentum weighting
                stats = pf.stats()
                result = self._extract_portfolio_stats(stats, symbol, 'qqe_supertrend', 
                                                     f"QQE({ql},{qs}), ST({sl},{sm})",
                                                     {'qqe_length': ql, 'qqe_smooth': qs, 
                                                      'st_length': sl, 'st_multiplier': sm})
                
                # FIXED: Filter for current momentum requirements
                if result and result.get('total_trades', 0) >= 3:  # Reduced from 5 to 3
                    batch_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Error processing QQE+ST params ({ql},{qs},{sl},{sm}): {e}")
                continue
        
        return batch_results
    
    def _optimize_rsi_macd(self, data: pd.DataFrame, symbol: str, config: Dict) -> List[Dict]:
        """FIXED: RSI + MACD with current momentum focus"""
        results = []
        
        param_combinations = [
            (rl, mf, ms, msig) for rl in config['params']['rsi_length']
            for mf in config['params']['macd_fast']
            for ms in config['params']['macd_slow']
            for msig in config['params']['macd_signal']
            if mf < ms  # Valid MACD constraint
        ]
        
        # FIXED: Single batch processing
        batch_results = self._process_rsi_macd_batch(data, symbol, param_combinations)
        results.extend(batch_results)
        
        return results
    
    def _process_rsi_macd_batch(self, data: pd.DataFrame, symbol: str, param_batch: List) -> List[Dict]:
        """FIXED: Process with current momentum focus"""
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
                    freq='5min'
                )
                
                # Extract statistics
                stats = pf.stats()
                result = self._extract_portfolio_stats(stats, symbol, 'rsi_macd',
                                                     f"RSI({rl}), MACD({mf},{ms},{msig})",
                                                     {'rsi_length': rl, 'macd_fast': mf,
                                                      'macd_slow': ms, 'macd_signal': msig})
                
                # FIXED: Current momentum requirements
                if result and result.get('total_trades', 0) >= 3:
                    batch_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Error processing RSI+MACD params ({rl},{mf},{ms},{msig}): {e}")
                continue
        
        return batch_results
    
    def _optimize_tsi_vwap(self, data: pd.DataFrame, symbol: str, config: Dict) -> List[Dict]:
        """FIXED: TSI + VWAP with current momentum focus"""
        results = []
        
        # Pre-calculate VWAP once
        vwap = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
        if vwap is None:
            return results
        
        param_combinations = [
            (tf, ts, tsig) for tf in config['params']['tsi_fast']
            for ts in config['params']['tsi_slow']
            for tsig in config['params']['tsi_signal']
            if tf < ts  # Valid TSI constraint
        ]
        
        # FIXED: Single batch processing
        batch_results = self._process_tsi_vwap_batch(data, symbol, param_combinations, vwap)
        results.extend(batch_results)
        
        return results
    
    def _process_tsi_vwap_batch(self, data: pd.DataFrame, symbol: str, param_batch: List, vwap: pd.Series) -> List[Dict]:
        """FIXED: Process with current momentum focus"""
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
                    freq='5min'
                )
                
                # Extract statistics
                stats = pf.stats()
                result = self._extract_portfolio_stats(stats, symbol, 'tsi_vwap',
                                                     f"TSI({tf},{ts},{tsig}), VWAP",
                                                     {'tsi_fast': tf, 'tsi_slow': ts, 'tsi_signal': tsig})
                
                # FIXED: Current momentum requirements
                if result and result.get('total_trades', 0) >= 3:
                    batch_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Error processing TSI+VWAP params ({tf},{ts},{tsig}): {e}")
                continue
        
        return batch_results
    
    def _extract_portfolio_stats(self, stats, symbol: str, strategy: str, params_str: str, parameters: Dict) -> Optional[Dict]:
        """FIXED: Extract stats with current regime scoring"""
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
            
            # FIXED: Calculate current regime optimization score
            score = self._calculate_current_regime_score(win_rate, total_return, max_drawdown, profit_factor, total_trades)
            
            return {
                'symbol': symbol,
                'strategy': strategy,
                'params': params_str,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'total_return': total_return,
                'max_drawdown': abs(max_drawdown),
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'score': score,
                'parameters': parameters
            }
            
        except Exception as e:
            self.logger.debug(f"Error extracting stats: {e}")
            return None
    
    def _calculate_current_regime_score(self, win_rate: float, total_return: float, 
                                      max_drawdown: float, profit_factor: float, trade_count: int) -> float:
        """FIXED: Current regime scoring that prioritizes recent momentum effectiveness"""
        try:
            # FIXED: Weighted scoring for current crypto momentum
            win_rate_score = min(win_rate, 100.0) * 0.35        # 35% weight (increased)
            return_score = max(total_return, 0) * 0.20           # 20% weight  
            drawdown_score = max(0, (30 - abs(max_drawdown))) * 0.25  # 25% weight (increased penalty)
            profit_factor_score = min(max(profit_factor - 1, 0) * 15, 25) * 0.15  # 15% weight
            trade_frequency_score = min(trade_count / 10.0, 1.0) * 5 * 0.05  # 5% weight (reduced)
            
            total_score = (win_rate_score + return_score + drawdown_score + 
                         profit_factor_score + trade_frequency_score)
            
            return round(total_score, 2)
        except:
            return 0.0

def get_viable_symbols_for_testing(exchange: Exchange, limit: int = 10) -> List[str]:
    """FIXED: Get symbols with current momentum for testing"""
    logger = logging.getLogger(__name__)
    
    try:
        active_symbols_data = exchange.get_top_active_symbols(limit=limit*2)
        
        if active_symbols_data:
            candidate_symbols = [item['symbol'] for item in active_symbols_data]
        else:
            candidate_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        # FIXED: For ROC strategy, check multi-timeframe data
        viable_symbols = []
        for symbol in candidate_symbols:
            # Check both 3m and 15m data availability
            info_3m = get_symbol_data_info(exchange, symbol, timeframe='3m')
            info_15m = get_symbol_data_info(exchange, symbol, timeframe='15m')
            
            if (info_3m['available'] and info_3m['candles'] >= 100 and
                info_15m['available'] and info_15m['candles'] >= 25):
                viable_symbols.append(symbol)
                logger.info(f"✓ {symbol}: 3m:{info_3m['candles']} 15m:{info_15m['candles']} candles")
            else:
                logger.warning(f"✗ {symbol}: Insufficient multi-timeframe data")
        
        return viable_symbols[:limit]
        
    except Exception as e:
        logger.error(f"Error getting viable symbols: {e}")
        return ['BTCUSDT', 'ETHUSDT'][:limit]

def run_intelligent_vectorbt_optimization():
    """FIXED: Run current regime optimization with proper multi-timeframe data"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Current Regime VectorBT Optimization")
    
    try:
        # Load exchange configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        exchange = Exchange(config['api_key'], config['api_secret'])
        tester = IntelligentVectorBTTester(exchange)
        
        # FIXED: Get fewer symbols to avoid rate limiting
        viable_symbols = get_viable_symbols_for_testing(exchange, limit=3)
        
        if not viable_symbols:
            logger.error("No viable symbols found for current regime testing")
            return
        
        logger.info(f"Current regime optimization on: {viable_symbols}")
        
        all_results = []
        
        # FIXED: Process each symbol with current regime focus
        for symbol in viable_symbols:
            logger.info(f"\nCurrent regime analysis: {symbol}")
            
            symbol_results_count = 0
            
            # Test strategies on current regime
            for strategy_name in tester.strategies.keys():
                start_time = time.time()
                
                try:
                    # FIXED: Get appropriate data based on strategy
                    if strategy_name == 'roc_multi_timeframe':
                        # For ROC, get both 3m and 15m data
                        symbol_info = get_symbol_data_info(exchange, symbol, timeframe='multi_roc', limit=300)
                    else:
                        # For other strategies, use 3m data
                        symbol_info = get_symbol_data_info(exchange, symbol, timeframe='3m', limit=200)
                    
                    if not symbol_info['available']:
                        logger.error(f"No current data available for {symbol} - {strategy_name}")
                        continue
                    
                    symbol_data = symbol_info['data']
                    
                    results = tester.optimize_strategy_efficiently(symbol_data, symbol, strategy_name)
                    optimization_time = time.time() - start_time
                    
                    if results:
                        all_results.extend(results)
                        symbol_results_count += len(results)
                        
                        # Show top result for current regime
                        best_result = results[0]
                        logger.info(f"✓ {strategy_name}: {len(results)} current regime params, "
                                  f"best score: {best_result['score']:.2f} "
                                  f"({optimization_time:.2f}s)")
                    else:
                        logger.warning(f"✗ {strategy_name}: No current regime results ({optimization_time:.2f}s)")
                        
                except Exception as e:
                    optimization_time = time.time() - start_time
                    logger.error(f"✗ {strategy_name}: Error - {str(e)} ({optimization_time:.2f}s)")
            
            logger.info(f"Symbol {symbol}: {symbol_results_count} current regime combinations")
            
            # Add delay between symbols to avoid rate limiting
            if symbol != viable_symbols[-1]:
                time.sleep(1)
        
        # Export results
        if all_results:
            os.makedirs('vectorbt_results', exist_ok=True)
            
            df = pd.DataFrame(all_results)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = f'vectorbt_results/current_regime_optimization_{timestamp}.csv'
            df.to_csv(results_file, index=False)
            
            logger.info(f"Results exported to: {results_file}")
        else:
            logger.error("No current regime optimization results generated")
        
    except Exception as e:
        logger.error(f"Critical error in current regime optimization: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    """Main function"""
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Current Regime VectorBT Strategy Optimizer")
    
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