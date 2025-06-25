"""
Integrated Crypto Backtester - Uses existing exchange.py and config.json
File: core/backtester.py
"""

import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
import json
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from core.exchange import Exchange

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 1.0
    position_size_pct: float = 100.0
    leverage: float = 20.0
    commission_pct: float = 0.075
    slippage_pct: float = 0.02
    take_profit_pct: float = 1.5
    stop_loss_pct: float = 1.0
    max_open_time_minutes: int = 60
    timeframe: str = '1m'
    limit: int = 1400  # Number of candles to fetch

@dataclass
class TradeResult:
    """Individual trade result"""
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_abs: float
    pnl_pct: float
    pnl_leveraged: float
    exit_reason: str
    duration_minutes: int
    entry_signals: Dict
    exit_signals: Dict

class IntegratedBacktester:
    """Integrated backtester using your existing exchange connection"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.exchange = None
        self.trades: List[TradeResult] = []
        self.equity_curve: List[Dict] = []
        
        # Load exchange configuration
        self._load_exchange()
    
    def _load_exchange(self):
        """Load exchange using your existing config.json"""
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Config file {self.config_file} not found")
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            if 'api_key' not in config or 'api_secret' not in config:
                raise ValueError("API credentials not found in config.json")
            
            # Initialize exchange using your existing Exchange class
            self.exchange = Exchange(
                api_key=config['api_key'],
                api_secret=config['api_secret']
            )
            
            self.logger.info(" Exchange connection established for backtesting")
            
        except Exception as e:
            self.logger.error(f" Failed to initialize exchange: {e}")
            raise
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1m', limit: int = 1400) -> pd.DataFrame:
        """Fetch OHLCV data using your existing exchange connection"""
        try:
            self.logger.info(f"Fetching {limit} {timeframe} candles for {symbol}...")
            
            # Use your existing exchange method
            ohlcv_data = self.exchange.get_ohlcv(symbol, timeframe, limit)
            
            if not ohlcv_data or len(ohlcv_data) < 50:
                raise ValueError(f"Insufficient data: {len(ohlcv_data) if ohlcv_data else 0} candles")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Ensure all price columns are float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df.sort_index().dropna()
            
            self.logger.info(f" Fetched {len(df)} candles for {symbol}")
            self.logger.info(f"Price range: ${df['close'].min():.8f} - ${df['close'].max():.8f}")
            self.logger.info(f"Time range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f" Error fetching OHLCV data for {symbol}: {e}")
            raise
    
    def add_indicators(self, df: pd.DataFrame, indicator_config: Dict[str, Dict]) -> pd.DataFrame:
        """Add indicators to dataframe"""
        df = df.copy()
        
        for indicator_name, params in indicator_config.items():
            try:
                if indicator_name == 'qqe':
                    result = ta.qqe(df['close'], **params)
                    if result is not None and not result.empty:
                        df[f'qqe_line'] = result.iloc[:, 0]
                        df[f'qqe_signal'] = result.iloc[:, 1]
                        self.logger.debug(f" Added QQE indicator")
                
                elif indicator_name == 'supertrend':
                    result = ta.supertrend(df['high'], df['low'], df['close'], **params)
                    if result is not None and not result.empty:
                        # Find direction column
                        dir_cols = [col for col in result.columns if 'SUPERTd' in col]
                        if dir_cols:
                            df['st_direction'] = result[dir_cols[0]]
                        
                        # Find trend line
                        trend_cols = [col for col in result.columns if 'SUPERT_' in col and 'SUPERTd' not in col]
                        if trend_cols:
                            df['st_line'] = result[trend_cols[0]]
                        
                        self.logger.debug(f" Added Supertrend indicator")
                
                elif indicator_name == 'rsi':
                    df['rsi'] = ta.rsi(df['close'], **params)
                    self.logger.debug(f" Added RSI indicator")
                
                elif indicator_name == 'macd':
                    result = ta.macd(df['close'], **params)
                    if result is not None and not result.empty:
                        df['macd_line'] = result.iloc[:, 0]
                        df['macd_histogram'] = result.iloc[:, 1]
                        df['macd_signal'] = result.iloc[:, 2]
                        self.logger.debug(f" Added MACD indicator")
                
                elif indicator_name == 'tsi':
                    result = ta.tsi(df['close'], **params)
                    if result is not None and not result.empty:
                        df['tsi_line'] = result.iloc[:, 0]
                        if result.shape[1] > 1:
                            df['tsi_signal'] = result.iloc[:, 1]
                        self.logger.debug(f" Added TSI indicator")
                
                elif indicator_name == 'vwap':
                    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                    self.logger.debug(f" Added VWAP indicator")
                
                elif indicator_name == 'adx':
                    result = ta.adx(df['high'], df['low'], df['close'], **params)
                    if result is not None and not result.empty:
                        df['adx'] = result.iloc[:, 0]
                        self.logger.debug(f" Added ADX indicator")
                
                elif indicator_name == 'stochrsi':
                    result = ta.stochrsi(df['close'], **params)
                    if result is not None and not result.empty:
                        df['stochrsi_k'] = result.iloc[:, 0]
                        df['stochrsi_d'] = result.iloc[:, 1]
                        self.logger.debug(f" Added StochRSI indicator")
                
                elif indicator_name == 'cci':
                    df['cci'] = ta.cci(df['high'], df['low'], df['close'], **params)
                    self.logger.debug(f" Added CCI indicator")
                
                elif indicator_name == 'bbands':
                    result = ta.bbands(df['close'], **params)
                    if result is not None and not result.empty:
                        df['bb_lower'] = result.iloc[:, 0]
                        df['bb_mid'] = result.iloc[:, 1]
                        df['bb_upper'] = result.iloc[:, 2]
                        self.logger.debug(f" Added Bollinger Bands")
                
            except Exception as e:
                self.logger.error(f" Error adding {indicator_name}: {e}")
        
        return df.dropna()
    
    def run_backtest(self, 
                     symbol: str,
                     signal_func: Callable[[pd.DataFrame, int], str],
                     indicator_config: Dict[str, Dict],
                     backtest_config: BacktestConfig = None) -> Dict:
        """Run backtest using live exchange data"""
        
        if backtest_config is None:
            backtest_config = BacktestConfig()
        
        try:
            # Reset state
            self.trades = []
            self.equity_curve = []
            current_capital = backtest_config.initial_capital
            
            # Fetch live data
            df = self.fetch_ohlcv_data(symbol, backtest_config.timeframe, backtest_config.limit)
            
            # Add indicators
            df = self.add_indicators(df, indicator_config)
            
            if len(df) < 50:
                return {"error": "Insufficient data after adding indicators"}
            
            self.logger.info(f"🚀 Starting backtest for {symbol} with {len(df)} candles")
            
            # Backtest variables
            position = None
            entry_idx = 0
            entry_price = 0.0
            entry_time = None
            
            # Main backtest loop
            for i in range(50, len(df)):  # Skip first 50 for indicator warmup
                current_time = df.index[i]
                current_price = df['close'].iloc[i]
                
                # Update equity curve
                self.equity_curve.append({
                    'timestamp': current_time,
                    'price': current_price,
                    'capital': current_capital,
                    'position': position
                })
                
                # Exit logic
                if position:
                    minutes_open = i - entry_idx
                    exit_reason = None
                    exit_price = current_price
                    
                    # Check TP/SL
                    if position == 'long':
                        if current_price >= entry_price * (1 + backtest_config.take_profit_pct / 100):
                            exit_reason = 'tp'
                        elif current_price <= entry_price * (1 - backtest_config.stop_loss_pct / 100):
                            exit_reason = 'sl'
                    else:  # short
                        if current_price <= entry_price * (1 - backtest_config.take_profit_pct / 100):
                            exit_reason = 'tp'
                        elif current_price >= entry_price * (1 + backtest_config.stop_loss_pct / 100):
                            exit_reason = 'sl'
                    
                    # Check timeout
                    if minutes_open >= backtest_config.max_open_time_minutes:
                        exit_reason = 'timeout'
                    
                    # Check signal exit
                    if not exit_reason:
                        signal = signal_func(df, i)
                        if (position == 'long' and signal == 'sell') or (position == 'short' and signal == 'buy'):
                            exit_reason = 'signal'
                    
                    # Execute exit
                    if exit_reason:
                        trade = self._close_position(
                            position, entry_time, current_time, entry_price, exit_price,
                            exit_reason, df, entry_idx, i, backtest_config
                        )
                        self.trades.append(trade)
                        
                        # Update capital
                        current_capital += trade.pnl_abs
                        position = None
                
                # Entry logic
                if not position:
                    signal = signal_func(df, i)
                    if signal in ['buy', 'sell']:
                        position = 'long' if signal == 'buy' else 'short'
                        entry_idx = i
                        entry_price = current_price
                        entry_time = current_time
            
            # Close remaining position
            if position:
                trade = self._close_position(
                    position, entry_time, df.index[-1], entry_price, df['close'].iloc[-1],
                    'end', df, entry_idx, len(df)-1, backtest_config
                )
                self.trades.append(trade)
                current_capital += trade.pnl_abs
            
            # Calculate metrics
            metrics = self._calculate_metrics(backtest_config)
            
            self.logger.info(f" Backtest completed: {len(self.trades)} trades, "
                           f"{metrics.get('win_rate', 0):.1f}% win rate, "
                           f"{metrics.get('total_return_pct', 0):.2f}% return")
            
            return {
                'symbol': symbol,
                'metrics': metrics,
                'trades_count': len(self.trades),
                'data_points': len(df),
                'config': backtest_config
            }
            
        except Exception as e:
            self.logger.error(f" Backtest error for {symbol}: {e}")
            return {"error": str(e)}
    
    def _close_position(self, side: str, entry_time, exit_time, entry_price: float,
                       exit_price: float, exit_reason: str, df: pd.DataFrame,
                       entry_idx: int, exit_idx: int, config: BacktestConfig) -> TradeResult:
        """Close position and calculate PnL"""
        
        # Calculate position value and quantity
        position_value = config.initial_capital * (config.position_size_pct / 100)
        quantity = (position_value * config.leverage) / entry_price
        
        # Calculate PnL
        if side == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Apply leverage
        pnl_leveraged_pct = pnl_pct * config.leverage
        
        # Calculate absolute PnL
        pnl_abs = position_value * pnl_leveraged_pct
        
        # Apply costs
        commission = position_value * (config.commission_pct / 100) * 2  # Entry + Exit
        slippage = position_value * (config.slippage_pct / 100)
        net_pnl = pnl_abs - commission - slippage
        
        # Extract signal values
        entry_signals = self._extract_signals(df, entry_idx)
        exit_signals = self._extract_signals(df, exit_idx)
        
        return TradeResult(
            entry_time=str(entry_time),
            exit_time=str(exit_time),
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl_abs=net_pnl,
            pnl_pct=pnl_pct * 100,
            pnl_leveraged=pnl_leveraged_pct * 100,
            exit_reason=exit_reason,
            duration_minutes=exit_idx - entry_idx,
            entry_signals=entry_signals,
            exit_signals=exit_signals
        )
    
    def _extract_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        """Extract indicator values at given index"""
        signals = {}
        row = df.iloc[idx]
        
        for col in df.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                try:
                    signals[col] = float(row[col]) if not pd.isna(row[col]) else None
                except:
                    signals[col] = None
        
        signals['price'] = float(row['close'])
        return signals
    
    def _calculate_metrics(self, config: BacktestConfig) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl_abs > 0]
        losing_trades = [t for t in self.trades if t.pnl_abs <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        # PnL metrics
        total_pnl = sum(t.pnl_abs for t in self.trades)
        total_return_pct = (total_pnl / config.initial_capital) * 100
        
        # Profit factor
        gross_profit = sum(t.pnl_abs for t in winning_trades)
        gross_loss = abs(sum(t.pnl_abs for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        returns = [t.pnl_abs / config.initial_capital for t in self.trades]
        if len(returns) > 1:
            volatility = np.std(returns)
            sharpe = np.mean(returns) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe = 0
        
        # Drawdown calculation
        equity_curve = [config.initial_capital]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade.pnl_abs)
        
        peak = config.initial_capital
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3),
            'total_return_pct': round(total_return_pct, 2),
            'total_pnl': round(total_pnl, 4),
            'gross_profit': round(gross_profit, 4),
            'gross_loss': round(gross_loss, 4),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 3),
            'avg_duration_minutes': round(np.mean([t.duration_minutes for t in self.trades]), 1),
            'final_capital': round(config.initial_capital + total_pnl, 4)
        }
    
    def export_results(self, symbol: str, strategy_name: str, output_dir: str = "backtest_results"):
        """Export results to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export trades
        if self.trades:
            trades_data = []
            for trade in self.trades:
                trade_dict = {
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'side': trade.side,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl_abs': trade.pnl_abs,
                    'pnl_pct': trade.pnl_pct,
                    'exit_reason': trade.exit_reason,
                    'duration_minutes': trade.duration_minutes
                }
                
                # Add signal values
                for key, value in trade.entry_signals.items():
                    trade_dict[f'entry_{key}'] = value
                for key, value in trade.exit_signals.items():
                    trade_dict[f'exit_{key}'] = value
                
                trades_data.append(trade_dict)
            
            trades_file = f"{output_dir}/{symbol}_{strategy_name}_trades_{timestamp}.csv"
            pd.DataFrame(trades_data).to_csv(trades_file, index=False)
            
            self.logger.info(f"📁 Trades exported: {trades_file}")
            return trades_file
        
        return None

# Signal functions (same as before but integrated)
def qqe_supertrend_signal_fixed(df: pd.DataFrame, idx: int) -> str:
    """FIXED QQE + Supertrend signal logic"""
    try:
        if idx < 1:
            return 'none'
        
        qqe_line = df['qqe_line'].iloc[idx]
        qqe_signal = df['qqe_signal'].iloc[idx]
        st_direction = df['st_direction'].iloc[idx]
        
        if pd.isna(qqe_line) or pd.isna(qqe_signal) or pd.isna(st_direction):
            return 'none'
        
        # FIXED QQE logic: qqe_line > qqe_signal = bullish
        qqe_bullish = qqe_line > qqe_signal
        st_bullish = st_direction == 1
        
        if qqe_bullish and st_bullish:
            return 'buy'
        elif not qqe_bullish and not st_bullish:
            return 'sell'
        else:
            return 'none'
    
    except Exception:
        return 'none'

def rsi_macd_signal(df: pd.DataFrame, idx: int) -> str:
    """RSI + MACD signal"""
    try:
        if idx < 1:
            return 'none'
        
        rsi = df['rsi'].iloc[idx]
        macd_line = df['macd_line'].iloc[idx]
        macd_signal = df['macd_signal'].iloc[idx]
        
        if pd.isna(rsi) or pd.isna(macd_line) or pd.isna(macd_signal):
            return 'none'
        
        if rsi < 35 and macd_line > macd_signal:
            return 'buy'
        elif rsi > 65 and macd_line < macd_signal:
            return 'sell'
        else:
            return 'none'
    
    except Exception:
        return 'none'

def tsi_vwap_signal(df: pd.DataFrame, idx: int) -> str:
    """TSI + VWAP signal"""
    try:
        if idx < 1:
            return 'none'
        
        tsi_line = df['tsi_line'].iloc[idx]
        tsi_signal = df['tsi_signal'].iloc[idx] if 'tsi_signal' in df.columns else 0
        price = df['close'].iloc[idx]
        vwap = df['vwap'].iloc[idx]
        
        if pd.isna(tsi_line) or pd.isna(price) or pd.isna(vwap):
            return 'none'
        
        if tsi_line > tsi_signal and price > vwap:
            return 'buy'
        elif tsi_line < tsi_signal and price < vwap:
            return 'sell'
        else:
            return 'none'
    
    except Exception:
        return 'none'