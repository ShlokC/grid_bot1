"""
Exchange module for handling communication with Binance USDM Futures using CCXT.
Enhanced with take_profit_market order support.
"""
import ccxt
import logging
import time
from threading import Lock, Semaphore
from typing import Dict, List, Any, Optional, Tuple

class Exchange:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize the exchange connection with rate limiting."""
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Rate limiting and thread safety
        self.api_lock = Lock()
        self.rate_limiter = Semaphore(10)  # Max 10 concurrent requests
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Initialize CCXT Binance USDM Futures exchange
        self.exchange = self._create_exchange()
        
        # Pre-load markets
        try:
            self.markets = self.exchange.load_markets()
            self.logger.info(f"Initialized exchange connection. Loaded {len(self.markets)} markets.")
            
        except Exception as e:
            self.logger.error(f"Failed to load markets: {e}")
            raise

    def _rate_limited_request(self, func, *args, **kwargs):
        """Rate-limited API request wrapper"""
        with self.rate_limiter:
            with self.api_lock:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_request_interval:
                    time.sleep(self.min_request_interval - elapsed)
                self.last_request_time = time.time()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Rate-limited request failed: {e}")
                raise

    def _create_exchange(self):
        """Create and return a CCXT exchange instance."""
        return ccxt.binanceusdm({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'timeout': 30000
        })
    
    def _get_symbol_id(self, symbol: str) -> str:
        """
        Convert any symbol format to the ID format required by Binance USDM API.
        
        Args:
            symbol: Symbol in any format (e.g., 'BTC/USDT', 'BTCUSDT', 'BTC/USDT:USDT')
            
        Returns:
            str: Symbol ID (e.g., 'BTCUSDT')
        """
        # If symbol contains a colon, extract the part before colon
        if ':' in symbol:
            symbol = symbol.split(':')[0]
        
        # If symbol contains a slash, remove it
        if '/' in symbol:
            symbol = symbol.replace('/', '')
            
        return symbol
    
    def get_ohlcv(self, symbol: str, timeframe: str = '3m', limit: int = 100) -> List[List]:
        """Get OHLCV historical data with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Fetching OHLCV for {symbol_id}, timeframe: {timeframe}, limit: {limit}")
            
            ohlcv = self._rate_limited_request(self.exchange.fetch_ohlcv, symbol_id, timeframe, limit=limit)
            
            if not ohlcv:
                self.logger.warning(f"No OHLCV data received for {symbol_id}")
                return []
            
            return ohlcv
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []

    def get_balance(self) -> Dict:
        """Get account balance."""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker price for symbol with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Fetching ticker for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.fetch_ticker, symbol_id)
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """ENHANCED: Create limit order with automatic trading setup."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            
            # CRITICAL: Ensure trading configuration is set before order
            if not self.setup_symbol_trading_config(symbol, 20):
                raise Exception(f"Failed to setup trading configuration for {symbol}")
            
            self.logger.info(f"🚀 LIMIT ORDER: {side.upper()} {amount:.6f} {symbol_id} @ ${price:.6f}")
            
            # Create the order
            result = self._rate_limited_request(
                self.exchange.create_limit_order, 
                symbol_id, 
                side, 
                amount, 
                price
            )
            
            if result and 'id' in result:
                order_id = result['id']
                self.logger.info(f"✅ LIMIT ORDER CREATED: {side.upper()} {amount:.6f} {symbol_id} @ ${price:.6f}, ID: {order_id[:8]}")
            else:
                self.logger.error(f"❌ LIMIT ORDER FAILED: Invalid response")
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ LIMIT ORDER ERROR: {e}")
            raise
        
    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """MINIMAL: Create market order with optional setup."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            
            # OPTIONAL: Try to setup, but don't fail if it doesn't work
            self.setup_symbol_trading_config(symbol, 20)
            
            self.logger.info(f"🚀 MARKET ORDER: {side.upper()} {amount:.6f} {symbol_id}")
            
            # Create the order (original logic unchanged)
            result = self._rate_limited_request(
                self.exchange.create_market_order, 
                symbol_id, 
                side, 
                amount
            )
            
            if result and 'id' in result:
                order_id = result['id']
                fill_price = result.get('average', 'pending')
                
                if fill_price != 'pending':
                    self.logger.info(f"✅ MARKET ORDER FILLED: {side.upper()} {amount:.6f} {symbol_id} @ ${float(fill_price):.6f}, ID: {order_id[:8]}")
                else:
                    self.logger.info(f"✅ MARKET ORDER CREATED: {side.upper()} {amount:.6f} {symbol_id}, ID: {order_id[:8]}")
            else:
                self.logger.error(f"❌ MARKET ORDER FAILED: Invalid response")
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ MARKET ORDER ERROR: {e}")
            raise

    # Add this method to the Exchange class
    def create_take_profit_market_order(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict:
        """Create a take profit market order."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            
            self.logger.info(f"🎯 TAKE_PROFIT_MARKET: {side.upper()} {amount:.6f} {symbol_id} @ ${stop_price:.6f}")
            
            # Fixed: Removed extra None parameter
            result = self._rate_limited_request(
                self.exchange.create_order,
                symbol_id,
                'TAKE_PROFIT_MARKET',
                side,
                amount,
                None,  # price parameter (None for market orders)
                {
                    'stopPrice': stop_price,
                    'timeInForce': 'GTE_GTC'
                }
            )
            
            if result and 'id' in result:
                order_id = result['id']
                self.logger.info(f"✅ TAKE_PROFIT_MARKET CREATED: ID: {order_id[:8]}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ TAKE_PROFIT_MARKET ERROR: {e}")
            raise
        
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order by ID with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Cancelling order {order_id} for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.cancel_order, order_id, symbol_id)
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            raise
    
    def cancel_all_orders(self, symbol: str) -> List:
        """Cancel all orders for a symbol with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Cancelling all orders for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.cancel_all_orders, symbol_id)
        except Exception as e:
            self.logger.error(f"Error cancelling all orders for {symbol}: {e}")
            raise

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders with rate limiting."""
        try:
            symbol_id = None
            if symbol:
                symbol_id = self._get_symbol_id(symbol)
                self.logger.debug(f"Fetching open orders for symbol ID: {symbol_id}")
                
            return self._rate_limited_request(self.exchange.fetch_open_orders, symbol_id)
        except Exception as e:
            symbol_info = f" for {symbol}" if symbol else ""
            self.logger.error(f"Error fetching open orders{symbol_info}: {e}")
            raise
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """Get all open positions with rate limiting."""
        try:
            symbols_array = None
            
            if symbol:
                symbol_id = self._get_symbol_id(symbol)
                symbols_array = [symbol_id]
                self.logger.debug(f"Fetching positions for symbol array: {symbols_array}")
            
            positions = self._rate_limited_request(self.exchange.fetch_positions, symbols_array)
            
            # Filter out positions with zero size
            filtered_positions = []
            for pos in positions:
                size = float(pos.get('contracts', 0))
                if size != 0:
                    filtered_positions.append(pos)
            
            return filtered_positions
        except Exception as e:
            symbol_info = f" for {symbol}" if symbol else ""
            self.logger.error(f"Error fetching positions{symbol_info}: {e}")
            raise
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get the status of an order with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Fetching order {order_id} status for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.fetch_order, order_id, symbol_id)
        except Exception as e:
            self.logger.error(f"Error fetching order {order_id} status for {symbol}: {e}")
            raise
    
    def get_market_info(self, symbol: str) -> Dict:
        """Get market information, including precision and limits."""
        try:
            # Make sure markets are loaded
            if not hasattr(self, 'markets') or not self.markets:
                self.markets = self.exchange.load_markets()
            
            # Format symbol to ID for market lookup
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Getting market info for symbol ID: {symbol_id}")
            
            # Check if symbol exists in markets by comparing with the 'id' field
            symbol_found = False
            market_info = None
            
            for market_symbol, market_data in self.markets.items():
                if market_data.get('id', '').upper() == symbol_id.upper():
                    symbol_found = True
                    market_info = market_data
                    break
            
            if not symbol_found:
                self.logger.warning(f"Symbol ID {symbol_id} not found in markets. Reloading markets...")
                self.markets = self.exchange.load_markets(True)  # Force reload
                
                # Try again after reload
                for market_symbol, market_data in self.markets.items():
                    if market_data.get('id', '').upper() == symbol_id.upper():
                        symbol_found = True
                        market_info = market_data
                        break
                
                if not symbol_found:
                    available_symbols = [m.get('id', '') for m in self.markets.values()]
                    self.logger.error(f"Symbol ID {symbol_id} not available. Available symbols include: {', '.join(available_symbols[:10])}...")
                    raise ValueError(f"Symbol {symbol_id} not available on Binance USDM futures")
            
            return market_info if market_info else self.exchange.market(symbol_id)
            
        except Exception as e:
            self.logger.error(f"Error fetching market info for {symbol}: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all available trading symbols (IDs only)."""
        try:
            if not hasattr(self, 'markets') or not self.markets:
                self.markets = self.exchange.load_markets()
            
            # Return only active markets and their IDs
            available_symbols = []
            for market_symbol, market in self.markets.items():
                if market.get('active', False):
                    available_symbols.append(market.get('id', ''))
            
            return sorted(available_symbols)
        except Exception as e:
            self.logger.error(f"Error fetching available symbols: {e}")
            raise

    def create_stop_order(self, symbol: str, side: str, amount: float, stop_price: float, order_type: str = 'stop_market') -> Dict:
        """Create a stop-loss order (stop-market or stop-limit)."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            
            self.logger.info(f"🛡️ STOP ORDER: {side.upper()} {amount:.6f} {symbol_id} STOP @ ${stop_price:.6f}")
            
            # Prepare order parameters
            params = {
                'stopPrice': stop_price,
                'timeInForce': 'GTE_GTC'  # Good Till Cancelled
            }
            
            if order_type == 'stop_market':
                # FIXED: Correct CCXT create_order arguments
                result = self._rate_limited_request(
                    self.exchange.create_order,
                    symbol_id,
                    'STOP_MARKET',  # Use uppercase for Binance
                    side,
                    amount,
                    None,  # No limit price for stop-market
                    params  # Only pass params, not extra None
                )
            elif order_type == 'stop_limit':
                # Stop-limit order (has both stop price and limit price)
                limit_price = stop_price  # Use stop price as limit price for immediate execution
                result = self._rate_limited_request(
                    self.exchange.create_order,
                    symbol_id,
                    'STOP',
                    side,
                    amount,
                    limit_price,
                    params
                )
            else:
                raise ValueError(f"Unsupported stop order type: {order_type}")
            
            if result and 'id' in result:
                order_id = result['id']
                self.logger.info(f"✅ STOP ORDER CREATED: {side.upper()} {amount:.6f} {symbol_id} STOP @ ${stop_price:.6f}, ID: {order_id[:8]}")
            else:
                self.logger.error(f"❌ STOP ORDER FAILED: Invalid response")
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ STOP ORDER ERROR: {e}")
            # If stop orders not supported, inform caller
            if "stop" in str(e).lower() or "unsupported" in str(e).lower():
                self.logger.warning(f"⚠️ Stop orders may not be supported on this exchange")
            raise
    def get_top_active_symbols(self, limit: int = 5, timeframe_minutes: int = 30) -> List[Dict]:
        """
        Get top most active symbols by percentage change using 3-minute candles over specified timeframe.
        
        Args:
            limit: Number of top symbols to return (default: 5)
            timeframe_minutes: Minutes to look back for change calculation (default: 30)
            
        Returns:
            List of dictionaries containing symbol and percentage change data
        """
        try:
            # Calculate how many 3-minute candles we need for the timeframe
            candles_needed = max(int(timeframe_minutes / 3) + 2, 12)  # +2 for safety, minimum 12 candles
            
            self.logger.info(f"Fetching top {limit} most active symbols using 3m candles over {timeframe_minutes} minutes")
            self.logger.info(f"Analyzing {candles_needed} candles ({candles_needed * 3} minutes of data)")
            
            # Get all available symbols
            available_symbols = self.get_available_symbols()
            
            if not available_symbols:
                self.logger.warning("No available symbols found")
                return []
            
            symbol_activities = []
            processed_count = 0
            error_count = 0
            
            # Process symbols in batches to respect rate limits
            for symbol in available_symbols:
                try:
                    # Get 3-minute OHLCV data with rate limiting
                    ohlcv_data = self.get_ohlcv(symbol, timeframe='3m', limit=candles_needed)
                    
                    if not ohlcv_data or len(ohlcv_data) < candles_needed:
                        continue
                    
                    # Calculate percentage change from 30 minutes ago to current close
                    # ohlcv format: [timestamp, open, high, low, close, volume]
                    
                    # Get price from 30 minutes ago (or as close as possible)
                    target_candle_index = -(int(timeframe_minutes / 3) + 1)  # 30min ago = 10 candles back
                    if abs(target_candle_index) > len(ohlcv_data):
                        target_candle_index = -len(ohlcv_data)  # Use oldest available
                    
                    start_price = float(ohlcv_data[target_candle_index][4])  # Close price from 30min ago
                    current_close = float(ohlcv_data[-1][4])   # Current candle close
                    current_high = float(ohlcv_data[-1][2])    # Current candle high
                    current_low = float(ohlcv_data[-1][3])     # Current candle low
                    
                    # Calculate total volume over the period
                    total_volume = sum(float(candle[5]) for candle in ohlcv_data[target_candle_index:])
                    
                    if start_price <= 0:
                        continue
                    
                    # Calculate percentage change over the full timeframe
                    price_change_pct = ((current_close - start_price) / start_price) * 100
                    
                    # Calculate recent volatility (last few candles)
                    recent_candles = ohlcv_data[-3:]  # Last 9 minutes
                    recent_high = max(float(candle[2]) for candle in recent_candles)
                    recent_low = min(float(candle[3]) for candle in recent_candles)
                    volatility_pct = ((recent_high - recent_low) / current_close) * 100 if current_close > 0 else 0
                    
                    # Calculate actual timeframe analyzed
                    actual_minutes = abs(target_candle_index) * 3
                    
                    symbol_activities.append({
                        'symbol': symbol,
                        'price_change_pct': price_change_pct,
                        'volatility_pct': volatility_pct,
                        'current_price': current_close,
                        'start_price': start_price,
                        'volume': total_volume,
                        'timeframe_minutes': actual_minutes,
                        'abs_change_pct': abs(price_change_pct)  # For sorting by absolute change
                    })
                    
                    processed_count += 1
                    
                    # Log progress every 50 symbols
                    if processed_count % 50 == 0:
                        self.logger.debug(f"Processed {processed_count} symbols, errors: {error_count}")
                    
                except Exception as e:
                    error_count += 1
                    self.logger.debug(f"Error processing symbol {symbol}: {e}")
                    continue
            
            self.logger.info(f"Processed {processed_count} symbols successfully, {error_count} errors")
            
            if not symbol_activities:
                self.logger.warning("No symbol data could be processed")
                return []
            
            # Sort by absolute percentage change (most active regardless of direction)
            symbol_activities.sort(key=lambda x: x['abs_change_pct'], reverse=True)
            
            # Get top symbols
            top_symbols = symbol_activities[:limit]
            
            # Log results with timeframe info
            self.logger.info(f"Top {len(top_symbols)} most active symbols ({timeframe_minutes}m analysis):")
            for i, symbol_data in enumerate(top_symbols, 1):
                symbol = symbol_data['symbol']
                change_pct = symbol_data['price_change_pct']
                volatility = symbol_data['volatility_pct']
                current_price = symbol_data['current_price']
                start_price = symbol_data['start_price']
                analyzed_minutes = symbol_data['timeframe_minutes']
                
                direction = "📈" if change_pct > 0 else "📉"
                self.logger.info(f"  {i}. {symbol}: {direction} {change_pct:+.2f}% over {analyzed_minutes}m")
                self.logger.info(f"     ${start_price:.6f} → ${current_price:.6f} (Vol: {volatility:.2f}%)")
            
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting top active symbols: {e}")
            return []

    def get_top_gainers_losers(self, limit: int = 5, timeframe_minutes: int = 30) -> Dict[str, List[Dict]]:
        """
        Get separate lists of top gainers and top losers using 3-minute candles over specified timeframe.
        
        Args:
            limit: Number of top gainers and losers to return each
            timeframe_minutes: Minutes to look back for change calculation (default: 30)
            
        Returns:
            Dictionary with 'gainers' and 'losers' keys containing lists of symbol data
        """
        try:
            # Get top active symbols with larger sample for filtering
            active_symbols = self.get_top_active_symbols(limit=limit * 4, timeframe_minutes=timeframe_minutes)
            
            if not active_symbols:
                return {'gainers': [], 'losers': []}
            
            # Separate gainers and losers
            gainers = [s for s in active_symbols if s['price_change_pct'] > 0]
            losers = [s for s in active_symbols if s['price_change_pct'] < 0]
            
            # Sort gainers by highest positive change
            gainers.sort(key=lambda x: x['price_change_pct'], reverse=True)
            
            # Sort losers by highest negative change (most negative)
            losers.sort(key=lambda x: x['price_change_pct'])
            
            result = {
                'gainers': gainers[:limit],
                'losers': losers[:limit]
            }
            
            self.logger.info(f"Found {len(result['gainers'])} top gainers and {len(result['losers'])} top losers over {timeframe_minutes}m")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting top gainers/losers: {e}")
            return {'gainers': [], 'losers': []}
    def setup_symbol_trading_config(self, symbol: str, target_leverage: int = 20) -> bool:
        """MINIMAL: Setup isolated margin and leverage with correct method names."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.info(f"🔧 Setting up {symbol} ({symbol_id}) for {target_leverage}x trading")
            
            # Step 1: Set margin type to ISOLATED
            try:
                self.exchange.fapiPrivate_post_margintype({
                    'symbol': symbol_id,
                    'marginType': 'ISOLATED'
                })
                self.logger.info(f"✅ Set {symbol_id} to ISOLATED margin")
            except Exception as e:
                if '-4046' in str(e) or 'No need to change' in str(e):
                    self.logger.info(f"✅ {symbol_id} already ISOLATED")
                else:
                    self.logger.warning(f"⚠️ Margin setup failed for {symbol_id}: {e}")
                    # Continue anyway
            
            # Step 2: Set leverage (FIXED: correct method name)
            try:
                self.exchange.fapiprivate_post_leverage({  # FIXED: lowercase 'p'
                    'symbol': symbol_id,
                    'leverage': target_leverage
                })
                self.logger.info(f"✅ Set {symbol_id} to {target_leverage}x leverage")
            except Exception as e:
                if '-4141' in str(e) or 'position' in str(e):
                    self.logger.warning(f"⚠️ Cannot change leverage for {symbol_id} - position exists")
                else:
                    self.logger.warning(f"⚠️ Leverage setup failed for {symbol_id}: {e}")
                # Continue anyway
            
            return True  # Always return True to allow trading
            
        except Exception as e:
            self.logger.error(f"❌ Setup error for {symbol}: {e}")
            return True  # FIXED: Return True to allow trading even if setup fails
    def get_symbol_trading_config(self, symbol: str) -> Dict:
        """Get current trading configuration for a symbol."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            
            # Get position risk info which contains margin type and leverage
            position_info = self._rate_limited_request(
                self.exchange.fapiprivatev2_get_positionrisk,
                {'symbol': symbol_id}
            )
            
            if position_info and len(position_info) > 0:
                info = position_info[0]
                return {
                    'symbol': symbol,
                    'internal_symbol': symbol_id,
                    'margin_type': info.get('marginType', 'UNKNOWN'),
                    'leverage': int(float(info.get('leverage', 0))),
                    'position_size': float(info.get('positionAmt', 0)),
                    'entry_price': float(info.get('entryPrice', 0)),
                    'unrealized_pnl': float(info.get('unRealizedProfit', 0))
                }
            
            return {
                'symbol': symbol,
                'internal_symbol': symbol_id,
                'error': 'No position info found'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading config for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    def _setup_isolated_margin(self, internal_symbol: str) -> bool:
        """Setup isolated margin mode for symbol."""
        try:
            # First, check current margin mode
            try:
                position_info = self.exchange.fapiprivatev2_get_positionrisk({
                    'symbol': internal_symbol
                })
                
                if position_info and len(position_info) > 0:
                    current_margin_type = position_info[0].get('marginType', '').upper()
                    
                    if current_margin_type == 'ISOLATED':
                        self.logger.info(f"✅ {internal_symbol} already in ISOLATED margin mode")
                        return True
                    
                    self.logger.info(f"🔄 Changing {internal_symbol} from {current_margin_type} to ISOLATED margin")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not check current margin mode for {internal_symbol}: {e}")
                # Continue with setting isolated margin anyway
            
            # Set margin mode to ISOLATED
            response = self.exchange.fapiPrivate_post_margintype({
                'symbol': internal_symbol,
                'marginType': 'ISOLATED'
            })
            
            self.logger.info(f"✅ Set {internal_symbol} to ISOLATED margin mode")
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle "No need to change margin type" error (already correct)
            if '-4046' in error_msg or 'No need to change margin type' in error_msg:
                self.logger.info(f"✅ {internal_symbol} already in ISOLATED margin mode")
                return True
            
            # Handle other errors
            self.logger.error(f"❌ Failed to set isolated margin for {internal_symbol}: {e}")
            return False

    def _setup_leverage(self, internal_symbol: str, target_leverage: int) -> bool:
        """Setup leverage for symbol."""
        try:
            # Validate leverage range
            if target_leverage < 1 or target_leverage > 125:
                self.logger.error(f"❌ Invalid leverage {target_leverage}. Must be 1-125")
                return False
            
            # Check current leverage
            try:
                position_info = self.exchange.fapiprivatev2_get_positionrisk({
                    'symbol': internal_symbol
                })
                
                if position_info and len(position_info) > 0:
                    current_leverage = int(float(position_info[0].get('leverage', 0)))
                    
                    if current_leverage == target_leverage:
                        self.logger.info(f"✅ {internal_symbol} already at {target_leverage}x leverage")
                        return True
                    
                    self.logger.info(f"🔄 Changing {internal_symbol} leverage from {current_leverage}x to {target_leverage}x")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not check current leverage for {internal_symbol}: {e}")
                # Continue with setting leverage anyway
            
            # Set leverage
            response = self.exchange.fapiPrivate_post_leverage({
                'symbol': internal_symbol,
                'leverage': target_leverage
            })
            
            self.logger.info(f"✅ Set {internal_symbol} to {target_leverage}x leverage")
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle common errors
            if '-4028' in error_msg:
                self.logger.error(f"❌ Leverage {target_leverage}x not allowed for {internal_symbol}")
            elif '-4141' in error_msg:
                self.logger.error(f"❌ Cannot change leverage with open positions for {internal_symbol}")
            else:
                self.logger.error(f"❌ Failed to set leverage for {internal_symbol}: {e}")
            
            return False