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
        """Create a limit order."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            
            self.logger.info(f"🚀 LIMIT ORDER: {side.upper()} {amount:.6f} {symbol_id} @ ${price:.6f}")
            
            # Simple limit order without hedge mode
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
        """Create a market order (for closing positions)."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.info(f"🚀 MARKET ORDER: {side.upper()} {amount:.6f} {symbol_id}")
            
            # Simple market order without hedge mode
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
                # Stop-market order (no limit price, executes at market when triggered)
                result = self._rate_limited_request(
                    self.exchange.create_order,
                    symbol_id,
                    'stop_market',
                    side,
                    amount,
                    None,  # No limit price for stop-market
                    None,  # No price parameter
                    params
                )
            elif order_type == 'stop_limit':
                # Stop-limit order (has both stop price and limit price)
                limit_price = stop_price  # Use stop price as limit price for immediate execution
                result = self._rate_limited_request(
                    self.exchange.create_order,
                    symbol_id,
                    'stop_limit',
                    side,
                    amount,
                    limit_price,
                    None,
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