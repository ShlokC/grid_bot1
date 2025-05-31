"""
Exchange module for handling communication with Binance USDM Futures using CCXT.
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
        
        # ADDED: Rate limiting and thread safety
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
    def create_stop_market_order(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict:
        """Create a stop-market order (stop-loss) with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Creating {side} stop-market order for symbol ID: {symbol_id} at stop price: {stop_price}")
            
            # Use CCXT's create_order with stop_market type
            return self._rate_limited_request(
                self.exchange.create_order,
                symbol_id, 
                'stop_market',  # Order type
                side, 
                amount, 
                None,  # No limit price for stop-market
                None,  # No limit price
                {'stopPrice': stop_price}  # Stop trigger price
            )
        except Exception as e:
            self.logger.error(f"Error creating {side} stop-market order for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    def _rate_limited_request(self, func, *args, **kwargs):
        """ADDED: Rate-limited API request wrapper"""
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
    
    def get_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 1400) -> List[List]:
        """MODIFIED: Get OHLCV historical data with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Fetching OHLCV for {symbol_id}, timeframe: {timeframe}, limit: {limit}")
            
            ohlcv = self._rate_limited_request(self.exchange.fetch_ohlcv, symbol_id, timeframe, limit=limit)
            
            if not ohlcv:
                self.logger.warning(f"No OHLCV data received for {symbol_id}")
                return []
            
            self.logger.info(f"Fetched {len(ohlcv)} OHLCV candles for {symbol_id}")
            return ohlcv
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            return []
    def get_historical_prices(self, symbol: str, timeframe: str = '5m', limit: int = 1400) -> List[float]:
        """
        Get historical closing prices for technical analysis.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (5m for 5-minute candles)  
            limit: Number of prices to fetch
            
        Returns:
            List of closing prices (most recent last)
        """
        try:
            ohlcv = self.get_ohlcv(symbol, timeframe, limit)
            if not ohlcv:
                return []
            
            # Extract closing prices (index 4 in OHLCV)
            prices = [float(candle[4]) for candle in ohlcv]
            
            self.logger.info(f"Extracted {len(prices)} historical prices for {symbol}")
            self.logger.debug(f"Price range: ${min(prices):.6f} - ${max(prices):.6f}")
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error extracting historical prices for {symbol}: {e}")
            return []
    def get_balance(self) -> Dict:
        """Get account balance."""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """MODIFIED: Get current ticker price for symbol with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Fetching ticker for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.fetch_ticker, symbol_id)
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """MODIFIED: Create a limit order with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Creating {side} limit order for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.create_limit_order, symbol_id, side, amount, price)
        except Exception as e:
            self.logger.error(f"Error creating {side} limit order for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """MODIFIED: Create a market order with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Creating {side} market order for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.create_market_order, symbol_id, side, amount)
        except Exception as e:
            self.logger.error(f"Error creating {side} market order for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """MODIFIED: Cancel an order by ID with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Cancelling order {order_id} for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.cancel_order, order_id, symbol_id)
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id} for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def cancel_all_orders(self, symbol: str) -> List:
        """MODIFIED: Cancel all orders for a symbol with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Cancelling all orders for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.cancel_all_orders, symbol_id)
        except Exception as e:
            self.logger.error(f"Error cancelling all orders for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """MODIFIED: Get all open orders with rate limiting."""
        try:
            symbol_id = None
            if symbol:
                symbol_id = self._get_symbol_id(symbol)
                self.logger.debug(f"Fetching open orders for symbol ID: {symbol_id}")
                
            return self._rate_limited_request(self.exchange.fetch_open_orders, symbol_id)
        except Exception as e:
            symbol_info = f" for {symbol} (ID: {self._get_symbol_id(symbol)})" if symbol else ""
            self.logger.error(f"Error fetching open orders{symbol_info}: {e}")
            raise
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """MODIFIED: Get all open positions with rate limiting."""
        try:
            symbols_array = None
            
            if symbol:
                symbol_id = self._get_symbol_id(symbol)
                symbols_array = [symbol_id]
                self.logger.debug(f"Fetching positions for symbol array: {symbols_array}")
            
            positions = self._rate_limited_request(self.exchange.fetch_positions, symbols_array)
            
            # Filter out positions with zero size
            return [pos for pos in positions if float(pos.get('contracts', 0)) != 0]
        except Exception as e:
            symbol_info = f" for {symbol} (ID: {self._get_symbol_id(symbol)})" if symbol else ""
            self.logger.error(f"Error fetching positions{symbol_info}: {e}")
            raise
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """MODIFIED: Get the status of an order with rate limiting."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Fetching order {order_id} status for symbol ID: {symbol_id}")
            return self._rate_limited_request(self.exchange.fetch_order, order_id, symbol_id)
        except Exception as e:
            self.logger.error(f"Error fetching order {order_id} status for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
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
            
            # Use the correct symbol from markets for the fetch
            return market_info if market_info else self.exchange.market(symbol_id)
            
        except Exception as e:
            self.logger.error(f"Error fetching market info for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
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