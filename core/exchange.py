"""
Exchange module for handling communication with Binance USDM Futures using CCXT.
"""
import ccxt
import logging
from typing import Dict, List, Any, Optional, Tuple

class Exchange:
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the exchange connection.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize CCXT Binance USDM Futures exchange
        self.exchange = self._create_exchange()
        
        # Pre-load markets
        try:
            self.markets = self.exchange.load_markets()
            self.logger.info(f"Initialized exchange connection. Loaded {len(self.markets)} markets.")
        except Exception as e:
            self.logger.error(f"Failed to load markets: {e}")
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
    
    def get_balance(self) -> Dict:
        """Get account balance."""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker price for symbol."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Fetching ticker for symbol ID: {symbol_id}")
            return self.exchange.fetch_ticker(symbol_id)
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """
        Create a limit order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT' or 'BTCUSDT')
            side: 'buy' or 'sell'
            amount: Order quantity
            price: Order price
            
        Returns:
            Order information dictionary
        """
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Creating {side} limit order for symbol ID: {symbol_id}")
            return self.exchange.create_limit_order(symbol_id, side, amount, price)
        except Exception as e:
            self.logger.error(f"Error creating {side} limit order for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """Create a market order."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Creating {side} market order for symbol ID: {symbol_id}")
            return self.exchange.create_market_order(symbol_id, side, amount)
        except Exception as e:
            self.logger.error(f"Error creating {side} market order for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order by ID."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Cancelling order {order_id} for symbol ID: {symbol_id}")
            return self.exchange.cancel_order(order_id, symbol_id)
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id} for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def cancel_all_orders(self, symbol: str) -> List:
        """Cancel all orders for a symbol."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Cancelling all orders for symbol ID: {symbol_id}")
            return self.exchange.cancel_all_orders(symbol_id)
        except Exception as e:
            self.logger.error(f"Error cancelling all orders for {symbol} (ID: {self._get_symbol_id(symbol)}): {e}")
            raise
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders, optionally filtered by symbol."""
        try:
            symbol_id = None
            if symbol:
                symbol_id = self._get_symbol_id(symbol)
                self.logger.debug(f"Fetching open orders for symbol ID: {symbol_id}")
                
            return self.exchange.fetch_open_orders(symbol_id)
        except Exception as e:
            symbol_info = f" for {symbol} (ID: {self._get_symbol_id(symbol)})" if symbol else ""
            self.logger.error(f"Error fetching open orders{symbol_info}: {e}")
            raise
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """Get all open positions, optionally filtered by symbol."""
        try:
            symbols_array = None
            
            if symbol:
                symbol_id = self._get_symbol_id(symbol)
                symbols_array = [symbol_id]
                self.logger.debug(f"Fetching positions for symbol array: {symbols_array}")
            
            # Use the symbols array parameter for fetching positions
            positions = self.exchange.fetch_positions(symbols_array)
            
            # Filter out positions with zero size
            return [pos for pos in positions if float(pos.get('contracts', 0)) != 0]
        except Exception as e:
            symbol_info = f" for {symbol} (ID: {self._get_symbol_id(symbol)})" if symbol else ""
            self.logger.error(f"Error fetching positions{symbol_info}: {e}")
            raise
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get the status of an order."""
        try:
            symbol_id = self._get_symbol_id(symbol)
            self.logger.debug(f"Fetching order {order_id} status for symbol ID: {symbol_id}")
            return self.exchange.fetch_order(order_id, symbol_id)
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