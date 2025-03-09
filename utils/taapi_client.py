"""
Taapi.io API Client

This module provides utilities for interacting with the taapi.io API
to fetch cryptocurrency market data and technical indicators.
"""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class TaapiClient:
    """
    Client for the taapi.io API, providing methods to fetch market data and technical indicators.
    """
    
    def __init__(self, api_key=None, api_url=None, cache_dir=None):
        """
        Initialize the taapi.io client.
        
        Args:
            api_key (str, optional): The API key for taapi.io. Defaults to TAAPI_API_KEY env variable.
            api_url (str, optional): The API base URL. Defaults to TAAPI_API_URL env variable.
            cache_dir (str, optional): Directory to cache API responses. Defaults to CACHE_DIR env variable.
        """
        self.api_key = api_key or os.getenv('TAAPI_API_KEY')
        self.api_url = api_url or os.getenv('TAAPI_API_URL')
        self.cache_dir = cache_dir or os.getenv('CACHE_DIR', 'cache')
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(exist_ok=True)
        
        # Check if API key is available
        if not self.api_key:
            raise ValueError("No API key provided. Set TAAPI_API_KEY in .env file or pass as argument.")
            
        # Validate API credentials
        self._validate_credentials()
        
    def _validate_credentials(self):
        """Validate API credentials by making a simple request."""
        try:
            # Make a simple request to verify API key
            response = self._make_request(
                endpoint="/metadata",
                params={"secret": self.api_key}
            )
            print(f"Taapi.io API connection successful. Status: {response.get('status', 'Unknown')}")
        except Exception as e:
            print(f"Warning: Couldn't validate taapi.io credentials: {e}")
    
    def _make_request(self, endpoint, params, use_cache=True, cache_ttl=3600):
        """
        Make a request to the taapi.io API with caching.
        
        Args:
            endpoint (str): API endpoint path.
            params (dict): Query parameters for the request.
            use_cache (bool, optional): Whether to use cached responses. Defaults to True.
            cache_ttl (int, optional): Time-to-live for cached responses in seconds. Defaults to 3600.
            
        Returns:
            dict: The response data in JSON format.
        """
        # Add API key to parameters
        params['secret'] = self.api_key
        
        # Create cache key from endpoint and parameters
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        cache_key = cache_key.replace('/', '_').replace('?', '_').replace('&', '_')
        cache_file = Path(self.cache_dir) / f"{cache_key}.json"
        
        # Check if cache exists and is not expired
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < cache_ttl:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error reading cache: {e}")
        
        # Make the API request
        url = f"{self.api_url}{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            data = response.json()
            
            # Save to cache
            if use_cache:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                    
            return data
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
    
    def get_indicator(self, symbol, indicator, interval, params=None):
        """
        Get a specific technical indicator value.
        
        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            indicator (str): Indicator name (e.g., 'rsi', 'macd', 'bb').
            interval (str): Timeframe interval (e.g., '1h', '4h', '1d').
            params (dict, optional): Additional parameters for the indicator.
            
        Returns:
            dict: The indicator data.
        """
        query_params = {
            'symbol': symbol,
            'interval': interval
        }
        
        # Add additional parameters
        if params:
            query_params.update(params)
            
        return self._make_request(
            endpoint=f"/v1/{indicator}",
            params=query_params
        )
    
    def get_candles(self, symbol, interval, limit=100, start=None, end=None):
        """
        Get candlestick data.
        
        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            interval (str): Timeframe interval (e.g., '1h', '4h', '1d').
            limit (int, optional): Number of candles to return. Defaults to 100.
            start (str, optional): Start time in ISO format.
            end (str, optional): End time in ISO format.
            
        Returns:
            pd.DataFrame: DataFrame containing candlestick data.
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start:
            params['start'] = start
        if end:
            params['end'] = end
            
        response = self._make_request(
            endpoint="/v1/candles",
            params=params
        )
        
        # Convert response to DataFrame
        if isinstance(response, list) and len(response) > 0:
            df = pd.DataFrame(response)
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        else:
            print(f"Warning: Empty or invalid response for candles: {response}")
            return pd.DataFrame()
    
    def get_multiple_indicators(self, symbol, interval, indicators, start=None, end=None):
        """
        Get multiple indicators in one request using the bulk endpoint.
        
        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            interval (str): Timeframe interval (e.g., '1h', '4h', '1d').
            indicators (list): List of indicator dictionaries.
            start (str, optional): Start time in ISO format.
            end (str, optional): End time in ISO format.
            
        Returns:
            dict: Dictionary containing all requested indicators.
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'indicators': json.dumps(indicators)
        }
        
        if start:
            params['start'] = start
        if end:
            params['end'] = end
            
        return self._make_request(
            endpoint="/v1/bulk",
            params=params
        )
    
    def get_ohlcv(self, symbol, interval, limit=100, start=None, end=None, backtrack=0, type='crypto'):
        """
        Get OHLCV (Open, High, Low, Close, Volume) data by using individual requests for each value.
        
        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            interval (str): Timeframe interval (e.g., '1h', '4h', '1d').
            limit (int, optional): Number of data points to return. Defaults to 100.
            start (str, optional): Start time in ISO format.
            end (str, optional): End time in ISO format.
            backtrack (int, optional): Number of candles to go back. Defaults to 0.
            type (str, optional): Asset type ('crypto' or 'stocks'). Defaults to 'crypto'.
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data.
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'backtracks': limit - 1,  # -1 because it includes current candle
            'backtrack': backtrack,
            'type': type
        }
        
        if type == 'crypto':
            params['exchange'] = 'binance'
        
        try:
            # Get open price
            open_data = self._make_request(
                endpoint="/open",
                params=params
            )
            
            # Get high price
            high_data = self._make_request(
                endpoint="/high",
                params=params
            )
            
            # Get low price
            low_data = self._make_request(
                endpoint="/low",
                params=params
            )
            
            # Get close price
            close_data = self._make_request(
                endpoint="/close",
                params=params
            )
            
            # Get volume
            volume_data = self._make_request(
                endpoint="/volume",
                params=params
            )
            
            # Create dataframe
            df = pd.DataFrame()
            
            # Set date range
            now = datetime.now()
            
            # The API returns data from newest to oldest, so we need to reverse
            dates = []
            for i in range(limit):
                # Calculate date by going back i intervals
                if interval.endswith('m'):
                    minutes = int(interval[:-1])
                    date = now - timedelta(minutes=minutes * i)
                elif interval.endswith('h'):
                    hours = int(interval[:-1])
                    date = now - timedelta(hours=hours * i)
                elif interval == '1d':
                    date = now - timedelta(days=i)
                elif interval == '1w':
                    date = now - timedelta(weeks=i)
                else:
                    date = now - timedelta(days=i)
                    
                dates.append(date)
            
            # Assign data to dataframe
            df['date'] = dates
            
            # Initialize with empty lists
            df['open'] = [None] * len(dates)
            df['high'] = [None] * len(dates)
            df['low'] = [None] * len(dates)
            df['close'] = [None] * len(dates)
            df['volume'] = [None] * len(dates)
            
            # Fill values from API responses
            if isinstance(open_data, list):
                for i, value in enumerate(open_data[:min(len(open_data), limit)]):
                    if i < len(df):
                        df.loc[i, 'open'] = float(value)
                        
            if isinstance(high_data, list):
                for i, value in enumerate(high_data[:min(len(high_data), limit)]):
                    if i < len(df):
                        df.loc[i, 'high'] = float(value)
                        
            if isinstance(low_data, list):
                for i, value in enumerate(low_data[:min(len(low_data), limit)]):
                    if i < len(df):
                        df.loc[i, 'low'] = float(value)
                        
            if isinstance(close_data, list):
                for i, value in enumerate(close_data[:min(len(close_data), limit)]):
                    if i < len(df):
                        df.loc[i, 'close'] = float(value)
                        
            if isinstance(volume_data, list):
                for i, value in enumerate(volume_data[:min(len(volume_data), limit)]):
                    if i < len(df):
                        df.loc[i, 'volume'] = float(value)
                        
            # Drop rows with missing values
            df = df.dropna()
            
            # Sort by date (oldest first)
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            print(f"Error getting OHLCV data: {e}")
            return pd.DataFrame()

    def get_price_history(self, symbol, interval, limit=100, start=None, end=None, type='crypto'):
        """
        Get historical price data and format it for the backtesting system.
        
        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            interval (str): Timeframe interval (e.g., '1h', '4h', '1d').
            limit (int, optional): Number of candles to return. Defaults to 100.
            start (str, optional): Start time in ISO format.
            end (str, optional): End time in ISO format.
            type (str, optional): Asset type ('crypto' or 'stocks'). Defaults to 'crypto'.
            
        Returns:
            pd.DataFrame: DataFrame ready for import into the backtesting system.
        """
        # Try to get candle data directly (might not be available in all plans)
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit,
                'type': type
            }
            
            if type == 'crypto':
                params['exchange'] = 'binance'
                
            if start:
                params['start'] = start
            if end:
                params['end'] = end
                
            response = self._make_request(
                endpoint="/candles",
                params=params
            )
            
            # If we got a valid response with candles, use it
            if isinstance(response, list) and len(response) > 0:
                df = pd.DataFrame(response)
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Format for our system
                formatted_df = pd.DataFrame()
                formatted_df['date'] = df['date']
                formatted_df['open'] = df['open'].astype(float)
                formatted_df['high'] = df['high'].astype(float)
                formatted_df['low'] = df['low'].astype(float)
                formatted_df['close'] = df['close'].astype(float)
                formatted_df['volume'] = df['volume'].astype(float)
                formatted_df['ticker'] = symbol.replace('/', '_')
                formatted_df['timeframe'] = interval
                
                return formatted_df
        except Exception as e:
            print(f"Candles endpoint not available: {e}")
        
        # If candles endpoint failed, try to get data using individual indicators
        print(f"Trying alternative method to get price data for {symbol}...")
        df = self.get_ohlcv(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start=start,
            end=end,
            type=type
        )
        
        if df.empty:
            return df
        
        # Create a formatted DataFrame for our backtesting system
        formatted_df = pd.DataFrame()
        formatted_df['date'] = df['date']
        formatted_df['open'] = df['open'].astype(float)
        formatted_df['high'] = df['high'].astype(float)
        formatted_df['low'] = df['low'].astype(float)
        formatted_df['close'] = df['close'].astype(float)
        formatted_df['volume'] = df['volume'].astype(float)
        formatted_df['ticker'] = symbol.replace('/', '_')
        formatted_df['timeframe'] = interval
        
        return formatted_df

# Helper function to instantiate the client
def get_taapi_client():
    """Create and return a taapi.io client instance."""
    return TaapiClient()

if __name__ == "__main__":
    # Example usage
    client = get_taapi_client()
    
    try:
        # Get BTC/USDT price data
        print("Fetching BTC/USDT 1-day candles...")
        btc_data = client.get_price_history(
            symbol="BTC/USDT",
            interval="1d",
            limit=30  # Last 30 days
        )
        
        if not btc_data.empty:
            print(f"Retrieved {len(btc_data)} candles")
            print(btc_data.head())
        else:
            print("No data received.")
            
        # Get RSI indicator
        print("\nFetching RSI for BTC/USDT...")
        rsi = client.get_indicator(
            symbol="BTC/USDT",
            indicator="rsi",
            interval="1d",
            params={"period": 14}
        )
        print(f"RSI value: {rsi.get('value', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}") 