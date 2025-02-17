import requests
import pandas as pd
import time

def fetch_binance_klines(symbol='BTCUSDT', interval='1h', start_time=None, end_time=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time
        
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error if the request failed
    return response.json()

def klines(symbol='BTCUSDT', interval='1h', start_str="1 Jan, 2025", end_str="1 Feb, 2025"):
  
    # Convert start and end date strings to Unix timestamps in milliseconds
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
    
    all_data = []
    while start_ts < end_ts:
        klines = fetch_binance_klines(symbol=symbol, interval=interval, start_time=start_ts)
        if not klines:
            break
        all_data.extend(klines)
        # Update start_ts: last kline's close time + 1ms to avoid overlapping data
        last_close_time = klines[-1][6]
        start_ts = last_close_time + 1
        # Respect Binance API rate limits
        time.sleep(0.5)
    return all_data
