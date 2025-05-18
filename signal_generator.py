import os
import time
import pandas as pd
import numpy as np
import ta
import joblib
import MetaTrader5 as mt5
from  sklearn.linear_model import LinearRegression
import time

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = mt5.TIMEFRAME_M15
NUM_BARS = 5000
CSV_OUTPUT_PATH = "../signal.csv"

loaded_models = {}

def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed, error code = {mt5.last_error()}")
    print("MT5 initialized")

def fetch_data(symbol):
    utc_to = pd.Timestamp.now(tz='UTC')
    rates = mt5.copy_rates_from(symbol, TIMEFRAME, utc_to, NUM_BARS)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data received for {symbol}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def detect_wedge_pattern(df, window=20, tolerance=0.05, min_slope=1e-4, min_ratio=0.1, min_start_dist=None):
    is_rising_wedge = np.zeros(len(df))
    is_falling_wedge = np.zeros(len(df))

    for i in range(window - 1, len(df)):
        highs = df['high'].iloc[i-window+1:i+1].values.reshape(-1,1)
        lows = df['low'].iloc[i-window+1:i+1].values.reshape(-1,1)
        X = np.arange(window).reshape(-1,1)

        reg_high = LinearRegression().fit(X, highs)
        reg_low = LinearRegression().fit(X, lows)

        slope_high = reg_high.coef_[0][0]
        slope_low = reg_low.coef_[0][0]

        start_dist = reg_low.predict([[0]])[0][0] - reg_high.predict([[0]])[0][0]
        end_dist = reg_low.predict([[window-1]])[0][0] - reg_high.predict([[window-1]])[0][0]

        if min_start_dist is not None and start_dist < min_start_dist:
            continue

        converging = end_dist < start_dist * (1 - tolerance)

        rising_condition = slope_high > min_slope and slope_low > min_slope and slope_high < slope_low * (1 - min_ratio)
        falling_condition = slope_high < -min_slope and slope_low < -min_slope and slope_high > slope_low * (1 + min_ratio)

        if converging and rising_condition:
            is_rising_wedge[i] = 1

        if converging and falling_condition:
            is_falling_wedge[i] = 1

    df['is_rising_wedge'] = is_rising_wedge
    df['is_falling_wedge'] = is_falling_wedge
    return df

def engineer_features(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']

    df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['macd'] = ta.trend.MACD(close, window_slow=12, window_fast=6, window_sign=4).macd_diff()
    df['sma1'] = close.rolling(window=15).mean()
    df['sma2'] = close.rolling(window=50).mean()
    df['ema_21'] = close.ewm(span=21, adjust=False).mean()
    df['ema_8'] = close.ewm(span=8, adjust=False).mean()
    df['ema_13'] = close.ewm(span=13, adjust=False).mean()

    df['ema_8_21_diff'] = df['ema_8'] - df['ema_21']
    df['ema_8_13_diff'] = df['ema_8'] - df['ema_13']
    df['ema_13_21_diff'] = df['ema_13'] - df['ema_21']

    df['ema_8_above_13'] = (df['ema_8'] > df['ema_13']).astype(int)
    df['ema_8_above_21'] = (df['ema_8'] > df['ema_21']).astype(int)
    df['ema_13_above_21'] = (df['ema_13'] > df['ema_21']).astype(int)

    df['returns'] = close.pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()

    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    df['adx'] = adx_indicator.adx()
    df['dmp'] = adx_indicator.adx_pos()
    df['dmn'] = adx_indicator.adx_neg()

    df = df=detect_wedge_pattern(df, window=100)
    
    df = df.dropna()
    return df

def predict_signal(model, features, df):
    X = df[features].iloc[-1:].to_numpy()
    pred = model.predict(X)[0]
    return "BUY" if pred == 1 else "SELL"

def main():
    initialize_mt5()
    signals = []

    for symbol in SYMBOLS:
        try:
            df = fetch_data(symbol)
            df = engineer_features(df)

            if symbol not in loaded_models:
                model_file = f"model_{symbol}.pkl"
                if not os.path.exists(model_file):
                    print(f"Model file {model_file} not found. Skipping {symbol}.")
                    continue
                loaded_models[symbol] = joblib.load(model_file)
            
            model, features = loaded_models[symbol]
            signal = predict_signal(model, features, df)
            print(f"{symbol} signal: {signal}")
            signals.append({"symbol": symbol, "signal": signal, "timestamp": pd.Timestamp.now()})

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if signals:
        signal_df = pd.DataFrame(signals)
        signal_df.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"Signals saved to {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(900)  # 15 minutes
