import pandas as pd
import MetaTrader5 as mt5
import ta
from datetime import datetime, timezone as tz
import joblib
import os

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = mt5.TIMEFRAME_M1
NUM_BARS = 40000
CSV_OUTPUT_PATH = "../signal.csv"

def fetch_data(symbol):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, NUM_BARS)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data received for {symbol} from MT5")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    return df

def engineer_features(df):
    close = df["Close"]
    df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
    df["macd"] = ta.trend.MACD(close).macd_diff()
    df["sma"] = close.rolling(window=20).mean()
    df["returns"] = close.pct_change()
    df["volatility"] = df["returns"].rolling(window=10).std()
    df = df.dropna()

    features = ["rsi", "macd", "sma", "returns", "volatility"]
    for f in features:
        df[f] = df[f].astype(float)

    return df, features

def predict_signal(model, features, latest_row):
    X_latest = latest_row[features].to_numpy().astype(float).reshape(1, -1)
    prediction = model.predict(X_latest)[0]
    return "BUY" if prediction == 1 else "SELL"

def run_once():
    signal_data = []

    for symbol in SYMBOLS:
        try:
            df = fetch_data(symbol)
            df, features = engineer_features(df)

            model_path = f"model_{symbol}.pkl"
            if os.path.exists(model_path):
                model, features = joblib.load(model_path)
            else:
                raise FileNotFoundError(f"Model file for {symbol} not found. Train models first.")

            latest_row = df.iloc[-1:].copy()
            signal = predict_signal(model, features, latest_row)

            signal_data.append({
                "timestamp": datetime.now(tz.UTC).isoformat(),
                "symbol": symbol,
                "signal": signal
            })
        except Exception as e:
            print(f"[{datetime.now(tz.UTC)}] Error for {symbol}: {e}")

    pd.DataFrame(signal_data).to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"[{datetime.now(tz.UTC)}] Signal file updated")

if __name__ == "__main__":
    import time
    while True:
        run_once()
        time.sleep(900)  # Run every 15 minute