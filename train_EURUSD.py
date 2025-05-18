import numpy as np
import pandas as pd
import ta
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

SYMBOL = "EURUSD"

# Load and combine CSVs - make sure Local time column is parsed correctly
df1 = pd.read_csv("EURUSD1.csv", parse_dates=['Local time'])
df2 = pd.read_csv("EURUSD2.csv", parse_dates=['Local time'])
df3 = pd.read_csv("EURUSD3.csv", parse_dates=['Local time'])
df = pd.concat([df1, df2, df3], ignore_index=True)


def detect_wedge_pattern(df, window=20, tolerance=0.05, min_slope=1e-4, min_ratio=0.1, min_start_dist=None):
    is_rising_wedge = np.zeros(len(df))
    is_falling_wedge = np.zeros(len(df))

    for i in range(window - 1, len(df)):
        highs = df['High'].iloc[i-window+1:i+1].values.reshape(-1,1)
        lows = df['Low'].iloc[i-window+1:i+1].values.reshape(-1,1)
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
    df = df.copy()  # avoid SettingWithCopyWarning
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["macd"] = ta.trend.MACD(close, window_slow=12, window_fast=6, window_sign=4).macd_diff()
    df["sma1"] = close.rolling(window=15).mean()
    df["sma2"] = close.rolling(window=50).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_8"] = close.ewm(span=8, adjust=False).mean()
    df["ema_13"] = close.ewm(span=13, adjust=False).mean()

    df["ema_8_21_diff"] = df["ema_8"] - df["ema_21"]
    df["ema_8_13_diff"] = df["ema_8"] - df["ema_13"]
    df["ema_13_21_diff"] = df["ema_13"] - df["ema_21"]

    df["ema_8_above_13"] = (df["ema_8"] > df["ema_13"]).astype(int)
    df["ema_8_above_21"] = (df["ema_8"] > df["ema_21"]).astype(int)
    df["ema_13_above_21"] = (df["ema_13"] > df["ema_21"]).astype(int)

    df["returns"] = close.pct_change()
    df["volatility"] = df["returns"].rolling(window=10).std()

    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_indicator.adx()
    df["dmp"] = adx_indicator.adx_pos()
    df["dmn"] = adx_indicator.adx_neg()

    df["target"] = (close.shift(-1) > close).astype(int)

    df = df=detect_wedge_pattern(df, window=100)

    df = df.dropna()

    features = [
        "rsi", 
        "macd", 
        "sma1", 
        # "sma2", 
        # "ema_21", 
        # "ema_8", 
        # "ema_13",
        # "ema_8_21_diff", 
        # "ema_8_13_diff", 
        # "ema_13_21_diff",
        # "ema_8_above_13", 
        "ema_8_above_21", 
        # "ema_13_above_21",
        # "returns", 
        "volatility", 
        # "adx", 
        # "dmp", 
        # "dmn",
        "is_rising_wedge", 
        "is_falling_wedge"
    ]
    for f in features:
        df[f] = df[f].astype(float)
    df["target"] = df["target"].astype(int)

    return df, features

def train_model(df, symbol):
    df, features = engineer_features(df)

    # df_sample = df.sample(n=min(len(df), 5000), random_state=42)
    
    # sns.pairplot(df_sample[features + ["target"]], hue="target", diag_kind="kde", plot_kws={"alpha":0.5})
    # plt.suptitle("Feature Relationships colored by Target", y=1.02)
    # plt.show()

    X = df[features].to_numpy()
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    ratio = sum(y_train == 0) / sum(y_train == 1)
    print(f"Class imbalance ratio (neg/pos): {ratio:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        early_stopping_rounds=10,
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=ratio,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print(f"Model class: {model.__class__}")
    print(f"Model module: {model.__module__}")

    # Try fit with early stopping and catch errors
    # try:
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    # except TypeError as e:
    #     print(f"Fit error: {e}")
    #     print("Retrying fit without early stopping...")
    #     model.fit(
    #         X_train, y_train,
    #         eval_set=[(X_test, y_test)],
    #         verbose=True
    #     )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"📊 [{symbol}] Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    joblib.dump((model, features), f"model_{symbol}.pkl")
    return model, features

if __name__ == "__main__":
    train_model(df, SYMBOL)
