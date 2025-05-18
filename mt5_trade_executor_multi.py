import time
import pandas as pd
import MetaTrader5 as mt5
import time
import csv

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
CSV_SIGNAL_PATH = "../signal.csv"
VOLUME = 5
SL_PIPS = 50
MAGIC_NUMBER = 20250517

def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed, error code = {mt5.last_error()}")
    print("MT5 initialized")

def get_signal_for(symbol, signals_df):
    row = signals_df[signals_df['symbol'] == symbol]
    if row.empty:
        return None
    return row.iloc[-1]['signal']

def get_current_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    # Assuming single position per symbol
    return positions[0]

def close_position(position):
    symbol = position.symbol
    lot = position.volume
    price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": price,
        "deviation": 10,
        "magic": MAGIC_NUMBER,
        "comment": "Close position"
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close position {position.ticket} on {symbol}, retcode={result.retcode}")
        return False
    print(f"Closed position {position.ticket} on {symbol}")
    return True

def log_trade_prices(symbol, price_before_close, price_before_open, price_after_open):
    with open("trade_prices_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([pd.Timestamp.now(), symbol, price_before_close, price_before_open, price_after_open])

def open_position(symbol, trade_type):
    price_info = mt5.symbol_info_tick(symbol)
    if not price_info:
        print(f"Failed to get price info for {symbol}")
        return False

    price = price_info.ask if trade_type == mt5.ORDER_TYPE_BUY else price_info.bid
    sl_price = price - SL_PIPS * mt5.symbol_info(symbol).point if trade_type == mt5.ORDER_TYPE_BUY else price + SL_PIPS * mt5.symbol_info(symbol).point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": VOLUME,
        "type": trade_type,
        "price": price,
        "sl": sl_price,
        "deviation": 10,
        "magic": MAGIC_NUMBER,
        "comment": "Open position"
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to open {'BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL'} position for {symbol}, retcode={result.retcode}")
        return False
    print(f"Opened {'BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL'} position for {symbol}")
    return True

def main():
    initialize_mt5()
    while True:
        try:
            signals_df = pd.read_csv(CSV_SIGNAL_PATH)
        except Exception as e:
            print(f"Failed to read signals file: {e}")
            time.sleep(60)
            continue

        for symbol in SYMBOLS:
            signal = get_signal_for(symbol, signals_df)
            if signal not in ["BUY", "SELL"]:
                print(f"No valid signal for {symbol}")
                continue

            desired_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

            position = get_current_position(symbol)

            # Get price before closing
            price_before_close = None
            if position is not None:
                price_info = mt5.symbol_info_tick(symbol)
                if price_info:
                    price_before_close = price_info.last  # or price_info.ask/bid

                print(f"Closing existing position for {symbol} at price {price_before_close}")
                if not close_position(position):
                    print(f"Failed to close position for {symbol}, skipping opening new position")
                    continue  # skip to next symbol if can't close

            # Add a short delay before opening new position (optional)
            time.sleep(1)

            # Get price before opening new position
            price_info = mt5.symbol_info_tick(symbol)
            price_before_open = price_info.last if price_info else None
            print(f"Opening {signal} position for {symbol} at price {price_before_open}")

            if not open_position(symbol, desired_type):
                print(f"Failed to open {signal} position for {symbol}")
            else:
                log_trade_prices(symbol, price_before_close, price_before_open, price_after_open)


            # Log price after opening (can check position again if needed)
            time.sleep(2)  # optional small delay
            price_info_after = mt5.symbol_info_tick(symbol)
            price_after_open = price_info_after.last if price_info_after else None
            print(f"Price after opening position for {symbol}: {price_after_open}")



        time.sleep(900)  # wait 15 minutes before next check

if __name__ == "__main__":
    main()
