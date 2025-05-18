import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, UTC
import time

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
VOLUME = 10  # 10 lots per trade
SL_PIPS = 50
MAGIC = 123456
CSV_PATH = "../signal.csv"

def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

def get_signal_for(symbol):
    df = pd.read_csv(CSV_PATH)
    row = df[df["symbol"] == symbol]
    return row.iloc[-1]["signal"] if not row.empty else "HOLD"

def get_current_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return positions[0] if positions else None

def calculate_sl_price(order_type, price, symbol):
    point = mt5.symbol_info(symbol).point
    if order_type == mt5.ORDER_TYPE_BUY:
        return price - SL_PIPS * point
    elif order_type == mt5.ORDER_TYPE_SELL:
        return price + SL_PIPS * point

def close_current_position(position):
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "price": mt5.symbol_info_tick(position.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask,
        "position": position.ticket,
        "deviation": 10,
        "magic": MAGIC,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })
    print(f"Closed {position.symbol} position: {result}")
    return result

def open_new_position(symbol, signal):
    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    sl_price = calculate_sl_price(order_type, price, symbol)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": VOLUME,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "deviation": 10,
        "magic": MAGIC,
        "comment": f"ML Signal {signal}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    print(f"[{datetime.now(UTC)}] Executed {symbol} {signal}: {result}")
    return result

if __name__ == "__main__":
    while True:
        try:
            initialize_mt5()

            for symbol in SYMBOLS:
                signal = get_signal_for(symbol)
                position = get_current_position(symbol)

                if signal not in ["BUY", "SELL"]:
                    print(f"[{datetime.now(UTC)}] {symbol}: No action required (signal={signal})")
                    continue

                if position:
                    current_type = "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL"
                    if current_type != signal:
                        close_current_position(position)
                        open_new_position(symbol, signal)
                    else:
                        print(f"[{datetime.now(UTC)}] {symbol}: Holding current {signal} position")
                else:
                    open_new_position(symbol, signal)

        except Exception as e:
            print(f"[{datetime.now(UTC)}] Error: {e}")

        time.sleep(900)  # check every 15 minute