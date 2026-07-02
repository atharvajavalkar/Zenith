"""XAUUSD Monte Carlo trading signal runner."""

import time

from config import ERROR_RETRY_SECONDS, POLL_INTERVAL_SECONDS
from strategy.signal import signal_once


def print_signal(sig: dict) -> None:
    print("\n==============================")
    print("XAUUSD MONTE CARLO INDICATION")
    print("==============================")
    print(f"LIVE_TIME: {sig['LIVE_TIME']}")
    print(f"LIVE_PRICE: {sig['LIVE_PRICE']}")
    print(f"SIGNAL: {sig['SIGNAL']}")
    print(f"BIAS_1H: {sig['BIAS_1H']}")
    print(f"BIAS_4H: {sig['BIAS_4H']}")
    print(f"CONTEXT_4H: {sig['CONTEXT_4H']}")
    print(f"STATE: {sig['STATE']}")
    print(f"SUCCESS_PERCENT: {sig['SUCCESS_PERCENT']}")
    print(f"RR: {sig['RR']}")
    print(f"ENTRY: {sig['ENTRY']}")
    print(f"SL: {sig['SL']}")
    print(f"TP: {sig['TP']}")
    print(f"DEMAND_TF: {sig['DEMAND_TF']}")
    print(f"SUPPLY_TF: {sig['SUPPLY_TF']}")
    print(f"EXIT: {sig['EXIT']}")
    print("==============================\n")


def run() -> None:
    last_candle_time = None
    first_run = True

    while True:
        try:
            sig = signal_once()

            if first_run or sig["CANDLE_TIME"] != last_candle_time:
                first_run = False
                last_candle_time = sig["CANDLE_TIME"]
                print_signal(sig)

            time.sleep(POLL_INTERVAL_SECONDS)

        except Exception as e:
            print("\nERROR:", e)
            time.sleep(ERROR_RETRY_SECONDS)


if __name__ == "__main__":
    run()
