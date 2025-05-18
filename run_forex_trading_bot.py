import subprocess
import time
import sys
import os

def run_process(command):
    return subprocess.Popen(command, shell=True)

def main():
    signal_gen_script = os.path.abspath("signal_generator.py")
    trade_exec_script = os.path.abspath("mt5_trade_executor_multi.py")

    signal_process = run_process(f'"{sys.executable}" "{signal_gen_script}"')
    print("Signal generator started.")
    time.sleep(30)
    trade_process = run_process(f'"{sys.executable}" "{trade_exec_script}"')
    print("Trade executor started")

    try:
        while True:
            if signal_process.poll() is not None:
                print("Signal generator stopped unexpectedly. Restarting...")
                signal_process = run_process(f"{sys.executable} {signal_gen_script}")

            if trade_process.poll() is not None:
                print("Trade executor stopped unexpectedly. Restarting...")
                trade_process = run_process(f"{sys.executable} {trade_exec_script}")

            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        print("Stopping bot...")

        signal_process.terminate()
        trade_process.terminate()

        signal_process.wait()
        trade_process.wait()

if __name__ == "__main__":
    main()