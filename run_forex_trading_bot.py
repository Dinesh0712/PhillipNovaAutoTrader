import subprocess
import time
import os

# Paths to scripts
SIGNAL_SCRIPT = "signal_generator.py"
EXECUTOR_SCRIPT = "mt5_trade_executor_multi.py"

VENV_PYTHON = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")

def launch_script(script_name):
    return subprocess.Popen([VENV_PYTHON, script_name])

if __name__ == "__main__":
    print("🚀 Starting Signal Generator and Trade Executor...")
    
    # Start both scripts
    signal_proc = launch_script(SIGNAL_SCRIPT)
    time.sleep(5)  # small delay to ensure signal starts first
    executor_proc = launch_script(EXECUTOR_SCRIPT)

    print("🟢 Both bots running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(10)
            # optionally check if either process has crashed
            if signal_proc.poll() is not None:
                print("⚠️ Signal Generator stopped.")
            if executor_proc.poll() is not None:
                print("⚠️ Trade Executor stopped.")
    except KeyboardInterrupt:
        print("🛑 Stopping both bots...")
        signal_proc.terminate()
        executor_proc.terminate()
        print("✅ Shutdown complete.")