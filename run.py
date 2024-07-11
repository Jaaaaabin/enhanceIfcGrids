import sys
import ctypes
import argparse

# Enable Virtual Terminal Processing to display ANSI colors in Windows console before numpy
if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    
import os
import shutil
import subprocess
from pathlib import Path
import threading
import time
import io

# Global variable to keep track of the last output time
last_output_time = time.time()
stop_event = threading.Event()  # Event to signal the monitoring thread to stop
original_stdout = sys.stdout

# Wrapper to capture print statements
class OutputMonitor(io.StringIO):
    def write(self, msg):
        global last_output_time
        last_output_time = time.time()
        super().write(msg)
        original_stdout.write(msg)

    def flush(self):
        original_stdout.flush()
        
# Save the original stdout
sys.stdout = OutputMonitor()

def main():

    # Define the directories
    project_directory = os.getcwd()
    source_directory = os.path.join(project_directory, "data/data_autocon_test")
    target_directory = os.path.join(project_directory, "data/data_autocon_ga")
    ga_script = os.path.join(project_directory, "ifc_ga_optimization.py")

    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run the GA optimization on specified mode.")
    parser.add_argument('--nr', type=int, default=0, help="the nr. of the ifc model.")
    args = parser.parse_args()
    
    # Iterate over all .ifc files in the source directory
    for ifc_file in Path(source_directory).glob("*.ifc"):

        # Define the source and target file paths
        source_file = ifc_file
        target_file = os.path.join(target_directory, ifc_file.name)

        if args.nr != 0:
            if str(args.nr) != ifc_file.name.split('-')[0]:
                continue

        try:
            # Step 1: Copy the file
            shutil.copy(source_file, target_file)
            print(f"\033[1m\033[94m{ifc_file.name}\033[0m is \033[1m\033[92mcopied\033[0m from TEST folder to GA folder")
            
            # Step 2: Run the main.py
            subprocess.run(["python", ga_script], check=True)
            print(f"\033[1m\033[94m{ifc_file.name}\033[0m has been analyzed with \033[1m\033[92mGA\033[0m.")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the main script: {e}")
        
        finally:
            # Step 3: Delete the copied file
            if os.path.exists(target_file):
                os.remove(target_file)
            print(f"\033[1m\033[94m{ifc_file.name}\033[0m is \033[1m\033[92mremoved\033[0m from the GA folder")
        
def monitor_output():
    global last_output_time
    while not stop_event.is_set():  # Check the event state
        if time.time() - last_output_time > 300:  # 300 seconds = 5 minutes
            sys.stdout.write('\n')  # Send Enter
            sys.stdout.flush()
            last_output_time = time.time()  # Reset the timer
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":

    monitor_thread = threading.Thread(target=monitor_output, daemon=True)
    monitor_thread.start()

    # Run the main function 
    main()

    stop_event.set()  # Signal the monitoring thread to stop
    monitor_thread.join() # Ensure the monitor thread finishes

    # Restore the original stdout
    sys.stdout = original_stdout
    sys.stdout.flush()