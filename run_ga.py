import sys
import ctypes
# Enable Virtual Terminal Processing to display ANSI colors in Windows console before numpy
if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    
import os
import shutil
import subprocess
from pathlib import Path

# Define the directories
project_directory = os.getcwd()
source_directory = os.path.join(project_directory, "data/data_autocon_test")
target_directory = os.path.join(project_directory, "data/data_autocon_ga")
main_script = os.path.join(project_directory, "main.py")

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

# Iterate over all .ifc files in the source directory
for ifc_file in Path(source_directory).glob("*.ifc"):
    # Define the source and target file paths
    source_file = ifc_file
    target_file = os.path.join(target_directory, ifc_file.name)

    try:
        # Step 1: Copy the file
        shutil.copy(source_file, target_file)
        print(f"\033[1m\033[94m{ifc_file.name}\033[0m is \033[1m\033[92mcopied\033[0m from TEST folder to GA folder")
        
        # Step 2: Run the main.py
        subprocess.run(["python", main_script, "--num_process", "24", "--set_plot", "True"], check=True)
        print(f"\033[1m\033[94m{ifc_file.name}\033[0m has been analyzed with \033[1m\033[92mGA\033[0m.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the main script: {e}")
    
    finally:
        # Step 3: Delete the copied file
        if os.path.exists(target_file):
            os.remove(target_file)
        print(f"\033[1m\033[94m{ifc_file.name}\033[0m is \033[1m\033[92mremoved\033[0m from the GA folder")
