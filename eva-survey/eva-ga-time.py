import os
from datetime import datetime

base_folder = r'C:\dev\phd\enrichIFC\enrichIFC\solution_ga'

for subfolder in os.listdir(base_folder):
    pareto_path = os.path.join(base_folder, subfolder, 'pareto_front')
    if os.path.isdir(pareto_path):
        modified_times = []
        for root, dirs, files in os.walk(pareto_path):
            for filename in files:
                if filename.endswith('.html'):
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        mtime = os.path.getmtime(file_path)
                        modified_times.append(mtime)
        
        print(f"\n=== Folder: {pareto_path} ===")
        if modified_times:
            earliest = min(modified_times)
            latest = max(modified_times)
            time_diff_seconds = latest - earliest

            print("Earliest modified time :", datetime.fromtimestamp(earliest))
            print("Latest modified time   :", datetime.fromtimestamp(latest))
            print("Time difference [min]  :", time_diff_seconds/60.0)
        else:
            print("No .html files found.")