import pandas as pd
import os
import json

# Define function to get the directory of the current script
def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

# Define CSV file path
csv_path = os.path.join(get_script_directory(), "data_useasis_or_change.csv")

# Load the CSV file
df = pd.read_csv(csv_path, header=None)

# Initialize the result dictionary
participant_change_solutions = {}

# Process each row (participant)
for idx, row in df.iterrows():
    participant_key = f"Participant {idx + 1}"
    participant_change_solutions[participant_key] = {}

    # Each pair of columns = one solution case
    num_cases = len(row) // 2
    for case in range(num_cases):
        col1 = row[2 * case]
        col2 = row[2 * case + 1]

        # Process col1 → Solution X.1
        if pd.notna(col1):
            items_1 = [item.strip() for item in str(col1).split(";")]
        else:
            items_1 = []

        # Process col2 → Solution X.2
        if pd.notna(col2):
            items_2 = [item.strip() for item in str(col2).split(";")]
        else:
            items_2 = []

        # Apply filtering rule
        clean_key = "Use as-is / Make only very minor changes"
        if clean_key in items_1:
            items_1 = [clean_key]
        if clean_key in items_2:
            items_2 = [clean_key]

        participant_change_solutions[participant_key][f"Solution {case + 1}.1"] = items_1
        participant_change_solutions[participant_key][f"Solution {case + 1}.2"] = items_2

# Save to JSON
output_path = os.path.join(get_script_directory(), "data_parsed_useasis_or_change_solutions.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(participant_change_solutions, f, indent=2)
