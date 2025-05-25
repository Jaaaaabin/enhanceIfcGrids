import pandas as pd
import os
import json

# Define function to get script directory
def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

# Define file path relative to the script location
csv_path = os.path.join(get_script_directory(), "data_reason_selection.csv")

# Load the raw CSV file
df = pd.read_csv(csv_path, header=None)

# Initialize dictionary to store parsed results
participant_dict = {}

# Process each row into a nested dictionary
for idx, row in df.iterrows():
    participant_key = f"Participant {idx + 1}"
    participant_dict[participant_key] = {}
    for case_index, cell in enumerate(row):
        if pd.isna(cell):
            participant_dict[participant_key][case_index + 1] = []
        else:
            participant_dict[participant_key][case_index + 1] = [item.strip() for item in str(cell).split(";")]

# Save the structured dictionary to a JSON file
output_json_path = os.path.join(get_script_directory(), "data_parsed_reason_selection.json")
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(participant_dict, f, indent=2)
