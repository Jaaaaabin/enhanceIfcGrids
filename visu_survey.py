import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

def get_script_directory():
    """
    Returns the directory where the script is located.
    """
    return os.path.dirname(os.path.abspath(__file__))

# Load the survey results.
survey_file_name = "results20250524.csv"
survey_dir = get_script_directory() + "/eva-survey"
survey_data_path = os.path.join(survey_dir, survey_file_name)
df_survey = pd.read_csv(survey_data_path, encoding='utf-8', low_memory=False)

# Define expected answers for each case
expected_answers = {
    1: "B",
    2: "C",
    3: "B",
    4: "B",
    5: "B",
    6: "C",
    7: "C",
    8: "C"
}

# Define valid answer options
valid_options = {"A", "B", "C"}

# Prepare summary collection
summary_data = []

for case in range(1, 9):
    col = f"Q1.{case}.1"
    if col in df_survey.columns:
        for response in df_survey[col].dropna():
            response_clean = str(response).strip()
            if response_clean == expected_answers[case]:
                label = "expected"
            elif response_clean in valid_options:
                label = "unexpected"
            else:
                label = "other"
            summary_data.append({"Case": case, "Response": response_clean, "Label": label})

# Convert to DataFrame and summarize
df_summary = pd.DataFrame(summary_data)
result_count = df_summary.groupby(["Case", "Label"]).size().unstack(fill_value=0)

