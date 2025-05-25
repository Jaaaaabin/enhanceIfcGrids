import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import seaborn as sns

# Get the current script directory
def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

# Load JSON file
json_path = os.path.join(get_script_directory(), "data_parsed_useasis_or_change_solutions.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define known feedback categories
known_categories = {
    "Remove unnecessary grid lines (that do not align with key elements)",
    "Simplify overly dense areas (where too many lines create functional confusion)",
    "Add missing references (i.e., structural grids)",
    "Add missing references (i.e., non-structural grids)",
    "Add partial grids to support specific areas (e.g., service zones, irregular spaces)",
    "Reassign structural and non-structural grid roles where misclassified",
    "Use as-is / Make only very minor changes"
}

# Count responses per sub-solution
solution_summary = defaultdict(Counter)
for participant_feedback in data.values():
    for sol_key, feedbacks in participant_feedback.items():
        for feedback in feedbacks:
            key = feedback if feedback in known_categories else "Other"
            solution_summary[sol_key][key] += 1

# Create DataFrame and order columns
df = pd.DataFrame(solution_summary).fillna(0).astype(int).T
use_key = "Use as-is / Make only very minor changes"
df = df[[c for c in df.columns if c != use_key] + [use_key]]

# Normalize to percentages
df_percent = df.div(df.sum(axis=1), axis=0).fillna(0)

# Sort labels and define x-positions with spacing between cases
def sort_key(label):
    c, s = label.replace("Solution ", "").split(".")
    return int(c), int(s)

sorted_labels = sorted(df_percent.index, key=sort_key)
df_percent = df_percent.loc[sorted_labels]

x_pos = []
case_seen = set()
current_x = 0
for label in df_percent.index:
    case = int(label.replace("Solution ", "").split(".")[0])
    if case in case_seen:
        current_x += 1
    else:
        if len(case_seen) > 0:
            current_x += 1.5
        case_seen.add(case)
    x_pos.append(current_x)
    current_x += 1

# Color settings
safe_pastels = ["#aec7e8", "#ffbb78", "#c5b0d5", "#f7b6d2", "#c49c94", "#9edae5"]
custom_colors = {}
for i, col in enumerate(df_percent.columns):
    if col == use_key:
        custom_colors[col] = "#4CAF50"  # Green
    elif col == "Other":
        custom_colors[col] = "#A9A9A9"  # Grey
    else:
        custom_colors[col] = safe_pastels[i % len(safe_pastels)]

# Plot
fig, ax = plt.subplots(figsize=(16, 6))
bottom = pd.Series(0, index=df_percent.index)
for col in df_percent.columns:
    ax.bar(x_pos, df_percent[col], bottom=bottom, label=col, color=custom_colors[col])
    bottom += df_percent[col]

# Format
ax.set_ylabel("Proportion of Responses", fontsize=16)
ax.set_xlabel("Solution Number", fontsize=16)
ax.set_xticks(x_pos)
ax.set_xticklabels([label.replace("Solution ", "") for label in df_percent.index], rotation=45, ha='right')
ax.set_ylim(0, 1.05)
ax.grid(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fontsize='large')
plt.tight_layout()

# Save
output_path = os.path.join(get_script_directory(), "survey_use_or_change.png")
plt.savefig(output_path, dpi=300)