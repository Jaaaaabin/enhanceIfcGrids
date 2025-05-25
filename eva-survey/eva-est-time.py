import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_script_directory():
    """
    Returns the directory where the script is located.
    """
    return os.path.dirname(os.path.abspath(__file__))

# Prepare the initial data again
df_time_matrix = pd.DataFrame([
    ["20 - 30 mins", "20 - 30 mins", "20 - 30 mins", "30 - 40 mins", "20 - 30 mins", "20 - 30 mins", "30 - 40 mins", "40 - 60 mins"],
    ["10 - 20 mins", "20 - 30 mins", "20 - 30 mins", "20 - 30 mins", "20 - 30 mins", "20 - 30 mins", "20 - 30 mins", "30 - 40 mins"],
    ["1 - 2 hours", "> 2 hours", "1 - 2 hours", "> 2 hours", "1 - 2 hours", "1 - 2 hours", "> 2 hours", "> 2 hours"],
    ["20 - 30 mins", "30 - 40 mins", "10 - 20 mins", "1 - 2 hours", "30 - 40 mins", "30 - 40 mins", "20 - 30 mins", "1 - 2 hours"],
    ["1 - 2 hours", "1 - 2 hours", "1 - 2 hours", "1 - 2 hours", "1 - 2 hours", "1 - 2 hours", "1 - 2 hours", "> 2 hours"],
    ["10 - 20 mins", "20 - 30 mins", "10 - 20 mins", "30 - 40 mins", "10 - 20 mins", "20 - 30 mins", "10 - 20 mins", "30 - 40 mins"],
    ["20 - 30 mins", "20 - 30 mins", "20 - 30 mins", "30 - 40 mins", "20 - 30 mins", "20 - 30 mins", "20 - 30 mins", "30 - 40 mins"],
    ["10 - 20 mins", "20 - 30 mins", "0 - 10 mins", "20 - 30 mins", "0 - 10 mins", "10 - 20 mins", "10 - 20 mins", "20 - 30 mins"],
    ["30 - 40 mins", "30 - 40 mins", "30 - 40 mins", "40 - 60 mins", "20 - 30 mins", "20 - 30 mins", "20 - 30 mins", "30 - 40 mins"],
    ["0 - 10 mins", "10 - 20 mins", "10 - 20 mins", "10 - 20 mins", "0 - 10 mins", "10 - 20 mins", "10 - 20 mins", "20 - 30 mins"],
    ["1 - 2 hours", "40 - 60 mins", "40 - 60 mins", "1 - 2 hours", "40 - 60 mins", "40 - 60 mins", "40 - 60 mins", "1 - 2 hours"]
], columns=[f"Model {i+1}" for i in range(8)])

# Reshape to long format
df_long = df_time_matrix.melt(var_name="Model", value_name="Time Estimate")

# Count data for bar plot
count_data = df_long.groupby(["Model", "Time Estimate"]).size().reset_index(name="Count")

paired_colors = sns.color_palette("Paired", 7)
paired_color_map = dict(zip([
    "0 - 10 mins",
    "10 - 20 mins",
    "20 - 30 mins",
    "30 - 40 mins",
    "40 - 60 mins",
    "1 - 2 hours",
    "> 2 hours"
], paired_colors))

# Save the plot as a high-resolution PNG file
output_path = get_script_directory() + "/survey_time_estimate_dist.png"

plt.figure(figsize=(12, 4))
ax = sns.barplot(
    data=count_data,
    x="Model",
    y="Count",
    hue="Time Estimate",
    hue_order=list(paired_color_map.keys()),
    palette=paired_color_map
)

plt.legend(bbox_to_anchor=(0.5, 1.12), loc="upper center", ncol=7, fontsize='medium')
plt.ylabel("Number of Participants")
plt.xlabel("")  # Remove x-axis label
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_path, dpi=300)