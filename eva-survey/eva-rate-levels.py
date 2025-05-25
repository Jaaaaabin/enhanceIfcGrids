
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Input data: [participant, overall rating, group1 rating, group2 rating]
ratings = [
    [1, 2, 2, 3],
    [2, 2, 2, 2],
    [3, 3, 2, 3],
    [4, 3, 2, 4],
    [5, 2, 3, 3],
    [6, 3, 2, 3],
    [7, 2, 2, 3],
    [8, 3, 2, 4],
    [9, 3, 2, 3],
    [10, 4, 4, 4],
    [11, 3, 2, 3]
]

# Create DataFrame
df_ratings = pd.DataFrame(ratings, columns=["ID", "Overall", "Group 1", "Group 2"])

# Count rating frequencies
overall_counts = df_ratings["Overall"].value_counts().sort_index()
group1_counts = df_ratings["Group 1"].value_counts().sort_index()
group2_counts = df_ratings["Group 2"].value_counts().sort_index()

# Hatching patterns and edge color mapping
hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
rating_edge_colors = {
    2: "navy",
    3: "maroon",
    4: "orange"
}

# Plotting function
def plot_hatched_pies(ax, data, title, edge_color_map):
    total = sum(data)
    start_angle = 90
    radius = 1
    for i, (label, count) in enumerate(data.items()):
        angle = 360 * count / total
        theta = (start_angle + angle / 2) * (np.pi / 180)

        wedge = mpatches.Wedge(
            center=(0, 0), r=radius, theta1=start_angle, theta2=start_angle + angle,
            facecolor='white', hatch=hatches[i % len(hatches)],
            edgecolor=edge_color_map.get(label, "black"), linewidth=1.4
        )
        ax.add_patch(wedge)

        x = 0.7 * radius * np.cos(theta)
        y = 0.7 * radius * np.sin(theta)
        ax.text(x, y, f"Level {label}\n({count})", ha='center', va='center', fontsize=11, weight='bold')

        start_angle += angle

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(10, 6))
plot_hatched_pies(axes[0], overall_counts, "Overall Rating", rating_edge_colors)
plot_hatched_pies(axes[1], group1_counts, "Rating – Group 1 \n (models 1, 2, 3, 7)", rating_edge_colors)
plot_hatched_pies(axes[2], group2_counts, "Rating – Group 2 \n (models 4, 5, 6, 8)", rating_edge_colors)

plt.tight_layout()
plt.show()

import os

def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

# Save the figure
output_path = get_script_directory() + "/survey_rating_dist.png"
fig.savefig(output_path, dpi=300)
