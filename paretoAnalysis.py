import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def collection_of_dfs(json_data, number_pft, storey_adjustment):

    data = []
    for key, value in json_data.items():
        st_values = value.get("st", [])
        ns_values = value.get("ns", [])
        storey = value.get("storey", 0) + storey_adjustment

        # Calculate statistics
        number_of_st_grids = len(st_values)
        sum_relation_all_st_grid = sum(st_values)
        average_relations_per_st_grid = sum_relation_all_st_grid / number_of_st_grids if number_of_st_grids != 0 else 0.0
        std_relations_per_st_grid = np.std(st_values) if st_values else np.nan
        sem_relations_per_st_grid = np.std(st_values, ddof=1) / np.sqrt(number_of_st_grids) if number_of_st_grids > 1 else np.nan
        
        # Extract key as floats
        float_keys = tuple(map(float, key.strip('()').split(',')))
        avg_rel_per_st_grid_per_storey = average_relations_per_st_grid / storey if storey != 0 else 0.0

        # Add relevant columns to data list
        data.append([
            float_keys[0], float_keys[1], len(ns_values), number_of_st_grids,
            sum(ns_values), sum_relation_all_st_grid, average_relations_per_st_grid,
            std_relations_per_st_grid, sem_relations_per_st_grid, storey,
            avg_rel_per_st_grid_per_storey
        ])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        "f unbound", "f distribution", "Num NSG", "Num STG", "Sum Rel all NSG",
        "Sum Rel all STG", "Avg Rel per STG", "Std Rel STG", "Sem Rel STG", "Storey",
        "Avg Rel per STG per Storey"
    ])
    
    # Split into Pareto and non-Pareto
    return df.iloc[:number_pft], df.iloc[number_pft:]

# ---------------------------------- metrics
def calculate_distances(points):

    return [np.linalg.norm(points[i] - points[i + 1]) for i in range(len(points) - 1)]

def delta_indicator(pareto_df):

    sorted_df = pareto_df.sort_values(by="f unbound")
    points = sorted_df[["f unbound", "f distribution"]].values
    distances = calculate_distances(points)
    avg_distance = np.mean(distances)
    delta = sum(abs(d - avg_distance) for d in distances) / ((len(points) - 1) * avg_distance)
    return delta

def diversity_metric(pareto_df):

    points = pareto_df[["f unbound", "f distribution"]].values
    nearest_distances = [
        min(np.linalg.norm(point - other) for j, other in enumerate(points) if i != j)
        for i, point in enumerate(points)
    ]
    avg_distance = np.mean(nearest_distances)
    spacing = np.mean([abs(d - avg_distance) for d in nearest_distances])
    return spacing

def calculate_diversity_metrics(nr, all_json_data, all_number_pareto_front, all_number_storey_adjustment):

    df_pareto_front, _ = collection_of_dfs(all_json_data[nr], all_number_pareto_front[nr], all_number_storey_adjustment[nr])
    delta_val = delta_indicator(df_pareto_front)
    diversity_val = diversity_metric(df_pareto_front)
    print(f"Project {nr + 1}, Delta Indicator: {delta_val:.3f}, Diversity Metric: {diversity_val:.3f}")
# ---------------------------------- metrics

def calculate_min_max_with_zscore(df, column_name, threshold=1.5):

    series = df[column_name]
    mean, std_dev = series.mean(), series.std()
    lower_bound, upper_bound = mean - threshold * std_dev, mean + threshold * std_dev

    filtered_series = series[(series >= lower_bound) & (series <= upper_bound)]
    pareto_min, pareto_max = filtered_series.min(), filtered_series.max()
    filtered_rows = df[(df[column_name] >= pareto_min) & (df[column_name] <= pareto_max)]

    return pareto_min, pareto_max, filtered_rows

def plot_multiple_cases(
    output_dir, all_json_data, all_number_pareto_front, all_number_storey_adjustment, all_highlight_points):

    fig, axes = plt.subplots(1, 8, figsize=(20, 8), sharey=True)
    plt.subplots_adjust(wspace=0.02)

    for nr in range(8):

        highlight_points = all_highlight_points[nr]
        df_pareto, df_non_pareto = collection_of_dfs(
            all_json_data[nr], all_number_pareto_front[nr], all_number_storey_adjustment[nr]
        )

        # - - - - - - - - - - - - - - - 
        # KDE plot for Pareto front points
        # sns.kdeplot(
        #     df_pareto['Avg Rel per STG per Storey'], 
        #     ax=axes[nr],
        #     vertical=True, 
        #     bw_adjust=0.95,
        #     alpha=0.6,
        #     color='blue', 
        #     fill=False,
        # )

        # KDE plot for Non Pareto front points
        kde = sns.kdeplot(
            df_non_pareto['Avg Rel per STG per Storey'], 
            ax=axes[nr],
            vertical=True, 
            bw_adjust=0.95,
            alpha=0.25,
            color='orange', 
            fill=True, 
        )

        # - - - - - - - - - - - - - - - 
        # Scatter plot for Non Pareto front
        axes[nr].scatter(
            df_non_pareto['f distribution'], 
            df_non_pareto['Avg Rel per STG per Storey'], 
            alpha=0.50,
            s=35,
            color='orange',
            edgecolor='none',
            label='non-Pareto front solutions ' if nr == 0 else ""
        )
        # Scatter plot for Pareto front
        axes[nr].scatter(
            df_pareto['f distribution'], 
            df_pareto['Avg Rel per STG per Storey'], 
            alpha=0.50,
            s=35,
            color='blue',
            edgecolor='none',
            label='Pareto front solutions' if nr == 0 else ""
        )

        # - - - - - - - - - - - - - - - 
        # Selected Visualization: Pareto front points
        for (x, y) in highlight_points[:2]:

            # Check if the point is in df_pareto
            pareto_matches = df_pareto[
                (abs(df_pareto['f unbound'] - x) < 0.0001) & 
                (abs(df_pareto['f distribution'] - y) < 0.0001)
            ]

            if not pareto_matches.empty:
                axes[nr].scatter(
                    pareto_matches['f distribution'],
                    pareto_matches['Avg Rel per STG per Storey'],
                    color='blue',
                    edgecolor='black',
                    s=35,
                    linewidth=0.5,
                    label='Pareto front solutions (selected for grid visualization)' if nr == 0 else "")

        # - - - - - - - - - - - - - - - 
        # Selected Visualization: Non Pareto front points
        for (x, y) in highlight_points[2:]:
            # Check if the point is in df_non_pareto
            non_pareto_matches = df_non_pareto[
                (abs(df_non_pareto['f unbound'] - x) < 0.0001) & 
                (abs(df_non_pareto['f distribution'] - y) < 0.0001)
            ]

            if not non_pareto_matches.empty:
                axes[nr].scatter(
                    non_pareto_matches['f distribution'],
                    non_pareto_matches['Avg Rel per STG per Storey'],
                    color='orange',
                    edgecolor='black',
                    s=35,
                    linewidth=0.5,
                    label='non-Pareto front solutions (selected for grid visualization)' if nr == 0 else "")
                
            # # Calculate and print filtered rows
            # pareto_min, pareto_max, filtered_rows = calculate_min_max_with_zscore(
            #     df_pareto, 'Avg Rel per STG per Storey'
            # )
            # # print for the visualization list.
            # print(f"Project {nr + 1}, {filtered_rows}")

        # # Draw horizontal lines for the filtered range
        # axes[nr].axhline(y=pareto_min, color='navy', linestyle='--', linewidth=0.5, alpha=0.7)
        # axes[nr].axhline(y=pareto_max, color='navy', linestyle='--', linewidth=0.5, alpha=0.7)

        # Set the title and limit the x-axis
        axes[nr].set_title(f"Case {nr + 1}")
        axes[nr].set_xlim(0, 1)

        # Remove individual x and y labels
        axes[nr].set_xlabel("")
        axes[nr].set_ylabel("")

        # Calculate occurrence percentages based on 'f distribution' of Pareto Front
        # total_points = len(df_non_pareto)
        # percent_values = np.linspace(0, 1, len(df_non_pareto))
        
        # Upper x-axis to show percentages based on point distribution
        secax = axes[nr].secondary_xaxis('top')
        step_positions = np.linspace(0, 5, 11)  # Smaller steps (0, 0.1, 0.2, ..., 1)
        secax.set_xticks(step_positions)
        percent_labels = [
            f"{int((step_positions[i] * 100))}%" for i in range(len(step_positions))
        ]
        secax.set_xticklabels(percent_labels)
        secax.tick_params(labeltop=False, labelbottom=True)

    # Shared x and y labels
    fig.supylabel(
        "Average number of bound relations per global grid \n per building storey (of each solution)",
        fontsize=15,
        x=0.02,
        ha='center')
    
    fig.supxlabel(
        "$f_{distribution}$",
        fontsize=15,
        y=0.03,
        ha='center')

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    unique_handles, unique_labels = [], []
    for i, label in enumerate(labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handles[i])
    fig.legend(unique_handles, unique_labels, bbox_to_anchor=(0.38, 0.92), fontsize=14)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_selection_for_visualization.png'), dpi=200)
    plt.close()



# SAVE
# def save_plot(plot_type, pareto_df, output_dir, file_prefix):
#     os.makedirs(output_dir, exist_ok=True)
#     filename = f"{file_prefix}_{plot_type}.png"
#     filepath = os.path.join(output_dir, filename)
#     sorted_df = pareto_df.sort_values(by="f unbound")
#     points = sorted_df[["f unbound", "f distribution"]].values

#     plt.figure(figsize=(8, 6))

#     if plot_type == "delta":
#         delta_value = delta_indicator(pareto_df)
#         plt.plot(points[:, 0], points[:, 1], 'bo-', label='Pareto Front', alpha=0.7)
#         plt.title(f'Pareto Front Spread (Delta Indicator: {delta_value:.2f})')

#     elif plot_type == "diversity":
#         diversity_value = diversity_metric(pareto_df)
#         nearest_distances = [
#             min(np.linalg.norm(point - other) for j, other in enumerate(points) if i != j)
#             for i, point in enumerate(points)
#         ]
#         plt.hist(nearest_distances, bins=10, edgecolor='black', alpha=0.7)
#         plt.title(f'Diversity Metric (Spacing): {diversity_value:.2f}')
    
#     plt.xlabel('f unbound')
#     plt.ylabel('f distribution')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(filepath, dpi=300)
#     plt.close()
