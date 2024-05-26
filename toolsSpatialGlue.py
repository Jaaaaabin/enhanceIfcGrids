import os
import numpy as np
import matplotlib.pyplot as plt

def spatial_get_directions(lines):
    """
    Calculate the direction vectors for given lines.
    """
    return lines[:, 2:4] - lines[:, 0:2]

def spatial_normalize(vectors):
    """
    Normalize direction vectors.
    """
    lengths = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
    return vectors / lengths

def spatial_line_intersection(p1, p2, q1, q2):
    """
    Find the intersection point of two lines (extended indefinitely).
    """
    d1 = p2 - p1
    d2 = q2 - q1
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if cross == 0:
        return None  # Lines are parallel or collinear
    t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / cross
    intersection = p1 + t * d1
    return intersection

def spatial_extend_line_to_point(line, point):
    """
    Extend a line segment to a specified point.
    """
    if np.linalg.norm(line[:2] - point) < np.linalg.norm(line[2:4] - point):
        line[:2] = point
    else:
        line[2:4] = point

def spatial_convert_to_2d_lines(lines_data):
    """
    Convert locations to 2D lines array.
    """
    return np.array([[loc[0][0], loc[0][1], loc[1][0], loc[1][1]] for loc in [line['location'] for line in lines_data]])

def spatial_plot_lines_side_by_side(before_lines, after_lines, title, fig_save_path):
    """
    Plot before and after lines side by side.
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) #size should be dependent on the max and min delt_x and delta_y.

    # Plot before lines
    for line in before_lines:
        axs[0].plot(
            [line[0], line[2]], [line[1], line[3]], color='darkgreen', linestyle='-', linewidth=1.5, 
            marker='o', mec='k', markersize=3)
        
    axs[0].set_title(f"{title} (Before)")
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].grid(True)
    axs[0].axis('scaled')

    # Plot after lines
    for line in after_lines:
        axs[1].plot(
            [line[0], line[2]], [line[1], line[3]], color='darkorange', linestyle='-', linewidth=1.5,
            marker='o', mec='k', markersize=3)
        
    axs[1].set_title(f"{title} (After)")
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].grid(True)
    axs[1].axis('scaled')

    plt.axis('scaled')
    plt.suptitle(title)
    plt.savefig(fig_save_path)
    plt.close()

def spatial_process_lines(original_lines_data, output_figure_folder, plot_adjustments=False, width_threshold=2.0, identical_threshold=1e-6):
    """
    Process lines to find intersections and extend them accordingly.
    """
    elevation_levels = set(line['elevation'] for line in original_lines_data)
    new_lines_data = []

    for elevation in elevation_levels:
        
        # Filter lines by current elevation
        lines_at_elevation = [line for line in original_lines_data if line['elevation'] == elevation]
        if len(lines_at_elevation) < 2:
            continue
        
        # Convert to 2D lines for processing
        after_lines = spatial_convert_to_2d_lines(lines_at_elevation)
        before_lines = np.copy(after_lines)
        
        # Get directions
        directions = spatial_normalize(spatial_get_directions(after_lines))

        for i in range(len(after_lines)):
            p1, p2 = after_lines[i][:2], after_lines[i][2:]
            direction_i = directions[i]

            for j in range(len(after_lines)):
                if i != j:
                    q1, q2 = after_lines[j][:2], after_lines[j][2:]
                    direction_j = directions[j]

                    # Check if lines are not parallel
                    if not np.allclose(np.cross(direction_i, direction_j), 0):
                        intersection = spatial_line_intersection(p1, p2, q1, q2)
                        
                        if intersection is not None:
                            
                            # Distance threshold for intersection points located on the original line segment (line_i)
                            threshold_int_intersection = (lines_at_elevation[i]['width'] + lines_at_elevation[j]['width']) * width_threshold
                            # Distance threshold for intersection points NOT located on the original line segment (line_i)
                            threshold_ext_intersection = max(lines_at_elevation[i]['width'], lines_at_elevation[j]['width'])
                            
                            # Check if intersection is on the original line segment
                            if np.linalg.norm(p1 - intersection) + np.linalg.norm(p2 - intersection) - np.linalg.norm(p1 - p2) < identical_threshold:
                                
                                # Check distance from intersection to endpoints of the line_j segment
                                if np.linalg.norm(q1 - intersection) < threshold_int_intersection:
                                    spatial_extend_line_to_point(after_lines[j], intersection)
                                elif np.linalg.norm(q2 - intersection) < threshold_int_intersection:
                                    spatial_extend_line_to_point(after_lines[j], intersection)
                                
                                # Check distance from intersection to endpoints of the line_i segment
                                if np.linalg.norm(p1 - intersection) < threshold_int_intersection:
                                    spatial_extend_line_to_point(after_lines[i], intersection)
                                elif np.linalg.norm(p2 - intersection) < threshold_int_intersection:
                                    spatial_extend_line_to_point(after_lines[i], intersection)
                            
                            # When the intersection is not located on the original line segment
                            # But it's very close to one of the edges of the original line segment
                            else:

                                # line_i
                                close_to_p1 = np.linalg.norm(p1 - intersection) < threshold_ext_intersection
                                close_to_p2 = np.linalg.norm(p2 - intersection) < threshold_ext_intersection
                                # line_j
                                close_to_q1 = np.linalg.norm(q1 - intersection) < threshold_ext_intersection
                                close_to_q2 = np.linalg.norm(q2 - intersection) < threshold_ext_intersection

                                if (close_to_p1 or close_to_p2) and (close_to_q1 or close_to_q2):

                                    # if both two lines have one edge points close to the intersection point, Then find those two points.

                                    # line_i
                                    if close_to_p1:
                                        spatial_extend_line_to_point(after_lines[i], intersection)
                                    elif close_to_p2:
                                        spatial_extend_line_to_point(after_lines[i], intersection)
                                    
                                    # line_j
                                    if close_to_q1:
                                        spatial_extend_line_to_point(after_lines[j], intersection)
                                    elif close_to_q2:
                                        spatial_extend_line_to_point(after_lines[j], intersection)
                                
                                    # here todo. any repetitions?
                                    
                                # if np.linalg.norm(p1 - intersection) < threshold_ext_intersection:
                                #     spatial_extend_line_to_point(after_lines[i], intersection)
                                # elif np.linalg.norm(p2 - intersection) < threshold_ext_intersection:
                                #     spatial_extend_line_to_point(after_lines[i], intersection)

        if plot_adjustments:
            spatial_plot_lines_side_by_side(
                before_lines,
                after_lines,
                f"Adjustment at Elevation {elevation}",
                os.path.join(output_figure_folder, f'Adjustment_Wall_Connections_Elevation_{elevation}.png'))

        for k, line in enumerate(lines_at_elevation):
            new_line = line.copy()
            new_line['location'] = [[after_lines[k][0], after_lines[k][1], line['location'][0][2]],
                                    [after_lines[k][2], after_lines[k][3], line['location'][1][2]]]
            new_lines_data.append(new_line)

    return new_lines_data