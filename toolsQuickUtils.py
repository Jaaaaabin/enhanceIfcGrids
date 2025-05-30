import functools
import time
import os
import math
import json
import numpy as np
from collections import Counter
import networkx as nx

def ensure_directory_exists(path):
    """Ensures that a directory exists, if not, creates it."""

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"\033[1m\033[94m[Created directory:] \033[93m{path}")
        
    # else:
    #     print(f"Directory already exists: {path}")

def time_decorator(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\033[1m\033[94m[Execution Time]\033[0m \033[92m{func.__name__}\033[0m completed in \033[93m{end_time - start_time:.2f} seconds\033[0m")
        return result

    return wrapper

def flatten_and_merge_lists(*lists):

    def flatten_a_list(lst): # check if there's a problem..
        """
        Recursively flattens a list.
        """
        for item in lst:
            if isinstance(item, list):
                yield from flatten_a_list(item)
            else:
                yield item
    
    # Filter out empty lists and flatten non-empty lists
    non_empty_lists = filter(None, lists)  # This removes empty lists
    flattened_items = list(flatten_a_list(non_empty_lists))
    return flattened_items

def check_repeats_in_list(container):
    """
    Check if a container (list, nested list, or any nested structure) contains
    unique values, regardless of its dimensionality.
    """
    seen = set()

    def check_unique(element):
        if isinstance(element, list):
            return all(check_unique(item) for item in element)
        else:
            if element in seen:
                return True
            seen.add(element)
            return False

    return check_unique(container)

def load_thresholds_from_json(json_file):
    """
    Loads threshold values from a JSON file.
    Returns a dictionary of thresholds or an empty dictionary if the file is not found or is invalid.
    """
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading thresholds from {json_file}: {e}")
        return {}

def deep_merge_dictionaries(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dictionaries(merged[key], value)
        elif key in merged and isinstance(merged[key], int) and isinstance(value, int):
            merged[key] += value
        else:
            merged[key] = value
    return merged

def remove_duplicate_dicts(list_of_dicts):

    unique_dicts = []
    seen = set()
    for d in list_of_dicts:
        dict_str = json.dumps(d, sort_keys=True)
        if dict_str not in seen:
            seen.add(dict_str)
            unique_dicts.append(d)
    return unique_dicts

def enrich_dict_with_another(dict_main, dict_new, remove_duplicate=False):

    for key, value in dict_new.items():
        if key in dict_main:
            if isinstance(value, int):
                continue
            elif isinstance(value, list):
                dict_main[key].extend(value)
                if remove_duplicate:
                    dict_main[key] = remove_duplicate_dicts(dict_main[key])
        else:
            dict_main[key] = value.copy()

    return dict_main

def remove_duplicate_points(points):
    
    def is_close(point1, point2, threshold=0.0001):
        """
        Check if two points are within a certain threshold.
        """
        return abs(point1.x - point2.x) <= threshold and abs(point1.y - point2.y) <= threshold

    unique_points = []
    for pt in points:
        if all(not is_close(pt, unique_pt) for unique_pt in unique_points):
            unique_points.append(pt)

    return unique_points

def find_most_common_value(values):

    value_counts = Counter(values)
    most_common_value, most_common_count = value_counts.most_common(1)[0]

    return most_common_value, most_common_count

def get_coords(point, round_demi=4):

    # Function to get the x and y values from points
    if isinstance(point, list) or isinstance(point, tuple):
        x, y = point[0], point[1]
    elif (hasattr(point, 'x') and hasattr(point, 'y')):
        x, y = point.x, point.y
    else:
        raise ValueError("Point objects must have 'x' and 'y' attributes")
    return round(x, round_demi), round(y, round_demi)

def get_line_slope_by_points(point1, point2):

    # Extract x and y values for both points
    x1, y1 = get_coords(point1)
    x2, y2 = get_coords(point2)
    
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) > 0.00001:
        slope = dy / dx
    else:
        slope = float('inf')  # Vertical line

    return slope

def slope_to_orientation_0_180(slope):

    if slope == float('inf'):
        degree = 90.0
    elif slope == 0:
        degree = 0.0
    else:
        degree = math.degrees(math.atan(slope))
    
    # Normalize the degree to be within 0 to 180
    if degree < 0:
        degree += 180
    elif degree > 180:
        degree -= 180
        
    return degree

def point_to_line_distance(point, line, tol_distance=0.001):

    def point_get_coords(point):
        return point[0], point[1], point[2]
    
    # line Ax+ By + C = 0 (x1,y1), (x2,y2), point (x0, y0).
    x1, y1, _ = point_get_coords(line[0])
    x2, y2, _ = point_get_coords(line[1])
    x0, y0, _ = point_get_coords(point)
    
    A = y1 - y2
    B = -(x1 - x2)
    C = x1 * y2 - x2 * y1
    distance = abs(A * x0 + B * y0 + C) / (A**2 + B**2)**0.5

    if distance < tol_distance:
        relative_distance = 0.
    else:
        # Calculate the direction by projecting the point onto the line segment
        dot_product = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
        length_squared = (x2 - x1)**2 + (y2 - y1)**2
        
        # Calculate the projection parameter t
        t = dot_product / length_squared
        
        if t < 0:
            closest_point = (x1, y1)
        elif t > 1:
            closest_point = (x2, y2)
        else:
            closest_point = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        
        direction = 1.0 if (x0 - closest_point[0]) * (x2 - x1) + (y0 - closest_point[1]) * (y2 - y1) > 0 else -1.0
        relative_distance = distance * direction

    return relative_distance

def perpendicular_distance(point1, point2):

    # Extract x and y values for both points
    x1, y1 = get_coords(point1)
    x2, y2 = get_coords(point2)
    
    A = y2 - y1
    B = -(x2 - x1)
    C = x1 * y2 - x2 * y1
    
    return C / math.sqrt(A**2 + B**2)
    
def is_sloped_point_on_lineby2points(point, line_point1, line_point2, slope, threshold):
    """
    Checks if a point is within a specified distance (threshold) from a line
    defined by two points (line_point1 and line_point2) and a slope.
    """
    return abs((point.y - line_point1.y) - slope * (point.x - line_point1.x)) <= threshold and \
            abs((point.y - line_point2.y) - slope * (point.x - line_point2.x)) <= threshold

# def line_points_are_same(p1, p2):

#     x1, y1 = p1.x, p1.y
#     x2, y2 = p2.x, p2.y

#     if abs(x1-x2) <=0.00001 and abs(y1-y2) <=0.00001:
#         return True
#     else:
#         return False
    
def line_points_are_collinear(p1, p2, p3, t):

    # are three points are collinear or not.
    
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y

    determinant = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) # "cross product" of 2D vectors
    
    if determinant <= t:
        return True
    else:
        return False

def line_points_are_close(p1, p2, p3, offset):
    # p1, p2 are from one line.
    # p3 is another point.

    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y

    # Coefficients of the line Ax + By + C = 0
    # A B C are from one line.
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # Distance from point3 to the line
    if A ==0 and B== 0:
        distance = 0
    else:
        distance = abs(A*x3 + B*y3 + C) / np.sqrt(A**2 + B**2)

    return distance<=offset

def close_parallel_lines(p1, p2, p3, p4, offset, t=0.00001):
    
    # line1: p1, p2
    # line2: p3, p4
    
    all_close = (line_points_are_close(p1, p2, p3, offset) and
                    line_points_are_close(p1, p2, p4, offset) and
                    line_points_are_close(p3, p4, p1, offset) and
                    line_points_are_close(p3, p4, p2, offset))
    
    # Check if all points are collinear
    all_collinear = (line_points_are_collinear(p1, p2, p3, t) and
                     line_points_are_collinear(p1, p2, p4, t) and
                     line_points_are_collinear(p3, p4, p1, t) and
                     line_points_are_collinear(p3, p4, p2, t))
    
    return all_close, all_collinear

def is_close_to_known_orientations(new_orientation, known_orientations, threshold=0.001):
    """
    specific constraints that column-based gridlines must be located following the main directions of the building.
    """
    if new_orientation in known_orientations:
        return True
    else:
        return any(abs(new_orientation - s) <= threshold for s in known_orientations)

def are_lines_cross(line1_bounds, line2_bounds):
    """
    Determines the intersection point of two line segments, if any.
    Returns the point of intersection or None if there is no intersection.
    """
    x1, y1, x2, y2 = line1_bounds
    x3, y3, x4, y4 = line2_bounds

    # Calculate denominators
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    num_x = (x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)
    num_y = (x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)
    
    # If den == 0, lines are parallel (including possibly overlapping)
    if den == 0:
        return None

    # Calculate the intersection point
    px = num_x / den
    py = num_y / den
    
    # Check if the intersection point is within both line segments
    if ((px - min(x1, x2)) * (px - max(x1, x2)) <= 0 and
        (py - min(y1, y2)) * (py - max(y1, y2)) <= 0 and
        (px - min(x3, x4)) * (px - max(x3, x4)) <= 0 and
        (py - min(y3, y4)) * (py - max(y3, y4)) <= 0):
        return (px, py)
    
    return None

def distance_between_points(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def point_on_line_segment(point, line, threshold_segment_percent):

    x1, y1, x2, y2 = line
    
    dist_to_start = distance_between_points(point, (x1, y1))
    dist_to_end = distance_between_points(point, (x2, y2))

    line_length = distance_between_points((x1, y1), (x2, y2))
    edge_length = (threshold_segment_percent / 100.0) * line_length
    
    if dist_to_start <= edge_length or dist_to_end <= edge_length:
        return True
    else:
        return False

def calculate_line_crosses(main_line, second_line, ignore_cross_edge=False, cross_threshold_percent=5):
    
    intersection_point = are_lines_cross(main_line.bounds, second_line.bounds)
    
    if intersection_point:
        if ignore_cross_edge:
            cross_on_edge = point_on_line_segment(intersection_point, second_line.bounds, cross_threshold_percent)
            if not cross_on_edge:
                return True
            else:
                return False
        else:
            return True
        
def get_rectangle_corners(points):

    # read them into np.array
    points = np.array(points)
    
    # remove duplicated points first.
    points = np.unique(points, axis=0)

    min_x, min_y, min_z = points.min(axis=0)
    max_x, max_y, max_z = points.max(axis=0)
    mid_point = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
    
    min_z_points = points[points[:, 2] == min_z]
    max_z_points = points[points[:, 2] == max_z]
    relevant_points = np.vstack((min_z_points, max_z_points))
    
    distances = np.sqrt(((points - mid_point) ** 2).sum(axis=1))
    
    farthest_points_indices = np.argsort(-distances)[:4]  # Get indices of the 4 largest distances
    farthest_points = points[farthest_points_indices]
    farthest_distances = distances[farthest_points_indices]
    
    return farthest_points

def a_is_subtuple_of_b(a, b):
   
    return set(a).issubset(set(b))


def find_closed_loops(segments, tolerance=1e-4):
    G = nx.Graph()
    
    def add_point(point):
        # Round the coordinates to avoid floating point issues
        return tuple(round(coord, 8) for coord in point)
    
    for segment in segments:
        start, end = map(add_point, segment)
        
        # Add both directions of the segment
        G.add_edge(start, end)
        G.add_edge(end, start)
    
    # Find all simple cycles in the graph
    cycles = list(nx.simple_cycles(G))
    
    # Filter and clean up the cycles
    cleaned_cycles = []
    for cycle in cycles:
        # Remove duplicate points (can happen due to bidirectional edges)
        cleaned_cycle = []
        for point in cycle:
            if not cleaned_cycle or not np.allclose(point, cleaned_cycle[-1], atol=tolerance):
                cleaned_cycle.append(point)
        
        # Ensure the cycle is closed
        if not np.allclose(cleaned_cycle[0], cleaned_cycle[-1], atol=tolerance):
            cleaned_cycle.append(cleaned_cycle[0])
        
        # Only keep cycles with at least 3 unique points
        if len(cleaned_cycle) > 3:
            cleaned_cycles.append(cleaned_cycle)
    
    return cleaned_cycles