import functools
import time
import json

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

def get_line_slope_by_points(point1, point2):
    
    # works for x and y, doesn't matter is z exits or not.
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    if abs(dx) > 0.0001:
        slope = dy / dx
    else:
        slope = float('inf')  # Verticalline
    return slope

def is_sloped_point_on_lineby2points(point, line_point1, line_point2, slope, threshold):
    """
    Checks if a point is within a specified distance (threshold) from a line
    defined by two points (line_point1 and line_point2) and a slope.
    """
    return abs((point.y - line_point1.y) - slope * (point.x - line_point1.x)) <= threshold and \
            abs((point.y - line_point2.y) - slope * (point.x - line_point2.x)) <= threshold

def is_close_to_known_slopes(new_slope, known_slopes, threshold=0.0001):
    """
    specific constraints that column-based gridlines must be located following the main directions of the building.
    """
    if new_slope in known_slopes:
        return True
    else:
        return any(abs(new_slope - s) <= threshold for s in known_slopes)

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

def calculate_line_crosses(main_line, second_line, ignore_cross_edge=False, threshold_percent=5):
    
    intersection_point = are_lines_cross(main_line.bounds, second_line.bounds)
    
    if intersection_point:
        if ignore_cross_edge:
            cross_on_edge = point_on_line_segment(intersection_point, second_line.bounds, threshold_percent)
            if not cross_on_edge:
                return True
            else:
                return False
        else:
            return True
    
    