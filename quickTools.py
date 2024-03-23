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

def flatten_a_list(lst):
    """
    Recursively flattens a list.
    """
    for item in lst:
        if isinstance(item, list):
            yield from flatten_a_list(item)
        else:
            yield item

def flatten_and_merge_lists(*lists):
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