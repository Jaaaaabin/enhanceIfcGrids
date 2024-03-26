import ifcopenshell
import os
import copy
import json
import math
import itertools
from tqdm import tqdm
from collections import Counter
import numpy as np
import shapely
from shapely.geometry import Point, LineString, MultiPoint

import bokeh.plotting

from wallExtractor import WallWidthExtractor
from quickTools import is_sloped_point_on_lineby2points, get_line_slope_by_points, is_close_to_known_slopes, remove_duplicate_points
from quickTools import time_decorator, check_repeats_in_list, flatten_and_merge_lists, enrich_dict_with_another, calculate_line_intersection

#===================================================================================================
#Grids ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

class GridGenerator:

    def __init__(self, figure_path, json_columns, json_walls):

        # initial data
        self.info_columns = []
        self.info_walls = []
        self.main_directions = {}
        self.main_storeys_from_columns = {}
        self.main_storeys_from_walls = {}
        self.main_storeys = {}

        # = = = = = = = = = = = = generation parameters = = = = = = = = = 
        # vertical threshold.
        self.t_z_storey_raise = 0.8         # semi-static / static
        self.t_z_crossing_columns = 0.1     # semi-static / static

        # columns related
        self.t_c_num = 6
        self.t_c_dist = 0.0001

        # walls related
        self.t_w_num = 2
        self.t_w_dist = 0.0001
        self.t_w_st_accumuled_length = 20
        self.t_w_nt_accumuled_length = 20

        # merging related
        # self.t_self_dist = 0.2
        # self.t_cross_dist = 0.5
        self.border_x = None
        self.border_y = None

        self.out_fig_path = figure_path
        self.read_infos(json_columns, json_walls)
        self.init_visualization_settings()

        print ("=============GridGenerator=============")
        print (self.out_fig_path)

    def read_infos(self, json_columns, json_walls):
        
        def read_json_file(file_path):
            try:
                with open(file_path, 'r') as file:
                    return json.load(file)
            except FileNotFoundError:
                print(f"File {file_path} not found.")
                return None
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}.")
                return None

        self.info_columns = read_json_file(json_columns) or []
        self.info_walls = read_json_file(json_walls) or []

    def init_visualization_settings(self):
        """
        Initializes visualization settings for various building components.
        """

        self.visualization_settings = {
            # points.
            'column_points_struc':{
                'legend_label':'Column Points',
                'color': "darkgreen",
                'size':8,
                'alpha':1,
            },

            # lines.
            'wall_lines_struc':{
                'legend_label':'Structural Wall Lines',
                'color': "black",
                'line_dash':'solid',
                'line_width':3,
                'alpha':1,
            },
            'wall_lines_nonst':{
                'legend_label':'Non-structural Wall Lines',
                'color': "dimgray",
                'line_dash':'solid',
                'line_width':3,
                'alpha':1,
            },

            # grid lines.
            'grids_column': {
                'legend_label':'Grids from IfcColumn',
                'color': "tomato",
                'line_dash':'dotted',
                'line_width':2,
                'alpha':0.85,
            },
            'grids_st_wall': {
                'legend_label': 'Grids from structural IfcWall',
                'color': "orange",
                'line_dash':'dashed',
                'line_width':2,
                'alpha':0.60,
            },
            'grids_ns_wall': {
                'legend_label': 'Grids from non-structural IfcWall',
                'color': "navy",
                'line_dash':'dashed',
                'line_width':2,
                'alpha':0.60,
            },
            'grids_st_merged': {
                'legend_label': 'Structural Grids',
                'color': "orange",
                'line_dash':'dotdash',
                'line_width':3,
                'alpha':0.85,
            },
            'grids_ns_merged': {
                'legend_label':'Non-structural Grids',
                'color': "navy",
                'line_dash':'dashed',
                'line_width':3,
                'alpha':0.85,
            },}

    # column-related
    def points2grids(self, component_pts, element_ids):
        """
        Return: Points from collinear pairs of points.
        """

        # todo.
        
        grid_elements = []
        ids_component_pts_per_grid = []

        # Iterate through each pair of points once
        for i, (point1, id1) in tqdm(enumerate(zip(component_pts[:-1],element_ids[:-1])), total=len(component_pts)-1, desc="Points -> Grid:"):
            for j, (point2,id2) in enumerate(zip(component_pts[i + 1:],element_ids[i + 1:]), start=i + 1):
                
                aligned_points = [point1, point2]
                id_components = {id1, id2}  # Use a set to avoid duplicate indices
                slope = get_line_slope_by_points(point1, point2)
                
                # for columns, ignore those pairs not located on the main directions.
                if not is_close_to_known_slopes(slope, self.main_directions):
                    continue
                else:
                    for k, (point,id3) in enumerate(zip(component_pts,element_ids)):
                        if id3 not in id_components:  # Skip points already considered in the line
                            if slope == float('inf'):  # Vertical line check
                                if abs(point.x - point1.x) <= self.t_c_dist:
                                    aligned_points.append(point)
                                    id_components.add(id3)
                            else:
                                # Use point-slope form of line equation to check alignment
                                if abs((point.y - point1.y) - slope * (point.x - point1.x)) <= self.t_c_dist:
                                    aligned_points.append(point)
                                    id_components.add(id3)

                    # Check for minimum number of points and uniqueness before adding to the grid
                    if len(aligned_points) >= self.t_c_num:
                        id_tuple = tuple(sorted(id_components))  # Convert to tuple for hashability
                        if id_tuple not in ids_component_pts_per_grid:
                            grid_elements.append(aligned_points)
                            ids_component_pts_per_grid.append(id_tuple)

        return grid_elements, ids_component_pts_per_grid
    
    # wall-related
    def lines2grids(self, component_lines, element_ids):
        """
        Return: Points from collinear lines.
        """

        grid_elements = []
        ids_component_pts_per_grid = []

        # Pre-calculate slopes for all lines
        line_slopes = [get_line_slope_by_points(Point(ln.bounds[:2]), Point(ln.bounds[2:])) for ln in component_lines]
        line_lengths = [ln.length for ln in component_lines]

        for i, (ln, slope, length, id1) in tqdm(enumerate(zip(component_lines[:-1], line_slopes[:-1], line_lengths[:-1], element_ids[:-1])), total=len(component_lines)-1, desc="Lines -> Grid:"):
            
            point1, point2 = Point(ln.bounds[:2]), Point(ln.bounds[2:])
            aligned_points = [[point1, point2]]
            accumulated_length = length
            id_components = {id1}  # Use a set for unique IDs

            # Sub-case of non-vertical lines.
            if slope != float('inf'):

                for j, (new_ln, new_slope, new_length, id2) in enumerate(zip(component_lines[i + 1:], line_slopes[i + 1:], line_lengths[i + 1:], element_ids[i + 1:]), start=i + 1):
                    point3, point4 = Point(new_ln.bounds[:2]), Point(new_ln.bounds[2:])
                    
                    # Check slope similarity and alignment within tolerance
                    if is_sloped_point_on_lineby2points(point3, point1, point2, slope, self.t_w_dist) and \
                        is_sloped_point_on_lineby2points(point4, point1, point2, slope, self.t_w_dist):
                        aligned_points.append([point3,point4])
                        accumulated_length += new_length
                        id_components.add(id2)

            # Sub-case of vertical lines.
            elif slope == float('inf'):

                for j, (new_ln, new_slope, new_length, id2) in enumerate(zip(component_lines[i + 1:], line_slopes[i + 1:], line_lengths[i + 1:], element_ids[i + 1:]), start=i + 1):
                    point3, point4 = Point(new_ln.bounds[:2]), Point(new_ln.bounds[2:])
                    
                    if new_slope == float('inf'):
                        if abs(point3.x-point1.x) <= self.t_w_dist and abs(point3.x-point2.x) <= self.t_w_dist and \
                        abs(point4.x-point1.x) <= self.t_w_dist and abs(point4.x-point2.x) <= self.t_w_dist:
                            aligned_points.append([point3,point4])
                            accumulated_length += new_length
                        id_components.add(id2)
            else:
                raise ValueError("No such subcases for the slope of a line.")
            
            id_components = sorted(id_components)

            # to get embedded in the secondary-loop.  
            # Add unique grid_elements to the potential grid_elements list
            if len(aligned_points) >= self.t_w_num and accumulated_length >= self.t_w_st_accumuled_length:
                id_tuple = tuple(id_components)  # Convert to tuple for hashability
                if id_tuple not in ids_component_pts_per_grid:
                    if ids_component_pts_per_grid and any([
                        set(id_tuple).issubset(set(existing_ids_component)) for existing_ids_component in ids_component_pts_per_grid]):
                        continue
                    else:
                        grid_elements.append(aligned_points)
                        ids_component_pts_per_grid.append(list(set(id_components)))
                        
        grid_elements = [[e for element in elements for e in element] for elements in grid_elements]
        grid_elements = [remove_duplicate_points(elements) for elements in grid_elements]

        return grid_elements, ids_component_pts_per_grid
    
    def generate_gridslines_from_points(self, grid_elements):
        """
        The logic is to get the grid line from all the points (either column points or points of wall lines.)
        """
        border_x = 0
        border_y = 0
        border_x = self.border_x
        border_y = self.border_y

        if len(border_x) != 2 or len(border_y) != 2: 
            raise ValueError("border_x and border_y must each have two elements.")

        grids = []
        for elements in grid_elements:

            slopes = [get_line_slope_by_points(p1, p2) for (p1, p2) in itertools.combinations(elements, 2)]
            
            if float('inf') in slopes:
                
                elements_x = [elem.x for elem in elements]
                mean_x = np.mean(elements_x)
                p_start = [mean_x, border_y[0]] 
                p_end = [mean_x, border_y[1]] 

            else:
                mean_slope = np.mean(slopes)
                
                mean_res = np.mean([pt.y - mean_slope * pt.x for pt in elements if mean_slope != float('inf')])
                p_start, p_end = [0, 0], [0, 0]

                # Calculating intersections
                border_x_p_start = [border_x[0], border_x[0] * mean_slope + mean_res] 
                border_x_p_end = [border_x[1], border_x[1] * mean_slope + mean_res] 
                if mean_slope != 0:  # Avoid division by zero
                    border_y_p_start = [(border_y[0] - mean_res) / mean_slope, border_y[0]] 
                    border_y_p_end = [(border_y[1] - mean_res) / mean_slope, border_y[1]] 
                else:  # For horizontal lines, choose points directly at y borders
                    border_y_p_start = [border_x[0], border_y[0]] 
                    border_y_p_end = [border_x[1], border_y[0]] 
                
                # Determining valid start and end points
                p_start, p_end = border_x_p_start, border_x_p_end  # Default to x-borders
                if not (border_y[0] <= border_x_p_start[1] <= border_y[1]): 
                    p_start = min(border_y_p_start, border_y_p_end, key=lambda p: p[0])
                if not (border_y[0] <= border_x_p_end[1] <= border_y[1]): 
                    p_end = max(border_y_p_start, border_y_p_end, key=lambda p: p[0])
            
            grids.append([p_start, p_end])
        
        grid_linestrings = [LineString(grid) for grid in grids]
        
        return grid_linestrings

    def get_gridline_points_and_alignments(self):

        for st_key, st_value in self.main_storeys.items():

            if st_value.get('column_points_struc', None) is not None:
                component_pts, element_ids = st_value['column_points_struc'], st_value['column_points_struc_ids']
                grid_elements_columns_struc, grid_elements_columns_struc_ids = self.points2grids(component_pts, element_ids)
                self.main_storeys[st_key].update({
                    'grid_elements_columns_struc': grid_elements_columns_struc,
                    'grid_elements_columns_struc_ids': grid_elements_columns_struc_ids,
                    })
                
            else:
                self.main_storeys[st_key].update({'grid_elements_columns_struc': []})
            
            if st_value.get('wall_lines_struc', None) is not None:
                component_lines, element_ids = st_value['wall_lines_struc'], st_value['wall_lines_struc_ids']
                grid_elements_walls_struc, grid_elements_walls_struc_ids = self.lines2grids(component_lines, element_ids)
                self.main_storeys[st_key].update({
                    'grid_elements_walls_struc': grid_elements_walls_struc,
                    'grid_elements_walls_struc_ids': grid_elements_walls_struc_ids,
                    })
            else:
                self.main_storeys[st_key].update({'grid_elements_walls_struc': []})
            
            if st_value.get('wall_lines_nonst', None) is not None:
                component_lines, element_ids = st_value['wall_lines_nonst'], st_value['wall_lines_nonst_ids']
                grid_elements_walls_nonst, grid_elements_walls_nonst_ids = self.lines2grids(component_lines, element_ids)
                self.main_storeys[st_key].update({
                    'grid_elements_walls_nonst': grid_elements_walls_nonst,
                    'grid_elements_walls_nonst_ids':grid_elements_walls_nonst_ids,
                    })
            else:
                self.main_storeys[st_key].update({'grid_elements_walls_nonst': []})

    #
    def get_main_storeys_from_columns(self, num_columns=4):
        """
        four columns for a room, making a minimum support system.
        """

        if not self.info_columns:
            print("No column information available.")
            return

        columns_by_elevation = {}
        for w in self.info_columns:
            try:
                elevation = w["elevation"]
            except KeyError:
                continue  # Skip the current iteration if elevation key is missing
            columns_by_elevation.setdefault(elevation, []).append(w)

        self.main_storeys_from_columns = {elevation: {"count": len(columns), "columns": columns}
                                          for elevation, columns in columns_by_elevation.items() if len(columns) > num_columns}

    def get_main_storeys_from_walls(self, num_walls=4):
        """
        four walls for a room, making a minimum space or storey.
        """

        if not self.info_walls:
            print("No wall information available.")
            return

        walls_by_elevation = {}
        for w in self.info_walls:
            try:
                elevation = w["elevation"]
            except KeyError:
                continue  # Skip if elevation key is missing
            walls_by_elevation.setdefault(elevation, []).append(w)

        self.main_storeys_from_walls = {elevation: {"count": len(walls), "walls": walls}
                                        for elevation, walls in walls_by_elevation.items() if len(walls) > num_walls}

    def get_main_storeys_init(self):

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

        self.get_main_storeys_from_columns()
        self.get_main_storeys_from_walls()
        self.main_storeys = deep_merge_dictionaries(self.main_storeys_from_columns, self.main_storeys_from_walls)

    def get_main_directions(self, num_directions):
            
        def degree2slope(degree):
            t_direction_degree = 0.0001
            slope = float('inf') if abs(degree-90.0)<t_direction_degree else math.radians(degree) # static threshold.
            return slope
        
        wall_orientations = [w['orientation'] for w in self.info_walls if 'orientation' in w]
        wall_orientations = [(v-180) if v>=180 else v for v in wall_orientations]
        main_directions = Counter(wall_orientations)
        main_directions = main_directions.most_common(num_directions)

        self.main_directions = [main_direct[0] for main_direct in main_directions]
        self.main_directions = [degree2slope(main_direct) for main_direct in self.main_directions]
     
    def identify_columns_cross_storeys(self):
        
        def filter_by_location_z(dicts, z):    
            filtered_dicts = []
            for d in dicts:
                try:
                    if (z -d['location'][0][-1]) >= self.t_z_crossing_columns and (d['location'][1][-1]-z) >= self.t_z_crossing_columns:
                        filtered_dicts.append(d)
                except KeyError:
                    print("Warning: One of the dictionaries is missing the 'location' key.")
                except IndexError:
                    print("Warning: 'location' data is improperly structured.")
                except TypeError:
                    print("Warning: Incompatible type encountered in 'location' data.")
            return filtered_dicts

        for st_key, st_value in self.main_storeys.items():
            
            if st_value.get('columns', None) is not None:
                pass
            else:
                crossed_columns = filter_by_location_z(self.info_columns, st_key)
                self.main_storeys[st_key].update({'columns': crossed_columns})
    
    def enrich_main_storeys_info_raised_area(self):
        """
        dynamic storey merging process.
        """

        storeys_with_raised_area = [sorted([st1,st2]) for (st1,st2) in itertools.combinations(self.main_storeys.keys(), 2) if abs(st1-st2)<=self.t_z_storey_raise]
        
        if storeys_with_raised_area:
            if check_repeats_in_list(storeys_with_raised_area):
                raise ValueError("Inter-dependent raised areas. when merging different storeys.")
            else:
                for (st_main,st_raise) in storeys_with_raised_area:
                    dict_main = self.main_storeys[st_main]
                    dict_new = self.main_storeys[st_raise]
                    self.main_storeys[st_main] = enrich_dict_with_another(dict_main, dict_new, remove_duplicate=True)
                    self.main_storeys.pop(st_raise, None)
        else:
            print("Non raised area according to the given 'self.t_z_storey_raise'.")
        
    def enrich_main_storeys_info_locations(self):

        for st_key, st_value in self.main_storeys.items():

            st_info_columns = st_value.get('columns', None) # todo. check what is the return when it doesn't exit
            st_info_walls = st_value.get('walls', None) # todo. check what is the return when it doesn't exit

            column_points_struc, wall_lines_struc, wall_lines_nonst = None, None, None

            #columns.
            if st_info_columns is not None and st_info_columns:

                columns_location_id_pairs = [(c['location'], c['id']) for c in st_info_columns]
                s_column_locations, column_points_struc_ids = zip(*columns_location_id_pairs)
                s_column_locations, column_points_struc_ids = list(s_column_locations), list(column_points_struc_ids)
                column_locations_struc = copy.deepcopy(s_column_locations)
                [column_loc.pop() for column_loc in column_locations_struc]
                column_locations_struc = [item for sublist in column_locations_struc for item in sublist]
                column_points_struc = [Point(column_loc) for column_loc in column_locations_struc]
                
                if column_points_struc is not None:
                    self.main_storeys[st_key].update({
                        'column_points_struc': column_points_struc,
                        'column_points_struc_ids': column_points_struc_ids,
                        })
                
            #walls.
            if st_info_walls is not None and st_info_walls:

                wall_s_location_id_pairs = [(w['location'], w['id']) for w in st_info_walls if w['loadbearing']]
                wall_ns_location_id_pairs = [(w['location'], w['id']) for w in st_info_walls if not w['loadbearing']]

                if wall_s_location_id_pairs:
                    s_wall_locations, wall_lines_struc_ids = zip(*wall_s_location_id_pairs)
                    s_wall_locations, wall_lines_struc_ids = list(s_wall_locations), list(wall_lines_struc_ids)
                    wall_locations_struc = copy.deepcopy(s_wall_locations)
                    [p.pop() for wall_loc in wall_locations_struc for p in wall_loc]
                    wall_lines_struc = [LineString(wall_location) for wall_location in wall_locations_struc]
                    
                    if wall_lines_struc is not None:
                        self.main_storeys[st_key].update({
                            'wall_lines_struc': wall_lines_struc,
                            'wall_lines_struc_ids': wall_lines_struc_ids,
                            })
                        
                if wall_ns_location_id_pairs:
                    ns_wall_locations, wall_lines_nonst_ids = zip(*wall_ns_location_id_pairs)
                    ns_wall_locations, wall_lines_nonst_ids = list(ns_wall_locations), list(wall_lines_nonst_ids)
                    wall_locations_nonst = copy.deepcopy(ns_wall_locations)
                    [p.pop() for wall_loc in wall_locations_nonst for p in wall_loc]
                    wall_lines_nonst = [LineString(wall_location) for wall_location in wall_locations_nonst]
        
                    if wall_lines_nonst is not None:
                        self.main_storeys[st_key].update({
                            'wall_lines_nonst': wall_lines_nonst,
                            'wall_lines_nonst_ids': wall_lines_nonst_ids,
                            })
    
    def update_display_borders(self, all_references_x, all_references_y, pad_x_y):

        def calculate_and_update_border(current_border, refs):
            if not refs:  # If no references, return current border or initialize it
                return current_border if current_border else [0, 0]
            # Calculate new border values
            new_min, new_max = min(refs) - pad_x_y, max(refs) + pad_x_y
            # If current borders are not set, use new values
            if not current_border:
                return [new_min, new_max]
            # Update borders to encompass both old and new values
            return [min(current_border[0], new_min), max(current_border[1], new_max)]

        # Update or initialize self.border_x and self.border_y
        self.border_x = calculate_and_update_border(self.border_x, all_references_x)
        self.border_y = calculate_and_update_border(self.border_y, all_references_y)

    def generate_gridlines_per_storey(self, storey):        
        
        all_grid_reference_points = flatten_and_merge_lists(
            self.main_storeys[storey]['grid_elements_columns_struc'],
            self.main_storeys[storey]['grid_elements_walls_struc'],
            self.main_storeys[storey]['grid_elements_walls_nonst'])
        
        pad_x_y = 2
        all_references_x, all_references_y = [pt.x for pt in all_grid_reference_points], [pt.y for pt in all_grid_reference_points]
        self.update_display_borders(all_references_x, all_references_y, pad_x_y) # necessary for the current calculation of grid lines.

        gridlines_elements_mapping = {
            'grids_column': 'grid_elements_columns_struc',
            'grids_st_wall': 'grid_elements_walls_struc',
            'grids_ns_wall': 'grid_elements_walls_nonst'}
        
        for gridline_key, element_key in gridlines_elements_mapping.items():
            grid_elements = self.main_storeys[storey].get(element_key)
            if grid_elements:
                self.main_storeys[storey][gridline_key] = self.generate_gridslines_from_points(grid_elements)
            else:
                self.main_storeys[storey][gridline_key] = []
    
    def display_per_storey(self, storey):

        # plot_name = f"\[Floor \, Plan \, of \, {storey.Name} \, (T_{{c,dist}}={t_c_dist}, \, T_{{c,num}}={t_c_num}, \, T_{{w,dist}}={t_w_dist}, \, T_{{w,num}}={t_w_num}) - Initial \]"
        # fig_save_name = f"Initial_{storey.Name}_t_c_dist_{t_c_dist}_t_c_num_{t_c_num}_t_w_dist_{t_w_dist}_t_w_num_{t_w_num}"
        plot_name = f"Floor Plan Elevation {str(round(storey, 4))} - Initial"
        fig_save_name = f"Floor_Plan_Elevation_{str(round(storey,4))}_Initial"

        fig = bokeh.plotting.figure(
            title=plot_name,
            title_location='above',
            x_axis_label='x',
            y_axis_label='y',
            width=800,
            height=800,
            match_aspect=True)
        fig.title.text_font_size = '11pt'
        fig.xgrid.visible = False
        fig.ygrid.visible = False

        # Mapping of main_storeys keys to their corresponding visualization settings.
        plot_configurations = [
            ('column_points_struc', 'square', None),
            ('wall_lines_struc', 'line', 'coords'),
            ('wall_lines_nonst', 'line', 'coords'),
            ('grids_column', 'line', 'coords'),
            ('grids_st_wall', 'line', 'coords'),
            ('grids_ns_wall', 'line', 'coords'),
        ]

        for config in plot_configurations:
            data_key, plot_type, attr = config
            element_data = self.main_storeys[storey].get(data_key)
            if element_data:
                g_plot = self.visualization_settings[data_key]
                for element in element_data:
                    x, y = (element.x, element.y) if not attr else getattr(element, attr).xy
                    
                    if plot_type == 'square':
                        fig.square(x, y, legend_label=g_plot['legend_label'], size=g_plot['size'], 
                                color=g_plot['color'], alpha=g_plot['alpha'])
                    elif plot_type == 'line':
                        fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], 
                                line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])

        # Save the figure.
        bokeh.plotting.output_file(filename=os.path.join(self.out_fig_path, fig_save_name + ".html"), title=fig_save_name)
        bokeh.plotting.save(fig)

    def prepare_wall_lengths(self):

        total_length = 0
        for item in self.info_walls:
            if "length" in item:
                total_length += item["length"]
        self.total_wall_lengths = total_length
        self.info_wall_length_by_id = {d['id']: d['length'] for d in self.info_walls if 'id' in d and 'length' in d}

    @time_decorator
    def get_main_storeys_and_directions(self, num_directions=2):
        """
        "static" initialization processes.
        """
        
        self.get_main_storeys_init() # static
        self.get_main_directions(num_directions) # static
        self.identify_columns_cross_storeys() # static          
    
    @time_decorator
    def enrich_main_storeys(self):
        """
        "semi-static" enrichment processes.
        """

        self.enrich_main_storeys_info_raised_area()
        self.enrich_main_storeys_info_locations()
    
    @time_decorator
    def calculate_cross_lost(self):
        
        crossed_wall_lengths = 0.0

        # find all unbound walls and their LINESTRING : st and nt
        for st_key, st_value in self.main_storeys.items():

            walls_bound_st = st_value['grid_elements_walls_struc_ids'] if st_value['grid_elements_walls_struc_ids'] else []
            if walls_bound_st:
                walls_bound_st = [item for sublist in walls_bound_st for item in sublist]
            walls_bound_ns = st_value['grid_elements_walls_nonst_ids'] if st_value['grid_elements_walls_nonst_ids'] else []
            if walls_bound_ns:
                walls_bound_ns = [item for sublist in walls_bound_ns for item in sublist]
            
            wall_not_bound_st, wall_not_bound_ns = {}, {}
            if st_value['wall_lines_struc_ids'] and st_value['wall_lines_struc']:
                for (wall_id, wall_line) in zip(st_value['wall_lines_struc_ids'], st_value['wall_lines_struc']):
                    if wall_id not in walls_bound_st:
                        wall_not_bound_st.update({
                            wall_id: wall_line
                            })
            if st_value['wall_lines_nonst_ids'] and st_value['wall_lines_nonst']:
                for (wall_id, wall_line) in zip(st_value['wall_lines_nonst_ids'], st_value['wall_lines_nonst']):
                    if wall_id not in walls_bound_st:
                        wall_not_bound_ns.update({
                            wall_id: wall_line
                            })
            all_grid_lines = st_value['grids_column'] + st_value['grids_st_wall'] + st_value['grids_ns_wall']

            # todo.
            # differentiate between structural and non-structural walls that are crossed by generated grid lines.
            crossed_walls = []
            for g_line in all_grid_lines:
                for w_st_id, w_st_line in wall_not_bound_ns.items():
                    if calculate_line_intersection(g_line,w_st_line,ignore_edge_intersecton=True):
                        crossed_walls.append(w_st_id)
                        crossed_wall_lengths += self.info_wall_length_by_id[w_st_id]

            # use percent vlaues as the final loss function.
            print ("stop")

    @time_decorator
    def create_grids(self):
        """
        "dynamic" generation processes.
        
        fundamental parameters:

        parameters in points2grids:
            self.t_c_num = 6
            self.t_c_dist = 0.0001
        
        parameters in lines2grids:
            self.t_w_num = 2
            self.t_w_dist = 0.0001
            self.t_w_st_accumuled_length = 20

        parameters in gridalignment:

        cost function:

        """

        self.get_gridline_points_and_alignments()
        [self.generate_gridlines_per_storey(st) for st in self.main_storeys.keys()]
        self.calculate_cross_lost()

    @time_decorator
    def display_grids(self):
        """
        plot.
        """
        [self.display_per_storey(st) for st in self.main_storeys.keys()]


        # grids_per_storey = {
        #     "structural": {
        #         "IfcColumn": [column_grid_linestrings, column_grid_components],
        #         "IfcWall": [wall_s_grid_linestrings, wall_s_grid_components]
        #     },
        #     "non-structural":{
        #         "IfcWall": [wall_ns_grid_linestrings, wall_ns_grid_components]
        #     }}
        
        # self.grids.update({storey.GlobalId: grids_per_storey})

#--------------------------
    def align_same_type(self, grid_linestrings, grid_componnets, tol=0.0):
        
        # find all the pairs.s
        aligned_idx = []

        for i, gd_1 in enumerate(grid_linestrings):

            for j, gd_2 in enumerate(grid_linestrings):
        
                # combination pairs (i,j)
                if i < j:

                    # count the pairs of ids if align.
                    if not shapely.intersects(gd_1,gd_2) and shapely.distance(gd_1,gd_2) < tol: # todo., this intersects might have some errors.
                        aligned_idx.append([i,j])
                    else:
                        continue
                else:
                    continue
                    
        # count and prioritize the alignment orders.
        id_frequency = Counter([item for sublist in aligned_idx for item in sublist])
        sorted_id_by_occurency = [item for item, count in id_frequency.most_common()]

        # clear the alignment relationships.
        logic_aligned_idx = []
        for id_host in sorted_id_by_occurency:
            
            existing_logic_aligned_idx = [item for sublist in logic_aligned_idx for item in sublist]
            
            if id_host not in existing_logic_aligned_idx:
                
                idx = [pair for pair in aligned_idx if id_host in pair]
                idx = list(set([item for sublist in idx for item in sublist]))

                idx.remove(id_host)
                [idx.remove(i) for i in existing_logic_aligned_idx if i in idx]

                new_logic_aligned_idx = [id_host,*idx]

                if len(new_logic_aligned_idx)>=2:
                    logic_aligned_idx.append(new_logic_aligned_idx)
            
            else:
                continue
        
        # alignment.
        grid_linestrings_aligned, grid_componnets_aligned = [], []

        for gd_id, gd_line in enumerate(grid_linestrings):
            
            # > not procesed yet
            if grid_linestrings[gd_id] not in grid_linestrings_aligned:
                
                # > > if it's related to alignments
                if gd_id in [item for sublist in logic_aligned_idx for item in sublist]:

                    for logic_pair in logic_aligned_idx:
                        
                        # if it's related to an alignment, and it's a host
                        if gd_id == logic_pair[0]:
                            grid_linestrings_aligned.append(grid_linestrings[gd_id])
                            new_components = [grid_componnets[id] for id in logic_pair]
                            grid_componnets_aligned.append([item for sublist in new_components for item in sublist])
                            break 
                        
                        # if it's related toan alignment, but it's not a host
                        elif gd_id in logic_pair:
                            break 
                        
                        # didn't find in this logic pair
                        else:
                            continue
                
                # > >if it's not related to any alignment.
                else:
                    grid_linestrings_aligned.append(grid_linestrings[gd_id])
                    grid_componnets_aligned.append(grid_componnets[gd_id])

            # > already procesed.
            else:
                continue

        return grid_linestrings_aligned, grid_componnets_aligned
#--------------------------
    
#todo.
    
    # def adjust_grids_per_storey(
    #     self,
    #     storey,
    #     t_self_dist=0.001,
    #     t_cross_dist=0.4,
    #     plot_fig=True,
    #     ):

    #     # get grids per storey.
    #     if storey.GlobalId in self.grids.keys():
    #         grids_per_storey = self.grids[storey.GlobalId]

    #     #---------------------------------------------------------------------------------------------------
    #     # Structural merge: merge overlapping structural grids from IfcColumn and IfcWall.
    #     gd_type = "structural" 
    #     st_grids_linestrings =  grids_per_storey[gd_type]["IfcColumn"][0] + grids_per_storey[gd_type]["IfcWall"][0]
    #     st_grids_componnets =  grids_per_storey[gd_type]["IfcColumn"][1] + grids_per_storey[gd_type]["IfcWall"][1]
        
    #     st_grids_linestrings_merged, st_grids_componnets_merged = self.align_same_type(
    #         grid_linestrings=st_grids_linestrings, grid_componnets=st_grids_componnets, tol=t_self_dist)

    #     self.grids[storey.GlobalId][gd_type].update({"self-merged": [st_grids_linestrings_merged, st_grids_componnets_merged]})
        
    #     #---------------------------------------------------------------------------------------------------
    #     # Non-structural merge: merge overlapping non-structural grids from  IfcWall.
    #     gd_type = "non-structural"
    #     ns_grids_linestrings =  grids_per_storey[gd_type]["IfcWall"][0]
    #     ns_grids_componnets =  grids_per_storey[gd_type]["IfcWall"][1]

    #     ns_grids_linestrings_merged, ns_grids_componnets_merged = self.align_same_type(
    #         grid_linestrings=ns_grids_linestrings, grid_componnets=ns_grids_componnets, tol=t_self_dist)

    #     self.grids[storey.GlobalId][gd_type].update({"self-merged": [ns_grids_linestrings_merged, ns_grids_componnets_merged]})

    #     #---------------------------------------------------------------------------------------------------
    #     # Align the Non-structural to structural: remove non-structural grids close to neighboring (merged) structural grids.
    #     gd_type = "non-structural"
    #     ns_grids_linestrings_merged =  grids_per_storey[gd_type]["self-merged"][0]
    #     ns_grids_componnets_merged =  grids_per_storey[gd_type]["self-merged"][1]

    #     aligned_ns_to_st=[]

    #     for ii, gd_st in enumerate(st_grids_linestrings_merged):

    #         for jj, gd_ns in enumerate(ns_grids_linestrings_merged):
                
    #             # if not aligned yet with structural grids.
    #             if jj not in aligned_ns_to_st:

    #                 # if parallel and too close < t_cross_dist.
    #                 if not shapely.intersects(gd_st,gd_ns) and shapely.distance(gd_st,gd_ns) < t_cross_dist:
                        
    #                     if ns_grids_componnets_merged[jj] not in st_grids_componnets_merged[ii]:
    #                         print (shapely.distance(gd_st,gd_ns))
    #                         st_grids_componnets_merged[ii]+=ns_grids_componnets_merged[jj]
    #                         aligned_ns_to_st.append(jj)
                
    #             else:
    #                 continue
        
    #     # final update of the merge.
    #     ns_grids_linestrings_merged = [e for i, e in enumerate(ns_grids_linestrings_merged) if i not in aligned_ns_to_st]
    #     ns_grids_componnets_merged = [e for i, e in enumerate(ns_grids_componnets_merged) if i not in aligned_ns_to_st]
    #     self.grids[storey.GlobalId]["structural"].update({"cross-merged": [st_grids_linestrings_merged, st_grids_componnets_merged]})
    #     self.grids[storey.GlobalId]["non-structural"].update({"cross-merged": [ns_grids_linestrings_merged, ns_grids_componnets_merged]})

    #     # =========================== visualization
    #     (wall_lines_struc,wall_lines_nonst,column_points) = self.get_info_elements_per_storey(storey=storey)
        
    #     plot_name = f"\[Floor \, Plan \, of \, {storey.Name} \, (T_{{self,dist}}={t_self_dist}, \, T_{{cross,dist}}={t_cross_dist}) - Gird \, Alignment \]"
    #     fig_save_name = f"Merge_{storey.Name}_t_self_dist_{t_self_dist}_t_cross_dist_{t_cross_dist}"

    #     fig = bokeh.plotting.figure(
    #         title=plot_name,
    #         title_location='above',
    #         x_axis_label='x',
    #         y_axis_label='y',
    #         width=800,
    #         height=800,
    #         match_aspect=True)
    #     fig.title.text_font_size = '11pt'

    #     #--------------------------
    #     # structural grids.
    #     st_grids_linestrings_merged = self.grids[storey.GlobalId]["structural"]["cross-merged"][0]
    #     g_plot = self.visualization_settings['grids_st_merged']
    #     for ls in st_grids_linestrings_merged:
    #         x, y = ls.coords.xy
    #         fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])
            
    #     # non-structural grids.
    #     ns_grids_linestrings_merged = self.grids[storey.GlobalId]["non-structural"]["cross-merged"][0]
    #     g_plot = self.visualization_settings['grids_ns_merged']
    #     for ls in ns_grids_linestrings_merged:
    #         x, y = ls.coords.xy
    #         fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])

    #     #--------------------------
    #     # columns
    #     g_plot = self.visualization_settings['column_points_struc']
    #     for point in column_points:
    #         fig.square(point.x, point.y, legend_label=g_plot['legend_label'], size=g_plot['size'], color=g_plot['color'], alpha=g_plot['alpha'])
        
    #     # structural walls
    #     g_plot = self.visualization_settings['wall_lines_struc']
    #     for ls in wall_lines_struc:
    #         x, y = ls.coords.xy
    #         fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])
        
    #     # non-structural walls
    #     g_plot = self.visualization_settings['wall_lines_nonst']
    #     for ls in wall_lines_nonst:
    #         x, y = ls.coords.xy
    #         fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])

    #     fig.xgrid.visible = False
    #     fig.ygrid.visible = False

    #     if plot_fig:
    #         bokeh.plotting.output_file(filename=os.path.join(self.out_fig_path, fig_save_name + ".html"), title=fig_save_name)
    #         bokeh.plotting.save(fig)

                
#Grids ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================
    
# old
    
     # def get_info_elements_per_storey(self, storey, tol_elevation=0.001):

    #     wall_info_per_storey = []
    #     for wall in self.info_walls:
    #         if abs(wall['location'][0][-1]-storey.Elevation) <= tol_elevation :
    #             wall_info_per_storey.append(wall)

    #     column_info_per_storey = []
    #     for column in self.info_columns:
    #         if abs(column['location'][-1]-storey.Elevation) <= tol_elevation :
    #             column_info_per_storey.append(column)

    #     # differentiate between structural and non-structural walls.
    #     s_wall_locations =  [w['location'] for w in wall_info_per_storey if w['loadbearing']]
    #     ns_wall_locations =  [w['location'] for w in wall_info_per_storey if not w['loadbearing']]
        
    #     wall_locations_struc = copy.deepcopy(s_wall_locations)
    #     wall_locations_nonst = copy.deepcopy(ns_wall_locations)
    #     [p.pop() for wall_loc in wall_locations_struc for p in wall_loc]
    #     [p.pop() for wall_loc in wall_locations_nonst for p in wall_loc]
        
    #     s_column_locations = [c['location'] for c in column_info_per_storey]

    #     column_locations_struc = copy.deepcopy(s_column_locations)
    #     [column_loc.pop() for column_loc in column_locations_struc]
        
    #     wall_lines_struc = [LineString(wall_location) for wall_location in wall_locations_struc]
    #     wall_lines_nonst = [LineString(wall_location) for wall_location in wall_locations_nonst]
       
    #     column_points = [Point(column_loc) for column_loc in column_locations_struc]

    #     return (wall_lines_struc,wall_lines_nonst,column_points)