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
from collections import defaultdict

import bokeh.plotting

from quickTools import get_line_slope_by_points, remove_duplicate_points, are_points_collinear, deep_merge_dictionaries
from quickTools import time_decorator, check_repeats_in_list, flatten_and_merge_lists, enrich_dict_with_another, calculate_line_crosses, a_is_subtuple_of_b

#===================================================================================================
#Grids ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

class GridGenerator:

    def __init__(self, figure_path, json_floors, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls):

        # initial data
        self.info_floors = []
        self.info_st_columns = []
        self.info_st_walls = []
        self.info_ns_walls = []
        self.info_ct_walls = []

        self.main_directions = {}
        self.main_storeys_from_ifc_columns = {}
        self.main_storeys_from_ifc_walls = {}
        self.main_storeys_from_ifc_ct_walls = {}
        self.main_storeys_from_ifc_floors = {}
        self.main_storeys = {}

        # = = = = = = = = = = = = generation parameters = = = = = = = = = 
        # semi-static: vertical threshold.
        self.z_storey_raise = 0.8
        self.z_crossing_columns = 0.1

        # per building. columns related
        self.st_c_num = 3
        self.st_c_dist = 0.001

        # per building. structural walls related
        self.st_w_num = 2
        self.st_w_dist = 0.001
        self.st_w_accumuled_length = 5

        # per building. non-structural walls related
        self.ns_w_num = 2
        self.ns_w_dist = 0.001
        self.ns_w_accumuled_length = 5
        #------
        # todo/
        #------
        # print('here is a todo.')
        # later switch the non-structural wall threshold values to floor-dependent?

        #------
        # todo/
        #------
        # print('here is a todo.')
        # merging related
        # self.t_self_dist = 0.2
        # self.t_cross_dist = 0.5
        
        self.border_x = None
        self.border_y = None

        self.out_fig_path = figure_path
        self.read_infos(json_floors, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls)
        self.init_visualization_settings()

        # print ("=============GridGenerator=============")
        # print (self.out_fig_path)

    def update_parameters(self, new_parameters):
        
        if not isinstance(new_parameters, dict):
            raise ValueError("the input 'new_parameters' must be a dictionary.")
        
        # List of valid parameter names for reference
        valid_parameters = {
            'st_c_num',
            'st_c_dist',
            'st_w_num',
            'st_w_dist',
            'st_w_accumuled_length',
            'ns_w_num',
            'ns_w_dist',
            'ns_w_accumuled_length',
            }
        
        for key, value in new_parameters.items():
            if key in valid_parameters:
                if isinstance(value, (int, float)):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"The value for {key} must be an int or float.")
            else:
                raise KeyError(f"{key} is not a valid parameter name.")
        
        return self

    def read_infos(self, json_floors, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls):
        
        def read_json_file(file_path):
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as file:
                        return json.load(file)
                except FileNotFoundError:
                    print(f"File {file_path} not found.")
                    return None
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {file_path}.")
                    return None
            else:
                return []

        self.info_floors = read_json_file(json_floors)
        self.info_st_columns = read_json_file(json_st_columns)
        self.info_st_walls = read_json_file(json_st_walls)
        self.info_ns_walls = read_json_file(json_ns_walls)
        self.info_ct_walls = read_json_file(json_ct_walls)
        self.info_all_walls = self.info_st_walls + self.info_ns_walls + self.info_ct_walls
        
        self.id_st_columns = [info['id'] for info in self.info_st_columns]
        self.id_st_walls = [info['id'] for info in self.info_st_walls]
        self.id_ns_walls = [info['id'] for info in self.info_ns_walls]
        self.id_ct_walls = [info['id'] for info in self.info_ct_walls]
        
    def init_visualization_settings(self):
        """
        Initializes visualization settings for various building components.
        """

        self.visualization_settings = {
            # points.
            'st_column_points':{
                'legend_label':'Column Locations',
                'color': "darkgreen",
                'size':8,
                'alpha':1,
            },

            # lines.
            'st_wall_lines':{
                'legend_label':'Structural Wall Locations',
                'color': "black",
                'line_dash':'solid',
                'line_width':3,
                'alpha':1,
            },
            'ns_wall_lines':{
                'legend_label':'Non-structural Wall Locations',
                'color': "dimgray",
                'line_dash':'solid',
                'line_width':3,
                'alpha':1,
            },

            # grid lines.
            'location_grids_st_c': {
                'legend_label':'Grids from structural Columns',
                'color': "tomato",
                'line_dash':'dotted',
                'line_width':2,
                'alpha':0.85,
            },
            'location_grids_st_w': {
                'legend_label': 'Grids from structural Walls',
                'color': "orange",
                'line_dash':'dashed',
                'line_width':2,
                'alpha':0.60,
            },
            'location_grids_ns_w': {
                'legend_label': 'Grids from non-structural Walls',
                'color': "navy",
                'line_dash':'dashed',
                'line_width':2,
                'alpha':0.60,
            },

            # processed grid lines.
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
    
    # column-related.
    def get_grids_from_column_points(self, element_pts, element_ids):
        """
        Return: Points from collinear pairs of points.
        """

        # check if the input data is feasible pairs.
        if len(element_pts)!= len(element_ids):
            return None, None
        
        def is_close(a, b, tolerance):
            return abs(a - b) <= tolerance
        
        def check_existence_with_index(new, seen, tolerance=0.001):
            
            if not seen:
                return False, None
            else:
                for i, sub_seen in enumerate(seen):
                    if all(is_close(sub_seen[j], new[j], tolerance) for j in range(len(new))):
                        return True, i
                return False, None
        
        # Preprocessing for identifying the "same located columns (below and above)."
        seen_xy, uniq_element_points, uniq_element_ids = [], [], []
        for (loc_pt, id_pt) in zip(element_pts, element_ids):
            loc_pt_xy = [loc_pt.x, loc_pt.y]
            existence, existence_index = check_existence_with_index(loc_pt_xy, seen_xy)
            
            if not existence:
                seen_xy.append(loc_pt_xy)
                uniq_element_points.append(loc_pt)
                uniq_element_ids.append([id_pt])
            else:
                uniq_element_ids[existence_index].append(id_pt)

        # Iterate through each pair of points once
        generated_grids, element_ids_per_grid, element_ids_per_grid_reps = [], [], []

        for i, (point1, id_group_1) in enumerate(zip(
            uniq_element_points[:-1], uniq_element_ids[:-1])):
    
            for j, (point2, id_group_2) in enumerate(
                zip(uniq_element_points[i + 1:], uniq_element_ids[i + 1:]), start=i+1):
            
                aligned_points = [point1, point2]
                related_element_ids_reps = [id_group_1[0], id_group_2[0]]
                related_element_ids = id_group_1+id_group_2  # Use a set to avoid duplicate indices
            
                slope = get_line_slope_by_points(point1, point2)
                
                # for columns, ignore those pairs not located on the main directions.
                # if not is_close_to_known_slopes(slope, self.main_directions):
                #     continue
                # else:

                for k, (point, id_group_3) in enumerate(zip(uniq_element_points[j + 1:], uniq_element_ids[j + 1: ]), start=j+1):

                    if slope == float('inf'):  # Vertical line check
                        if abs(point.x - point1.x) <= self.st_c_dist and abs(point.x - point2.x) <= self.st_c_dist:
                            aligned_points.append(point)
                            related_element_ids_reps.append(id_group_3[0])
                            related_element_ids+=id_group_3
                    else:
                        # Use point-slope form of line equation to check alignment
                        if abs((point.y - point1.y) - slope * (point.x - point1.x)) <= self.st_c_dist:
                            aligned_points.append(point)
                            related_element_ids_reps.append(id_group_3[0])
                            related_element_ids+=id_group_3

                # be careful with the hierarchy here.
                # Check for minimum number of points and uniqueness before adding to the grid
                if len(aligned_points) >= self.st_c_num:
                    id_tuple_reps = tuple(sorted(related_element_ids_reps)) 
                    id_tuple = tuple(sorted(related_element_ids))
                    
                    if any([a_is_subtuple_of_b(id_tuple_reps, existing_tuple_reps) for existing_tuple_reps in element_ids_per_grid_reps]):
                        continue
                    else:
                        generated_grids.append(aligned_points)
                        element_ids_per_grid_reps.append(id_tuple_reps)
                        element_ids_per_grid.append(id_tuple)
        
        # todo.
        # clean the points in generated_grids by considering:
        # 1. weights from element_ids_per_grid
        # 2. offset thresholds.

        return generated_grids, element_ids_per_grid
    
    def get_grids_from_wall_lines(self, element_lns, element_ids, line_type=[]):
        
        # check if the input data is feasible pairs.
        if len(element_lns)!= len(element_ids):
            return None, None
        
        if not line_type:
            raise ValueError("line_type hasn't been specified for 'lines2grids'.")
        if line_type == 'structural':
            minimum_accumuled_wall_length = self.st_w_accumuled_length
            minimum_alignment_number = self.st_w_num
            wall_offset_distance = self.st_w_dist  # area spanned by the triangle they would form
        elif line_type == 'non-structural':
            minimum_accumuled_wall_length = self.ns_w_accumuled_length
            minimum_alignment_number = self.ns_w_num
            wall_offset_distance = self.ns_w_dist # area spanned by the triangle they would form

        # ##################################################################################
        # todo. Preprocessing for identifying the "same located walls (below and above)."
        # ##################################################################################

        # Pre-calculate slopes for all lines
        line_slopes = [get_line_slope_by_points(list(ln.boundary.geoms)[0], list(ln.boundary.geoms)[1]) for ln in element_lns]
        line_slopes = [float('inf') if ln_slope==float('inf') else round(ln_slope, 4) for ln_slope in line_slopes] # might cause minor issues.
        line_lengths = [ln.length for ln in element_lns]

        # divide the data in to aligned groups for following iterative calculcations.
        grouped_data = defaultdict(list)
        for index, slope in enumerate(line_slopes):
            grouped_data[slope].append([element_lns[index], element_ids[index], line_lengths[index]])

        generated_grids, element_ids_per_grid = [], []

        for slope, data_value in grouped_data.items():
            grouped_data[slope] = list(map(list, zip(*data_value))) # update the data structure.
            
            # Walls of the same slope.
            element_lns, element_ids, ln_lengths = grouped_data[slope]
            for i, (ln_1, id_1, length_1) in enumerate(zip(
                element_lns[:-1], element_ids[:-1], ln_lengths[:-1])):
                
                point1, point2 = list(ln_1.boundary.geoms)[0],list(ln_1.boundary.geoms)[1]
                aligned_points = [[point1, point2]]
                accumulated_length = length_1
                aligned_element_ids = {id_1}

                # Sub-case of non-vertical lines.
                if slope != float('inf'):

                    for j, (ln_new, new_length, id_new) in enumerate(zip(
                        element_lns[i + 1:], ln_lengths[i + 1:], element_ids[i + 1:]), start=i + 1):

                        point3, point4 = list(ln_new.boundary.geoms)[0],list(ln_new.boundary.geoms)[1]

                        if are_points_collinear(point1, point2, point3, point4, t=wall_offset_distance):
                            aligned_points.append([point3,point4])
                            accumulated_length += new_length
                            aligned_element_ids.add(id_new)
                        else:
                            continue

                # Sub-case of vertical lines.
                elif slope == float('inf'):

                    for j, (ln_new, new_length, id_new) in enumerate(zip(
                        element_lns[i + 1:], ln_lengths[i + 1:], element_ids[i + 1:]), start=i + 1):

                        point3, point4 = list(ln_new.boundary.geoms)[0],list(ln_new.boundary.geoms)[1]
                        
                        if abs(point3.x-point1.x) <= wall_offset_distance and \
                            abs(point3.x-point2.x) <= wall_offset_distance and \
                                abs(point4.x-point1.x) <= wall_offset_distance and \
                                    abs(point4.x-point2.x) <= wall_offset_distance:
                            aligned_points.append([point3,point4])
                            accumulated_length += new_length
                            aligned_element_ids.add(id_new)
                        else:
                            continue
                
                aligned_element_ids = sorted(aligned_element_ids)

                # be careful with the hierarchy here.
                if len(aligned_element_ids) >= minimum_alignment_number and \
                    accumulated_length >= minimum_accumuled_wall_length:
                    
                    id_tuple = tuple(aligned_element_ids)  # Convert to tuple for hashability
                    if id_tuple not in element_ids_per_grid:
                        if element_ids_per_grid and any([
                            set(id_tuple).issubset(set(existing_ids_component)) for existing_ids_component in element_ids_per_grid]):
                            continue
                        else:
                            generated_grids.append(aligned_points)
                            element_ids_per_grid.append(list(set(aligned_element_ids)))
                            
        generated_grids = [[e for element in elements for e in element] for elements in generated_grids]
        generated_grids = [remove_duplicate_points(elements) for elements in generated_grids]

        # todo.
        # clean the points in generated_grids by considering:
        # 1. weights from element_ids_per_grid
        # 2. offset thresholds.

        return generated_grids, element_ids_per_grid
    
    def generate_grids_from_candidate_points(self, grid_elements):
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


    def get_element_information_for_grid(self):
        """
        grids_st_c. from IfcColumn.
        grids_st_w. from structural IfcWall.
        grids_ns_w. from non-structural IfcWall + IfcCurtainWall.
        """
        
        self.grids_all = {}
        
        # st_column
        # ------------------
        # by whole building.
        # ------------------
        st_column_points = self.info_all_locations.get('st_column_points')
        st_column_ids = self.info_all_locations.get('st_column_ids')

        if st_column_points and st_column_ids:
            grids_st_c, grids_st_c_ids = self.get_grids_from_column_points(st_column_points, st_column_ids)
        
            self.grids_all.update({
                'grids_st_c': grids_st_c,
                'grids_st_c_ids': grids_st_c_ids,
                })
        else:
            self.grids_all.update({
                'grids_st_c': [],
                'grids_st_c_ids': [],
                })
        
        # st_wall
        # ------------------
        # by whole building.
        # ------------------
        st_wall_lines = self.info_all_locations.get('st_wall_lines')
        st_wall_ids = self.info_all_locations.get('st_wall_ids')

        if st_wall_lines and st_wall_ids:
            grids_st_w, grids_st_w_ids = self.get_grids_from_wall_lines(st_wall_lines, st_wall_ids,line_type='structural')
        
            self.grids_all.update({
                'grids_st_w': grids_st_w,
                'grids_st_w_ids': grids_st_w_ids,
                })
        else:
            self.grids_all.update({
                'grids_st_w': [],
                'grids_st_w_ids': [],
                })

        # ns_wall
        # ------------------
        # by storey.
        # ------------------
        for key, value in self.info_all_locations_by_storey.items():

            # ns_wall
            ns_wall_lines = value.get('ns_wall_lines')
            ns_wall_ids = value.get('ns_wall_ids')
            
            # curtain wall
            ct_wall_lines = value.get('ct_wall_lines')
            ct_wall_ids = value.get('ct_wall_ids')
            
            ns_wall_lines += ct_wall_lines 
            ns_wall_ids += ct_wall_ids

            if ns_wall_lines and ns_wall_ids:
                grids_ns_w, grids_ns_w_ids = self.get_grids_from_wall_lines(ns_wall_lines, ns_wall_ids,line_type='non-structural')
            
                self.grids_all.update({
                    key: {
                        'grids_ns_w': grids_ns_w,
                        'grids_ns_w_ids': grids_ns_w_ids,
                    }})
            else:
                self.grids_all.update({
                    key: {
                        'grids_ns_w': [],
                        'grids_ns_w_ids': [],
                    }})
                
    def get_main_storeys_with_columns(self, num_columns=1):

        if not self.info_st_columns:
            print("No column information available.")
            return

        columns_by_elevation = {}
        for w in self.info_st_columns:
            try:
                elevation = w["elevation"]
            except KeyError:
                continue  # Skip the current iteration if elevation key is missing
            columns_by_elevation.setdefault(elevation, []).append(w)

        self.main_storeys_from_ifc_columns = {elevation: {"columns": columns}
                                          for elevation, columns in columns_by_elevation.items() if len(columns) >= num_columns}

    def get_main_storeys_with_walls(self, num_walls=1, include_ct_walls=False):

        considered_walls = self.info_ns_walls + self.info_st_walls

        if not considered_walls:
            print("No wall information available.")
            return

        walls_by_elevation = {}
        for w in considered_walls:
            try:
                elevation = w["elevation"]
            except KeyError:
                continue  # Skip if elevation key is missing
            walls_by_elevation.setdefault(elevation, []).append(w)

        self.main_storeys_from_ifc_walls = {elevation: {"walls": walls}
                                        for elevation, walls in walls_by_elevation.items() if len(walls) >= num_walls}

        # curtain walls.
        if include_ct_walls and self.info_ct_walls:
            
            considered_walls = self.info_ct_walls
            ct_walls_by_elevation = {}
            for w in considered_walls:
                try:
                    elevation = w["elevation"]
                except KeyError:
                    continue  # Skip if elevation key is missing
                ct_walls_by_elevation.setdefault(elevation, []).append(w)
        
            self.main_storeys_from_ifc_ct_walls = {elevation: {"curtain walls": walls}
                                            for elevation, walls in ct_walls_by_elevation.items() if len(walls) >= num_walls}
        
    def get_main_storeys_with_floors(self, num_floors=1):
        
        if not self.info_floors:
            print("No wall information available.")
            return
        
        floors_by_elevation = {}
        for fl in self.info_floors:
            try:
                elevation = fl["elevation"]
            except KeyError:
                continue  # Skip if elevation key is missing
            floors_by_elevation.setdefault(elevation, []).append(fl)

        self.main_storeys_from_ifc_floors = {elevation: {"floors": floors}
                                         for elevation, floors in floors_by_elevation.items() if len(floors) >= num_floors}

    def get_main_storeys_init(self, include_ct_walls=False):

        # here we take floor and wall elements to determine the initial main storeys.
        self.get_main_storeys_with_columns()
        self.get_main_storeys_with_floors()
        self.get_main_storeys_with_walls(include_ct_walls=include_ct_walls)

        # column + walls 
        self.main_storeys = deep_merge_dictionaries(self.main_storeys_from_ifc_columns, self.main_storeys_from_ifc_walls)
        # + floors.
        self.main_storeys = deep_merge_dictionaries(self.main_storeys, self.main_storeys_from_ifc_floors)
        # + curtain walls.
        if include_ct_walls:
            self.main_storeys = deep_merge_dictionaries(self.main_storeys, self.main_storeys_from_ifc_ct_walls)
        
        # ============================================================================
        # keep for now.
        # merge the curtain walls when they carry slightly different elevation values
        # if include_curtainwalls:
        #     merge_keys = set()
        #     for k_second in self.main_storeys_from_ifc_ct_walls.keys():
        #         for k_main in self.main_storeys.keys():
        #             if k_second!= k_main and abs(k_second-k_main) <= max_z_diff_curtainwall:
        #                 merge_keys.add((k_second,k_main))
            
        #     for k_couples in merge_keys:
        #         k_second, k_main = k_couples
        #         if k_second in self.main_storeys_from_ifc_ct_walls.keys() and \
        #         k_main not in self.main_storeys_from_ifc_ct_walls.keys():
        #             self.main_storeys_from_ifc_ct_walls[k_main] = self.main_storeys_from_ifc_ct_walls.pop(k_second)
        #             for i in range(len(self.main_storeys_from_ifc_ct_walls[k_main]['curtain walls'])):
        #                 self.main_storeys_from_ifc_ct_walls[k_main]['curtain walls'][i]['elevation'] = k_main
        # ============================================================================

    def enrich_main_storeys(self):
        
        for key, value in self.main_storeys.items():
            
            ids_per_storey = []
            bottom_z_ranges_per_storey = [None, None]

            for sub_key, sub_value in value.items():

                ids_per_sub_key = [v['id'] for v in sub_value]
                ids_per_storey += ids_per_sub_key

                if sub_key == 'floors':
                    z_values = [fl['elevation'] for fl in sub_value]
                    bottom_z_ranges_per_storey = [min(z_values),max(z_values)]

            self.main_storeys[key].update({
                'ids': ids_per_storey,
                'bottom_z_ranges': bottom_z_ranges_per_storey,
                })
        
        # remove the storeys that doesn't have any attached floors.
        self.main_storeys =  {key: value for key, value in self.main_storeys.items() if 'floors' in value.keys()}
         
    def get_main_directions(self, num_directions):
            
        def degree2slope(degree):
            t_direction_degree = 0.0001
            slope = float('inf') if abs(degree-90.0)<t_direction_degree else math.radians(degree) # static threshold.
            return slope
        
        wall_orientations = [w['orientation'] for w in self.info_all_walls if 'orientation' in w]
        wall_orientations = [(v-180) if v>=180 else v for v in wall_orientations]
        main_directions = Counter(wall_orientations)
        main_directions = main_directions.most_common(num_directions)

        self.main_directions = [main_direct[0] for main_direct in main_directions]
        self.main_directions = [degree2slope(main_direct) for main_direct in self.main_directions]
    
    # no need of this function if we treat the columns for the whole building.
    def identify_columns_cross_storeys(self):
        
        def filter_by_location_z(dicts, z):    
            filtered_dicts = []
            for d in dicts:
                try:
                    if (z -d['location'][0][-1]) >= self.z_crossing_columns and (d['location'][1][-1]-z) >= self.z_crossing_columns:
                        filtered_dicts.append(d)
                except KeyError:
                    print("Warning: One of the dictionaries is missing the 'location' key.")
                except IndexError:
                    print("Warning: 'location' data is improperly structured.")
                except TypeError:
                    print("Warning: Incompatible type encountered in 'location' data.")
            return filtered_dicts

        for storey_key, storey_value in self.main_storeys.items():
            
            if storey_value.get('columns', None) is not None:
                pass
            else:
                crossed_columns = filter_by_location_z(self.info_st_columns, storey_key)
                self.main_storeys[storey_key].update({'columns': crossed_columns})
    
    def enrich_main_storeys_info_raised_area(self):
        """
        dynamic storey merging process.
        """
        # todo.
        storeys_with_raised_area = [sorted([st1,st2]) for (st1,st2) in itertools.combinations(self.main_storeys.keys(), 2) if abs(st1-st2)<=self.z_storey_raise]
        
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
            print("Non raised area according to the given 'self.z_storey_raise'.")

    def get_element_locations(self):
        
        self.info_all_locations = {}
        self.info_all_locations_by_storey = {}

        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # info_st_columns.
        if self.info_st_columns:
        
            st_column_location_id_pairs = [(c['location'], c['id']) for c in self.info_st_columns]
            st_column_locations, st_column_ids = zip(*st_column_location_id_pairs)
            st_column_locations, st_column_ids = list(st_column_locations), list(st_column_ids)
            
            # to use only the base locations.
            st_column_base_locations = copy.deepcopy(st_column_locations)
            [column_loc.pop() for column_loc in st_column_base_locations] 
            st_column_base_locations = [item for sublist in st_column_base_locations for item in sublist] # flatten.
            
            # convert locations to points 
            st_column_points = [Point(column_loc) for column_loc in st_column_base_locations]
            
            if st_column_points is not None:
                self.info_all_locations.update({
                    'st_column_points': st_column_points,
                    'st_column_ids': st_column_ids,
                    })
            else:
                self.info_all_locations.update({
                    'st_column_points': [],
                    'st_column_ids': [],
                    })
            
            # classify by main storeys.
            if st_column_points is not None:

                for key in self.main_storeys.keys():

                    pts_per_storey, ids_per_storey = [],[]
                    
                    st_column_points = self.info_all_locations['st_column_points']
                    st_column_ids = self.info_all_locations['st_column_ids']
                    
                    for pt, id in zip(st_column_points, st_column_ids):
                        if id in self.main_storeys[key]['ids']:
                            pts_per_storey.append(pt)
                            ids_per_storey.append(id)

                    if key not in self.info_all_locations_by_storey.keys():
                        self.info_all_locations_by_storey[key] = {}
                    self.info_all_locations_by_storey[key].update({
                        'st_column_points': pts_per_storey,
                        'st_column_ids':ids_per_storey,
                    })
        
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # info_st_walls.
        if self.info_st_walls:

            st_wall_location_id_pairs = [(w['location'], w['id']) for w in self.info_st_walls]
            st_wall_locations, st_wall_ids = zip(*st_wall_location_id_pairs)
            st_wall_locations, st_wall_ids = list(st_wall_locations), list(st_wall_ids)
            
            # to use only the xy planar locations.
            st_wall_planar_locations = copy.deepcopy(st_wall_locations)
            [p.pop() for wall_loc in st_wall_planar_locations for p in wall_loc]
            
            # convert locations to lines 
            st_wall_lines = [LineString(wall_location) for wall_location in st_wall_planar_locations]

            if st_wall_lines is not None:
                self.info_all_locations.update({
                    'st_wall_lines': st_wall_lines,
                    'st_wall_ids': st_wall_ids,
                    })
            else:
                self.info_all_locations.update({
                    'st_wall_lines': [],
                    'st_wall_ids': [],
                    })
        
            # classify by main storeys.
            if st_wall_lines is not None:

                for key in self.main_storeys.keys():

                    ls_per_storey, ids_per_storey = [],[]
                    
                    st_wall_lines = self.info_all_locations['st_wall_lines']
                    st_wall_ids = self.info_all_locations['st_wall_ids']
                    
                    for l, id in zip(st_wall_lines, st_wall_ids):
                        if id  in self.main_storeys[key]['ids']:
                            ls_per_storey.append(l)
                            ids_per_storey.append(id)
                    
                    if key not in self.info_all_locations_by_storey.keys():
                        self.info_all_locations_by_storey[key] = {}
                    self.info_all_locations_by_storey[key].update({
                            'st_wall_lines': ls_per_storey,
                            'st_wall_ids':ids_per_storey,
                        })
                    
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # info_ns_walls.
        if self.info_ns_walls:

            ns_wall_location_id_pairs = [(w['location'], w['id']) for w in self.info_ns_walls]
            ns_wall_locations, ns_wall_ids = zip(*ns_wall_location_id_pairs)
            ns_wall_locations, ns_wall_ids = list(ns_wall_locations), list(ns_wall_ids)
            
            # to use only the xy planar locations.
            ns_wall_planar_locations = copy.deepcopy(ns_wall_locations)
            [p.pop() for wall_loc in ns_wall_planar_locations for p in wall_loc]
            
            # convert locations to lines 
            ns_wall_lines = [LineString(wall_location) for wall_location in ns_wall_planar_locations]

            if ns_wall_lines is not None:
                self.info_all_locations.update({
                    'ns_wall_lines': ns_wall_lines,
                    'ns_wall_ids': ns_wall_ids,
                    })
            else:
                self.info_all_locations.update({
                    'ns_wall_lines': [],
                    'ns_wall_ids': [],
                    })
            
            # classify by main storeys.
            if ns_wall_lines is not None:

                for key in self.main_storeys.keys():

                    ls_per_storey, ids_per_storey = [],[]
                    
                    ns_wall_lines = self.info_all_locations['ns_wall_lines']
                    ns_wall_ids = self.info_all_locations['ns_wall_ids']
                    
                    for l, id in zip(ns_wall_lines, ns_wall_ids):
                        if id  in self.main_storeys[key]['ids']:
                            ls_per_storey.append(l)
                            ids_per_storey.append(id)
                    
                    if key not in self.info_all_locations_by_storey.keys():
                        self.info_all_locations_by_storey[key] = {}

                    self.info_all_locations_by_storey[key].update({
                        'ns_wall_lines': ls_per_storey,
                        'ns_wall_ids':ids_per_storey,
                        })
        
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # info_ct_walls.
        if self.info_ct_walls:

            ct_wall_location_id_pairs = [(w['location'], w['id']) for w in self.info_ct_walls]
            ct_wall_locations, ct_wall_ids = zip(*ct_wall_location_id_pairs)
            ct_wall_locations, ct_wall_ids = list(ct_wall_locations), list(ct_wall_ids)
            
            # to use only the xy planar locations.
            ct_wall_planar_locations = copy.deepcopy(ct_wall_locations)
            [p.pop() for wall_loc in ct_wall_planar_locations for p in wall_loc]
            
            # convert locations to lines 
            ct_wall_lines = [LineString(wall_location) for wall_location in ct_wall_planar_locations]

            if ct_wall_lines is not None:
                self.info_all_locations.update({
                    'ct_wall_lines': ct_wall_lines,
                    'ct_wall_ids': ct_wall_ids,
                    })
            else:
                self.info_all_locations.update({
                    'ct_wall_lines': [],
                    'ct_wall_ids': [],
                    })
            
            # classify by main storeys.
            if ct_wall_lines is not None:

                for key in self.main_storeys.keys():

                    ls_per_storey, ids_per_storey = [],[]
                    
                    ct_wall_lines = self.info_all_locations['ct_wall_lines']
                    ct_wall_ids = self.info_all_locations['ct_wall_ids']
                    
                    for l, id in zip(ct_wall_lines, ct_wall_ids):
                        if id  in self.main_storeys[key]['ids']:
                            ls_per_storey.append(l)
                            ids_per_storey.append(id)
                    
                    if key not in self.info_all_locations_by_storey.keys():
                        self.info_all_locations_by_storey[key] = {}

                    self.info_all_locations_by_storey[key].update({
                        'ct_wall_lines': ls_per_storey,
                        'ct_wall_ids':ids_per_storey,
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
    

    def extract_grid_overall_borders(self):        
        
        # get the global borders.
        all_grid_reference_points = self.grids_all['grids_st_c'] + self.grids_all['grids_st_w']
        for storey in self.main_storeys.keys():
            if self.grids_all[storey]['grids_ns_w']:
                all_grid_reference_points.append(self.grids_all[storey]['grids_ns_w'])
            else:
                continue
        all_grid_reference_points = flatten_and_merge_lists(all_grid_reference_points)
        
        pad_x_y = 5
        all_references_x, all_references_y = [pt.x for pt in all_grid_reference_points], [pt.y for pt in all_grid_reference_points]
        self.update_display_borders(all_references_x, all_references_y, pad_x_y) # necessary for the current calculation of grid lines.


    def calculate_grid_locations(self):        
         
        st_grid_terms = ['grids_st_c', 'grids_st_w']
        ns_grid_terms = ['grids_ns_w']
        
        for grid_type_key in st_grid_terms:

            grid_location_key = 'location_' + grid_type_key
            elements_for_grids = self.grids_all.get(grid_type_key)

            if elements_for_grids:
                self.grids_all[grid_location_key] = self.generate_grids_from_candidate_points(elements_for_grids)
            else:
                self.grids_all[grid_location_key] = []

        ns_grids_by_storey = {key: self.grids_all[key] for key in self.main_storeys.keys()}

        for storey_key, storey_value in ns_grids_by_storey.items():
            
            for grid_type_key in ns_grid_terms:

                grid_location_key = 'location_' + grid_type_key
                elements_for_grids = storey_value.get(grid_type_key)

                if elements_for_grids:
                    self.grids_all[storey_key].update({
                        grid_location_key: self.generate_grids_from_candidate_points(elements_for_grids)
                    })
                else:
                   self.grids_all[storey_key].update({
                        grid_location_key: []
                    })

    def prepare_wall_total_lengths_numbers(self):

        # IfcWall and IfcCurtainWall.
        total_length = 0
        for item in self.info_all_walls:
            if "length" in item:
                total_length += item["length"]

        self.total_wall_lengths = total_length
        self.total_wall_numbers = len(self.info_all_walls)
        self.info_wall_length_by_id = {d['id']: d['length'] for d in self.info_all_walls if 'id' in d and 'length' in d}

    # @time_decorator
    def get_main_directions_and_storeys(self, num_directions=2):
        """
        "static" initialization processes.
        """
        
        self.prepare_wall_total_lengths_numbers() # serve for calculating the percent / total length of st ns ct walls.
        self.get_main_directions(num_directions) # keep it for now. not fully used yet for algorithm.
        self.get_main_storeys_init(include_ct_walls=True) # get the elements perfloor, to check if it gets all necessary information...
        self.enrich_main_storeys()

        # self.identify_columns_cross_storeys() # static [no need of this because we consider it globally, but later might need to check.]  
    
    # @time_decorator
    def enrich_all_element_locations(self):
        
        # step 1
        self.get_element_locations()

        ##############################
        # step 2:
        # get the outlines. to invesitgate afer clean the models.
        ##############################
        
        ##############################
        # Step 2 preprocessing on the main storeys: irst merge the actually connecting floors. [no need yet, but later use floor outlines.]
        # self.enrich_main_storeys_info_raised_area()
        ##############################
    
    # a loss about local performance of the grids.
    # @time_decorator
    def calculate_grid_wall_cross_loss(self, ignore_cross_edge=False, cross_threshold_percent=5):

        # the loss target.
        cross_w_lengths = 0.0
        
        # get all walls bound to grids.
        self.ids_wall_bound = self.grids_all['grids_st_w_ids']
        for storey_key, storey_value in self.grids_all.items():
            if 'grids_ns_w_ids' in storey_value:
                self.ids_wall_bound += storey_value['grids_ns_w_ids']
        self.ids_wall_bound = set([item for sublist in self.ids_wall_bound for item in sublist])
        
        # get all global st grids and then do calculation per storey.
        self.ids_wall_unbound_crossed = set()
        global_st_grids = self.grids_all.get('location_grids_st_c', []) + self.grids_all.get('location_grids_st_w', [])

        for storey_key, storey_value in self.info_all_locations_by_storey.items():
            
            # add the non-structural walls.
            all_grids_on_storey = global_st_grids + self.grids_all.get(storey_key, {}).get('location_grids_ns_w', [])

            ids_wall_on_storey = storey_value.get('st_wall_ids',[]) + storey_value.get('ns_wall_ids', [])
            lns_wall_on_storey = storey_value.get('st_wall_lines',[]) + storey_value.get('ns_wall_lines', [])

            for (wall_id, wall_line) in zip(ids_wall_on_storey, lns_wall_on_storey):
                if wall_id not in self.ids_wall_bound:
                    for g_line in all_grids_on_storey:
                        if calculate_line_crosses(g_line, wall_line, ignore_cross_edge, cross_threshold_percent):
                            self.ids_wall_unbound_crossed.add(wall_id)
    
        for wall_id in self.ids_wall_unbound_crossed:
            cross_w_lengths += self.info_wall_length_by_id[wall_id]
        
        # get the percent of total number of unbound walls.
        self.percent_unbound_w_numbers = (1 - len(self.ids_wall_bound) / self.total_wall_numbers) # [0,1], smaller, better
        
        # get the percent of total lengths of crossed unbound walls.
        self.percent_cross_unbound_w_lengths =  (cross_w_lengths/self.total_wall_lengths) # [0,1], smaller, better

    # todo. add this part into the OptimizerGA.
    # a loss about global performance of the grids.
    # @time_decorator
    def calculate_grid_distance_deviation_loss(self, min_size_group=3):
        """
        Calculate the grid distance deviation, requiring at 3 grids.
        |           |               |                   |
        |           |               |                   |
        <- dist_1 -> <-   dist_2   -> <-    dist_3    ->
        - abs(dist_1**2 - dist_2**2)
        - abs(dist_2**2 - dist_3**2)
        """
        
        def perpendicular_distance(point1, point2):
            A = point2.y - point1.y
            B = -(point2.x - point1.x)
            C = point1.x * point2.y - point2.x * point1.y
            return C / math.sqrt(A**2 + B**2)
    
        def get_distance_deviations(grids):

            grid_groups = defaultdict(list)
            square_root_of_distance_differences = []
            
            # calculate the relative locations within a group of grids.
            for ln in grids:
                
                point1, point2 = list(ln.boundary.geoms)[0], list(ln.boundary.geoms)[1]
                slope = get_line_slope_by_points(point1, point2)
                slope = round(slope, 4)
                grid_groups[slope].append([point1, point2, perpendicular_distance(point1, point2)])

            # filter the minor groups that cannot be calculated for the deviation.
            grid_groups = {key: value for key, value in grid_groups.items() if len(value) >= min_size_group}
            
            # check if there's grid_groups left alter filtering the minor groups.
            if grid_groups:

                # sort each group by the perpendicular_distance value
                for slope in grid_groups:
                    grid_groups[slope] = sorted(grid_groups[slope], key=lambda x: x[-1])

                for slope, grid_group in grid_groups.items():
                    for i in range(1, len(grid_group)-1):
                        dist_1 = (grid_group[i][-1] - grid_group[i-1][-1])
                        dist_2 = (grid_group[i+1][-1] - grid_group[i][-1])

                        square_diff = abs(dist_1**2 - dist_2**2)**0.5
                        # ====================================================
                        # problem of overlapping grids_st_c and grids_st_w..
                        # ====================================================
                        square_root_of_distance_differences.append(square_diff)

            return square_root_of_distance_differences
        
        def sigmoid_scale(d, d_max):
            return 1 / (1 + np.exp(-10 * (d / d_max - 0.5)))

        # the loss target.
        self.avg_deviation_distance_st = []
        self.avg_deviation_distance_ns = []

        # get all global st grids.
        global_st_grids = self.grids_all.get('location_grids_st_c', []) + self.grids_all.get('location_grids_st_w', [])

        # the accumulated distance deviation.
        self.avg_deviation_distance_st += get_distance_deviations(global_st_grids)

        for storey_key, storey_value in self.grids_all.items():
            
            if isinstance(storey_value, dict):

                storey_ns_grids = storey_value.get('location_grids_ns_w', [])
                if storey_ns_grids:
                    self.avg_deviation_distance_ns += get_distance_deviations(storey_ns_grids)
        
        # get the averaged rescaled deviation value.
        average_wall_length = sum(list(self.info_wall_length_by_id.values()))/len(list(self.info_wall_length_by_id.values()))
        
        # =============================
        # only test the st_grid relative distance differences for now.
        if self.avg_deviation_distance_st:
            # get average.
            self.avg_deviation_distance_st = sum(self.avg_deviation_distance_st)/len(self.avg_deviation_distance_st)
            # rescale it into [0, 1]
            self.avg_deviation_distance_st = sigmoid_scale(self.avg_deviation_distance_st, average_wall_length)
        else:
            # if not reach any "feasible grid group", return a close-to-1 value by the sigmoid scaling function.
            self.avg_deviation_distance_st = sigmoid_scale(average_wall_length, average_wall_length)
        
        
        # =============================
        # ignore the ns part for now.
        # if self.avg_deviation_distance_ns:
        #     self.avg_deviation_distance_ns = sum(self.avg_deviation_distance_ns)/len(self.avg_deviation_distance_ns)
        # print ("deviation of ns grid distances - loss calculation:", self.avg_deviation_distance_ns)

    # @time_decorator
    def create_grids(self):

        self.get_element_information_for_grid() # - > self.grids_all

        self.extract_grid_overall_borders() # -> self.border_x, self.border_y

        self.calculate_grid_locations() # -> self.grids_all with grid locations.

    # @time_decorator
    def visualization_2d(self):

        # plot_name = f"\[Floor \, Plan \, of \, {storey.Name} \, (T_{{c,dist}}={t_c_dist}, \, T_{{c,num}}={t_c_num}, \, T_{{w,dist}}={t_w_dist}, \, T_{{w,num}}={t_w_num}) - Initial \]"
        # fig_save_name = f"Initial_{storey.Name}_t_c_dist_{t_c_dist}_t_c_num_{t_c_num}_t_w_dist_{t_w_dist}_t_w_num_{t_w_num}"
        
        for storey in self.main_storeys.keys():

            # plotting settings.
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

            # plotting configurations of building elements.
            element_plot_configurations = [
                ('st_column_points', 'square', None),
                ('st_wall_lines', 'line', 'coords'),
                ('ns_wall_lines', 'line', 'coords'),
            ]

            for config in element_plot_configurations:
                data_key, plot_type, attr = config
                element_data = self.info_all_locations_by_storey[storey].get(data_key, []) # per storey.
                
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
                else:
                    continue
                    # raise ValueError("element_plot_configurations dont' lead to correct values.")
            
            # plotting configurations of grids of different types.
            grid_plot_configurations = [
                ('location_grids_st_c', 'line', 'coords'),
                ('location_grids_st_w', 'line', 'coords'),
                ('location_grids_ns_w', 'line', 'coords'),
            ]

            for config in grid_plot_configurations:
                data_key, plot_type, attr = config
                grid_data = self.grids_all[storey].get(data_key, []) # per storey.

                if not grid_data:
                    grid_data = self.grids_all.get(data_key, []) # per building.

                if grid_data:
                    g_plot = self.visualization_settings[data_key]
                    for grid in grid_data:
                        x, y = (grid.x, grid.y) if not attr else getattr(grid, attr).xy
                        
                        if plot_type == 'square':
                            fig.square(x, y, legend_label=g_plot['legend_label'], size=g_plot['size'], 
                                    color=g_plot['color'], alpha=g_plot['alpha'])
                        elif plot_type == 'line':
                            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], 
                                    line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])
                else:
                    continue
                    # raise ValueError("grid_plot_configurations dont' lead to correct values.")

            # Save the figure.
            bokeh.plotting.output_file(filename=os.path.join(self.out_fig_path, fig_save_name + ".html"), title=fig_save_name)
            bokeh.plotting.save(fig)

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