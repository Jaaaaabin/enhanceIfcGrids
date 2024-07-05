import os
import copy
import json
import math
import itertools
from collections import Counter
import numpy as np
import shapely
from shapely.geometry import Point, LineString, MultiPoint
from collections import defaultdict

import bokeh.plotting
from bokeh.io import export_svgs

import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import cairosvg

from toolsQuickUtils import get_line_slope_by_points, remove_duplicate_points, close_parallel_lines, deep_merge_dictionaries, is_close_to_known_slopes, perpendicular_distance
from toolsQuickUtils import time_decorator, check_repeats_in_list, flatten_and_merge_lists, enrich_dict_with_another, calculate_line_crosses, a_is_subtuple_of_b

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

        #----------------------------------
        self.st_c_num = 3
        self.st_w_num = 2
        self.ns_w_num = 2

        self.st_w_accumuled_length_percent = 0.0001 # to be more dynamic
        self.ns_w_accumuled_length_percent = 0.0001 # to be more dynamic

        self.st_st_merge = 0.3
        self.ns_st_merge = 0.3
        self.ns_ns_merge = 0.3

        self.st_c_align_dist = 0.001
        self.st_w_align_dist = 0.1
        self.ns_w_align_dist = 0.1
        
        self.border_x = None
        self.border_y = None

        self.out_fig_path = figure_path
        self.read_infos(json_floors, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls)
        self.init_visualization_settings()
        
        self.ifc_file_name = figure_path.split('\\')[-1]
        self.prefix = self._get_file_prefix_code(self.ifc_file_name)
        print(f"=====================GridGenerator=====================\n{self.ifc_file_name}\n=====================GridGenerator=====================")
            
    def _get_file_prefix_code(self, filename):
        parts = filename.split('-')
        return '-'.join(parts[:2])
    
    # @time_decorator
    def update_parameters(self, new_parameters):
        
        if not isinstance(new_parameters, dict):
            raise ValueError("the input 'new_parameters' must be a dictionary.")
        
        # List of valid parameter names for reference
        valid_parameters = {
            'st_c_num',
            'st_w_num',
            'ns_w_num',
            'st_w_accumuled_length_percent',
            'ns_w_accumuled_length_percent',
            'st_st_merge',
            'ns_st_merge',
            'ns_ns_merge',
            'st_c_align_dist',
            'st_w_align_dist',
            'ns_w_align_dist',
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
            # column points.
            'st_column_points':{
                'legend_label':'Column(S)',
                'color': "#006400",
                'size':5,
                'alpha':0.675,
            },

            # wall lines.
            'st_wall_lines':{
                'legend_label':'Wall(S)',
                'color': "black",
                'line_dash':'solid',
                'line_width':5,
                'alpha':0.675,
            },
            'ns_wall_lines':{
                'legend_label':'Wall(N)',
                'color': "#9a9a9a",
                'line_dash':'solid',
                'line_width':3,
                'alpha':0.675,
            },

            # grid lines.
            'location_grids_st_c': {
                'legend_label':'Grid-Column(S)',
                'color': "#B8860B", 
                'line_dash':'dashed',
                'line_width':2,
                'alpha':0.95,
            },
            'location_grids_st_w': {
                'legend_label': 'Grid-Wall(S)',
                'color': "#bc272d", # red.
                'line_dash':'dashed',
                'line_width':2,
                'alpha':0.95,
            },
            'location_grids_ns_w': {
                'legend_label': 'Grid-Wall(N)',
                'color': "#2e2eb8", # teal.
                'line_dash':'dotted',
                'line_width':2,
                'alpha':0.95,
            },

            # processed grid lines.
            'location_grids_st_merged': {
                'legend_label': 'Grid(S)',
                'color': "#bc272d", # red.
                'line_dash':'dashed',
                'line_width':2.5,
                'alpha':1.0,
            },
            'location_grids_ns_merged': {
                'legend_label':'Grid(N)',
                'color': "#0000a2", # blue
                'line_dash':'dotted',
                'line_width':2.5,
                'alpha':1.0,
            },}
    
#===================================================================================================
# Grid Creation ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 

    # column-related alignment.
    def get_grids_from_column_point_alignments(self, element_pts, element_ids):
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
                if not is_close_to_known_slopes(slope, self.main_directions):
                    continue
                else:
                #|<- - - 
                    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *     
                    for k, (point, id_group_3) in enumerate(zip(uniq_element_points[j + 1:], uniq_element_ids[j + 1: ]), start=j+1):

                        if slope == float('inf'):  # Vertical line check
                            if abs(point.x - point1.x) <= self.st_c_align_dist and abs(point.x - point2.x) <= self.st_c_align_dist:
                                aligned_points.append(point)
                                related_element_ids_reps.append(id_group_3[0])
                                related_element_ids+=id_group_3
                        else:
                            # Use point-slope form of line equation to check alignment
                            if abs((point.y - point1.y) - slope * (point.x - point1.x)) <= self.st_c_align_dist:
                                aligned_points.append(point)
                                related_element_ids_reps.append(id_group_3[0])
                                related_element_ids+=id_group_3
                    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                #|<- - - 

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
                        element_ids_per_grid.append(list(id_tuple))
        
        # todo. clean the points in generated_grids by considering:
        # weights from element_ids_per_grid

        return generated_grids, element_ids_per_grid
    
    def get_grids_from_wall_line_alignments(self, element_lns, element_ids, line_type=[]):
        
        # check if the input data is feasible pairs.
        if len(element_lns)!= len(element_ids):
            return None, None
        
        if not line_type:
            raise ValueError("line_type hasn't been specified for 'get_grids_from_wall_line_alignments'.")
        if line_type == 'structural':
            minimum_accumuled_wall_length_percent = self.st_w_accumuled_length_percent
            minimum_alignment_number = self.st_w_num
            wall_offset_distance = self.st_w_align_dist  # area spanned by the triangle they would form
        elif line_type == 'non-structural':
            minimum_accumuled_wall_length_percent = self.ns_w_accumuled_length_percent
            minimum_alignment_number = self.ns_w_num
            wall_offset_distance = self.ns_w_align_dist # area spanned by the triangle they would form

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
                collinear_points = []
                accumulated_length = []
                accumulated_length.append(length_1)
                aligned_element_ids = {id_1}

                # Sub-case of non-vertical lines. 
                if slope != float('inf'):

                    for j, (ln_new, new_length, id_new) in enumerate(zip(
                        element_lns[i + 1:], ln_lengths[i + 1:], element_ids[i + 1:]), start=i + 1):

                        point3, point4 = list(ln_new.boundary.geoms)[0],list(ln_new.boundary.geoms)[1]
                        are_close, are_collinear = close_parallel_lines(point1, point2, point3, point4, offset=wall_offset_distance)
                        if are_close:
                            collinear_points.append(are_collinear)
                            aligned_points.append([point3,point4])
                            accumulated_length.append(new_length)
                            aligned_element_ids.add(id_new)

                        else:
                            continue

                # Sub-case of vertical lines.
                elif slope == float('inf'):

                    for j, (ln_new, new_length, id_new) in enumerate(zip(
                        element_lns[i + 1:], ln_lengths[i + 1:], element_ids[i + 1:]), start=i + 1):

                        point3, point4 = list(ln_new.boundary.geoms)[0],list(ln_new.boundary.geoms)[1]
                        are_close, are_collinear = close_parallel_lines(point1, point2, point3, point4, offset=wall_offset_distance)
                        if are_close:
                            collinear_points.append(are_collinear)
                            aligned_points.append([point3,point4])
                            accumulated_length.append(new_length)
                            aligned_element_ids.add(id_new)

                        else:
                            continue
                
                aligned_element_ids = sorted(aligned_element_ids)

                # be careful with the hierarchy here.
                # satisfy both criteria?

                if len(aligned_element_ids) >= minimum_alignment_number and \
                    (sum(accumulated_length)/self.total_wall_lengths) >= minimum_accumuled_wall_length_percent:
                    
                    id_tuple = tuple(aligned_element_ids)  # Convert to tuple for hashability

                    if id_tuple not in element_ids_per_grid:
                        
                        # the case when it's already there.
                        if element_ids_per_grid and any([
                            set(id_tuple).issubset(set(existing_ids_component)) for existing_ids_component in element_ids_per_grid]):
                            continue
                        
                        # the case when it's not there yet.
                        else:
                            if not all(collinear_points):
                                # easiest option: reserve only the longest one.
                                for id, l in enumerate(accumulated_length):
                                    if l == max(accumulated_length):
                                        aligned_points = [aligned_points[id]]
                                        break
                                    else:
                                        continue

                            generated_grids.append(aligned_points)
                            element_ids_per_grid.append(list(set(aligned_element_ids)))
                            
        generated_grids = [[e for element in elements for e in element] for elements in generated_grids]
        generated_grids = [remove_duplicate_points(elements) for elements in generated_grids]

        # todo. clean the points in generated_grids by considering:
        # weights from element_ids_per_grid

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
        # by the whole building.
        # ------------------
        st_column_points = self.info_all_locations.get('st_column_points')
        st_column_ids = self.info_all_locations.get('st_column_ids')

        if st_column_points and st_column_ids:
            grids_st_c, grids_st_c_ids = self.get_grids_from_column_point_alignments(st_column_points, st_column_ids)
        
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
        # by the whole building.
        # ------------------
        st_wall_lines = self.info_all_locations.get('st_wall_lines')
        st_wall_ids = self.info_all_locations.get('st_wall_ids')

        if st_wall_lines and st_wall_ids:
            grids_st_w, grids_st_w_ids = self.get_grids_from_wall_line_alignments(st_wall_lines, st_wall_ids,line_type='structural')
        
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
        # by every storey.
        # ------------------
        for key, value in self.info_all_locations_by_storey.items():

            # ns_wall
            ns_wall_lines = value.get('ns_wall_lines', []) # including the case without ns.
            ns_wall_ids = value.get('ns_wall_ids', [])
            
            # curtain wall
            ct_wall_lines = value.get('ct_wall_lines', []) # including the case without ct. 
            ct_wall_ids = value.get('ct_wall_ids', [])
            
            ns_wall_lines += ct_wall_lines 
            ns_wall_ids += ct_wall_ids

            if ns_wall_lines and ns_wall_ids:
                grids_ns_w, grids_ns_w_ids = self.get_grids_from_wall_line_alignments(ns_wall_lines, ns_wall_ids,line_type='non-structural')
            
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

        self.main_storeys_from_ifc_columns = {round(elevation,2): {"columns": columns}
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

        self.main_storeys_from_ifc_walls = {round(elevation,2): {"walls": walls}
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
        
            self.main_storeys_from_ifc_ct_walls = {round(elevation,2): {"curtain walls": walls}
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

        self.main_storeys_from_ifc_floors = {round(elevation,2): {"floors": floors}
                                         for elevation, floors in floors_by_elevation.items() if len(floors) >= num_floors}

    def get_main_storeys_init(self, include_ct_walls=False):

        # here we take floor and wall elements to determine the initial main storeys.
        self.get_main_storeys_with_columns()
        self.get_main_storeys_with_floors()
        self.get_main_storeys_with_walls(include_ct_walls=include_ct_walls)

        # column + walls 
        self.main_storeys = deep_merge_dictionaries(self.main_storeys_from_ifc_columns, self.main_storeys_from_ifc_walls)
        # # + floors.
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
        
        # # refine step 1: remove the storeys that doesn't have any attached floors.
        # self.main_storeys =  {key: value for key, value in self.main_storeys.items() if 'floors' in value.keys()}
        
        # (old) refine step 2: only keep the storeys that 'either have walls' or 'either have columns' or 'have both walls and columns'.
        # self.main_storeys =  {key: value for key, value in self.main_storeys.items() if 'columns' in value.keys() or 'walls' in value.keys()}
        
        # refine step 2: only keep the storeys that  have walls
        self.main_storeys =  {key: value for key, value in self.main_storeys.items() if 'walls' in value.keys()}

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
        
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # collect all ids per storey.

        for key, sub_dict in self.info_all_locations_by_storey.items():
            
            # get all sub keys with 'ids'
            all_ids_per_storey = []
            sub_keys = [sub_key for sub_key in sub_dict.keys() if '_ids' in sub_key]

            if sub_keys:
                for sub_key in sub_keys:
                    all_ids_per_storey += sub_dict[sub_key]

            self.info_all_locations_by_storey[key].update({
                'all_ids':all_ids_per_storey
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
        
        # step1: calculate the st grids for the whole building.
        for grid_type_key in st_grid_terms:

            grid_location_key = 'location_' + grid_type_key
            elements_for_grids = self.grids_all.get(grid_type_key)

            if elements_for_grids:
                self.grids_all[grid_location_key] = self.generate_grids_from_candidate_points(elements_for_grids)
            else:
                self.grids_all[grid_location_key] = []

        ns_grids_by_storey = {key: self.grids_all[key] for key in self.main_storeys.keys()}

        # step2: calculate the ns grids per storey.
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
        
        # step3: related the st grids to each storey.        
        for storey_key, storey_value in self.grids_all.items():
            
            # if it's a storey key.    
            if storey_key in self.info_all_locations_by_storey.keys():
                
                # for grids_st_c.
                global_location_grids_st, global_grids_st_ids = self.grids_all['location_grids_st_c'], self.grids_all['grids_st_c_ids']
                self.grids_all[storey_key]['location_grids_st_c'], self.grids_all[storey_key]['grids_st_c_ids'] = self.relate_st_grids_to_each_storey(
                    storey_key=storey_key, global_location_grids_st=global_location_grids_st, global_grids_st_ids=global_grids_st_ids)
                
                # for grids_st_c.
                global_location_grids_st, global_grids_st_ids = self.grids_all['location_grids_st_w'], self.grids_all['grids_st_w_ids']
                self.grids_all[storey_key]['location_grids_st_w'], self.grids_all[storey_key]['grids_st_w_ids'] = self.relate_st_grids_to_each_storey(
                    storey_key=storey_key, global_location_grids_st=global_location_grids_st, global_grids_st_ids=global_grids_st_ids)
            else:
                continue

    def prepare_wall_total_lengths_numbers(self):

        # IfcWall and IfcCurtainWall.
        total_length = 0
        for item in self.info_all_walls:
            if "length" in item:
                total_length += item["length"]

        # some global wall information
        self.total_wall_lengths = total_length
        self.info_wall_length_by_id = {d['id']: d['length'] for d in self.info_all_walls if 'id' in d and 'length' in d}

        # count all the walls and columns.
        self.total_wall_numbers = len(self.info_all_walls)
        self.total_column_numbers = len(self.info_st_columns)
        self.total_canbound_element_numbers = self.total_wall_numbers + self.total_column_numbers
        
        self.total_canbound_elements = self.info_all_walls + self.info_st_columns
        self.id_canbound_elements = set([elem['id'] for elem in self.total_canbound_elements])

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
    
    # @time_decorator
    def create_grids(self):

        self.get_element_information_for_grid() # - > self.grids_all

        self.extract_grid_overall_borders() # -> self.border_x, self.border_y

        self.calculate_grid_locations() # -> self.grids_all with grid locations.


# Grid Creation ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
# Grid Merge ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 
    
    # @time_decorator
    def merge_same_grids(self, grid_linestrings, grid_componnets, tol=0.0, slope_tol=0.01):

        # the long running time coming from grids from non-main directions.
        # find all the pairs
        merged_id_pairs = []

        for i, gd_ln_1 in enumerate(grid_linestrings):

            for j, gd_ln_2 in enumerate(grid_linestrings):
        
                # combination pairs (i,j)
                if i < j:

                    slope_1 = get_line_slope_by_points(list(gd_ln_1.boundary.geoms)[0], list(gd_ln_1.boundary.geoms)[1])
                    slope_2 = get_line_slope_by_points(list(gd_ln_2.boundary.geoms)[0], list(gd_ln_2.boundary.geoms)[1])

                    # only consider merging among parallel lines.
                    if abs(slope_1-slope_2) <= slope_tol or slope_1==slope_2==float('inf'):
                    
                        # if "close enough"
                        if shapely.distance(gd_ln_1,gd_ln_2) < tol:
                            merged_id_pairs.append([i,j])
                            
                    else:
                        continue
                else:
                    continue

        # count and prioritize the merge orders by counting the number of merging connections.
        id_frequency = Counter([item for sublist in merged_id_pairs for item in sublist])
        sorted_id_by_occurency = [item for item, count in id_frequency.most_common()]
        
        # merge maps. 
        merge_maps = {}
        merge_built_ids = []

        for id_host in sorted_id_by_occurency:
    
            # iterate from the high prioritized 'id_host'.    
            counted_merge_ids = [item for sublist in list(merge_maps.values()) for item in sublist] 
            
            if counted_merge_ids:
                
                # the grid with id_host is already aligned with other grid.
                if id_host in counted_merge_ids:
                    continue # skip.

            ids_guest = []
            # the grid with 'id_host' is not yet aligned with other grids, thus, identify all potential guest ids.
            for jj, id_paris in enumerate(merged_id_pairs):

                if jj not in merge_built_ids and id_host in id_paris:

                    new_guest_id = [item for item in id_paris if item != id_host][0]
                    
                    if new_guest_id not in counted_merge_ids:
                        ids_guest.append(new_guest_id)
                        merge_built_ids.append(jj)
            
            if ids_guest:
                merge_maps.update({id_host: ids_guest})
        
        all_grids_marked_merged = list(set([item for sublist in list(merge_maps.values()) for item in sublist]))

        # add the guest grid_st(s) to the host grid_st.
        for gd_host, gd_guests in merge_maps.items():
            for gd_guest in gd_guests:
                grid_componnets[gd_host] += grid_componnets[gd_guest]
        
        # delete the "moved" grid_st(s).
        location_grids_st = copy.deepcopy(grid_linestrings)
        grids_st_ids = copy.deepcopy(grid_componnets)
        for jj in sorted(all_grids_marked_merged, reverse=True):
            del location_grids_st[jj]
            del grids_st_ids[jj]

        return location_grids_st, grids_st_ids
    
    # @time_decorator
    def merge_ns2st_grids(self, ns_grids, ns_grids_ids, tol=0.0, slope_tol=0.01):

        merge_maps = []
        all_grids_marked_merged = []

        for ii, gd_ln_st in enumerate(self.grids_merged['location_grids_st_merged']):

            for jj, gd_ln_ns in enumerate(ns_grids):
                
                # if not merged yet with structural grids.
                if jj not in all_grids_marked_merged:

                    slope_st = get_line_slope_by_points(list(gd_ln_st.boundary.geoms)[0], list(gd_ln_st.boundary.geoms)[1])
                    slope_ns = get_line_slope_by_points(list(gd_ln_ns.boundary.geoms)[0], list(gd_ln_ns.boundary.geoms)[1])

                    # only consider merge among parallel lines.
                    if abs(slope_st-slope_ns) <= slope_tol or slope_st==slope_ns==float('inf'):
                        
                        
                        if shapely.distance(gd_ln_st,gd_ln_ns) < tol:
                    
                            # store the merge maps.
                            merge_maps.append([ii,jj])
                            # marked as merged.
                            all_grids_marked_merged.append(jj)

                    else:
                        continue
                
                else:
                    continue
        
        # align the grid_ns to grid_st.
        # 'location_grids_st_merged' and 'grids_st_merged_ids' are updated here.
        for [ii,jj] in merge_maps:
            if ii < len(self.grids_merged['grids_st_merged_ids']) and jj < len(ns_grids_ids):
                self.grids_merged['grids_st_merged_ids'][ii] += ns_grids_ids[jj]

        # delete the initial grid_ns.
        location_grids_ns_w_storey = copy.deepcopy(ns_grids)
        grids_ns_w_ids_storey = copy.deepcopy(ns_grids_ids)
        for jj in sorted(all_grids_marked_merged, reverse=True):
            del location_grids_ns_w_storey[jj]
            del grids_ns_w_ids_storey[jj]

        return location_grids_ns_w_storey, grids_ns_w_ids_storey
    
    # @time_decorator
    def relate_st_grids_to_each_storey(self, storey_key, global_location_grids_st, global_grids_st_ids):

        location_grids_st_per_storey, grids_st_ids_per_storey = [], []
        all_element_ids_per_storey = self.info_all_locations_by_storey[storey_key]['all_ids']
        related_iis = []

        if global_location_grids_st and global_grids_st_ids and all_element_ids_per_storey:
            for ii, st_ids in enumerate(global_grids_st_ids):
                if bool(set(st_ids) & set(all_element_ids_per_storey)):
                    related_iis.append(ii)
                else:
                    continue
    
        if related_iis:
            location_grids_st_per_storey = [global_location_grids_st[i] for i in related_iis]
            grids_st_ids_per_storey = [global_grids_st_ids[i] for i in related_iis]
            
        return location_grids_st_per_storey, grids_st_ids_per_storey
    
    # @time_decorator
    def merge_grids(self):
        
        self.grids_merged = defaultdict(list)

        # ----------------------------------------------
        # step1: merge the st grids
        # take all information from 'grids_all'
        all_st_grids = self.grids_all['location_grids_st_c'] + self.grids_all['location_grids_st_w']
        all_st_grids_ids = self.grids_all['grids_st_c_ids'] + self.grids_all['grids_st_w_ids']

        # location_grids_st_merged, grids_st_merged_ids are created in 'grids_merged' per building.
        self.grids_merged['location_grids_st_merged'], self.grids_merged['grids_st_merged_ids'] = self.merge_same_grids(
            all_st_grids, all_st_grids_ids, tol = self.st_st_merge)

        # ----------------------------------------------
        # step2: merge ns to st grids per storey: location_grids_ns_merged, grids_ns_merged_ids
        # take 'location_grids_ns_w' and 'grids_ns_w_ids' from 'grids_all'
        # take 'location_grids_st_merged' and 'grids_st_merged_ids' from 'grids_merged'
        for storey_key, storey_value in self.grids_all.items():

            if 'grids_ns_w_ids' in storey_value:
                # 'location_grids_ns_merged' and 'grids_ns_merged_ids' are updated per (main) storey.
                self.grids_merged[storey_key] = {}
                self.grids_merged[storey_key]['location_grids_ns_merged'], self.grids_merged[storey_key]['grids_ns_merged_ids'] = self.merge_ns2st_grids(
                    storey_value['location_grids_ns_w'], storey_value['grids_ns_w_ids'], tol = self.ns_st_merge)
            else:
                continue
        
        # ----------------------------------------------
        # step3: merge the ns grids per storey:
        # take 'location_grids_ns_merged' and 'grids_ns_merged_ids' from 'grids_merged'
        for storey_key, storey_value in self.grids_merged.items():

            if 'grids_ns_merged_ids' in storey_value:
                # 'location_grids_ns_merged' and 'grids_ns_merged_ids' are updated per (main) storey.
                self.grids_merged[storey_key]['location_grids_ns_merged'], self.grids_merged[storey_key]['grids_ns_merged_ids'] = self.merge_same_grids(
                    storey_value['location_grids_ns_merged'], storey_value['grids_ns_merged_ids'], tol = self.ns_ns_merge)

            else:
                continue

        # ----------------------------------------------
        # step4: distribute the global st grids to all storeys:
        for storey_key, storey_value in self.grids_merged.items():
            
            # if it's a storey key.    
            if storey_key in self.info_all_locations_by_storey.keys():
                
                global_location_grids_st = self.grids_merged['location_grids_st_merged']
                global_grids_st_ids = self.grids_merged['grids_st_merged_ids']
                self.grids_merged[storey_key]['location_grids_st_merged'], self.grids_merged[storey_key]['grids_st_merged_ids'] = self.relate_st_grids_to_each_storey(
                    storey_key=storey_key, global_location_grids_st=global_location_grids_st, global_grids_st_ids=global_grids_st_ids)
            else:
                continue
    
# Grid Merge ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
# Grid Analyses  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    def analyze_grids(self):
        
        self.extract_grid_relationships()
        self.calculate_grid_distribution()
        self.extract_grid_neighbors()
        self.summarize_bounding_results()

    def extract_grid_relationships(self):
        
        self.grids_relationships = defaultdict(list)
        gds, gd_types, gd_ids, gd_storeys = [], [], [], []

        for storey_key, grids_per_storey in self.grids_merged.items():
            if storey_key in self.info_all_locations_by_storey:
                for grid_type in ['st', 'ns']:
                    location_key, ids_key = f'location_grids_{grid_type}_merged', f'grids_{grid_type}_merged_ids'
                    if location_key in grids_per_storey and ids_key in grids_per_storey:
                        for gd, ids in zip(grids_per_storey[location_key], grids_per_storey[ids_key]):
                            gd_storeys.append(storey_key)
                            gd_types.append(grid_type)
                            gds.append(gd)
                            gd_ids.append(ids)

        sublist_dict = defaultdict(list)
        for index, sublist in enumerate(gd_ids):
            sublist_dict[tuple(sublist)].append(index)
        grid_groups = list(sublist_dict.values())

        for i, grid_group in enumerate(grid_groups, start = 1):
            grid_type = gd_types[grid_group[0]]
            grid_ids = gd_ids[grid_group[0]]
            grid_location_line = gds[grid_group[0]]
            grid_location_points = list(grid_location_line.boundary.geoms)[:2]
            grid_location = [[round(pt.x, 4), round(pt.y, 4)] for pt in grid_location_points]
            grid_storeys = [gd_storeys[idx] for idx in grid_group]

            self.grids_relationships[i] = {
                'type': grid_type,
                'storey': grid_storeys,
                'location': grid_location,
                'ids': grid_ids,
            }

        if self.grids_relationships:
            
            # sort them by the number of related value['storey'] and value['ids'].
            sorted_grids_relationships = dict(
                sorted(
                    self.grids_relationships.items(),
                    key=lambda item: (len(item[1]['storey']), len(item[1]['ids'])),
                    reverse=True)) # in a descending order
            self.grids_relationships = {i + 1: v for i, (k, v) in enumerate(sorted_grids_relationships.items())}

    def calculate_grid_distribution(self):

        for storey_key in self.grids_merged.keys():
                    
            # iterate through all storeys.
            if storey_key in self.info_all_locations_by_storey:
                
                # get all involved grids per storey.
                involved_grids_indices = [index for index in self.grids_relationships.keys() if storey_key in self.grids_relationships[index]['storey']]
                grids_per_storey = {k: v for k, v in self.grids_relationships.items() if k in involved_grids_indices}

                # calculate the relative locations within a group of grids.
                grid_groups_per_storey = defaultdict(list)
                for grid_index, grid_values in grids_per_storey.items():

                    if len(grid_values['location']) != 2:
                        raise ValueError(f"Grid {grid_index} location does not contain exactly two points.")
                        
                    slope = get_line_slope_by_points(grid_values['location'][0], grid_values['location'][1])
                    grid_groups_per_storey[slope].append([
                        grid_index, # grid index,.
                        grid_values['location'][0], # location of endpoint 1.
                        grid_values['location'][1], # location of endpoint 2.
                        perpendicular_distance(grid_values['location'][0], grid_values['location'][1]) # perpendicular distance.
                        ])
                
                # check if there's grid_groups_per_storey achieved.
                if grid_groups_per_storey:

                    # sort each group by the perpendicular_distance value
                    for slope in grid_groups_per_storey:
                        grid_groups_per_storey[slope] = sorted(grid_groups_per_storey[slope], key=lambda x: x[-1]) # sort them via the perpendicular distance.
                
                self.grids_merged[storey_key].update({
                    'grid_groups': grid_groups_per_storey
                })
                
            else:
                continue

    def extract_grid_neighbors(self):

        def find_neighbors_for_index(group_data, index):

            position = None
            for i, item in enumerate(group_data):
                if item[0] == index:
                    position = i
                    break
                    
            # Initialize neighbors list
            neighbors = {}

            if position is not None:

                # Check the row before the given index
                if position > 0:
                    prev_item = group_data[position - 1]
                    prev_index = prev_item[0]
                    prev_diff = prev_item[3] - group_data[position][3]
                    neighbors.update(
                        {prev_index: prev_diff})
                
                # Check the row after the given index
                if position < len(group_data) - 1:
                    next_item = group_data[position + 1]
                    next_index = next_item[0]
                    next_diff = next_item[3] - group_data[position][3]
                    neighbors.update(
                        {next_index: next_diff})

            return neighbors

        for grid_index, grid_values in self.grids_relationships.items():
            
            self.grids_relationships[grid_index]['neighbor'] = {}
            for st in grid_values['storey']:
                if self.grids_merged[st].get('grid_groups', []):
                    for grid_group_values in self.grids_merged[st]['grid_groups'].values():
                        neighbor_indices = find_neighbors_for_index(grid_group_values, grid_index)
                        if not neighbor_indices:
                            # if the grid is not in the current grid_group:
                            continue
                        else:
                            # if it's found.
                            self.grids_relationships[grid_index]['neighbor'].update({
                                st: neighbor_indices
                            })

        try:
            with open(os.path.join(self.out_fig_path, 'info_grid_relationships.json'), 'w') as json_file:
                json.dump(self.grids_relationships, json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'info_grid_relationships.json'}: {e}")
    
    # also a loss function for merged grids.
    def summarize_bounding_results(self):

        # the loss target.
        # get all walls bound to grids.
        self.ids_elements_bound = self.grids_merged['grids_st_merged_ids']

        # get the numbers of ids bound to each grid.
        num_bound_ids_per_grid = []
        if self.grids_merged['grids_st_merged_ids']:
            num_bound_ids_per_grid += [len(sublist) for sublist in self.grids_merged['grids_st_merged_ids']]

        for storey_key, storey_value in self.grids_merged.items():
            if 'grids_ns_merged_ids' in storey_value:
                self.ids_elements_bound += storey_value['grids_ns_merged_ids']
                num_bound_ids_per_grid += [len(sublist) for sublist in storey_value['grids_ns_merged_ids']]

        self.ids_elements_bound = set([item for sublist in self.ids_elements_bound for item in sublist])
        self.ids_elements_unbound = self.id_canbound_elements.difference(self.ids_elements_bound)
        self.percent_unbound_elements = (1 - len(self.ids_elements_bound) / self.total_canbound_element_numbers) # [0,1] to minimize
        
        if num_bound_ids_per_grid:
            reversed_num_bound_ids_per_grid = [1/num for num in num_bound_ids_per_grid]
            self.avg_reversed_num_bound_ids_per_grid = sum(reversed_num_bound_ids_per_grid)/len(reversed_num_bound_ids_per_grid) # [0,1] to minimize
        else:
            self.avg_reversed_num_bound_ids_per_grid = 1.0
        
        # write the unbound elements.
        try:
            with open(os.path.join(self.out_fig_path, 'info_non_relationships.json'), 'w') as json_file:
                json.dump(list(self.ids_elements_unbound), json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'info_non_relationships.json'}: {e}")

# Grid Analyses  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#=================================================================================================== 

#===================================================================================================
# Loss with Merged Grids ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    def calculate_merged_losses(self):
        
        self.merged_loss_maxmin_deviation()
        self.merged_loss_distance_deviation()
    
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    # a loss about global performance of the grids.
    # to minimize.
    def merged_loss_maxmin_deviation(self):
        """
        Calculate the grid distance deviation, requiring at 3 grids.
        |           |               |                   |
        |           |               |                   |
        <- dist_1 -> <-   dist_2   -> <-    dist_3    ->
        |           |               |                   |
        |           |               |                   |
        - max = dist_3
        - min = dist_1
        """

        def get_maxmin_deviations(grid_groups, min_size_group=3):

            max_min_deviations = []
            grid_groups = {key: value for key, value in grid_groups.items() if len(value) >= min_size_group}

            # check if there's grid_groups left alter filtering the minor groups.
            if grid_groups:
                
                for slope, grid_group in grid_groups.items():
                    distance_per_group = []
                    for i in range(1, len(grid_group)):
                        dist = (grid_group[i][-1] - grid_group[i-1][-1])
                        distance_per_group.append(dist)

                    if distance_per_group:
                        max_min_distances_per_group = [max(distance_per_group), min(distance_per_group)]                        
                        max_min_deviations.append((1-max_min_distances_per_group[1]/max_min_distances_per_group[0]))
            
            return max_min_deviations
    
        # initialize the loss target.
        self.avg_deviation_maxmin = []
        for storey_key, storey_value in self.grids_merged.items():
            if isinstance(storey_value, dict) and storey_value.get('grid_groups'):
                grid_groups_per_storey = storey_value['grid_groups']
                self.avg_deviation_maxmin.append(get_maxmin_deviations(grid_groups_per_storey))
        
        self.avg_deviation_maxmin = [item for sublist in self.avg_deviation_maxmin for item in sublist]
        if self.avg_deviation_maxmin and len(self.avg_deviation_maxmin)!=0:
            self.avg_deviation_maxmin = sum(self.avg_deviation_maxmin)/len(self.avg_deviation_maxmin)
        else:
            print ("|---------------------------------->>>Warning: One penalty pic occurs in merged_loss_maxmin_deviation |")
            self.avg_deviation_maxmin = 1.0
        
        # print ("avg_deviation_maxmin", self.avg_deviation_maxmin) # might have errors after the updates
        
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    # a loss about global performance of the grids.
    # to minimize.
    def merged_loss_distance_deviation(self):
        """
        Calculate the grid distance deviation, requiring at 3 grids.
        |           |               |                   |
        |           |               |                   |
        <- dist_1 -> <-   dist_2   -> <-    dist_3    ->
        |           |               |                   |
        |           |               |                   |
        - abs(dist_1**2 - dist_2**2)
        - abs(dist_2**2 - dist_3**2)
        """
        
        def get_distance_deviations(grid_groups, min_size_group=3):

            square_root_of_distance_differences = []
            grid_groups = {key: value for key, value in grid_groups.items() if len(value) >= min_size_group}
            
            if grid_groups:
                for slope, grid_group in grid_groups.items():
                    for i in range(1, len(grid_group)-1):
                        dist_1 = (grid_group[i][-1] - grid_group[i-1][-1])
                        dist_2 = (grid_group[i+1][-1] - grid_group[i][-1])

                        square_root_diff = abs(dist_1**2 - dist_2**2)**0.5
                        square_root_of_distance_differences.append(square_root_diff)

            return square_root_of_distance_differences
        
        def sigmoid_scale(d, d_max):
            return 1 / (1 + np.exp(-10 * (d / d_max - 0.5)))
        
        # todo, reconsider if there's any better scale function.
        # get the averaged rescaled deviation value. 
        average_wall_length = sum(list(self.info_wall_length_by_id.values()))/len(list(self.info_wall_length_by_id.values()))
        
        # initialize the loss target.
        self.avg_deviation_distance = []
        for storey_key, storey_value in self.grids_merged.items():
            if isinstance(storey_value, dict) and storey_value.get('grid_groups'):
                grid_groups_per_storey = storey_value['grid_groups']
                self.avg_deviation_distance += get_distance_deviations(grid_groups_per_storey)

        if self.avg_deviation_distance:
            self.avg_deviation_distance = sum(self.avg_deviation_distance)/len(self.avg_deviation_distance)
            self.avg_deviation_distance = sigmoid_scale(self.avg_deviation_distance, average_wall_length)
        else:
            print ("|---------------------------------->>>Warning: One penalty pic occurs in merged_loss_distance_deviation |")
            self.avg_deviation_distance = sigmoid_scale(average_wall_length, average_wall_length)

        # print ("avg_deviation_distance", self.avg_deviation_distance) # might have errors after the updates
        
# Loss with Merged Grids ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ 
#===================================================================================================

#===================================================================================================
# Visualization ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    def _save_svg_chromedriver(self, fig, output_svg_path):

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        service = Service(chromedriver_binary.chromedriver_filename)
        driver = webdriver.Chrome(service=service, options=options)
        
        fig.output_backend = "svg"
        export_svgs(fig, filename=output_svg_path, webdriver=driver)
        driver.quit()

    def _convert_svg_to_pdf(self, svg_path, pdf_path):
        
        cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
        
    # @time_decorator
    def visualization_2d_before_merge(self, visualization_storage_path=None, add_strs=''):

        if visualization_storage_path is None:
            visualization_storage_path = self.out_fig_path
            
        for storey in self.main_storeys.keys():

            # plotting settings.
            plot_name = f"Elevation {str(round(storey, 4))} - before merging"
            fig_save_name = f"{self.prefix}_Elevation_{str(round(storey,4))}_creation" if not add_strs else \
                f"{self.prefix}_Elevation_{str(round(storey,4))}_creation_{add_strs}"
            fig = bokeh.plotting.figure(
                title=plot_name,
                title_location='above',
                x_axis_label='x',
                y_axis_label='y',
                width=800,
                height=800,
                match_aspect=True)
            
            # plotting tunning 
            fig.title.align = 'center'
            fig.title.text_font_size = '14pt'
            fig.xgrid.visible = False
            fig.ygrid.visible = False
            fig.xaxis.axis_label_standoff = -20
            fig.yaxis.axis_label_standoff = -25
            fig.xaxis.major_label_standoff = -20
            fig.yaxis.major_label_standoff = -20

            # plotting configurations of building elements.
            element_plot_configurations = [
                ('ns_wall_lines', 'line', 'coords'),
                ('st_wall_lines', 'line', 'coords'),
                ('st_column_points', 'square', None),
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
                ('location_grids_ns_w', 'line', 'coords'),
                ('location_grids_st_w', 'line', 'coords'),
                ('location_grids_st_c', 'line', 'coords'),
            ]

            for config in grid_plot_configurations:

                data_key, plot_type, attr = config
                grid_data = self.grids_all[storey].get(data_key, []) # per storey.

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

            # Adjust the legend settings.
            legend = fig.legend[0]
            legend.location = "top_center"
            legend.orientation = "horizontal"
            legend.spacing = 10
            legend.padding = 10
            legend.margin = 0
            legend.label_text_font_size = "12pt"

            # Save the figure.
            svg_file_path = os.path.join(visualization_storage_path, fig_save_name + ".svg")
            pdf_file_path = os.path.join(visualization_storage_path, fig_save_name + ".pdf")

            self._save_svg_chromedriver(fig, output_svg_path = svg_file_path)
            self._convert_svg_to_pdf(svg_file_path, pdf_file_path)
        
            # bokeh.plotting.output_file(filename=os.path.join(self.out_fig_path, fig_save_name + ".html"), title=fig_save_name)
            # bokeh.plotting.save(fig)

    # @time_decorator
    def visualization_2d_after_merge(self, visualization_storage_path=None, add_strs=''):
        
        if visualization_storage_path is None:
            visualization_storage_path = self.out_fig_path

        for storey in self.main_storeys.keys():

            # plotting settings.
            plot_name = f"Elevation {str(round(storey, 4))} - after merging"
            fig_save_name = f"{self.prefix}_Elevation_{str(round(storey,4))}_merging" if not add_strs else \
                f"{self.prefix}_Elevation_{str(round(storey,4))}_merging_{add_strs}"
            fig = bokeh.plotting.figure(
                title=plot_name,
                title_location='above',
                x_axis_label='x',
                y_axis_label='y',
                width=800,
                height=800,
                match_aspect=True)
            
            # plotting tunning 
            fig.title.align = 'center'
            fig.title.text_font_size = '14pt'
            fig.xgrid.visible = False
            fig.ygrid.visible = False
            fig.xaxis.axis_label_standoff = -20
            fig.yaxis.axis_label_standoff = -25
            fig.xaxis.major_label_standoff = -20
            fig.yaxis.major_label_standoff = -20

            # plotting configurations of building elements.
            element_plot_configurations = [
                ('ns_wall_lines', 'line', 'coords'),
                ('st_wall_lines', 'line', 'coords'),
                ('st_column_points', 'square', None),
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
                ('location_grids_ns_merged', 'line', 'coords'),
                ('location_grids_st_merged', 'line', 'coords'),
            ]

            for config in grid_plot_configurations:

                data_key, plot_type, attr = config
                grid_data = self.grids_merged[storey].get(data_key, []) # per storey.

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

            # Adjust the legend settings.
            legend = fig.legend[0]
            legend.location = "top_center"
            legend.orientation = "horizontal"
            legend.spacing = 10
            legend.padding = 10
            legend.margin = 0
            legend.label_text_font_size = "12pt"

            # Save the figure.
            svg_file_path = os.path.join(visualization_storage_path, fig_save_name + ".svg")
            pdf_file_path = os.path.join(visualization_storage_path, fig_save_name + ".pdf")

            self._save_svg_chromedriver(fig, output_svg_path = svg_file_path)
            self._convert_svg_to_pdf(svg_file_path, pdf_file_path)
            

#Visualization ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ 
#===================================================================================================