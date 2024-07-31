import ifcopenshell
from ifcopenshell.api import run
import math
import json
import os
import numpy as np
from collections import defaultdict
from toolsQuickUtils import time_decorator

class IfcSpatialGridEnrichment:

    def __init__(self, model_path, figure_path):

        self.model = ifcopenshell.open(model_path)
        self.ifc_file_name = os.path.basename(model_path)
        self.output_figure_path = figure_path

        # Initial setup.
        self.mode_unit = 1.0 # unit = 0.001 * meter
        self.initialize_ifc_structure()
        
    def initialize_ifc_structure(self):

        self.project = self.model.by_type('IfcProject')[0]
        self.building = self.model.by_type('IfcBuilding')[0]
        self.storeys = self.model.by_type('IfcBuildingStorey')

        max_z_values = max(st.Elevation for st in self.storeys)
        if max_z_values > 1000:
            self.mode_unit = 0.001  # unit = 0.001 * meter
        self.elev_storeys = {round(st.Elevation * self.mode_unit, 1): st for st in self.storeys}

    def enrich_grid_placement_information(self):
         
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
            
        def map_grid_direction_groups(degree,t_degree=0.01):
            if abs(degree)<t_degree:
                return 'U'
            elif abs(degree-90)<t_degree:
                return 'V'
            else:
                return 'W'
            
        self.all_grid_data = read_json_file(os.path.join(self.output_figure_path, 'hierarchical_data.json'))
        self.grid_placements = defaultdict(list)
        for grid_label, grid_data in self.all_grid_data.items():
            if grid_data["type"] in {"st_grid", "ns_grid"}:
                direction_group = map_grid_direction_groups(grid_data["plane_direction"])
                self.grid_placements[direction_group].append((grid_label, grid_data))
        
        if 'W' not in self.grid_placements:
            self.grid_placements['W'] = []

        for axis_direction, axis_grid_info in self.grid_placements.items():
            for axis_label, grid_info in axis_grid_info:
                radian = math.radians(grid_info['plane_direction'])
                grid_axis_IfcDirection = (math.cos(radian), math.sin(radian), 0.0)
                grid_info['location'] = (np.array(grid_info['location'])/self.mode_unit).tolist()
                grid_axis_IfcCoordinates = [grid_info['location'][0][:2], grid_info['location'][1][:2]]
                grid_axis_IfcStoreys = [self.elev_storeys[z*self.mode_unit] for z in grid_info['location'][0][2:]]
            
                grid_info.update({
                    'IfcDirection': grid_axis_IfcDirection,
                    'IfcCoordinates': grid_axis_IfcCoordinates,
                    'IfcStoreys': grid_axis_IfcStoreys,
                })

    def _create_grid_axis(self, label, info, SameSense_axes=True):
            
        point_list = self.model.create_entity("IfcCartesianPointList2D", CoordList=info['IfcCoordinates'])
        indexed_curve = self.model.create_entity("IfcIndexedPolyCurve", Points=point_list, SelfIntersect=False)
        
        return self.model.create_entity(
            'IfcGridAxis',
            AxisTag=label,
            AxisCurve=indexed_curve,
            SameSense=SameSense_axes,
        )
    
    def _add_grid_in_spatial_structures(self, grid, contained_in_storeys):

        for st in contained_in_storeys:

            existing_relation = None
            for relation in st.ContainsElements:
                if relation.is_a("IfcRelContainedInSpatialStructure"):
                    existing_relation = relation
                    break
                        
            if existing_relation:
                # Add the new grid to the existing relationship
                updated_related_elements = existing_relation.RelatedElements + (grid,)
                existing_relation.RelatedElements = updated_related_elements
            else:
                self.model.create_entity(
                    "IfcRelContainedInSpatialStructure",
                    GlobalId=ifcopenshell.guid.new(),
                    RelatingStructure=st,
                    RelatedElements=[grid])

    def create_reference_grids(self):

        axis_mapping = {'U': 'UAxes','V': 'VAxes', 'W': 'WAxes'}
        
        for axis_direction, axis_grid_info in self.grid_placements.items():
            for axis_label, grid_info in axis_grid_info:
                new_grid = self.model.create_entity("IfcGrid", GlobalId=ifcopenshell.guid.new(), Name=axis_direction+'_'+axis_label)
                self.all_grid_data[axis_label].update({'id':new_grid.GlobalId})
                same_sense_on_this_axis  = axis_direction != 'W'
                new_grid_axis = self._create_grid_axis(label=axis_label, info=grid_info, SameSense_axes=same_sense_on_this_axis)

                if axis_direction in axis_mapping:
                    setattr(new_grid, axis_mapping[axis_direction], [new_grid_axis])
                else:
                    ValueError("the axis_direction value is not as expected.")

                self._add_grid_in_spatial_structures(new_grid, grid_info['IfcStoreys'])
    
    @time_decorator
    def enrich_ifc_with_grids(self):
        
        # extract the grid placements.
        self.enrich_grid_placement_information()
        # create the grids.
        self.create_reference_grids()
    
    @time_decorator
    def enrich_reference_relationships(self):
       
        for grid_label, grid_data in self.all_grid_data.items():
            if grid_data["type"] in {"st_grid", "ns_grid"}:
                
                new_grid_element = self.model.by_guid(grid_data['id'])
                all_cd_element_ids = []
                for id_cd_grid in grid_data['children']:
                    cd_element_ids = self.all_grid_data[id_cd_grid]['children']
                    all_cd_element_ids+=cd_element_ids
                all_cd_elements = [self.model.by_guid(id) for id in all_cd_element_ids]

                self.model.create_entity(
                    "IfcRelReferencedInSpatialStructure",
                    GlobalId=ifcopenshell.guid.new(),
                    RelatedElements=all_cd_elements,
                    RelatingStructure=new_grid_element)
                    
    def save_the_enriched_ifc(self):
        self.model.write(os.path.join(self.output_figure_path, 'enriched_' + self.ifc_file_name))

    # self.model.create_entity("IfcRelReferencedInSpatialStructure", GlobalId=ifcopenshell.guid.new(), RelatedElements=[element], RelatingStructure=grid)

    # self.model.create_entity("IfcRelContainedInSpatialStructure", GlobalId=ifcopenshell.guid.new(), RelatedElements=[element], RelatingStructure=grid)
    #    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # def _create_grid_reference(self, grid):
    #     """Create a classification reference for a grid."""
    #     return self.model.create_entity(
    #         'IfcClassificationReference',
    #         Location=f"Grid:{grid.Name}",
    #         Identification=grid.GlobalId,
    #         Name=grid.Name
    #     )