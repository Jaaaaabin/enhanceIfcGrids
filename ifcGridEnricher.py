import ifcopenshell
from ifcopenshell.api import run
import math
import json
import os
import time
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
        self.initialize_ifc_owner_history()

    def initialize_ifc_structure(self):

        self.project = self.model.by_type('IfcProject')[0]
        self.building = self.model.by_type('IfcBuilding')[0]
        self.storeys = self.model.by_type('IfcBuildingStorey')

        # self.st_columns = self.model.by_type("IfcColumn")
        # self.slabs = self.model.by_type("IfcSlab")
        # self.roofs = self.model.by_type("IfcRoof")
        # self.beams = self.model.by_type("IfcBeam")
        self.walls = self.model.by_type("IfcWall")
        # self.curtainwalls = self.model.by_type("IfcCurtainWall")
        # self.plates = self.model.by_type("IfcPlate")
        # self.members = self.model.by_type("IfcMember")

        max_z_values = max(st.Elevation for st in self.storeys)
        if max_z_values > 1000:
            self.mode_unit = 0.001  # unit = 0.001 * meter
        self.elev_storeys = {round(st.Elevation * self.mode_unit, 1): st for st in self.storeys}

    def initialize_ifc_owner_history(self):

        person = self.model.create_entity("IfcPerson", GivenName="Jiabin", FamilyName="Wu")
        org = self.model.create_entity("IfcOrganization", Name="TUMCMS")
        person_and_org = self.model.create_entity("IfcPersonAndOrganization", ThePerson=person, TheOrganization=org)
        app = self.model.create_entity(
            "IfcApplication", ApplicationDeveloper=org, Version="1.0", ApplicationFullName="IFC ENRICHMENT", ApplicationIdentifier="MyApp")

        # Create IfcOwnerHistory
        self.new_owner_history = self.model.create_entity(
            "IfcOwnerHistory",
            # GlobalId=ifcopenshell.guid.new(),
            OwningUser=person_and_org,
            OwningApplication=app,
            ChangeAction="ADDED",
            CreationDate=int(time.time())
        )
    
    def save_the_enriched_ifc(self):
        self.model.write(os.path.join(self.output_figure_path, 'enriched_' + self.ifc_file_name))

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
        
                # OwnerHistory=self.new_owner_history,
        return self.model.create_entity(
            'IfcGridAxis',
            AxisTag=label,
            AxisCurve=indexed_curve,
            SameSense=SameSense_axes,
        )
    
    def _contain_a_grid_in_a_spatial_structure(self, grid, relating_structure):

        existing_containing_relation = None
        for relation in relating_structure.ContainsElements:
            if relation.is_a("IfcRelContainedInSpatialStructure"):
                existing_containing_relation = relation
                break
                    
        if existing_containing_relation:

            # Add the new grid to the existing relationship
            updated_related_elements = existing_containing_relation.RelatedElements + (grid,)
            existing_containing_relation.RelatedElements = updated_related_elements
            existing_containing_relation.OwnerHistory = self.new_owner_history
    
        else:
            
            self.model.create_entity(
                "IfcRelContainedInSpatialStructure",
                GlobalId=ifcopenshell.guid.new(),
                OwnerHistory=self.new_owner_history,
                RelatingStructure=relating_structure,
                RelatedElements=[grid])
        
    def create_reference_grids(self):

        grid_axis_mapping = {'U': 'UAxes','V': 'VAxes', 'W': 'WAxes'}
        
        for axis_direction, axis_grid_info in self.grid_placements.items():
            for axis_label, grid_info in axis_grid_info:
                new_grid = self.model.create_entity(
                    "IfcGrid",
                    GlobalId=ifcopenshell.guid.new(),
                    OwnerHistory=self.new_owner_history,
                    Name=axis_direction+'_'+axis_label)
                self.all_grid_data[axis_label].update({'id':new_grid.GlobalId})
                same_sense_on_this_axis  = axis_direction != 'W'
                new_grid_axis = self._create_grid_axis(label=axis_label, info=grid_info, SameSense_axes=same_sense_on_this_axis)

                if axis_direction in grid_axis_mapping:
                    setattr(new_grid, grid_axis_mapping[axis_direction], [new_grid_axis])
                else:
                    ValueError("the axis_direction value is not as expected.")

                if len(grid_info['IfcStoreys'])>1:

                    # relates to multiple storeys -> building.
                    self._contain_a_grid_in_a_spatial_structure(new_grid, self.building)
                elif len(grid_info['IfcStoreys'])==1:

                    # relates to one single storey -> storey
                    self._contain_a_grid_in_a_spatial_structure(new_grid, grid_info['IfcStoreys'][0])
                else:
                    
                    # there's no storeys, a case impossible.
                    ValueError("the number of related IfcStorey is less than 1 for the grid")
    
    @time_decorator
    def enrich_ifc_with_grids(self):
        
        # extract the grid placements.
        self.enrich_grid_placement_information()
        
        # create the grids.
        self.create_reference_grids()
    
    @time_decorator
    def enrich_reference_relationships_relconstraint(self):
        
        principal_constraint = self.model.create_entity(
                            "IfcConstraint",
                            Name="LogicalPlacementConstraint (Simulating IfcRelPositions)",
                            Description="Principal constrained placement relative to gloabl IfcGrid.",
                            ConstraintGrade="USERDEFINED",
                            UserDefinedGrade="PRINCIPAL", # 'UserDefinedGrade' must be asserted when IfcConstraintGradeEnum="USERDEFINED"
                            )
        supplementary_constraint = self.model.create_entity(
                            "IfcConstraint",
                            Name="LogicalPlacementConstraint (Simulating IfcRelPositions)",
                            Description="Supplementary constrained placement relative to local IfcGrid.",
                            ConstraintGrade="USERDEFINED",
                            UserDefinedGrade="SUPPLEMENTARY", # 'UserDefinedGrade' must be asserted when IfcConstraintGradeEnum="USERDEFINED"
                            )
        
        # the following parts are directly copied from 'enrich_reference_relationships_relref'
        for grid_label, grid_data in self.all_grid_data.items():
            if grid_data["type"] in {"st_grid", "ns_grid"}:
                
                new_grid_element = self.model.by_guid(grid_data['id'])
                all_cd_element_ids = []
                for id_cd_grid in grid_data['children']:
                    cd_element_ids = self.all_grid_data[id_cd_grid]['children']
                    all_cd_element_ids+=cd_element_ids
                all_cd_elements = [self.model.by_guid(id) for id in all_cd_element_ids]

                # Create the IfcConstraint entity to define the positioning rule or constraint
                
                    # ----------------------------------------------
                    # IfcRelConnets <- IfcRelPositions(
                    #     //GlobalId, OwnerHistory,Name, Description//,
                    #     RelatingPositioningElement:IfcPositioningElement,
                    #     RelatedProducts:IfcProduct)
                    # ----------------------------------------------
                    # IfcRelAssociates <- IfcRelAssociatesConstraint(
                    #     //GlobalId, OwnerHistory,Name, Description//,
                    #     IfcRelAssociates(
                    #         //GlobalId, OwnerHistory,Name, Description//, RelatedObjects:SetofObjects),
                    #     Intent:IfcLable,
                    #     RelatingConstraint:IfcConstraint)
                    # ----------------------------------------------
                    # IfcConstraint(
                    #     Name
                    #     Description
                    #     ConstraintGrade
                    #     UserDefinedGrade
                    # )
                    # IfcRelAssociatesConstraint
                    # 
    
                # Create the IfcRelAssociatesConstraint relationship
                relation_constraint = self.model.create_entity(
                    "IfcRelAssociatesConstraint",
                    GlobalId=ifcopenshell.guid.new(),
                    OwnerHistory=self.new_owner_history,
                    Name="ApplyIfcConstraint",
                    Description="Placement relative to IfcGrid",
                    RelatedObjects=all_cd_elements,
                    Intent=new_grid_element.Name,
                    RelatingConstraint=principal_constraint,  # The constraint we just created
                )

    @time_decorator
    def enrich_reference_relationships_relref(self):
    
    # store the reference dependencies in 'IfcRelReferencedInSpatialStructure'.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Attention: 'IfcRelReferencedInSpatialStructure' can only be used to reference spatial elements such as :
    # site as IfcSite, building as IfcBuilding, storey as IfcBuildingStorey, space as IfcSpace
    # self.model.create_entity("IfcRelReferencedInSpatialStructure", GlobalId=ifcopenshell.guid.new(), RelatedElements=[element], RelatingStructure=grid)
    # self.model.create_entity("IfcRelContainedInSpatialStructure", GlobalId=ifcopenshell.guid.new(), RelatedElements=[element], RelatingStructure=grid)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
                    OwnerHistory=self.new_owner_history,
                    RelatedElements=all_cd_elements,
                    RelatingStructure=new_grid_element)
    