import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

import os
import copy
import math
import itertools
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks

import OCC.Core
import OCC.Core.Units
import OCC.Core.GProp
import OCC.Core.BRepGProp
import OCC.Core.TopExp,OCC.Core.TopAbs,OCC.Core.TopoDS,OCC.Core.BRepBndLib,OCC.Core.BRep

import OCC.Extend.TopologyUtils
# from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
# from OCC.Core.Quantity import Quantity_Color, Quantity_NameOfColor

import shapely
from shapely.geometry import Point, LineString, MultiPoint
from shapely.geometry import CAP_STYLE, JOIN_STYLE

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import bokeh.plotting


# import geopandas as gpd


class IfcExtractor:

    def __init__(self, model_path, figure_path):

        try:
            self.model = ifcopenshell.open(model_path)
            self.ifc_file_name = os.path.basename(model_path)
            self.out_fig_path = figure_path
            os.makedirs(figure_path, exist_ok=True)
            self.version = self.model.schema
            self.storeys = self.model.by_type("IfcBuildingStorey")
            self.spaces = []
            self.walls = []
            self.columns = []
            self.existing_grids = []
            self.doors = []
            self.windows = []
            self.grids = {}
            self.extract_elements()

        except ifcopenshell.errors.FileNotFoundError:
            print(f"Error: File '{model_path}' not found.")

        except Exception as e:
            print(f"An error occurred: {e}")
    
    def extract_elements(self):

        """
        Extracts various IFC entities from the model and initializes visualization settings.
        """

        if self.model:

            self.existing_grids = self.model.by_type("IfcGrid")
            self.storeys = self.model.by_type("IfcBuildingStorey")
            self.spaces = self.model.by_type("IfcSpace")
            self.walls = self.model.by_type("IfcWall") + self.model.by_type("IfcCurtainWall")
            self.columns = self.model.by_type("IfcColumn")
            self.doors = self.model.by_type("IfcDoor")
            self.windows = self.model.by_type("IfcWindow")
            
            self.init_visualization_settings()

    def init_visualization_settings(self):

        """
        Initializes visualization settings for various building components.
        """

        self.visualization_settings = {
            # points.
            'points_column':{
                'legend_label':'Column Points',
                'color': "darkgreen",
                'size':8,
                'alpha':1,
            },

            # lines.
            'lines_st_wall':{
                'legend_label':'Structural Wall Lines',
                'color': "black",
                'line_dash':'solid',
                'line_width':3,
                'alpha':1,
            },
            'lines_ns_wall':{
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
    
    def get_object_elevation(self, object):

        """
        Retrieves the elevation of a given object based on its spatial containment within a building storey.

        Parameters:
            object (IfcObject): The object to find the elevation for.

        Returns:
            float or None: The elevation of the object if found, otherwise None.
        """
        if object and hasattr(object, 'ContainedInStructure'):
            for definition in object.ContainedInStructure:
                if definition.is_a('IfcRelContainedInSpatialStructure'):
                    element = definition.RelatingStructure
                    if element.is_a('IfcBuildingStorey'):
                        return element.Elevation
        return None
    
        # for definition in object.ContainedInStructure:
        #     if definition.is_a('IfcRelContainedInSpatialStructure'):
        #         element = definition.RelatingStructure
        #         if element.is_a('IfcBuildingStorey'):
        #             return element.Elevations

#===================================================================================================
#column
    
    def cacl_column_location_byplacement(self,column):
        
        column_location = column.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(column, "ObjectPlacement") else None
        return column_location
    
    def calc_column_location(self, column):
        
        column_location = None
        for r in column.Representation.Representations:
            if r.RepresentationIdentifier == 'Body':
                mapped_r = r.Items[0].MappingSource.MappedRepresentation
                if mapped_r.RepresentationType=='SweptSolid' and hasattr(mapped_r.Items[0],'Position'):
                    column_location = mapped_r.Items[0].Position.Location.Coordinates
        
        if column_location is None:
            column_location = self.cacl_column_location_byplacement(column)

        return list(column_location)

    def get_column_info(self):
        self.column_info = []
        for column in self.columns:
            column_location = self.calc_column_location(column)
            self.column_info.append({
                "id": column.GlobalId,
                "location": column_location,
            })

    # old            
    # def calc_column_location(self, column):
            
        # component = column
        # elevation = self.get_object_elevation(column)

        # location = []
        # for r in component.Representation.Representations:

        #     if r.RepresentationIdentifier =='Body':
        #         location = r.Items[0].MappingSource.MappedRepresentation.Items[0].Position.Location.Coordinates

        #     # location =  column_placement.PlacementRelTo.ReferencedByPlacements[-1].RelativePlacement.Location.Coordinates
        
        #     # if column_placement.is_a('IfcLocalPlacement') and column_placement.RelativePlacement:
        #     #     column_placement_rel_cords = column_placement.RelativePlacement.Location.Coordinates
        
        #     # if column_placement.PlacementRelTo.PlacesObject:
        #     #     column_placement_rel_object = column_placement.PlacementRelTo.PlacesObject

        #     # # if elevation == column_placement_rel_object[0].Elevation: ?
        #     # if elevation != None:
        #     #     location = [*column_placement_rel_cords[:-1], elevation]
        #         location = [*location]

        #         return location


#===================================================================================================
#wall information extraction based on highly-enriched semantics

    # def get_wall_width(self, wall):
        
    #     psets = ifcopenshell.util.element.get_psets(wall)
    #     if 'Qto_WallBaseQuantities' in psets.keys():
    #         if 'Width' in psets['Qto_WallBaseQuantities'].keys():
    #             width = round(psets['Qto_WallBaseQuantities']['Width'],2)
    #             return width
    #         else:
    #             return None
    #     else:
    #         return None

    def get_wall_loadbearing(self, wall):
        
        psets = ifcopenshell.util.element.get_psets(wall)
        if 'Pset_WallCommon' in psets.keys():
            if 'LoadBearing' in psets['Pset_WallCommon'].keys():
                load_bearing = psets['Pset_WallCommon']['LoadBearing']
                return load_bearing
            else:
                return None
        else:
            return None

#===================================================================================================
#wall - calculate
        
    def calc_wall_orientation(self, wall, deg_range=360, orien_dec=[]):
            
        orientation_deg = None

        if wall.is_a('IfcWall') or wall.is_a('IfcWallStandardCase'):
            component = wall
        elif wall.is_a('IfcCurtainWall'):
            component = wall.IsDecomposedBy[0].RelatedObjects[0]

        if component.ObjectPlacement.RelativePlacement.RefDirection != None:
            orientation_vec = component.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios
            orientation_rad = math.atan2(orientation_vec[1],orientation_vec[0])
            orientation_deg = math.degrees(orientation_rad) % deg_range
        else:
            orientation_deg = 0.0

        orientation_deg = round(orientation_deg, int(orien_dec)) if orien_dec else orientation_deg
        
        return orientation_deg
    

    def calc_wall_length_by_pset(self, wall):
        """Gets the length of a wall from its property sets."""
        
        wall_length = None
        
        # what about the IfcStandardcaseWall?
        if not wall.is_a('IfcWall') and not wall.is_a('IfcCurtainWall') and not wall.is_a('IfcWallStandardCase'):
            return 0.0

        wall_pset = ifcopenshell.util.element.get_psets(wall)
        if wall.is_a('IfcWall') or wall.is_a('IfcWallStandardCase'):
            for pset_key in wall_pset:
                if 'Length' in wall_pset[pset_key]:
                    return round(wall_pset[pset_key]['Length'], 4)
        
        elif wall.is_a('IfcCurtainWall'):
            for pset_key in wall_pset:
                if 'Length' in wall_pset[pset_key]:
                    return round(wall_pset[pset_key]['Length'], 4)
        else:
            return wall_length

    def find_farthest_linear_points(self, points):
        
        def distance_2d(point_a, point_b):
            """Calculate the Euclidean distance between two points in 2D."""
            return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
        max_distance = 0
        farthest_linear_points = None

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = distance_2d(points[i], points[j])
                if dist > max_distance:
                    max_distance = dist
                    farthest_linear_points = (points[i], points[j])

        return farthest_linear_points
    
    def calc_wall_location(self, wall):
        """to replace the calc_wall_dimensions."""

        # wall, elevation------------------------

        if wall.ContainedInStructure[0].RelatingStructure.is_a('IfcBuildingStorey'):
            wall_elevation = wall.ContainedInStructure[0].RelatingStructure.Elevation
        else:
            wall_elevation = None

        local_points = None,
        global_location_0, global_location_1 = None, None

        #'IfcWall' or 'IfcWallStandardCase' conditions.
        if wall.is_a('IfcWall') or wall.is_a('IfcWallStandardCase'):

            # local
            for r in wall.Representation.Representations:
                if r.RepresentationIdentifier =='Axis':
                    wall_axis = r
                    if wall_axis.Items[0].is_a('IfcPolyline'):
                        wall_pnts = wall_axis.Items[0].Points
                        local_points = wall_pnts.CoordList if hasattr(wall_pnts, "CoordList") else (wall_pnts[0].Coordinates,wall_pnts[1].Coordinates)
                    
                    #============ for curve walls=======================
                    elif wall_axis.Items[0].is_a('IfcTrimmedCurve'):
                        
                        return [0,0,wall_elevation],[0,0,wall_elevation]
                    #============ for curve walls=======================

            # global
            global_location_0 = wall.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(wall, "ObjectPlacement") else None
            orientation_deg = self.calc_wall_orientation(wall,deg_range=360)
            wall_length = self.calc_wall_length_by_pset(wall)
            if orientation_deg != None and wall_length != None:
                global_location_1 = (
                    global_location_0[0]+ wall_length*math.cos(math.radians(orientation_deg)),
                    global_location_0[1]+ wall_length*math.sin(math.radians(orientation_deg)),
                    global_location_0[2])
            else:
                global_location_1 = None
                    
            global_location_0 = [*global_location_0[:-1], wall_elevation]
            global_location_1 = [*global_location_1[:-1], wall_elevation]

        # 'IfcCurtainWall' conditions.
        elif wall.is_a('IfcCurtainWall'):
            
            related_components = wall.IsDecomposedBy[0].RelatedObjects
            if len(related_components)==1 and related_components[0].is_a('IfcPlate'):
                
                # local
                component = related_components[0]
                for r in component.Representation.Representations:
                    if r.RepresentationIdentifier =='FootPrint':
                        # local_points = r.Items[0].MappingSource.MappedRepresentation.Items[0].Points.CoordList
                        local_points = r.Items[0].MappingSource.MappedRepresentation.Items[0].Points
                        if isinstance(local_points,tuple):
                            local_points = [list(c) for pt in local_points for c in pt]
                        print ("Footprint:",local_points)
        
                # global
                global_location_0 = component.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(component, "ObjectPlacement") else None
                orientation_deg = self.calc_wall_orientation(wall,deg_range=360)
                wall_length = self.calc_wall_length_by_pset(wall)
                if orientation_deg != None and wall_length != None:
                    global_location_1 = (
                        global_location_0[0]+ wall_length*math.cos(math.radians(orientation_deg)),
                        global_location_0[1]+ wall_length*math.sin(math.radians(orientation_deg)),
                        global_location_0[2])
                else:
                    global_location_1 = None
                        
                global_location_0 = [*global_location_0[:-1], wall_elevation]
                global_location_1 = [*global_location_1[:-1], wall_elevation]
                        
            elif len(related_components) > 1:
                
                # direct global
                all_placement_points = []
                for component in related_components:
                    placement_point = component.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(component, "ObjectPlacement") else None
                    all_placement_points.append(list(placement_point))
                
                local_points = self.find_farthest_linear_points(all_placement_points)
                global_location_0, global_location_1 = local_points
                global_location_0 = [*global_location_0[:-1], wall_elevation]
                global_location_1 = [*global_location_1[:-1], wall_elevation]

        return global_location_0,global_location_1
                    
    # # length calculation
        
    #     # gloabl locations---------------------------
    #     orientation_deg = self.calc_wall_orientation(wall,deg_range=360) # 360 for location calculation
    #     global_location_0 = component.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(component, "ObjectPlacement") else None
        
    #     if orientation_deg != None and length != None:
    #         global_location_1 = (
    #             global_location_0[0]+ length*math.cos(math.radians(orientation_deg)),
    #             global_location_0[1]+ length*math.sin(math.radians(orientation_deg)),
    #             global_location_0[2])
    #     else:
    #         global_location_1 = None
                
    #     global_location_0 = [*global_location_0[:-1], wall_elevation]
    #     global_location_1 = [*global_location_1[:-1], wall_elevation]
        
    #     return length, [global_location_0,global_location_1]
    
    # #tempo
    # def calc_wall_dimensions(self, wall, length_dec=[]):
            
    #     # length---------------------------
    #     local_points, length = None, None 
    #     global_location_0, global_location_1 = None, None 

    #     # conditions.
    #     if wall.is_a('IfcWall') or wall.is_a('IfcWallStandardCase'):
    #         component = wall
    #     elif wall.is_a('IfcCurtainWall'):
    #         component = wall.IsDecomposedBy[0].RelatedObjects[0]

    #     if component.is_a('IfcPlate'):
    #         # IfcCurtainWall -> get the FootPrint of the related object IfcPlate.
    #         for r in component.Representation.Representations:
    #             if r.RepresentationIdentifier =='FootPrint':
    #                 # local_points = r.Items[0].MappingSource.MappedRepresentation.Items[0].Points.CoordList
    #                 local_points = r.Items[0].MappingSource.MappedRepresentation.Items[0].Points
    #                 if isinstance(local_points,tuple):
    #                     local_points = [list(c) for pt in local_points for c in pt]
    #                 print ("Footprint:",local_points)
           
    #     else:
    #         #IfcWall +IfcWallStandardCase - > get the axis representation of the IfcWall.
    #         for r in component.Representation.Representations:
    #             if r.RepresentationIdentifier =='Axis':
    #                 component_axis = r
    #                 if component_axis.Items[0].is_a('IfcPolyline'):
    #                     wall_pnts = component_axis.Items[0].Points
    #                     local_points = wall_pnts.CoordList if hasattr(wall_pnts, "CoordList") else (wall_pnts[0].Coordinates,wall_pnts[1].Coordinates)
                    
    #                 #============ for curve walls=======================
    #                 elif component_axis.Items[0].is_a('IfcTrimmedCurve'):
    #                     return 0, [[0,0,0],[0,0,0]]
    #                 #============ for curve walls=======================
        
    #     # length calculation
    #     if local_points != None:
    #         if len(local_points) > 2:
    #             local_points = local_points[:-1]
    #             x_coords = [point[0] for point in local_points]
    #             y_coords = [point[1] for point in local_points]
    #             length = max((max(x_coords) - min(x_coords)) , (max(y_coords) - min(y_coords)))
    #         else:
    #             length = np.max(tuple(abs(x-y) for x,y in zip(local_points[1],local_points[0])))
    #         length = round(length, int(length_dec)) if length_dec else length
    #         return length, [global_location_0,global_location_1]
        
    #     # gloabl locations---------------------------
    #     orientation_deg = self.calc_wall_orientation(wall,deg_range=360) # 360 for location calculation
    #     elevation = self.get_object_elevation(wall)
    #     global_location_0 = component.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(component, "ObjectPlacement") else None
        
    #     if orientation_deg != None and length != None:
    #         global_location_1 = (
    #             global_location_0[0]+ length*math.cos(math.radians(orientation_deg)),
    #             global_location_0[1]+ length*math.sin(math.radians(orientation_deg)),
    #             global_location_0[2])
    #     else:
    #         global_location_1 = None
                
    #     global_location_0 = [*global_location_0[:-1], elevation]
    #     global_location_1 = [*global_location_1[:-1], elevation]
        
    #     return length, [global_location_0,global_location_1]

    def calculate_width_from_sweptArea(self, sweptArea):
        
        geometry = []

        if hasattr(sweptArea, 'XDim') and hasattr(sweptArea, 'YDim'):
            return min(sweptArea.XDim, sweptArea.YDim)
        
        elif hasattr(sweptArea, 'OuterCurve') and sweptArea.OuterCurve.Points:
            
            if (hasattr(sweptArea.OuterCurve.Points,'Coordinates') and len(sweptArea.OuterCurve.Points)==1) \
                or len(sweptArea.OuterCurve.Points) > 1:
                geometry = [p.Coordinates for p in sweptArea.OuterCurve.Points]
            elif hasattr(sweptArea.OuterCurve.Points,'CoordList'):
                geometry = [p for p in sweptArea.OuterCurve.Points.CoordList]
        
        elif hasattr(sweptArea, 'OuterCurve') and sweptArea.OuterCurve.Segments:
            return 0.0
        
        if geometry:
            x_values, y_values = zip(*geometry)
            x_width, y_width = max(x_values) - min(x_values), max(y_values) - min(y_values)
            return min([x_width, y_width])
    
    def calculate_representation_width_from_ExtrudedAreaSolid(self, r):
        
        # Input validation
        ExtrudedAreaSolid = r.Items[0]
        if not ExtrudedAreaSolid or not ExtrudedAreaSolid.is_a('IfcExtrudedAreaSolid'):
            print("The provided object is not a valid IfcExtrudedAreaSolid.")
            return None

        # ================= for testing ======================
        # issues only with IFC 4 Reference View (Architectural/Structural/Building Service)
        # 0.0095 - PolygonalFaceSet - ['2yaEAQnL91IvnKfXbRCgji','3m7UEo_RD4nOFYw3Z96OaP']
        # 0.1355 - multiple IfcExtrudedAreaSolids - good ['2yaEAQnL91IvnKfXbRCgjj','2yaEAQnL91IvnKfXbRCgiP']
        # 7.71 - multiple IfcExtrudedAreaSolids - bads ['2yaEAQnL91IvnKfXbRCgpC']
        # ================= for testing ====================== 

        # handle cases for both single and multiple ExtrudedAreaSolids.
        num_ExtrudedAreaSolid = len(r.Items)
        if num_ExtrudedAreaSolid > 1: 
            # multiple IfcExtrudedAreaSolids.

            # ------ to recheck and resovle
            all_widths_Depth = [a.Depth for a in r.Items]
            width_Depth = sum(all_widths_Depth)    
            all_widths_sweptArea = [self.calculate_width_from_sweptArea(a.SweptArea) for a in r.Items]
            width_sweptArea = sum(all_widths_sweptArea)
            # print ("multiple IfcExtrudedAreaSolids | all_widths_Depth:",all_widths_Depth,"\n all_widths_sweptArea:",all_widths_sweptArea,"\n final width:", min(width_Depth, width_sweptArea))
            return min(width_Depth, width_sweptArea)
            # ------ to recheck and resovle
        
        else:
            # one single IfcExtrudedAreaSolid.
            sweptArea = r.Items[0].SweptArea
            width_Depth = r.Items[0].Depth
            width_sweptArea = self.calculate_width_from_sweptArea(sweptArea)
            # print ("single IfcExtrudedAreaSolid | width_Depth:",width_Depth,"\n width_sweptArea:",width_sweptArea,"\n final width:", min(width_Depth, width_sweptArea))
            return min(width_Depth, width_sweptArea)

    def calculate_width_from_PolygonalFaceSet(self, PolygonalFaceSet):

        # Access the Cartesian points list
        pointsList = PolygonalFaceSet.Coordinates.CoordList
        minY, maxY = float('inf'), float('-inf')
        for point in pointsList:
            y = point[1]
            minY, maxY = min(minY, y), max(maxY, y)
            
        # print ("PolygonalFaceSet | minY, maxY:", minY, maxY, "\n final width:", maxY - minY)   
        return maxY - minY
    
    def calculate_representation_width_from_PolygonalFaceSet(self, r):
        
        # Input validation
        faceSet = r.Items[0]
        if not faceSet or not faceSet.is_a('IfcPolygonalFaceSet'):
            print("The provided object is not a valid IfcPolygonalFaceSet.")
            return None
        
        width_PolygonalFaceSet = None
        num_PolygonalFaceSet = len(r.Items)
        if num_PolygonalFaceSet > 1: 
            width_PolygonalFaceSet = [self.calculate_width_from_PolygonalFaceSet(a) for a in r.Items]
            width_PolygonalFaceSet = sum(width_PolygonalFaceSet)    
        else:
            PolygonalFaceSet = r.Items[0]
            width_PolygonalFaceSet = self.calculate_width_from_sweptArea(PolygonalFaceSet)

        return width_PolygonalFaceSet


    def get_wall_width_by_geometry(self, wall):
        
        if not wall.Representation or not wall.Representation.Representations:
            
            if wall.is_a('IfcCurtainWall'):
                # IfcCurtainWall. Definition from ISO 6707-1:1989:
                # Non load bearing wall positioned on the outside of a building and enclosing it.
                return 0.0
                
                # # -------------------------------------------
                # # do we need to validate it as a non-structural wall placed externall.
                # for rel_properties in wall.IsDefinedBy:
                #     if hasattr(rel_properties.RelatingProertyDefinition, 'HasProerties'):
                #         for single_value in rel_properties.RelatingProertyDefinition.HasProerties:
                #             if single_value.Name == 'IsExternal' and single_value.NominalValue.wrappedValue == True:
                #                 return 0.0
                # # -------------------------------------------
            else:
                print("Wall has no Representation or no Representation.Representations.")
                return None
        
        # ifc 4 reference view structual view doesn't have any representation.
        width = None
        for r in wall.Representation.Representations:
            if r.RepresentationIdentifier == 'Body':
                try:
                    if r.Items[0].is_a('IfcExtrudedAreaSolid'):
                        width = self.calculate_representation_width_from_ExtrudedAreaSolid(r)
                    elif r.Items[0].is_a('IfcPolygonalFaceSet'):
                        width = self.calculate_representation_width_from_PolygonalFaceSet(r)
                    if width is not None:
                        width = round(width, 4)
                except AttributeError:
                    continue  # Handle missing attributes gracefully
        print ("wall_geometry_thickness:",width)
        return width
    

    def get_wall_width_by_material(self, wall):
        
        wall_material = ifcopenshell.util.element.get_material(wall)
        # for ifc4. Error processing model test-base-ifc4-rv-s.ifc: entity instance of type 'IFC4.IfcMaterial' has no attribute 'Thickness'

        if wall_material:

            # IFC Reference Views, the material layer sets are excluded.
            if wall_material.is_a('IfcMaterial') and hasattr(wall_material, 'Thickness'):
                # If the material is an IfcMaterial entity, append its name and thickness (if available)
                wall_material_thickness = round(wall_material.Thickness, 4)
            
            elif wall_material.is_a('IfcMaterialLayerSetUsage') and hasattr(wall_material, 'ForLayerSet'):
                wall_material_thickness = 0.0
                for layer in wall_material.ForLayerSet.MaterialLayers:
                    wall_material_thickness += layer.LayerThickness
                wall_material_thickness = round(wall_material_thickness, 4)
            
            elif wall_material.is_a('IfcMaterialLayerSet') and hasattr(wall_material, 'MaterialLayers'):
                wall_material_thickness = 0.0
                for layer in wall_material.MaterialLayers:
                    wall_material_thickness += layer.LayerThickness
                wall_material_thickness = round(wall_material_thickness, 4)
            
            else:
                # If the material is not an IfcMaterial or a layer set usage, just append its type and None for thickness
                wall_material_thickness = None
        else:
            # If no material is found, append None for both material and thickness
            wall_material_thickness = None
        
        print ("wall_material_thickness:",wall_material_thickness)
        print ("--------------------------------------")
        return wall_material_thickness
    

    def get_wall_width_by_pset(self, wall):
        """Gets the width of a wall from its property sets."""
        
        # what about the IfcStandardcaseWall?
        if not wall.is_a('IfcWall') and not wall.is_a('IfcCurtainWall') and not wall.is_a('IfcStandardcaseWall'):
            return None
        
        wall_pset = ifcopenshell.util.element.get_psets(wall)

        if wall.is_a('IfcWall') or wall.is_a('IfcStandardcaseWall'):
            for pset_key in wall_pset:
                if 'Width' in wall_pset[pset_key]:
                    return round(wall_pset[pset_key]['Width'], 4)
        
        elif wall.is_a('IfcCurtainWall'):
            try:
                wall_component = wall.IsDecomposedBy[0].RelatedObjects[0]
                wall_component_pset = ifcopenshell.util.element.get_psets(wall_component)
                if 'Dimensions' in wall_component_pset and 'Thickness' in wall_component_pset['Dimensions']:
                    return round(wall_component_pset['Dimensions']['Thickness'], 4)
            except (IndexError, KeyError):
                pass

    def calc_wall_width(self, wall):

        width = self.get_wall_width_by_pset(wall)

        if width == None:
            width = self.get_wall_width_by_material(wall)
            if width == None:
                width = self.get_wall_width_by_geometry(wall)

        return width

    def get_wall_info(self):
            
        self.wall_info = []

        for wall in self.walls:
            
            print (wall.GlobalId)
            wall_loadbearing = self.get_wall_loadbearing(wall)
            wall_width = self.calc_wall_width(wall)
            wall_length = self.calc_wall_length_by_pset(wall)            
            wall_orientation = self.calc_wall_orientation(wall, deg_range=180, orien_dec=1) # 180 for orientation printing
            wall_location = self.calc_wall_location(wall)

            
            # wall_width = None
            # if wall_width_by_geometry is None and wall_width_by_material is None:
            #     wall_width = 0.0
            # elif wall_width_by_geometry is None:
            #     wall_width = max(0.0, wall_width_by_material)
            # elif wall_width_by_material is None:
            #     wall_width = max(0.0, wall_width_by_geometry)
            # else:
            #     wall_width = max(wall_width_by_geometry, wall_width_by_material)

            # wall_length, wall_location = self.calc_wall_dimensions(wall, length_dec=2)

            self.wall_info.append({
                "id": wall.GlobalId,
                "loadbearing":wall_loadbearing,
                "width": wall_width,
                "orientation": wall_orientation,
                "length": wall_length,
                "location": wall_location,
            })
    
    def get_main_storeys(self, num_wall=0):

        wall_elevations = []
        [wall_elevations.append(w["location"][0][-1]) for w in self.wall_info]

        self.main_storeys = []
        if wall_elevations:
            [self.main_storeys.append(st) for st in self.storeys if wall_elevations.count(st.Elevation)>num_wall]

#===================================================================================================
#space

    def calc_space_location(self, space):
        
        location = space.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(space, "ObjectPlacement") else None
        return location
    
    def get_space_info(self):
            
        space_info = []
        for space in self.spaces:
            space_location = self.calc_space_location(space)
            space_info.append({
                "id": space.id(),
                "location": space_location, # not working.
            })
        return space_info
    
    def get_wall_shape(self, wall):

        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_PYTHON_OPENCASCADE, True)
        settings.set(settings.USE_WORLD_COORDS, True)
        settings.set(settings.INCLUDE_CURVES, True)
        settings.set(settings.SEW_SHELLS, True)
        shape = ifcopenshell.geom.create_shape(settings, wall)
        return shape
    
#===================================================================================================
#wall analysis
    
    def wall_width_histogram(self):
        
        values = [w['width'] for w in self.wall_info if 'width' in w]
        #  RV_A / RV_S, width of non-structural walls are lost. - > to check.

        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes((0.0875, 0.1, 0.875, 0.875))
        ax.hist(values, weights=np.ones(len(values)) / len(values), bins=20, color='#bcbd22', edgecolor='black')
        ax.set_xlabel('Width of IfcWalls', color='black', fontsize=12)
        ax.set_ylabel("Percentage Frequency Distribution", color="black", fontsize=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(xmin=0.0, xmax=max(values))

        plt.savefig(os.path.join(self.out_fig_path, 'wall_width_histogram.png'), dpi=200)
        plt.close(fig)
        
    def wall_length_histogram(self):

        values = [w['length'] for w in self.wall_info if 'length' in w]

        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes((0.0875, 0.1, 0.875, 0.875))
        ax.hist(values, weights=np.ones(len(values)) / len(values), bins=20, color='#bcbd22', edgecolor='black')
        ax.set_xlabel('Length of IfcWalls', color='black', fontsize=12)
        ax.set_ylabel("Percentage Frequency Distribution", color="black", fontsize=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(xmin=0.0, xmax=max(values))

        plt.savefig(os.path.join(self.out_fig_path, 'wall_length_histogram.png'), dpi=200)
        plt.close(fig)

    def wall_location_map(self):
        values = [w['location'] for w in self.wall_info if 'location' in w]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        for v in values:
            start_point, end_point = v
            xs, ys, zs = zip(start_point, end_point)
            ax.plot(xs, ys, zs, marker='o', color='black', linewidth=1, markersize=3)

        plt.savefig(os.path.join(self.out_fig_path, 'wall_location_map.png'), dpi=200)
        plt.close(fig)

    def wall_orientation_histogram(self):
        values = [w['orientation'] for w in self.wall_info if 'orientation' in w]

        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes((0.0875, 0.1, 0.875, 0.875))
        ax.hist(values, weights=np.ones(len(values)) / len(values), bins=20, color='#bcbd22', edgecolor='black')
        ax.set_xlabel('Orientation of IfcWalls [0°,180°)', color='black', fontsize=12)
        ax.set_ylabel("Percentage Frequency Distribution", color="black", fontsize=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks(np.arange(0, 180, 30))
        ax.set_xlim(xmin=0, xmax=180)

        plt.savefig(os.path.join(self.out_fig_path, 'wall_orientation_histogram.png'), dpi=200)
        plt.close(fig)

    
#===================================================================================================
# grid-related

    def get_line_slope(self, point1, point2):
        
        # works for x and y, doesn't matter is z exits or not.
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        if dx != 0:
            slope = dy / dx
        else:
            slope = float('inf')  # Verticalline
        return slope
        
    def get_grids_from_points(self, elements_of_grids, border_x=[], border_y=[]):

        if len(border_x) != 2 or len(border_y) != 2:
            raise ValueError("border_x and border_y must each have two elements.")

        grids = []
        for elements in elements_of_grids:
            slopes = [self.get_line_slope(p1, p2) for (p1, p2) in itertools.combinations(elements, 2)]
            valid_slopes = [slope for slope in slopes if slope != float('inf')]
            
            if not valid_slopes:  # Handle cases where no valid slope could be determined
                mean_slope = 0  # or some other default value or handling mechanism
            else:
                mean_slope = np.mean(valid_slopes)
                
            mean_res = np.mean([pt[1] - mean_slope * pt[0] for pt in elements if mean_slope != float('inf')])
            
            p_start, p_end = [0, 0], [0, 0]
            if border_x and border_y:
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
    
    # # old
    # def get_grids_from_points_old(self, elements_of_grids, border_x=[], border_y=[]):
    
    #     if len(border_x) != 2 or len(border_y) != 2:
    #         raise ValueError("border_x and border_y must each have two elements.")
        
    #     grids = []
    #     for elements in elements_of_grids:

    #         mean_slope = np.mean([self.get_line_slope(p1,p2) for (p1,p2) in list(itertools.combinations(elements, 2))
    #                               if self.get_line_slope(p1,p2) != float('inf')])
    #         mean_res = np.mean([pt[1]-mean_slope*pt[0] for pt in elements])
            
    #         p_start, p_end = [0,0], [0,0]
    #         # function to resolve.
    #         if border_x and border_y:
    #             border_x_p_start = [border_x[0], border_x[0] * mean_slope + mean_res]
    #             border_x_p_end = [border_x[1], border_x[1] * mean_slope + mean_res]
    #             border_y_p_start = [(border_y[0]- mean_res) / mean_slope, border_y[0]]
    #             border_y_p_end = [(border_y[1]- mean_res) / mean_slope, border_y[1]]
    
    #             if border_x_p_start[1]>border_y[1] or border_x_p_start[1]<border_y[0]:
    #                 p_start = border_y_p_end if border_y_p_end[0]<border_y_p_start[0] else border_y_p_start
    #             else:
    #                 p_start = border_x_p_start
    #             if border_x_p_end[1]>border_y[1] or border_x_p_end[1]<border_y[0]:
    #                 p_end = border_y_p_end if border_y_p_end[0]>border_y_p_start[0] else border_y_p_start
    #             else:
    #                 p_end = border_x_p_end
               
    #         grids.append([p_start, p_end])
    #     grid_linestrings = [LineString(grid) for grid in grids]

    #     return grid_linestrings
   
    def generate_grids(
        self,
        grid_type,
        t_c_dist=0,
        t_c_num=0,
        t_w_dist=0,
        t_w_num=0,
        only_structural_grid=True,
        ):
        
        elements_of_grids = []
        ids_components_per_grid = []
        components_of_grids= []

        if grid_type == 'IfcColumn':
            
            # the STRUCTURAL IfcColumn-oriented elements_of_grids are analyzed per floor.
            components = self.column_info
            component_pts = [w['location'] for w in components]

            # Iterate over all pairs of points on the same level to check alignments
            for i, point1 in enumerate(component_pts):
                for j, point2 in enumerate(component_pts):

                    if i != j and point1[-1] == point2[-1]:
                        
                        # Calculate the inclination (slope) of the line between two points in XY plane
                        slope = self.get_line_slope(point1, point2)

                        # Find points that align with this line within the tolerance
                        aligned_points = []
                        id_components = [i,j]
                        for k, point in enumerate(component_pts):
                            if point[-1] == point1[-1] and abs((point[1] - point1[1]) - slope * (point[0] - point1[0])) <= t_c_dist:
                                aligned_points.append(point)
                                id_components.append(k)

                        # Add unique elements_of_grids to the potential elements_of_grids list
                        aligned_points = sorted(aligned_points)
                        if aligned_points not in elements_of_grids and len(aligned_points) >= t_c_num:
                            elements_of_grids.append(aligned_points)
                            ids_components_per_grid.append(list(set(id_components)))

                    else:
                        continue
            
            for ids in ids_components_per_grid:
                components_of_grids.append([components[id] for id in ids])
        
        elif grid_type == 'IfcWall':

            components = self.wall_info
            if only_structural_grid:
                component_pts = [w['location'] for w in components if w['loadbearing']]
            else:
                component_pts = [w['location'] for w in components if not w['loadbearing']]
            
            # Iterate over all pairs of endpoints of walls
            for i,pts in enumerate(component_pts):

                point1,point2 = pts

                # Calculate the inclination (slope) of the line between two points in XY plane
                slope = self.get_line_slope(point1, point2)

                # Find pairs of points that overlap with this line within the tolerance
                aligned_points = [[point1,point2]]
                id_components = [i]

                for j, new_pts in enumerate(component_pts):
                    
                    point3, point4 = new_pts

                    if i != j :
                        
                        if abs(slope - self.get_line_slope(point3, point4)) <= t_w_dist and \
                            abs((point3[1] - point1[1]) - slope * (point3[0] - point1[0])) <= t_w_dist:
                            aligned_points.append([point3,point4])
                            id_components.append(j)

                # Add unique elements_of_grids to the potential elements_of_grids list
                if aligned_points not in elements_of_grids and len(aligned_points) >= t_w_num:
                    elements_of_grids.append(aligned_points)
                    ids_components_per_grid.append(list(set(id_components)))

            elements_of_grids = [[e for element in elements for e in element] for elements in elements_of_grids]
            for ids in ids_components_per_grid:
                components_of_grids.append([components[id] for id in ids])

        return (elements_of_grids, components_of_grids)
    
    def get_info_elements_per_storey(self, storey, tol_elevation=0.001):

        wall_info_per_storey = []
        for wall in self.wall_info:
            if abs(wall['location'][0][-1]-storey.Elevation) <= tol_elevation :
                wall_info_per_storey.append(wall)

        column_info_per_storey = []
        for column in self.column_info:
            if abs(column['location'][-1]-storey.Elevation) <= tol_elevation :
                column_info_per_storey.append(column)

        # differentiate between structural and non-structural walls.
        s_wall_locations =  [w['location'] for w in wall_info_per_storey if w['loadbearing']]
        ns_wall_locations =  [w['location'] for w in wall_info_per_storey if not w['loadbearing']]
        
        wall_locations_struc = copy.deepcopy(s_wall_locations)
        wall_locations_nonst = copy.deepcopy(ns_wall_locations)
        [p.pop() for wall_loc in wall_locations_struc for p in wall_loc]
        [p.pop() for wall_loc in wall_locations_nonst for p in wall_loc]
        
        s_column_locations = [c['location'] for c in column_info_per_storey]

        column_locations_struc = copy.deepcopy(s_column_locations)
        [column_loc.pop() for column_loc in column_locations_struc]
        
        wall_lines_struc = [LineString(wall_location) for wall_location in wall_locations_struc]
        wall_lines_nonst = [LineString(wall_location) for wall_location in wall_locations_nonst]
       
        column_points = [Point(column_loc) for column_loc in column_locations_struc]

        return (wall_lines_struc,wall_lines_nonst,column_points)

    def init_grids_per_storey(
        self,
        storey,
        t_c_dist=0.01,      # T_{c,dist}
        t_c_num=3,          # T_{c,num}
        t_w_dist=0.01,      # T_{w,dist}
        t_w_num=2,          # T_{w,num}
        plot_fig=True,
        ):        
        
        # generattion of grids and visualization
        (wall_lines_struc,wall_lines_nonst,column_points) = self.get_info_elements_per_storey(storey=storey)
        plot_name = f"\[Floor \, Plan \, of \, {storey.Name} \, (T_{{c,dist}}={t_c_dist}, \, T_{{c,num}}={t_c_num}, \, T_{{w,dist}}={t_w_dist}, \, T_{{w,num}}={t_w_num}) - Initial \]"
        fig_save_name = f"Initial_{storey.Name}_t_c_dist_{t_c_dist}_t_c_num_{t_c_num}_t_w_dist_{t_w_dist}_t_w_num_{t_w_num}"

        fig = bokeh.plotting.figure(
            title=plot_name,
            title_location='above',
            x_axis_label='x',
            y_axis_label='y',
            width=800,
            height=800,
            match_aspect=True)
        fig.title.text_font_size = '11pt'
        
        walllines_x = [[l.bounds[0],l.bounds[2]] for l in wall_lines_struc]
        walllines_y = [[l.bounds[1],l.bounds[3]] for l in wall_lines_struc]
        walllines_x = [item for row in walllines_x for item in row]
        walllines_y = [item for row in walllines_y for item in row]
        pad_x_y = 0.5
        border_x, border_y = [min(walllines_x)-pad_x_y,max(walllines_x)+pad_x_y], [min(walllines_y)-pad_x_y,max(walllines_y)+pad_x_y]

        #--------------------------
        # grids. IfcColumn
        only_structural_grid = True
        grid_type = 'IfcColumn'
        (column_grid_elements, column_grid_components) = self.generate_grids(
            grid_type=grid_type,
            only_structural_grid=only_structural_grid,
            t_c_dist=t_c_dist,
            t_c_num=t_c_num,)
        
        column_grid_linestrings = self.get_grids_from_points(column_grid_elements, border_x=border_x, border_y=border_y)
        g_plot = self.visualization_settings['grids_column']
        for ls in column_grid_linestrings:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])

        # grids. IfcWall
        only_structural_grid = True
        grid_type = 'IfcWall'
        (wall_s_grid_elements, wall_s_grid_components) = self.generate_grids(
            grid_type=grid_type,
            only_structural_grid=only_structural_grid,
            t_w_dist=t_w_dist,
            t_w_num=t_w_num,)
        
        wall_s_grid_linestrings = self.get_grids_from_points(wall_s_grid_elements, border_x=border_x, border_y=border_y)
        g_plot = self.visualization_settings['grids_st_wall']
        for ls in wall_s_grid_linestrings:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])
        
        # grids. IfcWall-nonstructural.
        only_structural_grid = False
        grid_type = 'IfcWall'
        (wall_ns_grid_elements, wall_ns_grid_components) = self.generate_grids(
            grid_type=grid_type,
            only_structural_grid=only_structural_grid,
            t_w_dist=t_w_dist,
            t_w_num=t_w_num,)
        
        wall_ns_grid_linestrings = self.get_grids_from_points(wall_ns_grid_elements, border_x=border_x, border_y=border_y)
        g_plot = self.visualization_settings['grids_ns_wall']
        for ls in wall_ns_grid_linestrings:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])
        
        #--------------------------
        # columns
        g_plot = self.visualization_settings['points_column']
        for point in column_points:
            fig.square(point.x, point.y, legend_label=g_plot['legend_label'], size=g_plot['size'], color=g_plot['color'], alpha=g_plot['alpha'])
        
        # structural walls
        g_plot = self.visualization_settings['lines_st_wall']
        for ls in wall_lines_struc:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])
        
        # non-structural walls
        g_plot = self.visualization_settings['lines_ns_wall']
        for ls in wall_lines_nonst:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])

        fig.xgrid.visible = False
        fig.ygrid.visible = False

        if plot_fig:

            bokeh.plotting.output_file(filename=os.path.join(self.out_fig_path, fig_save_name + ".html"), title=fig_save_name)
            bokeh.plotting.save(fig)
            
        #--------------------------

        grids_per_storey = {
            "structural": {
                "IfcColumn": [column_grid_linestrings, column_grid_components],
                "IfcWall": [wall_s_grid_linestrings, wall_s_grid_components]
            },
            "non-structural":{
                "IfcWall": [wall_ns_grid_linestrings, wall_ns_grid_components]
            }}
        
        self.grids.update({storey.GlobalId: grids_per_storey})

    def align_same_type(self, grid_linestrings, grid_componnets, tol=0.0):
        
        # find all the pairs.
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

    def adjust_grids_per_storey(
        self,
        storey,
        t_self_dist=0.001,
        t_cross_dist=0.4,
        plot_fig=True,
        ):

        # get grids per storey.
        if storey.GlobalId in self.grids.keys():
            grids_per_storey = self.grids[storey.GlobalId]

        #---------------------------------------------------------------------------------------------------
        # Structural merge: merge overlapping structural grids from IfcColumn and IfcWall.
        gd_type = "structural" 
        st_grids_linestrings =  grids_per_storey[gd_type]["IfcColumn"][0] + grids_per_storey[gd_type]["IfcWall"][0]
        st_grids_componnets =  grids_per_storey[gd_type]["IfcColumn"][1] + grids_per_storey[gd_type]["IfcWall"][1]
        
        st_grids_linestrings_merged, st_grids_componnets_merged = self.align_same_type(
            grid_linestrings=st_grids_linestrings, grid_componnets=st_grids_componnets, tol=t_self_dist)

        self.grids[storey.GlobalId][gd_type].update({"self-merged": [st_grids_linestrings_merged, st_grids_componnets_merged]})
        
        #---------------------------------------------------------------------------------------------------
        # Non-structural merge: merge overlapping non-structural grids from  IfcWall.
        gd_type = "non-structural"
        ns_grids_linestrings =  grids_per_storey[gd_type]["IfcWall"][0]
        ns_grids_componnets =  grids_per_storey[gd_type]["IfcWall"][1]

        ns_grids_linestrings_merged, ns_grids_componnets_merged = self.align_same_type(
            grid_linestrings=ns_grids_linestrings, grid_componnets=ns_grids_componnets, tol=t_self_dist)

        self.grids[storey.GlobalId][gd_type].update({"self-merged": [ns_grids_linestrings_merged, ns_grids_componnets_merged]})

        #---------------------------------------------------------------------------------------------------
        # Align the Non-structural to structural: remove non-structural grids close to neighboring (merged) structural grids.
        gd_type = "non-structural"
        ns_grids_linestrings_merged =  grids_per_storey[gd_type]["self-merged"][0]
        ns_grids_componnets_merged =  grids_per_storey[gd_type]["self-merged"][1]

        aligned_ns_to_st=[]

        for ii, gd_st in enumerate(st_grids_linestrings_merged):

            for jj, gd_ns in enumerate(ns_grids_linestrings_merged):
                
                # if not aligned yet with structural grids.
                if jj not in aligned_ns_to_st:

                    # if parallel and too close < t_cross_dist.
                    if not shapely.intersects(gd_st,gd_ns) and shapely.distance(gd_st,gd_ns) < t_cross_dist:
                        
                        if ns_grids_componnets_merged[jj] not in st_grids_componnets_merged[ii]:
                            print (shapely.distance(gd_st,gd_ns))
                            st_grids_componnets_merged[ii]+=ns_grids_componnets_merged[jj]
                            aligned_ns_to_st.append(jj)
                
                else:
                    continue
        
        # final update of the merge.
        ns_grids_linestrings_merged = [e for i, e in enumerate(ns_grids_linestrings_merged) if i not in aligned_ns_to_st]
        ns_grids_componnets_merged = [e for i, e in enumerate(ns_grids_componnets_merged) if i not in aligned_ns_to_st]
        self.grids[storey.GlobalId]["structural"].update({"cross-merged": [st_grids_linestrings_merged, st_grids_componnets_merged]})
        self.grids[storey.GlobalId]["non-structural"].update({"cross-merged": [ns_grids_linestrings_merged, ns_grids_componnets_merged]})

        # =========================== visualization
        (wall_lines_struc,wall_lines_nonst,column_points) = self.get_info_elements_per_storey(storey=storey)
        
        plot_name = f"\[Floor \, Plan \, of \, {storey.Name} \, (T_{{self,dist}}={t_self_dist}, \, T_{{cross,dist}}={t_cross_dist}) - Gird \, Alignment \]"
        fig_save_name = f"Merge_{storey.Name}_t_self_dist_{t_self_dist}_t_cross_dist_{t_cross_dist}"

        fig = bokeh.plotting.figure(
            title=plot_name,
            title_location='above',
            x_axis_label='x',
            y_axis_label='y',
            width=800,
            height=800,
            match_aspect=True)
        fig.title.text_font_size = '11pt'

        #--------------------------
        # structural grids.
        st_grids_linestrings_merged = self.grids[storey.GlobalId]["structural"]["cross-merged"][0]
        g_plot = self.visualization_settings['grids_st_merged']
        for ls in st_grids_linestrings_merged:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])
            
        # non-structural grids.
        ns_grids_linestrings_merged = self.grids[storey.GlobalId]["non-structural"]["cross-merged"][0]
        g_plot = self.visualization_settings['grids_ns_merged']
        for ls in ns_grids_linestrings_merged:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])

        #--------------------------
        # columns
        g_plot = self.visualization_settings['points_column']
        for point in column_points:
            fig.square(point.x, point.y, legend_label=g_plot['legend_label'], size=g_plot['size'], color=g_plot['color'], alpha=g_plot['alpha'])
        
        # structural walls
        g_plot = self.visualization_settings['lines_st_wall']
        for ls in wall_lines_struc:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])
        
        # non-structural walls
        g_plot = self.visualization_settings['lines_ns_wall']
        for ls in wall_lines_nonst:
            x, y = ls.coords.xy
            fig.line(x, y, legend_label=g_plot['legend_label'], color=g_plot['color'], line_dash=g_plot['line_dash'], line_width=g_plot['line_width'], alpha=g_plot['alpha'])

        fig.xgrid.visible = False
        fig.ygrid.visible = False

        if plot_fig:
            bokeh.plotting.output_file(filename=os.path.join(self.out_fig_path, fig_save_name + ".html"), title=fig_save_name)
            bokeh.plotting.save(fig)

# ===============================================================
# to improve.

    def connect_wall_location_points(self, wall_locations_2d, dist_per_bin=2):

        merged_points = []
        points = np.array([item for sublist in wall_locations_2d for item in sublist])

        # Calculate pairwise distances
        distances = pdist(points, metric='euclidean') # num_pts * (num_pts-1) / 2
        distance_matrix = squareform(distances) # num_pts * num_pts
        num_bins = int(distances.shape[0]/dist_per_bin)
        hist, bin_edges = np.histogram(distances, bins=num_bins)
                
        gaps = []
        for i in range(len(hist)):
            if hist[i] == 0 :
                gap_range = [bin_edges[i], bin_edges[i+1]]
                gaps.append(gap_range)

        # # Find peaks (peak bins).
        # peaks, _ = find_peaks(hist)
        # peak_values = 0.5 * (bin_edges[peaks] + bin_edges[peaks + 1])

        # Histogram plot.
        fig = plt.figure(figsize=(12, 7))  # unit of inch
        ax = plt.axes((0.075, 0.075, 0.90, 0.85))  # in range (0,1)
        ax.hist(distances, bins=num_bins, color='#bcbd22', edgecolor='black', label=str(num_bins), alpha=0.8)
        plt.savefig('hist_'+str(num_bins)+'.png', dpi=200)

        # Find 'gaps' among peaks.
        threshold = gaps[0][0]
        
        # Use DBSCAN for clustering, eps is set to the threshold
        # threshold = np.percentile(distances, threshold_percentile)
        dbscan = DBSCAN(eps=threshold, min_samples=1, metric='euclidean')
        clusters = dbscan.fit_predict(points)
        
        # are those merged ones or 
        merged_points = np.array([points[clusters == c].mean(axis=0) for c in set(clusters)])

        return merged_points.tolist()

#===================================================================================================
    # plotting

    # # merged points.?
        # wall_points = [Point(wall_location_pt) for wall_loc in wall_locations for wall_location_pt in wall_loc]
    # # wall_points_connected = self.connect_wall_location_points(wall_locations)
    # # wall_points_connected = [Point(p) for p in wall_points_connected]
    # # wall_multipoints = MultiPoint(wall_points_connected)

    # # find the boundary.
    # # interval = 2
    # # kwargs = {"cap_style": CAP_STYLE.square, "join_style": JOIN_STYLE.mitre}
    # # boundary = wall_multipoints.buffer(interval/2, **kwargs).buffer(-interval/2, **kwargs)
    
    # # convex hull. far from perfect.
    # # convex_hull = wall_multipoints.convex_hull
    # # convex_points_x, convex_points_y = convex_hull.exterior.xy


    # for merge.
    # for point in wall_points:
    #     fig.square(point.x, point.y, legend_label="wall points", size=2, color="maroon", alpha=1)

    # for point in wall_points_connected:
    #     fig.square(point.x, point.y, legend_label="merged points", size=2, color="green", alpha=1)
    
    # for i in range(len(convex_points_x)):
    #     fig.square(convex_points_x[i], convex_points_y[i], legend_label="convex points", size=5, color="navy", alpha=0.8)

    # # save TO BE SOLVED.
    # fig.background_fill_color = None
    # fig.border_fill_color = None
    # bokeh.io.export_png(fig, filename="plan.png")

    # xs, ys = self.extract_polygon_coords(boundary)
    # fig.patches(xs, ys, fill_alpha=0.2, line_color="grey", line_width=1)

#===================================================================================================
    # --------------------for existing grids.
    # def calc_grid_location(self, grid):

    #     location = []
    #     grid_items= grid.Representation.Representations[0].Items
    #     for it in grid_items:
    #         location.append([[[*p[0].Coordinates],[*p[1].Coordinates]] for p in it.Elements[0]])

    #     # grid_line_2 = grid.ObjectPlacement
    #     # grid_placement = 
    #     #         if r.RepresentationIdentifier =='FootPrint':
    #     return location
    
    # def get_grid_info(self):

    #     self.grid_info = []

    #     for grid in self.grids:
    #         grid_location = self.calc_grid_location(grid)
    #         grid_evlevation = self.get_object_elevation(grid)
    #         self.grid_info.append({
    #             "id": grid.GlobalId,
    #             "location": grid_location,
    #             "elevation": grid_evlevation,
    #         })

#===================================================================================================
    # some ifc connecting ... assuming there's no connecting elements can be directly used.
    # IfcRelConnectsElements
    # IfcRelConnectsPathElements
    # IfcRelConnectsStructuralElement
    # IfcRelConnectsWithEccentricity
    # IfcRelAggregates
    # IfcRelContainedInSpatialStructure

    # def extract_polygon_coords(self, multipolygon):
    #     xs = []
    #     ys = []
    #     if multipolygon.is_empty:
    #         return xs, ys
    #     else:
    #         exterior_coords = multipolygon.envelope.exterior.coords.xy
    #         xs.append(list(exterior_coords[0]))
    #         ys.append(list(exterior_coords[1]))
    #         return xs, ys
    
#===================================================================================================
#geometry analysis

    



    # def calc_shape_volume(self, shape):
    #     props = OCC.Core.GProp.GProp_GProps()
    #     OCC.Core.BRepGProp.brepgprop.SurfaceProperties(shape.geometry, props)
    #     return props.Mass()

    # def calc_shape_area(self, shape):
    #     props = OCC.Core.GProp.GProp_GProps()
    #     OCC.Core.BRepGProp.brepgprop.VolumeProperties(shape.geometry, props)
    #     return props.Mass()

    # def calc_wall_volume(self, wall):
    #     shape = self.get_wall_shape(wall)
    #     volume = self.calc_shape_volume(shape)
    #     return volume
    
    # def calc_wall_area(self, wall):
    #     shape = self.get_wall_shape(wall)
    #     area = self.calc_shape_area(shape)
    #     return area


    # def get_wall_volume(self, wall):

    #     psets = ifcopenshell.util.element.get_psets(wall)
    #     if 'Dimensions' in psets.keys():
    #         if 'Volume' in psets['Dimensions'].keys():
    #             volume = psets['Dimensions']['Volume']
    #             return volume
    #         else:
    #             return None
    #     else:
    #         return None
        
    # def get_wall_length(self, wall):

    #     psets = ifcopenshell.util.element.get_psets(wall)

    #     if 'Dimensions' in psets.keys():
    #         if 'Length' in psets['Dimensions'].keys():
    #             volume = psets['Dimensions']['Length']
    #             return volume
    #         else:
    #             return None
    #     else:
    #         return None
        
        # print(ifcopenshell.util.element.get_psets(wall, psets_only=True))
        # print(ifcopenshell.util.element.get_psets(wall, qtos_only=True))

        # settings = ifcopenshell.geom.settings()
        # shape = ifcopenshell.geom.create_shape(settings, wall)

        # print(shape.guid)
        # print(shape.id)
        # print(shape.geometry.id)

        # # A 4x4 matrix representing the location and rotation of the element, in the form:
        # # [ [ x_x, y_x, z_x, x   ]
        # #   [ x_y, y_y, z_y, y   ]
        # #   [ x_z, y_z, z_z, z   ]
        # #   [ 0.0, 0.0, 0.0, 1.0 ] ]
        # # The position is given by the last column: (x, y, z)
        # # The rotation is described by the first three columns, by explicitly specifying the local X, Y, Z axes.
        # # The first column is a normalised vector of the local X axis: (x_x, x_y, x_z)
        # # The second column is a normalised vector of the local Y axis: (y_x, y_y, y_z)
        # # The third column is a normalised vector of the local Z axis: (z_x, z_y, z_z)
        # # The axes follow a right-handed coordinate system.

        # # Objects are never scaled, so the scale factor of the matrix is always 1.
        # matrix = shape.transformation.matrix.data
        # # For convenience, you might want the matrix as a nested numpy array, so you can do matrix math.
        # matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
        # # You can also extract the XYZ location of the matrix.
        # location = matrix[:,3][0:3]

        # # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
        # verts = shape.geometry.verts

        # # Indices of vertices per edge e.g. [e1v1, e1v2, e2v1, e2v2, ...]
        # # If the geometry is mesh-like, edges contain the original edges.
        # # These may be quads or ngons and not necessarily triangles.
        # edges = shape.geometry.edges

        # # Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]
        # # Note that faces are always triangles.
        # faces = shape.geometry.faces

        # # Since the lists are flattened, you may prefer to group them like so depending on your geometry kernel
        # # A nested numpy array e.g. [[v1x, v1y, v1z], [v2x, v2y, v2z], ...]
        # grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        # # A nested numpy array e.g. [[e1v1, e1v2], [e2v1, e2v2], ...]
        # grouped_edges = ifcopenshell.util.shape.get_edges(shape.geometry)
        # # A nested numpy array e.g. [[f1v1, f1v2, f1v3], [f2v1, f2v2, f2v3], ...]
        # grouped_faces = ifcopenshell.util.shape.get_faces(shape.geometry)
                

        # settings = ifcopenshell.geom.settings()
        # settings.set(settings.USE_PYTHON_OPENCASCADE, True)
        
        # #f1
        # # product = ifcopenshell.geom.create_shape(settings, wall)
        # # shape = OCC.Core.TopoDS.TopoDS_Iterator(product.geometry).Value()
        # # trsf = shape.geometry.Location().Transformation()
        # # trsf.TranslationPart().X(), trsf.TranslationPart().Y(), trsf.TranslationPart.Z()

        # #f2
        # settings2 = ifcopenshell.geom.settings()
        # product = ifcopenshell.geom.create_shape(settings2, wall)
        # print (tuple(product.transformation.matrix.data))
        

    
    # def normalize(self, li):
    #     mean = np.mean(list(li))
    #     std = np.std(list(li))
    #     nor = abs(li-mean) / std
    #     return nor

    # def plot(self, model_path):

    #     ifc_file = ifcopenshell.open(model_path)
    #     walls = ifc_file.by_type("IfcWall")

    #     settings = ifcopenshell.geom.settings()
    #     settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    #     settings.set(settings.USE_WORLD_COORDS, True)
    #     settings.set(settings.INCLUDE_CURVES, True)

    #     wall_shapes = []
    #     bbox = OCC.Core.Bnd.Bnd_Box()

    #     occ_display = ifcopenshell.geom.utils.initialize_display()

    #     for wall in walls:

    #         shape = ifcopenshell.geom.create_shape(settings, wall).geometry
    #         tempo0 = wall.ObjectPlacement.RelativePlacement.Location
    #         tempo1 = wall.ObjectPlacement.RelativePlacement.Location.Coordinates
    #         tempo2 = wall.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios
    #         print(tempo1)
    #         wall_shapes.append((wall, shape))  
            
    #         ifcopenshell.geom.utils.display_shape(shape)
        
    #     occ_display.FitAll()
    #     ifcopenshell.geom.utils.main_loop()

    #     settings = ifcopenshell.geom.settings()
    #     settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    #     settings.set(settings.USE_WORLD_COORDS, True)
    #     settings.set(settings.INCLUDE_CURVES, True)

    #     # get the shape geometry by creating the shape.
    #     # wall_representation_axis = wall.Representation.Representations[0]
    #     # wall_representation_boday = wall.Representation.Representations[1]
    #     # wall_pnts = (wall_representation_axis.Items[0].Points[0].Coordinates,wall_representation_axis.Items[0].Points[1].Coordinates)

    #     # occ display initialization.
    #     # occ_display = ifcopenshell.geom.utils.initialize_display()
    #     # occ_display.FitAll() # Fit the model into view
    #     # ifcopenshell.geom.utils.main_loop() # Allow for user interaction

    #     shape = ifcopenshell.geom.create_shape(settings, wall).geometry
        
    #     # ==================================
    #     # List to store the faces of the wall
    #     exp_face = OCC.Core.TopExp.TopExp_Explorer(shape, OCC.Core.TopAbs.TopAbs_FACE)
    #     wall_faces = []

    #     while exp_face.More():
    #         face = exp_face.Current()
    #         # face = OCC.Core.TopoDS.topods.Face(exp_face.Current())
    #         wall_faces.append(face)
    #         exp_face.Next()

    #     for face in wall_faces:
    #         print("Face :", face)

        # # ==================================
        # # List to store the edges (axes) of the wall
        # exp_edge = OCC.Core.TopExp.TopExp_Explorer(shape, OCC.Core.TopAbs.TopAbs_EDGE)
        # wall_edges = []

        # while exp_edge.More():
        #     edge = OCC.Core.TopoDS.topods_Edge(exp_edge.Current())
        #     wall_edges.append(edge)
        #     exp_edge.Next()

        # for edge in wall_edges:
        #     print("Edge (Axis):", edge)
            
        # # ==================================
        # # List to store the vertices of the wall
        # exp_vertice = OCC.Core.TopExp.TopExp_Explorer(shape, OCC.Core.TopAbs.TopAbs_VERTEX)
        # wall_vertices = []

        # while exp_vertice.More():
        #     vertice = OCC.Core.TopoDS.TopoDS_Vertex(exp_vertice.Current())
        #     wall_vertices.append(vertice)
        #     exp_vertice.Next()

        # for vertex in wall_vertices:
        #     print("Vertex:", vertex)

        # pt = OCC.Core.BRep.Pnt(vertex)
        # print("Point:", pt)

    
        # shape = ifcopenshell.geom.create_shape(settings, wall).geometry
        # # shape = ifcopenshell.geom.create_shape(settings, wall_axis_representation)

        # # get the single edge of your wall axis representation
        # exp1 = OCC.Core.TopExp.TopExp_Explorer(shape, OCC.Core.TopAbs.TopAbs_FACE)
        
        # # get vertices
        # exp2 = OCC.Core.TopExp.topexp_Vertices(shape, OCC.Core.TopAbs.TopAbs_VERTEX)
        
        # # get the points associated to the vertices.
        # exp3 = OCC.Core.BRep.Pnt(shape)

        # out = []
        # display_shape = ifcopenshell.geom.utils.display_shape(shape)

        # faces = shape.geometry.faces
        # face = faces[0]
        
        # # calculate the wall volume and area.
        # volume = self.calc_volume(shape) 
        # area = self.calc_area(shape)

        # # feature = self.normalize(map(operator.truediv, area, volume))
    
        # # color = RED if feature > 1. else GRAY
        # # ifcopenshell.geom.utils.display_shape(shape, clr = color)
        
        # exp = OCC.Core.TopExp.TopExp_Explorer(shape, OCC.Core.TopAbs.TopAbs_FACE)
        # while exp.More():
        #     face = OCC.Core.TopoDS.topods.Face(exp.Current())
        #     prop = OCC.Core.BRepGProp.BRepGProp_Face(face)
        #     p = OCC.gp.gp_Pnt()
        #     normal_direction = OCC.gp.gp_Vec()
        #     prop.Normal(0.,0.,p,normal_direction)
        #     if abs(1. - normal_direction.Z()) < 1.e-5:
        #         ifcopenshell.geom.utils.display_shape(face)
        #     exp.Next()

        # # Fit the model into view
        # occ_display.FitAll()
        
        # # # Allow for user interaction
        # ifcopenshell.geom.utils.main_loop()


        # exp = OCC.Core.TopoDS.TopoDS_Shape.Location()
    
        # product = ifcopenshell.geom.create_shape(settings, wall)
        # representation = wall.Representation.Axis
        # shape = ifcopenshell.geom.create_shape(wall, representation)
        
        # # to get a single edge of your wall axis representation (ideally there should only be one).
        # OCC.Core.TopExp.TopExp_Explorer

        # # to get its vertices.
        # OCC.Core.TopExp.topexp_Vertices

        # # to get the points associated to the vertices.
        # OCC.Core.BRep.Pnt 
        
            #     placement = object_placement.RelativePlacement
            #     if placement and placement.is_a("IfcAxis2Placement3D"):
            #         # Extract the transformation matrix
            #         matrix = placement.Axis2Placement.Matrix
            #         if matrix:
            #             # Extract the orientation components
            #             x_direction = matrix[0][0]
            #             y_direction = matrix[1][0]
            #             z_direction = matrix[2][0]
            #             print(f"Orientation of Wall {wall.Name}:")
            #             print(f"X Direction: {x_direction}")
            #             print(f"Y Direction: {y_direction}")
            #             print(f"Z Direction: {z_direction}")
            # elif object_placement.is_a("IfcGridPlacement"):
            #     # Handle other types of placements as needed
            #     pass

        # halfspaces = []

        # for wall, shape in wall_shapes:

        #     # topo = OCC.Utils.Topo(shape) # old
        #     # topo = OCC.Extend.TopologyUtils.TopologyExplorer(shape)

        #     exp_face = OCC.Core.TopExp.TopExp_Explorer(shape, OCC.Core.TopAbs.TopAbs_FACE)
        #     wall_faces = []

        #     while exp_face.More():
        #         # face = exp_face.Current()
        #         face = OCC.Core.TopoDS.topods_Edge(exp_face.Current())
        #         wall_faces.append(face)
        #         exp_face.Next()
                
        #     for face in wall_faces:
        #         surf = OCC.Core.BRep.BRep_Tool.Surface(face)
        #         obj = surf.GetObject()
        #         assert obj.DynamicType().GetObject().Name() == "Geom_Plane"
                
        #         plane = OCC.Core.Geom.Handle_Geom_Plane.DownCast(surf).GetObject()
                
        #         if plane.Axis().Direction().Z() == 0:
        #             face_bbox = OCC.Core.Bnd.Bnd_Box()
        #             OCC.Core.BRepBndLib.brepbndlib_Add(face, face_bbox)
        #             face_center = ifcopenshell.geom.utils.get_bounding_box_center(face_bbox).XYZ()
                    
        #             face_normal = plane.Axis().Direction().XYZ()
        #             face_towards_center = bounding_box_center.XYZ() - face_center
        #             face_towards_center.Normalize()
                    
        #             dot = face_towards_center.Dot(face_normal)
                    
        #             if dot < -0.8:
                        
        #                 ifcopenshell.geom.utils.display_shape(face)
                        
        #                 face_plane = plane.Pln()
        #                 new_face = OCC.Core.BRepBuilderAPI.BRepBuilderAPI_MakeFace(face_plane).Face()
        #                 halfspace = OCC.Core.BRepPrimAPI.BRepPrimAPI_MakeHalfSpace(
        #                     new_face, bounding_box_center).Solid()
        #                 halfspaces.append(halfspace)

        # for wall in walls:
        #     if wall.Representation:
        #         # tempo = wall.Representation
        #         shape = ifcopenshell.geom.create_shape(settings, wall).geometry
        #         # OCC.Core.BRepBndLib.brepbndlib_Add(shape, bbox)
        #         # display_shape = ifcopenshell.geom.utils.display_shape(shape)
        
        # check: https://sourceforge.net/p/ifcopenshell/discussion/1782716/thread/014c820c23/
        # check: https://sourceforge.net/p/ifcopenshell/discussion/1782717/thread/409ef11620/

