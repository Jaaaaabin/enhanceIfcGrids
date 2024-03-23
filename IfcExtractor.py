import ifcopenshell
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from wallExtractor import WallWidthExtractor
from quickTools import time_decorator, remove_duplicate_dicts

# import ifcopenshell.geom
# import ifcopenshell.util.shape
# import OCC.Core.TopExp,OCC.Core.TopAbs,OCC.Core.TopoDS,OCC.Core.BRepBndLib,OCC.Core.BRep

#===================================================================================================
#IfcGeneral ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
class IfcExtractor:

    def __init__(self, model_path, figure_path):

        try:
            self.model = ifcopenshell.open(model_path)
            self.ifc_file_name = os.path.basename(model_path)
            self.out_fig_path = figure_path
            os.makedirs(figure_path, exist_ok=True)

            self.version = self.model.schema
            self.project = []
            self.site = []
            self.storeys = []
            self.slabs = []
            self.spaces = []
            self.columns = []
            self.walls = []
            self.curtainwalls = []
            self.doors = []
            self.windows = []

            self.existing_grids = []
            self.grids = {}
            self.extract_all_ifc_elements()
            
            print ("=============IfcExtractor=============")
            print (self.ifc_file_name)
            
        except ifcopenshell.errors.FileNotFoundError: # type: ignore
            print(f"Error: File '{model_path}' not found.")

        except Exception as e:
            print(f"An error occurred: {e}")
    
    def extract_all_ifc_elements(self):

        """
        Extracts various IFC entities from the model and initializes visualization settings.
        """

        if self.model:

            # self.project = self.model.by_type("IfcProject")
            # self.site = self.model.by_type("IfcSite")
            # self.doors = self.model.by_type("IfcDoor")
            # self.windows = self.model.by_type("IfcWindow")
            # self.existing_grids = self.model.by_type("IfcGrid")

            self.storeys = self.model.by_type("IfcBuildingStorey")
            self.slabs = self.model.by_type("IfcSlab")
            self.spaces = self.model.by_type("IfcSpace")
            self.columns = self.model.by_type("IfcColumn")
            self.walls = self.model.by_type("IfcWall") + self.model.by_type('IfcWallStandardCase')
            self.curtainwalls = self.model.by_type("IfcCurtainWall")
            
    def write_dict_columns(self):

        dict_info_columns = remove_duplicate_dicts(self.info_columns)
        try:
            with open(os.path.join(self.out_fig_path, 'info_columns.json'), 'w') as json_file:
                json.dump(dict_info_columns, json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'columns.json'}: {e}")

    def write_dict_walls(self):
        
        dict_info_walls = remove_duplicate_dicts(self.info_walls)
        try:
            with open(os.path.join(self.out_fig_path, 'info_walls.json'), 'w') as json_file:
                json.dump(dict_info_walls, json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'columns.json'}: {e}")
    
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

#IfcGeneral ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#display ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    def wall_width_histogram(self):
        
        values = [w['width'] for w in self.info_walls if 'width' in w]
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

        values = [w['length'] for w in self.info_walls if 'length' in w]

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
        values = [w['location'] for w in self.info_walls if 'location' in w]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')  # type: ignore

        for v in values:
            start_point, end_point = v
            xs, ys, zs = zip(start_point, end_point)
            ax.plot(xs, ys, zs, marker='o', color='black', linewidth=1, markersize=3)

        plt.savefig(os.path.join(self.out_fig_path, 'wall_location_map.png'), dpi=200)
        plt.close(fig)

    def wall_orientation_histogram(self):
        values = [w['orientation'] for w in self.info_walls if 'orientation' in w]

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

    def wall_display(self):

        self.wall_width_histogram()
        self.wall_length_histogram()
        self.wall_location_map()
        self.wall_orientation_histogram()
        
#display ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#column ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    # function notes.
    def cacl_column_geometry(self,column):
        
        column_location, column_length = None, None

        if not column.Representation or not column.Representation.Representations:
            print("Column has no Representation or no Representation.Representations.")
        
        else:
            # have the 'Body' representation.
            for r in column.Representation.Representations:
                if r.RepresentationIdentifier == 'Body':
                    mapped_r = r.Items[0].MappingSource.MappedRepresentation
                    if mapped_r.RepresentationType=='SweptSolid' and all(hasattr(mapped_r.Items[0], attr) for attr in ["Depth", "Position"]):
                       
                        #length
                        column_length = mapped_r.Items[0].Depth
                        
                        #orientation
                        if mapped_r.Items[0].ExtrudedDirection and hasattr(mapped_r.Items[0].ExtrudedDirection,'DirectionRatios'):
                            column_orientation = mapped_r.Items[0].ExtrudedDirection.DirectionRatios
                        
                        #location
                        column_location_p1 = mapped_r.Items[0].Position.Location.Coordinates
                        if abs(column_location_p1[0])<0.001 and abs(column_location_p1[1])<0.001: ## to improve.
                            column_location = None
                        else:
                            column_location_p2 = tuple((direction_val * column_length) + p1_val for p1_val, direction_val in \
                                                    zip(column_location_p1, column_orientation))
                            column_location = [list(column_location_p1),list(column_location_p2)]
                    
                    elif mapped_r.RepresentationType=='Brep' or mapped_r.RepresentationType=='Tessellation' or mapped_r.RepresentationType=='AdvancedBrep':
                        # RepresentationType can also be 'Brep', 'Tessellation' and 'AdvancedBrep'. for TUM_Gebaude_models.
                        column_length = 100.0 # to be corrected.
                        column_orientation = (0.0, 0.0, 1.0) # to be corrected.
                    else:
                        print ("test")
                
            # how to get the column_orientation
                            
            # doesnt have the 'Body' representation.
            if column_location == None:
                if hasattr(column, "ObjectPlacement"):

                    column_location_p1 = column.ObjectPlacement.RelativePlacement.Location.Coordinates 
                    # if column_location_p1[-1] != column.ObjectPlacement.PlacementRelTo.PlacesObject[0].Elevation:
                    rel_storey_elevation = column.ObjectPlacement.PlacementRelTo.PlacesObject[0].Elevation
                    rel_placement = column.ObjectPlacement.PlacementRelTo.RelativePlacement.Location.Coordinates
                    column_location_p1 = tuple(sum(x) for x in zip(column_location_p1, rel_placement))
                    column_location_p1 = column_location_p1[:2] + (column_location_p1[2] + rel_storey_elevation,)
                    
                    column_location_p2 = tuple((direction_val * column_length) + p1_val for p1_val, direction_val in \
                                    zip(column_location_p1, column_orientation))  # type: ignore
                    column_location = [list(column_location_p1),list(column_location_p2)]
                else:
                    raise ValueError("column doesn't have ObjectPlacement.")
                    
        return column_location, column_length

    # function notes.
    def extract_a_column(self, ifc_column):
        
        info_a_column = dict()
        column_location, column_length = self.cacl_column_geometry(ifc_column)
        info_a_column.update({
            "id": ifc_column.GlobalId,
            "elevation":self.get_object_elevation(ifc_column),
            "location": column_location,
            "length": column_length,
        })
        
        return info_a_column

    @time_decorator
    def extract_all_columns(self):

        self.info_columns = []
        for c in self.columns:
            info_a_column = self.extract_a_column(c)
            self.info_columns.append(info_a_column)

#column ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#wall ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    # function notes.
    def calc_wall_geometry(self, wall):
        
        wall_location, wall_length = None, None
        
        try:
            # if wall.GlobalId=='0BZD7xrIL3h80Kncn0Xa2a':
            #     print("test")

            if not wall.Representation or not wall.Representation.Representations:
                print("Wall has no Representation or no Representation.Representations.")
            
            else:
                wall_elevation = wall.ContainedInStructure[0].RelatingStructure.Elevation
                wall_location_p1, wall_location_p2 = None, None

                orientation_deg = self.calc_wall_orientation(wall, deg_range=360)
                wall_location_p1 = wall.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(wall, "ObjectPlacement") else None
                
                if wall_location_p1 != None:
                    
                    # have the 'Axis' representation.
                    rHasAxis = False
                    for r in wall.Representation.Representations:
                        if r.RepresentationIdentifier =='Axis':
                            rHasAxis = True
                            wall_axis = r
                            if wall_axis.Items[0].is_a('IfcPolyline'):
                                wall_pnts = wall_axis.Items[0].Points
                                coord_wall_pnts = wall_pnts.CoordList if hasattr(wall_pnts, "CoordList") else (wall_pnts[0].Coordinates,wall_pnts[1].Coordinates)
                                wall_length = math.sqrt((coord_wall_pnts[1][0] - coord_wall_pnts[0][0])**2 + (coord_wall_pnts[1][1] - coord_wall_pnts[0][1])**2)
                            
                            elif wall_axis.Items[0].is_a('IfcTrimmedCurve'): #todo.
                                #============ for curve walls=======================
                                wall_length = 0 # use wall_length for curved ones yet.
                                #============ for curve walls=======================
                    
                    # don't have the 'Axis' representation.
                    if not rHasAxis:
                        for r in wall.Representation.Representations:
                            if r.RepresentationIdentifier =='Body':
                                wall_body = r
                                if wall_body.Items[0].is_a('IfcExtrudedAreaSolid') and hasattr(wall_body.Items[0], 'Depth'):
                                    wall_length = wall_body.Items[0].Depth

                    if orientation_deg != None and wall_length != None:
                        wall_location_p2 = (
                            wall_location_p1[0]+ wall_length*math.cos(math.radians(orientation_deg)),
                            wall_location_p1[1]+ wall_length*math.sin(math.radians(orientation_deg)),
                            wall_location_p1[2])
                        
                    # wall_length = self.calc_wall_length_by_pset(wall)
                    wall_location_p1 = [*wall_location_p1[:-1], wall_elevation]  # type: ignore
                    wall_location_p2 = [*wall_location_p2[:-1], wall_elevation]  # type: ignore
                    wall_location = [list(wall_location_p1),list(wall_location_p2)]
        
        except AttributeError as e:
            print(f"calc_wall_geometry: An attribute error occurred: {e}")
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"calc_wall_geometry: An unexpected error occurred: {e}")

        return wall_location, wall_length
    
    # function notes.
    def calc_wall_loadbearing(self, wall):
        
        psets = ifcopenshell.util.element.get_psets(wall)
        if 'Pset_WallCommon' in psets.keys():
            if 'LoadBearing' in psets['Pset_WallCommon'].keys():
                load_bearing = psets['Pset_WallCommon']['LoadBearing']
                return load_bearing
            else:
                return None
        else:
            return None
    
    # function notes.
    def calc_wall_orientation(self, wall, deg_range=360):
            
        orientation_deg = None

        if wall.ObjectPlacement.RelativePlacement.RefDirection != None:
            orientation_vec = wall.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios
            orientation_rad = math.atan2(orientation_vec[1],orientation_vec[0])
            orientation_deg = math.degrees(orientation_rad) % deg_range
        else:
            orientation_deg = 0.0

        return round(orientation_deg, 4)
    
    # function notes.
    def extract_a_wall(self, ifc_wall):
        
        # ifc4.X vs ifc2x3. missing ifcwallstandardcase entities.
        info_a_wall = dict()

        wall_location, wall_length = self.calc_wall_geometry(ifc_wall)
        self.WallWidthExtractor = WallWidthExtractor(ifc_wall)
        self.WallWidthExtractor.calc_wall_width_from_representation()
    
        info_a_wall.update({
            "id": ifc_wall.GlobalId,
            "loadbearing": self.calc_wall_loadbearing(ifc_wall),
            "elevation": self.get_object_elevation(ifc_wall),
            "orientation": self.calc_wall_orientation(ifc_wall),
            "location": wall_location,
            "length": round(wall_length,4),
            "width":round(self.WallWidthExtractor.width,4), # type: ignore
        })

        return info_a_wall

    @time_decorator
    def extract_all_walls(self):

        self.info_walls = []
        for w in self.walls:
            info_a_wall = self.extract_a_wall(w)
            self.info_walls.append(info_a_wall)

#wall ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================


#===================================================================================================
#curtainwall ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
# ??????????????????????????????????????????????????????????????????????????????????????????????????
#curtainwall ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================


#===================================================================================================
#alloldwall ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
# to check again and make reuse of them.
            
    # def connect_wall_location_points(self, wall_locations_2d, dist_per_bin=2):

    #     merged_points = []
    #     points = np.array([item for sublist in wall_locations_2d for item in sublist])

    #     # Calculate pairwise distances
    #     distances = pdist(points, metric='euclidean') # num_pts * (num_pts-1) / 2
    #     distance_matrix = squareform(distances) # num_pts * num_pts
    #     num_bins = int(distances.shape[0]/dist_per_bin)
    #     hist, bin_edges = np.histogram(distances, bins=num_bins)
                
    #     gaps = []
    #     for i in range(len(hist)):
    #         if hist[i] == 0 :
    #             gap_range = [bin_edges[i], bin_edges[i+1]]
    #             gaps.append(gap_range)

    #     # # Find peaks (peak bins).
    #     # peaks, _ = find_peaks(hist)
    #     # peak_values = 0.5 * (bin_edges[peaks] + bin_edges[peaks + 1])

    #     # Histogram plot.
    #     fig = plt.figure(figsize=(12, 7))  # unit of inch
    #     ax = plt.axes((0.075, 0.075, 0.90, 0.85))  # in range (0,1)
    #     ax.hist(distances, bins=num_bins, color='#bcbd22', edgecolor='black', label=str(num_bins), alpha=0.8)
    #     plt.savefig('hist_'+str(num_bins)+'.png', dpi=200)

    #     # Find 'gaps' among peaks.
    #     threshold = gaps[0][0]
        
    #     # Use DBSCAN for clustering, eps is set to the threshold
    #     # threshold = np.percentile(distances, threshold_percentile)
    #     dbscan = DBSCAN(eps=threshold, min_samples=1, metric='euclidean')
    #     clusters = dbscan.fit_predict(points)
        
    #     # are those merged ones or 
    #     merged_points = np.array([points[clusters == c].mean(axis=0) for c in set(clusters)])

    #     return merged_points.tolist()
    
    # def calc_wall_length_by_pset(self, wall):
    #     """Gets the length of a wall from its property sets."""
        
    #     wall_length = None
        
    #     # what about the IfcStandardcaseWall?
    #     if not wall.is_a('IfcWall') and not wall.is_a('IfcCurtainWall') and not wall.is_a('IfcWallStandardCase'):
    #         return 0.0

    #     wall_pset = ifcopenshell.util.element.get_psets(wall)
    #     if wall.is_a('IfcWall') or wall.is_a('IfcWallStandardCase'):
    #         for pset_key in wall_pset:
    #             if 'Length' in wall_pset[pset_key]:
    #                 return round(wall_pset[pset_key]['Length'], 4)
        
    #     elif wall.is_a('IfcCurtainWall'):
    #         for pset_key in wall_pset:
    #             if 'Length' in wall_pset[pset_key]:
    #                 return round(wall_pset[pset_key]['Length'], 4)
    #     else:
    #         return wall_length

    # def find_farthest_linear_points(self, points):
        
    #     def distance_2d(point_a, point_b):
    #         """Calculate the Euclidean distance between two points in 2D."""
    #         return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    #     max_distance = 0
    #     farthest_linear_points = None

    #     for i in range(len(points)):
    #         for j in range(i + 1, len(points)):
    #             dist = distance_2d(points[i], points[j])
    #             if dist > max_distance:
    #                 max_distance = dist
    #                 farthest_linear_points = (points[i], points[j])

    #     return farthest_linear_points
    
    # def calc_wall_location(self, wall):
    #     """to replace the calc_wall_dimensions."""

    #     # wall, elevation------------------------

    #     if wall.ContainedInStructure[0].RelatingStructure.is_a('IfcBuildingStorey'):
    #         wall_elevation = wall.ContainedInStructure[0].RelatingStructure.Elevation
    #     else:
    #         wall_elevation = None

    #     local_points = None,
    #     global_location_0, global_location_1 = None, None

    #     #'IfcWall' or 'IfcWallStandardCase' conditions.
    #     if wall.is_a('IfcWall') or wall.is_a('IfcWallStandardCase'):

    #         # local
    #         for r in wall.Representation.Representations:
    #             if r.RepresentationIdentifier =='Axis':
    #                 wall_axis = r
    #                 if wall_axis.Items[0].is_a('IfcPolyline'):
    #                     wall_pnts = wall_axis.Items[0].Points
    #                     local_points = wall_pnts.CoordList if hasattr(wall_pnts, "CoordList") else (wall_pnts[0].Coordinates,wall_pnts[1].Coordinates)
                    
    #                 #============ for curve walls=======================
    #                 elif wall_axis.Items[0].is_a('IfcTrimmedCurve'):
                        
    #                     return [0,0,wall_elevation],[0,0,wall_elevation]
    #                 #============ for curve walls=======================

    #         # global
    #         global_location_0 = wall.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(wall, "ObjectPlacement") else None
    #         orientation_deg = self.calc_wall_orientation(wall,deg_range=360)
    #         wall_length = self.calc_wall_length_by_pset(wall)
    #         if orientation_deg != None and wall_length != None:
    #             global_location_1 = (
    #                 global_location_0[0]+ wall_length*math.cos(math.radians(orientation_deg)),
    #                 global_location_0[1]+ wall_length*math.sin(math.radians(orientation_deg)),
    #                 global_location_0[2])
    #         else:
    #             global_location_1 = None
                    
    #         global_location_0 = [*global_location_0[:-1], wall_elevation]
    #         global_location_1 = [*global_location_1[:-1], wall_elevation]

    #     # 'IfcCurtainWall' conditions.
    #     elif wall.is_a('IfcCurtainWall'):
            
    #         related_components = wall.IsDecomposedBy[0].RelatedObjects
    #         if len(related_components)==1 and related_components[0].is_a('IfcPlate'):
                
    #             # local
    #             component = related_components[0]
    #             for r in component.Representation.Representations:
    #                 if r.RepresentationIdentifier =='FootPrint':
    #                     # local_points = r.Items[0].MappingSource.MappedRepresentation.Items[0].Points.CoordList
    #                     local_points = r.Items[0].MappingSource.MappedRepresentation.Items[0].Points
    #                     if isinstance(local_points,tuple):
    #                         local_points = [list(c) for pt in local_points for c in pt]
    #                     print ("Footprint:",local_points)
        
    #             # global
    #             global_location_0 = component.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(component, "ObjectPlacement") else None
    #             orientation_deg = self.calc_wall_orientation(wall,deg_range=360)
    #             wall_length = self.calc_wall_length_by_pset(wall)
    #             if orientation_deg != None and wall_length != None:
    #                 global_location_1 = (
    #                     global_location_0[0]+ wall_length*math.cos(math.radians(orientation_deg)),
    #                     global_location_0[1]+ wall_length*math.sin(math.radians(orientation_deg)),
    #                     global_location_0[2])
    #             else:
    #                 global_location_1 = None
                        
    #             global_location_0 = [*global_location_0[:-1], wall_elevation]
    #             global_location_1 = [*global_location_1[:-1], wall_elevation]
                        
    #         elif len(related_components) > 1:
                
    #             # direct global
    #             all_placement_points = []
    #             for component in related_components:
    #                 placement_point = component.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(component, "ObjectPlacement") else None
    #                 all_placement_points.append(list(placement_point))
                
    #             local_points = self.find_farthest_linear_points(all_placement_points)
    #             global_location_0, global_location_1 = local_points
    #             global_location_0 = [*global_location_0[:-1], wall_elevation]
    #             global_location_1 = [*global_location_1[:-1], wall_elevation]

    #     return global_location_0,global_location_1

    # def get_wall_info(self):
            
    #     self.wall_info = []

    #     for wall in self.walls:
            
    #         print (wall.GlobalId)
    #         wall_loadbearing = self.get_wall_loadbearing(wall)
    #         wall_width = self.calc_wall_width(wall)
    #         wall_length = self.calc_wall_length_by_pset(wall)            
    #         wall_orientation = self.calc_wall_orientation(wall, deg_range=180, orien_dec=1) # 180 for orientation printing
    #         wall_location = self.calc_wall_location(wall)

            
    #         # wall_width = None
    #         # if wall_width_by_geometry is None and wall_width_by_material is None:
    #         #     wall_width = 0.0
    #         # elif wall_width_by_geometry is None:
    #         #     wall_width = max(0.0, wall_width_by_material)
    #         # elif wall_width_by_material is None:
    #         #     wall_width = max(0.0, wall_width_by_geometry)
    #         # else:
    #         #     wall_width = max(wall_width_by_geometry, wall_width_by_material)

    #         # wall_length, wall_location = self.calc_wall_dimensions(wall, length_dec=2)

    #         self.wall_info.append({
    #             "id": wall.GlobalId,
    #             "loadbearing":wall_loadbearing,
    #             "width": wall_width,
    #             "orientation": wall_orientation,
    #             "length": wall_length,
    #             "location": wall_location,
    #         })

#alloldwall ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================     

#===================================================================================================
#space

    # def calc_space_location(self, space):
        
    #     location = space.ObjectPlacement.RelativePlacement.Location.Coordinates if hasattr(space, "ObjectPlacement") else None
    #     return location
    
    # def get_space_info(self):
            
    #     space_info = []
    #     for space in self.spaces:
    #         space_location = self.calc_space_location(space)
    #         space_info.append({
    #             "id": space.id(),
    #             "location": space_location, # not working.
    #         })
    #     return space_info


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

