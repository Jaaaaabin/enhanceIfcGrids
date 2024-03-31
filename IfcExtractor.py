import multiprocessing
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from quickTools import time_decorator
from quickTools import remove_duplicate_dicts, find_most_common_value
from quickTools import get_rectangle_corners, distance_between_points

#===================================================================================================
#IfcGeneral ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
class IfcExtractor:

    def __init__(self, model_path, figure_path, read='triangle'):

        try:

            self.type_geo = read
            self.settings = self._configure_settings()

            self.model = ifcopenshell.open(model_path)

            self.ifc_file_name = os.path.basename(model_path)
            self.out_fig_path = figure_path
            os.makedirs(figure_path, exist_ok=True)

            self.version = self.model.schema

            # Initialization of lists for various IFC elements
            self.project = []
            self.site = []
            self.storeys = []
            self.spaces = []

            self.columns = []
            self.walls = []
            self.curtainwalls = []
            self.plates = []
            self.members = []

            self.slabs = []
            self.roofs = []
            self.floors = []
            
            self.doors = []
            self.windows = []

            self._extract_all_ifc_elements()
            
            print(f"=============IfcExtractor=============\n{self.ifc_file_name}")
            
        except ifcopenshell.errors.FileNotFoundError: # type: ignore
            print(f"Error: File '{model_path}' not found.")

        except Exception as e:
            print(f"An error occurred: {e}")
    
    def _configure_settings(self):
        
        if self.type_geo == 'triangle':
            settings = ifcopenshell.geom.settings()
            settings.USE_PYTHON_OPENCASCADE = True
            # only option to activate both triangulation and the generation of normals
            settings.DISABLE_TRIANGULATION = 1
            settings.set(settings.DISABLE_TRIANGULATION, False)
            settings.WELD_VERTICES = False
            settings.NO_NORMALS = False
            settings.GENERATE_UVS = True
            settings.S = False
            settings.USE_WORLD_COORDS = True
        
        elif self.type_geo == 'brep':
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_PYTHON_OPENCASCADE, True)
            settings.set(settings.DISABLE_TRIANGULATION, True)
            settings.set(settings.USE_BREP_DATA, True)
            settings.USE_WORLD_COORDS = True
            
        return settings
    
    def _extract_all_ifc_elements(self):

        """
        Extracts various IFC entities from the model and initializes visualization settings.
        """

        if self.model:

            self.storeys = self.model.by_type("IfcBuildingStorey")
            self.slabs = self.model.by_type("IfcSlab")
            self.roofs = self.model.by_type("IfcRoof")
            
            self.columns = self.model.by_type("IfcColumn")
            # self.beams = self.model.by_type("IfcBeam")
            
            self.walls = self.model.by_type("IfcWall") + self.model.by_type('IfcWallStandardCase')
            
            self.curtainwalls = self.model.by_type("IfcCurtainWall")            
            self.plates = self.model.by_type("IfcPlate")
            self.members = self.model.by_type("IfcMember")
        
            self.spaces = self.model.by_type("IfcSpace")


    # function notes.
    def _calc_element_orientation(self, element, deg_range=360):

        if element.ObjectPlacement.RelativePlacement.RefDirection is None:
            return 0.0
        
        orientation_vec = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios
        orientation_rad = math.atan2(orientation_vec[1], orientation_vec[0])
        orientation_deg = math.degrees(orientation_rad) % deg_range
        
        return round(orientation_deg, 4)
    
    # function notes.
    def _vertices2dimensions(self, vertices, deci=6):
        verts_array = np.array(vertices)
        x_range, y_range, z_range = np.ptp(verts_array, axis=0)
        length, width = max(x_range, y_range), min(x_range, y_range)
        height = z_range

        return {
            'length': round(length, deci),
            'width': round(width, deci),
            'height': round(height, deci),
        }
    
    # function notes.
    def _shape2_location_verts(self, shape):

        matrix = shape.transformation.matrix.data
        matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
        location = matrix[:,3][0:3]
        grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        return location, grouped_verts
    

#IfcGeneral ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#slab ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    def _floorshape_reasoning(self, shape):
        
        # todos.
        # Figure out what is the output for each line...
        grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        verts_per_face = ifcopenshell.util.shape.get_faces(shape.geometry)
        
        # those edge that occurs only once, is the outlines.
        grouped_edges = ifcopenshell.util.shape.get_edges(shape.geometry)

        locations_per_face = []
        for sublist in verts_per_face:
            new_sublist = [grouped_verts[i].tolist() for i in sublist]
            locations_per_face.append(new_sublist)

        locations_per_face = np.array(locations_per_face, dtype=object)
        all_z_values = np.array([vertex[2] for face in locations_per_face for vertex in face])
        z_min, z_max = np.min(all_z_values), np.max(all_z_values)
        faces_side, faces_upper, faces_lower = [], [], []

        for face in locations_per_face:
            z_values = [vertex[2] for vertex in face]  # Extract z-values of all vertices in the face
            if len(set(z_values)) == 1:  # Check if all z-values are the same
                if z_values[0] == z_max:
                    faces_upper.append(face)
                if z_values[0] == z_min:
                    faces_lower.append(face)
            else:
                faces_side.append(face)
        
        # output values.
        floor_width = z_max - z_min

        print ("tss")

    def get_floor_dimensions(self):

        iterator = ifcopenshell.geom.iterator(
                    self.settings, self.model, multiprocessing.cpu_count(), include=self.floors)

        if self.type_geo == 'triangle':

            if iterator.initialize():

                while True:
                
                    shape = iterator.get()
                    self._floorshape_reasoning(shape)
                    # draw_3d_points(grouped_verts)
                    # dimensions = self._vertices2dimensions(grouped_verts)
                    # location_p1 = location.tolist()

                    if not iterator.next():
                        break

    def extract_all_floors_via_triangulation(self):
        
        def draw_3d_points(points):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            z_coords = [point[2] for point in points]
    
            ax.scatter(x_coords, y_coords, z_coords)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            plt.show()

        if self.version == 'IFC2X3':
            self.floors = self.slabs
        else:
            self.floors = self.slabs + self.roofs

        self.get_floor_dimensions()

        # consider the IFC versions. IfcSlab and IfcRoof.
        # merge / correlate two floors if they're vertically close and locational separated.
        # To determine main storeys. considering that the IfcCurtainWall has vertical shifts.


        # for storey in self.storeys:
        #     print (storey.Elevation)
        
        print('step')

#slab ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#beam ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

        
#beam ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================
      
#===================================================================================================
#wall ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    # function notes.
    def calc_wall_loadbearing(self, wall):
        psets = ifcopenshell.util.element.get_psets(wall)
        return psets.get('Pset_WallCommon', {}).get('LoadBearing')
    
    # function notes.
    def get_wall_dimensions(self):

        iterator = ifcopenshell.geom.iterator(
            self.settings, self.model, multiprocessing.cpu_count(), include=self.walls)
        
        if self.type_geo == 'triangle':

            if iterator.initialize():

                while True:
                
                    shape = iterator.get()
                    wall_location_p1, grouped_verts = self._shape2_location_verts(shape)
                    wall_dimensions = self._vertices2dimensions(grouped_verts)
                    wall_location_p1 = wall_location_p1.tolist()

                    dict_of_a_wall = {
                        "id": shape.guid,
                        "elevation": wall_location_p1[-1],
                        "location":[wall_location_p1],
                        }
                    dict_of_a_wall.update(wall_dimensions)
                    self.info_walls.append(dict_of_a_wall)

                    if not iterator.next():
                        break
    
    # function notes.
    def enrich_wall_information(self):
        id_to_wall = {w.GlobalId: w for w in self.walls}
        self.info_walls = [
            self._update_wall_info(id_to_wall[info['id']], info) \
                for info in self.info_walls if \
                    info['id'] in id_to_wall and \
                        self._calc_element_orientation(id_to_wall[info['id']]) is not None and \
                            info['length'] is not None
        ]
    
    # function notes.
    def _update_wall_info(self, wall, info_w):

        orientation_deg = self._calc_element_orientation(wall)
        wall_location_p1 = info_w['location'][0]
        wall_location_p2 = (
            wall_location_p1[0] + info_w['length'] * math.cos(math.radians(orientation_deg)),
            wall_location_p1[1] + info_w['length'] * math.sin(math.radians(orientation_deg)),
            wall_location_p1[2]
        )

        info_w.update({
            'location': [wall_location_p1, wall_location_p2],
            'orientation': self._calc_element_orientation(wall,deg_range=180), # Here we use degree among 0-180 degree.
            'loadbearing': self.calc_wall_loadbearing(wall),
        })

        return info_w

    @time_decorator
    def extract_all_walls_via_triangulation(self):
        
        self.info_walls = []
        self.info_curtainwalls = []

        # walls
        self.get_wall_dimensions()
        self.enrich_wall_information()

        # curtainwalls.
        self.process_curtainwall_subelements()
        self.get_curtainwall_information()

        self.info_all_walls = self.info_walls + self.info_curtainwalls
   
#wall ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#curtainwall ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    def process_curtainwall_subelements(self):

        self.id2orientation_plates = dict()
        for plate in self.plates:
            self.id2orientation_plates.update({
                plate.GlobalId: self._calc_element_orientation(plate)
            })

        self.id2orientation_members = dict()
        for member in self.members:
            self.id2orientation_members.update({
                member.GlobalId: self._calc_element_orientation(member)
            })

    def get_curtainwall_information(self):
        
        try:
            for cw in self.curtainwalls:
                if hasattr(cw,'IsDecomposedBy') and len(cw.IsDecomposedBy)==1 and cw.IsDecomposedBy[0].is_a('IfcRelAggregates'):

                    cw_related_objects = cw.IsDecomposedBy[0].RelatedObjects
                    cw_related_plates = [ob for ob in cw_related_objects if ob.is_a('IfcPlate')]
                    
                    if not cw_related_plates:
                        raise ValueError(f"no related plates found in the IfcCurtainWall {cw.GlobalId}.")

                    plate_iterator = ifcopenshell.geom.iterator(
                        self.settings, self.model, multiprocessing.cpu_count(), include=cw_related_plates)
                    
                    plate_location_per_cw, plate_orientation_per_cw, plate_width_per_cw = [], [], []

                    if self.type_geo == 'triangle':
                        if plate_iterator.initialize():
                            while True:

                                shape = plate_iterator.get()
                                location, grouped_verts = self._shape2_location_verts(shape)
                                dimensions = self._vertices2dimensions(grouped_verts)
                                
                                orientation_deg = self.id2orientation_plates[shape.guid]

                                location_p1 = location.tolist()
                                location_p2 = [
                                    location_p1[0] + dimensions['length'] * math.cos(math.radians(orientation_deg)),
                                    location_p1[1] + dimensions['length'] * math.sin(math.radians(orientation_deg)),
                                    location_p1[2]]
                                
                                location_p3 = location_p1[:2] + [location_p1[2] + dimensions['height']]
                                location_p4 = location_p2[:2] + [location_p2[2] + dimensions['height']]
                                
                                plate_location_per_cw.extend([location_p1, location_p2, location_p3, location_p4]) # no need for further flatten.
                                plate_orientation_per_cw.append(orientation_deg)
                                plate_width_per_cw.append(dimensions['width'])
                                
                                if not plate_iterator.next():
                                    break

                    cw_corner_points = get_rectangle_corners(plate_location_per_cw)
                    cw_elevation = min(pt[2] for pt in cw_corner_points)
                    cw_location = [pt.tolist() for pt in cw_corner_points if pt[2] == cw_elevation]
                    cw_length = distance_between_points(cw_location[0], cw_location[1])
                    cw_width = find_most_common_value(plate_width_per_cw)[0]
                    cw_orientation = find_most_common_value(plate_orientation_per_cw)[0]
                    
                    self.info_curtainwalls.append({
                        'id': cw.GlobalId,
                        'elevation': cw_elevation,
                        'location': cw_location,
                        'length': cw_length,
                        'width': round(cw_width, 4),
                        'orientation': cw_orientation % 180.0,
                        'loadbearing': False,
                        })
                
                else:
                    # Logging the missing or malformed attribute rather than raising an exception to allow processing to continue
                    print(f'Attribute check failed for IfcCurtainWall with guid {cw.GlobalId}.')
        
        except Exception as e:
            # General exception handling to catch unexpected errors
            print(f"An error occurred: {str(e)}")

#curtainwall ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#column ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    # function notes.
    def get_column_dimensions(self):
        
        iterator = ifcopenshell.geom.iterator(
            self.settings, self.model, multiprocessing.cpu_count(), include=self.columns)
        
        if self.type_geo == 'triangle':

            if iterator.initialize():

                while True:

                    shape = iterator.get()
                    column_location_p1, grouped_verts = self._shape2_location_verts(shape)
                    column_dimensions = self._vertices2dimensions(grouped_verts)
                    column_location_p1 = column_location_p1.tolist()

                    dict_of_a_column = {
                        "id": shape.guid,
                        "location":[column_location_p1],
                        }
                    dict_of_a_column.update(column_dimensions)
                    self.info_columns.append(dict_of_a_column)

                    if not iterator.next():
                        break
    
    # function notes.
    def enrich_column_information(self):
        id_to_column = {c.GlobalId: c for c in self.columns}
        self.info_columns = [
            self._update_column_info(id_to_column[info['id']], info) \
                for info in self.info_columns if \
                    info['id'] in id_to_column and \
                            info['height'] is not None
        ]
    
    # function notes.
    def _get_location_column(self, column):
        c_location_pt_from_body = None
        c_location_pt_directplacement = None
        
        for r in column.Representation.Representations:
            if r.RepresentationIdentifier == 'Body':
                mapped_r = r.Items[0].MappingSource.MappedRepresentation
                c_location_pt_from_body = mapped_r.Items[0].Position.Location.Coordinates
                break  # Assuming we only need the first matching 'Body'
        
        if abs(c_location_pt_from_body[0])<0.001 and abs(c_location_pt_from_body[1])<0.001:
            # risk: what will happen if it's really 0,0,x.
            c_location_pt_directplacement = column.ObjectPlacement.RelativePlacement.Location.Coordinates
            rel_storey_elevation = column.ObjectPlacement.PlacementRelTo.PlacesObject[0].Elevation
            rel_placement = column.ObjectPlacement.PlacementRelTo.RelativePlacement.Location.Coordinates
            c_location_pt_directplacement = tuple(sum(x) for x in zip(c_location_pt_directplacement, rel_placement))
            c_location_pt_directplacement = c_location_pt_directplacement[:2] + (c_location_pt_directplacement[2] + rel_storey_elevation,)
            return c_location_pt_directplacement
        else:
            return c_location_pt_from_body

    # function notes.
    def _update_column_info(self, column, info_c):

        column_location_p1 = self._get_location_column(column)
        column_location_p2 = column_location_p1[:2] + (column_location_p1[2] + info_c['height'],)

        info_c.update({
            "elevation": column_location_p1[-1],
            'location': [column_location_p1, column_location_p2],
        })

        return info_c
    
    @time_decorator
    def extract_all_columns_via_triangulation(self):
        
        self.info_columns = []
        self.get_column_dimensions()
        self.enrich_column_information()

#column ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#displayandexport ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    # function notes.
    def wall_width_histogram(self):
        
        values = [w['width'] for w in self.info_all_walls if 'width' in w]
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
        
    # function notes.
    def wall_length_histogram(self):

        values = [w['length'] for w in self.info_all_walls if 'length' in w]

        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes((0.0875, 0.1, 0.875, 0.875))
        ax.hist(values, weights=np.ones(len(values)) / len(values), bins=20, color='#bcbd22', edgecolor='black')
        ax.set_xlabel('Length of IfcWalls', color='black', fontsize=12)
        ax.set_ylabel("Percentage Frequency Distribution", color="black", fontsize=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(xmin=0.0, xmax=max(values))

        plt.savefig(os.path.join(self.out_fig_path, 'wall_length_histogram.png'), dpi=200)
        plt.close(fig)

    # function notes.
    def wall_location_map(self):

        values = [w['location'] for w in self.info_all_walls if 'location' in w]

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

    # function notes.
    def wall_orientation_histogram(self):
        values = [w['orientation'] for w in self.info_all_walls if 'orientation' in w]

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

    @time_decorator
    def wall_display(self):

        self.wall_width_histogram()
        self.wall_length_histogram()
        self.wall_location_map()
        self.wall_orientation_histogram()
    
    # function notes.
    def write_dict_columns(self):

        dict_info_columns = remove_duplicate_dicts(self.info_columns)
        try:
            with open(os.path.join(self.out_fig_path, 'info_columns.json'), 'w') as json_file:
                json.dump(dict_info_columns, json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'columns.json'}: {e}")

    # function notes.
    def write_dict_walls(self):
        
        dict_info_walls = remove_duplicate_dicts(self.info_all_walls)
        try:
            with open(os.path.join(self.out_fig_path, 'info_walls.json'), 'w') as json_file:
                json.dump(dict_info_walls, json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'walls.json'}: {e}")
        
#displayandexport ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
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

