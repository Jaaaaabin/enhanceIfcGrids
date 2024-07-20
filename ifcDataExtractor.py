import multiprocessing
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

import os
import math
import json
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from collections import defaultdict
from shapely.geometry import Polygon
from sklearn.decomposition import PCA

from toolsQuickUtils import time_decorator
from toolsQuickUtils import remove_duplicate_dicts, find_most_common_value
from toolsQuickUtils import get_rectangle_corners, distance_between_points, find_closed_loops
from toolsSpatialGlue import spatial_process_lines

#===================================================================================================
#IfcGeneral ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
class IfcDataExtractor:

    def __init__(self, model_path, figure_path, read='triangle'):

        try:

            self.type_geo = read
            self.settings = self._configure_settings()

            self.model = ifcopenshell.open(model_path)

            self.ifc_file_name = os.path.basename(model_path)
            self.authoring_tool = 'non-ar' if '-AR-' not in self.ifc_file_name else 'ar'

            self.out_fig_path = figure_path
            os.makedirs(figure_path, exist_ok=True)

            self.version = self.model.schema
            self.storeys = []

            self.st_columns = []

            self.curtainwalls = []
            self.plates = []
            self.members = []

            self.slabs = []
            self.roofs = []
            self.floors = []
            
            self.doors = []
            self.windows = []

            self._extract_all_ifc_elements()
            
            # hard_coded_part.
            self._read_view_settings()
            self._remove_slab_outliers()
            # hard_coded_part.

            print(f"=====================IfcExtractor=====================\n{self.ifc_file_name}\n=====================IfcExtractor=====================")
            
        except ifcopenshell.errors.FileNotFoundError: # type: ignore
            print(f"Error: File '{model_path}' not found.")

        except Exception as e:
            print(f"An error occurred: {e}")

    def _get_file_prefix_code(self, filename):
        parts = filename.split('-')
        return '-'.join(parts[:2])
    
    def _remove_slab_outliers(self):

        slab_outliers ={
            "11103190":[
                '1lPJHttHj0t93Rx9JW$x8L',
            ],
            "11103085":[
                '0lLezB9Eo$GvWuGLIrVqQT',
                '2dgQT7c8heJAthKIf57922',
                '2$TzztUz_nI9DzZezaA0Uf',
                '19S$35tbDXGBvHnQP5VyE9',
                '1cM2g9vEKfIuKkoOh4wzLq',
                '3B4yd04k$TGAbpDhg7t6cm',
                '216BSagR2mHRRWNp1ebRcl',
                '0Qo2NFvaCRJOHrKtNiM2UJ',
                '3j$3sa8m0eIAMLU9FdP81l',
                '2qKTvaLYmZHOA8KyeHuD1p'
            ],
            "11103186":[
                '1gQ_y3GLj6yQZm$NpWfoHW'
            ],
            "11103035":[
                '3Nvn$pqh7mHx_H$clYhRTH',
                '3_uFSRcG9_IvnATDmaTf65',
            ],
            "11103223":[
                '3gU8YbK9XBrRI57teoH1G_',
                '0mSabJq$mkQti5Wbrc58uO',
                '2POJPR4dfAxRjFwwgsyMmv',
                '3$utHrBT4ZQgz5TTy1Dw69',
            ],
            "11103438":[
                '2EoIrT87f9BffvlijK50Zv',
                '2EoIrT87f9BffvlijK50dX',
                '2EoIrT87f9BffvlijK50i5',
                '2EoIrT87f9BffvlijK50kH',
                '2EoIrT87f9BffvlijK50Xj',
                '2EoIrT87f9BffvlijK50bg',
                '2EoIrT87f9BffvlijK50hC',
                '2EoIrT87f9BffvlijK53CR',
                '2EoIrT87f9BffvlijK53CJ',
                '1pR$bNwgP9swl7km4GAk2q',
                '1pR$bNwgP9swl7km4GAk1O',
            ]
        }
        
        # filter out non-relevant slabs with specific guids.
        for key, values in slab_outliers.items():
            if key in self.ifc_file_name:
                self.slabs = [sl for sl in self.slabs if sl.GlobalId not in values]
                break
            else:
                continue
        
        slab_type_outliers = {
            "11103186": "Footing",
            "11103438": "Landing",}
        
        # filter out non-relevant slabs with specific slab types "string".
        for key, type_name_string in slab_type_outliers.items():
            if key in self.ifc_file_name:
                self.slabs = [sl for sl in self.slabs if type_name_string not in sl.Name]
                break
            else:
                continue
    
    def _remove_wall_outliers(self, min_height=1.0):

        self.info_walls = [w for w in self.info_walls if w['height'] > min_height]

    def _read_view_settings(self):

        view_settings = {
            "11103093": {
                "type": 1,
                "elev": 25,
                "azim": -60
            },
            "11103024": {
                "type": 2,
                "elev": 30,
                "azim": -15
            },
            "11103190": {
                "type": 3,
                "elev": 35,
                "azim": -30
            },
            "11103085": {
                "type": 4,
                "elev": 25,
                "azim": -40
            },
            "11103186": {
                "type": 5,
                "elev": 25,
                "azim": -110
            },
            "11103035": {
                "type": 6,
                "elev": 20,
                "azim": -75
            },
            "11103223": {
                "type": 7,
                "elev": 25,
                "azim": 170
            },
            "11103438": {
                "type": 8,
                "elev": 25,
                "azim": 20
            }
        }
        self.elev, self.azim = 25, 50
        for key, values in view_settings.items():
            if key in self.ifc_file_name:
                self.elev = values["elev"]
                self.azim = values["azim"]
                break
            else:
                continue

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
            
            # self.beams = self.model.by_type("IfcBeam")
            self.st_columns = self.model.by_type("IfcColumn")
            self.walls = self.model.by_type("IfcWall") + self.model.by_type('IfcWallStandardCase')

            self.curtainwalls = self.model.by_type("IfcCurtainWall")
            self.plates = self.model.by_type("IfcPlate")
            self.members = self.model.by_type("IfcMember")

    # function notes.
    def _calc_element_orientation(self, element, deg_range=360):

        if element.ObjectPlacement.RelativePlacement.RefDirection is None:
            return 0.0
        
        orientation_vec = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios
        orientation_rad = math.atan2(orientation_vec[1], orientation_vec[0])
        orientation_deg = math.degrees(orientation_rad) % deg_range
        
        return round(orientation_deg, 4)
    
    def _calc_element_orientation_from_reference_points(self, endpoint, midpoint, deg_range=360):

        # Convert points to numpy arrays
        p1 = np.array(endpoint)
        p2 = np.array(midpoint)
        
        # Calculate the direction vector from point1 to point2
        direction_vector = p2 - p1
        angle_deg = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
        if angle_deg < 0:
            angle_deg += deg_range
        
        return angle_deg

    # function notes.
    def _vertices2dimensions(self, vertices, deci=4):
        verts_array = np.array(vertices)
        x_range, y_range, z_range = np.ptp(verts_array, axis=0)
        length, width = max(x_range, y_range), min(x_range, y_range)
        height = z_range

        return {
            'length': round(length, deci),
            'width': round(width, deci),
            'height': round(height, deci),
        }
    
    def _vertices2dimensions_pca_advanced(self, vertices, deci=4):

        # Perform PCA to find the main directions
        pca = PCA(n_components=3)
        pca.fit(vertices)
        
        # Transform the points to the new coordinate system
        transformed_points = pca.transform(vertices)
        
        # Calculate the differences along each principal component
        max_transformed = np.max(transformed_points, axis=0)
        min_transformed = np.min(transformed_points, axis=0)
        
        # The dimensions along the principal components
        dimensions = max_transformed - min_transformed
        
        # Sort the dimensions from max to min
        sorted_dimensions = sorted(dimensions, reverse=True)
        
        # Calculate the height by finding the dimension close to the max distance in z axis
        max_z = np.max(vertices[:, 2])
        min_z = np.min(vertices[:, 2])
        max_distance_z = max_z - min_z
        
        # Determine the height as the dimension closest to max_distance_z
        height = min(sorted_dimensions, key=lambda x: abs(x - max_distance_z))
        
        # Remove the height from the sorted dimensions list
        sorted_dimensions.remove(height)
        
        # Assign the remaining dimensions to length and width
        length, width = sorted(sorted_dimensions, reverse=True)
        
        # Identify the principal component corresponding to the length
        length_index = np.argmax(dimensions == length)
        length_vector = pca.components_[length_index]
        
        # Calculate the direction of the length in the xy-plane
        angle_rad = np.arctan2(length_vector[1], length_vector[0]) # in the range [-pi, pi]
        angle_deg = np.degrees(angle_rad)

        return {
            'length': round(length, deci),
            'width': round(width, deci),
            'height': round(height, deci),
            'orientation': round(angle_deg, 4),
        }

    # function notes.
    def _shape2_location_verts(self, shape):

        matrix = shape.transformation.matrix.data
        matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
        location = matrix[:,3][0:3]
        grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        return location, grouped_verts
    
    # function notes.
    def _divide_unit(self, loc):

        for value in loc:
            if abs(value) > 1000:
                loc = tuple([value / 1000 for value in loc])
                break
            else:
                continue
        return loc
#IfcGeneral ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#slab ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    def _floorshape_reasoning(self, shape):
        
        # shape_grouped_verts = ifcopenshell.util.shape.get_shape_vertices(shape, shape.geometry)
        # also check https://blenderbim.org/docs-python/ifcopenshell-python/geometry_processing.html

        # A nested numpy array e.g. [[v1x, v1y, v1z], [v2x, v2y, v2z], ...]
        grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)

        # A nested numpy array e.g. [[f1v1, f1v2, f1v3], [f2v1, f2v2, f2v3], ...]
        grouped_faces = ifcopenshell.util.shape.get_faces(shape.geometry)
        
        # # A nested numpy array e.g. [[e1v1, e1v2], [e2v1, e2v2], ...]
        # grouped_edges = ifcopenshell.util.shape.get_edges(shape.geometry)

        locations_per_face = []
        for sublist in grouped_faces:
            new_sublist = [grouped_verts[i].tolist() for i in sublist]
            locations_per_face.append(new_sublist)
        locations_per_face = np.array(locations_per_face, dtype=object)
        all_z_values = np.array([vertex[2] for face in locations_per_face for vertex in face])
        z_min, z_max = np.min(all_z_values), np.max(all_z_values)
        
        floor_width = z_max - z_min
        if floor_width > 0.5:
            return None, None
        
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
        
        # slab outline in the xy plane. can be used when merging the slab with raised areas.
        floor_location_xy = []

        for f_side in faces_side:
            xy_locations_per_face_s =[]
            sliced_f = f_side[:, :2]
            
            keep_mask = np.ones(sliced_f.shape[0], dtype=bool)
            for i in range(sliced_f.shape[0]):
                 for j in range(i + 1, sliced_f.shape[0]):
                    if np.all(sliced_f[i] == sliced_f[j]):
                        keep_mask[j] = False  # Mark row j for removal
            xy_locations_per_face_s = sliced_f[keep_mask]
            floor_location_xy.append(xy_locations_per_face_s.tolist())

        # tempo intervention.
        # dirty but works for now....
        floor_location_xy = [sublist for sublist in floor_location_xy if len(sublist) <= 2]

        return floor_width, floor_location_xy
    
    def get_floor_dimensions_non_ar(self):

        def add_values_to_points(points, add_values):
            updated_points = []
            for outer_point in points:
                updated_outer = []
                for point in outer_point:
                    new_point = [
                        point[0] + add_values[0],  # Add the first dimension
                        point[1] + add_values[1],  # Add the second dimension
                        add_values[2]              # Use the third value directly
                    ]
                    updated_outer.append(new_point)
                updated_points.append(updated_outer)
            return updated_points
        
        iterator = ifcopenshell.geom.iterator(
                    self.settings, self.model, multiprocessing.cpu_count(), include=self.floors)

        if self.type_geo == 'triangle':
            if iterator.initialize():
                while True:
                
                    shape = iterator.get()
                    floor_width, floor_location_xy = self._floorshape_reasoning(shape)

                    tempo_matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
                    tempo_location = tempo_matrix[:,3][0:3]
                    floor_location_xy = add_values_to_points(floor_location_xy, tempo_location)

                    dict_of_a_floor = {
                        "id": shape.guid,
                        "location": floor_location_xy,
                        "width": floor_width,
                        "elevation": tempo_location[-1]
                        }
                    
                    self.info_floors.append(dict_of_a_floor)
                    if not iterator.next():
                        break

    def get_floor_dimensions_ar(self):

        iterator = ifcopenshell.geom.iterator(
                    self.settings, self.model, multiprocessing.cpu_count(), include=self.floors)

        if self.type_geo == 'triangle':
            if iterator.initialize():
                while True:
                
                    shape = iterator.get()
                    floor_width, floor_location_xy = self._floorshape_reasoning(shape) # erros happen around the last iterations.. maybe a problem with floor.
                    
                    dict_of_a_floor = {
                        "id": shape.guid,
                        "location": floor_location_xy,
                        "width": floor_width,
                        }
                    
                    self.info_floors.append(dict_of_a_floor)
                    if not iterator.next():
                        break
    
    def _update_floor_info(self, floor, info_f):
        
        # use this IFC query for temporary solution.
        floor_elevation = floor.ObjectPlacement.PlacementRelTo.RelativePlacement.Location.Coordinates[-1]
        if abs(floor_elevation) > 100:
            floor_elevation = floor_elevation / 1000
    
        for lcs in info_f['location']:
            for lc in lcs:
                lc.append(floor_elevation)

        info_f.update({
            'elevation': round(floor_elevation,4)
        })
        return info_f

    def enrich_floor_information(self):

        id_to_floor = {f.GlobalId: f for f in self.floors}
        self.info_floors = [
            self._update_floor_info(id_to_floor[info['id']], info) \
                for info in self.info_floors if \
                    info['id'] in id_to_floor and info['width'] is not None \
                    and info['location'] is not None]
    
    @time_decorator
    def extract_all_floors(self):

        self.info_floors = []
        self.floors = self.slabs

        # this part needs to be improved systematically before any publsihment.....
        if self.authoring_tool != 'ar':
            # case of non-Autodesk Revit as the initial authoring tools.
            self.get_floor_dimensions_non_ar()
        else:
            # case of Autodesk Revit as the initial authoring tools.
            self.get_floor_dimensions_ar()
            self.enrich_floor_information()
    
        # save data and display
        self.write_dict_floors()

    def _refine_slab_shapes_per_storey(self, thin_slab_limit=0.05):
        
        self.info_floor_slabs = []
        
        floor_slabs_data = self.info_floors
        storey_values = list(set(item['elevation'] for item in floor_slabs_data))
        slab_z_groups = defaultdict(list)
        segment_z_groups = defaultdict(list)
        
        # Iterate over each slab definition in the data
        for st_z in storey_values:
            slab_per_st_z = []
            for slab in floor_slabs_data:
                if slab['elevation'] == st_z and slab['width']>thin_slab_limit:
                    slab_per_st_z.append(slab)
                else:
                    continue
            slab_z_groups[st_z] = slab_per_st_z

        for st_z, values in slab_z_groups.items():
            
            if len(values)>1:
                segment_z_groups[st_z] = [value['location'] for value in values]
                segment_z_groups[st_z] = [item for sublist in segment_z_groups[st_z] for item in sublist]

            elif len(values)==1:
                segment_z_groups[st_z] = values[0]['location']

            else:
                return None
        
        # Create a new 3D figure
        max_area = 0
        floor_slab_closed_loops = []
        
        # Calculate areas of loops and find the maximum area
        for st_z, segments in segment_z_groups.items():
            loops = find_closed_loops(segments)
            for loop in loops:
                polygon = Polygon(loop)
                area = polygon.area
                floor_slab_closed_loops.append((st_z, loop, area))
                if area > max_area:
                    max_area = area
        
        # Sort loops by area
        floor_slab_closed_loops.sort(key=lambda x: x[2], reverse=True)
        self.info_floor_slabs = [(st_z, loop, area / max_area) for st_z, loop, area in floor_slab_closed_loops]

    def _update_floor_with_refined_slabs(self, splitter_percent=0.8):

        def generate_point_pairs(points):
            point_pairs = []
            num_points = len(points)
            if num_points < 2:
                return point_pairs
            for i in range(num_points - 1):
                point_pairs.append((points[i], points[i+1]))
            
            # Note: no need to get the -1 to 0 since it's a closed loop, and the first point is the same as the last point.
            # point_pairs.append((points[-1], points[0]))
            
            return point_pairs
        
        tempo_dict = defaultdict(dict)
        unique_slab_width = max(fl['width'] for fl in self.info_floors)

        for (v, segments, ratio) in self.info_floor_slabs:

            if ratio > splitter_percent:
                tempo_dict[v] = {
                    'points': generate_point_pairs(segments),
                    'area_ratio': ratio,
                    'width':unique_slab_width,
                }

        self.info_unique_floor_slabs = tempo_dict
        
    def post_processing_floors_to_slabs(self):
        
        self._refine_slab_shapes_per_storey()
        self._update_floor_with_refined_slabs()
        self.write_dict_floor_slabs()
        self.refined_floor_slabs_dispaly()
        
    def refined_floor_slabs_dispaly(self):

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Normalize loop areas to get the color values
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('coolwarm')

        # Plot loops with colors based on their area percentage
        for st_z, loop, area in self.info_floor_slabs:
            color = cmap(norm(area))
            loop.append(loop[0])  # To close the loop
            loop_array = np.array(loop)
            ax.plot(loop_array[:, 0], loop_array[:, 1], loop_array[:, 2], color=color, alpha=0.5)
        
        # Create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Closed area of slab geometries (as % of the maximum closed area)', fontsize=14)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0%', '50%', '100%'])
        
        # Set labels and title
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.axis('off')

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.out_fig_path, self._get_file_prefix_code(self.ifc_file_name)+ '_refined_floor_slabs.png'), dpi=200)
        plt.close()

#slab ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#beam ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
# not investigated yet.
        
#beam ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================
      
#===================================================================================================
#wall ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    # function notes.
    def calc_wall_is_int_ext(self, wall):
        psets = ifcopenshell.util.element.get_psets(wall)
        return psets.get('Pset_WallCommon', {}).get('IsExternal')
    
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
                        "elevation": round(wall_location_p1[-1],4),
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
        })

        return info_w
    
    def _split_st_ns_ct_wall_information(self):
        
        self.id_st_walls  = [w.GlobalId for w in self.walls if self.calc_wall_loadbearing(w)]
        self.id_ns_walls  = [w.GlobalId for w in self.walls if not self.calc_wall_loadbearing(w)]
        self.id_ct_walls  = [w.GlobalId for w in self.curtainwalls] if self.curtainwalls else []

        for info_w in self.info_walls:
            value_to_check = info_w.get('id', None)
            if value_to_check in self.id_st_walls:
                self.info_st_walls.append(info_w)
            elif value_to_check in self.id_ns_walls:
                self.info_ns_walls.append(info_w)

    def _glue_wall_connections(self):
        
        all_wall_data = self.info_st_walls + self.info_ns_walls + self.info_curtainwalls
        glued_wall_data = spatial_process_lines(
            all_wall_data,
            plot_adjustments=True,
            output_figure_folder=self.out_fig_path)
        
        new_info_st_walls = []
        new_info_ns_walls = []
        new_info_curtainwalls = []
        
        for info_w in glued_wall_data:

            wall_id = info_w.get('id', None)
            if wall_id in self.id_st_walls:
                new_info_st_walls.append(info_w)
            elif wall_id in self.id_ns_walls:
                new_info_ns_walls.append(info_w)
            elif wall_id in self.id_ct_walls:
                new_info_curtainwalls.append(info_w)
            else:
                continue

        self.info_st_walls = new_info_st_walls
        self.info_ns_walls = new_info_ns_walls
        self.info_curtainwalls = new_info_curtainwalls

    def _get_main_directions_from_walls(self, num_directions=4):
        
        info_all_walls = self.info_ns_walls + self.info_st_walls + self.info_curtainwalls

        wall_orientations = [w['orientation'] for w in info_all_walls if 'orientation' in w]
        wall_orientations = [(v-180) if v>=180 else v for v in wall_orientations] # to 0 - 180 degree.
        main_directions = Counter(wall_orientations)
        main_directions = main_directions.most_common(num_directions)
        self.main_directions = [main_direct[0] for main_direct in main_directions]

        wall_x_values = [w['location'][0][0] for w in info_all_walls if 'location' in w]
        wall_y_values = [w['location'][0][1] for w in info_all_walls if 'location' in w]

        self.corner_x_value = (max(wall_x_values) + min(wall_x_values)) * 0.5
        self.corner_y_value = (max(wall_y_values) + min(wall_y_values)) * 0.5
        self.corner_z_value = max([w['location'][0][-1] for w in info_all_walls if 'location' in w]) * 1.1
    
    def _get_all_wall_int_ext(self):
        
        # for IfcWalls
        # in the scope of internal / external, we only consider the walls and curtain walls.
        for info in self.info_walls:
            element = self.model.by_guid(info['id'])
            if self.calc_wall_is_int_ext(element):
                info.update({'external': 1})
            else:
                info.update({'external': 0})

        # for IfcCurtainWalls
        # we assume that all IfcCurtainWall will be placed as external building elements.
        for info in self.info_curtainwalls:
            info.update({'external': 1})

    @time_decorator
    def extract_all_walls(self):
        
        # first write to info_walls and then split by st_walls or ns_walls.
        self.info_walls = []
        self.info_st_walls = []
        self.info_ns_walls = []

        self.get_wall_dimensions()
        self.enrich_wall_information()

    @time_decorator
    def extract_all_curtainwalls(self):

        self.info_curtainwalls = []
        self.process_curtainwall_subelements()
        self.get_curtainwall_information()

    def post_processing_walls(self):

        self._get_all_wall_int_ext()
        self._remove_wall_outliers()
        self._split_st_ns_ct_wall_information()
        self._glue_wall_connections()
        self._get_main_directions_from_walls()
        self.write_dict_walls()
    
#wall ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#curtainwall ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    # curtain wall with sub elements. ----------------------------------------------------------------
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
    # curtain wall with sub elements. ----------------------------------------------------------------

    def _process_ifc_members(self, cw_related_members):
        member_z_location_per_cw = []
        if self.type_geo == 'triangle' and cw_related_members:
            member_iterator = ifcopenshell.geom.iterator(self.settings, self.model, multiprocessing.cpu_count(), include=cw_related_members)
            if member_iterator.initialize():
                while True:
                    shape = member_iterator.get()
                    location, _ = self._shape2_location_verts(shape)
                    member_z_location_per_cw.append(list(location)[-1])
                    if not member_iterator.next():
                        break
        return min(member_z_location_per_cw) if member_z_location_per_cw else None

    def _process_ifc_plates(self, cw_related_plates):

        plate_location_per_cw, plate_orientation_per_cw, plate_width_per_cw, plate_z_ranges_per_cw= [], [], [], []
        if self.type_geo == 'triangle' and cw_related_plates:
            plate_iterator = ifcopenshell.geom.iterator(self.settings, self.model, multiprocessing.cpu_count(), include=cw_related_plates)
            if plate_iterator.initialize():
                while True:
                    shape = plate_iterator.get()
                    location, grouped_verts = self._shape2_location_verts(shape)
                    dimensions = self._vertices2dimensions(grouped_verts)
                    orientation_deg = self.id2orientation_plates[shape.guid]
                    plate_locations = self._calculate_plate_locations(location, dimensions, orientation_deg)
                    plate_location_per_cw.extend(plate_locations)
                    plate_orientation_per_cw.append(orientation_deg)
                    plate_width_per_cw.append(dimensions['width'])
                    plate_z_values = [sublist[2] for sublist in plate_locations]
                    plate_z_ranges = [min(plate_z_values), max(plate_z_values)]
                    plate_z_ranges_per_cw.extend(plate_z_ranges)
                    if not plate_iterator.next():
                        break
        return plate_location_per_cw, plate_orientation_per_cw, plate_width_per_cw, plate_z_ranges_per_cw

    def _calculate_plate_locations(self, location, dimensions, orientation_deg):

        location_p_center = location.tolist()
        location_p1 = [
            location_p_center[0] - 0.5 * dimensions['length'] * math.cos(math.radians(orientation_deg)),
            location_p_center[1] - 0.5 * dimensions['length'] * math.sin(math.radians(orientation_deg)),
            location_p_center[2]
        ]
        location_p2 = [
            location_p_center[0] + 0.5 * dimensions['length'] * math.cos(math.radians(orientation_deg)),
            location_p_center[1] + 0.5 * dimensions['length'] * math.sin(math.radians(orientation_deg)),
            location_p_center[2]
        ]
        location_p3 = location_p1[:2] + [location_p1[2] + dimensions['height']]
        location_p4 = location_p2[:2] + [location_p2[2] + dimensions['height']]
        return [location_p1, location_p2, location_p3, location_p4]

    def _get_closest_points_to_elevation(self, corner_points, elevation):
        corner_points = np.array(corner_points)
        differences = np.abs(corner_points[:, 2] - elevation)
        sorted_indices = np.argsort(differences)
        closest_indices = sorted_indices[:2]
        return [corner_points[idx].tolist() for idx in closest_indices]

    # curtain wall as assembly (representation) ----------------------------------------------------------------
    def _adjust_curtainwall_z_location(self, wall_location_p1, wall_dimensions, wall_location_centroid):
        
        half_h = wall_dimensions['height'] * 0.5
        if abs(wall_location_p1[-1]-wall_location_centroid[-1]) > 0.5: # tempo hard-coded threshold value.
            wall_location_p1[-1] = round(wall_location_centroid[-1] - half_h, 6)
        
        return wall_location_p1
        
    def _get_curtainwall_dimensions(self, curtainwall_element):

        shape = ifcopenshell.geom.create_shape(self.settings, curtainwall_element)
        wall_location_p1, grouped_verts = self._shape2_location_verts(shape)

        wall_location_p1 = wall_location_p1.tolist()
        location_centroid = ifcopenshell.util.shape.get_shape_bbox_centroid(shape, shape.geometry)
        wall_dimensions = self._vertices2dimensions_pca_advanced(grouped_verts)

        wall_location_p1 = self._adjust_curtainwall_z_location(wall_location_p1, wall_dimensions, location_centroid)
        wall_orientation = self._calc_element_orientation_from_reference_points(wall_location_p1, location_centroid)
        orientation_in180 = wall_orientation-180 if wall_orientation>=180 else wall_orientation

        wall_location_p2 = (
            wall_location_p1[0] + wall_dimensions['length'] * math.cos(math.radians(wall_orientation)),
            wall_location_p1[1] + wall_dimensions['length'] * math.sin(math.radians(wall_orientation)),
            wall_location_p1[2]
        ) 
    
        dict_of_a_curtainwall = {
            "id": shape.guid,
            "elevation": round(wall_location_p1[-1],4),
            "location": [wall_location_p1, wall_location_p2],
            "orientation": orientation_in180,
            }
        
        dict_of_a_curtainwall.update(wall_dimensions)
        self.info_curtainwalls.append(dict_of_a_curtainwall)
    
    # curtain wall as assembly (representation) ----------------------------------------------------------------
    
    # final processing functions ----------------------------------------------------------------
    
    def _process_curtainwall_with_sub_elements(self, cw, deci=4):

        cw_related_objects = cw.IsDecomposedBy[0].RelatedObjects
        cw_related_members = [ob for ob in cw_related_objects if ob.is_a('IfcMember')]
        cw_related_plates = [ob for ob in cw_related_objects if ob.is_a('IfcPlate')]

        min_z_per_cw = self._process_ifc_members(cw_related_members)
        plate_location_per_cw, plate_orientation_per_cw, plate_width_per_cw, plate_z_ranges_per_cw= self._process_ifc_plates(cw_related_plates)

        cw_corner_points = get_rectangle_corners(plate_location_per_cw)
        cw_elevation = min(pt[2] for pt in cw_corner_points)
        cw_location = [pt.tolist() for pt in cw_corner_points if pt[2] == cw_elevation]
        cw_height = max(plate_z_ranges_per_cw) - min(plate_z_ranges_per_cw)
        
        if len(cw_location) < 2:
            cw_location = self._get_closest_points_to_elevation(cw_corner_points, cw_elevation)

        cw_length = distance_between_points(cw_location[0], cw_location[1])
        cw_width = find_most_common_value(plate_width_per_cw)[0]
        cw_orientation = find_most_common_value(plate_orientation_per_cw)[0]

        if min_z_per_cw is not None and min_z_per_cw < cw_elevation:
            cw_elevation = min_z_per_cw
            for sub_loc in cw_location:
                sub_loc[-1] = cw_elevation

        self.info_curtainwalls.append({
            'id': cw.GlobalId,
            'elevation': round(cw_elevation, deci),
            'location': cw_location,
            'height': round(cw_height, deci),
            'length': cw_length,
            'width': round(cw_width, deci),
            'orientation': cw_orientation % 180.0,
        })
        # todo. get the height of the IfcCurtainWall with "sub elements."

    def _process_curtainwall_with_representation(self, cw):
        if hasattr(cw.Representation, 'Representations'):
            for r in cw.Representation.Representations:
                if r.RepresentationIdentifier == 'Body':
                    self._get_curtainwall_dimensions(cw)
                    break
    
    # final processing functions ----------------------------------------------------------------

    def get_curtainwall_information(self):
        try:
            for cw in self.curtainwalls:
                if hasattr(cw, 'IsDecomposedBy') and len(cw.IsDecomposedBy) == 1 and cw.IsDecomposedBy[0].is_a('IfcRelAggregates'):
                    self._process_curtainwall_with_sub_elements(cw)
                elif hasattr(cw, 'Representation'):
                    self._process_curtainwall_with_representation(cw)
                else:
                    print(f'Attribute check failed for IfcCurtainWall with guid {cw.GlobalId}.')
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    # old function----------------------------------------------------------------
    # def get_curtainwall_information(self):
        # try:
            
        #     for cw in self.curtainwalls:
                
        #         
        #         if hasattr(cw,'IsDecomposedBy') and len(cw.IsDecomposedBy)==1 and cw.IsDecomposedBy[0].is_a('IfcRelAggregates'):

        #             cw_related_objects = cw.IsDecomposedBy[0].RelatedObjects
        #             cw_related_members = [ob for ob in cw_related_objects if ob.is_a('IfcMember')]
        #             cw_related_plates = [ob for ob in cw_related_objects if ob.is_a('IfcPlate')]
                    
        #             # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        #             # IfcMembers.
        #             ### here might exist one issue. case happend that there's actually only one IfcMember rendering bizzare min_z_per_cw. 
        #             member_z_location_per_cw = []
        #             min_z_per_cw = None

        #             if self.type_geo == 'triangle':

        #                 if cw_related_members:

        #                     member_iterator = ifcopenshell.geom.iterator(self.settings, self.model, multiprocessing.cpu_count(), include=cw_related_members)
                            
        #                     if member_iterator.initialize():
        #                         while True:

        #                             shape = member_iterator.get()
        #                             location, grouped_verts = self._shape2_location_verts(shape)
        #                             # dimensions = self._vertices2dimensions(grouped_verts)
        #                             member_z_location_per_cw.append(list(location)[-1])
                                    
        #                             if not member_iterator.next():
        #                                 break
        #                     if member_z_location_per_cw:
        #                         min_z_per_cw = min(member_z_location_per_cw)

        #             # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        #             # IfcPlates.
        #             plate_location_per_cw, plate_orientation_per_cw, plate_width_per_cw = [], [], []

        #             if self.type_geo == 'triangle':

        #                 if cw_related_plates:
                            
        #                     plate_iterator = ifcopenshell.geom.iterator(self.settings, self.model, multiprocessing.cpu_count(), include=cw_related_plates)
                            
        #                     if plate_iterator.initialize():
        #                         while True:

        #                             shape = plate_iterator.get()
        #                             location, grouped_verts = self._shape2_location_verts(shape)
        #                             dimensions = self._vertices2dimensions(grouped_verts)
                                    
        #                             orientation_deg = self.id2orientation_plates[shape.guid]

        #                             location_p_center = location.tolist() # center part of the IfcPlate.
        #                             location_p1 = [
        #                                 location_p_center[0] - 0.5 * dimensions['length'] * math.cos(math.radians(orientation_deg)),
        #                                 location_p_center[1] - 0.5 * dimensions['length'] * math.sin(math.radians(orientation_deg)),
        #                                 location_p_center[2]]
                                    
        #                             location_p2 = [
        #                                 location_p_center[0] + 0.5 * dimensions['length'] * math.cos(math.radians(orientation_deg)),
        #                                 location_p_center[1] + 0.5 * dimensions['length'] * math.sin(math.radians(orientation_deg)),
        #                                 location_p_center[2]]
                                    
        #                             location_p3 = location_p1[:2] + [location_p1[2] + dimensions['height']]
        #                             location_p4 = location_p2[:2] + [location_p2[2] + dimensions['height']]
                                    
        #                             plate_location_per_cw.extend([location_p1, location_p2, location_p3, location_p4]) # no need for further flatten.
        #                             plate_orientation_per_cw.append(orientation_deg) # ok
        #                             plate_width_per_cw.append(dimensions['width']) # ok
                                    
        #                             if not plate_iterator.next():
        #                                 break
                    
        #             cw_corner_points = get_rectangle_corners(plate_location_per_cw)
        #             cw_elevation = min(pt[2] for pt in cw_corner_points)
                    
        #             cw_location = [pt.tolist() for pt in cw_corner_points if pt[2] == cw_elevation]
        #             # this will return an invalid if the curtain wall has been cut somewhere.
                    
        #             # ----------------- switched version.
        #             if len(cw_location) < 2:
        #                 corner_points = np.array(cw_corner_points)
        #                 differences = np.abs(corner_points[:, 2] - cw_elevation)
        #                 sorted_indices = np.argsort(differences)
        #                 closest_indices = sorted_indices[:2]
        #                 cw_location = [corner_points[idx].tolist() for idx in closest_indices]
                                        
        #             cw_length = distance_between_points(cw_location[0], cw_location[1])
        #             cw_width = find_most_common_value(plate_width_per_cw)[0]
        #             cw_orientation = find_most_common_value(plate_orientation_per_cw)[0]
                    
        #             # use the IfcMember bottom to update the elevation and 3D location.
        #             if min_z_per_cw is not None and min_z_per_cw < cw_elevation:
                        
        #                 cw_elevation = min_z_per_cw

        #                 for sub_loc in cw_location:
        #                     if sub_loc:
        #                         sub_loc[-1] = cw_elevation

        #             self.info_curtainwalls.append({
        #                 'id': cw.GlobalId,
        #                 'elevation': round(cw_elevation,4),
        #                 'location': cw_location,
        #                 'length': cw_length,
        #                 'width': round(cw_width, 4),
        #                 'orientation': cw_orientation % 180.0,
        #                 })
        #         
        #         elif hasattr(cw, 'Representation'):

        #             if hasattr(cw.Representation,'Representations'):
        #                 for r in cw.Representation.Representations:
        #                     if r.RepresentationIdentifier == 'Body':
        #                         self._get_curtainwall_dimensions(cw)
        #                         break
        #             
        #         else:
        #             # Logging the missing or malformed attribute rather than raising an exception to allow processing to continue
        #             print(f'Attribute check failed for IfcCurtainWall with guid {cw.GlobalId}.')

        # except Exception as e:
        #     # General exception handling to catch unexpected errors
        #     print(f"An error occurred: {str(e)}")

#curtainwall ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
#column ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    # function notes.
    def get_column_dimensions(self):
        
        iterator = ifcopenshell.geom.iterator(
            self.settings, self.model, multiprocessing.cpu_count(), include=self.st_columns)
        
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
                    self.info_st_columns.append(dict_of_a_column)

                    if not iterator.next():
                        break
    
    # function notes.
    def enrich_column_information(self):
        id_to_column = {c.GlobalId: c for c in self.st_columns}
        self.info_st_columns = [
            self._update_column_info(id_to_column[info['id']], info) \
                for info in self.info_st_columns if \
                    info['id'] in id_to_column and \
                            info['height'] is not None
        ]
    
    # function notes.
    def _get_location_column(self, column):
        c_location_pt_from_body = None
        c_location_pt_directplacement = None
        
        for r in column.Representation.Representations:
            if r.RepresentationIdentifier == 'Body':
                if hasattr(r.Items[0],'MappingSource'):
                    mapped_r = r.Items[0].MappingSource.MappedRepresentation
                    if hasattr(mapped_r.Items[0], 'Position'):
                        c_location_pt_from_body = mapped_r.Items[0].Position.Location.Coordinates
                        break  # Assuming we only need the first matching 'Body'
                else:
                    continue

        # ---------------------------------------------------------------------------------------------------------------------------
        # # old version (save): some z coordinations are doubled due to the unclear reference storeys.
        # if c_location_pt_from_body==None or (abs(c_location_pt_from_body[0])<0.001 and abs(c_location_pt_from_body[1])<0.001):
        #     c_location_pt_directplacement = column.ObjectPlacement.RelativePlacement.Location.Coordinates
        #     rel_storey_elevation = column.ObjectPlacement.PlacementRelTo.PlacesObject[0].Elevation
        #     rel_placement = column.ObjectPlacement.PlacementRelTo.RelativePlacement.Location.Coordinates
        #     c_location_pt_directplacement = tuple(sum(x) for x in zip(c_location_pt_directplacement, rel_placement))
        #     c_location_pt_directplacement = c_location_pt_directplacement[:2] + (c_location_pt_directplacement[2] + rel_storey_elevation,)
        # ---------------------------------------------------------------------------------------------------------------------------
        
        # directly use the "rel_storey_elevation" as the point location z value.
        if c_location_pt_from_body==None or (abs(c_location_pt_from_body[0])<0.001 and abs(c_location_pt_from_body[1])<0.001):
            c_location_pt_directplacement = column.ObjectPlacement.RelativePlacement.Location.Coordinates
            rel_storey_elevation = column.ObjectPlacement.PlacementRelTo.PlacesObject[0].Elevation
            rel_placement = column.ObjectPlacement.PlacementRelTo.RelativePlacement.Location.Coordinates
            c_location_pt_directplacement = tuple(sum(x) for x in zip(c_location_pt_directplacement, rel_placement))
            c_location_pt_directplacement = c_location_pt_directplacement[:2] + (rel_storey_elevation,)
            return c_location_pt_directplacement
        else:
            return c_location_pt_from_body
    
    # function notes.
    def _update_column_info(self, column, info_c):

        column_location_p1 = self._get_location_column(column)
        column_location_p1 = self._divide_unit(column_location_p1)
        
        column_location_p2 = column_location_p1[:2] + (column_location_p1[2] + info_c['height'],)

        info_c.update({
            "elevation": round(column_location_p1[-1],3),
            'location': [column_location_p1, column_location_p2],
        })

        return info_c
    
    @time_decorator
    def extract_all_columns(self):
        
        # all the columns are considered structural columns
        self.info_st_columns = []

        # get id and locations of IfcColumns.
        self.get_column_dimensions()

        # update hte elevation and location ends of IfcColumns
        self.enrich_column_information()

        # save data and display
        self.write_dict_columns()
        
#column ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
# write to dictionaries ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    # function notes.
    def write_dict_floors(self):
        
        dict_info_floors = remove_duplicate_dicts(self.info_floors)
        try:
            with open(os.path.join(self.out_fig_path, 'info_floors.json'), 'w') as json_file:
                json.dump(dict_info_floors, json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'floors.json'}: {e}")
    
    # function notes.
    def write_dict_floor_slabs(self):

        try:
            with open(os.path.join(self.out_fig_path, 'info_floor_slabs.json'), 'w') as json_file:
                json.dump(self.info_unique_floor_slabs, json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'floor_slabs.json'}: {e}")

    # function notes.
    def write_dict_columns(self):

        dict_info_columns = remove_duplicate_dicts(self.info_st_columns)
        try:
            with open(os.path.join(self.out_fig_path, 'info_columns.json'), 'w') as json_file:
                json.dump(dict_info_columns, json_file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write to {self.out_fig_path + 'columns.json'}: {e}")

    # function notes.
    def write_dict_walls(self):
        
        # st_walls
        dict_info_walls = remove_duplicate_dicts(self.info_st_walls)
        if dict_info_walls:
            try:
                with open(os.path.join(self.out_fig_path, 'info_st_walls.json'), 'w') as json_file:
                    json.dump(dict_info_walls, json_file, indent=4)
            except IOError as e:
                raise IOError(f"Failed to write to {self.out_fig_path + 'st_walls.json'}: {e}")
        
        # ns_walls
        dict_info_walls = remove_duplicate_dicts(self.info_ns_walls)
        if dict_info_walls:
            try:
                with open(os.path.join(self.out_fig_path, 'info_ns_walls.json'), 'w') as json_file:
                    json.dump(dict_info_walls, json_file, indent=4)
            except IOError as e:
                raise IOError(f"Failed to write to {self.out_fig_path + 'ns_walls.json'}: {e}")

        # curtain_walls
        dict_info_walls = remove_duplicate_dicts(self.info_curtainwalls)
        if dict_info_walls:
            try:
                with open(os.path.join(self.out_fig_path, 'info_ct_walls.json'), 'w') as json_file:
                    json.dump(dict_info_walls, json_file, indent=4)
            except IOError as e:
                raise IOError(f"Failed to write to {self.out_fig_path + 'ct_walls.json'}: {e}")
    
# write to dictionaries ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ 
#===================================================================================================

#===================================================================================================
# wall visualizations ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    # for walls
    def wall_width_histogram(self, display_walls=[]):
        
        if display_walls:

            values = [w['width'] for w in display_walls if 'width' in w]

            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes((0.0875, 0.1, 0.875, 0.875))
            ax.hist(values, weights=np.ones(len(values)) / len(values), bins=20, color='#bcbd22', edgecolor='black')
            ax.set_xlabel('Width of IfcWalls', color='black', fontsize=12)
            ax.set_ylabel("Percentage Frequency Distribution", color="black", fontsize=12)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlim(xmin=0.0, xmax=max(values)+0.1)

            plt.savefig(os.path.join(self.out_fig_path, 'wall_width_histogram.png'), dpi=200)
            plt.close(fig)
        
    # function notes.
    def wall_length_histogram(self, display_walls=[]):

        if display_walls:
            
            values = [w['length'] for w in display_walls if 'length' in w]

            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes((0.0875, 0.1, 0.875, 0.875))
            ax.hist(values, weights=np.ones(len(values)) / len(values), bins=20, color='#bcbd22', edgecolor='black')
            ax.set_xlabel('Length of IfcWalls', color='black', fontsize=12)
            ax.set_ylabel("Percentage Frequency Distribution", color="black", fontsize=12)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlim(xmin=0.0, xmax=max(values)+1.0)

            plt.savefig(os.path.join(self.out_fig_path, 'wall_length_histogram.png'), dpi=200)
            plt.close(fig)

    # function notes.
    def wall_orientation_histogram(self, display_walls=[]):

        if display_walls:

            values = [w['orientation'] for w in display_walls if 'orientation' in w]

            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes((0.0875, 0.1, 0.875, 0.875))
            ax.hist(values, weights=np.ones(len(values)) / len(values), bins=20, color='#bcbd22', edgecolor='black')
            ax.set_xlabel('Orientation of IfcWalls [0°,180°)', color='black', fontsize=12)
            ax.set_ylabel("Percentage Frequency Distribution", color="black", fontsize=12)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xticks(np.arange(0, 180, 30))
            ax.set_xlim(xmin=0, xmax=180+1.0)

            plt.savefig(os.path.join(self.out_fig_path, 'wall_orientation_histogram.png'), dpi=200)
            plt.close(fig)

    @time_decorator
    def wall_display(self):
        self.wall_width_histogram(display_walls=self.info_walls+self.info_curtainwalls)
        self.wall_length_histogram(display_walls=self.info_walls+self.info_curtainwalls)
        self.wall_orientation_histogram(display_walls=self.info_walls+self.info_curtainwalls)

# wall visualizations ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ 
#===================================================================================================

#===================================================================================================
# displayandexport ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 

    @time_decorator
    def wall_column_floor_location_display(
        self, view_elev=None, view_azim=None, plot_main_plane_directions=False, plane_vector_length=1):

        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        self.elev = view_elev if view_elev is not None else self.elev 
        self.azim = view_azim if view_azim is not None else self.azim
         
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.axis('off')
        
        # Display Walls
        display_walls = self.info_st_walls + self.info_ns_walls + self.info_curtainwalls
        if display_walls:
            st_wall_values, ns_wall_values, ct_wall_values = [], [], []
            for w in display_walls:
                if 'location' in w:
                    if w['id'] in self.id_st_walls:
                        st_wall_values.append(w['location'])
                    elif  w['id'] in self.id_ns_walls:
                        ns_wall_values.append(w['location'])
                    elif  w['id'] in self.id_ct_walls:
                        ct_wall_values.append(w['location'])

            for st_v in st_wall_values:
                start_point, end_point = st_v
                xs, ys, zs = zip(start_point, end_point)
                ax.plot(xs, ys, zs, marker='s', color='black', linewidth=1.5, markersize=1, alpha=0.875, label="IfcWall(S)")
        
            for ns_v in ns_wall_values:
                start_point, end_point = ns_v
                xs, ys, zs = zip(start_point, end_point)
                ax.plot(xs, ys, zs, marker='s', color='grey', linewidth=1, markersize=1, label="IfcWall(N)")

            for ct_v in ct_wall_values:
                start_point, end_point = ct_v
                xs, ys, zs = zip(start_point, end_point)
                ax.plot(xs, ys, zs, marker='s', color='royalblue', linewidth=1, markersize=1, label="IfcWall(CT)")
        
        # Display Columns
        display_columns = self.info_st_columns
        if display_columns:
            column_values = [c['location'] for c in display_columns if 'location' in c]
            for v in column_values:
                start_point, end_point = v
                xs, ys, zs = zip(start_point, end_point)
                ax.plot(xs, ys, zs, marker='o', color='olivedrab', linewidth=1.5, markersize=1, alpha=0.375, label="IfcColumn(S)")
        
        # Display Floors
        values = [w['location'] for w in self.info_floors if 'location' in w]
        values = [x for xs in values for x in xs]
        
        # Use a set to track plotted lines
        plotted_lines = set()
        
        for v in values:
            start_point, end_point = tuple(v[0]), tuple(v[1])  # Convert to tuples
            # Create a sorted tuple of the points to avoid duplicating lines
            line = tuple(sorted((start_point, end_point)))
            if line not in plotted_lines:
                plotted_lines.add(line)
                xs, ys, zs = zip(start_point, end_point)
                ax.plot(xs, ys, zs, marker='x', color='salmon', linewidth=3, markersize=0.25, alpha=0.35, label="IfcSlab")
        
        # Display Main Plane Directions
        if plot_main_plane_directions:

            ax.quiver(
                self.corner_x_value,
                self.corner_y_value,
                self.corner_z_value,
                0,
                0,
                plane_vector_length*0.5,
                color='tomato', arrow_length_ratio=0.3)
            if self.main_directions:
                for degree in self.main_directions:
                    radian = np.deg2rad(degree)
                    x = np.cos(radian)
                    y = np.sin(radian)
                    ax.quiver(
                        self.corner_x_value, 
                        self.corner_y_value,
                        self.corner_z_value, 
                        plane_vector_length*x, 
                        plane_vector_length*y,
                        0,
                        color='navy', arrow_length_ratio=0.3)

        # Ensure tight layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(
            self.out_fig_path, self._get_file_prefix_code(self.ifc_file_name)+'_wall_column_floor_location_map.png'), dpi=200)
        plt.close(fig)
        
# displayandexport ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================

#===================================================================================================
# triangle display.

    @time_decorator
    def export_triangle_geometry(self, id=[], x_box=1, y_box=1, z_box=1):
        
        if not id:
            raise ValueError("please specify the Guid of the element")
        else:
            element = self.model.by_guid(id)
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            matrix = shape.transformation.matrix.data
            matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
            location = matrix[:,3][0:3]
        
            # A nested numpy array e.g. [[v1x, v1y, v1z], [v2x, v2y, v2z], ...]
            grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
            # A nested numpy array e.g. [[e1v1, e1v2], [e2v1, e2v2], ...]
            # grouped_edges = ifcopenshell.util.shape.get_edges(shape.geometry)
            # A nested numpy array e.g. [[f1v1, f1v2, f1v3], [f2v1, f2v2, f2v3], ...]
            grouped_faces = ifcopenshell.util.shape.get_faces(shape.geometry)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect([x_box,y_box,z_box])  # Equal aspect ratio

            for fc in grouped_faces:
                vertices = grouped_verts[fc]
                vertices = np.vstack([vertices, vertices[0]])
                x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
                ax.plot(x, y, z, linewidth=1, color='black', alpha=0.5)
            
            plt.title(id)
            plt.show()
    
    # tempo visualization
    def tempo_export_triangle_geometry_single_wall(self, ids=[], x_box=1, y_box=1, z_box=1, elev=30, azim=45):
        
        if not ids:
            raise ValueError("Please specify the GUID of the element")
        
        for id in ids:

            element = self.model.by_guid(id)
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            matrix = shape.transformation.matrix.data
            matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
            location = matrix[:, 3][0:3]
            
            # Get vertices and faces
            grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
            grouped_faces = ifcopenshell.util.shape.get_faces(shape.geometry)

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect([x_box, y_box, z_box])  # Equal aspect ratio
            
            # Plot each face
            for fc in grouped_faces:
                vertices = grouped_verts[fc]
                poly3d = [[tuple(vertices[j]) for j in range(len(vertices))]]
                ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=1, edgecolors='black', alpha=0.5))
            
            # Determine ranges for each axis
            x_range = np.max(grouped_verts[:, 0]) - np.min(grouped_verts[:, 0])
            y_range = np.max(grouped_verts[:, 1]) - np.min(grouped_verts[:, 1])
            z_range = np.max(grouped_verts[:, 2]) - np.min(grouped_verts[:, 2])
            
            max_range = max(x_range, y_range, z_range)
            
            # Set larger limits for the axes to keep the box aspect consistent
            ax.set_xlim([np.min(grouped_verts[:, 0]) - max_range * 0.1, np.max(grouped_verts[:, 0]) + max_range * 0.1])
            ax.set_ylim([np.min(grouped_verts[:, 1]) - max_range * 0.1, np.max(grouped_verts[:, 1]) + max_range * 0.1])
            ax.set_zlim([np.min(grouped_verts[:, 2]) - max_range * 0.1, np.max(grouped_verts[:, 2]) + max_range * 0.1])

            ax.view_init(elev=elev, azim=azim)
            
            ax.set_title(id)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(self.out_fig_path, str(id) + 'triangle_meshes.png'), dpi=100)
            plt.close(fig)
    
    def triangle_display_of_one_element(self, element):

        shape = ifcopenshell.geom.create_shape(self.settings, element)
        matrix = shape.transformation.matrix.data
        matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
        location = matrix[:, 3][0:3]

        location_array = np.array(location)
        # Get vertices and faces
        grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        grouped_verts = location_array + grouped_verts
        grouped_faces = ifcopenshell.util.shape.get_faces(shape.geometry)
        
        # Plot each face
        poly3ds_per_element = []
        for fc in grouped_faces:
            vertices = grouped_verts[fc]
            poly3d = [[tuple(vertices[j]) for j in range(len(vertices))]]
            poly3ds_per_element.append(poly3d)
    
        return poly3ds_per_element, grouped_verts
    
    def export_triangle_geometry_of_all_walls(self, x_box=1, y_box=1, z_box=1, elev=45, azim=50):
        
        all_wall_ids = list(set(self.id_ns_walls)) # carefully!
        all_plate_ids = [wall_plate.GlobalId for wall_plate in self.plates]
        all_element_ids = all_wall_ids + all_plate_ids
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([x_box, y_box, z_box])  # Equal aspect ratio

        all_grouped_verts = None
        started = False
        for id in all_element_ids:

            # color setup. 
            # wall_ids = ['1ikpL435zCYA4QPkDWyd7c', '1ikpL435zCYA4QPkDWyd4L', '1ikpL435zCYA4QPkDWydy1']
            element_edgecolor = None
            if id == '1ikpL435zCYA4QPkDWyd7c':
                element_edgecolor = 'maroon'
            elif id == '1ikpL435zCYA4QPkDWyd4L':
                element_edgecolor = 'gray'
            else:
                element_edgecolor = 'lightskyblue'

            element = self.model.by_guid(id)
            poly3ds_per_element, grouped_verts_per_element = self.triangle_display_of_one_element(element)
            
            if started:
                all_grouped_verts = np.vstack((all_grouped_verts, grouped_verts_per_element))
            else:
                all_grouped_verts = grouped_verts_per_element
                started = True

            # Plot each face
            for poly3d in poly3ds_per_element:
                ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=1, edgecolors=element_edgecolor, alpha=0.5))
                
        x_range = np.max(all_grouped_verts[:, 0]) - np.min(all_grouped_verts[:, 0])
        y_range = np.max(all_grouped_verts[:, 1]) - np.min(all_grouped_verts[:, 1])
        z_range = np.max(all_grouped_verts[:, 2]) - np.min(all_grouped_verts[:, 2])
        max_range = max(x_range, y_range, z_range)

        # Set larger limits for the axes to keep the box aspect consistent
        ax.set_xlim([np.min(all_grouped_verts[:, 0]) - max_range * 0.1, np.max(all_grouped_verts[:, 0]) + max_range * 0.1])
        ax.set_ylim([np.min(all_grouped_verts[:, 1]) - max_range * 0.1, np.max(all_grouped_verts[:, 1]) + max_range * 0.1])
        ax.set_zlim([np.min(all_grouped_verts[:, 2]) - max_range * 0.1, np.max(all_grouped_verts[:, 2]) + max_range * 0.1])

        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.axis('off')

        # Activate axis values
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_fig_path, self.ifc_file_name + str(elev) + str(azim) +'_triangle_meshes.png'), dpi=300)
        plt.close(fig)

    def export_triangle_geometry_of_a_curtainwall(self, x_box=1, y_box=1, z_box=1, elev=45, azim=50):
        
        all_plate_ids = [wall_plate.GlobalId for wall_plate in self.plates]
        all_element_ids = all_plate_ids
        
        fig = plt.figure(figsize=(30, 20))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([x_box, y_box, z_box])  # Equal aspect ratio

        all_grouped_verts = None
        started = False
        for id in all_element_ids:

            # color setup. 
            # wall_ids = ['1ikpL435zCYA4QPkDWyd7c', '1ikpL435zCYA4QPkDWyd4L', '1ikpL435zCYA4QPkDWydy1']
            element_edgecolor = None
            if id == '1ikpL435zCYA4QPkDWyd7c':
                element_edgecolor = 'maroon'
            elif id == '1ikpL435zCYA4QPkDWyd4L':
                element_edgecolor = 'gray'
            else:
                element_edgecolor = 'lightskyblue'

            element = self.model.by_guid(id)
            poly3ds_per_element, grouped_verts_per_element = self.triangle_display_of_one_element(element)
            
            if started:
                all_grouped_verts = np.vstack((all_grouped_verts, grouped_verts_per_element))
            else:
                all_grouped_verts = grouped_verts_per_element
                started = True

            # Plot each face
            for poly3d in poly3ds_per_element:
                ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=1, edgecolors=element_edgecolor, alpha=0.95))
                 
        # Highlight vertices
        ax.scatter(all_grouped_verts[:, 0], all_grouped_verts[:, 1], all_grouped_verts[:, 2], color='black', s=8)

        x_range = np.max(all_grouped_verts[:, 0]) - np.min(all_grouped_verts[:, 0])
        y_range = np.max(all_grouped_verts[:, 1]) - np.min(all_grouped_verts[:, 1])
        z_range = np.max(all_grouped_verts[:, 2]) - np.min(all_grouped_verts[:, 2])
        max_range = max(x_range, y_range, z_range)

        # Set larger limits for the axes to keep the box aspect consistent
        ax.set_xlim([np.min(all_grouped_verts[:, 0]) - max_range * 0.1, np.max(all_grouped_verts[:, 0]) + max_range * 0.1])
        ax.set_ylim([np.min(all_grouped_verts[:, 1]) - max_range * 0.1, np.max(all_grouped_verts[:, 1]) + max_range * 0.1])
        ax.set_zlim([np.min(all_grouped_verts[:, 2]) - max_range * 0.1, np.max(all_grouped_verts[:, 2]) + max_range * 0.1])

        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.axis('off')

        # Activate axis values
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_fig_path, 'curtainwall_triangle_meshes.png'), dpi=300)
        plt.close(fig)
    
# triangle display ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#===================================================================================================


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