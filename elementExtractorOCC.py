import OCC.Core
import OCC.Core.Units
import OCC.Core.GProp
import OCC.Core.BRepGProp
import OCC.Core.TopoDS, OCC.Core.BRep
import OCC.Extend.DataExchange
import OCC.Display.SimpleGui

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import multiprocessing

import os
import json
import copy
import numpy as np

# important references: Tri&Brep https://blenderbim.org/docs-python/ifcopenshell-python/geometry_processing.html
# important references: Brep https://sourceforge.net/p/ifcopenshell/discussion/1782717/thread/f1d8c8cc/

class GeometryProcessor:

    def __init__(self, model_path=[], read='triangle'):

        self.model_path = model_path
        self.folder_path = '\\'.join(model_path.split('\\')[:-1])

        # # =========================brep or triangle=========================        
        self.type_geo = read
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
        
        self.settings  = settings
        self.model = ifcopenshell.open(self.model_path)
        self.info_walls = []

        self.read_geometry()
        self.write_dict_walls()

    def vertices2wall(self, vertices):

        # Convert the list of vertices into a NumPy array for easier manipulation
        verts_array = np.array(vertices)
        x_range = np.ptp(verts_array[:, 0])
        y_range = np.ptp(verts_array[:, 1])
        z_range = np.ptp(verts_array[:, 2])
        
        # Assuming the primary orientation of the wall is along the axis with the largest range
        # and that the height is along the z-axis
        if x_range > y_range:
            length = x_range
            thickness = y_range
        else:
            length = y_range
            thickness = x_range
        height = z_range
        
        return {
            'length': length,
            'height': height,
            'thickness': thickness
        }

    def read_geometry(self):

        # todo. combine the wall geometry together with the location, direction
        # what information can we get from the columns via OCC?

        all_wall_elements = self.model.by_type("IfcWall") + self.model.by_type("IfcCurtainWall") 
        iterator = ifcopenshell.geom.iterator(
            self.settings, self.model, multiprocessing.cpu_count(), include=all_wall_elements)
        self.info_walls = []
        
        if self.type_geo == 'triangle':            
            if iterator.initialize():
                while True:

                    shape = iterator.get()
                    matrix = shape.transformation.matrix.data
                    matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
                    
                    # materials = shape.geometry.materials
                    # material_ids = shape.geometry.material_ids
                    # faces = shape.geometry.faces
                    # edges = shape.geometry.edges
                    # verts = shape.geometry.verts
                    # grouped_faces = ifcopenshell.util.shape.get_faces(shape.geometry)
                    # grouped_edges = ifcopenshell.util.shape.get_edges(shape.geometry)
                    
                    location = matrix[:,3][0:3]
                    grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
                    wall_dimensions = self.vertices2wall(grouped_verts)
                    dict_of_a_wall = {
                        "id": shape.guid,
                        "location":location.tolist(),
                        }
                    dict_of_a_wall.update(wall_dimensions)
                    self.info_walls.append(dict_of_a_wall)

                    # ... write code to process geometry here ...
                    if not iterator.next():
                        break
        
        elif self.type_geo == 'brep':            
            if iterator.initialize():
                while True:
                    shape = iterator.get()
                    geometry = shape.geometry
                    shape_gpXYZ = geometry.Location().Transformation().TranslationPart()
                    print(shape_gpXYZ.X(), shape_gpXYZ.Y(), shape_gpXYZ.Z())
                    if not iterator.next():
                        break
    
    def write_dict_walls(self):

        try:
            info_walls_file = copy.deepcopy(self.model_path)
            info_walls_file = info_walls_file.replace('.ifc','_info_walls.json')
            
            with open(info_walls_file, 'w') as json_file:
                json.dump(self.info_walls, json_file, indent=4)

        except IOError as e:
            raise IOError(f"Failed to write to {self.model_path}: {e}")

TEST_FOLDER = r'C:\dev\phd\enrichIFC\preparedata\rvt2ifc\test4occ\simple'
model_paths = [filename for filename in os.listdir(TEST_FOLDER) if os.path.isfile(os.path.join(TEST_FOLDER, filename)) and filename.endswith(".ifc")]

for model_path in model_paths:
    ifc_geometry_processor = GeometryProcessor(model_path=os.path.join(TEST_FOLDER, model_path))


    # Initialize the visualization, if necessary
    # self.display, self.start_display, self.add_menu, self.add_function_to_menu = OCC.Display.SimpleGui.init_display()
    
    # def load_geometry(self, file_path):
    #     """Loads geometry from a file."""
    #     shape = OCC.Extend.DataExchange.read_step_file(file_path)
    #     return shape
    
    # def process_geometry(self, shape):
    #     """Process the geometry in some way. Placeholder for actual operations."""
    #     # Example operation: Get bounding box
    #     bbox = OCC.Core.Bnd.Bnd_Box()
    #     OCC.Core.BRepBndLib.brepbndlib_Add(shape, bbox)
    #     x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
    #     return (x_min, y_min, z_min, x_max, y_max, z_max)
    
    # def display_geometry(self, shape):
    #     """Display the geometry using the built-in viewer."""
    #     self.display.DisplayShape(shape, update=True)
    #     self.start_display()


# -------------------------------------------------------------------
# from Fiona.

# def set_settings(brep):
#     if brep:
#         # can convert almost all IFC representations into a brep
#         settings = ifcopenshell.geom.settings()
#         settings.set(settings.USE_PYTHON_OPENCASCADE, True)
#         settings.set(settings.DISABLE_TRIANGULATION, True)
#         settings.USE_WORLD_COORDS = True
#         settings.set(settings.USE_BREP_DATA, True)
#     else:
#         # can convert almost all IFC representations into a triangulated mesh
#         settings = ifcopenshell.geom.settings()
#         settings.USE_PYTHON_OPENCASCADE = True
#         # very weird behaviour here... but this seems to be the only option to activate both triangulation and the generation of normals
#         settings.DISABLE_TRIANGULATION = 1
#         settings.set(settings.DISABLE_TRIANGULATION, False)
#         settings.WELD_VERTICES = False
#         settings.NO_NORMALS = False
#         settings.GENERATE_UVS = True
#         settings.S = False
#         # settings.USE_WORLD_COORDS = True
#         settings.set(settings.USE_WORLD_COORDS, True)


# if include_geometry:
#     instance_iterator = ifcopenshell.geom.iterator(self.settings, self.model, multiprocessing.cpu_count(),
#                                                    include=all_products)
#     instance_iterator.initialize()
#     instance_counter = 0
#     guid_seen = []
    # while True:
    #     # every 20th instance print the progress
    #     if instance_counter % 20 == 0:
    #         print(f"parsing instance {instance_counter}")
    #     i_geometry = instance_iterator.get()
    #     if self.brep:
    #         i_type_ifc: str = i_geometry.data.type
    #         i_guid_ifc: str = i_geometry.data.guid
    #     else:
    #         i_type_ifc: str = i_geometry.type
    #         i_guid_ifc: str = i_geometry.guid
    #     #todo: the elements seem to be duplicated in the ifc file, so we need a special stopping criterion
    #     if i_guid_ifc in guid_seen:
    #         if not instance_iterator.next():
    #             break
    #         else:
    #             continue
    #     else:
    #         guid_seen.append(i_guid_ifc)
    #     #instance = BuildingElement(instance_counter, i_guid_ifc, i_type_ifc, geometry=i_geometry, geometry_type="occ_triangulation")
    #     instance = BuildingElement(instance_counter, i_guid_ifc, i_type_ifc, geometry=i_geometry.geometry, geometry_type="occ_brep")
    #     instance_counter += 1
    #     all_instances.append(instance)