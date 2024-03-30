import OCC.Core
import OCC.Core.Units
import OCC.Core.GProp
import OCC.Core.BRepGProp
import OCC.Core.TopoDS, OCC.Core.BRep
import OCC.Extend.DataExchange
import OCC.Display.SimpleGui

import numpy as np
import math
import os
import json
import copy
import multiprocessing
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

from quickTools import get_rectangle_corners

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
    
        self.plates = self.model.by_type("IfcPlate")
        self.members = self.model.by_type("IfcMember")
        
        self.process_curtainwall_elements()
        self.read_curtainwall_geometry()    
        # self.write_dict_walls()
    
    def _calc_element_orientation(self, element, deg_range=360):
        
        if element.ObjectPlacement.RelativePlacement.RefDirection is None:
            return 0.0
        
        orientation_vec = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios
        orientation_rad = math.atan2(orientation_vec[1], orientation_vec[0])
        orientation_deg = math.degrees(orientation_rad) % deg_range
        
        return round(orientation_deg, 4)
    
    def process_curtainwall_elements(self):

        self.id2orientation_plates = dict()
        for plate in self.plates:
            self.id2orientation_plates.update({
                plate.GlobalId: self._calc_element_orientation(plate,)
            })

        self.id2orientation_members = dict()
        for member in self.members:
            self.id2orientation_members.update({
                member.GlobalId: self._calc_element_orientation(member,)
            })

    def _shape2_location_verts(self, shape):

        matrix = shape.transformation.matrix.data
        matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
        location = matrix[:,3][0:3]
        grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        return location, grouped_verts
    
    def _vertices2dimensions(self, vertices):

        verts_array = np.array(vertices)
        x_range, y_range, z_range = np.ptp(verts_array, axis=0)
        length, width = max(x_range, y_range), min(x_range, y_range)
        height = z_range
        return {'length': length, 'height': height, 'width': width}

    def read_curtainwall_geometry(self):

        all_curtainwalls = self.model.by_type("IfcCurtainWall")
        
        for cw in all_curtainwalls:

            if hasattr(cw,'IsDecomposedBy') and len(cw.IsDecomposedBy)==1 and cw.IsDecomposedBy[0].is_a('IfcRelAggregates'):

                cw_related_objects = cw.IsDecomposedBy[0].RelatedObjects
                cw_related_plates = [ob for ob in cw_related_objects if ob.is_a('IfcPlate')]

                plate_iterator = ifcopenshell.geom.iterator(
                    self.settings, self.model, multiprocessing.cpu_count(), include=cw_related_plates)
                plate_location_per_cw = []

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
                            plate_location_per_cw.append([location_p1,location_p2,location_p3, location_p4])

                            if not plate_iterator.next():
                                break

                plate_location_per_cw = [x for xs in plate_location_per_cw for x in xs]
                corner_points = get_rectangle_corners(plate_location_per_cw)
                print ("test")
            
            else:
                raise ValueError(f'Please check the attribute of the IfcCurtainWall with guid {cw.GlobalId}.')


            
                
                


                # # local
            # component = related_components[0]
            # for r in component.Representation.Representations:
            #     if r.RepresentationIdentifier =='FootPrint':
            #         # local_points = r.Items[0].MappingSource.MappedRepresentation.Items[0].Points.CoordList
            #         local_points = r.Items[0].MappingSource.MappedRepresentation.Items[0].Points
            #         if isinstance(local_points,tuple):
            #             local_points = [list(c) for pt in local_points for c in pt]
            #         print ("Footprint:",local_points)

    
    def read_geometry(self):
        
        # todo. combine the wall geometry together with the location, direction
        # what information can we get from the columns via OCC?

        all_walls = self.model.by_type("IfcWall") + self.model.by_type("IfcCurtainWall") 
        iterator = ifcopenshell.geom.iterator(
            self.settings, self.model, multiprocessing.cpu_count(), include=all_walls)
        self.info_walls = []
        
        if self.type_geo == 'triangle':            
            if iterator.initialize():
                while True:
                    
                    shape = iterator.get()
                    # materials = shape.geometry.materials
                    # material_ids = shape.geometry.material_ids
                    # faces = shape.geometry.faces
                    # edges = shape.geometry.edges
                    # verts = shape.geometry.verts
                    # grouped_faces = ifcopenshell.util.shape.get_faces(shape.geometry)
                    # grouped_edges = ifcopenshell.util.shape.get_edges(shape.geometry)
                    
                    location, grouped_verts = self._shape2_location_dimensions(shape)
                    dimensions = self._vertices2dimensions(grouped_verts)
                    dict_of_a_wall = {
                        "id": shape.guid,
                        "location":location.tolist(),
                        }
                    dict_of_a_wall.update(dimensions)
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


TEST_FOLDER = r'C:\dev\phd\enrichIFC\preparedata\data_icccbe'
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