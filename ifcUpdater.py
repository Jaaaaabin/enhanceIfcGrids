import os
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element

class IfcUpdater:

    def __init__(self, file_in, file_out):
        
        self.ifc_file_path_in = file_in
        self.ifc_file_path_out = file_out

        self.settings = self._configure_settings()
        self.ifc_model = ifcopenshell.open(self.ifc_file_path_in)
        self.settings = ifcopenshell.geom.settings()
        
        self.info_cleaning = {} 

    def _configure_settings(self):

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

        return settings

    def _set_one_wall_common_property(self, wall, property_name, new_property):

        if not wall.IsDefinedBy:
            raise ValueError("Wall does not have any associated properties.")
        
        try:

            n_Pset_WallCommon = 0
            
            for p_set in list(wall.IsDefinedBy):
                
                if p_set.RelatingPropertyDefinition.Name == 'Pset_WallCommon':
                    
                    idx = []
                    notfound_property = True
                    n_Pset_WallCommon +=1

                    for ii, p in enumerate(p_set.RelatingPropertyDefinition.HasProperties):
                        if p.Name == property_name:
                            notfound_property = False
                            idx = ii
                            break # in one Pset_WallCommon, there's only one attribute with the 'property_name'

                    # if it's found, do update.
                    if idx:
                        new_p_set = list(p_set.RelatingPropertyDefinition.HasProperties)
                        new_p_set[idx] = new_property[0]
                        p_set.RelatingPropertyDefinition.HasProperties = tuple(new_p_set)

                    # if it doesn't exist, add it to the end.
                    elif notfound_property:
                        new_p_set = list(p_set.RelatingPropertyDefinition.HasProperties)
                        new_p_set.append(new_property[0])
                        p_set.RelatingPropertyDefinition.HasProperties = tuple(new_p_set)
            
            if n_Pset_WallCommon != 1:
                raise ValueError(f"The number of 'Pset_WallCommon' queried on wall{wall.GlobalId} is not equal to 1.")

        except Exception as e:
            print(f"Error updating wall {wall.GlobalId}: {str(e)}")
        
    def modify_common_property_walls(self, property_name, property_value,):
        
        # create the property: name, description, entity_value.
        new_property = self.ifc_model.createIfcPropertySingleValue(property_name, None, self.ifc_model.create_entity("IfcBoolean", property_value), None),
        
        wall_guids = self.info_cleaning.get(property_name, [])

        if wall_guids:
            walls = []
            for guid in wall_guids:
                walls.append(self.ifc_model.by_guid(guid))
        else:
            walls = self.ifc_model.by_type("IfcWall") + self.ifc_model.by_type('IfcWallStandardCase')

        for wall in walls:
            self._set_one_wall_common_property(wall, property_name, new_property)

    def collect_cleaning_information(self, property_cleaning, file_cleaning):
        
        info_all = []
        with open(file_cleaning, 'r') as file:
            for line in file:
                str_value = str(line.strip())
                info_all.append(str_value)

        self.info_cleaning.update({
            property_cleaning: info_all,
        })
    
    def save_updated_model(self):

        self.ifc_model.write(self.ifc_file_path_out)


PROJECT_DATA_PATH = r'C:\dev\phd\enrichIFC\enrichIFC\data'
DATA_INPUT_PATH = os.path.join(PROJECT_DATA_PATH, 'saved_set1')
DATA_OUTPUT_PATH = os.path.join(PROJECT_DATA_PATH, 'data_cleaned')

ifc_file_name = '3776779.ifc'

file_input = os.path.join(DATA_INPUT_PATH,ifc_file_name)
file_output = os.path.join(DATA_OUTPUT_PATH,ifc_file_name)

ifc_cleaning_file = ifc_file_name + '.txt'
file_cleaning = os.path.join(DATA_OUTPUT_PATH,ifc_cleaning_file)

ifc_updater = IfcUpdater(file_in=file_input, file_out=file_output,)

# LoadBearing.
ifc_updater.modify_common_property_walls(property_name='LoadBearing', property_value=True)
ifc_updater.collect_cleaning_information(property_cleaning='LoadBearing', file_cleaning=file_cleaning)
ifc_updater.modify_common_property_walls(property_name='LoadBearing', property_value=False)
ifc_updater.save_updated_model()

# # IsExternal.
# ifc_updater.modify_common_property_walls(property_name='IsExternal', property_value=True)
# ifc_updater.modify_common_property_walls(property_name='IsExternal', property_value=False, wall_guids=test_wall_guids)