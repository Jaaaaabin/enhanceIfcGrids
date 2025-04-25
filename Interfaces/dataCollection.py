
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import pandas as pd
import os

class DataCollector:
    """
    The DataCollector class extracts and compiles basic metadata and object counts from 
    all .ifc files in a specified folder. It gathers schema info, descriptions, and originating systems, 
    and optionally counts occurrences of key building components (e.g., walls, doors, stairs).
    The results are written into a basic_infos.csv file.
    The class handles missing files and general errors gracefully using exception handling.
    """
    def __init__(self, folder_path):

        try:
            
            self.folder_path = folder_path
            self.infos = []

            self.file_names = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".ifc")]
            
            self.required_objects = [
                'IfcSlab',
                'IfcRoof',
                'IfcWallStandardCase',
                'IfcWall',
                'IfcCurtainWall',
                'IfcColumn',
                'IfcBeam', 
                'IfcDoor',
                'IfcWindow',
                'IfcStair',
                ]
            
            self._collect_infos(read_design_counts=True)
            
            
        except ifcopenshell.errors.FileNotFoundError:
            print(f"Error: File '{folder_path}' not found.")

        except Exception as e:
            print(f"An error occurred: {e}")


    def _collect_infos(self, read_design_counts=False):
        
        df_headers_basic = ['File Name', 'Schema Identifier', 'Description', 'Originating System']

        for file_name in self.file_names:
            
            # read the ifc file.
            ifc_file = ifcopenshell.open(os.path.join(self.folder_path,file_name))
            
            # read the basci infos
            model_name = file_name.replace('.ifc', '')
            ifc_version = ifc_file.schema
            ifc_description = ifc_file.wrapped_data.header.file_description.description
            ifc_originating_system = ifc_file.wrapped_data.header.file_name.originating_system

            info_per_file = [model_name, ifc_version, ifc_description, ifc_originating_system]

            if read_design_counts:
                design_counts = self._collect_object_counts(ifc_file)
                info_per_file += design_counts

            self.infos.append(info_per_file)
            print(f"==========DataCollector===========\n{file_name}")
        
        # write to a csv file.
        df_headers = df_headers_basic + self.required_objects if read_design_counts else df_headers_basic
        df_all_info = pd.DataFrame(self.infos, columns=df_headers)
        df_all_info.to_csv(os.path.join(self.folder_path,'basic_infos.csv'), index=False)

    def _collect_object_counts(self, ifc_file):
        
        count_elems: dict = {}
        all_ifc_elements = ifc_file.by_type("IfcProduct", include_subtypes=True)
        for element in all_ifc_elements:
            ifc_type = element.get_info()['type']
            if ifc_type not in count_elems.keys():
                count_elems[ifc_type] = 1
            else:
                count_elems[ifc_type] += 1
        [count_elems.update({k:0}) for k in self.required_objects if k not in count_elems.keys()]

        object_counts = [count_elems[key] for key in self.required_objects if key in self.required_objects]

        # when there's no roof.
        return object_counts

if __name__ == "__main__":

    DATA_VALIDATION_1_PATH = r'C:\dev\phd\enrichIFC\enrichIFC\data\data_validation_1'
    data_collector = DataCollector(DATA_VALIDATION_1_PATH)
            
    