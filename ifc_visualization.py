import os
from ifcDataExtractor import IfcExtractor

DATA_PATH = r'C:\dev\phd\enrichIFC\enrichIFC\paperfigures\multilayerwalls\data'
DATA_RES_PATH = r'C:\dev\phd\enrichIFC\enrichIFC\paperfigures\multilayerwalls\res'

def process_ifc_file(input_path, output_path):

    extractor = IfcExtractor(input_path, output_path)

    extractor.extract_all_walls()

    # extractor.export_triangle_geometry_of_all_walls(elev=45, azim=50)

    extractor.export_triangle_geometry_of_a_curtainwall(elev=35, azim=60)

# ----------
try:
    model_paths = [filename for filename in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, filename))]
    
    for model_path in model_paths:
        
        process_ifc_file(
            os.path.join(DATA_PATH, model_path),
            os.path.join(DATA_RES_PATH, model_path))
        break

except Exception as e:
    print(f"Error accessing directory {DATA_PATH}: {e}")
