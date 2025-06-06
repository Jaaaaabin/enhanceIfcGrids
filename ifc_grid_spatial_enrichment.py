import os
from EnhancingComponent.ifcGridEnricher import IfcSpatialGridEnrichment

PROJECT_PATH = os.getcwd()
# DATA_FOLDER_PATH = PROJECT_PATH + r'\data\enriched'
DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_test'
DATA_RES_PATH = PROJECT_PATH + r'\res_extraction'

def enrich_ifc_file(input_path, output_path):

    enricher = IfcSpatialGridEnrichment(input_path, output_path)

    enricher.enrich_ifc_with_grids()
    
    # # store the reference dependencies in 'IfcRelReferencedInSpatialStructure'.
    # enricher.enrich_reference_relationships_relref()
    
    # store the reference dependencies in 'FootPrint' of the building elements.
    enricher.enrich_reference_relationships_relconstraint()

    enricher.save_the_enriched_ifc()

# ----------
try:

    model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
    
    for model_path in model_paths:
        enrich_ifc_file(
            os.path.join(DATA_FOLDER_PATH, model_path),
            os.path.join(DATA_RES_PATH, model_path))
        
        # break
        
except Exception as e:
    print(f"Error accessing directory {DATA_FOLDER_PATH}: {e}")