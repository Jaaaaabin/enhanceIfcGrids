import os
from EnhancingComponent.ifcDataExtractor import IfcDataExtractor

# import itertools
from Interfaces.dataJsonAnalysis import JsonFileComparator

PROJECT_PATH = os.getcwd()
DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_autocon_test_no_grids'
# DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_test'
DATA_RES_PATH = PROJECT_PATH + r'\res_extraction'

def process_ifc_file(input_path, output_path):

    extractor = IfcDataExtractor(input_path, output_path)

    extractor.extract_all_floors()
    extractor.post_processing_floors_to_slabs()

    extractor.extract_all_columns()
    extractor.extract_all_walls()
    extractor.extract_all_curtainwalls()
    extractor.post_processing_walls()
    
    # extractor.wall_column_floor_location_display(plot_main_plane_directions=True, plane_vector_length=10)

    # # ==================
    # for the layout example.
    extractor.wall_column_floor_location_display(
        view_elev=40, view_azim=-130, plot_main_plane_directions=True, plot_main_plane=True, plane_vector_length=1)

# def compare_ifc_infos(data_path, ifc_a, ifc_2, json_name):

#     infoComparator = JsonFileComparator(data_path, ifc_a, ifc_2, json_name)
#     infoComparator.run_comparison()

# def combinations_from_shared_ifc_basis(all_ifcs):
    
#     ifc_groups = {}
#     basis_combinations = {}

#     for s in all_ifcs:
#         basis = s.rsplit('_', 1)[0] if '_' in s else s
#         if basis in ifc_groups:
#             ifc_groups[basis].append(s)
#         else:
#             ifc_groups[basis] = [s]

#     shared_ifc_groups = {basis: strings for basis, strings in ifc_groups.items() if len(strings) > 1}

#     for basis, strings in shared_ifc_groups.items():
#         basis_combinations[basis] = list(itertools.combinations(strings, 2))

#     return basis_combinations

# ----------
try:
    model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
    
    for model_path in model_paths:
        
        process_ifc_file(
            os.path.join(DATA_FOLDER_PATH, model_path),
            os.path.join(DATA_RES_PATH, model_path))

except Exception as e:
    print(f"Error accessing directory {DATA_FOLDER_PATH}: {e}")

# # ----------
# try:
#     model_paths = [filename for filename in os.listdir(DATA_RES_PATH) if not os.path.isfile(os.path.join(DATA_RES_PATH, filename))]
#     json_names = ['info_walls.json']

#     combinnations_ifc_variants = combinations_from_shared_ifc_basis(model_paths)
#     for basis, combos in combinnations_ifc_variants.items():
#         for combo in combos:
#             [compare_ifc_infos(DATA_RES_PATH, combo[0], combo[1], json_name) for json_name in json_names]

# except Exception as e:
#     print(f"Error accessing directory {DATA_RES_PATH}: {e}")
