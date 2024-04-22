from ifcExtractor import IfcExtractor
from infoAnalysis import JsonFileComparator
from gridGenerator import GridGenerator
import os
import itertools

PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_test'
DATA_RES_PATH = PROJECT_PATH + r'\res'


def process_ifc_file(input_path, output_path):

    extractor = IfcExtractor(input_path, output_path)

    # extractor.export_triangle_geometry(id='2rzvwssmPB$Ogkk_N5eE98', z_box=.1)
    extractor.extract_all_floors()
    extractor.extract_all_columns()
    extractor.extract_all_walls()
    extractor.wall_and_column_location_display()

def compare_ifc_infos(data_path, ifc_a, ifc_2, json_name):

    infoComparator = JsonFileComparator(data_path, ifc_a, ifc_2, json_name)
    infoComparator.run_comparison()

def combinations_from_shared_ifc_basis(all_ifcs):
    
    ifc_groups = {}
    basis_combinations = {}

    for s in all_ifcs:
        basis = s.rsplit('_', 1)[0] if '_' in s else s
        if basis in ifc_groups:
            ifc_groups[basis].append(s)
        else:
            ifc_groups[basis] = [s]

    shared_ifc_groups = {basis: strings for basis, strings in ifc_groups.items() if len(strings) > 1}

    for basis, strings in shared_ifc_groups.items():
        basis_combinations[basis] = list(itertools.combinations(strings, 2))

    return basis_combinations

def preparation_of_grid_generation(
    work_path,
    ifc_model,
    info_floors='info_floors.json',
    info_st_columns='info_columns.json',
    info_st_walls='info_st_walls.json',
    info_ns_walls='info_ns_walls.json',
    info_ct_walls='info_ct_walls.json'):

    # initialization
    generator = GridGenerator(
        os.path.join(work_path, ifc_model),
        os.path.join(work_path, ifc_model, info_floors),
        os.path.join(work_path, ifc_model, info_st_columns),
        os.path.join(work_path, ifc_model, info_st_walls),
        os.path.join(work_path, ifc_model, info_ns_walls),
        os.path.join(work_path, ifc_model, info_ct_walls),
        )
    
    # preparation.
    generator.get_main_directions_and_storeys(num_directions=2) # static.
    generator.enrich_all_element_locations() # static.

    return generator

def building_grid_generation(basic_generator, new_parameters):
    
    # update the parameters.
    new_generator = basic_generator.update_parameters(new_parameters)
    
    # generate the grids
    new_generator.create_grids()

    # # calculate the losses
    new_generator.calculate_grid_wall_cross_loss(ignore_cross_edge=True)    # loss calculation.
    new_generator.calculate_grid_distance_deviation_loss()

    # display the grids
    new_generator.visualization_2d()
    
# main start path.
if __name__ == "__main__":

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
        
    # ----------
    try:
        model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
        
        for model_path in model_paths:

            # for each building model
            init_grid_generator = preparation_of_grid_generation(DATA_RES_PATH, model_path)

            best_thresholds = {
                'st_c_num': 3,
                'st_c_dist': 0.0001,
                'st_w_num': 2,
                'st_w_dist': 0.0001,
                'st_w_accumuled_length': 5,
                'ns_w_num': 1,
                'ns_w_dist': 0.0001,
                'ns_w_accumuled_length': 3,
            }
            
            building_grid_generation(init_grid_generator, best_thresholds)

    except Exception as e:
        print(f"Error accessing directory {DATA_RES_PATH}: {e}")

# tbd: also consider the wall width ? (at which stage) when generating the grid lines.
        
# to do
# part 1
# when generating the grids, also register the source components, for both columns and walls.
# what can be the criteria toward the 'global optimal' of grid lines.? -> lead to the adoption of an optimization algorithm.
# should we first do data extraction from the z(vertical) direction or we frist start from horizontal plans.

# part 2
# create parameters, based on the estimated grid lines and the related relationships (with building components), in the design authoring tool, i.e. Autodesk Revit.
# and add something else.

# part 3
# final ending is the user interface...?

# outline detection
# https://www.cgal.org/
# https://stackoverflow.com/questions/2741589/given-a-large-set-of-vertices-in-a-non-convex-polygon-how-can-i-find-the-edges
# https://stackoverflow.com/questions/25585401/travelling-salesman-in-scipy
# https://gis.stackexchange.com/questions/417467/how-to-extract-the-boundaries-of-shapely-multipoint