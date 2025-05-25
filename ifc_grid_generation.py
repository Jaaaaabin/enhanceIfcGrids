import os
from EnhancingComponent.gridSystemGenerator import GridGenerator
from toolsQuickUtils import time_decorator, load_thresholds_from_json

PROJECT_PATH = os.getcwd()
DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_autocon_test_no_grids'
# DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_test'
DATA_RES_PATH = PROJECT_PATH + r'\res_extraction'

# final solution related supplement.
DATA_GA_RES_PATH = PROJECT_PATH + r'\res_ga_new'
FIT_PARETO_VALUE_FLOAT = [] # default.

@time_decorator
def preparation_of_grid_generation(
    work_path,
    ifc_model,
    info_floors='info_floors.json',
    info_st_columns='info_columns.json',
    info_st_walls='info_st_walls.json',
    info_ns_walls='info_ns_walls.json',
    info_ct_walls='info_ct_walls.json'):

    # initialization.
    generator = GridGenerator(
        os.path.join(work_path, ifc_model),
        os.path.join(work_path, ifc_model, info_floors),
        os.path.join(work_path, ifc_model, info_st_columns),
        os.path.join(work_path, ifc_model, info_st_walls),
        os.path.join(work_path, ifc_model, info_ns_walls),
        os.path.join(work_path, ifc_model, info_ct_walls),
        )
     
    # preparation.
    generator.get_main_directions_and_storeys(num_directions=4) # static.
    generator.enrich_all_element_locations() # static.

    return generator

@time_decorator
def building_grid_generation(
    basic_generator, new_parameters, set_visualization=True, set_analysis=False, set_additional_indicator=False):
    
    # update the parameters.
    new_generator = basic_generator.update_parameters(new_parameters)
    
    # generate the grids.
    new_generator.create_grids()

    # merge the grids.
    new_generator.merge_grids()

    # visualization of the grids.
    if set_visualization:
        new_generator.visualization_2d_before_merge(visual_type='html') # visual_type='html' or 'pdf'
        new_generator.visualization_2d_after_merge(visual_type='html') # visual_type='html' or 'pdf'
    
    if set_analysis:

        # [------for the ga optimization------]
        # extract the relationships from merged grids.
        new_generator.analyze_grids()
        # calculate the losses for merged girds.
        new_generator.calculate_losses()  
        # [------for the ga optimization------]
        
    if set_additional_indicator:

        new_generator.calculate_grid_indicator()
        return new_generator.additional_indicator


# # ----------
# main.

if __name__ == "__main__":
    
    try:
        model_paths = [filename for filename in os.listdir(DATA_RES_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
        
        for model_path in model_paths:

            if model_path.startswith("4-"):
                
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # customized threshold input for grid generation 
                # for each building model
                init_grid_generator = preparation_of_grid_generation(DATA_RES_PATH, model_path)
                best_thresholds = { 
        "st_c_num": 2,
        "st_w_num": 7,
        "ns_w_num": 2,
        "st_w_accumuled_length_percent": 0.0014,
        "ns_w_accumuled_length_percent": 0.0049,
        "st_st_merge": 0.5,
        "ns_st_merge": 0.5,
        "ns_ns_merge": 0.2,
        "st_w_align_dist": 0.0742,
        "ns_w_align_dist": 0.0904
                }

                # best_thresholds = {
                #     'st_c_num': 2,          # SCC
                #     'st_w_num': 2,          # SWC 
                #     'ns_w_num': 2,          # NWC
                #     'st_w_accumuled_length_percent': 0.005,         # SWL
                #     'ns_w_accumuled_length_percent': 0.0005,        # NWL
                #     'st_st_merge': 0.2,                             # SSM
                #     'ns_st_merge': 0.4,                             # NSM
                #     'ns_ns_merge': 0.2,                             # NNM
                #     'st_c_align_dist': 0.001,                       # SAC fixed value,
                #     'st_w_align_dist': 0.01,                        # SWA fixed value, to be decided per project
                #     'ns_w_align_dist': 0.01,                        # NWA fixed value, to be decided per project.
                # }
                building_grid_generation(init_grid_generator, best_thresholds)
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    except Exception as e:
        print(f"Error accessing directory {DATA_RES_PATH}: {e}")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Maybe we will not need them anymore....
# autcon saving.
# the thresholds used for intemediate results visualization
# best_thresholds = {
#     'st_c_num': 6,
#     'st_w_num': 2,
#     'ns_w_num': 3,
#     'st_w_accumuled_length_percent': 0.005,
#     'ns_w_accumuled_length_percent': 0.0005,
#     'st_st_merge': 0.5,
#     'ns_st_merge': 0.5,
#     'ns_ns_merge': 0.5,
#     'st_c_align_dist': 0.001,     # fixed value,
#     'st_w_align_dist': 0.1,       # fixed value, to be decided per project
#     'ns_w_align_dist': 0.1,       # fixed value, to be decided per project.
# }
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -