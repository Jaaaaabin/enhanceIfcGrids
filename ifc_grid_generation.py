import os
from gridSystemGenerator import GridGenerator
from toolsQuickUtils import time_decorator, load_thresholds_from_json

PROJECT_PATH = os.getcwd()
# DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_autocon_test_no_grids'
DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_test'
DATA_RES_PATH = PROJECT_PATH + r'\res'

# final solution related supplement.
DATA_GA_RES_PATH = PROJECT_PATH + r'\res_ga'
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
    generator.get_main_directions_and_storeys(num_directions=4) # static.
    generator.enrich_all_element_locations() # static.

    return generator

@time_decorator
def building_grid_generation(
    basic_generator, new_parameters, set_visualization=True, set_analysis=True):
    
    # update the parameters.
    new_generator = basic_generator.update_parameters(new_parameters)
    
    # generate the grids.
    new_generator.create_grids()

    # merge the grids.
    new_generator.merge_grids()

    # visualization
    if set_visualization:
        new_generator.visualization_2d_before_merge(visual_type='html') # visual_type='svg'
        new_generator.visualization_2d_after_merge(visual_type='html') # visual_type='svg'
    
    if set_analysis:   
        # extract the relationships from merged grids.
        new_generator.analyze_grids()
        # calculate the losses for merged girds.
        new_generator.calculate_losses()  #[for the ga optimization.]

# # ----------
# main.

if __name__ == "__main__":
    
    try:
        model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
        
        for model_path in model_paths:            

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # customized threshold input for grid generation 
            # for each building model
            init_grid_generator = preparation_of_grid_generation(DATA_RES_PATH, model_path)
            best_thresholds = {
                'st_c_num': 2,
                'st_w_num': 2,
                'ns_w_num': 2,
                'st_w_accumuled_length_percent': 0.005,
                'ns_w_accumuled_length_percent': 0.0005,
                'st_st_merge': 0.2,
                'ns_st_merge': 0.4,
                'ns_ns_merge': 0.2,
                'st_c_align_dist': 0.001,     # fixed value,
                'st_w_align_dist': 0.01,       # fixed value, to be decided per project
                'ns_w_align_dist': 0.01,       # fixed value, to be decided per project.
            }
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