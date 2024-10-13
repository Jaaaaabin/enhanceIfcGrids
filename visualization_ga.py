
import os
import copy
import json
from gaTools import ga_decodeInteger_x, ga_adjustReal_x
from gaTools import calculate_pareto_front, meta_visualization_pareto_frontier

from toolsQuickUtils import load_thresholds_from_json, ensure_directory_exists
from ifc_grid_generation import preparation_of_grid_generation, building_grid_generation

def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def get_file_path(base_path, file_template, enable_rr):
    return os.path.join(base_path, file_template.format("True" if enable_rr else "False"))

def square_sum(tup):
    return sum(x**2 for x in tup)

PROJECT_PATH = os.getcwd()
ALL_POSITION_DATA_PATH = os.path.join(PROJECT_PATH, 'res')
ALL_GA_RES_PATH = os.path.join(PROJECT_PATH, 'res_ga')
ALL_GA_SOLUTION_PATH = os.path.join(PROJECT_PATH, 'solution_ga')
SUBDIRS = get_subdirectories(ALL_GA_RES_PATH)

JSON_ALL_RES_RR_TRUE = os.path.join(ALL_GA_RES_PATH, 'all_rr_true.json')
FIG_ALL_RES_RR_TRUE = os.path.join(ALL_GA_RES_PATH, 'all_rr_true.png')
JSON_ALL_RES_RR_FALSE = os.path.join(ALL_GA_RES_PATH, 'all_rr_false.json')
FIG_ALL_RES_RR_FALSE = os.path.join(ALL_GA_RES_PATH, 'all_rr_false.png')

def collect_meta_results():

    for ENABLE_GA_RR in [True, False]:
        
        all_results = dict()

        for nr, subdir in enumerate(SUBDIRS, start=1):

            result_data = dict()

            GA_RES_PATH = os.path.join(ALL_GA_RES_PATH, subdir)

            # Using helper function to avoid redundant path logic
            GENERATION_IND_FIT_FILE = get_file_path(GA_RES_PATH, "ga_inds_fit_rr_{}.txt", ENABLE_GA_RR)
            GENERATION_IND_GEN_FILE = get_file_path(GA_RES_PATH, "ga_inds_gen_rr_{}.txt", ENABLE_GA_RR)
            GENERATION_PARETO_FIG_FILE = get_file_path(GA_RES_PATH, "ga_pareto_fitness_rr_{}.png", ENABLE_GA_RR)

            pareto_front_data, pareto_front_non_data = calculate_pareto_front(
                    gen_file_path=GENERATION_IND_GEN_FILE,
                    fit_file_path=GENERATION_IND_FIT_FILE,
                    pareto_front_fig_output_file=GENERATION_PARETO_FIG_FILE,
            )
            
            sorted_pareto_front_gene = dict(sorted(pareto_front_data.items(), key=lambda item: square_sum(item[0])))
            sorted_pareto_front_non_gene = dict(sorted(pareto_front_non_data.items(), key=lambda item: square_sum(item[0])))

            sorted_pareto_front_data = {key: [
                ga_adjustReal_x(
                    ga_decodeInteger_x(sublist)) for sublist in sublists] for key, sublists in sorted_pareto_front_gene.items()}
            sorted_pareto_front_non_data = {key: [
                ga_adjustReal_x(
                    ga_decodeInteger_x(sublist)) for sublist in sublists] for key, sublists in sorted_pareto_front_non_gene.items()}
            
            sorted_pareto_front_size = {key: len(sublists) for key, sublists in sorted_pareto_front_data.items()}
            sorted_pareto_front_non_size =  {key: len(sublists) for key, sublists in sorted_pareto_front_non_data.items()}

            result_data['result_path'] = GA_RES_PATH

            result_data['near_gene'] = {str(key): value for key, value in sorted_pareto_front_gene.items()}
            result_data['near_data'] = {str(key): value for key, value in sorted_pareto_front_data.items()}
            result_data['near_size'] = {str(key): value for key, value in sorted_pareto_front_size.items()}
            
            result_data['rest_gene'] = {str(key): value for key, value in sorted_pareto_front_non_gene.items()}
            result_data['rest_data'] = {str(key): value for key, value in sorted_pareto_front_non_data.items()}
            result_data['rest_size'] = {str(key): value for key, value in sorted_pareto_front_non_size.items()}

            all_results[str(nr)] = result_data

        json_output_path = JSON_ALL_RES_RR_TRUE if ENABLE_GA_RR else JSON_ALL_RES_RR_FALSE

        with open(json_output_path, 'w') as json_file:
            json.dump(all_results, json_file, indent=4)

def plot_meta_results():

    MARKERS = [
        'o',
        's',
        'D',
        '^',
        'v',
        '<',
        '>',
        'p',
        '*']

    COLORS =[
        "#00c3a7",
        "#aabd79",
        "#eecc16",
        "#f256d9",
        "#c1272d",
        "#0669ff",
        "#9b51ce",
        "#818181"]
    
    meta_visualization_pareto_frontier(JSON_ALL_RES_RR_TRUE, FIG_ALL_RES_RR_TRUE, MARKERS, COLORS)
    meta_visualization_pareto_frontier(JSON_ALL_RES_RR_FALSE, FIG_ALL_RES_RR_FALSE, MARKERS, COLORS)


def process_solutions(
    ga_re_path, ga_solution_path, subsub_dir_name, enable_rr, threshold_file_prefix, grid_generator_init):

    # Prepare paths and load threshold values
    threshold_json_file = os.path.join(ga_re_path, f"{threshold_file_prefix}_rr_{str(enable_rr)}.json")
    values_thresholds = load_thresholds_from_json(threshold_json_file)

    for fit_pair, t_values in values_thresholds.items():

        # Select the first combination in the group
        thresholds_for_generation = t_values[0]

        # Create result directory for each fit_pair
        ga_solution_per_fit_pair_path = os.path.join(ga_solution_path, subsub_dir_name, fit_pair)
        ensure_directory_exists(ga_solution_per_fit_pair_path)

        # Adjust the grid generator and set output path
        grid_generator_active = copy.deepcopy(grid_generator_init)
        grid_generator_active.out_fig_path = ga_solution_per_fit_pair_path

        # Start the grid generation
        building_grid_generation(
            grid_generator_active, thresholds_for_generation, set_visualization=True, set_analysis=False
        )

def produce_all_solutions(
    enable_rr=True, plot_pareto_front=False, plot_non_pareto_front=False):

    for nr, subdir in enumerate(SUBDIRS, start=1):
        
        # Directory preparation
        GA_RES_PATH = os.path.join(ALL_GA_RES_PATH, subdir)
        GA_SOLUTION_PATH = os.path.join(ALL_GA_SOLUTION_PATH, subdir)
        ensure_directory_exists(GA_SOLUTION_PATH)

        # Initialize the grid generator
        grid_generator_init = preparation_of_grid_generation(ALL_POSITION_DATA_PATH, subdir)

        # Process Pareto front solutions if enabled
        if plot_pareto_front:
            process_solutions(
                GA_RES_PATH, GA_SOLUTION_PATH, 'pareto_front', enable_rr, "ga_pareto_inds", grid_generator_init)

        # Process non-Pareto front solutions if enabled
        if plot_non_pareto_front:
            process_solutions(
                GA_RES_PATH, GA_SOLUTION_PATH, 'non_pareto_front', enable_rr, "ga_pareto_non_inds", grid_generator_init)                

if __name__ == "__main__":


    # # collect the GA results from the res_ga.
    # collect_meta_results()
    
    # # visualize near-optimal solutions in a meta visualization mode. (in the res_ga subfolder.)
    # plot_meta_results()

    # # provide two options for solution production as final alternative.
    produce_all_solutions(plot_non_pareto_front=True)



# save.
# def produce_all_solutions(
#     enable_rr=True,
#     plot_pareto_front=False,
#     plot_non_pareto_front=False):
    
#     for nr, subdir in enumerate(SUBDIRS, start=1):
        
#         # directory preparation
#         GA_RES_PATH = os.path.join(ALL_GA_RES_PATH, subdir)
#         GA_SOLUTION_PATH = os.path.join(ALL_GA_SOLUTION_PATH, subdir)
#         if not os.path.exists(GA_SOLUTION_PATH):
#             os.makedirs(GA_SOLUTION_PATH)

#         # initialize the grid generator.
#         grid_generator_init = preparation_of_grid_generation(ALL_POSITION_DATA_PATH, subdir)

#         # ---------------------------------------
#         # For the Pareto front solutions, we collect all of them
#         if plot_pareto_front:

#             subsub_dir_name = 'pareto_front'
#             threshold_json_file = os.path.join(GA_RES_PATH, f"ga_pareto_inds_rr_{str(enable_rr)}.json")
#             values_thresholds = load_thresholds_from_json(threshold_json_file)
#             for fit_pair, t_values in values_thresholds.items():
                
#                 # select the first combination within the same group (based on the fitness values.)
#                 thresholds_for_generation = t_values[0]

#                 # prepare the sub-directory for results.
#                 ga_solution_per_fit_pair_path = os.path.join(GA_SOLUTION_PATH, subsub_dir_name, fit_pair)
#                 if not os.path.exists(ga_solution_per_fit_pair_path):
#                     os.makedirs(ga_solution_per_fit_pair_path)
                
#                 # adjust the target directory of the grid generation
#                 grid_generator_active = copy.deepcopy(grid_generator_init) # save computing time.
#                 grid_generator_active.out_fig_path = ga_solution_per_fit_pair_path
                
#                 # start the grid generation 
#                 building_grid_generation(
#                     grid_generator_active, thresholds_for_generation, set_visualization=True, set_analysis=False)
        
#         # ---------------------------------------
#         # For the non-Pareto-front solutions, we collect all of them
#         if plot_non_pareto_front:

#             subsub_dir_name = 'non_pareto_front'
#             threshold_json_file = os.path.join(GA_RES_PATH, f"ga_pareto_non_inds_rr_{str(enable_rr)}.json")
#             values_thresholds = load_thresholds_from_json(threshold_json_file)
#             for fit_pair, t_values in values_thresholds.items():
                
#                 # select the first combination within the same group (based on the fitness values.)
#                 thresholds_for_generation = t_values[0]

#                 # prepare the sub-directory for results.
#                 ga_solution_per_fit_pair_path = os.path.join(GA_SOLUTION_PATH, subsub_dir_name, fit_pair)
#                 if not os.path.exists(ga_solution_per_fit_pair_path):
#                     os.makedirs(ga_solution_per_fit_pair_path)
                
#                 # adjust the target directory of the grid generation
#                 grid_generator_active = copy.deepcopy(grid_generator_init) # save computing time.
#                 grid_generator_active.out_fig_path = ga_solution_per_fit_pair_path
                
#                 # start the grid generation 
#                 building_grid_generation(
#                     grid_generator_active, thresholds_for_generation, set_visualization=True, set_analysis=False)