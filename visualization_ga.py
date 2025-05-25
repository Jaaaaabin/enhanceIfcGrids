import os
import copy
import json
from toolsGA import ga_decodeInteger_x, ga_adjustReal_x
from toolsGA import calculate_pareto_front, meta_visualization_pareto_frontier

from toolsQuickUtils import load_thresholds_from_json, ensure_directory_exists
from ifc_grid_generation import preparation_of_grid_generation, building_grid_generation
from paretoAnalysis import calculate_diversity_metrics, plot_multiple_cases

def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def get_file_path(base_path, file_template, enable_rr):
    return os.path.join(base_path, file_template.format("True" if enable_rr else "False"))

def square_sum(tup):
    return sum(x**2 for x in tup)

PROJECT_PATH = os.getcwd()
ALL_POSITION_DATA_PATH = os.path.join(PROJECT_PATH, 'res_extraction')
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

# - - - - - - - 
# process the solution.
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
# - - - - - - - 

# - - - - - - - 
# calculate the additional indicator.
def single_indicator_calculation(ga_re_path, ga_solution_path, subsub_dir_name, enable_rr, threshold_file_prefix, grid_generator_init):

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
        additional_indicator = building_grid_generation(
            grid_generator_active, thresholds_for_generation,
            set_visualization=False,
            set_analysis=False,
            set_additional_indicator=True,
        )
        
        with open(os.path.join(ga_solution_per_fit_pair_path,'indicator.json'), 'w') as fp:
            json.dump(additional_indicator, fp)

def summarize_indicator(main_folder):

    # Define the output file
    output_file = os.path.join(main_folder, 'indicators.json')
    combined_data = {}

    # Traverse through the directory structure
    for subfolder in ["pareto_front", "non_pareto_front"]:
        subfolder_path = os.path.join(main_folder, subfolder)
        
        if not os.path.exists(subfolder_path):
            continue
        
        for subsubfolder in os.listdir(subfolder_path):
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            
            if not os.path.isdir(subsubfolder_path):
                continue
            
            # Check if indicator.json exists in this subsubfolder
            json_file_path = os.path.join(subsubfolder_path, 'indicator.json')
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r') as json_file:
                        data = json.load(json_file)
                        # Use the subsubfolder as the key directly
                        combined_data[subsubfolder] = data
                except Exception as e:
                    print(f"Error reading {json_file_path}: {e}")
    
    # Write the combined data to the output file
    try:
        with open(output_file, 'w') as output_json:
            json.dump(combined_data, output_json, indent=4)
        print(f"Combined data written to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

def calculate_all_indicators(
    enable_rr=True, plot_pareto_front=True, plot_non_pareto_front=True):

    for nr, subdir in enumerate(SUBDIRS, start=1):
        
        # Directory preparation
        GA_RES_PATH = os.path.join(ALL_GA_RES_PATH, subdir)
        GA_SOLUTION_PATH = os.path.join(ALL_GA_SOLUTION_PATH, subdir)
        ensure_directory_exists(GA_SOLUTION_PATH)

        # Initialize the grid generator
        grid_generator_init = preparation_of_grid_generation(ALL_POSITION_DATA_PATH, subdir)

        # Process Pareto front solutions if enabled
        if plot_pareto_front:
            single_indicator_calculation(
                GA_RES_PATH, GA_SOLUTION_PATH, 'pareto_front', enable_rr, "ga_pareto_inds", grid_generator_init)

        # # Process non-Pareto front solutions if enabled
        if plot_non_pareto_front:
            single_indicator_calculation(
                GA_RES_PATH, GA_SOLUTION_PATH, 'non_pareto_front', enable_rr, "ga_pareto_non_inds", grid_generator_init)

        if plot_pareto_front and plot_non_pareto_front:
            
            summarize_indicator(GA_SOLUTION_PATH)

# -------------------- new -------------------- 
def calculate_metrics_and_additional_indicators():

    all_json_data = []
    all_number_pareto_front = [8, 11, 2, 5, 5, 3, 5, 4] # automated count from the GA results.
    all_number_storey_adjustment = [0, -1, 0, -1, 0, 0, 0, 0] # modification because some of the "mezzanie"/"roof".

    all_visualization_fitness = [
        [(0.39, 0.103), (0.485, 0.053), (0.715, 0.573)],
        [(0.116, 0.303), (0.232, 0.231), (0.453, 0.36)],
        [(0.009, 0.138), (0.124, 0.112), (0.493, 0.224)],
        [(0.171, 0.366),(0.236, 0.249), (0.531, 0.648)],
        [(0.138, 0.02), (0.163, 0.019), (0.284, 0.334)],
        [(0.586, 0.136), (0.646, 0.03), (0.684, 0.41)],
        [(0.299, 0.197), (0.563, 0.025), (0.322, 0.15)],
        [(0.05, 0.227), (0.488, 0.19), (0.268, 0.336)],]
    
    # by manual selection.
    for nr in range(1, 9):
        json_file = os.path.join(ALL_GA_SOLUTION_PATH, f'indicators_{nr}.json')
        with open(json_file, 'r') as file:
            json_data = json.load(file)
        all_json_data.append(json_data)

    # # Calculate diversity metrics
    # for nr in range(len(all_json_data)):
    #     calculate_diversity_metrics(nr, all_json_data, all_number_pareto_front, all_number_storey_adjustment)

    # Plot multiple cases
    plot_multiple_cases(
        ALL_GA_SOLUTION_PATH, all_json_data, all_number_pareto_front, all_number_storey_adjustment, all_visualization_fitness)

# - - - - - - - 
#     
if __name__ == "__main__":


    # # collect the GA results from the res_ga.
    # collect_meta_results()
    
    # # visualize near-optimal solutions in a meta visualization mode. (in the res_ga subfolder.)
    # plot_meta_results()

    # # provide two options for solution production as final alternative.
    # produce_all_solutions(plot_non_pareto_front=True)
    
    # 
    # calculate_all_indicators()
    # 
    calculate_metrics_and_additional_indicators()
