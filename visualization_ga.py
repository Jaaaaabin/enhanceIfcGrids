
import os
import json
from gaTools import ga_decodeInteger_x, ga_adjustReal_x
from gaTools import calculate_pareto_front, meta_visualization_pareto_frontier

def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def get_file_path(base_path, file_template, enable_rr):
    return os.path.join(base_path, file_template.format("True" if enable_rr else "False"))

def square_sum(tup):
    return sum(x**2 for x in tup)

PROJECT_PATH = os.getcwd()
ALL_MODEL_GA_RES_PATH = os.path.join(PROJECT_PATH, 'res_ga')
subdirs = get_subdirectories(ALL_MODEL_GA_RES_PATH)

JSON_ALL_RES_RR_TRUE = os.path.join(ALL_MODEL_GA_RES_PATH, 'all_rr_true.json')
FIG_ALL_RES_RR_TRUE = os.path.join(ALL_MODEL_GA_RES_PATH, 'all_rr_true.png')
JSON_ALL_RES_RR_FALSE = os.path.join(ALL_MODEL_GA_RES_PATH, 'all_rr_false.json')
FIG_ALL_RES_RR_FALSE = os.path.join(ALL_MODEL_GA_RES_PATH, 'all_rr_false.png')

def collect_meta_results():

    for ENABLE_GA_RR in [True, False]:
        
        all_results = dict()

        for nr, subdir in enumerate(subdirs, start=1):

            result_data = dict()

            MODEL_GA_RES_PATH = os.path.join(ALL_MODEL_GA_RES_PATH, subdir)

            # Using helper function to avoid redundant path logic
            GENERATION_IND_FIT_FILE = get_file_path(MODEL_GA_RES_PATH, "ga_inds_fit_rr_{}.txt", ENABLE_GA_RR)
            GENERATION_IND_GEN_FILE = get_file_path(MODEL_GA_RES_PATH, "ga_inds_gen_rr_{}.txt", ENABLE_GA_RR)
            GENERATION_PARETO_FIG_FILE = get_file_path(MODEL_GA_RES_PATH, "ga_pareto_fitness_rr_{}.png", ENABLE_GA_RR)

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

            result_data['result_path'] = MODEL_GA_RES_PATH

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

    MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']

    COLORS =["#00c3a7",
        "#aabd79",
        "#eecc16",
        "#f256d9",
        "#c1272d",
        "#0669ff",
        "#9b51ce",
        "#818181"]
    
    meta_visualization_pareto_frontier(JSON_ALL_RES_RR_TRUE, FIG_ALL_RES_RR_TRUE, MARKERS, COLORS)
    meta_visualization_pareto_frontier(JSON_ALL_RES_RR_FALSE, FIG_ALL_RES_RR_FALSE, MARKERS, COLORS)

if __name__ == "__main__":

    # collect_meta_results()
    plot_meta_results()