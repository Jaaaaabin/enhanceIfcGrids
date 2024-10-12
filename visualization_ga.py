
import os
import json
from gaTools import calculate_pareto_front
from ifc_ga_optimization import  ga_decodeInteger_x, ga_adjustReal_x

def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories

PROJECT_PATH = os.getcwd()
ALL_MODEL_GA_RES_PATH = os.path.join(PROJECT_PATH, 'res_ga')
subdirs = get_subdirectories(ALL_MODEL_GA_RES_PATH)

ENABLE_GA_RR=True

for nr in range(1,9,1):
    
    subdir = subdirs[nr-1]
    nr_subir = str(nr)
    MODEL_GA_RES_PATH = os.path.join(ALL_MODEL_GA_RES_PATH,subdir)
    GENERATION_IND_FIT_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_inds_fit_rr_True.txt") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_inds_fit_rr_False.txt") 
    GENERATION_IND_GEN_FILE =  os.path.join(MODEL_GA_RES_PATH, "ga_inds_gen_rr_True.txt") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_inds_gen_rr_False.txt")
    GENERATION_PARETO_IND_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_inds_rr_True.json") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_inds_rr_False.json")
    GENERATION_NON_PARETO_IND_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_non_inds_rr_True.json") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_non_inds_rr_False.json")
    GENERATION_PARETO_FIG_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_fitness_rr_True.png") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_fitness_rr_False.png")

    break 
print("s")

# INI_GENERATION_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_ini_inds_integer.txt")
# GENERATION_LOG_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_log" + PLOT_KEYS + ".json")
# GENERATION_FIT_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_fitness" + PLOT_KEYS + ".png")
# GENERATION_IND_FIT_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_inds_fit_rr_True.txt") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_inds_fit_rr_False.txt") 
# GENERATION_IND_GEN_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_inds_gen_rr_True.txt") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_inds_gen_rr_False.txt")
# GENERATION_PARETO_IND_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_inds_rr_True.json") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_inds_rr_False.json")
# GENERATION_NON_PARETO_IND_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_non_inds_rr_True.json") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_non_inds_rr_False.json")
# GENERATION_PARETO_FIG_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_fitness_rr_True.png") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_fitness_rr_False.png")


# GENERATION_IND_GEN_FILE = 0
# GENERATION_IND_FIT_FILE = 0
# GENERATION_PARETO_FIG_FILE = 0
# GENERATION_PARETO_IND_FILE = 0
# GENERATION_NON_PARETO_IND_FILE = 0

pareto_front_data, non_pareto_front_data =  calculate_pareto_front(
        gen_file_path = GENERATION_IND_GEN_FILE,
        fit_file_path = GENERATION_IND_FIT_FILE,
        pareto_front_fig_output_file = GENERATION_PARETO_FIG_FILE,
        )
    
# sort the keys
def square_sum(tup):
    return sum(x**2 for x in tup)

sorted_pareto_front_data = dict(sorted(pareto_front_data.items(), key=lambda item: square_sum(item[0])))
sorted_non_pareto_front_data = dict(sorted(non_pareto_front_data.items(), key=lambda item: square_sum(item[0])))

# Pareto
for key, sublists in sorted_pareto_front_data.items():
    sorted_pareto_front_data[key] = [
        ga_adjustReal_x(ga_decodeInteger_x(sublist)) for sublist in sublists]
# with open(GENERATION_PARETO_IND_FILE, 'w') as json_file:
#     json.dump({str(key): value for key, value in sorted_pareto_front_data.items()}, json_file, indent=4)

# Non-pareto.
for key, sublists in sorted_non_pareto_front_data.items():
    sorted_non_pareto_front_data[key] = [
        ga_adjustReal_x(ga_decodeInteger_x(sublist)) for sublist in sublists]
# with open(GENERATION_NON_PARETO_IND_FILE, 'w') as json_file:
#     json.dump({str(key): value for key, value in sorted_non_pareto_front_data.items()}, json_file, indent=4)