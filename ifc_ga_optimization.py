import sys
import ctypes

# Enable Virtual Terminal Processing to display ANSI colors in Windows console before numpy
if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

import os
import copy
import array
import json
import argparse
import random
import logging
import multiprocessing
import numpy as np
from deap import base, creator, tools, algorithms

from ifc_grid_generation import preparation_of_grid_generation

from gaTools import getIfcModelPaths, getParameterScales, getParameterVarLimits
from gaTools import createInds, ga_loadInds, saveLogbook, visualizeGenFitness, visualizeGenFitness_multiobjectives
from gaTools import ga_rr_eaSimple, calculate_pareto_front

#===================================================================================================
# Genetic Algorithm Configuration - Constants

ENABLE_GA_RR = True

POPULATION_SIZE = 20 # population size or no of individuals or solutions being considered in each generation.
NUM_GENERATIONS = 300 #00 # number of iterations.

TOURNAMENT_SIZE = 3 # number of participants in tournament selection.
CROSS_PROB = 0.6 # the probability with which two individuals are crossed or mated
MUTAT_PROB = 0.1 # the probability for mutating an individual

NUM_GENERATIONS_THRESHOLD_RESTART = 50
RANDOM_RESTART_POPULATION_SIZE = int(POPULATION_SIZE*0.8)
NUM_GENERATIONS_CONVERGE = 10000 # only use 'maximum number of generation as the convergence condition/

PLOT_KEYS = "_rr_"  + str(ENABLE_GA_RR) + "_pop_size_" + str(POPULATION_SIZE) + "_cross_" + str(CROSS_PROB) + "_mutate_" + str(MUTAT_PROB)
#===================================================================================================
# Paths setup and Log registration.

PROJECT_PATH = os.getcwd()
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')

DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_autocon_ga')
MODEL_NAME = getIfcModelPaths(folder_path=DATA_FOLDER_PATH, only_first=True)

MODEL_GA_RES_PATH = os.path.join(PROJECT_PATH, 'res_ga', MODEL_NAME)
os.makedirs(MODEL_GA_RES_PATH, exist_ok=True)

gridGeneratorInit = preparation_of_grid_generation(DATA_RES_PATH, MODEL_NAME)
logging.basicConfig(filename='ga.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s') # reconfigurate the logging file.

INI_GENERATION_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_ini_inds_integer.txt")
GENERATION_LOG_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_log" + PLOT_KEYS + ".json")
GENERATION_FIT_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_fitness" + PLOT_KEYS + ".png")
GENERATION_IND_FIT_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_inds_fit_rr_True.txt") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_inds_fit_rr_False.txt") 
GENERATION_IND_GEN_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_inds_gen_rr_True.txt") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_inds_gen_rr_False.txt")
GENERATION_PARETO_IND_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_inds_rr_True.json") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_inds_rr_False.json")
GENERATION_NON_PARETO_IND_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_non_inds_rr_True.json") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_non_inds_rr_False.json")
GENERATION_PARETO_FIG_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_pareto_fitness_rr_True.png") if ENABLE_GA_RR else os.path.join(MODEL_GA_RES_PATH, "ga_pareto_fitness_rr_False.png")

GENERATION_BEST_IND_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_log_best_inds" + PLOT_KEYS + ".json")

# todo.
# Take the initial ranges from the extracted ifc information.
# PARAMS = {
#     'st_c_num': (2, 10), # [3,10)  # min = 3
#     'st_w_num': (2, 10), # [3,10)  # min = num_main_floors.
#     'ns_w_num': (2, 10), # [2,10)  # min = 2.
#     'st_w_accumuled_length_percent': (0.0001, 0.05), # should be more "dependent" on the average length.
#     'ns_w_accumuled_length_percent': (0.0001, 0.05), # should be more "dependent" on the average length.
#     'st_st_merge': (0.1, 1.00), # ....god sick differentiate between merge
#     'ns_st_merge': (0.1, 1.00), # ....god sick differentiate between merge
#     'ns_ns_merge': (0.1, 1.00), # ....god sick differentiate between merge
#     # 'st_c_align_dist': (0.0001, 0.1), # fixed : 0.001
#     'st_w_align_dist': (0.0001, 0.1), # fixed?
#     'ns_w_align_dist': (0.0001, 0.1), # fixed?
# }
#===================================================================================================
# Basic parameter & Customized Population setup:
PARAMS = {
    'st_c_num': (2, 10), # [3,10)  # min = 3
    'st_w_num': (2, 10), # [3,10)  # min = num_main_floors.
    'ns_w_num': (2, 10), # [2,10)  # min = 2.
    'st_w_accumuled_length_percent': (0.0001, 0.05), # should be more "dependent" on the average length.
    'ns_w_accumuled_length_percent': (0.0001, 0.05), # should be more "dependent" on the average length.
    'st_st_merge': (0.1, 1.00), # ....god sick differentiate between merge
    'ns_st_merge': (0.1, 1.00), # ....god sick differentiate between merge
    'ns_ns_merge': (0.1, 1.00), # ....god sick differentiate between merge
    # 'st_c_align_dist': (0.0001, 0.1), # fixed : 0.001
    'st_w_align_dist': (0.0001, 0.1), # fixed?
    'ns_w_align_dist': (0.0001, 0.1), # fixed?
}

# Get the additional Constant Values.
PARAMS_SCALE, PARAMS_INTEGER = getParameterScales(param_ranges=PARAMS)
MinVals, MaxVals = getParameterVarLimits(param_ranges=PARAMS_INTEGER)

# ===================================================================================================
# Basic Functions of GA

def ga_decodeInteger_x(individual: list) -> list:

    decoded_xs = []
    count_parameter_placeholders = []
    for key, (lower, upper) in PARAMS_INTEGER.items():
        upper -= 1
        l = len(str(abs(upper)))
        count_parameter_placeholders.append(l)

    if sum(count_parameter_placeholders) == len(individual):
        
        start_i = 0
        for c in count_parameter_placeholders:
            decoded_x = int(''.join(str(digit) for digit in individual[int(start_i):int(start_i+c)]))
            decoded_xs.append(decoded_x)
            start_i += c
    else:
        raise ValueError("The integer deocder is wrong")
    return decoded_xs

def ga_adjustReal_x(decoded_x):

    decoded_parameters = dict(zip(PARAMS.keys(), decoded_x))
    
    for key, value in list(decoded_parameters.items()):
        
        # rescale back to the real values.
        if PARAMS_SCALE[key] > 1:
            decoded_parameters[key] /= PARAMS_SCALE[key]

        # correct the integer values.
        if '_num' in key:
            decoded_parameters[key] = int(value)

    return decoded_parameters

# Objective Functions of genetic algorithm (GA)
# @time_decorator
def ga_objective(individual: list) -> tuple:

    # decode the integer values back to real parameter values for calculating the objective function.
    decoded_individual = ga_decodeInteger_x(list(individual))
    decoded_parameters = ga_adjustReal_x(decoded_individual)

    # build the gridGenerator for the current individual, create grids and calculate the losses.
    gridGenerator = copy.deepcopy(gridGeneratorInit) # save computing time.
    gridGenerator.update_parameters(decoded_parameters)
    gridGenerator.create_grids()
    gridGenerator.merge_grids()
    gridGenerator.analyze_grids()
    gridGenerator.calculate_losses()
    
    # for our problem, it might be a "dominated" problem.
    # individual_fitness = gridGenerator.percent_unbound_elements*0.5 + gridGenerator.avg_deviation_maxmin*0.25 + gridGenerator.avg_deviation_adjacent*0.25
    # return (individual_fitness,) # the return value must be a list / tuple, even it's only one fitness value.
    f_unbound = gridGenerator.percent_unbound_elements
    f_distribution = 0.5*gridGenerator.avg_deviation_maxmin + 0.5*gridGenerator.avg_deviation_adjacent
    return (f_unbound, f_distribution)

# ===================================================================================================
# main.
def main():

    parser = argparse.ArgumentParser(description="Run genetic algorithm with a specified variable value.")
    parser.add_argument('--random_seed', type=int, default=2021, help='Random seed for creatomg initial individuals.')
    parser.add_argument('--num_process', type=int, default=max(1, int(multiprocessing.cpu_count()*0.5)), help='Number of processes for multi processing.')
    parser.add_argument('--set_plot', type=bool, default=True, help='plot the the generated grids')
    parser.add_argument('--set_rr', type=bool, default=ENABLE_GA_RR, help='enable the random restart')

    args = parser.parse_args()
    print("------------ number of processes employed :", args.num_process, "------------")
    print("------------ enable the random restart :", args.set_rr, "------------")

    if args.random_seed:
        random.seed(args.random_seed)

    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMin) 
    toolbox = base.Toolbox()

    createInds(n=POPULATION_SIZE, param_rangs=PARAMS_INTEGER, filename=INI_GENERATION_FILE)
    toolbox.register("population", ga_loadInds, creator.Individual, n=POPULATION_SIZE, filename=INI_GENERATION_FILE)

    # Evaluator initializer
    toolbox.register("evaluate", ga_objective)
    # <To check if there's other more sustanible Penalty methods>, for example: toolbox.decorate("evaluate", tools.DeltaPenalty(feasible_fxn, 1.0))
    
    # Registerbasic processes using DEAP bulit-in functions
    toolbox.register("mate", tools.cxUniform, indpb=CROSS_PROB)
    toolbox.register("mutate", tools.mutUniformInt, low=MinVals, up=MaxVals, indpb=MUTAT_PROB)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Clear the old generation individual file.
    if os.path.exists(GENERATION_IND_FIT_FILE):
        os.remove(GENERATION_IND_FIT_FILE)
        
    # Process Pool of multi workers
    if args.num_process > 1:
        pool = multiprocessing.Pool(processes=args.num_process)
        toolbox.register("map", pool.map)
    
    # History to track all the individuals produced in the evolution.
    history = tools.History()
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)
    # # toolbox.decorate("select", history.decorator)

    pop = toolbox.population()
    history.update(pop)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # hof = tools.HallOfFame(1)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    final_pop, logbook, restart_rounds = ga_rr_eaSimple(
        pop, creator, toolbox, set_random_restart=args.set_rr,
        cxpb=CROSS_PROB, mutpb=MUTAT_PROB, ngen=NUM_GENERATIONS,
        initial_generation_file=INI_GENERATION_FILE, fitness_file=GENERATION_IND_FIT_FILE, threshold_file=GENERATION_IND_GEN_FILE,
        stats=stats, verbose=True,
        param_limits = PARAMS_INTEGER, ngen_threshold_restart=NUM_GENERATIONS_THRESHOLD_RESTART,
        pop_restart=RANDOM_RESTART_POPULATION_SIZE, ngen_converge=NUM_GENERATIONS_CONVERGE)
    
    if args.num_process > 1:
        pool.close()

    # Analysis of the GA results.
    saveLogbook(
        logbook=logbook, log_file=GENERATION_LOG_FILE)

    # - - - - - - - - previous version of fitness visualization
    # visualizeGenFitness(
    #     output_file=GENERATION_FIT_FILE, logbook=logbook, restart_rounds=restart_rounds, ind_file=GENERATION_IND_FIT_FILE, generation_size=POPULATION_SIZE)
    # save_genealogy(toolbox, history, genealogy_file=GENERATION_GENEALOGY_FILE) # genealogy for plotting crossover and mutation.
    # - - - - - - - - previous version of fitness visualization
    
    # - - - - - - - - previous version of the best individual selection
    # best_ind = tools.selBest(final_pop, 1)[0] # Pick the best individual
    # best_ind_decoded = ga_decodeInteger_x(best_ind)
    # decoded_parameters = ga_adjustReal_x(best_ind_decoded)
    # gridGeneratorInit.update_parameters(decoded_parameters) # Call back the grid generator.
    # gridGeneratorInit.create_grids()
    # gridGeneratorInit.merge_grids()
    # with open(GENERATION_BEST_IND_FILE, 'w') as json_file: # Visualization of the generated grids.
    #     json.dump(decoded_parameters, json_file, indent=4)
    # if args.set_plot:
    #     gridGeneratorInit.visualization_2d_before_merge(visual_type='html', visualization_storage_path=MODEL_GA_RES_PATH, add_strs='ga')
    #     gridGeneratorInit.visualization_2d_after_merge(visual_type='html', visualization_storage_path=MODEL_GA_RES_PATH, add_strs='ga')
    # - - - - - - - - previous version of the best individual selection

    visualizeGenFitness_multiobjectives(
        output_file=GENERATION_FIT_FILE, logbook=logbook, restart_rounds=restart_rounds, ind_file=GENERATION_IND_FIT_FILE, generation_size=POPULATION_SIZE)
     
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
    with open(GENERATION_PARETO_IND_FILE, 'w') as json_file:
        json.dump({str(key): value for key, value in sorted_pareto_front_data.items()}, json_file, indent=4)
    
    # Non-pareto.
    for key, sublists in sorted_non_pareto_front_data.items():
        sorted_non_pareto_front_data[key] = [
            ga_adjustReal_x(ga_decodeInteger_x(sublist)) for sublist in sublists]
    with open(GENERATION_NON_PARETO_IND_FILE, 'w') as json_file:
        json.dump({str(key): value for key, value in sorted_non_pareto_front_data.items()}, json_file, indent=4)
    
if __name__ == "__main__":

    main()