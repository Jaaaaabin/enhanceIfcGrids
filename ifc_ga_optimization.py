import sys
import ctypes

# Enable Virtual Terminal Processing to display ANSI colors in Windows console before numpy
if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

import os
import copy
import array
import argparse
import random
import logging
import multiprocessing
import numpy as np
from deap import base, creator, tools, algorithms

from ifc_grid_generation import preparation_of_grid_generation

from gaTools import getIfcModelPaths, getParameterScales, getParameterVarLimits
from gaTools import createInds, ga_loadInds, saveLogbook, visualizeGenFitness, visualizeGenFitnessViolin
from gaTools import ga_eaSimple, ga_rr_eaSimple

#===================================================================================================
# Genetic Algorithm Configuration - Constants
POPULATION_SIZE = 40 # population size or no of individuals or solutions being considered in each generation.
NUM_GENERATIONS = 50 # number of iterations.

TOURNAMENT_SIZE = 3 # number of participants in tournament selection.
CROSS_PROB = 0.5 # the probability with which two individuals are crossed or mated
MUTAT_PROB = 0.3 # the probability for mutating an individual

NUM_GENERATIONS_NO_IMPROVEMENT = 5
NUM_GENERATIONS_CONVERGE = 10
STD_CONVERGE = 0.02

NUM_PROCESS = 8
RANDOM_SEED = 20001

#===================================================================================================
# Paths setup and Log registration.

PROJECT_PATH = os.getcwd()
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')

DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_autocon_test')
MODEL_NAME = getIfcModelPaths(folder_path=DATA_FOLDER_PATH, only_first=True)

MODEL_GA_RES_PATH = os.path.join(PROJECT_PATH, 'res_ga', MODEL_NAME)
os.makedirs(MODEL_GA_RES_PATH, exist_ok=True)

gridGeneratorInit = preparation_of_grid_generation(DATA_RES_PATH, MODEL_NAME)
logging.basicConfig(filename='ga.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s') # reconfigurate the logging file.

INI_GENERATION_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_ini_inds_integer.txt")
GENERATION_LOG_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_log.json")
GENERATION_FIT_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_fitness.png")
GENERATION_IND_FILE = os.path.join(MODEL_GA_RES_PATH, "ga_inds.txt")
GENERATION_IND_VIOLIN_FLE = os.path.join(MODEL_GA_RES_PATH, "ga_ind_violin.png")

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
    individual_fitness = gridGenerator.percent_unbound_elements*0.5 + gridGenerator.avg_deviation_maxmin*0.25 + gridGenerator.avg_deviation_adjacent*0.25

    return (individual_fitness,) # the return value must be a list / tuple, even it's only one fitness value.

# ===================================================================================================
# main.
def main():
    
    parser = argparse.ArgumentParser(description="Run genetic algorithm with a specified variable value.")
    parser.add_argument('--random_seed', type=int, default=20001, help='Random seed for creatomg initial individuals.')
    parser.add_argument('--num_process', type=int, default=12, help='Number of processes for multi processing.')
    parser.add_argument('--set_plot', type=bool, default=False, help='plot the the generated grids')
    args = parser.parse_args()
    
    if args.random_seed:
        random.seed(args.random_seed)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMin) 
    toolbox = base.Toolbox()

    createInds(n=POPULATION_SIZE, param_rangs=PARAMS_INTEGER, filename=INI_GENERATION_FILE)
    toolbox.register("population", ga_loadInds, creator.Individual, n=POPULATION_SIZE, filename=INI_GENERATION_FILE)

    # Evaluator initializer
    toolbox.register("evaluate", ga_objective)
    # <To check if there's other more sustanible Penalty methods>, for example: toolbox.decorate("evaluate", tools.DeltaPenalty(feasible_fxn, 1.0))
    
    # registering basic processes using DEAP bulit-in functions
    toolbox.register("mate", tools.cxUniform, indpb=CROSS_PROB)
    toolbox.register("mutate", tools.mutUniformInt, low=MinVals, up=MaxVals, indpb=MUTAT_PROB)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # toolbox.register("select", tools.selNSGA2, k=int(POPULATION_SIZE/5), nd='log')

    # NSGA-II: Non-dominated Sorting Genetic Algorithm.
    # https://github.com/DEAP/deap/issues/505
    # https://medium.com/@rossleecooloh/optimization-algorithm-nsga-ii-and-python-package-deap-fca0be6b2ffc
    # A has higher height than B but lower salary. In the contrary, B confronts the same situation. We call the situation “non-dominated”.
    # If we can find a set of solutions that they don’t dominate each other and not dominated by any other solutions, we call them “Pareto-optimal” solutions.
    # In each iteration, we combine the parent and the offspring after GA operations.
    # Through the Non-dominated Sorting, we classify all individuals to different Pareto-optimal front level
    # We then select individuals as the next population from Pareto-optimal front in the order of different levels.
    # As for "diversity" preservation, the “Crowding Distance” is also computed.
    # The crowded distance comparison guides the selection process at the various stages of the algorithm toward a uniformly "spread-out" Pareto-optimal front.
    
    # Clear the old generation individual file.
    if os.path.exists(GENERATION_IND_FILE):
        os.remove(GENERATION_IND_FILE)
        
    # Process Pool of multi workers
    if args.num_process > 1:
        pool = multiprocessing.Pool(processes=args.num_process)
        toolbox.register("map", pool.map)
    
    # history to track all the individuals produced in the evolution.
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
    
    # final_pop, logbook = ga_eaSimple(pop, toolbox, cxpb=CROSS_PROB, mutpb=MUTAT_PROB, 
    #     ngen=NUM_GENERATIONS, fitness_file=GENERATION_IND_FILE, stats=stats, verbose=True)
    final_pop, logbook = ga_rr_eaSimple(
        pop, creator, toolbox, 
        cxpb=CROSS_PROB, mutpb=MUTAT_PROB, ngen=NUM_GENERATIONS,
        initial_generation_file = INI_GENERATION_FILE,
        fitness_file=GENERATION_IND_FILE,
        stats=stats, verbose=True,
        param_limits = PARAMS_INTEGER, ngen_no_improve=NUM_GENERATIONS_NO_IMPROVEMENT,
        ngen_converge=NUM_GENERATIONS_CONVERGE, std_converge=STD_CONVERGE)
    
    if args.num_process > 1:
        pool.close()
    
    # Analysis of the GA results.
    saveLogbook(logbook=logbook, log_file=GENERATION_LOG_FILE)
    visualizeGenFitness(logbook=logbook, fitness_file=GENERATION_FIT_FILE, show_max=False)
    visualizeGenFitnessViolin(violin_file=GENERATION_IND_VIOLIN_FLE, ind_file=GENERATION_IND_FILE, generation_size=POPULATION_SIZE)
    # save_genealogy(toolbox, history, genealogy_file=GENERATION_GENEALOGY_FILE) # genealogy for plotting crossover and mutation.

    # Pick the best individual
    best_ind = tools.selBest(final_pop, 1)[0]
    best_ind_decoded = ga_decodeInteger_x(best_ind)
    decoded_parameters = ga_adjustReal_x(best_ind_decoded)
    
    # Call back the grid generator.
    gridGeneratorInit.update_parameters(decoded_parameters)
    gridGeneratorInit.create_grids()
    gridGeneratorInit.merge_grids()

    # Visualization of the generated grids.
    print("best ind decoded parameter values:", decoded_parameters)

    if args.set_plot:
        gridGeneratorInit.visualization_2d_before_merge(visualization_storage_path=MODEL_GA_RES_PATH, add_strs='ga')
        gridGeneratorInit.visualization_2d_after_merge(visualization_storage_path=MODEL_GA_RES_PATH, add_strs='ga')

if __name__ == "__main__":

    main()

# ========================references===========================
# https://github.com/DEAP/deap/blob/master/examples/ga/onemax_mp.py
# https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
# https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000384464-Problem-using-multiprocess-with-IPython
# Check papers:
# Automated optimization of steel reinforcement in RC building frames using building information modeling and hybrid genetic algorithm
# check real-valued link. + the existing method of correcting non-integer to integer.
# https://www.researchgate.net/post/How_can_I_encode_and_decode_a_real-valued_problem-variable_in_Genetic_Algorithms
# # have to take care that the numbers do not go outside your range
# https://gitlab.com/santiagoandre/deap-customize-population-example/-/blob/master/AGbasic.py?ref_type=heads
# links for constraint handling:
# https://stackoverflow.com/questions/20301206/enforce-constraints-in-genetic-algorithm-with-deap
# https://github.com/deap/deap/issues/30
# save all individuals of all generations.
# https://groups.google.com/g/deap-users/c/9IHsKGR9Daw
# networkx.graphviz_layout for printing history, the source code:
# https://gist.github.com/fmder/5505431
# importance of invalid fitness in DEAP.
# https://stackoverflow.com/questions/44708050/whats-the-importance-of-invalid-fitness-in-deap 

# ========================default_initialization===========================
# save for default initialization.
# PARAM_BOUNDS = [value for value in PARAMS.values()]
# NUM_PARAMS = len(PARAM_BOUNDS)
# CHROMOSOME_LENGTH = 20 # length of the chromosome (individual),  which should be divisible by no. of variables (in bit form). when this length gets smaller, it only returns integers..
# if CHROMOSOME_LENGTH % NUM_PARAMS != 0:
#     raise ValueError(f"The value {CHROMOSOME_LENGTH} should be divisible by no. of variables")
# logging.info("CHROMOSOME_LENGTH: %s", CHROMOSOME_LENGTH)

# ========================save===========================
# def decode_binary_x(individual: list) -> list:
#     """Decode binary list to parameter values based on defined bounds."""
#     len_chromosome = len(individual)
#     len_chromosome_one_var = int(len_chromosome/NUM_PARAMS)
#     bound_index = 0
#     x = []
#     for i in range(0,len_chromosome,len_chromosome_one_var):
#         # converts binary to decimial using 2**place_value
#         chromosome_string = ''.join((str(xi) for xi in  individual[i:i+len_chromosome_one_var]))
#         binary_to_decimal = int(chromosome_string,2)
#         # the decoding method that 
#         # a. can implement lower and upper bounds for each variable
#         # b. can choose chromosome of any length (more the no. of bits, more precise the decoded value).
#         lb, ub= PARAM_BOUNDS[bound_index]
#         precision = (ub-lb)/((2**len_chromosome_one_var)-1)
#         decoded = (binary_to_decimal*precision)+lb
#         x.append(decoded)
#         bound_index +=1
#     return x

# ========================save===========================
# # Multiprocesssing Functions
# def monitor_resources():
#     """Function to log CPU and Memory usage of the current process."""
#     process = psutil.Process()
#     memory_info = process.memory_info()
#     logging.info(f"Process {process.pid}: Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")
#     # logging.info(f"Process {process.pid}: CPU usage: {process.cpu_percent(interval=1)}%") # always 0%...
# def worker_init():
#     """Initialize worker process to monitor its resources."""
#     logging.info(f"Worker {multiprocessing.current_process().pid} started.")
#     monitor_resources()  # Monitor the current worker

# ========================save===========================
# ncore = "1"
# os.environ["OMP_NUM_THREADS"] = ncore
# os.environ["OPENBLAS_NUM_THREADS"] = ncore
# os.environ["MKL_NUM_THREADS"] = ncore
# os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
# os.environ["NUMEXPR_NUM_THREADS"] = ncore
# =======================================================

# ========================save===========================
# Individual initializer
    # <default>
    # creator.create("Individual", list, fitness=creator.FitnessMulti)
    # Attribute initializer
    # <default>
    # toolbox.register("attr_bool", random.randint, 0, 1) # attribute generator with toolbox.attr_bool() drawing a random integer between 0 and 1
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, CHROMOSOME_LENGTH)  # depending upon decoding strategy, which uses precision
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ========================save===========================
# def save_genealogy(toolbox, history, genealogy_file):
#     print ("history.genealogy_history:", history.genealogy_history)
#     print ("history.genealogy_tree:", history.genealogy_tree)
    # # log = history.genealogy_tree
    # # genealogy_history 

    # with open(GENERATION_LOG_GENEALOGY_FILE, "w") as output_file:
    #     json.dump(log, output_file)

    # # # visualization.
    # graph = nx.DiGraph(history.genealogy_tree)
    # graph = graph.reverse()  # Make the graph top-down
    # plt.figure(figsize=(10, 15)) 
    # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]

    # positions = graphviz_layout(graph, prog="dot")
    # nx.draw(graph, positions, node_color=colors)
    # plt.savefig(genealogy_file)
    # plt.close() # Close the figure to free up memory