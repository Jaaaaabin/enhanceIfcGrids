import sys
import ctypes

# Enable Virtual Terminal Processing to display ANSI colors in Windows console before numpy
if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

import os
import copy
import array
import random
import logging
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from ifc_grid_generation import preparation_of_grid_generation

from gaTools import getIfcModelPaths, getParameterScales, getParameterVarLimits
from gaTools import createInds, saveLogbook, visualizeGenFitness, visualizeGenFitnessViolin
from gaTools import ga_eaSimple

#===================================================================================================
# Genetic Algorithm Configuration - Constants
POPULATION_SIZE = 30 # population size or no of individuals or solutions being considered in each generation.
NUM_GENERATIONS = 20 # number of iterations.

TOURNAMENT_SIZE = 3 # number of participants in tournament selection.
CROSS_PROB = 0.5 # the probability with which two individuals are crossed or mated
MUTAT_PROB = 0.1 # the probability for mutating an individual

NUM_PROCESS = 8
RANDOM_SEED = 20001
#===================================================================================================
# Paths setup and Log registration.
PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_test')
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')

MODEL_PATH = getIfcModelPaths(folder_path=DATA_FOLDER_PATH, only_first=True)

gridGeneratorInit = preparation_of_grid_generation(DATA_RES_PATH, MODEL_PATH)
logging.basicConfig(filename='ga.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s') # reconfigurate the logging file.

INI_GENERATION_FILE = os.path.join(DATA_RES_PATH, MODEL_PATH,'GA_generation_ini_inds_integer.txt')
GENERATION_LOG_FILE = os.path.join(DATA_RES_PATH, MODEL_PATH, "GA_generation_log.json")
GENERATION_FIT_FILE = os.path.join(DATA_RES_PATH, MODEL_PATH, "GA_generation_fitness.png")
GENERATION_IND_FILE = os.path.join(DATA_RES_PATH, MODEL_PATH, "GA_generation_inds.txt")
GENERATION_IND_VIOLIN_FLE = os.path.join(DATA_RES_PATH, MODEL_PATH, "GA_generation_ind_violin.png")

#===================================================================================================
# Basic parameter & Customized Population setup:
PARAMS = {
    'st_c_num': (3, 10), # [3,10) 
    'st_w_num': (1, 10), # [1,10)
    'ns_w_num': (2, 10), # [2,10)
    'st_w_accumuled_length_percent': (0.0001, 0.0100),
    'ns_w_accumuled_length_percent': (0.0001, 0.0100),
    'st_st_merge': (0.01, 0.30),
    'ns_st_merge': (0.10, 1.50),
    # 'st_c_dist': (0.00001, 0.0001), # fixed as 0.001
    # 'st_w_dist': (0.00001, 0.0001), # fixed as 0.001
    # 'ns_w_dist': (0.00001, 0.0001), # fixed as 0.001
}

# Get the additional Constant Values.
PARAMS_SCALE, PARAMS_INTEGER = getParameterScales(param_ranges=PARAMS)
MinVals, MaxVals = getParameterVarLimits(param_ranges=PARAMS_INTEGER)

# ===================================================================================================
# Basic Functions of GA
def ga_loadInds(creator, n=POPULATION_SIZE, filename=INI_GENERATION_FILE):
    individuals = []
    try:
        # Since the individual data is expected to be a list of integers:
        with open(filename, 'r') as file:
            for line in file:
                individual = list(map(int, line.strip().strip('[]').split(',')))
                individual = creator(individual)
                individuals.append(individual)
    except FileNotFoundError:
        logging.error(f"The file {filename} was not found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading individuals: {e}")
        return None
    
    if len(individuals) == n:
        return individuals
    else:
        logging.warning(f"Loaded {len(individuals)} individuals, expected {n}.")
        return None
    
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
    gridGenerator.merged_loss_unbound_elements2grids()
    gridGenerator.merged_loss_distance_deviation()

    individual_fitness = gridGenerator.percent_unbound_elements/2 + gridGenerator.avg_deviation_distance/2

    return (individual_fitness,) # the return value must be a list / tuple, even it's only one fitness value.

# ===================================================================================================
# main.
def main(random_seed=[], num_processes=1):

    if random_seed:
        random.seed(random_seed)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMin) 
    toolbox = base.Toolbox()

    createInds(n=POPULATION_SIZE, param_rangs=PARAMS_INTEGER, filename=INI_GENERATION_FILE)
    toolbox.register("population", ga_loadInds, creator.Individual)

    # Evaluator initializer
    toolbox.register("evaluate", ga_objective) # privide the objective function here
    # <To check if there's other more sustanible Penalty methods>, for example: toolbox.decorate("evaluate", tools.DeltaPenalty(feasible_fxn, 1.0))
    
    # registering basic processes using DEAP bulit-in functions
    toolbox.register("mate", tools.cxUniform, indpb=CROSS_PROB)
    toolbox.register("mutate", tools.mutUniformInt, low=MinVals, up=MaxVals, indpb=MUTAT_PROB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    # Clear the old generation individual file.
    if os.path.exists(GENERATION_IND_FILE):
        os.remove(GENERATION_IND_FILE)
        
    # Process Pool of multi workers
    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
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
    
    # final_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CROSS_PROB, mutpb=MUTAT_PROB, 
    #     ngen=NUM_GENERATIONS, stats=stats, verbose=True)
    
    final_pop, logbook = ga_eaSimple(pop, toolbox, cxpb=CROSS_PROB, mutpb=MUTAT_PROB, 
        ngen=NUM_GENERATIONS, fitness_file=GENERATION_IND_FILE, stats=stats, verbose=True)
    
    if num_processes > 1:
        pool.close()
    
    # Analysis of the GA.
    saveLogbook(logbook=logbook, log_file=GENERATION_LOG_FILE) 
    visualizeGenFitness(logbook=logbook, fitness_file=GENERATION_FIT_FILE)
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
    gridGeneratorInit.visualization_2d_before_merge()
    gridGeneratorInit.visualization_2d_after_merge()

if __name__ == "__main__":

    main(random_seed=RANDOM_SEED, num_processes=NUM_PROCESS)


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