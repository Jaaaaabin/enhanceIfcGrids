import os
import sys
import copy
import ctypes
import array

# Enable Virtual Terminal Processing to display ANSI colors in Windows console
if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ncore = "1"
# os.environ["OMP_NUM_THREADS"] = ncore
# os.environ["OPENBLAS_NUM_THREADS"] = ncore
# os.environ["MKL_NUM_THREADS"] = ncore
# os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
# os.environ["NUMEXPR_NUM_THREADS"] = ncore

import numpy as np
import random
import logging
import psutil
import multiprocessing

import matplotlib.pyplot as plt
from math import ceil, log10
from deap import base, creator, tools, algorithms

from quickTools import time_decorator
from ifc_grid_generation import preparation_of_grid_generation

# references.s
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
#===================================================================================================
# Paths setup:
PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_test_ga')
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')

def get_ifc_model_paths(folder_path: str) -> list:
    model_paths = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
    if not model_paths:
        logging.error("No existing model paths found.")
        raise FileNotFoundError("No model files in DATA_FOLDER_PATH")
    return model_paths

MODEL_PATHS = get_ifc_model_paths(DATA_FOLDER_PATH)
MODEL_PATH = MODEL_PATHS[0]  # Assuming we take only one model every time for the ga testing..
gridGeneratorInit = preparation_of_grid_generation(DATA_RES_PATH, MODEL_PATH) # initial gridGenerator to be "deep" copied...
INI_GENERATION_TXT = os.path.join(DATA_RES_PATH, MODEL_PATH,'ini_int_individuals.txt')
                                  
#===================================================================================================
# Log registration.
# reconfigurate the logging file.
logging.basicConfig(filename='ga.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

#===================================================================================================
# Basic parameter / vairbales and the preset bounds
PARAMS = {
    'st_w_accumuled_length_percent': (0.0001, 0.0100),
    'ns_w_accumuled_length_percent': (0.0001, 0.0100),
    'st_c_num': (2, 9),
    'st_w_num': (2, 9),
    'ns_w_num': (2, 9),
    # 'st_c_dist': (0.00001, 0.0001),
    # 'st_w_dist': (0.00001, 0.0001),
    # 'ns_w_dist': (0.00001, 0.0001),
}
PARAM_BOUNDS = [value for value in PARAMS.values()]

#===================================================================================================
# Genetic Algorithm Configuration - Constants
NUM_PARAMS = len(PARAM_BOUNDS)

POPULATION_SIZE = 50 # population size or no of individuals or solutions being considered in each generation.
NUM_GENERATIONS = 30 # number of iterations.

CHROMOSOME_LENGTH = 20 # length of the chromosome (individual), which should be divisible by no. of variables (in bit form). when this length gets smaller, it only returns integers..
TOURNAMENT_SIZE = 5 # number of participants in tournament selection.

# todo.. how different it can lead by different probabilities of crossover and mutation
CROSS_PROB = 0.5 # the probability with which two individuals are crossed or mated
MUTAT_PROB = 0.1 # the probability for mutating an individual

if CHROMOSOME_LENGTH % NUM_PARAMS != 0:
    raise ValueError(f"The value {CHROMOSOME_LENGTH} should be divisible by no. of variables")

logging.info("POPULATION_SIZE: %s", POPULATION_SIZE)
logging.info("NUM_GENERATIONS: %s", NUM_GENERATIONS)
logging.info("CHROMOSOME_LENGTH: %s", CHROMOSOME_LENGTH)
logging.info("TOURNAMENT_SIZE: %s", TOURNAMENT_SIZE)
logging.info("CROSS_PROB: %s", CROSS_PROB)
logging.info("MUTAT_PROB: %s", MUTAT_PROB)

#===================================================================================================
# Customize Population setup:
def get_parameter_scales(param_ranges=PARAMS):

    # convert the dict ranges.
    integer_param_ranges = {}
    scale_param_ranges = {}

    for key, (lower, upper) in param_ranges.items():

        # Determine the smallest power of 10 that converts both limits to integers
        lower_decimals = ceil(-log10(lower % 1)) if lower % 1 != 0 else 0
        upper_decimals = ceil(-log10(upper % 1)) if upper % 1 != 0 else 0

        # Choose the maximum number of decimal places to determine scale factor
        max_decimals = max(lower_decimals,upper_decimals)
        
        # Apply the scale factor to convert both limits to integers
        scale_factor = 10 ** max_decimals
        new_lower, new_upper = int(lower * scale_factor), int(upper * scale_factor)

        # store the new dictionarys and the scale values.
        scale_param_ranges[key] = scale_factor
        integer_param_ranges[key] = (new_lower, new_upper)

    return scale_param_ranges, integer_param_ranges

PARAMS_SCALE, PARAMS_INTEGER = get_parameter_scales()

def generate_one_individual(ranges=PARAMS_INTEGER):

    # convert the dict ranges.
    integer_individual = []

    for key, (lower, upper) in ranges.items():
        size_max = len(str(abs(upper-1)))
        integer_v = str(random.randint(lower, upper-1)).zfill(size_max)
        integer_individual.append(integer_v)
        
    integer_individual = list(''.join(v for v in integer_individual))
    integer_individual = list(map(int, integer_individual))

    return integer_individual

def create_and_store_individuals(n=POPULATION_SIZE, filename=INI_GENERATION_TXT):
    
    int_lists = [generate_one_individual() for _ in range(n)]

    with open(filename, 'w') as file:
        for int_list in int_lists:
            file.write(str(int_list) + '\n')

def read_and_load_individuals(creator, n=POPULATION_SIZE, filename=INI_GENERATION_TXT):
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

# ===================================================================================================
# Fitness Visualization.
def visualize_fitness(logbook):
    """Visualize and save the evolution of fitness over generations."""
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")

    plt.figure(figsize=(10, 5))
    plt.plot(gen, min_fitness, 'b-', label="Minimum Fitness")
    plt.plot(gen, max_fitness, 'r-', label="Maximum Fitness")
    plt.plot(gen, avg_fitness, 'g-', label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(os.path.join(DATA_RES_PATH, MODEL_PATH, "GA_fitness_over_generations.png"))
    plt.close()  # Close the figure to free up memory

    logging.info("Fitness evolution plot saved as 'fitness_over_generations.png'.")

# ===================================================================================================
# Basic Decode Functions of GA
def decode_integer_x(individual: list) -> list:
    decoded_xs = []
    count_parameter_placeholders = []
    for key, (lower_limit, upper_limit) in PARAMS_INTEGER.items():
        l = len(str(abs(upper_limit-1)))
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

def adjust_x_values(decoded_x):

    decoded_parameters = dict(zip(PARAMS.keys(), decoded_x))
    
    for key, value in list(decoded_parameters.items()):
        
        # rescale back to the real values.
        if PARAMS_SCALE[key] > 1:
            decoded_parameters[key] /= PARAMS_SCALE[key]

        # correct the integer values.
        if '_num' in key:
            decoded_parameters[key] = int(value)

    return decoded_parameters

def feasible_fxn(individual: list) -> bool:
    """penalty decorator."""
    # to improve toward a more generic version. since this really makes the evolution stick to a "maybe" local-optimal

    for item in individual[-3:]:
        if item < 1:
            return False
        else:
            continue
    return True

# Objective Functions of genetic algorithm (GA)
# @time_decorator
def objective_fxn(individual: list) -> tuple:

    """Evaluate the fitness of an individual based on grid performance metrics."""
    # decode the binary individuals to real parameter values.
    
    # pre-constraints checking.
    if not feasible_fxn(individual):
        return (0.999, 0.999,)

    # normal objective function.
    decoded_individual = decode_integer_x(list(individual))
    decoded_parameters = adjust_x_values(decoded_individual)

    # build the gridGenerator.
    gridGenerator = copy.deepcopy(gridGeneratorInit)
    gridGenerator.update_parameters(decoded_parameters)
    
    # create grids and calculate the losses.
    gridGenerator.create_grids() # gets slower.
    gridGenerator.calculate_grid_wall_cross_loss(ignore_cross_edge=True)
    gridGenerator.calculate_grid_distance_deviation_loss()
    
    # print ("self.percent_unbound_w_numbers:", gridGenerator.percent_unbound_w_numbers)
    # print ("self.percent_cross_unbound_w_lengths:", gridGenerator.percent_cross_unbound_w_lengths)
    # print ("self.avg_deviation_distance_st:", gridGenerator.avg_deviation_distance_st)

    decoded_parameters =  {k: round(v, 4) for k, v in decoded_parameters.items()}
    logging.info("The Ifc input parameters: %s", decoded_parameters)

    # the return value must be a list / tuple, even it's only one fitness value.
    return (
        gridGenerator.percent_unbound_w_numbers,
        gridGenerator.percent_cross_unbound_w_lengths,)
        # gridGenerator.avg_deviation_distance_st)

# ===================================================================================================
# main.
def main(random_seed, num_processes=1):

    random.seed(random_seed)

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0,))
    
    # Individual initializer
    # <default>
    # creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # <customize>
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMulti) 
    
    toolbox = base.Toolbox()

    # Attribute initializer
    # <default>
    # toolbox.register("attr_bool", random.randint, 0, 1) # attribute generator with toolbox.attr_bool() drawing a random integer between 0 and 1
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, CHROMOSOME_LENGTH)  # depending upon decoding strategy, which uses precision
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # <customize>
    create_and_store_individuals()
    toolbox.register("population", read_and_load_individuals, creator.Individual)

    # Evaluator initializer
    toolbox.register("evaluate", objective_fxn) # privide the objective function here

    # <To check if there's other more sustanible Penalty methods>
    # toolbox.decorate("evaluate", tools.DeltaPenalty(feasible_fxn, 1.0))

    # registering basic processes using DEAP bulit-in functions
    toolbox.register("mate", tools.cxUniform, indpb=CROSS_PROB) # strategy for crossover, this classic two point crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTAT_PROB) # mutation strategy with probability of mutation
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE) # selection startegy
    
    # Process Pool of multi workers
    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
        toolbox.register("map", pool.map)

    pop = toolbox.population()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    hof = tools.HallOfFame(1)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    final_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CROSS_PROB, mutpb=MUTAT_PROB, 
        ngen=NUM_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    
    if num_processes > 1:
        pool.close()
    
    # plot the grids.
    visualize_fitness(logbook)
    best_ind = tools.selBest(final_pop, 1)[0]

    best_ind_decoded = decode_integer_x(best_ind)
    decoded_parameters = adjust_x_values(best_ind_decoded)
    
    gridGeneratorInit.update_parameters(decoded_parameters)
    gridGeneratorInit.create_grids()
    gridGeneratorInit.visualization_2d()
    print("best ind decoded parameter values:", decoded_parameters)

if __name__ == "__main__":

    main(random_seed=20, num_processes=8)





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
#
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
