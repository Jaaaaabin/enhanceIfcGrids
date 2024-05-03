import os
import sys
import copy
import ctypes

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
from deap import base, creator, tools, algorithms

from quickTools import time_decorator
from ifc_grid_generation import preparation_of_grid_generation

# references.s
# https://github.com/DEAP/deap/blob/master/examples/ga/onemax_mp.py
# https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
# https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000384464-Problem-using-multiprocess-with-IPython

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

#===================================================================================================
# Log registration.
# reconfigurate the logging file.
logging.basicConfig(filename='ga.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

#===================================================================================================
# Basic parameter / vairbales and the preset bounds
PARAMS = {
    'st_c_num': (2, 6),
    # 'st_c_dist': (0.00001, 0.0001),
    'st_w_num': (2, 4),
    # 'st_w_dist': (0.00001, 0.0001),
    'st_w_accumuled_length': (2.0, 20.0),
    'ns_w_num': (2, 5),
    # 'ns_w_dist': (0.00001, 0.0001),
    'ns_w_accumuled_length': (2.0, 20.0),
}
PARAM_BOUNDS = [value for value in PARAMS.values()]

#===================================================================================================
# Genetic Algorithm Configuration - Constants
NUM_PARAMS = len(PARAM_BOUNDS)

POPULATION_SIZE = 20 # population size or no of individuals or solutions being considered in each generation.
NUM_GENERATIONS = 15 # number of iterations.

CHROMOSOME_LENGTH = 40 # length of the chromosome (individual), which should be divisible by no. of variables (in bit form). when this length gets smaller, it only returns integers..
TOURNAMENT_SIZE = 3 # number of participants in tournament selection.

# todo.. how different it can lead by different probabilities of crossover and mutation
CROSS_PROB = 0.2 # the probability with which two individuals are crossed or mated, high means more random jumps or deviation from parents, which is generally not desired
MUTAT_PROB = 0.5 # the probability for mutating an individual

if CHROMOSOME_LENGTH % NUM_PARAMS != 0:
    raise ValueError(f"The value {CHROMOSOME_LENGTH} should be divisible by no. of variables")

logging.info("POPULATION_SIZE: %s", POPULATION_SIZE)
logging.info("NUM_GENERATIONS: %s", NUM_GENERATIONS)
logging.info("CHROMOSOME_LENGTH: %s", CHROMOSOME_LENGTH)
logging.info("TOURNAMENT_SIZE: %s", TOURNAMENT_SIZE)
logging.info("CROSS_PROB: %s", CROSS_PROB)
logging.info("MUTAT_PROB: %s", MUTAT_PROB)

# ===================================================================================================
# Multiprocesssing Functions
def monitor_resources():
    """Function to log CPU and Memory usage of the current process."""
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"Process {process.pid}: Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")
    # logging.info(f"Process {process.pid}: CPU usage: {process.cpu_percent(interval=1)}%") # always 0%...

def worker_init():
    """Initialize worker process to monitor its resources."""
    logging.info(f"Worker {multiprocessing.current_process().pid} started.")
    monitor_resources()  # Monitor the current worker

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
# Basic Functions of GA
def decode_binary_x(individual: list) -> list:
    """Decode binary list to parameter values based on defined bounds."""
    len_chromosome = len(individual)
    len_chromosome_one_var = int(len_chromosome/NUM_PARAMS)
    bound_index = 0
    x = []
    
    for i in range(0,len_chromosome,len_chromosome_one_var):
        
        # converts binary to decimial using 2**place_value
        chromosome_string = ''.join((str(xi) for xi in  individual[i:i+len_chromosome_one_var]))
        binary_to_decimal = int(chromosome_string,2)
        
        # the decoding method that 
        # a. can implement lower and upper bounds for each variable
        # b. can choose chromosome of any length (more the no. of bits, more precise the decoded value).

        lb, ub= PARAM_BOUNDS[bound_index]
        precision = (ub-lb)/((2**len_chromosome_one_var)-1)
        decoded = (binary_to_decimal*precision)+lb
        x.append(decoded)
        bound_index +=1
    
    return x

def adjust_x_values(decoded_x):

    decoded_parameters = dict(zip(PARAMS.keys(), decoded_x))
    
    for key, value in list(decoded_parameters.items()):
        if '_num' in key:
            decoded_parameters[key] = int(value)

    return decoded_parameters

# Objective Functions of genetic algorithm (GA)
# @time_decorator
def objective_fxn(individual: list) -> tuple:

    """Evaluate the fitness of an individual based on grid performance metrics."""
    # decode the binary individuals to real parameter values.
    decoded_individual = decode_binary_x(individual)
    decoded_parameters = adjust_x_values(decoded_individual)

    # build the gridGenerator.
    gridGenerator = copy.deepcopy(gridGeneratorInit)
    gridGenerator.update_parameters(decoded_parameters)
    
    # create grids and calculate the losses.
    gridGenerator.create_grids() # gets slower.
    gridGenerator.calculate_grid_wall_cross_loss(ignore_cross_edge=True)
    # gridGenerator.calculate_grid_distance_deviation_loss()
    
    # print ("self.percent_unbound_w_numbers:", gridGenerator.percent_unbound_w_numbers)
    # print ("self.percent_cross_unbound_w_lengths:", gridGenerator.percent_cross_unbound_w_lengths)
    # print ("self.avg_deviation_distance_st:", gridGenerator.avg_deviation_distance_st)

    decoded_parameters =  {k: round(v, 4) for k, v in decoded_parameters.items()}
    logging.info("The Ifc input parameters: %s", decoded_parameters)

    # the return value must be a list / tuple, even it's only one fitness value.
    return (
        # gridGenerator.percent_unbound_w_numbers,)
        gridGenerator.percent_cross_unbound_w_lengths,)
        # gridGenerator.avg_deviation_distance_st)

# ===================================================================================================
# main.
def main(random_seed, num_processes=1):

    random.seed(random_seed)

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # Attribute initializer
    toolbox.register("attr_bool", random.randint, 0, 1) # attribute generator with toolbox.attr_bool() drawing a random integer between 0 and 1
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, CHROMOSOME_LENGTH)  # depending upon decoding strategy, which uses precision
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_fxn) # privide the objective function here

    # registering basic processes using DEAP bulit-in functions
    toolbox.register("mate", tools.cxUniform, indpb=CROSS_PROB) # strategy for crossover, this classic two point crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTAT_PROB) # mutation strategy with probability of mutation
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE) # selection startegy

    # Process Pool of multi workers
    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes, initializer=worker_init)
        toolbox.register("map", pool.map)

    pop = toolbox.population(n=POPULATION_SIZE)
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
    best_ind_decoded = decode_binary_x(best_ind)
    decoded_parameters = adjust_x_values(best_ind_decoded)
    print("best ind decoded parameter values:", decoded_parameters)

    gridGeneratorInit.update_parameters(decoded_parameters)
    gridGeneratorInit.create_grids()
    gridGeneratorInit.visualization_2d()

if __name__ == "__main__":

    main(random_seed=100, num_processes=8)