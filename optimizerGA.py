import os
import copy
import logging
import random

import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from gridGenerator import GridGenerator
from main import preparation_of_grid_generation

#===================================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and paths setup
PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_test')
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')

# Utility to read model paths
def get_ifc_model_paths(folder_path: str) -> list:
    model_paths = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
    if not model_paths:
        logging.error("No existing model paths found.")
        raise FileNotFoundError("No model files in DATA_FOLDER_PATH")
    return model_paths

MODEL_PATHS = get_ifc_model_paths(DATA_FOLDER_PATH)
MODEL_PATH = MODEL_PATHS[0]  # Assuming we use the first model path
gridGeneratorInit = preparation_of_grid_generation(DATA_RES_PATH, MODEL_PATH)

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
# Genetic Algorithm Configuration
NUM_PARAMS = len(PARAM_BOUNDS)

POPULATION_SIZE = 10 # population size or no of individuals or solutions being considered in each generation.
NUM_GENERATIONS = 3 # number of iterations.

CHROMOSOME_LENGTH = 40 # length of the chromosome (individual), which should be divisible by no. of variables (in bit form).
TOURNAMENT_SIZE = 5 # number of participants in tournament selection.

# todo..
# what is the difference.....
crossover_prob = 0.5 # the probability with which two individuals are crossed or mated, high means more random jumps or deviation from parents, which is generally not desired
mutation_prob = 0.2 # the probability for mutating an individual

if CHROMOSOME_LENGTH % NUM_PARAMS != 0:
    raise ValueError(f"The value {CHROMOSOME_LENGTH} should be divisible by no. of variables")

# ===================================================================================================
# Basic Functions of GA
def decode_all_x(individual: list) -> list:
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

def objective_fxn(individual: list) -> tuple:     
    """Evaluate the fitness of an individual based on grid performance metrics."""
    decoded_individual = decode_all_x(individual)
    decoded_parameters = dict(zip(PARAMS.keys(), decoded_individual))
    
    for key, value in list(decoded_parameters.items()):
        if '_num' in key:
            decoded_parameters[key] = int(value)
    
    gridGeneratorInit.update_parameters(decoded_parameters)
    gridGeneratorInit.create_grids()
    gridGeneratorInit.calculate_grid_wall_cross_loss(ignore_cross_edge=True)

    # the return value must be a list / tuple, even it's only one fitness value.
    return (gridGeneratorInit.percent_unbound_w_numbers, gridGeneratorInit.percent_cross_unbound_w_lengths)

# ===================================================================================================
# Optimizaton Functions of GA
# Population Initialization.
def initializerGA():

    ini_pop = toolbox.population(n=POPULATION_SIZE)

    return ini_pop

# Population Optimization.
def optimizerGA(pop):

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    hall_of_fame = tools.HallOfFame(1)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    final_pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=NUM_GENERATIONS,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True)
    
    best_ind = tools.selBest(pop, 1)[0]
    logging.info(f"Best individual is {best_ind}, Fitness: {best_ind.fitness.values}")

    visualize_fitness(logbook)

    return final_pop, best_ind, logbook

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
# Genetic Algorithm Set up with DEAP library.
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# ===================================================================================================
# Genetic Algorithm Registration with DEAP library.
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1) # attribute generator with toolbox.attr_bool() drawing a random integer between 0 and 1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, CHROMOSOME_LENGTH)  # depending upon decoding strategy, which uses precision
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", objective_fxn) # privide the objective function here

# registering basic processes using DEAP bulit-in functions
toolbox.register("mate", tools.cxUniform, indpb=crossover_prob) # strategy for crossover, this classic two point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_prob) # mutation strategy with probability of mutation
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE) # selection startegy

# run.
initial_population = initializerGA()
final_pop, best_ind, logbook = optimizerGA(initial_population)
best_ind_decoded = decode_all_x(best_ind)

# plot the final best.
decoded_parameters = dict(zip(PARAMS.keys(), best_ind_decoded))
for key, value in list(decoded_parameters.items()):
    if '_num' in key:
        decoded_parameters[key] = int(value)
gridGeneratorInit.update_parameters(decoded_parameters)
gridGeneratorInit.create_grids()
gridGeneratorInit.visualization_2d()

print("checkpoint.")
