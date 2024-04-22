import os
import copy
import logging
import random
import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from gridGenerator import GridGenerator
from main import preparation_of_grid_generation

#===================================================================================================
# Constants and paths setup
PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_test')
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')

# Utility to read model paths
def get_ifc_model_paths(folder_path):
    model_paths = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
    if not model_paths:
        logging.error("No existing model paths found.")
        raise FileNotFoundError("No model files in DATA_FOLDER_PATH")
    return model_paths

model_paths = get_ifc_model_paths(DATA_FOLDER_PATH)
model_path = model_paths[0]  # Assuming we use the first model path

# Preparation of the initial grid generator
gridGeneratorInit = preparation_of_grid_generation(DATA_RES_PATH, model_path)

#===================================================================================================
# Basic parameter / vairbales and the preset bounds
PARAMS = {
    'st_c_num': (2, 8),
    # 'st_c_dist': (0.00001, 0.0001),
    'st_w_num': (2, 8),
    # 'st_w_dist': (0.00001, 0.0001),
    'st_w_accumuled_length': (2.0, 12.0),
    'ns_w_num': (2, 5),
    # 'ns_w_dist': (0.00001, 0.0001),
    'ns_w_accumuled_length': (2.0, 10.0),
}

param_bounds = [value for value in PARAMS.values()]

#===================================================================================================
# Genetic Algorithm Configuration
NUM_PARAMS = len(param_bounds)

POPULATION_SIZE = 10 # population size or no of individuals or solutions being considered in each generation.
NUM_GENERATIONS = 3 # number of iterations.

CHROMOSOME_LENGTH = 40 # length of the chromosome (individual), which should be divisible by no. of variables (in bit form).
TOURNAMENT_SIZE = 5 # number of participants in tournament selection.

MUTATION_RATE = 0.05 # the probability that a gene will mutate or flip: generally kept low, high means more random jumps or deviation from parents, which is generally not desired
CROSSOVER_RATE = 0.7
CXPB = 0.5 # the probability with which two individuals are crossed or mated
MUTPB = 0.2 # the probability for mutating an individual

if CHROMOSOME_LENGTH % NUM_PARAMS != 0:
    raise ValueError(f"The value {CHROMOSOME_LENGTH} should be divisible by no. of variables")

# ===================================================================================================
# Basic Functions of GA
def decode_all_x(individual: list, NUM_PARAMS: int, bounds: list) -> list:
    '''
    returns list of decoded x in same order as we have in binary format in chromosome
    bounds should have upper and lower limit for each variable in same order as we have in binary format in chromosome 
    '''
    
    len_chromosome = len(individual)
    len_chromosome_one_var = int(len_chromosome/NUM_PARAMS)
    bound_index = 0
    x = []
    
    for i in range(0,len_chromosome,len_chromosome_one_var):
        
        # converts binary to decimial using 2**place_value
        chromosome_string = ''.join((str(xi) for xi in  individual[i:i+len_chromosome_one_var]))
        binary_to_decimal = int(chromosome_string,2)
        
        # decoding methods that 
        # can implement lower and upper bounds for each variable
        # can choose chromosome of any length (more the no. of bits, more precise the decoded value).
        lb = bounds[bound_index][0]
        ub = bounds[bound_index][1]
        precision = (ub-lb)/((2**len_chromosome_one_var)-1)
        decoded = (binary_to_decimal*precision)+lb
        x.append(decoded)
        bound_index +=1
    
    return x

def objective_fxn(individual: list) -> tuple:     
    
    decoded_individual = decode_all_x(individual, NUM_PARAMS, param_bounds)
    decoded_parameters = dict(zip(PARAMS.keys(), decoded_individual))
    
    for key, value in list(decoded_parameters.items()):
        if '_num' in key:
            decoded_parameters[key] = int(value)
    
    gridGeneratorInit.update_parameters(decoded_parameters)
    gridGeneratorInit.create_grids()
    gridGeneratorInit.calculate_grid_wall_cross_loss(ignore_cross_edge=True)

    # the return value must be a list / tuple, even it's only one fitness value.
    return (gridGeneratorInit.percent_cross_w_lengths,)

# ===================================================================================================
# Optimizaton Functions of GA

def optimizerGA(
    population_size: int,
    num_generations: int,
    crossover_prob: float,
    mutation_prob: float):
    
    try:

        ini_pop = toolbox.population(n=population_size)
        pop = copy.deepcopy(ini_pop)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, log = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=num_generations,
            stats=stats,
            verbose=True)
        
        best_ind = tools.selBest(pop, 1)[0]
        logging.info(f"Best individual is {best_ind}, Fitness: {best_ind.fitness.values}")
        
        return ini_pop, pop, log, best_ind
    
    except Exception as e:
        logging.error(f"Error occurs during GA optimization: {e}")
        raise

# ===================================================================================================
# Genetic Algorithm Set up with DEAP library.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1) # attribute generator with toolbox.attr_bool() drawing a random integer between 0 and 1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, CHROMOSOME_LENGTH)  # depending upon decoding strategy, which uses precision
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ===================================================================================================
# Genetic Algorithm Registration with DEAP library.
# registering objetive function with constraint
toolbox.register("evaluate", objective_fxn) # privide the objective function here

# registering basic processes using bulit in functions in DEAP
toolbox.register("mate", tools.cxTwoPoint) # strategy for crossover, this classic two point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_RATE) # mutation strategy with probability of mutation
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE) # selection startegy

initial_pop, pop, log, best_ind = optimizerGA(
    population_size=POPULATION_SIZE,
    num_generations=NUM_GENERATIONS,
    crossover_prob=CXPB,
    mutation_prob=MUTPB)