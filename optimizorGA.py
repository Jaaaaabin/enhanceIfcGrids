import os
import random
import numpy as np

from deap import base, creator, tools, algorithms
from gridGenerator import GridGenerator

PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_test'
DATA_RES_PATH = PROJECT_PATH + r'\res'

# Threshold parameter variation ranges for GA
# to be collected externally from fixed values determined from the main branch.
GA_PARAMS = {
    't_c_num': (1, 5),
    't_c_dist': (0.00005, 0.00015),
    't_w_num': (1, 5),
    't_w_dist': (0.00005, 0.00015),
    't_w_st_accumuled_length': (20.0, 30.0),
    't_w_ns_accumuled_length': (10.0, 30.0)
}

def preparation_of_grid_generation(work_path, ifc_model, info_columns='info_columns.json', info_walls='info_walls.json'):

    generator = GridGenerator(
        os.path.join(work_path, ifc_model),
        os.path.join(work_path, ifc_model, info_columns),
        os.path.join(work_path, ifc_model, info_walls),
        )
    generator.prepare_wall_lengths()
    generator.get_main_storeys_and_directions(num_directions=2) # static.
    generator.enrich_main_storeys() # static.

    return generator

class AcceleratorGA:
    def __init__(self, init_grid_generator, problem_size, population_size, generations, crossover_prob, mutation_prob):

        self.set_random_seed(RANDOM_SEED)

        self.init_grid_generator = init_grid_generator
        self.problem_size = problem_size
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # Setting up the DEAP toolbox
        self.toolbox = base.Toolbox()
        self.setup()
    
    def set_random_seed(self, seed):
        """Set the random seed for numpy and random modules."""
        random.seed(seed)
        np.random.seed(seed)
    
    def attr_generator(self, param_range):
        """Generalized attribute generator for both int and float ranges."""
        return lambda: random.uniform(*param_range)

    def setup(self):

        def generate_individual(toolbox):
            individual = [toolbox.__getattribute__(param)() for param in GA_PARAMS]
            return individual
    
        # Register attribute generators based on predefined ranges
        for param, param_range in GA_PARAMS.items():
            self.toolbox.register(param, self.attr_generator(param_range))
        
        for param, (start, end) in GA_PARAMS.items():
            if isinstance(start, int):
                self.toolbox.register(param, random.randint, start, end)
            else:
                self.toolbox.register(param, random.uniform, start, end)

        # Registering genetic operators
        self.toolbox.register("individual", generate_individual, self.toolbox)
        self.toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (self.toolbox.__getattribute__(param) for param in GA_PARAMS), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=self.mutation_prob)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual):
        """
        Evaluate an individual's fitness by updating parameters and calculating loss.
        """
        parameter_values = dict(zip(GA_PARAMS.keys(), individual))
        self.init_grid_generator.update_parameters(parameter_values)
        self.init_grid_generator.create_grids()
        self.init_grid_generator.calculate_cross_lost()
        
        return self.init_grid_generator.perct_crossing_ns_wall + self.init_grid_generator.perct_crossing_st_wall,
        # return (1.0,)

    def run(self):
        """
        Execute the genetic algorithm.
        """
        population = self.toolbox.population(n=self.population_size)

        # Statistics for monitoring
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Genetic algorithm execution
        population, log = algorithms.eaSimple(population, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob, ngen=self.generations, stats=stats, verbose=True)

        # Identifying the best solution
        best_ind = tools.selBest(population, 1)[0]
        print(f"Best individual is {best_ind}, Fitness: {best_ind.fitness.values}")
        
        return best_ind
#     
model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
model_path = model_paths[0]
init_grid_generator = preparation_of_grid_generation(DATA_RES_PATH, model_path)



# Meta-data prameters for GA
S_PROBLEM = len(GA_PARAMS.keys())
S_POPULATION = 4
N_GENERATION = 2

# Random Seeds
RANDOM_SEED = 2024
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


ga_optimizer = AcceleratorGA(
    init_grid_generator=init_grid_generator,
    problem_size=S_PROBLEM,
    population_size=S_POPULATION,
    generations=N_GENERATION,
    crossover_prob=0.7,
    mutation_prob=0.2)

best_solution = ga_optimizer.run()