import os
import logging

import random, numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

from gridGenerator import GridGenerator
from main import preparation_of_grid_generation


PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'  # *****************to change to your own local path*****************
DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_test')
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')

model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
if not model_paths:
    logging.error("No exiting model paths found.")
    raise FileNotFoundError("No model files in DATA_FOLDER_PATH")
model_path = model_paths[0]

try:
    init_grid_generator = preparation_of_grid_generation(DATA_RES_PATH, model_path)
except Exception as e:
    logging.error(f"Failed to initialize grid generator: {e}")
    exit()

no_of_generations = 10 # decide, iterations

# decide, population size or no of individuals or solutions being considered in each generation
population_size = 20 

# chromosome (also called individual) in DEAP
# length of the individual or chrosome should be divisible by no. of variables 
# is a series of 0s and 1s in Genetic Algorithm

# here, one individual may be 
# [1,0,1,1,1,0,......,0,1,1,0,0,0,0] of length 100
# each element is called a gene or allele or simply a bit
# Individual in bit form is called a Genotype and is decoded to attain the Phenotype i.e. the 
size_of_individual = 40 

# above, higher the better but uses higher resources

# we are using bit flip as mutation,
# probability that a gene or allele will mutate or flip, 
# generally kept low, high means more random jumps or deviation from parents, which is generally not desired
probability_of_mutation = 0.05 

# no. of participants in Tournament selection
# to implement strategy to select parents which will mate to produce offspring
tournSel_k = 5 

# no, of variables which will vary,here we have x and y
# this is so because both variables are of same length and are represented by one individual
# here first 50 bits/genes represent x and the rest 50 represnt y.
no_of_params = 8

PARAMS = {
    'st_c_num': (2, 5),
    'st_c_dist': (0.00001, 0.0001),
    'st_w_num': (2, 5),
    'st_w_dist': (0.00001, 0.0001),
    'st_w_accumuled_length': (2.0, 5.0),
    'ns_w_num': (2, 5),
    'ns_w_dist': (0.00001, 0.0001),
    'ns_w_accumuled_length': (2.0, 5.0),
}

param_bounds = [value for value in PARAMS.values()]
# same for both variables in our problem 

# CXPB  is the probability with which two individuals
#       are crossed or mated
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# an Individual is a list with one more attribute called fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

# instance of Toolbox class
toolbox = base.Toolbox()

# Attribute generator, generation function 
# toolbox.attr_bool(), when called, will draw a random integer between 0 and 1
# it is equivalent to random.randint(0,1)
toolbox.register("attr_bool", random.randint, 0, 1)

# here give the no. of bits in an individual i.e. size_of_individual, here 100
# depends upon decoding strategy, which uses precision
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, size_of_individual) 
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

### ultra important.!
def decode_all_x(individual,no_of_params,bounds):
    '''
    returns list of decoded x in same order as we have in binary format in chromosome
    bounds should have upper and lower limit for each variable in same order as we have in binary format in chromosome 
    '''
    
    len_chromosome = len(individual)
    len_chromosome_one_var = int(len_chromosome/no_of_params)
    bound_index = 0
    x = []
    
    # one loop each for x(first 50 bits of individual) and y(next 50 of individual)
    for i in range(0,len_chromosome,len_chromosome_one_var):
        # converts binary to decimial using 2**place_value
        chromosome_string = ''.join((str(xi) for xi in  individual[i:i+len_chromosome_one_var]))
        binary_to_decimal = int(chromosome_string,2)
        
        # this formula for decoding gives us two benefits
        # we are able to implement lower and upper bounds for each variable
        # gives us flexibility to choose chromosome of any length, 
        #      more the no. of bits for a variable, more precise the decoded value can be
        lb = bounds[bound_index][0]
        ub = bounds[bound_index][1]
        precision = (ub-lb)/((2**len_chromosome_one_var)-1)
        decoded = (binary_to_decimal*precision)+lb
        x.append(decoded)
        bound_index +=1
    
    # returns a list of solutions in phenotype o, here [x,y]
    return x

gridGeneratorInit = preparation_of_grid_generation(DATA_RES_PATH, model_path)

def objective_fxn(individual):     
    
    # decoding chromosome to get decoded x in a list
    decode_individual = decode_all_x(individual, no_of_params, param_bounds)
    parameter_values = dict(zip(PARAMS.keys(), decode_individual))
    
    for key, value in list(parameter_values.items()):
        if '_num' in key:
            parameter_values[key] = int(value) 
    
    gridGeneratorInit.update_parameters(parameter_values)
    gridGeneratorInit.create_grids()
    gridGeneratorInit.calculate_grid_wall_cross_loss(ignore_cross_edge=True)

    return [gridGeneratorInit.percent_cross_w_lengths]

def check_feasiblity(individual):
    '''
    Feasibility function for the individual. 
    Returns True if individual is feasible (or constraint not violated),
    False otherwise
    '''
    var_list = decode_all_x(individual,no_of_params,param_bounds)
    if sum(var_list) < 0:
        return True
    else:
        return False

def penalty_fxn(individual):
    '''
    Penalty function to be implemented if individual is not feasible or violates constraint
    It is assumed that if the output of this function is added to the objective function fitness values,
    the individual has violated the constraint.
    '''
    var_list = decode_all_x(individual,no_of_params,param_bounds)
    return sum(var_list)**2


# registering objetive function with constraint
toolbox.register("evaluate", objective_fxn) # privide the objective function here

# registering basic processes using bulit in functions in DEAP
toolbox.register("mate", tools.cxTwoPoint) # strategy for crossover, this classic two point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=probability_of_mutation) # mutation strategy with probability of mutation
toolbox.register("select", tools.selTournament, tournsize=tournSel_k) # selection startegy

hall_of_fame = tools.HallOfFame(1)

stats = tools.Statistics()

# registering the functions to which we will pass the list of fitness's of a gneration's offspring
# to ge the results
stats.register('Min', np.min)
stats.register('Max', np.max)
stats.register('Avg', np.mean)
stats.register('Std', np.std)

logbook = tools.Logbook()

# create poppulation as coded in population class
# no. of individuals can be given as input
pop = toolbox.population(n=population_size)

# The next thing to do is to evaluate our brand new population.

# use map() from python to give each individual to evaluate and create a list of the result
fitnesses = list(map(toolbox.evaluate, pop)) 

# ind has individual and fit has fitness score
# individual class in deap has fitness.values attribute which is used to store fitness value
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# evolve our population until we reach the number of generations

# Variable keeping track of the number of generations
g = 0
# clearing hall_of_fame object as precaution before every run
hall_of_fame.clear()

# Begin the evolution
while g < no_of_generations:
    # A new generation
    g = g + 1
    
    #The evolution itself will be performed by selecting, mating, and mutating the individuals in our population.
    
    # the first step is to select the next generation.
    # Select the next generation individuals using select defined in toolbox here tournament selection
    # the fitness of populations is decided from the individual.fitness.values[0] attribute
    #      which we assigned earlier to each individual
    # these are best individuals selected from population after selection strategy
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals, this needs to be done to create copy and avoid problem of inplace operations
    # This is of utter importance since the genetic operators in toolbox will modify the provided objects in-place.
    offspring = list(map(toolbox.clone, offspring))
    
    # Next, we will perform both the crossover (mating) and the mutation of the produced children with 
    #        a certain probability of CXPB and MUTPB. 
    # The del statement will invalidate the fitness of the modified offspring as they are no more valid 
    #       as after crossover and mutation, the individual changes
    
    # Apply crossover and mutation on the offspring
    # note, that since we are not cloning, the changes in child1, child2 and mutant are happening inplace in offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values 
                
    # Evaluate the individuals with an invalid fitness (after we use del to make them invalid)
    # again note, that since we did not use clone, each change happening is happening inplace in offspring
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # To check the performance of the evolution, we will calculate and print the 
    # minimal, maximal, and mean values of the fitnesses of all individuals in our population 
    # as well as their standard deviations.
    # Gather all the fitnesses in one list and print the stats
    #this_gen_fitness = [ind.fitness.values[0] for ind in offspring]
    this_gen_fitness = [] # this list will have fitness value of all the offspring
    for ind in offspring:
        this_gen_fitness.append(ind.fitness.values[0])            
    
    
    #### SHORT METHOD
    
    # will update the HallOfFame object with the best individual 
    #   according to fitness value and weight (while creating base.Fitness class)
    hall_of_fame.update(offspring)
    
    # pass a list of fitnesses 
    # (basically an object on which we want to perform registered functions)
    # will return a dictionary with key = name of registered function and value is return of the registered function
    stats_of_this_gen = stats.compile(this_gen_fitness)
    
    # creating a key with generation number
    stats_of_this_gen['Generation'] = g
    
    # printing for each generation
    print(stats_of_this_gen)
    
    # recording everything in a logbook object
    # logbook is essentially a list of dictionaries
    logbook.append(stats_of_this_gen)
    
    # now one generation is over and we have offspring from that generation
    # these offspring wills serve as population for the next generation
    # this is not happening inplace because this is simple python list and not a deap framework syntax
    pop[:] = offspring


# print the best solution using HallOfFame object
for best_indi in hall_of_fame:
    # using values to return the value and
    # not a deap.creator.FitnessMin object
    best_obj_val_overall = best_indi.fitness.values[0]
    print('Minimum value for function: ',best_obj_val_overall)
    print('Optimum Solution: ',decode_all_x(best_indi,no_of_params,param_bounds))

# finding the fitness value of the fittest individual of the last generation or 
# the solution at which the algorithm finally converges
# we find this from logbook

# select method will return value of all 'Min' keys in the order they were logged,
# the last element will be the required fitness value since the last generation was logged last
best_obj_val_convergence = logbook.select('Min')[-1]

# plotting Generations vs Min to see convergence for each generation

plt.figure(figsize=(20, 10))

# using select method in logbook object to extract the argument/key as list
plt.plot(logbook.select('Generation'), logbook.select('Min'))

plt.title("Minimum values of f(x,y) Reached Through Generations",fontsize=20,fontweight='bold')
plt.xlabel("Generations",fontsize=18,fontweight='bold')
plt.ylabel("Value of Himmelblau's Function",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# the red line at lowest value of f(x,y) in the last generation or the solution at which algorithm converged
plt.axhline(y=best_obj_val_convergence,color='r',linestyle='--')

# the red line at lowest value of f(x,y)
plt.axhline(y=best_obj_val_overall,color='r',linestyle='--')

#
if best_obj_val_convergence > 2:
    k = 0.8
elif best_obj_val_convergence > 1:
    k = 0.5
elif best_obj_val_convergence > 0.5:
    k = 0.3
elif best_obj_val_convergence > 0.3:
    k = 0.2
else:
    k = 0.1

# location of both text in terms of x and y coordinate
# k is used to create height distance on y axis between the two texts for better readability

# for best_obj_val_convergence
xyz1 = (no_of_generations/2.4,best_obj_val_convergence) 
xyzz1 = (no_of_generations/2.2,best_obj_val_convergence+(k*3)) 

plt.annotate("At Convergence: %0.5f" % best_obj_val_convergence,xy=xyz1,xytext=xyzz1,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=18,fontweight='bold')

# for best_obj_val_overall
xyz2 = (no_of_generations/6,best_obj_val_overall)
xyzz2 = (no_of_generations/5.4,best_obj_val_overall+(k/0.1))

plt.annotate("Minimum Overall: %0.5f" % best_obj_val_overall,xy=xyz2,xytext=xyzz2,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=18,fontweight='bold')

plt.show()