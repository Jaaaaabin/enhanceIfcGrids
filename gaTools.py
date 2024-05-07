import os
import random
from deap import tools

def clearPopulationFitnesses(file_path):

    if os.path.exists(file_path):
        os.remove(file_path)

def savePopulationFitnesses(file_path, values):

    try:
        # Open the file in append mode
        with open(file_path, 'a') as file:
            for v in values:
                file.write(f"{v}\n")

    except Exception as e:
        print(f"Failed to append to the file: {e}")

# DEAP - varAnd
def varAnd(population, toolbox, cxpb, mutpb):

    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring

# DEAP - eaSimple [adjusted.]
def ga_eaSimple(population, toolbox, cxpb, mutpb, ngen, fitness_file=[], stats=None,
             halloffame=None, verbose=__debug__):
    
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  * 
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
    if fitness_file:
        clearPopulationFitnesses(file_path=fitness_file)
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  * 
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
    # Collect the fitness of per population.
    population_fitnesses = [ind.fitness.values[0] for ind in population]
    savePopulationFitnesses(file_path=fitness_file,values=population_fitnesses)
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        
        # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
        # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
        # Collect the fitness of per population.
        population_fitnesses = [ind.fitness.values[0] for ind in population]
        savePopulationFitnesses(file_path=fitness_file,values=population_fitnesses)
        # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
        # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}

        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

    return population, logbook