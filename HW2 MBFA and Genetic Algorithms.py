# Peter Lie
# AERO 500 / 470

# Homework 2: MBFA and Genetic Algorithms

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()

# Imports
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm 


# Knapsack data:


# You have 20 items with the following value/mass properties:
values = [23, 21, 8, 1, 3, 7, 18, 19, 17, 15, 24, 22, 6, 28, 4, 2, 27, 20, 5, 10] # dollars
weights = [7, 2, 6, 9, 1, 5, 6, 1, 3, 4, 7, 9, 3, 7, 3, 4, 5, 1, 5, 4] # kg
# The total weight of the knapsack is limited to 45 kg
weight_limit = 45 # kg


# Task 1:
# Code - Develop a Modified Brute-Force Algorithm (MBFA) in Python.
# Results - Use you MBFA to solve the KNAPSACK problem with values 
# and weights given below.
# Submit your code and a list of the three best solutions you found.


# Given values and weights

# Modified brute-force parameters
num_trials = 100000  # You can increase this to 1 million for better results
n_items = len(values)

# Store best solutions
best_solutions = []

for i in range(num_trials):
    # Randomly generate a binary selection vector
    selection = np.random.randint(0, 2, size = n_items)
    
    total_weight = np.dot(selection, weights)
    total_value = np.dot(selection, values)
    
    if total_weight <= weight_limit:
        best_solutions.append((total_value, total_weight, selection.copy()))

# Sort by value, descending
best_solutions.sort(reverse=True, key=lambda x: x[0])

# Show top 3 solutions
print("Top 3 MBFA Solutions:")
for i in range(3):
    value, weight, combo = best_solutions[i]
    print(f"Solution {i+1}: Value = {value}, Weight = {weight}, Items = {combo}")

# Seems to sometimes find 213, often at 210 or 207 with 100,000 points
# For faster results, could use only selections that have at least 6 entries (least required to get to max weight)
# Also selections that have less than 13 entries (most required under max weight)



# Task 2:

# Code - Develop a simple Genetic Algorithm (GA) in Python using the 
# template supplied.
# Results - Use your GA to solve the KNAPSACK problem with values and 
# weights given below.
# Graph - Submit your code and a graph of the average population and 
# best member fitness versus trial for 50 generations.
# Discussion - Run your code several times and comment on how the GA 
# performs on different runs, and for runs with different numbers of 
# generations.
# To use the GA module, you simply need to have the two Python files 
# in the same folder.  You DO NOT need to use pip.  

# Template Code:  genentic_alg.py,  Download genentic_alg.py,GA.py



import GA
# Step 1 - Initialize Population
myPop = GA.Population(populationSize = 20, numGenes = 20)
for c in myPop.members:
    GA.myFitnessFunction(c)


# Step 2 - Selection
# Supose we selected Chrom 1 and 5


# Step 3 - Crossover and Mutate
# Crossover will generate new Chromosomes and will look like:
# newChrom = myPop[1] + myPop[5]


# Mutation will look like this, but only a small percent of the
# population Chromosomes will be mutated:
# newChrom.mutate()


# Step 4 - repeat




# Fitness function
def knapsack_fitness(Chromosome):
    gene = GA.Chromosome.__init__()  # array of 0s and 1s
    total_value = np.dot(gene, values)
    total_weight = np.dot(gene, weights)
    
    # Penalize overweight
    if total_weight > weight_limit:
        fitness = 0
    else:
        fitness = total_value
    
    Chromosome.fitness = fitness
    return fitness



def myFitnessFunction(chrom):
    values = [23, 21, 8, 1, 3, 7, 18, 19, 17, 15, 24, 22, 6, 28, 4, 2, 27, 20, 5, 10]
    weights = [7, 2, 6, 9, 1, 5, 6, 1, 3, 4, 7, 9, 3, 7, 3, 4, 5, 1, 5, 4]
    weight_limit = 45

    gene = chrom.getBinary()
    total_weight = sum([g * w for g, w in zip(gene, weights)])
    total_value = sum([g * v for g, v in zip(gene, values)])

    if total_weight > weight_limit:
        chrom.fitness = 0  # Penalize
    else:
        chrom.fitness = total_value


# GA settings
population_size = 20
generations = 50
mutation_rate = 0.1  # 10% of children mutate


# Initialize population
myPop = GA.Population(population_size, numGenes=20)
for chrom in myPop.members:
    knapsack_fitness(chrom)


# Track fitness over time
avg_fitness_list = []
best_fitness_list = []

# Run GA
for g in range(generations):
    new_members = []

    # Record fitness stats
    fitnesses = [c.fitness for c in myPop.members]
    avg_fitness_list.append(np.mean(fitnesses))
    best_fitness_list.append(np.max(fitnesses))

    # Create new generation
    for _ in range(population_size // 2):
        p1, p2 = myPop.selectParents()
        child1 = p1 + p2
        child2 = p2 + p1
        
        # Mutate with small chance
        if np.random.rand() < mutation_rate:
            child1.mutate()
        if np.random.rand() < mutation_rate:
            child2.mutate()
        
        # Evaluate fitness
        knapsack_fitness(child1)
        knapsack_fitness(child2)

        new_members.extend([child1, child2])

    # Replace population
    myPop.members = new_members

# Final best solution
best = max(myPop.members, key=lambda c: c.fitness)
print("Best GA solution:")
print("Fitness:", best.fitness)
print("Selected items:", best.getBinary())
print("Total weight:", np.dot(best.getBinary(), weights))

# Plot fitness evolution
plt.plot(avg_fitness_list, label="Average Fitness")
plt.plot(best_fitness_list, label="Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("GA Fitness Evolution for Knapsack")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






