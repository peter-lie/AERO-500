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


values = [23, 21, 8, 1, 3, 7, 18, 19, 17, 15, 24, 22, 6, 28, 4, 2, 27, 20, 5, 10]
weights = [7, 2, 6, 9, 1, 5, 6, 1, 3, 4, 7, 9, 3, 7, 3, 4, 5, 1, 5, 4]
max_weight = 45

# Parameters
generations = 50
pop_size = 50

import GA

# Initialize Population
myPop = GA.Population(populationSize=pop_size, numGenes=20)
for c in myPop.members:
    GA.myFitnessFunction(c)

best_fitness_history = []
avg_fitness_history = []

# Run Genetic Algorithm
top_ratio = 0.1

for gen in range(generations):
    # Evaluate fitness
    for c in myPop.members:
        GA.myFitnessFunction(c)

    # Track stats
    fitness_scores = [c.fitness for c in myPop.members]
    best_fitness_history.append(max(fitness_scores))
    avg_fitness_history.append(sum(fitness_scores) / len(fitness_scores))

    # Selection
    parent_pool = myPop.selection(ratio=top_ratio)

    # Generate new population
    # new_members = []
    # new_members = GA.Population(populationSize=1, numGenes=20)
    new_members = parent_pool
    while len(new_members) < 10 * pop_size:
        p1 = random.choice(parent_pool)
        p2 = random.choice(parent_pool)
        child = p1 + p2
        if random.random() < .01:
            child.mutate()
        GA.myFitnessFunction(child)
        new_members.append(child)

    myPop.members = new_members
    myPop.members = myPop.selection(ratio=.1) # changing this ratio has a large effect of the average fitness


print("Top GA Solution")

print(f"Solution 1: Value = {best_fitness_history[-1]}")

# Plot fitness over generations
plt.plot(best_fitness_history, label='Best Fitness')
plt.plot(avg_fitness_history, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('GA: Best and Average Fitness Over Generations')
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.show()

# Discussion:
# The GA always increases its best fitness, but seems to plateau its average
# fitness depending on how many children there are. This suggests there 
# is something just a tiny bit off, but still finds good solutions in general
# For more generations, the GA can continue to get a little closer but still
# plateaus for long periods of time waiting to get lucky with better children



# Task 3
# Compare and contrast the two solution methods, MBFA and GA.
# Comment on how each method converges on a solution versus number 
# of trials/generations.  Also, provide your thoughts on why one 
# method may work better than the other for solving this problem.


# MBFA takes 100000 interations to converge to a value somewhere from 207-215 (rare)
# Seems to sometimes find 213, often at 210 or 207 with 100,000 points
# For faster results, could use only selections that have at least 6 entries (least required to get to max weight), 
# or selections that have less than 13 entries (most required under max weight)

# GA uses 200 generations with 100 population to reach similar numbers
# There is still some variance with what number it reaches, but the average
# fitness values are much higher than the averge for the MBFA

# Generally, GA gets a better solution than MBFA. MBFA really only works consistently
# once you reach a critical number or trials, which is closer to a million 
# than one hundred thousand. The MBFA will usaually need to get lucky, whereas 
# the GA consistently gets better. The GA will plateau at a certain point based 
# on the input parameters, but can be made to reach 215



