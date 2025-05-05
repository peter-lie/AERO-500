# Peter Lie
# AERO 500 / 470

# Homework 3: Biomimetic Algorithms

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



# Task 1 (25 points):
# Implement Ant Colony Optimization (ACO) to solve the 5 node network 
# example starting on slide 16 from class lecture notes.

# Submit your Python code and any relevant plots or other documents.
# Find and print the final optimal path your ants found and the 
# value of that path. 

# Step 1:

# Distance Matrix
# d(r,s)
distanceMatrix = [[0,  10, 12, 11, 14],
                  [10, 0,  13, 15, 8 ],
                  [12, 13, 0,  9,  14], 
                  [11, 15, 9,  0,  16], 
                  [14, 8,  14, 16, 0 ]]

# Initial Pheromone Matrix
# tau(r,s)
tau = np.ones((5,5))


# Step 2

# Visibility Matrix
# eta(r,s)
visibilityMatrix = [[0,    1/10, 1/12, 1/11, 1/14],
                    [1/10, 0,    1/13, 1/15, 1/8 ],
                    [1/12, 1/13, 0,    1/9,  1/14], 
                    [1/11, 1/15, 1/9,  0,    1/16], 
                    [1/14, 1/8,  1/14, 1/16, 0   ]]





# Task 2 (25 points):
# Implement either the Particle Swarm Optimization (PSO) or the 
# Bees Algorithm (BA) as discussed in class.
# Use your implementation of PSO or BA to find an estimate of 
# the three dimensional Ackley' FunctionLinks to an external site.

# Submit your Python code and any relevant plots or other documents.
# Find and print the final minimal value of the Ackley Function 
# found by your algorithm 



