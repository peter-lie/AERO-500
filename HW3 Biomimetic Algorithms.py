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
from mpl_toolkits.mplot3d import Axes3D


# Task 1 (25 points):
# Implement Ant Colony Optimization (ACO) to solve the 5 node network 
# example starting on slide 16 from class lecture notes.

# Submit your Python code and any relevant plots or other documents.
# Find and print the final optimal path your ants found and the 
# value of that path. 

import numpy as np
import random

# Distance matrix from the image
DistanceMatrix = np.array([
    [0, 10, 12, 11, 14],
    [10, 0, 13, 15, 8],
    [12, 13, 0, 9, 14],
    [11, 15, 9, 0, 16],
    [14, 8, 14, 16, 0]
])

# Initial pheromone matrix
tau = np.ones((5, 5))

# Parameters
alpha = 1.0  # Pheromone importance
beta = 2.0   # Distance importance
rho = 0.5    # Evaporation rate
Q = 100      # Pheromone deposit factor
number_ants = 5
number_iterations = 100

n = DistanceMatrix.shape[0] # Should be five cities
best_path = None
best_cost = 100

for iteration in range(number_iterations):
    all_paths = []
    all_costs = []
    
    for ant in range(number_ants):
        visited_cities = [0]  # Start at city 0
        current_city = 0

        while len(visited_cities) < n:
            probabilities = []
            for j in range(n):
                if j not in visited_cities:
                    pheromone = tau[current_city][j] ** alpha
                    visibility = (1 / DistanceMatrix[current_city][j]) ** beta
                    probabilities.append((j, pheromone * visibility))
            
            total = sum(p for _, p in probabilities)
            probabilities = [(city, p / total) for city, p in probabilities]
            r = random.random()
            cumulative = 0
            for city, prob in probabilities:
                cumulative += prob
                if r <= cumulative:
                    next_city = city
                    break
            visited_cities.append(next_city)
            current_city = next_city

        visited_cities.append(0)  # return to start
        cost = sum(DistanceMatrix[visited_cities[i]][visited_cities[i+1]] for i in range(n))
        all_paths.append(visited_cities)
        all_costs.append(cost)

        if cost < best_cost:
            best_cost = cost
            best_path = visited_cities

    # Update pheromones
    tau *= (1 - rho)
    for path, cost in zip(all_paths, all_costs):
        for i in range(n):
            r, s = path[i], path[i+1]
            tau[r][s] += Q / cost
            tau[s][r] += Q / cost  # symmetric


print("Task 1: ")
# Either finds [0, 3, 2, 4, 1, 0] or [0, 1, 4, 2, 3, 0] (backwards)
print("Best path:    ", best_path)
# Cost of 52
print("Best cost:    ", best_cost)




# Task 2 (25 points):
# Implement either the Particle Swarm Optimization (PSO) or the 
# Bees Algorithm (BA) as discussed in class.
# Use your implementation of PSO or BA to find an estimate of 
# the three dimensional Ackley' FunctionLinks to an external site.

# Submit your Python code and any relevant plots or other documents.
# Find and print the final minimal value of the Ackley Function 
# found by your algorithm 


# Ackley function definition
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum_sq_term + cos_term + a + np.e

# Particle class
class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_pos = self.position.copy()
        self.best_val = ackley(self.position)

    def update_velocity(self, global_best, w, c1, c2):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = c1 * r1 * (self.best_pos - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

    def evaluate(self):
        val = ackley(self.position)
        if val < self.best_val:
            self.best_val = val
            self.best_pos = self.position.copy()
        return val

# PSO main function
def pso(num_particles=20, dim=3, bounds=(-5, 5), max_iter=50, w=.8, c1=.1, c2=.1):
    swarm = [Particle(dim, bounds) for _ in range(num_particles)]
    global_best_pos = swarm[0].position.copy()
    global_best_val = ackley(global_best_pos)

    history = []
    initial_positions = [particle.position.copy() for particle in swarm]

    for _ in range(max_iter):
        for particle in swarm:
            particle.update_velocity(global_best_pos, w, c1, c2)
            particle.update_position(bounds)
            val = particle.evaluate()
            if val < global_best_val:
                global_best_val = val
                global_best_pos = particle.position.copy()

        history.append(global_best_val)

    final_positions = [particle.position.copy() for particle in swarm]
    return global_best_pos, global_best_val, history, initial_positions, final_positions

# Run PSO
best_pos, best_val, history, initial_positions, final_positions = pso()

# print(" ")
print("Task 2:")
print("Best Position:", best_pos) # Pretty close to [0,0,0]
print("Best Value:   ", best_val) # I think this gets to MachE in 300 interations or so

# Plot Convergence
plt.semilogy(history)
plt.xlabel("Iteration")
plt.ylabel("Best Ackley Value")
plt.title("PSO Optimization of 3D Ackley Function")
plt.grid(True)
plt.show()

# 3D scatter plot of particle positions
initial_positions = np.array(initial_positions)
final_positions = np.array(final_positions)

fig = plt.figure(figsize=(12, 6))

# Initial
ax = fig.add_subplot(projection='3d')
ax.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2], c='blue', label='Initial Particles')
ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], c='red', label='Final Particles')
ax.set_title('Particle Positions')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


