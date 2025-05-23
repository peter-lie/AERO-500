# Peter Lie
# AERO 500 / 470

# Homework 5: Fast food Take out Queue

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()

import numpy as np
import DESQueuingTemplate as DES

# np.random.seed(0)

sim = DES.Simulation(T=float('inf'))

for _ in range(100):
    sim.advance_time()

print(f"Arrivals: {sim.N_arrivals}")
print(f"Departures: {sim.N_departs}")
print(f"Total Wait Time: {sim.total_wait:.2f}")
print(f"Average Wait per Customer: {sim.total_wait / sim.N_departs:.2f}")


