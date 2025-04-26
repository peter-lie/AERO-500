# Peter Lie
# AERO 500 / 470

# Homework 1: Stochastic Methods

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()

# Task 1:
# Use the Monte Carlo Integration Method to estimate the value of pi
# Submit your Python code and a plot of your results
# Approximately how many iterations does your code take to converge to a solution?
# How many trials did your simulation need to converge within 1% of the known value of pi?

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm 


# Parameters
num_points = 10000
inside_circle = 0
pi_estimates = []

# Monte Carlo simulation
for i in range(1, num_points):
    x, y = random.random(), random.random()  # Point in unit square
    if x**2 + y**2 <= 1:
        inside_circle += 1
    pi_estimate = 4 * inside_circle / i
    pi_estimates.append(pi_estimate)

# print(f"Final estimate of π after {num_points} iterations: {pi_estimates[-1]}")

# Yes, it does appear to converge

# Plotting the convergence
plt.figure(figsize=(10, 6))
plt.plot(pi_estimates, label='Estimated ${\\pi}$')
plt.axhline(y=np.pi, color='r', linestyle='--', label='Actual ${\\pi}$')
plt.xlabel('Number of Points')
plt.ylabel('Estimated ${\\pi}$')
plt.title('Monte Carlo Estimation of ${\\pi}$')
# plt.legend()
plt.grid(True)
plt.show()


# Clear terminal
import os
clear = lambda: os.system('clear')
clear()
# Not sure why it's reporting this up to here, but this works

print("Task 1:")
# Convergence within 1 percent of pi
within_1_percent = [i for i, est in enumerate(pi_estimates) if abs(est - np.pi) / np.pi < 0.01]
if within_1_percent:
    print(f"Estimate converged within 1% after approximately {within_1_percent[0]} iterations.")
else:
    print("${\\pi}$ estimate did not converge within one percent in given trials.")

print(f"Approximately converges to π after 1000 iterations ")

print(f"Final estimate of π after {num_points} iterations: {pi_estimates[-1]}")




# Task 2 (10 points):

# Use the Monte Carlo Integration Method to integrate the function x^3 sin(x)
# All positive on -2 < x < 3

import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x**3 * np.sin(x)

# Integration limits
a, b = -2, 3
N = 10000
integral_estimate = []
under_area = 0

# x_random = np.random.uniform(a, b, N)
# y_val = f(x_random)

for i in range(1, N):
    x, y = random.uniform(a,b), random.uniform(0,10)  # Point in unit square
    ytrue = f(x)
    if y <= ytrue:
        under_area += 1
    integral_area = 50 * under_area / i
    integral_estimate.append(integral_area)


# Number of trials for plotting
trials = np.arange(1, N)

rand_x = np.random.uniform(a,b,N)
f_val = f(rand_x)
est = (b - a) * np.mean(f_val)
truth, _ = quad(f, a, b)
err = abs((est - truth)/truth) * 100

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(trials, integral_estimate, label="Monte Carlo Approximation")
plt.axhline(y=truth, color='r', linestyle='--', label="Reference Line")
plt.xlabel("Number of Samples (N)")
plt.ylabel("Approximate Integral Value")
plt.title("Monte Carlo Integration of $x^3 \\sin(x)$ over $[-2,3]$")
# plt.legend()
plt.grid(True)
plt.show()


print(" ")
print("Task 2:")

print("Truth: ", truth)
print(f"MVT Estimation: {est}")
print(f"Approximately converges after 2000 iterations")

print(f"MVT Error: {err} %")



# Task 3

Rair = 287.053 # J / kg K
T = 15 # C
TKelv = 15 + 273.15 # K
rho = 1.225 # kg / m^3

numTrials = 10000

rho_est = []

for i in range(numTrials):
    T_c = np.random.normal(25, 0.2)
    T_k = T_c + 273.15
    P = np.random.normal(104847, 52)
    rho_est.append(P/(Rair*T_k)) 

rho_mean = np.mean(rho_est)
rho_std = np.std(rho_est)

rho_mean_al = 104847 / (TKelv * Rair)

T_mean_K = 25 + 273.15
d_rho_dP = 1/(Rair*T_mean_K)
d_rho_dT = -104847/(Rair*T_mean_K**2)
rho_std_al = np.sqrt((d_rho_dP * 52)**2 + (d_rho_dT * 0.2)**2)


pressure_error = np.linspace(-800, 800, 33)  
rho_stds = []

for P_std in pressure_error:
    rho_est = []
    for _ in range(numTrials):
        T_c = np.random.normal(25, 0.2)
        T_k = T_c + 273.15
        P = np.random.normal(104847, abs(P_std))
        rho_est.append(P /(Rair*T_k))

    rho_std = np.std(rho_est)
    rho_stds.append(rho_std)

# Plot sensitivity curve
plt.figure(figsize=(10, 6))
plt.plot(pressure_error, rho_stds, marker='o')
plt.title("Sensitivity of Density Error to Pressure Uncertainty")
plt.xlabel("Pressure Error")
plt.ylabel("Density Standard Deviation")
plt.grid(True)
plt.show()


print(" ")
print("Task 3:")

print(f"Analytical: ρ = {rho_mean_al:.5f} ± {rho_std_al:.5f} kg/m^3")
print(f"Monte Carlo Result: ρ = {rho_mean:.5f} ± {rho_std:.5f} kg/m^3")



# Task 4

N = 10000

Unit_Price = np.random.triangular(50, 55, 70, N)
Unit_Sales = np.random.triangular(2000, 2440, 3000, N)
Variable_Costs = np.random.triangular(50000, 55200, 65000, N)
Fixed_Costs = np.random.triangular(10000, 14000, 20000, N)

# Earnings equation given in book
Earnings = Unit_Price * Unit_Sales - (Variable_Costs + Fixed_Costs)

# Earnings parameters
mean_Earnings = np.mean(Earnings)
median_Earnings = np.median(Earnings)
std_Earnings = np.std(Earnings)
var_Earnings = np.var(Earnings)
min_Earnings = np.min(Earnings)
max_Earnings = np.max(Earnings)


# Confidence Intervals
z_alpha2 = norm.ppf(1 - 0.05 / 2)  # 95% confidence - two tailed
ci_lower = mean_Earnings - z_alpha2 * std_Earnings/np.sqrt(N)
ci_upper = mean_Earnings + z_alpha2 * std_Earnings/np.sqrt(N)

# Sensitivity Analysis 
sigma_Unit_Price = np.std(Unit_Price)
sigma_Unit_Sales = np.std(Unit_Sales)
sigma_Variable_Costs = np.std(Variable_Costs)
sigma_Fixed_Costs = np.std(Fixed_Costs)
combined_Sales_Price = Unit_Price * Unit_Sales
sigma_Combined = np.std(combined_Sales_Price, ddof=1)


fig, ax = plt.subplots(3, 1, figsize=(10, 6))

# plt.title("Sensitivity Analysis")

ax[0].plot(Variable_Costs, Earnings, '.')
# ax[0].title("Earnings vs Variable Costs")
ax[0].set_xlabel("Variable Costs")
ax[0].set_ylabel("Earnings")
ax[0].grid(True)

ax[1].plot(Fixed_Costs, Earnings, '.')
# ax[1].title("Earnings vs Fixed Costs")
ax[1].set_xlabel("Fixed Costs")
ax[1].set_ylabel("Earnings")
ax[1].grid(True)

ax[2].plot(Unit_Price * Unit_Sales, Earnings, '.')
# ax[2].title("Earnings vs Unit Price x Sales")
ax[2].set_xlabel("Unit Price x Unit Sales")
ax[2].set_ylabel("Earnings")
ax[2].grid(True)

plt.show()


# Talked with Kevin about how to use this
def partial_deriv(Y, X):
    # use covariance over variance
    return np.cov(Y, X)[0, 1] / np.var(X)

dY_dVcosts = partial_deriv(Earnings, Variable_Costs)
dY_dFcosts = partial_deriv(Earnings, Fixed_Costs)
dY_dCombined = partial_deriv(Earnings, combined_Sales_Price)

S_vcosts = (sigma_Variable_Costs/std_Earnings) * dY_dVcosts
S_fcosts = (sigma_Fixed_Costs/std_Earnings) * dY_dFcosts
S_combined = (sigma_Combined/std_Earnings) * dY_dCombined

print(" ")
print("Task 4:")

# Print summary
print(f"Monte Carlo Simulation Summary {N} runs:")
print(f"Mean Earnings:      {mean_Earnings:,.0f}")
print(f"Median Earnings:    {median_Earnings:,.0f}")
print(f"Standard Deviation: {std_Earnings:,.0f}")
print(f"Variance:           {var_Earnings:,.0f}")
print(f"Min Earnings:       {min_Earnings:,.0f}")
print(f"Max Earnings:       {max_Earnings:,.0f}")
print(f"95% confidence interval is from: {ci_lower} to {ci_upper}")
print(f"Variable Costs: {S_vcosts**2}")
print(f"Fixed Costs: {S_fcosts**2}")
print(f"Unit Sales x Unit Price: {S_combined**2}")

# The sigma normalized derivatives agree with what the book has
# .03 for variable, .01 for fixed, and .96 for combined

# The book confidence interval goes from 75,223 to 75,869, pretty close

