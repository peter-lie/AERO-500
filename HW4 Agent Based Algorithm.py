# Peter Lie
# AERO 500 / 470

# Homework 4: Agent - Based Algorithms
# VPython Simulation

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()

from vpython import *
import numpy as np

# Parameters
N = 45
a = 1
gamma = 0.05
xOffset, yOffset = 0.05, 0.05
Dnom = np.sqrt(2*xOffset**2 + (xOffset*(1.1 + np.pi/4))**2)

# Randomly generate 2D position
x_bar_k = np.zeros((2, N))
sides = np.zeros(N)
for i in range(N):
    if i == 0:
        x_k = 0
        y_k = 10
        side = 0
    else:
        # These ensure even split - same amount of agents on either side
        # side = -1 if i % 2 == 0 else 1
        # x_k = side * np.random.uniform(0, 10) # even V
        x_k = np.random.uniform(-10, 10) # uneven V
        side = -1 if x_k < 0 else 1 # keep on same side from primary leader
        y_k = np.random.uniform(-10, 10)
    x_bar_k[:, i] = [x_k, y_k]
    sides[i] = side

# Sort by y (descending) and identify leader as highest y value
sort_indices = np.argsort(-x_bar_k[1, :])
x_bar_k = x_bar_k[:, sort_indices]
sides = sides[sort_indices.astype(int)]
front_leader_ixOffset = 0

# Rendering Setup
scene = canvas(title="Goose V-Formation", width=1000, height=600)
scene.range = 30
bodies = []
for i in range(N):
    color_i = color.blue if i == front_leader_ixOffset else color.white
    boyOffset = sphere(pos=vector(*x_bar_k[:, i], 0), radius=0.3, make_trail=True, color=color_i, retain=100)
    bodies.append(boyOffset)

# Functions
def update_pos(i, x_bar, leader_pos, front_leader_pos, side):
    xi = x_bar[:, i]
    delta_r = leader_pos - xi
    dist = np.linalg.norm(delta_r)

    # far field
    if dist > Dnom:
        delta = gamma * delta_r
        F = np.zeros(4)
        if side == -1:
            F[0] = 1  # move left
        elif side == 1:
            F[1] = 1  # move right
        else:
            F[np.random.choice([0, 1])] = 1

        F[2] = 1  # always move back
        if np.isclose(xi[0], leader_pos[0], atol=0.5) or np.isclose(xi[1], leader_pos[1], atol=0.5):
            F[3] = 1

        B = np.array([
            [-xOffset if F[0] else 0, xOffset if F[1] else 0, 0, 0],
            [0, 0, -yOffset if F[2] else 0, -yOffset if F[3] else 0]
        ])
        delta = delta + B @ F

    # near field
    else:
        if side != 0:
            offset = np.array([side * 0.75 * a, 0.75 * a])
        else:
            offset = np.array([0, 0.75 * a])
        delta = gamma * (delta_r - offset)

    return delta


def get_relative_leader(i, x_bar, side):
    xi = x_bar[:, i]
    min_dist = np.inf
    rel_leader = None
    for j in range(N):
        if j == i:
            continue
        xj = x_bar[:, j]
        if xj[1] > xi[1]:
            # If front leader, allow it
            if j == front_leader_ixOffset:
                dist = np.linalg.norm(xj - xi)
                if dist < min_dist:
                    min_dist = dist
                    rel_leader = j
            elif np.sign(xj[0] - x_bar_k[0, front_leader_ixOffset]) == side:
                dist = np.linalg.norm(xj - xi)
                if dist < min_dist:
                    min_dist = dist
                    rel_leader = j
    return rel_leader

# Simulation Loop
while True:
    rate(40)
    front_leader_pos = x_bar_k[:, front_leader_ixOffset]
    for i in range(N):
        if i == front_leader_ixOffset:
            continue

        rel_leader_ixOffset = get_relative_leader(i, x_bar_k, sides[i])
        if rel_leader_ixOffset is None:
            continue
        leader_pos = x_bar_k[:, rel_leader_ixOffset].copy()
        update = update_pos(i, x_bar_k, leader_pos, front_leader_pos, sides[i])
        x_bar_k[:, i] += update

    for i in range(N):
        bodies[i].pos = vector(*x_bar_k[:, i], 0)

# V's are uneven because birds only look at local leaders - 
# meaning that it depends on where birds are initially with 
# respect to the global leader

# V's even if birds are evenly split across

