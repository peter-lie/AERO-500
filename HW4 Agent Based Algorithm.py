# Peter Lie
# AERO 500 / 470

# Homework 4: Agent Based Algorithms

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()

# Imports
from vpython import *
import random
import numpy as np

# Simulation parameters
N = 30
dt = 0.5
gamma = 0.05
dnorm = 1.5
wing_offset = 0.5
back_offset = 1.0

# Initialize birds and velocities
birds = []
velocities = []
for _ in range(N):
    bird = sphere(pos=vector(random.uniform(-5, 5),
                             random.uniform(-5, 5),
                             random.uniform(0, 2)),
                  radius=0.2,
                  color=color.white)
    birds.append(bird)
    velocities.append(vector(0, 0, 0))

# Fix primary leader at start
primary_leader_index = max(range(N), key=lambda i: birds[i].pos.y)
primary_leader = birds[primary_leader_index]
primary_leader.color = color.cyan

# FIXED: Capture leader's initial X at side-assignment time
fixed_leader_x = primary_leader.pos.x

# Assign fixed side for each bird at t=0
sides = []
for i in range(N):
    if i == primary_leader_index:
        sides.append(0)
    else:
        side = np.sign(birds[i].pos.x - fixed_leader_x)
        if side == 0:
            side = random.choice([-1, 1])
        sides.append(side)

# Choose nearest leader in front (greater y)
def choose_leader(i):
    self_pos = birds[i].pos
    min_dist = float('inf')
    leader = None
    for j in range(N):
        candidate = birds[j]
        if candidate.pos.y > self_pos.y:
            dist = mag(candidate.pos - self_pos)
            if dist < min_dist:
                min_dist = dist
                leader = candidate
    return leader

# Compute draft position
def compute_draft_position(leader_pos, side):
    return leader_pos + vector(side * wing_offset, -back_offset, 0.1)

# Simulation loop
while True:
    rate(30)
    for i in range(N):
        bird = birds[i]
        if i == primary_leader_index:
            velocities[i] = vector(0, 0.1, 0)
        else:
            leader = choose_leader(i)
            if leader is None:
                velocities[i] = vector(0, 0.05, 0)
            else:
                dist = mag(leader.pos - bird.pos)
                if dist > dnorm:
                    velocities[i] = norm(leader.pos - bird.pos) * 0.2
                else:
                    side = sides[i]
                    draft_pos = compute_draft_position(leader.pos, side)
                    delta_s = draft_pos - bird.pos
                    velocities[i] = gamma * delta_s

        bird.pos += velocities[i] * dt