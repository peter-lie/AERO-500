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
import random

# Parameters
N = 30
leader_speed = 0.1
follower_gain = 0.1
near_field_dist = 1.5
wing_offset = 0.8
back_offset = 1.2
leader_back_offset = 0.6  # tighter spacing behind the global leader
leader_wing_offset = 0.6  # equalize left/right spacing for global leader
dt = 0.2

# Initialize birds
birds = []
velocities = []

# Followers clustered behind the leader
for _ in range(N - 1):
    b = sphere(pos=vector(random.uniform(-2, 2),
                         random.uniform(-2, 1),
                         random.uniform(-0.2, 0.2)),
               radius=0.2, color=color.white)
    birds.append(b)
    velocities.append(vector(0, 0, 0))

# Leader at the front-center
leader = sphere(pos=vector(0, 2.0, 0), radius=0.2, color=color.cyan)
birds.insert(0, leader)
velocities.insert(0, vector(0, 0, 0))

primary_leader_index = 0

# Assign each bird a permanent left/right side
fixed_leader_x = birds[primary_leader_index].pos.x
sides = []
for i in range(N):
    if i == primary_leader_index:
        sides.append(0)
    else:
        side = np.sign(birds[i].pos.x - fixed_leader_x)
        if side == 0:
            side = random.choice([-1, 1])
        sides.append(side)

# Choose the closest same-side bird ahead, or global leader if none

def choose_leader(i):
    self_pos = birds[i].pos
    self_side = sides[i]
    min_dist = float('inf')
    leader = None
    for j in range(N):
        if j == i:
            continue
        candidate = birds[j]
        if candidate.pos.y > self_pos.y:
            if sides[j] == self_side:
                dist = mag(candidate.pos - self_pos)
                if dist < min_dist:
                    min_dist = dist
                    leader = candidate

    # If no same-side leader ahead, follow the global leader
    if leader is None:
        leader = birds[primary_leader_index]
    return leader

# Compute draft target position
def compute_draft_position(leader_pos, side, is_global_leader=False):
    offset = leader_back_offset if is_global_leader else back_offset
    lateral = leader_wing_offset if is_global_leader else wing_offset
    return leader_pos + vector(side * lateral, -offset, 0.1)

# Simulation loop
while True:
    rate(30)

    # Compute average follower Y
    follower_y = [bird.pos.y for i, bird in enumerate(birds) if i != primary_leader_index]
    avg_follower_y = sum(follower_y) / len(follower_y)

    # Compute max follower y to keep leader ahead
    max_follower_y = max(follower_y)

    for i in range(N):
        bird = birds[i]

        if i == primary_leader_index:
            # Leader slows down or stops if anyone catches up
            dy = bird.pos.y - max_follower_y
            if dy < 0.5:
                velocities[i] = vector(0, 0.0, 0)
            elif dy < 1.5:
                velocities[i] = vector(0, 0.05, 0)
            else:
                velocities[i] = vector(0, leader_speed, 0)
        else:
            leader_bird = choose_leader(i)
            if leader_bird is None:
                velocities[i] = vector(0, 0.05, 0)
            else:
                dist = mag(leader_bird.pos - bird.pos)
                if dist > near_field_dist:
                    direction = norm(leader_bird.pos - bird.pos)
                    velocities[i] = direction * 0.2
                else:
                    side = sides[i]
                    is_global = (leader_bird == birds[primary_leader_index])
                    target = compute_draft_position(leader_bird.pos, side, is_global_leader=is_global)
                    delta = target - bird.pos
                    velocities[i] = follower_gain * delta

        bird.pos += velocities[i] * dt
