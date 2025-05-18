# Peter Lie
# AERO 500 / 470

# ICA 8

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()


from vpython import *
import numpy as np

# Constants
AU = 1  # Astronomical Unit in AU
earthOrbitRadius = 1 * AU
# moonOrbitRadius = 0.00257 * AU  # 384400 km in AU
moonOrbitRadius = 0.0257 * AU  # scaled up to see behavior


# Periods
earthOrbitalPeriod = 12  # Earth orbits the Sun in 12 months
moonOrbitalPeriod = 1    # Moon orbits Earth in 1 month

# Angular velocities
omegaEarth = 2 * np.pi / earthOrbitalPeriod
omegaMoon = 2 * np.pi / moonOrbitalPeriod

# Visual elements
scene.background = color.black
scene.width = 800
scene.height = 600
scene.title = "Sun-Earth-Moon Parametric Simulation"

# Camera settings
scene.forward = vector(0, 2, -1)  # view from above
scene.up = vector(0, 1, 0)
scene.center = vector(0, 0, 0)
scene.range = 1.5

sun = sphere(pos=vector(0,0,0), radius=0.05, color=color.yellow, emissive=True)
earth = sphere(pos=vector(earthOrbitRadius,0,0), radius=0.03, color=color.blue, make_trail=True, trail_color=color.white)
moon = sphere(pos=earth.pos + vector(moonOrbitRadius,0,0), radius=0.02, color=color.gray(0.8), make_trail=True, trail_color=color.gray(0.5))

# Time step
dt = 0.01
t = 0

# Simulation loop
while True:
    rate(100)
    t += dt

    # Earth position around Sun
    earth.pos = vector(earthOrbitRadius * np.cos(omegaEarth * t),
                       earthOrbitRadius * np.sin(omegaEarth * t), 0)

    # Moon position around Earth
    moon.pos = earth.pos + vector(moonOrbitRadius * np.cos(omegaMoon * t),
                                  moonOrbitRadius * np.sin(omegaMoon * t), 0)

