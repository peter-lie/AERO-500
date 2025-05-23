import numpy as np

class Simulation:
    def __init__(self, T):
        # System States
        self.N = 0  # number of customers in system

        # Simulation Variables
        self.clock = 0.0
        self.T = T
        self.t_arrival = self.generate_arrival()
        self.t_depart = T  # initially no one to depart

        # Statistical Counters
        self.N_arrivals = 0
        self.N_departs = 0
        self.total_wait = 0.0

    def advance_time(self):
        t_event = min(self.t_arrival, self.t_depart)
        # Update total wait time
        self.total_wait += self.N * (t_event - self.clock)
        self.clock = t_event

        if self.t_arrival <= self.t_depart:
            self.handle_arrival()
        else:
            self.handle_departure()

    def handle_arrival(self):
        self.N += 1
        self.N_arrivals += 1

        self.t_arrival = self.clock + self.generate_arrival()

        if self.N == 1:
            self.t_depart = self.clock + self.generate_service()

        if self.clock > self.T:
            self.t_arrival = self.T  # Turn off arrivals

    def handle_departure(self):
        self.N -= 1
        self.N_departs += 1

        if self.N > 0:
            self.t_depart = self.clock + self.generate_service()
        else:
            self.t_depart = self.T  # Turn off departures

    def generate_arrival(self):
        return np.random.exponential(1.0 / 3.0)  # Arrival rate: 3 per unit time

    def generate_service(self):
        return np.random.exponential(1.0 / 4.0)  # Service rate: 4 per unit time
    
    