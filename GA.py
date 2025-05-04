import numpy as np
import random

values = [23, 21, 8, 1, 3, 7, 18, 19, 17, 15, 24, 22, 6, 28, 4, 2, 27, 20, 5, 10]
weights = [7, 2, 6, 9, 1, 5, 6, 1, 3, 4, 7, 9, 3, 7, 3, 4, 5, 1, 5, 4]
max_weight = 45
mutation_rate = 0.01

class Chromosome:
    def __init__(self, numGenes=20):
        self.genes = random.choices([0, 1], k=numGenes)
        self.fitness = 0

    def mutate(self):
        # Implement random mutation
        self.genes = [1 - bit if random.random() < mutation_rate else bit for bit in self.genes]

    def __add__(self, o):
        # Implement single point crossover with random crossover point
        point = random.randint(1, len(self.genes) - 1)
        new_genes = self.genes[:point] + o.genes[point:]
        return Chromosome(len(new_genes))
    
        # point = random.randint(1, len(values) - 1)
        # return p1[point] + p2[point]


class Population:
    def __init__(self, populationSize, numGenes=20):
        self.members = [Chromosome(numGenes) for i in range(populationSize)]


    def selection(self, ratio):
        # Implement Selection
        # Step 1 - Sort members by fitness
        self.members.sort(key=lambda x: x.fitness, reverse=True)
        # Step 2 - return some number of members based on the ratio provided
        count = max(1, int(len(self.members) * ratio))
        return self.members[:count]


def myFitnessFunction(chrom: Chromosome):
    # chrom.fitness = some function of chrom.genes
    total_value = sum(v for v, bit in zip(values, chrom.genes) if bit)
    total_weight = sum(w for w, bit in zip(weights, chrom.genes) if bit)
    chrom.fitness = total_value if total_weight <= max_weight else 0

