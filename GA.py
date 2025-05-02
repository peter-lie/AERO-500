import numpy as np
import random

class Chromosome:

    def __init__(self, numGenes = 20):
        self.genes = random.choices([0, 1], k = numGenes)
        self.fitness = 0

    def mutate(self):
        # Implement random mutation
        idx = random.randint(0, len(self.genes) - 1)
        self.genes[idx] = 1 - self.genes[idx]  # Flip 0<->1

    def __add__(self, o):
        # Implement single point crossover with random crossover point
                # Single point crossover
        point = random.randint(1, len(self.genes) - 1)
        child = Chromosome(numGenes=len(self.genes))
        child.genes = self.genes[:point] + o.genes[point:]
        return child
    

class Population:

    def __init__(self, populationSize, numGenes = 20):
        self.members = [Chromosome(numGenes) for i in range(populationSize)]

    def selection(self, ratio):
        # Implement Selection
        # Step 1 - Sort members by fitness
        self.members.sort(key=lambda x:x.fitness)
        # Step 2 - return some number of members based on the ratio provided
        top_10_percent = max(1, len(self.members) // 10)
        top_pool = self.members[:top_10_percent]

        # Randomly select two parents from the best 10%
        parent1 = random.choice(top_pool)
        parent2 = random.choice(top_pool)
        return parent1, parent2

def myFitnessFunction(chrom: Chromosome):
    # def myFitnessFunction(chrom):
    values = [23, 21, 8, 1, 3, 7, 18, 19, 17, 15, 24, 22, 6, 28, 4, 2, 27, 20, 5, 10]
    weights = [7, 2, 6, 9, 1, 5, 6, 1, 3, 4, 7, 9, 3, 7, 3, 4, 5, 1, 5, 4]
    weight_limit = 45

    gene = chrom.getBinary()
    total_weight = sum([g * w for g, w in zip(gene, weights)])
    total_value = sum([g * v for g, v in zip(gene, values)])

    if total_weight > weight_limit:
        chrom.fitness = 0  # Penalize
    else:
        chrom.fitness = total_value

