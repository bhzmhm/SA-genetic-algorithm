from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, v):
        self.v = v
        self.graph = defaultdict(list)

    def addEdge(self, s, d):
        self.graph[s].append(d)

    def countEdge(self):
        b = 0
        for i in range (self.v):
            a = len(self.graph[i])+1
            b = b + a
        return b

    def parent(self, s, d):  # check if s eating d
        for v in self.graph[s]:
            if (v == d):
                return True
        return False

lines = []
with open("input.txt") as f:
    lines = f.readlines()
# INITIAL DATA & PARAMETERS
n = int(lines[0])
lines.pop(0)
graph = Graph(n)
for l in lines:
    e = l.split(' ')
    graph.addEdge(int(e[0]), int(e[1]))

population_size = 100
max_pop_size = 200
crossover_coeff = 0.8
mutation_coeff = 0.4
max_iteration = 500
num_crossover = round(population_size*crossover_coeff)
num_mutation = round(population_size*mutation_coeff)
total = population_size+num_crossover+num_mutation
population = []
object_values = []
best_objective = 0
best_chromosome = np.zeros(n)

# OBJECTIVE FUNCTION
def objective(sol):
    e = graph.countEdge()
    for i in range(n):
        a = sol[i]
        for j in range(i,n):
            b = sol[j]
            if(graph.parent(b,a) and i!=j):
                e -= 1
    return e

def recombination(parent1, parent2):
    l = np.random.randint(n)
    r = np.random.randint(n)
    while(l==r):
        l = np.random.randint(n)
    if(l>r):
        temp = l
        l = r
        r = temp

    child1 = np.zeros(n, dtype=int)
    child2 = child1.copy()
    contain1 = child1.copy()
    contain2 = child1.copy()

    for i in range(l, r+1):
        child1[i] = parent1[i]
        contain1[parent1[i]-1] = 1

    for i in range(l, r+1):
        child2[i] = parent2[i]
        contain2[parent2[i]-1] = 1

    index_child = 0
    index_parent = 0
    while index_child < n and index_parent < n:
        if index_child >= l and index_child <= r:
            index_child = index_child + 1
            continue
        while index_parent < n and contain1[parent2[index_parent]-1]:
            index_parent = index_parent + 1
        if index_parent == n:
            break
        child1[index_child] = parent2[index_parent]
        index_child = index_child + 1
        index_parent = index_parent + 1

    index_child = 0
    index_parent = 0
    while index_child < n and index_parent < n:
        if index_child >= l and index_child <= r:
            index_child = index_child + 1
            continue
        while index_parent < n and contain2[parent1[index_parent]-1]:
            index_parent = index_parent + 1
        if index_parent == n:
            break
        child2[index_child] = parent1[index_parent]
        index_child = index_child + 1
        index_parent = index_parent + 1

    return (child1, child2)


# initial population
while len(population) < population_size:
    sequence = [i for i in range(1, n + 1)]
    temp = random.sample(sequence, n)
    population.append(temp)
    object_values.append(objective(temp))

# main loop of genetic algorithm
iteration = 0
pl = []
while iteration < max_iteration:
    summation = sum(object_values)
    pr = []
    cumulative_pr = []
    for i in range(population_size):
        pr.append(object_values[i] / summation)
    cumulative_pr.append(pr[0])
    for i in range(1, population_size - 1):
        temp = cumulative_pr[i - 1] + pr[i]
        cumulative_pr.append(temp)
    cumulative_pr.append(1)
    for i in range(0, num_crossover, 2):
        p1 = 0
        temp = np.random.rand()
        while cumulative_pr[p1] < temp:
            p1 = p1 + 1
        p2 = p1
        while p1 == p2:
            temp = np.random.rand()
            p = 0
            while cumulative_pr[p] < temp:
                p = p + 1
            p2 = p
        parent1 = population[p1]
        parent2 = population[p2]
        children = recombination(parent1, parent2)
        child1 = children[0]
        child2 = children[1]
        population.append(child1)
        object_values.append(objective(child1))
        population.append(child2)
        object_values.append(objective(child2))
    # mutation
    for i in range(num_mutation):
        temp = np.random.randint(num_crossover)
        temp = temp + population_size
        mutated = population[temp]
        temp1 = np.random.randint(n)
        temp2 = np.random.randint(n)
        while ((graph.parent(mutated[temp1], mutated[temp2]) and (temp1 < temp2))):
            temp1 = np.random.randint(n)
            temp2 = np.random.randint(n)
        temp_data = mutated[temp1]
        mutated[temp1] = mutated[temp2]
        mutated[temp2] = temp_data
        population.append(mutated)
        object_values.append(objective(mutated))
    # update best solution
    best_objective = max(object_values)
    best_arg = np.argmax(object_values)
    best_chromosome = population[best_arg]
    # keep best chromosomes
    if len(population) > max_pop_size:
        temp_population = []
        temp_objective = []
        args = np.argsort(object_values)
        for i in range(max_pop_size):
            temp = len(population) - 1 - i
            temp_population.append(population[args[temp]])
            temp_objective.append(object_values[args[temp]])

        population = temp_population
        object_values = temp_objective
        population_size = max_pop_size
    # print(best_objective)
    iteration = iteration + 1
    # print('graph : ',graph.countEdge()-1)
    pl.append(best_objective)
    if(graph.countEdge() - 1 == best_objective):
        break
print(best_chromosome)
print(best_objective)
# print(pl)
plt.plot(pl)
plt.show()


