from collections import defaultdict
import numpy as np
import random
import math


class Graph:
    def __init__(self, v):
        self.v = v
        self.graph = defaultdict(list)

    def addEdge(self, s, d):
        self.graph[s].append(d)

    def parent(self, s, d):  # check if s eating d
        for v in self.graph[s]:
            if (v == d):
                return True
        return False


# OBJECTIVE FUNCTION
def objective(sol):
    q = 0
    for i in range(n):
        a = sol[i]
        for j in range(i, n):
            b = sol[j]
            if (graph.parent(b, a)):
                q = q + 1
    return q


# TXT
lines = []
with open("input.txt") as f:
    lines = f.readlines()

# INITIAL DATA & PARAMETERS
n = int(lines[0])
lines.pop(0)
T = 100000
t_change = 0.99

graph = Graph(n)
for l in lines:
    e = l.split(' ')
    graph.addEdge(int(e[0]), int(e[1]))

# INITIAL SOLUTION
sequence = [i for i in range(1, n + 1)]
solution = random.sample(sequence, n)
fitness = objective(solution)

# MAIN LOOP OF SA ALGORITHM
while T > 0:
    neighbor = solution.copy()
    temp = np.random.randint(n)
    temp2 = np.random.randint(n)

    while ((graph.parent(neighbor[temp], neighbor[temp2]) and (temp < temp2))):
        temp = np.random.randint(n)
        temp2 = np.random.randint(n)

    temp_data = neighbor[temp]
    neighbor[temp] = neighbor[temp2]
    neighbor[temp2] = temp_data

    fit = objective(neighbor)

    delta = fitness - fit
    if delta >= 0:
        solution = neighbor
        fitness = fit
    else:
        pr = math.exp(delta / T)
        if pr >= .95:
            solution = neighbor
            fitness = fit

    # print(fitness)
    T = int(T * t_change)

print(solution)
print(fitness)

