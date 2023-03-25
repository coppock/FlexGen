from pulp import *

def objective_function(x, y, z):
    return x - y + z

def optimize():
    prob = LpProblem("Example", LpMinimize)
    x = LpVariable("x", 0)
    y = LpVariable("y", 0)
    z = LpVariable("z", 0)

    prob += objective_function(x, y, z), "Objective Function"
    prob += x + y - z <= 1.0
    prob += x >= 0
    prob += y >= 0
    prob += z >= 0

    prob.solve()

    for v in prob.variables():
        print(v.name, "=", v.varValue)

optimize()