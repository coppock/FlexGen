from cost_model import Policy
from pulp import *

def fake_cost_function(policy, arch, prof):
    return policy[0] + policy[1] + policy[2] + arch + prof

def search_optimal_policy(cost_model, arch, prof, bls):
    prob = LpProblem("PolicySearch", LpMinimize)

    wg = LpVariable("Weight Placement (GPU)", 0)
    wc = LpVariable("Weight Placement (CPU)", 0)
    wd = LpVariable("Weight Placement (Disk)", 0)
    cg = LpVariable("Cache Placement (GPU)", 0)
    cc = LpVariable("Cache Placement (CPU)", 0)
    cd = LpVariable("Cache Placement (Disk)", 0)
    hg = LpVariable("Activation Placement (GPU)", 0)
    hc = LpVariable("Activation Placement (CPU)", 0)
    hd = LpVariable("Activation Placement (Disk)", 0)

    policy = (wg, wc, wd, cg, cc, cd, hg, hc, hd)

    prob += cost_model(policy, arch, prof) / bls, "Objective Function"
    
    prob.solve()

    for v in prob.variables():
        print(v.name, "=", v.varValue)

search_optimal_policy(fake_cost_function, 0, 0, 1)