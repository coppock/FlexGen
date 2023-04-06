from pulp import *
from cost_model import *

def gpu_peak_memory_p(policy, arch):
    gpu_home = policy.wg * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
        policy.hg * 2 * arch.s * arch.h_1 * policy.bls + \
        4 * (arch.s + arch.n) * arch.h_1 * policy.cg * policy.bls * arch.l
    
    qkv = policy.gbs * 8 * arch.s * arch.h_1
    att_1 = policy.cg * policy.gbs * (2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1 + 2 * arch.n * arch.h * (arch.s ** 2))
    att_2 = policy.cg * policy.gbs * (2 * arch.n * arch.h * (arch.s ** 2) + 2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1)
    embed = policy.gbs * 4 * arch.s * arch.h_1
    mlp_1 = 2 * policy.gbs * arch.s * (arch.h_1 + arch.h_2)
    mlp_2 = 2 * policy.gbs * arch.s * (arch.h_1 + arch.h_2)

    gpu_w = 2 * (1 - policy.wg) * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
        (1 - policy.hg) * 2 * arch.s * arch.h_1 * policy.gbs + \
        max(qkv, att_1, att_2, embed, mlp_1, mlp_2)

    gpu_peak = gpu_home + gpu_w

    return gpu_peak

def gpu_peak_memory_g(policy, arch):
    gpu_home = policy.wg * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
        policy.hg * 2 * arch.h_1 * policy.bls + \
        4 * (arch.s + arch.n) * arch.h_1 * policy.cg * policy.bls * arch.l
    
    qkv = policy.gbs * 8 * arch.h_1
    att_1 = policy.cg * policy.gbs *  \
        (2 * arch.h_1 + 2 * (arch.s + arch.n) * arch.h_1 + 2 * arch.nh * (arch.s + arch.n))
    att_2 = policy.cg * policy.gbs * (2 * arch.nh * (arch.s + arch.n) + 2 * (arch.s + arch.n) * arch.h_1 + 2 * arch.h_1)
    embed = 4 * policy.gbs * arch.h_1
    mlp_1 = 2 * policy.gbs * (arch.h_1 + arch.h_2)
    mlp_2 = 2 * policy.gbs * (arch.h_2 + arch.h_1)

    gpu_w = 2 * (1 - policy.wg) * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
        (1 - policy.hg) * 2 * arch.s * arch.h_1 * policy.gbs + \
        max(qkv, att_1, att_2, embed, mlp_1, mlp_2)

    gpu_peak = gpu_home + gpu_w

    return gpu_peak

def cpu_peak_memory_p(policy, arch):
    cpu_home = policy.wc * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
        policy.hc * 2 * arch.s * arch.h_1 * policy.bls + \
        4 * (arch.s + arch.n) * arch.h_1 * policy.cc * policy.bls * arch.l
    
    cpu_w = (1 - policy.wg) * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
        (1 - policy.hg) * 2 * arch.s * arch.h_1 * policy.gbs
    
    cpu_peak = cpu_home + cpu_w

    return cpu_peak

def cpu_peak_memory_g(policy, arch):
    cpu_home = policy.wc * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
        policy.hc * 2 * arch.h_1 * policy.bls + \
        4 * (arch.s + arch.n) * arch.h_1 * policy.cc * policy.bls * arch.l
    
    cpu_w = policy.wd * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
        2 * policy.hd * 2 * arch.h_1 * policy.gbs + \
        2 * policy.cd * 4 * (arch.s + arch.n) * arch.h_1 * policy.gbs + \
        2 * arch.nh * (arch.s + arch.n) * policy.gbs + \
        2 * arch.h_1 * policy.gbs
    
    cpu_peak = cpu_home + cpu_w

    return cpu_peak

def disk_peak_memory(policy, arch):
    nvme_peak = (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * policy.wd * arch.l + \
        policy.hd * 2 * arch.s * arch.h_1 * policy.bls + \
        policy.cd * 4 * (arch.s + arch.n) * arch.h_1 * policy.bls * arch.l

    return nvme_peak

def search_optimal_policy(arch, prof):
    bls_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    gbs_list = [4, 8, 16, 32, 48, 60]

    gpu_mem_capacity = 0
    cpu_mem_capacity = 0
    disk_mem_capacity = 0

    for bls in bls_list:
        for gbs in gbs_list:
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

            policy = Policy(bls, gbs, wg, wc, wd, cg, cc, cd, hg, hc, hd)

            prob += cost(policy, arch, prof) / bls, "Objective Function"
            prob += gpu_peak_memory_g(policy, arch) < gpu_mem_capacity
            prob += cpu_peak_memory_g(policy, arch) < cpu_mem_capacity
            prob += disk_peak_memory(policy, arch) < disk_mem_capacity
            prob += (wg + wc + wd) == 1
            prob += (cg + cc + cd) == 1
            prob += (hg + hc + hd) == 1
            
            prob.solve()

            for v in prob.variables():
                print(v.name, "=", v.varValue)

# arch = Architecture(16, 1024, 1024, 32, 64, 16)
# prof = Profile(1024, 1024, 128, 128, 2048, 1024, 512)
# search_optimal_policy(arch, prof)