from pulp import *
from cost_model import *
from policy_search import cpu_peak_memory_p, disk_peak_memory

from decimal import Decimal

def init_T_pre(arch, bls, gbs):
    w_size = 8 * arch.h_1 ** 2 + 4 * arch.h_1 * arch.h_2
    h_size = 2 * arch.h_1 * bls
    c_size = 4 * arch.h_1 * bls

    ctog_pre = ((0 + 0) * w_size + (0 + 0) * arch.s * h_size) / prof.ctog_bdw
    gtoc_pre = ((0 + 0) * (arch.s + 1) * c_size + (0 + 0) * arch.s * h_size) / prof.gtoc_bdw
    dtoc_pre = (0 * w_size + 0 * arch.s * h_size) / prof.dtoc_bdw
    ctod_pre = (0 * (arch.s + 1) * c_size + 0 * arch.s * h_size) / prof.ctod_bdw
    comp_pre = ((bls * arch.s * w_size) / prof.mm_flops) + ((arch.s ** 2 * c_size) / prof.bmm_flops)

    return max(ctog_pre, gtoc_pre, dtoc_pre, ctod_pre, comp_pre)

def init_T_gen(arch, bls, gbs):
    w_size = 8 * arch.h_1 ** 2 + 4 * arch.h_1 * arch.h_2
    h_size = 2 * arch.h_1 * bls
    c_size = 4 * arch.h_1 * bls

    ctog_gen = ((0 + 0) * w_size + (0 + 0) * h_size) / prof.ctog_bdw
    gtoc_gen = (0 + 0) * h_size / prof.gtoc_bdw
    dtoc_gen = (0 * (arch.s + arch.n / 2) * c_size + 0 * w_size + 0 * h_size) / prof.dtoc_bdw
    ctod_gen = (0 * c_size + 0 * h_size) / prof.ctod_bdw
    gpu_comp_gen = (bls * w_size / prof.mm_flops) + (1 * (arch.s + arch.n/2) * c_size / prof.bmm_flops)
    cpu_comp_gen = (0 + 0) * (arch.s + arch.n/2) * c_size / prof.cpu_flops
    comp_gen = gpu_comp_gen + cpu_comp_gen
    
    return max(ctog_gen, gtoc_gen, dtoc_gen, ctod_gen, comp_gen)

def init_gpu_max(arch, bls, gbs):
    w_size = 8 * arch.h_1 ** 2 + 4 * arch.h_1 * arch.h_2
    h_size = 2 * arch.h_1 * bls
    c_size = 4 * arch.h_1 * bls

    qkv = gbs * 8 * arch.s * arch.h_1
    att_1 = 1 * gbs * (2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1 + 2 * arch.nh * (arch.s ** 2))
    att_2 = 1 * gbs * (2 * arch.nh * (arch.s ** 2) + 2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1)
    embed = gbs * 4 * arch.s * arch.h_1
    mlp_1 = 2 * gbs * arch.s * (arch.h_1 + arch.h_2)
    mlp_2 = 2 * gbs * arch.s * (arch.h_1 + arch.h_2)
    
    return max(qkv, att_1, att_2, embed, mlp_1, mlp_2)

def search_optimal_policy(arch, prof):
    bls_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    gbs_list = [4, 8, 16, 32, 48, 60]

    gpu_mem_capacity = 0.5 * (2 ** 30)
    cpu_mem_capacity = 64 * (2 ** 30)
    disk_mem_capacity = 512 * (2 ** 30)

    optimal_cost = Decimal('inf')
    optimal_policy = Policy(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    for bls in bls_list:
        for gbs in gbs_list:
            prob = LpProblem("PolicySearch", LpMinimize)

            wg = LpVariable("Weight Placement (GPU)", lowBound=0, upBound=1)
            wc = LpVariable("Weight Placement (CPU)", lowBound=0, upBound=1)
            wd = LpVariable("Weight Placement (Disk)", lowBound=0, upBound=1)
            cg = LpVariable("Cache Placement (GPU)", lowBound=0, upBound=1)
            cc = LpVariable("Cache Placement (CPU)", lowBound=0, upBound=1)
            cd = LpVariable("Cache Placement (Disk)", lowBound=0, upBound=1)
            hg = LpVariable("Activation Placement (GPU)", lowBound=0, upBound=1)
            hc = LpVariable("Activation Placement (CPU)", lowBound=0, upBound=1)
            hd = LpVariable("Activation Placement (Disk)", lowBound=0, upBound=1)

            T_pre = LpVariable("Cost (Prefill)", lowBound=0)
            T_gen = LpVariable("Cost (Generate)", lowBound=0)
            gpu_max = LpVariable("GPU Maximum Factor (Prefill)", lowBound=0)

            policy = Policy(bls, gbs, wg, wc, wd, cg, cc, cd, hg, hc, hd)

            # Cost
            w_size = 8 * arch.h_1 ** 2 + 4 * arch.h_1 * arch.h_2
            h_size = 2 * arch.h_1 * policy.bls
            c_size = 4 * arch.h_1 * policy.bls

            ctog_pre = ((policy.wc + policy.wd) * w_size + (policy.hc + policy.hd) * arch.s * h_size) / prof.ctog_bdw
            gtoc_pre = ((policy.cc + policy.cd) * (arch.s + 1) * c_size + (policy.hc + policy.hd) * arch.s * h_size) / prof.gtoc_bdw
            dtoc_pre = (policy.wd * w_size + policy.hd * arch.s * h_size) / prof.dtoc_bdw
            ctod_pre = (policy.cd * (arch.s + 1) * c_size + policy.hd * arch.s * h_size) / prof.ctod_bdw
            comp_pre = ((policy.bls * arch.s * w_size) / prof.mm_flops) + ((arch.s ** 2 * c_size) / prof.bmm_flops)

            ctog_gen = ((policy.wc + policy.wd) * w_size + (policy.hc + policy.hd) * h_size) / prof.ctog_bdw
            gtoc_gen = (policy.hc + policy.hd) * h_size / prof.gtoc_bdw
            dtoc_gen = (policy.cd * (arch.s + arch.n / 2) * c_size + policy.wd * w_size + policy.hd * h_size) / prof.dtoc_bdw
            ctod_gen = (policy.cd * c_size + policy.hd * h_size) / prof.ctod_bdw
            gpu_comp_gen = (policy.bls * w_size / prof.mm_flops) + (policy.cg * (arch.s + arch.n/2) * c_size / prof.bmm_flops)
            cpu_comp_gen = (policy.cc + policy.cd) * (arch.s + arch.n/2) * c_size / prof.cpu_flops
            comp_gen = gpu_comp_gen + cpu_comp_gen
        
            T = (T_pre * arch.l + T_gen * (arch.n - 1) * arch.l)

            # GPU Peak
            gpu_home = policy.wg * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
                policy.hg * 2 * arch.s * arch.h_1 * policy.bls + \
                4 * (arch.s + arch.n) * arch.h_1 * policy.cg * policy.bls * arch.l
            
            qkv = policy.gbs * 8 * arch.s * arch.h_1
            att_1 = policy.cg * policy.gbs * (2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1 + 2 * arch.nh * (arch.s ** 2))
            att_2 = policy.cg * policy.gbs * (2 * arch.nh * (arch.s ** 2) + 2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1)
            embed = policy.gbs * 4 * arch.s * arch.h_1
            mlp_1 = 2 * policy.gbs * arch.s * (arch.h_1 + arch.h_2)
            mlp_2 = 2 * policy.gbs * arch.s * (arch.h_1 + arch.h_2)

            # gpu_max >= qkv, att_1, att_2, embed, mlp_1, mlp_2
            gpu_peak = gpu_home + 2 * (1 - policy.wg) * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
                (1 - policy.hg) * 2 * arch.s * arch.h_1 * policy.gbs + \
                gpu_max

            # Optimization
            prob += (T / bls), "Objective Function"

            # Cost constraints
            prob += T_pre >= ctog_pre
            prob += T_pre >= gtoc_pre
            prob += T_pre >= dtoc_pre
            prob += T_pre >= ctod_pre
            prob += T_pre >= comp_pre
            prob += T_gen >= ctog_gen
            prob += T_gen >= gtoc_gen
            prob += T_gen >= dtoc_gen
            prob += T_gen >= ctod_gen
            prob += T_gen >= comp_gen

            # GPU constraints   
            prob += gpu_max >= qkv
            prob += gpu_max >= att_1
            prob += gpu_max >= att_2
            prob += gpu_max >= embed
            prob += gpu_max >= mlp_1
            prob += gpu_max >= mlp_2
            
            # Optimization contraints
            prob += gpu_peak <= gpu_mem_capacity
            prob += cpu_peak_memory_p(policy, arch) <= cpu_mem_capacity
            prob += disk_peak_memory(policy, arch) <= disk_mem_capacity
            prob += (wg + wc + wd) == 1
            prob += (cg + cc + cd) == 1
            prob += (hg + hc + hd) == 1
            
            status = prob.solve(PULP_CBC_CMD(msg=0))

            tmp_dict = {}
            for v in prob.variables():
                tmp_dict[v.name] = v.varValue

            optimal_T = (tmp_dict['Cost_(Prefill)'] * arch.l + tmp_dict['Cost_(Generate)'] * (arch.n - 1) * arch.l)
            if optimal_T < optimal_cost and status == 1:
                optimal_cost = optimal_T
                optimal_policy = Policy(bls, gbs, tmp_dict['Weight_Placement_(GPU)'], tmp_dict['Weight_Placement_(CPU)'], tmp_dict['Weight_Placement_(Disk)'], tmp_dict['Cache_Placement_(GPU)'], tmp_dict['Cache_Placement_(CPU)'], tmp_dict['Cache_Placement_(Disk)'], tmp_dict['Activation_Placement_(GPU)'], tmp_dict['Activation_Placement_(CPU)'], tmp_dict['Activation_Placement_(Disk)'])
    
    print(optimal_cost)
    print(optimal_policy)

arch = Architecture(12, 4096, 32, 768, 3072, 12)
prof = Profile(29.332109968558033, 24.100319935277458, 0.3529573969391976, 0.04779374014020742, 53.061556728960184, 53.061556728960184, 2.766593744737228)
search_optimal_policy(arch, prof)