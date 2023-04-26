from pulp import *
from flexgen.policy.cost_model import *
from flexgen.policy.policy_search import cpu_peak_memory_p, cpu_peak_memory_g, disk_peak_memory

from decimal import Decimal

def search_optimal_policy(arch, prof, gpu_mem, cpu_mem, disk_mem):
    bls_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # bls_list = [10]
    gbs_list = [4, 8, 16, 32, 48, 60]
    # gbs_list = [32]

    gpu_mem_capacity = gpu_mem * (2 ** 30)
    cpu_mem_capacity = cpu_mem * (2 ** 30)
    disk_mem_capacity = disk_mem * (2 ** 30)

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
            T_gen = LpVariable("Cost (After Prefill)", lowBound=0)
            gpu_max_p = LpVariable("GPU Maximum Factor (Prefill)", lowBound=0)
            gpu_max_g = LpVariable("GPU Maximum Factor (After Prefill)", lowBound=0)

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
            gpu_home_p = policy.wg * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
                policy.hg * 2 * arch.s * arch.h_1 * policy.bls + \
                4 * (arch.s + arch.n) * arch.h_1 * policy.cg * policy.bls * arch.l
            
            qkv_p = policy.gbs * 8 * arch.s * arch.h_1
            att_1_p = policy.cg * policy.gbs * (2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1 + 2 * arch.nh * (arch.s ** 2))
            att_2_p = policy.cg * policy.gbs * (2 * arch.nh * (arch.s ** 2) + 2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1)
            embed_p = policy.gbs * 4 * arch.s * arch.h_1
            mlp_1_p = 2 * policy.gbs * arch.s * (arch.h_1 + arch.h_2)
            mlp_2_p = 2 * policy.gbs * arch.s * (arch.h_1 + arch.h_2)

            # gpu_max >= qkv, att_1, att_2, embed, mlp_1, mlp_2
            gpu_w_g = 2 * (1 - policy.wg) * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
                (1 - policy.hg) * 2 * arch.s * arch.h_1 * policy.gbs + \
                gpu_max_p

            gpu_peak_p = gpu_home_p + gpu_w_g

            gpu_home_g = policy.wg * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
                policy.hg * 2 * arch.h_1 * policy.bls + \
                4 * (arch.s + arch.n) * arch.h_1 * policy.cg * policy.bls * arch.l
            
            qkv_g = policy.gbs * 8 * arch.h_1
            att_1_g = policy.cg * policy.gbs *  \
                (2 * arch.h_1 + 2 * (arch.s + arch.n) * arch.h_1 + 2 * arch.nh * (arch.s + arch.n))
            att_2_g = policy.cg * policy.gbs * (2 * arch.nh * (arch.s + arch.n) + 2 * (arch.s + arch.n) * arch.h_1 + 2 * arch.h_1)
            embed_g = 4 * policy.gbs * arch.h_1
            mlp_1_g = 2 * policy.gbs * (arch.h_1 + arch.h_2)
            mlp_2_g = 2 * policy.gbs * (arch.h_2 + arch.h_1)

            gpu_w_g = 2 * (1 - policy.wg) * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
                (1 - policy.hg) * 2 * arch.s * arch.h_1 * policy.gbs + \
                gpu_max_g

            gpu_peak_g = gpu_home_g + gpu_w_g

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
            prob += gpu_max_p >= qkv_p
            prob += gpu_max_p >= att_1_p
            prob += gpu_max_p >= att_2_p
            prob += gpu_max_p >= embed_p
            prob += gpu_max_p >= mlp_1_p
            prob += gpu_max_p >= mlp_2_p
            prob += gpu_max_g >= qkv_g
            prob += gpu_max_g >= att_1_g
            prob += gpu_max_g >= att_2_g
            prob += gpu_max_g >= embed_g
            prob += gpu_max_g >= mlp_1_g
            prob += gpu_max_g >= mlp_2_g
            
            # Optimization contraints
            prob += gpu_peak_p <= gpu_mem_capacity
            prob += gpu_peak_g <= gpu_mem_capacity
            prob += cpu_peak_memory_p(policy, arch) <= cpu_mem_capacity
            prob += cpu_peak_memory_g(policy, arch) <= cpu_mem_capacity
            prob += disk_peak_memory(policy, arch) <= disk_mem_capacity
            prob += (wg + wc + wd) == 1
            prob += (cg + cc + cd) == 1
            prob += (hg + hc + hd) == 1
            
            status = prob.solve(PULP_CBC_CMD(msg=0))

            tmp_dict = {}
            for v in prob.variables():
                tmp_dict[v.name] = v.varValue

            tmp_policy = Policy(bls, gbs, tmp_dict['Weight_Placement_(GPU)'], tmp_dict['Weight_Placement_(CPU)'], tmp_dict['Weight_Placement_(Disk)'], tmp_dict['Cache_Placement_(GPU)'], tmp_dict['Cache_Placement_(CPU)'], tmp_dict['Cache_Placement_(Disk)'], tmp_dict['Activation_Placement_(GPU)'], tmp_dict['Activation_Placement_(CPU)'], tmp_dict['Activation_Placement_(Disk)'])
            optimal_T = (tmp_dict['Cost_(Prefill)'] * arch.l + tmp_dict['Cost_(After_Prefill)'] * (arch.n - 1) * arch.l) / bls
            
            # print(f"Cost bls={bls}, gbs={gbs}: {optimal_T}, policy={tmp_policy}, status={status}")
            if optimal_T < optimal_cost and status == 1:
                optimal_cost = optimal_T
                optimal_policy = tmp_policy
    
    print("Optimal Cost:", optimal_cost)
    print("Optimal Policy:", optimal_policy)

    return optimal_policy

def search_optimal_bls(arch, gpu_mem, policy, elastic_coeff):
    bls_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    gpu_mem_capacity = elastic_coeff * gpu_mem * (2 ** 30)

    for bls in bls_list:
        # GPU Peak
        gpu_home_p = policy.wg * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
            policy.hg * 2 * arch.s * arch.h_1 * bls + \
            4 * (arch.s + arch.n) * arch.h_1 * policy.cg * bls * arch.l
        
        print("[1]", policy.hg * 2 * arch.s * arch.h_1 * bls, "hg:", policy.hg)
        print("[2]", 4 * (arch.s + arch.n) * arch.h_1 * policy.cg * bls * arch.l, policy.hg)
        
        qkv_p = policy.gbs * 8 * arch.s * arch.h_1
        att_1_p = policy.cg * policy.gbs * (2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1 + 2 * arch.nh * (arch.s ** 2))
        att_2_p = policy.cg * policy.gbs * (2 * arch.nh * (arch.s ** 2) + 2 * arch.s * arch.h_1 + 2 * arch.s * arch.h_1)
        embed_p = policy.gbs * 4 * arch.s * arch.h_1
        mlp_1_p = 2 * policy.gbs * arch.s * (arch.h_1 + arch.h_2)
        mlp_2_p = 2 * policy.gbs * arch.s * (arch.h_1 + arch.h_2)

        gpu_w_g = 2 * (1 - policy.wg) * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
            (1 - policy.hg) * 2 * arch.s * arch.h_1 * policy.gbs + \
            max(qkv_p, att_1_p, att_2_p, embed_p, mlp_1_p, mlp_2_p)

        gpu_peak_p = gpu_home_p + gpu_w_g
    
        gpu_home_g = policy.wg * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) * arch.l + \
            policy.hg * 2 * arch.h_1 * bls + \
            4 * (arch.s + arch.n) * arch.h_1 * policy.cg * bls * arch.l
        
        qkv_g = policy.gbs * 8 * arch.h_1
        att_1_g = policy.cg * policy.gbs *  \
            (2 * arch.h_1 + 2 * (arch.s + arch.n) * arch.h_1 + 2 * arch.nh * (arch.s + arch.n))
        att_2_g = policy.cg * policy.gbs * (2 * arch.nh * (arch.s + arch.n) + 2 * (arch.s + arch.n) * arch.h_1 + 2 * arch.h_1)
        embed_g = 4 * policy.gbs * arch.h_1
        mlp_1_g = 2 * policy.gbs * (arch.h_1 + arch.h_2)
        mlp_2_g = 2 * policy.gbs * (arch.h_2 + arch.h_1)

        gpu_w_g = 2 * (1 - policy.wg) * (8 * (arch.h_1 ** 2) + 4 * arch.h_1 * arch.h_2) + \
            (1 - policy.hg) * 2 * arch.s * arch.h_1 * policy.gbs + \
            max(qkv_g, att_1_g, att_2_g, embed_g, mlp_1_g, mlp_2_g)

        gpu_peak_g = gpu_home_g + gpu_w_g

        # if gpu_peak_p > gpu_mem_capacity or gpu_peak_g > gpu_mem_capacity:
        #     continue

        print(f"Max GPU Memory Available: {gpu_mem_capacity}, bls: {bls}, Prefill Peak: {gpu_peak_p}, Generate Peak: {gpu_peak_g}")

if __name__ == "__main__":
    arch = Architecture(96, 512, 32, 12288, (12288 * 4), 96)
    prof = Profile(29.332109968558033, 24.100319935277458, 0.3529573969391976, 0.04779374014020742, 53.061556728960184, 53.061556728960184, 2.766593744737228)
    # optimal_policy = search_optimal_policy(arch, prof, 24, 208, 1500)
    search_optimal_policy(arch, prof, 24, 208, 1500)
    search_optimal_policy(arch, prof, 24 * 0.75, 208, 1500)
    # search_optimal_bls(arch, 24, optimal_policy, 0.9)

# config = OptConfig(name=name,
#             max_seq_len=2048, num_hidden_layers=12, n_head=12,
#             hidden_size=768, input_dim=768, ffn_embed_dim=768 * 4,
#         )

# config = OptConfig(name=name,
#             max_seq_len=2048, num_hidden_layers=96, n_head=96,
#             hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
#         )