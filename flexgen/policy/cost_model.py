import collections

Policy = collections.namedtuple('Policy', ['bls', 'gbs', 'wg', 'wc', 'wd', 'cg', 'cc', 'cd', 'hg', 'hc', 'hd'])
Architecture = collections.namedtuple('Architecture', ['l', 's', 'n', 'h_1', 'h_2', 'nh'])
Profile = collections.namedtuple('Profile', ['ctog_bdw', 'gtoc_bdw', 'dtoc_bdw', 'ctod_bdw', 'mm_flops', 'bmm_flops', 'cpu_flops'])

def cost(policy, arch, prof):
    w_size = 8 * arch.h_1 ** 2 + 4 * arch.h_1 * arch.h_2
    h_size = 2 * arch.h_1 * policy.bls
    c_size = 4 * arch.h_1 * policy.bls

    ctog_pre = ((policy.wc + policy.wd) * w_size + (policy.hc + policy.hd) * arch.s * h_size) / prof.ctog_bdw
    gtoc_pre = ((policy.cc + policy.cd) * (arch.s + 1) * c_size + (policy.hc + policy.hd) * arch.s * h_size) / prof.gtoc_bdw
    dtoc_pre = (policy.wd * w_size + policy.hd * arch.s * h_size) / prof.dtoc_bdw
    ctod_pre = (policy.cd * (arch.s + 1) * c_size + policy.hd * arch.s * h_size) / prof.ctod_bdw
    comp_pre = policy.bls * arch.s * w_size / prof.mm_flops + arch.s ** 2 * c_size / prof.bmm_flops
    T_pre = max(ctog_pre, gtoc_pre, dtoc_pre, ctod_pre, comp_pre)

    ctog_gen = ((policy.wc + policy.wd) * w_size + (policy.hc + policy.hd) * h_size) / prof.ctog_bdw
    gtoc_gen = (policy.hc + policy.hd) * h_size / prof.gtoc_bdw
    dtoc_gen = (policy.cd * (arch.s + arch.n/2) * c_size + policy.wd * w_size + policy.hd * h_size) / prof.dtoc_bdw
    ctod_gen = (policy.cd * c_size + policy.hd * h_size) / prof.ctod_bdw
    gpu_comp_gen = policy.bls * w_size / prof.mm_flops + policy.cg * (arch.s + arch.n/2) * c_size / prof.bmm_flops
    cpu_comp_gen = (policy.cc + policy.cd) * (arch.s + arch.n/2) * c_size / prof.cpu_flops
    comp_gen = gpu_comp_gen + cpu_comp_gen
    T_gen = max(ctog_gen, gtoc_gen, dtoc_gen, ctod_gen, comp_gen)

    T = T_pre * arch.l + T_gen * (arch.n - 1) * arch.l
    return T
