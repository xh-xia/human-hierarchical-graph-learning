"""
CCS visualizations.

Created: Monday, ‎April ‎5, ‎2021, ‏‎3:59:47 PM (EDT; maybe earlier actually)
@author: Xiaohuan (Pixel) X.
"""

import warnings
# warnings.filterwarnings("error")  # raise even on warnings

from scipy import stats  # for corr stats
from itertools import product  # for nested loops
from concurrent.futures import ProcessPoolExecutor  # for multi-processing
from concurrent.futures import ThreadPoolExecutor  # for multi-threading

from utility427.helper427 import set_dir427, mkdir_p, get_params, partial_427_decorator
from utility427.math427 import log_b, findNearest, rank_eigvals, W_norm, np, pval_star, npdivide0
from utility427.plt427 import plt, Normalize, LinearSegmentedColormap, GridSpec, Line2D  # mpl
from utility427.plt427 import saveNclose427, colors_selector, cbrLabel427, get_violin_pw  # mpl helpers
from utility427.plt427 import load_CCS_stat, save_masks
from utility427.Sierpinski427 import make_Sierpinski427, p_ary, make_SierpinskiGraph427
from utility427.sim_params427 import make_sim_params, make_beta_boundry
from utility427.CCS_num import CCS_ep  # newly added on 2022.1.13

FS_TICK = 11
FS_LAB = 11
FS_MAIN = 17


def main_Sierpinski427():
    """
    get parameters from params_CCS427.json:
    - err_type (str): "std" or "ste", type of spread of CCS
    - CCS_type (str): "mean" or "std", type of stat as part of def of CCS
        use mean for the most part, as std is not steady
    - CCS_plot_type (str): "CCS" or "sum" or "both", type of y in CCS plot
        - CCS: vanilla method; show all level
        - sum: show only the sum of CCS across levels
    - raw_method (str): None (in json, it's null) or "violin"
    - dpi (int): used in plot generation; if None, default to both 300 and lossless .pdf
    - sub_fo_name (str): folder in "sim427/output" folder containing CCS_stat .npy files

    get parameters from sim427/input/params.json:
    - key_class (str): "reg_n_p" or something, only a label to classify batch of jobs in sim
    - n_agents (int): num of agents per subject parameter (like a repetition experiment)
    - var_betas: should be effectively the same as that in sim427/input/params.json


    - hierDict (dict): to recreate the graph parameters set (i.e., (regType, p, n))
    """
    set_dir427()  # script dir
    # get /sim427/input parameters (2 .json files)
    p = get_params(fname="sim427\\input\\params", default_dir=False)
    # overwrite key_class
    # can be whatever as long as it (was once defined in params.json) was run in sim427
    # otherwise will be error in load_CCS_stat(), which loads results generated from sim427
    # p["key_class"] = "max_beta"
    p = make_sim_params(p, binned=False)
    p.update(get_params(fname="input\\params_CCS427", default_dir=False))  # get parameters for CCS427.py

    # add more parameters into p
    p["colors"] = colors_selector(str="5-class Greens")
    p["beta_arr"] = np.geomspace(4e-4, 2e1, 400)  # for analytical curve only
    hierDict = dict()
    # hierDict['n'] = [[3],[3],[3,4,5]]
    # hierDict['p'] = [[3],[3,4,5],[3]]
    # hierDict['reg_n'] = [[0, 1, 2, 3], [3], [3, 4, 5]]
    # hierDict['reg_p'] = [[3],[3,4,5],[3]]
    # hierDict['r'] = [[0,1,3],[3],[3]]
    hierDict['emp'] = [[3],[3],[3]]  # experiment only
    kw_loop = dict(sub_fo_name=p["sub_fo_name"], CCS_type=p["CCS_type"], key_class=p["key_class"])
    kw_loop.update(dict(n_agents=p["n_agents"], err_type=p["err_type"], hierDict=hierDict))
    kw_loop.update(dict(beta_arr=p["beta_arr"], colors=p["colors"], dpi=p["dpi"]))
    kw_loop.update(dict(raw_method=p["raw_method"], CCS_plot_type=p["CCS_plot_type"]))
    kw_loop.update(dict(mode427=p["mode427"]))  # 0 analytical; 1 analytical+sim, 2 analytical+exposure

    # just graph (graphs_singleton)
    # betas = [np.inf, 0.33, 0.70]
    # plot_Graph((3, 3, 3), betas, dpi=300, sub_folder_name="", transparent=True, annotate=None)
    # single-processing | w/o mp in plot_main() ~155 sec
    for beta_class in p["beta_classes"]:
        plot_main(**kw_loop, mp=False)(beta_class)
    # multi-threading | w/o mp in plot_main() ~?? sec (too slow)
    # with ThreadPoolExecutor() as executor:
    #     executor.map(plot_main(**kw_loop, mp=False), p["beta_classes"])


@partial_427_decorator
def plot_main(beta_class, sub_fo_name, CCS_type, key_class, n_agents, hierDict,
              beta_arr, err_type, raw_method, CCS_plot_type, colors, dpi, mode427, mp=False):
    """this function is I/O bound | suitable for multi-threading
    should be in script dir when this function is run

    Kwarg
    -----
    mp (bool): whether to use multi-processing on plot_side() | doesn't work, so use False for now

    Intermediary
    ------------
    DD (dict): a data dict created only to be used in graphing (i.e., plot_Graph_CCS())
        most results from numerical calculations are put in DD
    """
    npy_sub_path = f"{sub_fo_name}\\CCS_stat_{CCS_type}_{key_class}_{beta_class}_{n_agents}"
    if mode427 == 0:  # analytical only
        CCS_stat = None
    else:
        CCS_stat = load_CCS_stat(sim_path="sim427\\output", fname=npy_sub_path)  # load sim results (<noise>)

    for key in hierDict.keys():
        kw_main = dict(beta_arr=beta_arr)
        kw_main["beta_arr2"] = CCS_stat["mean"][(3, 3, 3)][0, 0, :] if mode427==2 else None
        if mp:
            with ProcessPoolExecutor() as executor:
                dd_list = executor.map(plot_side(**kw_main), product(*hierDict[key]))
        else:
            dd_list = map(plot_side(**kw_main), product(*hierDict[key]))
        DD = dict()
        for dd in dd_list:  # aggregate all keys of dd in dd_list into DD
            DD.update(dd)

        kw_plot = dict(CCS_stat=CCS_stat, err_type=err_type, CCS_type=CCS_type)
        kw_plot.update(dict(beta_arr2=kw_main["beta_arr2"]))
        if key.startswith("reg_"):
            regCCS = 3
        elif key == "emp":
            regCCS = 5
        else:
            regCCS = 2
        if mode427 == 0:
            regCCS = 6
        kw_plot.update(dict(colors=colors, dpi=dpi, regCCS=regCCS))
        kw_plot.update(dict(CCS_plot_type=CCS_plot_type, sub_folder_name=sub_fo_name))
        kw_plot.update(dict(raw_method=raw_method))

        if regCCS in [0, 1]:
            n_sample = CCS_stat["mean"][list(DD.keys())[0]].shape[0]
            for spl in range(n_sample):
                kw_plot.update(dict(spl=spl))
                plot_Graph_CCS(DD, beta_arr, key, **kw_plot)
        elif regCCS == 2:  # no need for spl since it will be forced to be [0, -1]
            plot_Graph_CCS(DD, beta_arr, key, **kw_plot)
        elif regCCS == 3:
            kw_plot.update(dict(spl=slice(0, None)))
            plot_Graph_CCS(DD, beta_arr, key, **kw_plot)
        elif regCCS in [4, 5]:
            kw_plot.update(dict(spl=0))  # use 1500 step one for sim s.t. it matches experiment
            plot_Graph_CCS(DD, beta_arr, key, **kw_plot)
        elif regCCS == 6:
            kw_plot.update(dict(mode427=0))  # simple, no simulation or whatnot involved
            plot_Graph_CCS(DD, beta_arr, key, **kw_plot)
        else:
            raise NotImplementedError(f"<regCCS>={regCCS} is not implemented")


@partial_427_decorator
def plot_side(tup, beta_arr, beta_arr2=None):
    """this function is numerical calculation heavy | suitable for multi-processing

    Args
    ----
    - tup (tuple): (regType, p, lv), used to create the only key in <dd>
    - beta_arr, beta_arr2 (nparr): 1st one is for analytical curve, 400 pts, 2nd is scatter, 10 only

    Return
    ------
    - dd (dict of dict): it's in lowercase because in contrast to DD, dd only has one key-val pair
        this will be later merged with other dd from the same function to create DD
    """
    regType, p, lv = tup  # unpack tup
    dd = dict()  # = DataDict = {(regType,p,lv):{'GTDict'=GTDict,etc.}}
    dd[tup] = dict()
    dd[tup]["GTDict"] = make_SierpinskiGraph427(p, lv, norm=True, regType=regType)
    save_masks(dd[tup]["GTDict"], regType, p, lv)
    dd[tup]["A_hat_list"] = [
        make_A_hat_beta(dd[tup]["GTDict"]["A"], beta) for beta in beta_arr
    ]
    dd[tup]["CCS_arr"] = CCS_analysis(
        dd[tup]["GTDict"], beta_arr, dd[tup]["A_hat_list"], analytic=False
    )
    if beta_arr2 is None:
        dd[tup]["CCS_stat_ep"] = None  # plot sim as scatter points
    else:  # plot numerical approximation as scatter points
        kw = dict(T=[1500, 3000, 4500, 6000, 7500], mp=False)
        # kw = dict(T=[500, 1000, 1500, 2000, 3000], mp=False)
        A_hat_list2 = [make_A_hat_beta(dd[tup]["GTDict"]["A"], beta) for beta in beta_arr2]
        dd[tup]["CCS_stat_ep"] = CCS_ep(dd[tup]["GTDict"], A_hat_list2, beta_arr2, **kw)
        print(f"DEBUG CCS_ep 1 tup done: {tup}")
    Sier = make_Sierpinski427(p, lv, x0=[0.0, 0.0], s0=1.0, c=1.0, regType=regType)
    Sier.Layout_Sierpinski427()
    dd[tup]["Sier"] = Sier

    return dd


def powers_MC427(noise, params, res=1000, n_samp_grid=None, pval=0.05):
    """ it's a two-way simulation to find powers (i.e., prob of rejecting null if alt is true)
    if to do it in R, one needs to input noise, the only actual data input for this function, to R
    and output 3D arr back to python
    currently only implemented for the first sample slice (spl=0): 1500 walk length
    1) was done in getting the noise
    2) MC simulatino to draw from noise
    for each beta in noise:
        for each sample size n_samp in n_samp_grid:
            find power of Wilcoxon signed-rank test:
                MC sample data from noise w/ replacement, do it for res * n_samp times

    Kwargs
    ------
    - res (int): number of draws from noise; this determines the resolution of MC simulation
    - n_samp_grid (arr-like): the sample size to be swept through to get powers of
    - pval (float): pval threshold for power
    
    Return
    ------
    - powers (3D nparr): dim=(b, s, c)
    b = len(noise["mean"][params][spl, 0, :]) (1st is sample period; 2nd means 3rd is beta)
    s = len(n_samp_grid)
    c = len(noise["raw"][params][spl, :, 0, 0]); number of CCS lvs
        the entry is then power for each corresponding (beta, n_samp, CCS_lv) pair
    """
    RNG = np.random.default_rng()  # no specific seed
    spl = 0
    s = len(n_samp_grid)
    b = len(noise["mean"][params][spl, 0, :])  # beta arr
    c = len(noise["raw"][params][spl, :, 0, 0])
    powers = np.zeros((b, s, c))
    for i in range(b):  # beta loop
        for j in range(s):  # sample grid loop
            for k in range(c):  # CCS lv loop
                count = 0  # count instances of rejecting null (i.e., p_wilcoxon < pval)
                temp = noise["raw"][params][spl, k, i, :]  # full noise data
                full_size = len(temp)
                for _ in range(res):  # MC loop; 1-sample Wilcoxon: subtract baseline 1
                    idx_arr = RNG.choice(full_size, size=n_samp_grid[j])  # MC sample w/ replace
                    result = stats.wilcoxon(temp[idx_arr] - 1, alternative="greater").pvalue
                    count += (result < pval)
                powers[i, j, k] = count / res  # find power

    return powers


def FX_MC427(noise, params, res=1000, n_samp_grid=None, fx=1, all_beta=True):
    """ the algorithm is almost identical to powers_MC427();
    for res=1000 and n_samp_grid=geomspace=(1,100000,1000), it took ~2200 sec
    for res=1000 and n_samp_grid=1,...,200+geomspace=(201,100000,1000), it took almost 3200 sec
    "FX" stands for "effect", like SFX (sound effect)
    except the whole MC samples (effect size & p-val) are kept;
    thus the pval kwarg is changed to specify what effect size (stat test) to use
    another difference is that this is testing one CCS level to the next,
    instead of comparing one to baseline
    1) was done in getting the noise
    2) MC simulatino to draw from noise
    for each beta in noise:
        for each sample size n_samp in n_samp_grid:
            find effect size & p-val of specified test:
                MC sample data from noise w/ replacement, do it for res * n_samp times

    Kwargs
    ------
    - res (int): number of draws from noise; this determines the resolution of MC simulation
    - n_samp_grid (arr-like): the sample size to be swept through
    - fx (int): int code for effect size; corr is default; currently unused
    - all_beta (bool): whether we do per beta bin corr or among all of them
    
    Return
    ------
    - fxs / fxp (4D nparr): dim=(b, s, c-1, res)
    b = len(noise["mean"][params][spl, 0, :]) (1st is sample period; 2nd means 3rd is beta)
    s = len(n_samp_grid)
    c = len(noise["raw"][params][spl, :, 0, 0]); number of CCS lvs
        the entry is then effect size / p-val for each corresponding (beta, n_samp, CCS_lv) pair
    """
    RNG = np.random.default_rng()  # no specific seed
    spl = 0
    s = len(n_samp_grid)
    c = len(noise["raw"][params][spl, :, 0, 0])

    if all_beta:
        fxs = np.zeros((s, c - 1, res))
        fxp = np.zeros((s, c - 1, res))
        for j in range(s):  # sample grid loop
            for k in range(c - 1):  # cross CCS lv loop
                temp = noise["raw"][params][spl, [k, k + 1], :, :]  # full noise data (all beta)
                temp = np.reshape(temp, (2, -1))  # unspecified is collapsed number of points
                full_size = temp.shape[-1]
                for l in range(res):  # MC loop
                    idx_arr = RNG.choice(full_size, size=n_samp_grid[j])  # MC sample w/ replace
                    result = stats.spearmanr(temp[:, idx_arr], axis=1)
                    fxs[j, k, l] = result.correlation
                    fxp[j, k, l] = result.pvalue
    else:
        b = len(noise["mean"][params][spl, 0, :])  # beta arr
        fxs = np.zeros((b, s, c - 1, res))
        fxp = np.zeros((b, s, c - 1, res))
        for i in range(b):  # beta loop
            for j in range(s):  # sample grid loop
                for k in range(c - 1):  # cross CCS lv loop
                    temp = noise["raw"][params][spl, [k, k + 1], i, :]  # full noise data
                    full_size = temp.shape[-1]
                    for l in range(res):  # MC loop
                        idx_arr = RNG.choice(full_size, size=n_samp_grid[j])  # MC sample w/ replace
                        result = stats.spearmanr(temp[:, idx_arr], axis=1)
                        fxs[i, j, k, l] = result.correlation
                        fxp[i, j, k, l] = result.pvalue

    return fxs, fxp


def make_A_hat_beta(A, beta):
    """generate A_hat according to Max Entropy Model

    Args
    ----
    - A (2D nparr; symmetric): adjacency/weight matrix
    - beta (any number): complexity-accuracy trade-off param

    Return
    ------
    - A_hat (2D nparr, shape = np.shape (A)):
        assuming infinite walks on A, this is the resulting A_hat learned based on beta
        A_hat = (1-e^(-β)) * A * (I - (e^(-β))A)^(-1)
        undirected, weighted 3-regular graph with:
        lv hierarchies:
        level 1: base level; smallest communities of (3) nodes
        ...
        level lv-1: 3 communities of (3) level-lv-2 units
        level lv: 1 community of (3) level-lv-1 unit (coarsest level)
    """
    n = np.shape(A)[0]  # # of rows, but assuming symmetric, thus also cols (=nodes)
    A_ = W_norm(A)
    return (1 - np.exp(-beta)) * A_ @ np.linalg.inv(np.eye(n) - np.exp(-beta) * A_)


def get_S_kl(n, A, beta_arr, edgeList, lvList, pList=None, nList=None):
    """

    Args
    ----
    - n: power, whereas p is base
    - A: GroundTruth Transition Prob matrix.
    - beta_arr: list of β
    - pList: list of p in L_p(l)
    - nList: list of n in I_n(l)

    Intermediary
    ------------
    get L_pl ≡ L_p(l) = sum over all eigenvalues: (λ/(1-λ))**p * S_kl
    where (λ_k/(1-λ_k))**p = pk[p,k]
    where we also need to compute S_kl first:
    "structure factor":
    S_kl: S_kl[k,l] = mean_ij(v[i]*v[j]) for k-th eigvec v; v[i],v[j] ∈ level-(l+1) community (edge is level-(l+1) only)
    with corresponding eigvals
    eigval_kβ: eigval_kβ[k,β] = λ_A_hat[k]
               eigval_kβ[k,-1] = λ_A[k]
    "finite-step surprisal":
    ΔI_n(l1,l2) = sum_k (λ_k^n*(S_kl1-S_kl2))
    I_n(l) = sum_k (λ_k^n*S_kl)
    """
    if pList is None:
        pList = [1, 2, 3, 4]
    if nList is None:
        nList = np.arange(2, 47)
    res = dict()  # results of the calculations

    res["eigvals"], res["eigvecs"], res["eigvals_rank"] = rank_eigvals(A)  # GroundTruth
    res["num_eigval"] = max(res["eigvals_rank"]) + 1  # num of unique eigvals
    res["S_kl"] = np.zeros((res["num_eigval"], max(lvList)))
    res["pk"] = np.zeros(
        (len(pList), res["num_eigval"])
    )  # largest eigval will not be calculated, thus always zeroes in here
    res["nk"] = np.zeros((len(nList), res["num_eigval"]))  # ditto
    res["eigval_kβ"] = np.zeros((res["num_eigval"], len(beta_arr) + 1))
    # res['L_pl'] = np.zeros((len(pList), max(lvList)))
    # res['I_nl'] = np.zeros((len(nList), max(lvList)))
    # res['ΔI_n'] = np.zeros((len(nList), max(lvList)-1))

    """
    S_kl calculation (no β involved) & eigval_kβ calculation (no group assignment involved)
    note: (algebraic=geometric in our case) multiplicity should be the same for any β
    """
    for eig_k in range(res["num_eigval"]):
        idx = res["eigvals_rank"].index(eig_k)  # first one that matches
        res["eigval_kβ"][eig_k, -1] = res["eigvals"][idx]
        res["eigval_kβ"][eig_k, :-1] = (
            (1 - np.exp(-beta_arr))
            * res["eigvals"][idx]
            / (1 - (np.exp(-beta_arr)) * res["eigvals"][idx])
        )
        if eig_k >= 1:  # excluding largest eigval (i.e., eig_k = 0)
            for i in range(len(pList)):
                res["pk"][i, eig_k] = (res["eigvals"][idx] / (1 - res["eigvals"][idx])) ** pList[i]
            for i in range(len(nList)):
                res["nk"][i, eig_k] = (res["eigvals"][idx]) ** nList[i]
        kth_eigvec = res["eigvecs"][:, idx]  # nparr
        for l in range(1, max(lvList) + 1):
            b_ = [x == l for x in lvList]  # boolean mask
            b_edgeList = [e for (e, v) in zip(edgeList, b_) if v]  # edges in level l
            res["S_kl"][eig_k, l - 1] = np.mean(
                [kth_eigvec[v_i] * kth_eigvec[v_j] for (v_i, v_j) in b_edgeList]
            )
    res["L_pl"] = res["pk"] @ res["S_kl"]
    res["I_nl"] = res["nk"] @ res["S_kl"]
    res["ΔI_n"] = np.diff(res["I_nl"][:, ::-1], axis=1)[:, ::-1]
    return res


def CCS_analysis(GTDict, beta_arr, A_hat_list=None, analytic=False):
    """
    This function finds simulated (or analytic) CCS for all beta in beta_arr.

    Args
    ----
    - GroundTruthOnly (bool): not implemented ∵ those zero in original don't have well-defined hierarchies
        True: only calculate mean over the edges that are non-zero in original Sierpiński
        False: calculate mean over all appropriate edges
    - analytic (bool): if True, find analytic CCS
        not implemented if there is dynamic beta (i.e., beta_arr contains negative beta)
    - A_hat_list (np.arr):
        None: analytical prediction from Eigen-decomposition (uses beta_arr)
        np.arr: simulated result (vanilla method; doesn't use beta_arr)

    Return
    ------
    - CCS_arr (3D nparr):
        since CCS is for every 2 consecutive lvs, we have only (lv-1) entries out of lv levels
        NOTE: means, or stds, is part of CCS definition:
            we can find certain 1-val statistic (mean, std, etc) for a given level,
            and then we subtract or divide (MentalErrors paper used divide)
            and "CCS for level l" is a function of that statistic of level l and l+1
        CCS_arr[s,0,l-1]: CCS of means at beta s for level l (f"{'lv'}{l}{'-'}{l+1}")
        CCS_arr[s,1,l-1]: CCS of stds at beta s for level l (f"{'lv'}{l}{'-'}{l+1}")

    Miscellany
    ----------
    Copy Paste from make_SierpinskiGraph427() documentation:
    edgeList (a list of size-2 tuples (v_i,v_j))
        node index in edgeList is simply p2ten(s, p=p)
        where s is nodel p-ary string label
    lvList (a list of hierarchy labels): finest level is 1
    """
    edgeList, lvList = GTDict["edgeList"], GTDict["lvList"]
    n = len(beta_arr)  # number of beta (which is also number of graphs)
    lv = max(lvList)  # (max) hierarchical level (also the coarsest level)
    CCS_arr = np.zeros(
        (n, 2, lv - 1)
    )  # since CCS is for every 2 consecutive lvs, we have only (lv-1) entries out of lv levels

    if A_hat_list is None:  # mental matrix is not supplied, we have to go analytic
        analytic = True
    if analytic:  # if analytic, we can't have dynamic beta (i.e., negative entries in beta_arr)
        for i in range(-1, -n - 1, -1):
            if beta_arr[i] < 0:
                raise NotImplementedError("analytic CCS not implemented for dynamic beta")

    if not analytic:  # simulated CCS
        for i in range(n):
            mean_weights = [0.0 for _ in range(lv)]
            std_weights = [0.0 for _ in range(lv)]
            for l in range(1, lv + 1):
                b_ = [x == l for x in lvList]  # boolean mask
                b_edgeList = [e for (e, v) in zip(edgeList, b_) if v]  # edges in level l
                temp_list = [A_hat_list[i][v_i, v_j] for (v_i, v_j) in b_edgeList]
                mean_weights[l - 1] = np.mean(temp_list)
                std_weights[l - 1] = np.std(temp_list)
            # temp = -np.diff(mean_weights) # diff: all >0 if edge weights in finer level > coarser level
            # temp = np.exp(-np.diff(np.log(mean_weights))) # ratio: all >1 if edge weights in finer level > coarser level
            # if i == findNearest(beta_arr, 0.3, is_arg = True):
            #     print("DEBUG [analytical] mean_weights: {} | std_weights: {}".format(mean_weights,std_weights))
            temp1 = np.divide(mean_weights[:-1], mean_weights[1:])  # ditto, but more explicit
            temp2 = npdivide0(std_weights[:-1], std_weights[1:], atol=1.e-14, filler=np.nan)
            for l in range(0, lv - 1):  # l is index, corresponding CCS lv is l+1
                CCS_arr[i, 0, l] = temp1[l]
                CCS_arr[i, 1, l] = temp2[l]
    else:  # analytic CCS
        eigvals, eigvecs = np.linalg.eigh(GTDict["A"])
        eigvals = np.diag(eigvals)  # convert eigval list into eigval matrix 𝚲
        for i in range(n):
            mean_weights = [0.0 for _ in range(lv)]
            std_weights = [0.0 for _ in range(lv)]
            EB = np.exp(
                -beta_arr[i]
            )  # coefficient (e^-β) to find the eigenvalue of learned matrix A_hat
            Lambda = (1 - EB) * eigvals / (1 - EB * eigvals)
            A_hat = eigvecs @ Lambda @ (eigvecs.T)  # this works if A is symmetric (regularized)
            # for some reason it is all nan when p=3, lv=4
            # A_hat = eigvecs @ Lambda @ (np.linalg.inv(eigvecs))
            for l in range(1, lv + 1):
                b_ = [x == l for x in lvList]  # boolean mask
                b_edgeList = [e for (e, v) in zip(edgeList, b_) if v]  # edges in level l
                temp_list = [A_hat[v_i, v_j] for (v_i, v_j) in b_edgeList]
                mean_weights[l - 1] = np.mean(temp_list)
                std_weights[l - 1] = np.std(temp_list)
            # temp = -np.diff(mean_weights) # diff: all >0 if edge weights in finer level > coarser level
            # temp = np.exp(-np.diff(np.log(mean_weights))) # ratio: all >1 if edge weights in finer level > coarser level
            temp1 = np.divide(mean_weights[:-1], mean_weights[1:])  # ditto, but more explicit
            temp2 = npdivide0(std_weights[:-1], std_weights[1:], atol=1.e-14, filler=np.nan)
            for l in range(0, lv - 1):
                CCS_arr[i, 0, l] = temp1[l]
                CCS_arr[i, 1, l] = temp2[l]

    return CCS_arr


def plot_Graph_CCS(
    DD, beta_arr, key, CCS_stat=None, raw_method=None, spl=-1, CCS_type="mean", err_type="ste",
    CCS_plot_type="CCS", colors=None, dpi=None, regCCS=0, sub_folder_name="", beta_arr2=None, mode427=1
):
    """It produces both CCS plot and Graph (node-edge graph, not graph graph) plot

    Args
    ----
    - DD (dict):
        DD.keys (tuple): (regType,p,n)
    - key (str): those in hierDict.keys()

    Kwargs
    ------
    - CCS_stat (dict): CCS_stat['mean'], CCS_stat['std'], and CCS_stat['ste'] have:
        same keys as DD; value is 3D nparr (see RW_CCS_stat.py for description)
    - raw_method (str): if prefix in ["violin", "box"], assume CCS_stat has "raw" key
        - "violin": vanilla violin plot
        - "violin_median": show median in red, and only 2nd lv CCS as well (1st lv is as expected)
        - "box": box plot; same settings as ECCS box plot
    - spl (int or list): what indices of sample to draw data from CCS_stat; -1 is the last sample
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    - CCS_plot_type (str):
        - "CCS": vanilla
        - "sum": sum of CCS across levels
    - colors (list of color hex strings):
        e.g., plt.rcParams['axes.prop_cycle'].by_key()['color'] is default color in pyplot:
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    - regCCS (int):
        0: display 1 row of 2 cols of CCCS with no graphs; each col corresponds to a CCS level
        1: display 4 rows of CCS with no graphs
        assume:
            hierDict['reg_n'] = [[0,1,2,3],[3],[3,4,5]]
            or
            hierDict['reg_p'] = [[0,1,2,3],[3,4,5],[3]]
        2: display 3 rows: 1 graph 2 CCS; no varb implemented for this yet; spl is forced to be list
        3: display 5 rows of CCS with no graphs; all of regType=3 since that's the exp setup
        4: display 5 rows of n cols of CCS with no graphs; each but one col corresponds to a CCS level
            row varies in number of agents per bin: np.array([1, 5, 10, 50, 100]) * (n_agents // 100)
        5: display n rows of 3 cols of CCS with no graphs (n is number of CCS level)
            column differs in sample size per bin: 10, 100, 10,000
        6: display 2 rows: 1 graph 1 CCS; spl unused
    - sub_folder_name (str):
        the name of the folder in f"output/{whatnot}/" to store the plots
        where <whatnot> is defined in saveNclose427()
    - mode427 (int):
        - 0: analytical curve only
        - 1: analytical + simulation
        - 2: analytical + exposure

    Save
    ----
    CCS plot:
    - fixed beta
    - variable beta (controlled by varb ≡ var_beta)
        current implementation is a bit awkward & verbose (spamming if varb: everywhere lol)
        only affects ax_CCS since Graph has nothing to do with beta in current ver.
    """
    varb = False
    DD_keys = sorted(DD.keys(), reverse=False)  # ascending (default)
    if regCCS==2:  # force spl to be a len-2 list [0, -1]
        spl = [0, -1]
    elif regCCS==3:
        spl = list(range(0, 5))  # 5 walk sizes
        DD_keys = [k for k in DD_keys if k[0]==3]  # we only need regType=3
    if CCS_stat is not None and regCCS not in [2, 4, 5]:
        # generate variable beta version if there is negative beta (just need to check last item)
        if CCS_stat["mean"][DD_keys[0]][spl[0], 0, -1] < 0:  # 2nd dim idx=0 is beta dim
            varb = True

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if regCCS==0:  # 2 rows
        n_rows = 2
    elif regCCS==1:  # 4 rows
        n_rows = 4
    elif regCCS==2:  # 3 rows
        n_rows = 3
    elif regCCS in [3, 4]:  # 5 rows
        n_rows = 5
    elif regCCS==5:  # n_level rows
        n_rows = DD_keys[0][2] - 1
    elif regCCS==6:  # 2 rows: 1 graph 1 analytical CCS
        n_rows = 2
    else:  # 1 row; n_level cols
        n_rows = 1
    if regCCS in [0, 1, 2, 3, 5, 6]:
        n_cols = 3
    elif regCCS == 4:  # (n_level=n-1) + 1 cols
        n_cols = DD_keys[0][2]
    else:  # n_level cols
        n_cols = DD_keys[0][2] - 1

    fig = plt.figure(figsize=[6.5 * n_cols, 4.5 * n_rows])  # initialize
    if varb:
        fig2 = plt.figure(figsize=[6.5 * n_cols, 4.5 * n_rows])  # initialize

    ds = 0.2  # dummy axes for spacing between the visible plots
    hf = 9  # relative height of sub-figure
    cbW = 1  # colorbar width; need it if we draw networks
    height_ratios = ([1, hf, ds, 1] * n_rows)[:-1]  # remove last sub-row
    if mode427 == 0:
        width_ratios = [ds, 19, cbW] * n_cols
    else:
        width_ratios = [ds, 19, ds] * n_cols

    kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
    kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
    gs = GridSpec(**kw1)

    axes = dict()
    if varb:
        axes2 = dict()
    if regCCS == 0:
        axes["Graph"], axes["CCS"], axes["Colorbar"] = [None] * 3, [None] * 3, [None] * 3
        if varb:
            axes2["Graph"], axes2["CCS"], axes2["Colorbar"] = [None] * 3, [None] * 3, [None] * 3
    elif regCCS in [1, 3]:
        for row in range(4 + (regCCS == 3)):
            axes[f"CCS_row{row}"] = [None] * 3
            if varb:
                axes2[f"CCS_row{row}"] = [None] * 3
    elif regCCS == 2:
        axes["Graph"], axes["CCS"], axes["Colorbar"] = [None] * 3, [None] * 6, [None] * 3
    elif regCCS == 6:
        axes["Graph"], axes["CCS"], axes["Colorbar"] = [None] * 3, [None] * 3, [None] * 3
    elif regCCS == 4:
        for row in range(5):
            axes[f"CCS_row{row}"] = [None] * DD_keys[0][2]
    elif regCCS == 5:
        for row in range(DD_keys[0][2] - 1):
            axes[f"CCS_row{row}"] = [None] * 3
    else:
        axes["CCS"] = [None] * (DD_keys[0][2] - 1)

    fname = f"CCS_{key}"
    if CCS_stat is not None:
        # both group size and walk length are the same across all beta
        if regCCS!=2:
            if regCCS in [0, 1]:  # spl (int)
                n_agents = round(CCS_stat["mean"][DD_keys[0]][spl, 1, 0])
                n_steps = round(CCS_stat["mean"][DD_keys[0]][spl, 2, 0])
                fname += f"_{n_agents}_{n_steps}_{err_type}_{CCS_type}"
            elif regCCS == 3:  # spl (list)
                n_agents = round(CCS_stat["mean"][DD_keys[0]][spl[-1], 1, 0])  # same for any spl
                fname += f"_{n_agents}_allsteps_{err_type}_{CCS_type}"
            elif regCCS in [4, 5]:  # spl (int; 0); group size is diff tho
                n_steps = round(CCS_stat["mean"][DD_keys[0]][spl, 2, 0])
                fname += f"_allagents_{n_steps}_{err_type}_{CCS_type}"
        else:
            n_agents = round(CCS_stat["mean"][DD_keys[0]][spl[-1], 1, 0])  # same for any spl
            n_steps1 = round(CCS_stat["mean"][DD_keys[0]][spl[0], 2, 0])
            n_steps2 = round(CCS_stat["mean"][DD_keys[0]][spl[1], 2, 0])
            n_steps = (n_steps1, n_steps2)
            fname += f"_{n_agents}_{n_steps}_{err_type}_{CCS_type}"

    if regCCS==1:  # assuming DD_keys has 12 entries
        for i in range(3):  # col
            for regType in [0, 1, 2, 3]:  # row
                temp = gs[regType * 4 : regType * 4 + 3, i * 3 + 1]
                axes[f"CCS_row{regType}"][i] = fig.add_subplot(temp)
                if varb:
                    axes2[f"CCS_row{regType}"][i] = fig2.add_subplot(temp)
                params = DD_keys[i + 3 * regType]

                kw2 = dict(ax=axes[f"CCS_row{regType}"][i], x=beta_arr)
                kw2.update(dict(CCS_plot_type=CCS_plot_type))
                kw2.update(dict(params=params, key=key))
                kw2.update(dict(CCS_arr=DD[params]["CCS_arr"], noise=CCS_stat))
                kw2.update(dict(noise_ep=DD[params]["CCS_stat_ep"]))
                kw2.update(dict(err_type=err_type, dpi=dpi, CCS_type=CCS_type))
                kw2.update(dict(spl=spl, is_log=True, colors=colors, has_xlabel=regType==3))
                kw2.update(dict(raw_method=raw_method, has_title=True))
                ax_CCS(**kw2)
                if varb:
                    kw2.update(dict(ax=axes2[f"CCS_row{regType}"][i], varb=varb))
                    ax_CCS(**kw2)
    elif regCCS==3:  # assuming DD_keys has 3 entries (-> 3 col)
        for i in range(3):  # col
            for row in spl:  # row
                temp = gs[row * 4 : row * 4 + 3, i * 3 + 1]
                axes[f"CCS_row{row}"][i] = fig.add_subplot(temp)
                if varb:
                    axes2[f"CCS_row{row}"][i] = fig2.add_subplot(temp)
                params = DD_keys[i]  # unlike regCCS==1, 1 DD_keys per col

                kw2 = dict(ax=axes[f"CCS_row{row}"][i], x=beta_arr)
                kw2.update(dict(CCS_plot_type=CCS_plot_type))
                kw2.update(dict(params=params, key=key))
                kw2.update(dict(CCS_arr=DD[params]["CCS_arr"], noise=CCS_stat))
                kw2.update(dict(noise_ep=DD[params]["CCS_stat_ep"]))
                kw2.update(dict(err_type=err_type, dpi=dpi, CCS_type=CCS_type))
                kw2.update(dict(spl=row, is_log=True, colors=colors, has_xlabel=row==spl[-1]))
                kw2.update(dict(raw_method=raw_method, show_legend=i == 1, has_title=True))
                ax_CCS(**kw2)
                if varb:
                    kw2.update(dict(ax=axes2[f"CCS_row{regType}"][i], varb=varb))
                    ax_CCS(**kw2)
    elif regCCS in [0, 2, 6]:  # have graphs as first row
        lr = (regCCS in [0, 6])  # True if regCCS is 0/6, meaning first CCS row is the last row
        spl_current = spl if (regCCS in [0, 6]) else spl[0]
        for i in range(3):
            axes["Graph"][i] = fig.add_subplot(gs[0:3, i * 3 + 1])
            axes["CCS"][i] = fig.add_subplot(gs[4:7, i * 3 + 1])
            axes["Colorbar"][i] = fig.add_subplot(gs[0:3, i * 3 + 2])  # Edge Type colorbar
            params = DD_keys[i]
            kw3 = dict(ax=axes["Graph"][i], axcb=axes["Colorbar"][i], fig=fig, dpi=dpi)
            kw3.update(dict(params=params, colors=colors))
            kw3.update(dict(nodeList=DD[params]["Sier"].nodeList, GTDict=DD[params]["GTDict"]))
            ax_Graph(**kw3)
            kw4 = dict(ax=axes["CCS"][i], x=beta_arr, params=params)
            kw4.update(dict(spl=spl_current, CCS_plot_type=CCS_plot_type))
            kw4.update(dict(CCS_arr=DD[params]["CCS_arr"], noise=CCS_stat))
            kw4.update(dict(noise_ep=DD[params]["CCS_stat_ep"]))
            kw4.update(dict(key=key, err_type=err_type, CCS_type=CCS_type))
            kw4.update(dict(show_legend=i == 1, is_log=True, colors=colors, dpi=dpi, has_xlabel=lr))
            kw4.update(dict(raw_method=raw_method, has_title=True))
            ax_CCS(**kw4)
            if varb:
                axes2["Graph"][i] = fig2.add_subplot(gs[0:3, i * 3 + 1])
                axes2["CCS"][i] = fig2.add_subplot(gs[4:7, i * 3 + 1])
                axes2["Colorbar"][i] = fig2.add_subplot(gs[0:3, i * 3 + 2])  # Edge Type colorbar
                kw3.update(dict(ax=axes2["Graph"][i], axcb=axes2["Colorbar"][i], fig=fig2))
                ax_Graph(**kw3)
                kw4.update(dict(ax=axes2["CCS"][i], varb=varb))
                ax_CCS(**kw4)
        if regCCS==2:  # two CCS rows: one with walk_length=1500, one with walk_length=7500
            spl_current = spl[1]  # 2nd CCS row; 1st was done above
            for i in range(3):
                axes["CCS"][i+3] = fig.add_subplot(gs[8:11, i * 3 + 1])
                params = DD_keys[i]
                kw4.update(dict(ax=axes["CCS"][i+3]))
                kw4.update(dict(CCS_arr=DD[params]["CCS_arr"], params=params))
                kw4.update(dict(spl=spl_current, show_legend=i == 1, has_xlabel=True))
                ax_CCS(**kw4)
    elif regCCS == 4:  # DD_keys has 1 entry: (3, 3, 3)
        temp_n_agents = CCS_stat["mean"][DD_keys[0]][0, 1, 0]  # total number of agents
        temp_n_agents = np.array([0.1, 0.3, 0.5, 0.7, 1]) * (temp_n_agents // 100)
        # temp_n_agents = np.array([0.1, 0.5, 1, 10, 100]) * (temp_n_agents // 100)
        # temp_n_agents = np.array([1, 5, 10, 50, 100]) * (temp_n_agents // 100)
        temp_n_agents = np.around(temp_n_agents, decimals=0).astype(int, casting="unsafe")
        plot_corr(DD_keys[0], CCS_stat, colors, sub_folder_name, res=1000, res_grid=1000, dpi=dpi, truncated=True)
        plot_power(DD_keys[0], CCS_stat, colors, sub_folder_name, res=1000, pvals=None, dpi=dpi)
        for i in range(DD_keys[0][2]):  # col: all CCS level + per CCS level
            for row in range(5):  # row
                temp = gs[row * 4 : row * 4 + 3, i * 3 + 1]
                axes[f"CCS_row{row}"][i] = fig.add_subplot(temp)
                params = DD_keys[0]
                temp_emp_lv = None if i == 0 else i

                kw2 = dict(ax=axes[f"CCS_row{row}"][i], x=beta_arr)
                kw2.update(dict(CCS_plot_type=CCS_plot_type))
                kw2.update(dict(params=params, key=key))
                kw2.update(dict(CCS_arr=DD[params]["CCS_arr"], noise=CCS_stat))
                kw2.update(dict(noise_ep=None, emp_lv=temp_emp_lv))
                kw2.update(dict(err_type=err_type, dpi=dpi, CCS_type=CCS_type))
                kw2.update(dict(spl=spl, is_log=True, colors=colors, has_xlabel=row==4, has_title=row==0))
                kw2.update(dict(raw_method=raw_method, show_legend=i==0, n_agents=temp_n_agents[row]))
                ax_CCS(**kw2)
    elif regCCS == 5:  # DD_keys has 1 entry: (3, 3, 3)
        temp_n_agents = CCS_stat["mean"][DD_keys[0]][0, 1, 0]  # total number of agents per bin
        temp_n_agents = np.array([10, 100, temp_n_agents], dtype=int)
        plot_corr(DD_keys[0], CCS_stat, colors, sub_folder_name, res=1000, res_grid=1000, dpi=dpi, truncated=True)
        plot_power(DD_keys[0], CCS_stat, colors, sub_folder_name, res=1000, pvals=[0.05], dpi=dpi)
        for i in range(3):  # col: each n_per_bin
            for row in range(DD_keys[0][2] - 1):  # row
                temp = gs[row * 4 : row * 4 + 3, i * 3 + 1]
                axes[f"CCS_row{row}"][i] = fig.add_subplot(temp)
                params = DD_keys[0]
                temp_emp_lv = row + 1

                kw2 = dict(ax=axes[f"CCS_row{row}"][i], x=beta_arr)
                kw2.update(dict(CCS_plot_type=CCS_plot_type))
                kw2.update(dict(params=params, key=key))
                kw2.update(dict(CCS_arr=DD[params]["CCS_arr"], noise=CCS_stat))
                kw2.update(dict(noise_ep=None, emp_lv=temp_emp_lv))
                kw2.update(dict(err_type=err_type, dpi=dpi, CCS_type=CCS_type))
                kw2.update(dict(spl=spl, is_log=True, colors=colors, has_xlabel=row==(DD_keys[0][2] - 2), has_title=row==0))
                kw2.update(dict(raw_method=raw_method, show_legend=True, n_agents=temp_n_agents[i]))
                ax_CCS(**kw2)
    else:  # DD_keys has 1 entry: (3, 3, 3)
        for i in range((DD_keys[0][2] - 1)):  # col
            temp = gs[0 : 3, i * 3 + 1]
            axes["CCS"][i] = fig.add_subplot(temp)
            params = DD_keys[0]

            kw2 = dict(ax=axes["CCS"][i], x=beta_arr)
            kw2.update(dict(CCS_plot_type=CCS_plot_type))
            kw2.update(dict(params=params, key=key))
            kw2.update(dict(CCS_arr=DD[params]["CCS_arr"], noise=CCS_stat))
            kw2.update(dict(noise_ep=None, emp_lv=i + 1))
            kw2.update(dict(err_type=err_type, dpi=dpi, CCS_type=CCS_type))
            kw2.update(dict(spl=spl, is_log=True, colors=colors, has_xlabel=True))
            kw2.update(dict(raw_method=raw_method, show_legend=True, has_title=True))
            ax_CCS(**kw2)
    # panel label list
    regCCS2row = {0:2, 1:4, 2:3, 3:5, 4:5, 5:DD_keys[0][2] - 1, 6:2}  # k:v -> regCCS:num_of_rows
    # except for 4, which is regCCS:num_of_cols
    if regCCS in regCCS2row.keys():
        text_labels = [chr(ord("A") + i) for i in range(regCCS2row[regCCS])]
    else:
        raise NotImplementedError(f"<regCCS>={regCCS} is not implemented")
    for i in range(len(text_labels)):
        if regCCS != 427:  # list label for each col
            axlabels = [fig.add_subplot(gs[i * 4 : i * 4 + 3, 0])]
        else:  # list label for each col; currently unused
            axlabels = [fig.add_subplot(gs[0 : 3, i * 3])]
        if varb:
            axlabels.append(fig2.add_subplot(gs[i * 4 : i * 4 + 3, 0]))
        for axlabel in axlabels:
            axlabel.set_frame_on(False)
            axlabel.set_axis_off()  # same as ax.axis('off')
            kw5 = dict(fontsize=FS_MAIN, horizontalalignment="center", transform=axlabel.transAxes)
            axlabel.text(-2.4, 1.05, f"{text_labels[i]}", **kw5)
    fname += "_sim" if beta_arr2 is None else "_ep"
    saveNclose427(fig, fname, dpi=dpi, sub_folder_name=sub_folder_name)
    if varb:
        saveNclose427(fig2, fname + "_var", dpi=dpi, sub_folder_name=sub_folder_name)


def plot_corr(params, noise, colors, sub_folder_name, res=1000, res_grid=1000, dpi=None, truncated=False):
    """
    in plotting: show only significant ones (which excludes p>0.05 or p=np.nan)

    Kwargs
    ------
    - res (int): resolution (num of points) of MC sampling
    - res_grid (int): resolution (num of points) on n_samp_grid
    - truncated (bool): if True, only zoom in on xlim=(20,n_agents) and ylim=(-1,0)
    """
    spl = 0
    (regType, p, n) = params
    n_level = n-1
    cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:n_level], N=n_level)
    n_agents = noise["raw"][params][spl, 0, :, :].size  # total number of points in noise all beta
    # n_samp_grid = np.geomspace(1, n_agents, num=res_grid, dtype=int)
    # ASSUME n_agents = 100000 (or any number way larger than 100) AND res_grid > 200
    n_samp_grid = np.geomspace(201, n_agents, num=res_grid - 200, dtype=int)
    n_samp_grid = np.concatenate((np.arange(1, 201), n_samp_grid))
    if truncated:
        xlim = (20, n_agents)
        ylim = (-1, 0)
        yticks = np.linspace(-1, 0, 11)
        legend_loc = "lower right"
    else:
        xlim = (1, n_agents)
        ylim = (-1, 1)
        yticks = np.linspace(-1, 1, 21)
        legend_loc = "upper right"

    fname_s = f"spearr=({params[0]:d},{params[1]:d},{params[2]:d})"
    fname_p = f"spearp=({params[0]:d},{params[1]:d},{params[2]:d})"
    prefix = "input\\npy_files\\"
    temp_fname_s = fname_s + f"_res={res:d}_200+geomspace=(101,{n_agents:d},{res_grid:d})"
    temp_fname_p = fname_p + f"_res={res:d}_200+geomspace=(101,{n_agents:d},{res_grid:d})"
    try:
        sr = np.load(f"{prefix}{temp_fname_s}.npy", allow_pickle=True)
        sp = np.load(f"{prefix}{temp_fname_p}.npy", allow_pickle=True)
    except OSError:  # no file, calculate and save
        print(f"DEBUG running FX_MC427()")
        sr, sp = FX_MC427(noise, params, res=res, n_samp_grid=n_samp_grid, all_beta=True)
        np.save(f"{prefix}{temp_fname_s}.npy", sr)
        np.save(f"{prefix}{temp_fname_p}.npy", sp)
    print(f"DEBUG {temp_fname_s} done")
    print(f"DEBUG {temp_fname_p} done")

    fig = plt.figure(figsize=[6.5 * (n_level - 1), 4.5 * 1])  # initialize
    ds = 0.2  # dummy axes for spacing between the visible plots
    hf = 9  # relative height of sub-figure
    height_ratios = ([1, hf, ds, 1] * 1)[:-1]  # remove last sub-row
    width_ratios = [ds, 19, ds] * (n_level - 1)
    axes = [None] * (n_level - 1)

    kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
    kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
    gs = GridSpec(**kw1)
    sty1 = dict(alpha=0.74, linewidth=2)
    sty2 = dict(alpha=0.74, linewidth=0)

    for i in range(n_level - 1):  # do it for each two CCS lvs
        axes[i] = fig.add_subplot(gs[0 : 3, i * 3 + 1])
        sty1.update(dict(color=cmap(i), label=f"Median of $r_s$(lv{i+1},lv{i+2})"))
        # dim is n_samp_grid by res_grid; sort by corr, ascending; if percentile(), then no need to sort
        # temp_idx = np.argsort(sr[:, i, :], axis=-1)
        # temp_r = np.take_along_axis(sr[:, i, :], temp_idx, axis=-1)
        # temp_p = np.take_along_axis(sp[:, i, :], temp_idx, axis=-1)
        temp_r = sr[:, i, :]
        temp_p = sp[:, i, :]
        mask = temp_p < 0.05  # p<0.05 (this excludes both p>=0.05 and p=np.nan)
        to_draw = np.sum(mask, axis=1) >= 1  # what x to draw: at least 1 point at that x
        x = n_samp_grid[to_draw]
        temp_r[np.invert(mask)] = np.nan  # set all excluded points to np.nan
        temp_r = temp_r[to_draw]  # removed rows that all cols are insignificant
        axes[i].plot(x, np.nanpercentile(temp_r, 50, axis=1), **sty1)  # median = 50%
        kw1 = dict(color="dimgrey", zorder=1, label="50% Interval", **sty2)
        kw1.update(dict(y1 = np.nanpercentile(temp_r, 25, axis=1)))  # Q1 = 25%
        kw1.update(dict(y2 = np.nanpercentile(temp_r, 75, axis=1)))  # Q3 = 75%
        axes[i].fill_between(x=x, **kw1)  # 50% CI: IQR
        kw2 = dict(color="darkgrey", zorder=2, label="95% Interval", **sty2)
        kw2.update(dict(y1 = np.nanpercentile(temp_r, 2.5, axis=1)))  # 2.5%
        kw2.update(dict(y2 = np.nanpercentile(temp_r, 97.5, axis=1)))  # 97.5%
        axes[i].fill_between(x=x, **kw2)  # 95% CI
        # kw3 = dict(color="red", zorder=2, label="p < 0.05", **sty2)
        # kw3.update(dict(y1 = temp_r[:, whatnot1]))  # pval crossing 1
        # kw3.update(dict(y2 = temp_r[:, whatnot2]))  # pval crossing 2
        # axes[i].fill_between(x=n_samp_grid, **kw3)  # pval<0.05 region
        sty1.update(dict(color="red", label=f"Empirical $r_s$(lv{i+1},lv{i+2})"))  # ECCS2 sample size = 80; r = -0.542
        axes[i].scatter([80], [-0.542], s=6, **sty1)
        # axes[i].plot([80, 80], [ylim[0], -0.542], "-", **sty1)
        # sty1.pop("label")
        # axes[i].plot([xlim[0], 80], [-0.542, -0.542], "-", **sty1)
        # sty1.update(dict(color="magenta", label="ECCS1"))  # ECCS1 sample size = 71; r = -0.457
        # axes[i].plot([71, 71], [ylim[0], -0.457], "-", **sty1)
        # sty1.pop("label")
        # axes[i].plot([xlim[0], 71], [-0.457, -0.457], "-", **sty1)
        axes[i].set_xscale("log")
        axes[i].set_title("Cross CCS Level Spearman Correlation")
        axes[i].set_xlim(xlim)
        axes[i].set_ylim(ylim)
        axes[i].set_yticks(yticks)
        axes[i].tick_params(labelsize=FS_TICK)
        axes[i].set_xlabel(r"Total Sample Size (Across All $\beta$)")
        axes[i].set_ylabel("Spearman Correlation")
        temp_li, temp_la = axes[i].get_legend_handles_labels()
        axes[i].legend(temp_li, temp_la, loc=legend_loc)

    temp_str = "_truncated" * truncated
    fname = f"corr{temp_str}_param=({params[0]:d},{params[1]:d},{params[2]:d})"
    saveNclose427(fig, fname, dpi=dpi, sub_folder_name=sub_folder_name)


def plot_power(params, noise, colors, sub_folder_name, res=1000, pvals=None, dpi=None):
    """
    NOTE the empirical sample sizes are hand coded (from ECCS427.py)

    Intermediary
    ------------
    - power_dict(dict): k:v -> k is same as those in DD
    v (3D nparr): dim=(b, s, c)

    Plotting
    row: beta bin | col: pval
    for each subplot:
        each color -> CCS lv
    """
    (regType, p, n) = params
    n_level = n-1
    cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:n_level], N=n_level)
    spl = 0
    n_samp_grid = np.arange(1, 100)
    n_samp_emp = [1, 1, 6, 7, 24, 22, 16, 2, 1, 0]  # hand coded
    if pvals is None:
        # pvals = [0.05, 0.01, 0.001]
        pvals = [0.05, 0.01]
    vs = [None] * len(pvals)
    fname = f"powers=({params[0]:d},{params[1]:d},{params[2]:d})"
    prefix = "input\\npy_files\\"
    for i in range(len(pvals)):
        temp_fname = fname + f"_pval={len(pval_star(pvals[i]-1e-10, star=True)):d}stars"
        try:
            vs[i] = np.load(f"{prefix}{temp_fname}.npy", allow_pickle=True)
        except OSError:  # no file, calculate and save
            vs[i] = powers_MC427(noise, params, res=res, n_samp_grid=n_samp_grid, pval=pvals[i])
            np.save(f"{prefix}{temp_fname}.npy", vs[i])
        print(f"DEBUG pval={pvals[i]:.5f} done")
    # plot the power
    if len(pvals) == 1:  # 2 row by 3 col; only show when emp samp size > 1, which is 6 bins
        fname += "emp_samp_geq2"
        fig = plt.figure(figsize=[6.5 * 3, 4.5 * 2])  # initialize
        ds = 0.2  # dummy axes for spacing between the visible plots
        hf = 9  # relative height of sub-figure
        height_ratios = ([1, hf, ds, 2] * 2)[:-1]  # remove last sub-row
        width_ratios = [ds, 19, ds] * 3
        axes = [[None] * 3] * 2  # dim=(2,3)

        kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
        kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
        gs = GridSpec(**kw1)
        sty1 = dict(alpha=0.74, linewidth=2)

        xtick_loc = np.arange(n_samp_grid[0] - 1, n_samp_grid[-1] + 2, 5)  # both major and minor
        xtick_loc_minor = [x for t, x in enumerate(xtick_loc) if t%2==1]
        xtick_loc_major = [x for t, x in enumerate(xtick_loc) if t%2==0]
        xtick_lab_minor = [f"{x:.0f}" for x in xtick_loc_minor]
        xtick_lab_major = [f"{x:.0f}" for x in xtick_loc_major]

        b = len(noise["mean"][params][spl, 0, :])  # beta arr
        for i in range(2):  # row loop
            for j in range(3):  # col loop
                pad = 0
                axes[i][j] = fig.add_subplot(gs[i * 4 : i * 4 + 3, j * 3 + 1])
                axes[i][j].set_xlim((n_samp_grid[0] - 1, n_samp_grid[-1] + 1))
                axes[i][j].set_ylim((0 - pad, 1 + pad))  # power is prob, thus [0, 1]
                axes[i][j].set_xticks(xtick_loc_minor, minor=True)
                axes[i][j].set_xticks(xtick_loc_major, minor=False)
                # axes[i][j].set_xticklabels(xtick_lab_minor, minor=True)
                axes[i][j].set_xticklabels(xtick_lab_major, minor=False)
                axes[i][j].set_yticks(np.arange(0, 1.1, 0.1))
                axes[i][j].tick_params(labelsize=FS_TICK)

                bidx = i * 3 + j + 2  # beta bin idx start from 0; but display to start from 1
                axes[i][j].set_title(f"Beta Bin Index {bidx+1:d}", fontsize=FS_MAIN)
                if i == 1:  # 2nd row
                    axes[i][j].set_xlabel("Sample Size", fontsize=FS_MAIN)
                for l in range(n_level):
                    sty1.update(dict(color=cmap(l), label=f"lv{l+1}/lv{l+2}"))
                    axes[i][j].plot(n_samp_grid, vs[0][bidx, :, l], **sty1)
                xlim = axes[i][j].get_xlim()
                axes[i][j].plot(xlim, (0.95, 0.95), "--", color="black", zorder=0, label="95% Power")
                axes[i][j].plot(xlim, (0.9, 0.9), "--", color="dimgrey", zorder=0, label="90% Power")
                axes[i][j].plot(xlim, (0.8, 0.8), "--", color="grey", zorder=0, label="80% Power")
                xlim = (n_samp_emp[bidx], n_samp_emp[bidx])
                if xlim[0] > 1:  # emp samp size > 1
                    axes[i][j].plot(xlim, (0, 1), "--", color="red", zorder=0, label="Empirical Sample Size")
                if i == 0 and j == 1:  # show legend only in 2nd panel
                    temp_li, temp_la = axes[i][j].get_legend_handles_labels()
                    axes[i][j].legend(temp_li, temp_la, loc="lower right")
                axes[i][j].set_ylabel("Power", fontsize=FS_LAB)
    else:  # each column corresponds to a pval
        fig = plt.figure(figsize=[6.5 * (len(pvals)), 4.5 * 10])  # initialize
        ds = 0.2  # dummy axes for spacing between the visible plots
        hf = 9  # relative height of sub-figure
        height_ratios = ([1, hf, ds, 1] * 10)[:-1]  # remove last sub-row
        width_ratios = [ds, 19, ds] * (len(pvals))
        axes = [[None] * len(pvals)] * 10  # dim=(10,len(pvals))

        kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
        kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
        gs = GridSpec(**kw1)
        sty1 = dict(alpha=0.74, linewidth=2)

        xtick_loc = np.arange(n_samp_grid[0] - 1, n_samp_grid[-1] + 2, 5)  # both major and minor
        xtick_loc_minor = [x for t, x in enumerate(xtick_loc) if t%2==1]
        xtick_loc_major = [x for t, x in enumerate(xtick_loc) if t%2==0]
        xtick_lab_minor = [f"{x:.0f}" for x in xtick_loc_minor]
        xtick_lab_major = [f"{x:.0f}" for x in xtick_loc_major]

        b = len(noise["mean"][params][spl, 0, :])  # beta arr
        for i in range(b):  # row loop
            for j in range(len(pvals)):  # col loop
                axes[i][j] = fig.add_subplot(gs[i * 4 : i * 4 + 3, j * 3 + 1])
                axes[i][j].set_xlim((n_samp_grid[0] - 1, n_samp_grid[-1] + 1))
                axes[i][j].set_ylim((0, 1))  # power is prob, thus [0, 1]
                axes[i][j].set_xticks(xtick_loc_minor, minor=True)
                axes[i][j].set_xticks(xtick_loc_major, minor=False)
                # axes[i][j].set_xticklabels(xtick_lab_minor, minor=True)
                axes[i][j].set_xticklabels(xtick_lab_major, minor=False)
                axes[i][j].set_yticks(np.arange(0, 1.1, 0.1))
                axes[i][j].tick_params(labelsize=FS_TICK)
                axes[i][j].set_title(f"Beta Bin Index {i:d} (p={pvals[j]:.5g})", fontsize=FS_MAIN)
                if i == b - 1:
                    axes[i][j].set_xlabel("Sample Size", fontsize=FS_MAIN)
                for l in range(n_level):
                    sty1.update(dict(color=cmap(l), label=f"lv{l+1}/lv{l+2}"))
                    axes[i][j].plot(n_samp_grid, vs[j][i, :, l], **sty1)
                xlim = axes[i][j].get_xlim()
                axes[i][j].plot(xlim, (0.95, 0.95), "--", color="black", zorder=0, label="95% Power")
                axes[i][j].plot(xlim, (0.9, 0.9), "--", color="dimgrey", zorder=0, label="90% Power")
                axes[i][j].plot(xlim, (0.8, 0.8), "--", color="grey", zorder=0, label="80% Power")
                xlim = (n_samp_emp[i], n_samp_emp[i])
                if xlim[0] > 0:
                    axes[i][j].plot(xlim, (0, 1), "--", color="red", zorder=0)  # emp
                # axes[i][j].set_xscale("log")
                axes[i][j].set_ylabel("Power", fontsize=FS_LAB)

    saveNclose427(fig, fname, dpi=dpi, sub_folder_name=sub_folder_name)


def plot_Graph(tup, beta_arr, dpi=None, sub_folder_name="", transparent=True, annotate=None):
    """
    generates a graph for each beta in beta_arr

    Kwargs
    ------
    - sub_folder_name (str):
        the name of the folder in f"output/{whatnot}/" to store the plots
        where <whatnot> is defined in saveNclose427()
    - transparent (bool): whether we would use transparent background for output
    """
    if sub_folder_name == "":
        sub_folder_name = "graphs_singletons"
    regType, p, n = tup  # unpack tup
    DD = plot_side(beta_arr=beta_arr, beta_arr2=None)(tup)  # TODO needs to have proper beta_arr2
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    height_ratios, width_ratios = [1, 4, 5], [17, 1]
    kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
    kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
    gs = GridSpec(**kw1)

    # calculate A_hat based on analytic prediction
    eigvals, eigvecs = np.linalg.eigh(DD[tup]["GTDict"]["A"])
    eigvals = np.diag(eigvals)  # convert eigval list into eigval matrix 𝚲
    for b in beta_arr:
        if np.isinf(b):  # inf is the same as ground truth
            fname = f"SierpinskiGraph(beta=inf,regType={regType},p={p},n={n})"
            A = DD[tup]["GTDict"]["A"]
        else:
            fname = f"SierpinskiGraph(beta={b:.3f},regType={regType},p={p},n={n})"
            Lambda = (1 - np.exp(-b)) * eigvals / (1 - np.exp(-b) * eigvals)
            A = eigvecs @ Lambda @ (eigvecs.T)  # this works if A is symmetric (regularized)

        fig = plt.figure(figsize=[6, 4.5])  # initialize
        ax = fig.add_subplot(gs[:, :])
        axcb = fig.add_subplot(gs[1, -1])  # Edge Weight colorbar
        # axt = fig.add_subplot(gs[0, :])  # title axis
        kw3 = dict(ax=ax, axcb=axcb, axt=None, fig=fig, dpi=dpi, bA=[b, A])
        kw3.update(dict(params=tup, colors=colors, annotate=annotate))
        kw3.update(dict(nodeList=DD[tup]["Sier"].nodeList, GTDict=DD[tup]["GTDict"]))
        ax_Graph(**kw3)
        saveNclose427(fig, fname, dpi=dpi, sub_folder_name=sub_folder_name, transparent=transparent)


def ax_CCS(ax, x, CCS_arr, params, key, CCS_plot_type="CCS", raw_method=None, emp_lv=None,
           noise=None, spl=-1, varb=False, err_type='ste', CCS_type='mean', show_legend=False,
           is_log=True, colors=None, dpi=None, has_xlabel=True, has_title=True, noise_ep=None, n_agents=None):
    """
    Args
    ----
    - ax: axis object
    - x: a list of beta
    - params (tuple): (regType, p, n)
        - regType (int):
            0: default Sierpiński graph
            x: Sierpiński-like graph of type x regularization
    - key (str): 'n','p','reg_n', or 'reg_p', this only affects ax.set_ylim() line

    Kwargs
    ------
    - CCS_plot_type (str):
        - "CCS": vanilla
        - "sum": sum of CCS across levels
            propagation of uncertainty is involved;
            f = aA + bB | s=sd | sA=sd for A | sAB=cov for A & B (correlation)
            s = sqrt(a^2*sA^2 + b^2*sB^2 + 2ab*sAB)
            even though input could be standard error, output is standard deviation (assumption 2)
            assumptions:
            1. errors for different levels are uncorrelated (could calculate it in RW_CCS_stat.py)
            2. same formula for both standard deviation and standard error
            f = A + B
            then s = sqrt(sA^2 + sB^2)
    - raw_method (str): see plot_Graph_CCS() docstring
    - emp_lv (int): if not None, this will be the CCS lv to plot
    - noise (dict of 3D nparr): noise['mean'][params][s,i,beta]
        it is a synonym for CCS_stat (see doc in RW_CCS_stat.py)
    - spl (int or list): what indices of sample to draw data from CCS_stat; -1 is the last sample
    - n_agents (None or int): number of agents to be included
    - varb (bool): whether we draw negative beta (variable beta) or not
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    - show_legend (bool): whether we show simulation parameters
    - is_log (bool): if True then use log scale on x axis
    - has_xlabel/has_title (bool): whether we show xlabel/title
    - noise_ep (dict of 3D/4D nparr): numerical approximation using exposure theory
        for point estimation, it's 3D; for pdf and support, it's 4D obviously
        the CCS plot will not involve any simulation results if noise_ep is not None
        noise_ep['mean'][s,i,beta]
        s: similar to simulation, s should correspond to that in simulation; s=0 <-> T=1500, etc
        i: CCS level
        beta: beta, what else could it mean

    Intermediary
    ------------
    variable beta case:
        -whatnot[-9:-1] contains num of beta = [1, 2, 3, 4, 5, 6, 8, 10, 12] (abandoned)
        whatnot[-1] = -1, which means (n-1) number of beta
        <whatnot> := noise[err_type][params][-1, 0, :],
        which is a list of actual beta (including negative codename beta)

    Return
    ------
    - xmaxs (list): list of beta that maximizes CCS at each level
        xmaxs[1]: beta that maximizes CCS at lv2/lv3
        xmaxs[i]: beta that maximizes CCS at lv(i+1)/lv(i+2)
    """
    # set up beta range (analytical) for plot
    if is_log:
        x_range = (min(x) * 1.00, max(x) * 1.00)
        x_scale = 10  # base used for get_violin_pw(); same as when beta was first initialized
    else:
        delta = (max(x) - min(x)) * 0.05
        x_range = (min(x) - delta, max(x) + delta)
        x_scale = None  # linear
    ax.set_xlim(x_range)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    (regType, p, n) = params
    n_level = CCS_arr.shape[2]
    if n_level == n-1:  # normally there are only n-1* levels (since CCS is function of 2 lvs)
        pass
    elif n_level == n:  # *it could be violated, due to regularization
        n_level -= 1  # don't show higher level introduced by regularization
    else:
        raise Exception("something went wrong in <CCS_arr.shape[2]>")
    if CCS_type == "mean":
        CCS_type_slice = 0
    else:
        CCS_type_slice = 1
    cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:n_level], N=n_level)
    xlabel = r"Shuffling Parameter $\beta$"
    ylabel = "CCS"
    if CCS_plot_type == "sum":
        ylabel = "Sum of CCS Across Levels"
    if regType in [0, 1, 2, 3]:
        title = fr"CCS of $^{regType}S_{p:d}^{n:d}$"
    else:
        raise NotImplementedError(f"<regType>={regType} is not implemented")
    if n_agents is not None:
        title += f", n={n_agents:d} per Bin"


    if noise is not None:
        bs = None  # find where <0 beta starts (negative index)
        d = noise["mean"][params].shape[2]  # total number of beta
        for i in range(-1, -d - 1, -1):  # count from right because that's where <0 beta lies
            if noise["mean"][params][spl, 0, i] > 0:
                bs = i + 1
                break
        # if bs != -1:  # not sure why such condition; it's varb case, currently unused anyway
        #     import warnings  # may want to comment the warnings out since it gets verbose
        #     msg = "only <noise[\"mean\"][params][spl, 0, -1]> can be < 0\n"
        #     msg += f"currently <noise[\"mean\"][params][spl, 0, {bs}:]> are all < 0\n"
        #     msg += "using last val instead"
        #     warnings.warn(msg)
        #     bs = -1  # force use last val
        # bs1 = slice(bs, None)  # if no <0 beta, slice(bs, None) = [0:], which is >0 beta
        bs2 = slice(0, d + bs)  # get >0 beta slice object

    if noise is not None:  # ax.text()
        styles_txt = dict(fontsize=FS_LAB, horizontalalignment="right", transform=ax.transAxes, zorder=4)
        topy, s = 0.94, 0.06
        if show_legend:
            if emp_lv is None:  # show walk length too
                # n_agents = noise["mean"][params][spl, 1, 0]  # since all β have same group size, take 0
                n_steps = noise["mean"][params][spl, 2, 0]  # ditto but w/ walk length
                # beta_type = "both" if varb else "constant"
                ax.text(0.99, topy - s * (n_level + 2), f"walk length={n_steps:.0f}", **styles_txt)
                # if raw_method[-6:] == "median":  # show errorbar type to be ste_median
                #     ax.text(0.5, topy - 2 * s, "errorbar=standard error", **styles_txt)
                # else: # identical: standard error (which is std of the statistic)
                #     ax.text(0.5, topy - 2 * s, "errorbar=standard error", **styles_txt)
                # ax.text(0.5, topy - 3 * s, f"CCS_type={CCS_type}", **styles_txt)
                # ax.text(0.5, topy - 4 * s, f"beta_type={beta_type}", **styles_txt)
            # below shows agents per bin
            n_per_bin = noise["raw"][params][spl, 0, 0, :n_agents].shape[0]  # all emp_lv are same
            # ax.text(0.99, topy - s * (n_level + 3), f"n={n_per_bin:.0f} per bin", **styles_txt)
        else:
            if emp_lv is None:  # only show corr between one CCS lv and the next
                for i in range(n_level - 1):
                    n_steps = noise["mean"][params][spl, 2, 0]  # ditto but w/ walk length
                    temp_shape = noise["raw"][params][spl, i:i+2, bs2, :n_agents].shape
                    new_shape = (temp_shape[0], temp_shape[1] * temp_shape[2])
                    temp = np.reshape(noise["raw"][params][spl, i:i+2, bs2, :n_agents], new_shape)
                    spearmanr = stats.spearmanr(temp, axis=1)
                    temp_var = f"$r_s$(lv{i+1},lv{i+2})={spearmanr.correlation:.3f} p={spearmanr.pvalue:.3f}"
                    ax.text(0.99, topy - s * i, temp_var, **styles_txt)
            else:  # show Wilcoxon signed-rank tests (both per bin and global)
                temp = noise["raw"][params][spl, emp_lv - 1, bs2, :n_agents]
                temp_betas = noise["mean"][params][spl, 0, bs2]
                # global
                diff = np.ravel(temp) - 1  # compared to baseline of 1
                result = stats.wilcoxon(diff, alternative="greater")
                wstat, pval = result.statistic, result.pvalue
                # ax.text(0.99, topy - s * 0, f"Wilcoxon={wstat:d} {pval_star(pval, star=False)}", **styles_txt)
                ax.text(0.99, topy - s * 0, f"Wilcoxon {pval_star(pval, star=False)}", **styles_txt)
                pval2 = 1 - sum(diff > 0) / diff.shape[0]
                # ax.text(0.99, topy - s * 1, f"Proportion {pval_star(pval2, star=False)}", **styles_txt)
                ax.text(0.99, topy - s * 1, f"Proportion p={pval2:.3f}", **styles_txt)
                # per bin
                styles_txt.pop("transform", None)
                styles_txt.update(dict(horizontalalignment="center"))
                ax.text(6e-4, 0.954, f"Wilc.", **styles_txt)
                ax.text(6e-4, 0.924, f"Prop.", **styles_txt)
                for i in range(temp.shape[0]):  # loop over each bin (beta)
                    diff = temp[i, :] - 1  # compared to baseline of 1
                    result = stats.wilcoxon(diff, alternative="greater")
                    wstat, pval = result.statistic, result.pvalue
                    ax.text(temp_betas[i], 0.95, pval_star(pval, star=True), **styles_txt)
                    pval2 = 1 - sum(diff > 0) / diff.shape[0]
                    ax.text(temp_betas[i], 0.92, pval_star(pval2, star=True), **styles_txt)

    sty1 = dict(alpha= 0.74, linewidth=2, zorder=0)

    def temp_a1(i):  # draw CCS: analytical, noise constant, noise dynamic
        sty1.update(dict(color=cmap(i)))
        if show_legend:
            sty1.update(dict(label=f"lv{i+1}/lv{i+2}"))
        ax.plot(x, CCS_arr[:, CCS_type_slice, i], **sty1)  # plot analytical curve
        if noise is not None:  # put scatter points
            if varb:  # plot horizontal line; only plot one dynamic beta; draw it first
                kw1 = dict(color=cmap(i), zorder=1)
                y_mean = noise["mean"][params][spl, 3 + i, bs]
                y_err = noise[err_type][params][spl, 3 + i, bs]
                ax.plot(ax.get_xlim(), [y_mean] * 2, **kw1)  # draw line
                kw1.update(dict(y1 = [y_mean - y_err] * 2, y2 = [y_mean + y_err] * 2))
                ax.fill_between(x=ax.get_xlim(), **kw1, alpha=0.27)  # draw colored region
            # scatter plot
            sty2 = dict(linestyle="None", capsize=4.0, marker=".", markersize=11)
            sty2.update(dict(alpha = 0.74, linewidth = 2, zorder=2))
            sty2.update(dict(markeredgecolor=cmap(i), markerfacecolor=cmap(i), ecolor=cmap(i)))
            # if show_legend:
            #     sty2.update(dict(label='Stochastic '+labels[i].replace('-','/lv')))
            if raw_method is None:
                return 0
            elif raw_method.startswith("box"):  # box plot
                bdata = [noise["raw"][params][spl, i, b, :n_agents] for b in range(bs2.start, bs2.stop)]
                positions = noise["mean"][params][spl, 0, bs2]  # beta
                max_ECCS = [max(j) for j in bdata]
                bbdry = make_beta_boundry(positions.shape[0], b=10)  # log-uniform
                if is_log:  # create a dummy axis and plot boxes on that axis
                    ax_bx = ax.twiny()  # instantiate a separate x-axis for equal spacing boxes
                    # below position kwarg are crucial steps, turn exp to linear (log scale)
                    widths = log_b(bbdry[0, 2], 10) - log_b(bbdry[0, 1], 10)  # full width
                    kwargs = dict(widths=widths, manage_ticks=False, whis=(2.5, 97.5), zorder=1)  # 95% range whisker
                    ax_bx.set_xlim(log_b(ax.get_xlim(), 10))  # s.t. box matches the scatter
                    parts = ax_bx.boxplot(bdata, positions=log_b(positions, 10), showfliers=False, **kwargs)
                    ax_bx.xaxis.set_visible(False)  # hide twin axis, and doesn't take up space
                if True: # match box colors to scatter
                    for pc in parts.values():
                        for line in pc:
                            line.set_color(cmap(i))
                            line.set_alpha(1)
            elif raw_method.startswith("violin"):  # stochastic violin (overlaid)
                # TODO haven't implemented it in temp_a2()
                data_vl = [noise["raw"][params][spl, i, b, :n_agents] for b in range(bs2.start, bs2.stop)]
                pos_vl = noise["mean"][params][spl, 0, bs2]  # beta
                pos_vl, widths = get_violin_pw(pos_vl, x_scale=x_scale)
                kw_vl = dict(showmeans=False, showmedians=False, showextrema=False, widths=widths)
                if is_log:  # create a dummy axis and plot violins on that axis
                    ax_vl = ax.twiny()  # instantiate a separate x-axis for equal spacing violins
                    parts = ax_vl.violinplot(data_vl, positions=pos_vl, **kw_vl)
                    x_range_vl = log_b(ax.get_xlim(), x_scale)
                    ax_vl.set_xlim(x_range_vl)  # crucial step, map ax_vl data loc onto x
                    ax_vl.xaxis.set_visible(False)  # hide twin axis, and doesn't take up space
                    # NOTE: ax_vl.set_axis_off() also hides it, but the space is still reserved
                else:
                    parts = ax.violinplot(data_vl, positions=pos_vl, **kw_vl)
                for pc in parts['bodies']:
                    pc.set_facecolor(cmap(i))
                    pc.set_alpha(0.27)
                if raw_method[-6:] == "median" and noise_ep is None:  # plot median and ste_median instead
                    kw_erb.update(dict(y=noise["median"][params][spl, 3 + i, bs2]))
                    kw_erb.update(dict(yerr=noise["ste_median"][params][spl, 3 + i, bs2]))

            else:
                msg2 = "<raw_method>: currently only supports scatter(None)/violin('violin') plot"
                raise ValueError(msg2)

            # simulated point estimate: mean/median + std/ste/ste_median
            if not raw_method.startswith("box"):  # no need for this if box plot
                kw_erb = dict(x=noise["mean"][params][spl, 0, bs2])
                # calculate on the fly, instead of using noise["mean"][params][spl, 3 + i, bs2])
                # kw_erb.update(dict(y=noise["mean"][params][spl, 3 + i, bs2]))
                # kw_erb.update(dict(yerr=noise[err_type][params][spl, 3 + i, bs2]))
                kw_erb.update(dict(y=np.mean(noise["raw"][params][spl, i, bs2, :n_agents], axis=-1)))
                kw_erb.update(dict(yerr=np.std(noise["raw"][params][spl, i, bs2, :n_agents], axis=-1)))
                ax.errorbar(**kw_erb, **sty2)  # python 3.5+ PEP 448 (Unpacking Generalizations)
            if noise_ep is not None:  # ideal violin (overlaid): pdf; numerical approximation (deprecated)
                kw_erb = dict(x=noise_ep["beta"])  # beta
                kw_erb.update(dict(y=noise_ep["mean"][spl, i, :]))  # mean of CCS
                kw_erb.update(dict(yerr=noise_ep["std"][spl, i, :]))  # std of CCS
                kw_temp = dict(color=cmap(i), alpha=0.27, linewidth=0)
                bbdry = make_beta_boundry(noise_ep["beta"].shape[0], b=10)  # log-uniform
                for b, beta in enumerate(bbdry[:, 0]):
                    x0 = noise_ep["ps"][spl, i, b, :]  # min-max rescaled pdf
                    x0 = x0 / TODO(x0, noise_ep["rs"][spl, i, b, :])  # TODO norm by a variable
                    idx0, idx1 = 0, None
                    for w in range(len(x0)):  # find the middle where p > 0
                        if not np.isclose(x0[w], 0, rtol=0, atol=1e-10):
                            idx0 = w
                            break
                    for w in range(len(x0)):  # find the middle where p > 0
                        if not np.isclose(x0[-w - 1], 0, rtol=0, atol=1e-10):
                            idx1 = -w - 1
                            break
                    idx1 = None if idx1 == -1 else idx1 + 1
                    kw_temp["y"] = noise_ep["rs"][spl, i, b, idx0:idx1]
                    x0 = x0[idx0:idx1]
                    kw_temp["x1"] = bbdry[b, 1] + (1 - x0) * (bbdry[b, 0] - bbdry[b, 1])
                    kw_temp["x2"] = bbdry[b, 0] + x0 * (bbdry[b, 2] - bbdry[b, 0])
                    ax.fill_betweenx(**kw_temp)
                ax.errorbar(**kw_erb, **sty2)  # python 3.5+ PEP 448 (Unpacking Generalizations)
    
    def temp_a2():  # draw total CCS: analytical, noise constant, noise dynamic
        sty1.update(dict(color=cmap(0)))
        if show_legend:
            sty1.update(dict(label="Total CCS"))
        ts1 = slice(0, n_level)
        ysal = np.sum(CCS_arr[:, CCS_type_slice, ts1], axis=1)  # sum across levels
        ax.plot(x, ysal, **sty1)  # plot analytical curve
        if noise is not None:  # put scatter points of simulated results in
            ts2 = slice(3, 3 + n_level)
            if varb:  # plot horizontal line; only plot one dynamic beta; draw it first
                kw1 = dict(color=cmap(0), zorder=1)
                ysal_mean = np.sum(noise["mean"][params][spl, ts2, bs], axis=0)
                ysal_err = np.sqrt(np.sum(noise[err_type][params][spl, ts2, bs]**2, axis=0))
                ax.plot(ax.get_xlim(), [ysal_mean] * 2, **kw1)  # draw line
                kw1.update(dict(y1 = [ysal_mean - ysal_err] * 2, y2 = [ysal_mean + ysal_err] * 2))
                ax.fill_between(x=ax.get_xlim(), **kw1, alpha=0.27)  # draw colored region
            # plot fixed beta
            if raw_method is None:  # scatter plot
                sty2 = dict(linestyle="None", capsize=4.0, marker=".", markersize=11)
                sty2.update(dict(alpha = 0.74, linewidth = 2, zorder=2))
                sty2.update(dict(markeredgecolor=cmap(0), markerfacecolor=cmap(0), ecolor=cmap(0)))
                # if show_legend:
                #     sty2.update(dict(label='Stochastic '+labels[i].replace('-','/lv')))
                kw_erb = dict(x=noise["mean"][params][spl, 0, bs2])
                kw_erb.update(dict(
                    y=np.sum(noise["mean"][params][spl, ts2, bs2], axis=0),
                    yerr=np.sqrt(np.sum(noise[err_type][params][spl, ts2, bs2]**2, axis=0))
                    ))
                ax.errorbar(**kw_erb, **sty2)  # python 3.5+ PEP 448 (Unpacking Generalizations)

    def temp_b1(i):  # annotate CCS plot with maxima
        xmaxs[i] = x[np.argmax(CCS_arr[:, CCS_type_slice, i])]
        ymaxs[i] = np.max(CCS_arr[:, CCS_type_slice, i])
        # texts[i] = f"({xmaxs[i]:.3f},{ymaxs[i]:.3f})"  # show both beta and CCS
        texts[i] = f"β={xmaxs[i]:.3f}"  # show only beta
        kw_text.update(dict(color=cmap(i), xy=(xmaxs[i], ymaxs[i])))
        if noise is None:
            kw_text.update(dict(xytext=(0.85 - i * 0.2, 0.90)))
        else:
            kw_text.update(dict(xytext=(0.85 - i * 0.2, 0.85)))
        ax.annotate(texts[i], **kw_text)

    def temp_b2():  # annotate total CCS plot with maxima
        ysal = np.sum(CCS_arr[:, CCS_type_slice, :n_level], axis=1)  # sum across levels
        xmaxs = x[np.argmax(ysal)]
        ymaxs = np.max(ysal)
        # texts = f"({xmaxs:.3f},{ymaxs:.3f})"  # show both beta and CCS
        texts = f"{xmaxs:.3f}"  # show only beta
        kw_text.update(dict(color=cmap(0), xy=(xmaxs, ymaxs), xytext=(0.50, 0.05)))
        ax.annotate(texts, **kw_text)
        return xmaxs
    
    def temp_b3():  # annotate total CCS plot with mean beta for var_beta
        temp_fname = set_dir427() + "\\sim427\\input\\params_max_beta"
        max_beta_dict = get_params(fname=temp_fname, default_dir=False)  # from hi to low
        beta_mean = np.mean(max_beta_dict[f"{regType},{p},{n}"])
        blue_ = "cornflowerblue"
        styles3 = dict(label=r"mean $\beta$", color=blue_)
        ax.plot((beta_mean, beta_mean), ax.get_ylim(), "--", zorder=0, **styles3)
        kw_text.update(dict(color=blue_, xy=(beta_mean, ax.get_ylim()[0]), xytext=(0.75, 0.05)))
        ax.annotate(f"{beta_mean:.3f}", **kw_text)

    # argmax = beta that maximizes CCS at different permissible levels
    arrowprops = dict(arrowstyle="simple", facecolor="grey", edgecolor="grey")
    arrowprops.update(dict(linewidth=1 / 3, alpha=0.74))
    kw_text = dict(textcoords="axes fraction", fontsize=FS_LAB, arrowprops=arrowprops)
    kw_text.update(dict(ha="center", va="center"))

    if emp_lv is not None:
        xmaxs, ymaxs, texts = [None] * (n-1), [None] * (n-1), [None] * (n-1)
        temp_a1(emp_lv - 1)  # -1 because this is level not idx
        temp_b1(emp_lv - 1)  # put peak val on plot
        ax.set_ylim((0.9, 1.63))
        ax.plot(ax.get_xlim(), (1, 1), "--", color="grey", zorder=0)  # draw y=1 line in grey
    elif CCS_plot_type == "CCS":  # plot vanilla CCS graph
        xmaxs, ymaxs, texts = [None] * (n-1), [None] * (n-1), [None] * (n-1)
        for i in range(n_level):  # n_level is always n-1; see earlier code for why this is true
            temp_a1(i)
            temp_b1(i)  # put peak val on plot

        if key in ["n", "reg_n"]:
            ax.set_ylim((0.9, 1.3))  # for regType=3, p=3, n=3 max CCS is <1.3
        else:
            ax.set_ylim((0.9, 1.63))  # for regType=3, p=5, n=3 max CCS is about 1.63
        ax.plot(ax.get_xlim(), (1, 1), "--", color="grey", zorder=0)  # draw y=1 line in grey
    elif CCS_plot_type == "sum":  # plot summation CCS graph
        temp_a2()
        xmaxs = temp_b2()

        if key in ["n", "reg_n"]:
            delta2 = 0.5
            if n == 3:
                delta2 = 0.36  # for params=(3,3,3) max total CCS is ~2.36
            elif n == 4:
                delta2 = 0.50  # for params=(3,3,4) max total CCS is ~3.30
            elif n == 5:
                delta2 = 0.80  # for params=(3,3,5) max total CCS is ~4.25
        else:
            delta2 = 0.5
            if p == 3:
                delta2 = 0.36  # for params=(3,3,3) max total CCS is ~2.36
            elif p == 4:
                delta2 = 0.52  # for params=(3,4,3) max total CCS is ~2.52
            elif p == 5:
                delta2 = 0.75  # for params=(3,5,3) max total CCS is ~2.70
        ax.set_ylim((n_level-0.1, n_level + delta2))
        ax.plot(ax.get_xlim(), (n_level, n_level), "--", color="grey", zorder=0)
        temp_b3()
    else:
        raise NotImplementedError(f"<CCS_plot_type>=\"{CCS_plot_type}\" is not implemented yet")


    if has_title:
        ax.set_title(title, fontsize=FS_MAIN)
    if has_xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LAB)
    ax.set_ylabel(ylabel, fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    if is_log:
        ax.set_xscale("log")  # set x to log scale
        if CCS_plot_type == "CCS":
            loc="upper right"
        elif CCS_plot_type == "sum":
            loc="lower left"
        if noise is None:
            loc="upper left"
    else:
        loc="center right"
    temp_li, temp_la = ax.get_legend_handles_labels()
    if show_legend:
        l = ax.legend(temp_li, temp_la, loc=loc)
        l.set_zorder(100)  # on top of everything
    ax.grid(False)
    return xmaxs


def ax_Graph(ax, axcb, fig, params, nodeList, GTDict,
             axt=None, colors=None, dpi=None, annotate=None, bA=None):
    """
    Args
    ----
    - ax/axcb: axis object
    - axt: title axis
    - params (tuple): (regType, p, n)
    - nodeList: [(i,x,y),...] (x,y) is coordinate
    - GTDict (dict): contains 'A', 'edgeList', 'lvList' (all GroundTruth)
    - annotate (int):
        None: we don't label the nodes
        -1: Decimal
        p (0<p<10): base-p expansion
    - bA (list=[float, np.arr]):
        -bA[0]: beta
        -bA[1]: transition prob matrix to draw; if not None, draw a single graph of A
    """
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    (regType, p, n) = params
    # n is used here to scale node size properly
    # scale = 200 * (5*n**2+4*n)/(2.7**(1.46*n*p/3)-(n*2*(p/3))**2)
    # scale = 240 / np.log(0.1*(p+n)**n)
    scale = 240 / np.log(0.1 * (p + n) ** n) - n ** 2 / 2
    # num_nodes = round(p**n)
    all_levels = [x for x in range(1, n + 1)]  # get all levels, starting from 1, but may end at n+1
    nu = len(all_levels)  # num of all levels minus -1 level
    if regType == 1:  # regularized edges are level n+1
        all_levels.append(n + 1)
        nu += 1
    elif regType == 3:  # regularized edges are level -1
        all_levels.insert(0, -1)

    if regType in [0, 1, 2, 3]:
        title = "Sierpiński Graph of " + r"$^{}$".format(regType) + r"$S_{:d}^{:d}$".format(p, n)
    else:
        raise ValueError("<regType> is unclear.")

    # use default PMMM theme color
    # draw nodes
    if annotate is not None:
        scale *= 4.7  # make node larger to fit annotation
    marker_style = dict(
        facecolor="#f48ea5", edgecolor="#7f7596", marker="o", alpha=1, s=scale
    )  # previous CSS colors: lightcoral, cornflowerblue
    annokw = dict(horizontalalignment="center", verticalalignment="center", color="k", fontsize=10)
    if annotate is None:
        for i, x, y in nodeList:
            ax.scatter(x, y, zorder=2, **marker_style)
    else:
        base_p = annotate if annotate > 0 else 10
        for i, x, y in nodeList:
            ax.scatter(x, y, zorder=2, **marker_style)
            ax.annotate(str(p_ary(i, p=base_p, L=n)), xy=(x, y), xytext=(x, y), **annokw)
    # draw edges
    if bA is not None:  # a singleton graph with transition prob edge weight coloring
        beta, A = bA
        title += fr" ($\beta=+\infty$)" if np.isinf(beta) else fr" ($\beta={beta:.3f}$)"
        n_ = np.shape(A)[0]  # number of nodes
        RGBA = np.zeros((round(n_*(n_-1)/2), 4))
        RGBA[:,1] = 0.5 # for green (not sure if this is color 'g')
        max_prob = A.max()
        counter = 0
        for i in range(0,n_):
            for j in range(i+1,n_): # undirected (A is symmetric)
                # 🔴 assuming nodeList[i][0] = i
                x = [nodeList[i][1],nodeList[j][1]]
                y = [nodeList[i][2],nodeList[j][2]]
                RGBA[counter, 3] = A[i, j] / max_prob  # set alpha to 1 (opaque) if max prob
                ax.plot(x, y, color=RGBA[counter,:], zorder=1)
                counter += 1
        rgba, nu = [[0,0.5,0,0], [0,0.5,0,1]], 5
        cmap = LinearSegmentedColormap.from_list("custom edge color", rgba, N=nu)
        norm = Normalize(vmin=0, vmax=nu - 1)
        kwargs_cax = dict(cax=axcb, drawedges=False)
        cbr = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), **kwargs_cax)
        cbr.set_ticks([(nu - 1) * (2 * i + 1) / (2 * nu) for i in range(nu)])
        cbr.set_ticklabels([f"{x:.2f}" for x in np.linspace(0, max_prob, nu)])
        cbr.ax.tick_params(labelsize=FS_TICK)
        cbrLabel427(axcb, "transition probability")
    else:  # edge level coloring
        cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:nu], N=nu)
        norm = Normalize(vmin=0, vmax=nu - 1)
        cbr = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axcb)
        cbr.set_ticks([(nu - 1) * (2 * i + 1) / (2 * nu) for i in range(nu)])
        cbr.set_ticklabels([f"{x:d}" for x in np.arange(1, nu + 1)])
        cbr.ax.tick_params(labelsize=FS_TICK)
        cbrLabel427(axcb, "Edge Level")
        for lv in all_levels:
            b_ = [x == lv for x in GTDict["lvList"]]  # boolean mask
            b_edgeList = [e for (e, v) in zip(GTDict["edgeList"], b_) if v]  # edges in level lv
            xcoords, ycoords = np.zeros((2, len(b_edgeList))), np.zeros((2, len(b_edgeList)))
            for i, (v_i, v_j) in enumerate(b_edgeList):
                xcoords[:, i] = [nodeList[v_i][1], nodeList[v_j][1]]
                ycoords[:, i] = [nodeList[v_i][2], nodeList[v_j][2]]
            ax.plot(
                xcoords, ycoords, color=cmap(lv - 1), zorder=1
            )  # lower int -> drawn on the canvas earlier
    # Grid setting and save
    axcb.set_frame_on(False)
    # axcb.set_axis_off()  # same as ax.axis('off')
    ax.set_frame_on(False)
    ax.set_axis_off()  # same as ax.axis('off')
    ax.axis("equal")  # so that regular polygons appear to be regular as well
    if axt is None:  # use ax.set_title()
        if annotate is None:
            ax.set_title(title, fontsize=FS_MAIN)
        else:
            if annotate == -1:
                suffix = "Decimal)"
            else:
                suffix = f"Base-{base_p:d})"
            if bA is None:
                ax.set_title(title[20:-12] + suffix, fontsize=FS_MAIN)
            else:
                ax.set_title(title[:-1] + ", " + suffix, fontsize=FS_MAIN)
        ax.grid(False)
        # plt.legend(loc='upper left')
    else:
        axt.set_frame_on(False)
        axt.set_axis_off()
        kwargs_axt = dict(fontsize=FS_MAIN, transform=axt.transAxes)
        if annotate is None:
            axt.text(0, 0, title, **kwargs_axt)
        else:
            if annotate == -1:
                suffix = "Decimal)"
            else:
                suffix = f"Base-{base_p:d})"
            if bA is None:
                axt.text(0, 0, title[20:-12] + suffix, **kwargs_axt)
            else:
                axt.text(0, 0, title[:-1] + ", " + suffix, **kwargs_axt)
        ax.grid(False)
        # plt.legend(loc='upper left')


if __name__ == "__main__":
    main_Sierpinski427()
