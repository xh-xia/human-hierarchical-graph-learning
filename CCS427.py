"""
CCS visualizations.

Created: Monday, ‎April ‎5, ‎2021, ‏‎3:59:47 PM (EDT; maybe earlier actually)
@author: Xiaohuan (Pixel) X.
"""

from itertools import product  # for nested loops
from concurrent.futures import ProcessPoolExecutor  # for multi-processing
from concurrent.futures import ThreadPoolExecutor  # for multi-threading

from utility427.helper427 import set_dir427, mkdir_p, get_params, partial_427_decorator
from utility427.math427 import findNearest, rank_eigvals, W_norm, np
from utility427.plt427 import plt, Normalize, LinearSegmentedColormap, GridSpec  # matplotlib
from utility427.plt427 import saveNclose427, colors_selector, cbrLabel427  # matplotlib helpers
from utility427.plt427 import load_CCS_stat, save_masks
from utility427.Sierpinski427 import make_Sierpinski427, p_ary, make_SierpinskiGraph427
from utility427.sim_params427 import make_sim_params


def main_Sierpinski427():
    """
    get parameters from params_CCS427.json:
    - err_type (str): "std" or "ste", type of spread of CCS
    - CCS_type (str): "mean" or "std", type of stat as part of def of CCS
        use mean for the most part, as std is not steady
    - CCS_plot_type (str): "CCS" or "sum" or "both", type of y in CCS plot
        - CCS: vanilla method; show all level
        - sum: show only the sum of CCS across levels
    - dpi (int): used in plot generation; if None, default to both 300 and lossless .pdf
    - sub_fo_name (str): folder in "input" folder containing CCS_stat .npy files

    get parameters from sim427/input/params.json:
    - key_class (str): "reg_n_p" or something, only a label to classify batch of jobs in sim
    - n_agents (int): num of agents per subject parameter (like a repetition experiment)
    - var_betas: should be effectively the same as that in sim427/input/params.json


    - hierDict (dict): to recreate the graph parameters set (i.e., (regType, p, n))
    """
    temp_dir = set_dir427(dir_ = set_dir427() + "\\sim427\\input")
    p = get_params(fname=temp_dir + "\\params", default_dir=False)
    # overwrite key_class
    # can be whatever as long as it (was once defined in params.json) was run in sim427
    # otherwise will be error in load_CCS_stat(), which loads results generated from sim427
    # p["key_class"] = "max_beta"
    # p["key_class"] = "max_beta_hi2lo"

    make_sim_params(p)
    p.update(get_params(fname="params_CCS427"))
    set_dir427()  # go back to script dir
    # add more parameters into p
    p["colors"] = colors_selector(str="5-class Greens")
    p["beta_arr"] = np.geomspace(0.0001, 10, 400)  # for analytical curve only
    hierDict = dict()
    # hierDict['n'] = [[0],[3],[3,4,5]]
    # hierDict['p'] = [[3],[3,4,5],[3]]
    # hierDict["reg_n"] = [[0, 1, 2, 3], [3], [3, 4, 5]]
    # hierDict['reg_p'] = [[0,1,2,3],[3,4,5],[3]]
    hierDict['r'] = [[0,1,3],[3],[3]]
    kw_loop = dict(sub_fo_name=p["sub_fo_name"], CCS_type=p["CCS_type"], key_class=p["key_class"])
    kw_loop.update(dict(n_agents=p["n_agents"], err_type=p["err_type"], hierDict=hierDict))
    kw_loop.update(dict(beta_arr=p["beta_arr"], colors=p["colors"], dpi=p["dpi"]))
    kw_loop.update(dict(CCS_plot_type=p["CCS_plot_type"]))

    # single-processing | w/o mp in plot_main() ~155 sec
    for beta_class in p["beta_classes"]:
        plot_main(**kw_loop, mp=False)(beta_class)
    # multi-threading | w/o mp in plot_main() ~?? sec (too slow)
    # with ThreadPoolExecutor() as executor:
    #     executor.map(plot_main(**kw_loop, mp=False), p["beta_classes"])


@partial_427_decorator
def plot_main(beta_class, sub_fo_name, CCS_type, key_class, n_agents, hierDict,
              beta_arr, err_type, CCS_plot_type, colors, dpi, mp=False):
    """this function is I/O bound | suitable for multi-threading

    Kwarg
    -----
    mp (bool): whether to use multi-processing on plot_side() | doesn't work, so use False for now

    Intermediary
    ------------
    DD (dict): a data dict created only to be used in graphing (i.e., plot_Graph_CCS())
        most results from numerical calculations are put in DD
    """
    npy_sub_path = f"{sub_fo_name}\\"
    npy_sub_path += f"CCS_stat_{CCS_type}_{key_class}_{beta_class}_{n_agents}"
    CCS_stat = load_CCS_stat(fname=npy_sub_path)  # load sim results
    for key in hierDict.keys():
        kw_main = dict(beta_arr=beta_arr)
        if mp:
            with ProcessPoolExecutor() as executor:
                dd_list = executor.map(plot_side(**kw_main), product(*hierDict[key]))
        else:
            dd_list = map(plot_side(**kw_main), product(*hierDict[key]))
        DD = dict()
        for dd in dd_list:  # aggregate all keys of dd in dd_list into DD
            DD.update(dd)

        kw_plot = dict(CCS_stat=CCS_stat, err_type=err_type, CCS_type=CCS_type)
        kw_plot.update(dict(colors=colors, dpi=dpi, regCCS=len(key) > 1))
        kw_plot.update(dict(CCS_plot_type=CCS_plot_type, sub_folder_name=key_class))

        n_sample = CCS_stat["mean"][list(DD.keys())[0]].shape[0]
        for spl in range(n_sample):
            kw_plot.update(dict(spl=spl))
            plot_Graph_CCS(DD, beta_arr, key, **kw_plot)


@partial_427_decorator
def plot_side(tup, beta_arr):
    """this function is numerical calculation heavy | suitable for multi-processing

    Args
    ----
    - tup (tuple): (regType, p, lv), used to create the only key in <dd>

    Return
    ------
    dd (dict of dict): it's in lowercase because in contrast to DD, dd only has one key-val pair
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
    Sier = make_Sierpinski427(p, lv, x0=[0.0, 0.0], s0=1.0, c=1.0, regType=regType)
    Sier.Layout_Sierpinski427()
    dd[tup]["Sier"] = Sier

    return dd

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
        level lv: base level; smallest communities/clusters of (3) nodes
        ...
        level 2: 3 clusters of (3) level-3 units
        level 1: 1 cluster of (3) level-2 unit (coarsest level)
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
            temp2 = np.divide(std_weights[:-1], std_weights[1:])  # ditto, but more explicit
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
            temp2 = np.divide(std_weights[:-1], std_weights[1:])  # ditto, but more explicit
            for l in range(0, lv - 1):
                CCS_arr[i, 0, l] = temp1[l]
                CCS_arr[i, 1, l] = temp2[l]

    return CCS_arr


def plot_Graph_CCS(
    DD, beta_arr, key, CCS_stat=None, spl=-1, CCS_type="mean", err_type="ste",
    CCS_plot_type="CCS", colors=None, dpi=None, regCCS=False, sub_folder_name=""
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
    - spl (int): what index of sample to draw data from CCS_stat; -1 means the last sample
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    - CCS_plot_type (str):
        - "CCS": vanilla
        - "sum": sum of CCS across levels
    - colors (list of color hex strings):
        e.g., plt.rcParams['axes.prop_cycle'].by_key()['color'] is default color in pyplot:
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    - regCCS (bool):
        whether we will display 4 rows of CCS with no graphs
        assume:
            hierDict['reg_n'] = [[0,1,2,3],[3],[3,4,5]]
            or
            hierDict['reg_p'] = [[0,1,2,3],[3,4,5],[3]]
    - sub_folder_name (str):
        the name of the folder in f"output/{whatnot}/" to store the plots
        where <whatnot> is defined in saveNclose427()

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
    if CCS_stat is not None:
        # generate variable beta version if there is negative beta (just need to check last item)
        if CCS_stat["mean"][DD_keys[0]][spl, 0, -1] < 0:  # 2nd dim idx=0 is beta dim
            varb = True

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if regCCS:
        fig = plt.figure(figsize=[20, 18])  # initialize
        if varb:
            fig2 = plt.figure(figsize=[20, 18])  # initialize
    else:
        fig = plt.figure(figsize=[20, 9])  # initialize
        if varb:
            fig2 = plt.figure(figsize=[20, 9])  # initialize

    ds = 0.2  # dummy axes for spacing between the visible plots
    cbW = 1  # colorbar width
    hf = 9  # relative height of (sub)figure
    width_ratios = [ds, 19, cbW] * 3  # ≡ [ds,19,cbW,ds,19,cbW,ds,19,cbW]
    height_ratios = [1, hf, ds, 1, 1, hf, ds]
    if regCCS:
        height_ratios = [1, hf, ds, 1, 1, hf, ds, 1] + height_ratios
    kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
    kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
    gs = GridSpec(**kw1)

    axes = dict()
    if varb:
        axes2 = dict()
    if not regCCS:
        axes["Graph"], axes["CCS"], axes["Colorbar"] = [None] * 3, [None] * 3, [None] * 3
        if varb:
            axes2["Graph"], axes2["CCS"], axes2["Colorbar"] = [None] * 3, [None] * 3, [None] * 3
    else:
        for regType in [0, 1, 2, 3]:
            axes[f"CCS_reg{regType}"] = [None] * 3
            if varb:
                axes2[f"CCS_reg{regType}"] = [None] * 3

    fname = f"CCS_{key}"
    if CCS_stat is not None:
        # both group size and walk length are the same across all beta
        n_agents = round(CCS_stat["mean"][DD_keys[0]][spl, 1, 0])
        n_steps = round(CCS_stat["mean"][DD_keys[0]][spl, 2, 0])
        fname += f"_{n_agents}_{n_steps}_{err_type}_{CCS_type}"

    if regCCS:  # assuming DD_keys has 12 entries
        for i in range(3):
            for regType in [0, 1, 2, 3]:
                temp = gs[regType * 4 : regType * 4 + 3, i * 3 + 1]
                axes[f"CCS_reg{regType}"][i] = fig.add_subplot(temp)
                if varb:
                    axes2[f"CCS_reg{regType}"][i] = fig2.add_subplot(temp)
                params = DD_keys[i + 3 * regType]

                kw2 = dict(ax=axes[f"CCS_reg{regType}"][i], x=beta_arr)
                kw2.update(dict(CCS_plot_type=CCS_plot_type))
                kw2.update(dict(CCS_arr=DD[params]["CCS_arr"], params=params, key=key))
                kw2.update(dict(noise=CCS_stat, err_type=err_type, dpi=dpi, CCS_type=CCS_type))
                kw2.update(dict(spl=spl, is_log=True, colors=colors, regCCS=regType))
                ax_CCS(**kw2)
                if varb:
                    kw2.update(dict(ax=axes2[f"CCS_reg{regType}"][i], varb=varb))
                    ax_CCS(**kw2)
    else:
        for i in range(3):
            axes["Graph"][i] = fig.add_subplot(gs[0:3, i * 3 + 1])
            axes["CCS"][i] = fig.add_subplot(gs[4:7, i * 3 + 1])
            axes["Colorbar"][i] = fig.add_subplot(gs[0:3, i * 3 + 2])  # Edge Type colorbar
            params = DD_keys[i]
            kw3 = dict(ax=axes["Graph"][i], axcb=axes["Colorbar"][i], fig=fig, dpi=dpi)
            kw3.update(dict(params=params, colors=colors))
            kw3.update(dict(nodeList=DD[params]["Sier"].nodeList, GTDict=DD[params]["GTDict"]))
            ax_Graph(**kw3)
            kw4 = dict(ax=axes["CCS"][i], x=beta_arr, CCS_arr=DD[params]["CCS_arr"], params=params)
            kw4.update(dict(spl=spl, CCS_plot_type=CCS_plot_type))
            kw4.update(dict(key=key, noise=CCS_stat, err_type=err_type, CCS_type=CCS_type))
            kw4.update(dict(show_sim_param=i == 1, is_log=True, colors=colors, dpi=dpi, regCCS=3))
            ax_CCS(**kw4)
            if varb:
                axes2["Graph"][i] = fig2.add_subplot(gs[0:3, i * 3 + 1])
                axes2["CCS"][i] = fig2.add_subplot(gs[4:7, i * 3 + 1])
                axes2["Colorbar"][i] = fig2.add_subplot(gs[0:3, i * 3 + 2])  # Edge Type colorbar
                kw3.update(dict(ax=axes2["Graph"][i], axcb=axes2["Colorbar"][i], fig=fig2))
                ax_Graph(**kw3)
                kw4.update(dict(ax=axes2["CCS"][i], varb=varb))
                ax_CCS(**kw4)
    # panel label list
    text_labels = ["A", "B", "C", "D"] if regCCS else ["A", "B"]
    for i in range(len(text_labels)):
        axlabels = [fig.add_subplot(gs[i * 4 : i * 4 + 3, 0])]
        if varb:
            axlabels.append(fig2.add_subplot(gs[i * 4 : i * 4 + 3, 0]))
        for axlabel in axlabels:
            axlabel.set_frame_on(False)
            axlabel.set_axis_off()  # same as ax.axis('off')
            kw5 = dict(fontsize=17, horizontalalignment="center", transform=axlabel.transAxes)
            axlabel.text(-2.4, 1.05, f"{text_labels[i]}", **kw5)
    saveNclose427(fig, fname + "_const", dpi=dpi, sub_folder_name=sub_folder_name)
    if varb:
        saveNclose427(fig2, fname + "_var", dpi=dpi, sub_folder_name=sub_folder_name)


def ax_CCS(ax, x, CCS_arr, params, key, CCS_plot_type="CCS",
           noise=None, spl=-1, varb=False, err_type='ste', CCS_type='mean', show_sim_param=False,
           is_log=True, colors=None, dpi=None, regCCS=None):
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
    - noise (dict of 3D nparr): noise['mean'][params][s,i,beta]
        it is a synonym for CCS_stat (see doc in RW_CCS_stat.py)
    - spl (int): what index of sample to draw data from CCS_stat; -1 means the last sample
    - varb (bool): whether we draw negative beta (variable beta) or not
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    - show_sim_param (bool): whether we show simulation parameters
    - is_log (bool): if True then use log scale on x axis.
    - regCCS (int): reusing same var name,
        but in this function = (regCCS+1)th row.

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
        x_range = (min(x) * 1.00, max(x) * 1.25)
    else:
        delta = (max(x) - min(x)) * 0.05
        x_range = (min(x) - delta, max(x) + delta)
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
    ylabel = "Ratio of Means of Two Consecutive Levels"
    if CCS_plot_type == "sum":
        ylabel = "Sum of CCS Across Levels"
    if regType in [0, 1, 2, 3]:
        title = fr"Cross-Cluster Surprisal of $^{regType}S_{p:d}^{n:d}$"
    else:
        raise NotImplementedError(f"<regType>={regType} is not implemented")

    if noise is not None and show_sim_param:
        styles_txt = dict(fontsize=11, horizontalalignment="center", transform=ax.transAxes)
        n_agents = noise["mean"][params][spl, 1, 0]  # since all β have same group size, take 0
        n_steps = noise["mean"][params][spl, 2, 0]  # ditto but w/ walk length
        topy, s = 0.94, 0.06
        beta_type = "both" if varb else "constant"
        ax.text(0.5, topy, f"n_agents={n_agents:.0f}", **styles_txt)
        ax.text(0.5, topy - 1 * s, f"walk_length={n_steps:.0f}", **styles_txt)
        ax.text(0.5, topy - 2 * s, f"errorbar={err_type}", **styles_txt)
        ax.text(0.5, topy - 3 * s, f"CCS_type={CCS_type}", **styles_txt)
        ax.text(0.5, topy - 4 * s, f"beta_type={beta_type}", **styles_txt)

    if noise is not None:
        bs = None  # find where <0 beta starts (negative index)
        d = noise["mean"][params].shape[2]  # total number of beta
        for i in range(-1, -d - 1, -1):  # count from right because that's where <0 beta lies
            if noise["mean"][params][spl, 0, i] > 0:
                bs = i + 1
                break
        if bs != -1:
            # import warnings  # may want to comment the warnings out since it gets verbose
            # msg = "only <noise[\"mean\"][params][spl, 0, -1]> can be < 0\n"
            # msg += f"currently <noise[\"mean\"][params][spl, 0, {bs}:]> are all < 0\n"
            # msg += "using last val instead"
            # warnings.warn(msg)
            bs = -1  # force use last val
        # bs1 = slice(bs, None)  # if no <0 beta, slice(bs, None) = [0:], which is >0 beta
        bs2 = slice(0, d + bs)  # get >0 beta slice object

    styles1 = dict(alpha= 0.74, linewidth= 2)

    def temp_a1(i):  # draw CCS: analytical, noise constant, noise dynamic
        styles1.update({"label": f"lv{i+1}/lv{i+2}", "color": cmap(i)})
        ax.plot(x, CCS_arr[:, CCS_type_slice, i], **styles1)  # plot analytical curve
        if noise is not None:  # put scatter points of simulated results in
            if varb:  # plot horizontal line; only plot one dynamic beta; draw it first
                kw1 = dict(color=cmap(i), zorder=1)
                y_mean = noise["mean"][params][spl, 3 + i, bs]
                y_err = noise[err_type][params][spl, 3 + i, bs]
                ax.plot(ax.get_xlim(), [y_mean] * 2, **kw1)  # draw line
                kw1.update(dict(y1 = [y_mean - y_err] * 2, y2 = [y_mean + y_err] * 2))
                ax.fill_between(x=ax.get_xlim(), **kw1, alpha=0.27)  # draw colored region
            # plot fixed beta (scatter plot)
            styles2 = dict(linestyle="None", capsize=4.0, marker=".", markersize=11)
            styles2.update(dict(alpha = 0.74, linewidth = 2, zorder=2))
            styles2.update(dict(markeredgecolor=cmap(i), markerfacecolor=cmap(i), ecolor=cmap(i)))
            # styles2.update(dict(label='Stochastic '+labels[i].replace('-','/lv')))
            kw_erb = dict(x=noise["mean"][params][spl, 0, bs2])
            kw_erb.update(dict(y=noise["mean"][params][spl, 3 + i, bs2]))
            kw_erb.update(dict(yerr=noise[err_type][params][spl, 3 + i, bs2]))
            ax.errorbar(**kw_erb, **styles2)  # python 3.5+ PEP 448 (Unpacking Generalizations)
    
    def temp_a2():  # draw total CCS: analytical, noise constant, noise dynamic
        styles1.update(dict(label="Total CCS", color=cmap(0)))
        ts1 = slice(0, n_level)
        ysal = np.sum(CCS_arr[:, CCS_type_slice, ts1], axis=1)  # sum across levels
        ax.plot(x, ysal, **styles1)  # plot analytical curve
        if noise is not None:  # put scatter points of simulated results in
            ts2 = slice(3, 3 + n_level)
            if varb:  # plot horizontal line; only plot one dynamic beta; draw it first
                kw1 = dict(color=cmap(0), zorder=1)
                ysal_mean = np.sum(noise["mean"][params][spl, ts2, bs], axis=0)
                ysal_err = np.sqrt(np.sum(noise[err_type][params][spl, ts2, bs]**2, axis=0))
                ax.plot(ax.get_xlim(), [ysal_mean] * 2, **kw1)  # draw line
                kw1.update(dict(y1 = [ysal_mean - ysal_err] * 2, y2 = [ysal_mean + ysal_err] * 2))
                ax.fill_between(x=ax.get_xlim(), **kw1, alpha=0.27)  # draw colored region
            # plot fixed beta (scatter plot)
            styles2 = dict(linestyle="None", capsize=4.0, marker=".", markersize=11)
            styles2.update(dict(alpha = 0.74, linewidth = 2, zorder=2))
            styles2.update(dict(markeredgecolor=cmap(0), markerfacecolor=cmap(0), ecolor=cmap(0)))
            # styles2.update(dict(label='Stochastic '+labels[i].replace('-','/lv')))
            kw_erb = dict(x=noise["mean"][params][spl, 0, bs2])
            kw_erb.update(dict(
                y=np.sum(noise["mean"][params][spl, ts2, bs2], axis=0),
                yerr=np.sqrt(np.sum(noise[err_type][params][spl, ts2, bs2]**2, axis=0))
                ))
            ax.errorbar(**kw_erb, **styles2)  # python 3.5+ PEP 448 (Unpacking Generalizations)

    def temp_b1(i):  # annotate CCS plot with maxima
        xmaxs[i] = x[np.argmax(CCS_arr[:, CCS_type_slice, i])]
        ymaxs[i] = np.max(CCS_arr[:, CCS_type_slice, i])
        # texts[i] = f"({xmaxs[i]:.3f},{ymaxs[i]:.3f})"  # show both beta and CCS
        texts[i] = f"{xmaxs[i]:.3f}"  # show only beta
        kw_text.update(dict(color=cmap(i), xy=(xmaxs[i], ymaxs[i])))
        kw_text.update(dict(xytext=(0.85 - i * 0.2, 0.05)))
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
    kw_text = dict(textcoords="axes fraction", fontsize=11, arrowprops=arrowprops)
    kw_text.update(dict(ha="center", va="center"))

    if CCS_plot_type == "CCS":  # plot vanilla CCS graph
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


    ax.set_title(title, fontsize=17)
    if regCCS == 3:  # only have xlabel if bottom row
        ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if is_log:
        ax.set_xscale("log")  # set x to log scale
        if CCS_plot_type == "CCS":
            ax.legend(loc="upper left")
        elif CCS_plot_type == "sum":
            ax.legend(loc="lower left")
    else:
        ax.legend(loc="center right")
    ax.grid(False)
    return xmaxs


def ax_Graph(ax, axcb, fig, params, nodeList, GTDict, colors=None, dpi=None, annotate=None):
    """
    Args
    ----
    - ax/axcb: axis object
    - params (tuple): (regType, p, n)
    - nodeList: [(i,x,y),...] (x,y) is coordinate
    - GTDict (dict): contains 'A', 'edgeList', 'lvList' (all GroundTruth)
    - annotate (int):
        None: we don't label the nodes
        -1: Decimal
        p (0<p<10): base-p expansion
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
    annokw = dict(horizontalalignment="center", verticalalignment="center", color="b", fontsize=10)
    if annotate is None:
        for i, x, y in nodeList:
            ax.scatter(x, y, zorder=2, **marker_style)
    else:
        for i, x, y in nodeList:
            ax.scatter(x, y, zorder=2, **marker_style)
            ax.annotate(str(p_ary(i, p=p, L=n)), xy=(x, y), xytext=(x, y), **annokw)
    # draw edges & Edge Weight Coloring
    """ this is for transition prob edge drawing, which is not used anymore
    A = GTDict['A']
    n_ = np.shape(A)[0]
    RGBA = np.zeros((round(n_*(n_-1)/2),4))
    RGBA[:,1] = 0.5 # for green (not sure if this is color 'g')
    counter = 0
    for i in range(0,n_):
        for j in range(i+1,n_): # undirected (A is symmetric)
            # 🔴 assuming nodeList[i][0] = i
            x = [nodeList[i][1],nodeList[j][1]]
            y = [nodeList[i][2],nodeList[j][2]]
            RGBA[counter,3] = 1 if A[i,j]>0 else 0 # set alpha to 1 (opaque) if edge exists
            ax.plot(x,y,color=RGBA[counter,:],zorder=1) # lower int means drawn on the canvas earlier
            counter += 1
    """

    cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:nu], N=nu)
    cbr = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=nu - 1), cmap=cmap),
        cax=axcb,
        format="%.2f",
    )
    cbr.set_ticks([(nu - 1) * (2 * i + 1) / (2 * nu) for i in range(nu)])
    cbr.set_ticklabels(np.arange(1, nu + 1))
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
        )  # lower int means drawn on the canvas earlier
    # Grid setting and save
    axcb.set_frame_on(False)
    axcb.set_axis_off()  # same as ax.axis('off')
    ax.set_frame_on(False)
    ax.set_axis_off()  # same as ax.axis('off')
    ax.axis("equal")  # so that regular polygons appear to be regular as well
    if annotate is None:
        ax.set_title(title, fontsize=17)
    elif annotate == -1:
        ax.set_title(title[20:-12] + "Decimal Representation)", fontsize=17)
    else:  # since axET is always GroundTruthOnly, title will just be GroundTruth
        ax.set_title(title[20:-12] + "Base {:d} Representation)".format(p), fontsize=17)
    ax.grid(False)
    # plt.legend(loc='upper left')


if __name__ == "__main__":
    main_Sierpinski427()
