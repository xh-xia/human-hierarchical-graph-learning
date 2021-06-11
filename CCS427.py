"""
CCS visualizations.

Created: Monday, ‎April ‎5, ‎2021, ‏‎3:59:47 PM (EDT; maybe earlier actually)
@author: Xiaohuan (Pixel) X.
"""

from itertools import product  # for nested loops

from utility427.helper427 import set_dir427, mkdir_p
from utility427.math427 import findNearest, rank_eigvals, W_norm, np
from utility427.plt427 import plt, Normalize, LinearSegmentedColormap, GridSpec  # matplotlib
from utility427.plt427 import saveNclose427, colors_selector, cbrLabel427  # matplotlib helpers
from utility427.plt427 import load_CCS_stat, save_Masks
from utility427.Sierpinski427 import make_Sierpinski427, p_ary, make_SierpinskiGraph427
from stims427 import Hamiltonian_cycle

# define some global constants (lowercase/mixedcase tho)
err_type = "ste"
CCS_type = "mean"  # 'mean' or 'std'
key_class, n_agents, dpi = "reg_n_p", 100, 300
sub_folder_name = "step_funct"  # folder in "input" folder containing CCS_stat .npy files
# recreate beta_classes from "var_betas" in sim427/input/params.json
var_betas = [[0.001, 10], [0.002, 0.37], [0.37, 0.002]]
beta_classes = [None] * len(var_betas)
for i, var_beta in enumerate(var_betas):
    beta_classes[i] = "step_{}to{}".format(*var_beta)


def main_Sierpinski427():
    set_dir427()  # make sure cwd is the one this script is in
    colors = colors_selector(str="5-class Greens")
    beta_arr = np.geomspace(0.0001, 10, 400)  # for analytical curve only
    hierDict = dict()
    # hierDict['n'] = [[0],[3],[3,4,5]]
    # hierDict['p'] = [[3],[3,4,5],[3]]
    hierDict["reg_n"] = [[0, 1, 2, 3], [3], [3, 4, 5]]
    hierDict['reg_p'] = [[0,1,2,3],[3,4,5],[3]]
    hierDict['r'] = [[0,1,3],[3],[3]]
    for beta_class in beta_classes:
        npy_sub_path = f"{sub_folder_name}\\"
        npy_sub_path += f"CCS_stat_{CCS_type}_{key_class}_{beta_class}_{n_agents}"
        CCS_stat = load_CCS_stat(fname=npy_sub_path)  # load sim results
        for key in hierDict.keys():
            DD = dict()  # = DataDict = {(regType,p,lv):{'GTDict'=GTDict,etc.}}
            hierLists = hierDict[key]
            for regType, p, lv in product(*hierLists):
                tup = (regType, p, lv)
                DD[tup] = dict()
                DD[tup]["GTDict"] = make_SierpinskiGraph427(p, lv, norm=True, regType=regType)
                save_Masks(DD[tup]["GTDict"], regType, p, lv)
                DD[tup]["A_hat_list"] = [
                    make_A_hat_beta(DD[tup]["GTDict"]["A"], beta) for beta in beta_arr
                ]
                DD[tup]["CCS_arr"] = CCS_analysis(
                    DD[tup]["GTDict"], beta_arr, DD[tup]["A_hat_list"], analytic=False
                )
                Sier = make_Sierpinski427(p, lv, x0=[0.0, 0.0], s0=1.0, c=1.0, regType=regType)
                Sier.Layout_Sierpinski427()
                DD[tup]["Sier"] = Sier

            kw_plot = dict(CCS_stat=CCS_stat, err_type=err_type, CCS_type=CCS_type)
            kw_plot.update(dict(colors=colors, dpi=dpi, regCCS=len(key) > 1))
            kw_plot.update(dict(sub_folder_name=beta_class))
            plot_Graph_CCS(DD, beta_arr, key, **kw_plot)


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
    DD, beta_arr, key, CCS_stat=None, CCS_type="mean", err_type="ste",
    colors=None, dpi=None, regCCS=False, sub_folder_name=""
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
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
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
        if CCS_stat["mean"][DD_keys[0]][-1, 0, -1] < 0:  # 2nd dim idx=0 is beta dim
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
    width_ratios = [ds, 19, cbW] * 3  # ≡ [ds,19,cbW,ds,19,cbW,ds,19,cbW]
    height_ratios = [1, 9, ds, 1, 1, 9, ds]
    if regCCS:
        height_ratios = [1, 9, ds, 1, 1, 9, ds, 1] + height_ratios
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
        n_agents = round(CCS_stat["mean"][DD_keys[0]][-1, 1, 0])
        n_steps = round(CCS_stat["mean"][DD_keys[0]][-1, 2, 0])
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
                kw2.update(dict(CCS_arr=DD[params]["CCS_arr"], params=params, key=key))
                kw2.update(dict(noise=CCS_stat, err_type=err_type, dpi=dpi, CCS_type=CCS_type))
                kw2.update(dict(is_log=True, colors=colors, regCCS=regType))
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


def ax_CCS(ax, x, CCS_arr, params, key,
           noise=None, varb=False, err_type='ste', CCS_type='mean', show_sim_param=False,
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
    - noise (dict of 3D nparr): noise['mean'][params][s,i,beta]
        it is a synonym for CCS_stat (see doc in RW_CCS_stat.py)
    - varb (bool): whether we consider only negative beta (variable beta) or not
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    - show_sim_param (bool): whether we show simulation parameters
    - is_log (bool): if True then use log scale on x axis.
    - regCCS (int): reusing same var name,
        but in this function = (regCCS+1)th row.

    Intermediary
    ------------
    variable beta case:
        -whatnot[-9:-1] contains num of changes = [1, 2, 3, 4, 5, 6, 8, 10, 12]
        <whatnot> := noise[err_type][params][-1, 0, :],
        which is a list of actual beta (including negative codename beta)
    """
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
    if regType in [0, 1, 2, 3]:
        title = fr"Cross-Cluster Surprisal of $^{regType}S_{p:d}^{n:d}$"
    else:
        raise NotImplementedError(f"<regType>={regType} is not implemented")

    if noise is not None and show_sim_param:
        styles_txt = dict(fontsize=11, horizontalalignment="center", transform=ax.transAxes)
        n_agents = noise["mean"][params][-1, 1, 0]  # since all β have same group size, take 0
        n_steps = noise["mean"][params][-1, 2, 0]  # ditto but w/ walk length
        topy, s = 0.94, 0.06
        beta_type = "dynamic" if varb else "constant"
        ax.text(0.5, topy, f"n_agents={n_agents:.0f}", **styles_txt)
        ax.text(0.5, topy - 1 * s, f"walk_length={n_steps:.0f}", **styles_txt)
        ax.text(0.5, topy - 2 * s, f"errorbar={err_type}", **styles_txt)
        ax.text(0.5, topy - 3 * s, f"CCS_type={CCS_type}", **styles_txt)
        ax.text(0.5, topy - 4 * s, f"beta_type={beta_type}", **styles_txt)
    styles1 = {"alpha": 0.74, "linewidth": 2}
    # find where <0 beta starts and create beta slice object (i.e., bs)
    if noise is not None:
        bs = None
        d = noise["mean"][params].shape[2]  # total number of beta
        for i in range(-1, -d - 1, -1):  # count from right because that where <0 beta lies
            if noise["mean"][params][-1, 0, i] > 0:
                bs = i + 1
                break
        if varb:  # if there is no <0 beta, slice(bs, None) is then [0:], which is >0 beta
            bs = slice(bs, None)
            # this is separate operation: make twin axis for varb since it is on different scale
            ax_twin = ax.twiny()  # instantiate a twin x axis sharing same y-axis
            ax_twin.set_xticks(-noise["mean"][params][-1, 0, bs])  # tick where data point is at
        else: # get >0 beta
            bs = slice(0, d + bs)
    for i in range(n_level):  # ↓ first plot analytical curve
        # print('DEBUG: CCS_arr_lv1 - mean {}'.format(CCS_arr[:,0,i]))
        # print('DEBUG: CCS_arr_lv1 - std {}'.format(CCS_arr[:,1,i]))
        styles1.update({"label": f"lv{i+1}/lv{i+2}", "color": cmap(i)})
        ax.plot(x, CCS_arr[:, CCS_type_slice, i], **styles1)
        if noise is not None:
            styles2 = dict(yerr=noise[err_type][params][-1, 3 + i, bs])
            styles2.update(dict(linestyle="None", capsize=4.0, marker=".", markersize=11))
            styles2.update(dict(markeredgecolor=cmap(i), markerfacecolor=cmap(i), ecolor=cmap(i)))
            # styles2.update(dict(label='Stochastic '+labels[i].replace('-','/lv')))
            styles2.update({"alpha": 0.74, "linewidth": 2})
            if varb:
                kw_erb = dict(x=-noise["mean"][params][-1, 0, bs])
            else:
                kw_erb = dict(x=noise["mean"][params][-1, 0, bs])
            kw_erb.update(dict(y=noise["mean"][params][-1, 3 + i, bs]))
            if varb:
                ax_twin.errorbar(**kw_erb, **styles2)
            else:
                ax.errorbar(**kw_erb, **styles2)  # python 3.5+ PEP 448 (Unpacking Generalizations)

    # argmax = beta that maximizes bottom 3 level diffs (2 diffs)
    xmax3 = x[np.argmax(CCS_arr[:, CCS_type_slice, 0])]
    ymax3 = np.max(CCS_arr[:, CCS_type_slice, 0])
    xmax2 = x[np.argmax(CCS_arr[:, CCS_type_slice, 1])]
    ymax2 = np.max(CCS_arr[:, CCS_type_slice, 1])
    text3 = f"({xmax3:.3f},{ymax3:.3f})"
    text2 = f"({xmax2:.3f},{ymax2:.3f})"
    arrowprops = dict(arrowstyle="simple", facecolor="grey", edgecolor="grey")
    arrowprops.update(dict(linewidth=1 / 3, alpha=0.74))
    kw = dict(textcoords="axes fraction", fontsize=11, arrowprops=arrowprops)
    kw.update(dict(ha="center", va="center"))
    ax.annotate(text3, color=colors[0], xy=(xmax3, ymax3), xytext=(0.85, 0.95), **kw)
    ax.annotate(text2, color=colors[1], xy=(xmax2, ymax2), xytext=(0.15, 0.65), **kw)
    if regCCS == 3:  # only have xlabel if bottom row
        ax.set_xlabel(xlabel, fontsize=11)
        if varb:
            ax_twin.xaxis.set_ticks_position("top")  # move 2nd axis to the top
            ax_twin.xaxis.set_label_position("top")  # move 2nd axis to the top
            ax_twin.set_xlabel(xlabel + " (dynamic)")
    ax.set_ylabel(ylabel, fontsize=11)
    if key in ["n", "reg_n"]:
        ax.set_ylim((0.9, 1.3))  # for regType=3, p=3, n=3 max CCS is <1.3
    else:
        ax.set_ylim((0.9, 1.63))  # for regType=3, p=5, n=3 max CCS is about 1.63
    ax.plot(ax.get_xlim(), (1, 1), "--", color="grey", zorder=0)  # draw y=1 line in grey
    ax.set_title(title, fontsize=17)
    if is_log:
        ax.set_xscale("log")  # set x to log scale
        ax.legend(loc="upper left")
    else:
        ax.legend(loc="center right")
    ax.grid(False)
    return xmax2, xmax3


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
