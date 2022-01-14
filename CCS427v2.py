"""
CCS, CCSC (coarse-graining); it also removes some commented codes in CCS427.py

Created: Monday, ‎November ‎1, ‎2021, ‏‎9:52:16 AM (EDT)
@author: Xiaohuan (Pixel) X.
"""

from itertools import product  # for nested loops
from concurrent.futures import ProcessPoolExecutor  # for multi-processing
from concurrent.futures import ThreadPoolExecutor  # for multi-threading

from utility427.helper427 import set_dir427, mkdir_p, get_params, partial_427_decorator
from utility427.math427 import log_b, findNearest, rank_eigvals, W_norm, np
from utility427.plt427 import plt, Normalize, LinearSegmentedColormap, GridSpec, Line2D  # mpl
from utility427.plt427 import saveNclose427, colors_selector, cbrLabel427, get_violin_pw  # mpl helpers
from utility427.plt427 import load_CCS_stat, save_masks
from utility427.Sierpinski427 import make_Sierpinski427, p_ary, make_SierpinskiGraph427, W_CG
from utility427.sim_params427 import make_sim_params
from utility427.CCS_num import CCS_ep  # newly added on 2022.1.13
from sim427.RW_Graph_Class import CCS, CCPS, make_masks

def CCS_main():
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
    make_sim_params(p)
    p.update(get_params(fname="input\\params_CCS427", default_dir=False))  # get parameters for CCS427.py

    # add more parameters into p
    p["colors"] = colors_selector(str="5-class Greens")
    p["beta_arr"] = np.geomspace(0.0001, 10, 400)  # for analytical curve only
    hierDict = dict()
    hierDict['b'] = [[3], [3, 4, 5], [3, 4]]
    kw_loop = dict(sub_fo_name=p["sub_fo_name"], CCS_type=p["CCS_type"], key_class=p["key_class"])
    kw_loop.update(dict(n_agents=p["n_agents"], err_type=p["err_type"], hierDict=hierDict))
    kw_loop.update(dict(beta_arr=p["beta_arr"], colors=p["colors"], dpi=p["dpi"]))
    kw_loop.update(dict(raw_method=p["raw_method"], CCS_plot_type=p["CCS_plot_type"]))
    kw_loop.update(dict(use_sim=p["use_sim"]))

    for beta_class in p["beta_classes"]:
        plot_main(**kw_loop)(beta_class)


@partial_427_decorator
def plot_main(beta_class, sub_fo_name, CCS_type, key_class, n_agents, hierDict, use_sim,
              beta_arr, err_type, raw_method, CCS_plot_type, colors, dpi):
    """this function is I/O bound | suitable for multi-threading
    should be in script dir when this function is run

    Intermediary
    ------------
    DD (dict): a data dict created only to be used in graphing (i.e., plot_Graph_CCS())
        most results from numerical calculations are put in DD
    """

    if use_sim:
        CCS_stats = [None] * 2
        for i, CCS_variant in enumerate(["CCS", "CCSC"]):
            npy_sub_path = f"{sub_fo_name}\\{CCS_variant}_stat_{CCS_type}_{key_class}_{beta_class}_{n_agents}"
            CCS_stats[i] = load_CCS_stat(sim_path="sim427\\output", fname=npy_sub_path)  # get sim results (<noise>)
    else:
        CCS_stats = None

    for key in hierDict.keys():
        kw_main = dict(beta_arr=beta_arr)
        dd_list = map(plot_side(**kw_main), product(*hierDict[key]))
        DD = dict()
        for dd in dd_list:  # aggregate all keys of dd in dd_list into DD
            for k in dd:  # dd & DD have same structure, only DD is larger (since it's an aggregate)
                if k in DD:  # DD[k] is a dict whose keys will be expanded by dd[k]'s keys
                    DD[k].update(dd[k])
                else:
                    DD[k] = dd[k]

        kw_plot = dict(CCS_stats=CCS_stats, err_type=err_type, CCS_type=CCS_type)
        kw_plot.update(dict(colors=colors, dpi=dpi))
        kw_plot.update(dict(CCS_plot_type=CCS_plot_type, sub_folder_name=sub_fo_name))
        kw_plot.update(dict(raw_method=raw_method))

        plot_Graph_CCS(DD, beta_arr, key, **kw_plot)


@partial_427_decorator
def plot_side(tup, beta_arr):
    """this function is numerical calculation heavy | suitable for multi-processing

    Args
    ----
    - tup (tuple): (regType, p, lv), used to create the only key in <dd>

    Return
    ------
    - dd (dict of dict): it's in lowercase because in contrast to DD, dd only has one key-val pair
        this will be later merged with other dd from the same function to create DD

    NOTE
    "GTDict" & "Sier": are for ground truth, and we don't CG ground truth, so always vanilla
    "A_hat_list" & "CCS_arrs": are for beta version, and we CG beta version; so both vanilla & CG
    x in suffix="_CGx" is meaningful: because CGed version can come from different tup
        e.g., (3,3,4) with CG_lv=1 results in (3,3,3) | _CG1
      whereas (3,3,5) with CG_lv=2 results in (3,3,3) | _CG2
    """
    dd = dict()  # = DataDict = {(regType,p,lv):{'GTDict'=GTDict,etc.}}
    coarse_grain = [0, 1]  # 0 is vanilla; 1 is level 1 coarse-graining (lv1CG AKA CG1)
    regType, p, lv = tup  # unpack original tup
    for CG_lv in coarse_grain:
        tup_CG = (regType, p, lv-CG_lv)  # get coarse-grained tup; coarse-grain -> level-=CG_lv
        dd[tup_CG] = dict()
        if CG_lv == 0:
            suffix = ""
        else:
            suffix = f"_CG{CG_lv}"
        # vanilla only (NOTE can be vanilla for either original tup or CGed tup)
        dd[tup_CG]["GTDict"] = make_SierpinskiGraph427(tup_CG[1], tup_CG[2], norm=True, regType=regType)
        save_masks(dd[tup_CG]["GTDict"], *tup_CG)
        Sier = make_Sierpinski427(tup_CG[1], tup_CG[2], x0=[0.0, 0.0], s0=1.0, c=1.0, regType=regType)
        Sier.Layout_Sierpinski427()
        dd[tup_CG]["Sier"] = Sier
        # can be CG or vanilla
        kwargs = dict(A=dd[tup]["GTDict"]["A"], CG_lv=CG_lv, tup=tup)  # use original tup to CG
        dd[tup_CG][f"A_hat_list{suffix}"] = [make_A_hat_beta(beta=beta, **kwargs) for beta in beta_arr]
        dd[tup_CG][f"CCS_arrs{suffix}"] = CCS_analysis_v2(tup_CG, dd[tup_CG][f"A_hat_list{suffix}"])

    return dd


def make_A_hat_beta(A, beta, CG_lv=0, tup=None):
    """generate A_hat according to Max Entropy Model

    Args
    ----
    - A (2D nparr; symmetric): adjacency/weight matrix
    - beta (any number): complexity-accuracy trade-off param
    - CG_lv (>=0 int): coarse-graining level
    - tup (tup): corresponding to that of ground truth A;
                 cannot be None if CV_lv>0 since coarse-graining depends on p

    Return
    ------
    - A_hat (2D nparr):
        assuming infinite walks on A, this is the resulting A_hat learned based on beta
        assume the graph is connected, meaning beta -> 0 A_hat = 1/n * np.ones((n, n))
        A_hat = (1-e^(-β)) * A * (I - (e^(-β))A)^(-1)
        undirected, weighted 3-regular graph with:
        lv hierarchies:
        level 1: base level; smallest communities of (3) nodes
        ...
        level lv-1: 3 communities of (3) level-lv-2 units
        level lv: 1 community of (3) level-lv-1 unit (coarsest level)
    """
    n = np.shape(A)[0]  # # of rows, but assuming symmetric, thus also cols (=nodes)
    if np.isclose(0, beta, rtol=0, atol=1e-32):  # abs(a - b) <= (atol + rtol * abs(b))
        return 1/n * np.ones((n, n), dtype=float)
    A_ = W_norm(A)
    A_hat = (1 - np.exp(-beta)) * A_ @ np.linalg.inv(np.eye(n) - np.exp(-beta) * A_)
    for i in range(CG_lv):
        A_hat = W_CG(A_hat, *tup[:2], tup[2] - i)
    return A_hat



def CCS_analysis_v2(tup, A_hat_list=None, CCS_only=True):
    """
    it finds CCS, CCPS, CCTS given A_hat_list (len(A_hat_list)=num of betas)

    Arg
    ---
    - tup (tuple): regType, p, n
    - A_hat_list (np.arr):
        np.arr: analytic computation to find A_hat (transition probability matrix)

    Return
    ------
    - CCS_arrs (dict of 3D nparr): each key (str) is a variant of CCS
        for CCS_arr in CCS_arrs:
            CCS_arr[s,0,l-1]: CCS of means at beta s for level l (f"{'lv'}{l}{'-'}{l+1}")
            CCS_arr[s,1,l-1]: CCS of stds at beta s for level l (f"{'lv'}{l}{'-'}{l+1}")
    """
    regType, p, n = tup
    num_beta = len(A_hat_list)  # number of beta (which is also number of graphs)
    if regType in [0, 3]:
        lv = n  # (max) hierarchical level (also the coarsest level) for node community
    elif regType in [1]:
        lv = n + 1
    else:
        raise NotImplementedError(f"regType={regType} is not implemented yet")
    CCS_arrs = dict()
    CCS_arrs["CCS"] = np.zeros((num_beta, 2, lv - 1))
    CCS_arrs["CCPS"] = np.zeros((num_beta, 2, lv - 1))
    CCS_arrs["CCTS"] = np.zeros((num_beta, 2, lv - 1))

    if CCS_only:
        for i in range(num_beta):
            temp0 = CCS(A_hat_list[i], regType, p, n, seed=0, analytic_comp=True)
            CCS_arrs["CCS"][i, ...] = temp0[0, ...]
    else:
        for i in range(num_beta):
            temp0 = CCS(A_hat_list[i], regType, p, n, seed=0, analytic_comp=True)
            temp1 = CCPS(A_hat_list[i], regType, p, n, ccps_type=1, analytic_comp=True, scale=1)
            temp2 = CCPS(A_hat_list[i], regType, p, n, ccps_type=2, analytic_comp=True, scale=1)
            CCS_arrs["CCS"][i, ...] = temp0[0, ...]
            CCS_arrs["CCPS"][i, ...] = temp1[0, ...]
            CCS_arrs["CCTS"][i, ...] = temp2[0, ...]

    return CCS_arrs


def plot_Graph_CCS(
    DD, beta_arr, key, CCS_stats=None, raw_method=None, CCS_type="mean", err_type="ste",
    CCS_plot_type="CCS", colors=None, dpi=None, sub_folder_name=""
):
    """It produces both CCS plot and Graph (node-edge graph, not graph graph) plot

    display 3 rows: 1 graph 1 CCPS 1 CCTS

    Args
    ----
    - DD (dict):
        DD.keys (tuple): (regType,p,n)
    - key (str): those in hierDict.keys()

    Kwargs
    ------
    - CCS_stats (list dict): CCS, CCSC
        each dict:
            CCS_stat['mean'], CCS_stat['std'], and CCS_stat['ste'] have:
            same keys as DD; value is 3D nparr (see RW_CCS_stat.py for description)
    - raw_method (str): if prefix="violin", assume CCS_stat has "raw" key
        - violin: vanilla violin plot
        - violin_median: show median in red, and only 2nd lv CCS as well (1st lv is as expected)
    - spl (int or list): what indices of sample to draw data from CCS_stat; -1 is the last sample
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    - CCS_plot_type (str):
        - "CCS": vanilla
        - "sum": sum of CCS across levels
    - colors (list of color hex strings):
        e.g., plt.rcParams['axes.prop_cycle'].by_key()['color'] is default color in pyplot:
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    - sub_folder_name (str):
        the name of the folder in f"output/{whatnot}/" to store the plots
        where <whatnot> is defined in saveNclose427()

    Save
    ----
    CCS plot:
    - fixed beta
    - variable beta (controlled by varb ≡ var_beta): not implemented
    """
    if key == "b":  # hierDict['b'] = [[3], [3, 4, 5], [3, 4]]
        # given how plot_side() is written, DD will have (3, x, 2) for x in [3, 4, 5]
        # we will remove (3, x, 2)
        DD_keys = sorted(DD.keys(), reverse=False, key=lambda x: (x[0], x[2], x[1]))
        power_min = min({x[2] for x in DD_keys})  # remove one with the least power
        DD_keys = [x for x in DD_keys if x[2] != power_min]
    else:
        DD_keys = sorted(DD.keys(), reverse=False)  # ascending (default)
    if CCS_stats is None:  # meaning purely analytical, no sim results
        CCS_stats = [None, None]
        has_sim = False
    else:
        has_sim = True
    spl = [0, -1]

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=[20, 13.5])  # initialize

    ds = 0.2  # dummy axes for spacing between the visible plots
    cbW = 1  # colorbar width
    hf = 9  # relative height of (sub)figure
    width_ratios = [ds, 19, cbW] * 3  # ≡ [ds,19,cbW,ds,19,cbW,ds,19,cbW]
    height_ratios = [1, hf, ds, 1] * 2 + [1, hf, ds]  # 3 rows
    kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
    kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
    gs = GridSpec(**kw1)

    axes = dict()
    axes["Graph"], axes["CCS"], axes["Colorbar"] = [None] * 3, [None] * 6, [None] * 3

    fname = f"CCSC_{key}"
    if len(CCS_stats) != 2:
        raise ValueError("CCS_stats has to be a list of CCS_stat, CCSC_stat")
    CCS_stat, CCSC_stat = CCS_stats
    if has_sim:
        # both group size and walk length are the same across all beta
        n_agents = round(CCS_stat["mean"][DD_keys[0]][spl[-1], 1, 0])  # same for any spl
        n_steps1 = round(CCS_stat["mean"][DD_keys[0]][spl[0], 2, 0])
        n_steps2 = round(CCS_stat["mean"][DD_keys[0]][spl[1], 2, 0])
        n_steps = (n_steps1, n_steps1)
        fname += f"_{n_agents}_{n_steps}_{err_type}_{CCS_type}"

    # draw 1 transition graph + 2 CCS rows: CCS, CCSC (coarse-grained), analytical (no sim involved)
    # which means no noise (arg for ax_CCS()) involved
    # - suffix (str): coarse-graining suffix, if None, then empty (i.e., "")
    #     only DD[params]["A_hat_list"] and DD[params]["CCS_arrs"] have suffix versions
    spl_current = spl[0]  # 0 is 1500, 1 is 7500
    betas = [0.01, 0.33, 1]
    # for 1st row: (3, 3, 2), (3, 3, 3) - 1, (3, 3, 4) -2 | only one beta = betas[1]
    A_GTs = [make_SierpinskiGraph427(3, x, norm=True, regType=3, use_set=False)["A"] for x in [2,3,4]]
    for i in range(3):
        axes["Graph"][i] = fig.add_subplot(gs[0:3, i * 3 + 1])
        axes["CCS"][i] = fig.add_subplot(gs[4:7, i * 3 + 1])
        axes["Colorbar"][i] = fig.add_subplot(gs[0:3, i * 3 + 2])  # Edge Type colorbar
        # transition graph (all of the same beta, but different param)
        bA = [betas[1], make_A_hat_beta(A_GTs[i], betas[1], CG_lv=i, tup=(3, 3, 2 + i))]
        params = (3, 3, 2)  # regType=3, p=3, n=2; ground truth param (i.e., non-CG version)
        kw3 = dict(ax=axes["Graph"][i], axcb=axes["Colorbar"][i], fig=fig, dpi=dpi)
        kw3.update(dict(params=params, colors=colors, bA=bA, col=i, CG_lv=i))
        kw3.update(dict(nodeList=DD[params]["Sier"].nodeList, GTDict=DD[params]["GTDict"]))
        ax_Graph(**kw3)
        params = DD_keys[i]
        # 1st CCS row: CCS
        kw4 = dict(ax=axes["CCS"][i], x=beta_arr, CCS_arrs=DD[params]["CCS_arrs"], params=params)
        kw4.update(dict(spl=spl_current, CCS_plot_type=CCS_plot_type))
        kw4.update(dict(key=key, raw_method=raw_method, err_type=err_type, CCS_type=CCS_type))
        kw4.update(dict(show_legend=i == 1, is_log=True, colors=colors, dpi=dpi, last_row=False))
        kw4.update(dict(noise=CCS_stat, CCS_variant="CCS", CG_lv=0))
        ax_CCS(**kw4)
    # 2nd CCS row: CCSC (CG_lv=1)
    CG_lv = 1
    suffix = f"_CG{CG_lv}"
    for i in range(3):
        axes["CCS"][i+3] = fig.add_subplot(gs[8:11, i * 3 + 1])
        params = DD_keys[i]
        kw4.update(dict(ax=axes["CCS"][i+3]))
        kw4.update(dict(CCS_arrs=DD[params][f"CCS_arrs{suffix}"], params=params))
        kw4.update(dict(spl=spl_current, show_legend=i == 1, last_row=True))
        kw4.update(dict(noise=CCSC_stat, CCS_variant="CCS", CG_lv=CG_lv))
        ax_CCS(**kw4)
    text_labels = ["A", "B", "C"]  # panel label list
    for i in range(len(text_labels)):
        axlabels = [fig.add_subplot(gs[i * 4 : i * 4 + 3, 0])]
        for axlabel in axlabels:
            axlabel.set_frame_on(False)
            axlabel.set_axis_off()  # same as ax.axis('off')
            kw5 = dict(fontsize=17, horizontalalignment="center", transform=axlabel.transAxes)
            axlabel.text(-2.4, 1.05, f"{text_labels[i]}", **kw5)
    saveNclose427(fig, fname + "_const", dpi=dpi, sub_folder_name=sub_folder_name)


def ax_CCS(ax, x, CCS_arrs, params, key, CCS_plot_type="CCS", raw_method=None,
           noise=None, spl=-1, err_type='ste', CCS_type='mean', CCS_variant="CCS", CG_lv=0,
           show_legend=False, is_log=True, colors=None, dpi=None, last_row=False):
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
    - raw_method (str): if prefix="violin", assume CCS_stat has "raw" key
        - violin: vanilla violin plot; show mean and SD
        - violin_median: show median and ste_median instead
    - noise (dict of 3D nparr): noise['mean'][params][s,i,beta]
        it is a synonym for CCS_stat (see doc in RW_CCS_stat.py)
    - spl (int or list): what indices of sample to draw data from CCS_stat; -1 is the last sample
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    - CCS_variant (str): CCS, CCPS, CCTS; this refers to type of noise in <noise> & key in CCS_arrs
    - show_legend (bool): whether we show simulation parameters
    - is_log (bool): if True then use log scale on x axis.
    - last_row (bool): whether it is at last row

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
        x_scale = 10  # base used for get_violin_pw(); same as when beta was first initialized
    else:
        delta = (max(x) - min(x)) * 0.05
        x_range = (min(x) - delta, max(x) + delta)
        x_scale = None  # linear
    ax.set_xlim(x_range)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    (regType, p, n) = params
    CCS_arr = CCS_arrs[CCS_variant]  # this could be CCS_arr, CCPS_arr, or CCTS_arr
    n_level = CCS_arr.shape[2]
    if n_level == n - 1:  # normally there are only n-1* levels (since CCS is function of 2 lvs)
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
    ylabel = f"{CCS_variant}"
    if CCS_plot_type == "sum":
        ylabel = "Sum of CCS Across Levels"
    if regType in [0, 1, 2, 3]:
        if CG_lv == 0:
            title = fr"{CCS_variant} of $^{regType}S_{p:d}^{n:d}$"
        elif CG_lv > 0:
            title = fr"{CCS_variant} of $_{CG_lv:d}^{regType}S_{p:d}^{n:d}$"
        else:
            raise NotImplementedError(f"CG_lv={CG_lv} is not implemented")
    else:
        raise NotImplementedError(f"<regType>={regType} is invalid")

    if noise is not None and show_legend:  # only show walk length
        styles_txt = dict(fontsize=11, horizontalalignment="center", transform=ax.transAxes)
        # n_agents = noise["mean"][params][spl, 1, 0]  # since all β have same group size, take 0
        n_steps = noise["mean"][params][spl, 2, 0]  # ditto but w/ walk length
        topy, s = 0.94, 0.06
        ax.text(0.80, topy - 1 * s * 0, f"walk length={n_steps:.0f}", **styles_txt)

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

    sty1 = dict(alpha= 0.74, linewidth= 2)

    def temp_a1(i):  # draw CCS: analytical, noise constant, noise dynamic
        sty1.update(dict(color=cmap(i)))
        if show_legend:
            if CCS_variant == "CCPS":
                sty1.update(dict(label=f"lv{i+1}-lv{i+2}"))
            else:
                sty1.update(dict(label=f"lv{i+1}/lv{i+2}"))
        ax.plot(x, CCS_arr[:, CCS_type_slice, i], **sty1)  # plot analytical curve
        if noise is not None:  # put scatter points of simulated results in
            # plot fixed beta (scatter plot)
            sty2 = dict(linestyle="None", capsize=4.0, marker=".", markersize=11)
            sty2.update(dict(alpha = 0.74, linewidth = 2, zorder=2))
            sty2.update(dict(markeredgecolor=cmap(i), markerfacecolor=cmap(i), ecolor=cmap(i)))
            # if show_legend:
            #     sty2.update(dict(label='Stochastic '+labels[i].replace('-','/lv')))
            kw_erb = dict(x=noise["mean"][params][spl, 0, bs2])
            kw_erb.update(dict(y=noise["mean"][params][spl, 3 + i, bs2]))
            kw_erb.update(dict(yerr=noise[err_type][params][spl, 3 + i, bs2]))
            if raw_method is None:
                return 0
            elif raw_method.startswith("violin"):  # violin plot (overlaid)
                # TODO haven't implemented it in temp_a2()
                data_vl = [noise["raw"][params][spl, i, b, :] for b in range(bs2.start, bs2.stop)]
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
                if raw_method[-6:] == "median":  # plot median and ste_median instead
                    kw_erb.update(dict(y=noise["median"][params][spl, 3 + i, bs2]))
                    kw_erb.update(dict(yerr=noise["ste_median"][params][spl, 3 + i, bs2]))

            else:
                msg2 = "<raw_method>: currently only supports scatter(None)/violin('violin') plot"
                raise ValueError(msg2)

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
        if CCS_variant != "CCS":  # annotation only for CCS
            return 0  # no annotation for CCPS, CCTS
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

        if CCS_variant == "CCS":
            if key in ["n", "reg_n"]:
                ax.set_ylim((0.9, 1.3))  # for regType=3, p=3, n=3 max CCS is <1.3
            else:
                ax.set_ylim((0.9, 1.63))  # for regType=3, p=5, n=3 max CCS is about 1.63
            ax.plot(ax.get_xlim(), (1, 1), "--", color="grey", zorder=0)  # draw y=1 line in grey
        elif CCS_variant == "CCPS":  # only key in ["r"]: [0,1,3] [3] [3]
            ax.set_ylim((-0.07, 0.05))  # for regType=0,1,3, p=3, n=3 CCPS range is -0.07 ~ 0.05
            ax.plot(ax.get_xlim(), (0, 0), "--", color="grey", zorder=0)  # draw y=0 line in grey
        elif CCS_variant == "CCTS":  # only key in ["r"]: [0,1,3] [3] [3]
            ax.set_ylim((0, 11))  # for regType=0,1,3, p=3, n=3 CCTS range is 0 ~ 12
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
    if last_row:  # only have xlabel if bottom row
        ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if is_log:
        ax.set_xscale("log")  # set x to log scale
        if CCS_plot_type == "CCS":
            loc="upper left"
        elif CCS_plot_type == "sum":
            loc="lower left"
    else:
        loc="center right"
    temp_li, temp_la = ax.get_legend_handles_labels()
    if show_legend:
        ax.legend(temp_li, temp_la, loc=loc)
    ax.grid(False)
    return xmaxs


def ax_Graph(ax, axcb, fig, params, nodeList, GTDict,
             axt=None, colors=None, dpi=None, annotate=None, bA=None, col=None, CG_lv=0):
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
    - col (int): column of panel; should be corresponding to 3 different bA;
        if bA is not None, col should not be None as well
    - CG_lv (int): coarse-graining level; 0 means no coarse-graining
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
        title = fr"Sierpiński Graph of $^{regType}S_{p:d}^{n:d}$"
    else:
        raise ValueError(f"regType={regType} is invalid.")
    if bA is not None:
        if CG_lv == 0:
            title = fr"Learned Graph of $^{regType}S_{p:d}^{n:d}$"
        elif CG_lv > 0:
            title = fr"Learned Graph of $_{CG_lv:d}^{regType}S_{p:d}^{n:d}$"
        else:
            raise NotImplementedError(f"CG_lv={CG_lv} is not implemented")

    if bA is not None:  # singleton graph with transition prob coloring (edge) + self-loop (node)
        beta, A = bA  # assume it's from regType=3, p=3, n=3
        # max_prob = A.max()
        max_prob = 1
        title += fr" ($\beta=+\infty$)" if np.isinf(beta) else fr" ($\beta={beta:.3f}$)"
        n_ = np.shape(A)[0]  # number of nodes
        RGBA_within, RGBA_between = set_community_color(n_)
        colors427 = [[x for x in RGBA_within[0, :]], [x for x in RGBA_between[0, :]]]
        # draw nodes; use RGBA_within
        if annotate is not None:
            scale *= 4.7  # make node larger to fit annotation
        marker_style = dict(marker="o", s=scale)
        annokw = dict(horizontalalignment="center", verticalalignment="center", color="k", fontsize=10)
        if annotate is None:
            for i, x, y in nodeList:
                RGBA_within[i, 3] = A[i, i] / max_prob  # size = max number of edges -> some unused
                marker_style["alpha"] = RGBA_within[i, 3]
                marker_style["facecolor"] = RGBA_within[i, :3]
                marker_style["edgecolor"] = RGBA_within[i, :3]
                ax.scatter(x, y, zorder=2, **marker_style)
        else:
            base_p = annotate if annotate > 0 else 10
            for i, x, y in nodeList:
                RGBA_within[i, 3] = A[i, i] / max_prob  # size = max number of edges -> some unused
                marker_style["alpha"] = RGBA_within[i, 3]
                marker_style["facecolor"] = RGBA_within[i, :3]
                marker_style["edgecolor"] = RGBA_within[i, :3]
                ax.scatter(x, y, zorder=2, **marker_style)
                ax.annotate(str(p_ary(i, p=base_p, L=n)), xy=(x, y), xytext=(x, y), **annokw)
        # draw edges
        counter = 0
        for i in range(0, n_):
            for j in range(i+1, n_): # undirected (A is symmetric)
                # 🔴 assuming nodeList[i][0] = i
                x = [nodeList[i][1],nodeList[j][1]]
                y = [nodeList[i][2],nodeList[j][2]]
                # set alpha to 1 (opaque) if max prob
                RGBA_between[counter, 3] = A[i, j] / max_prob
                ax.plot(x, y, color=RGBA_between[counter, :], zorder=1)
                counter += 1
        nu = 5
        tran_prob_title = ["Self-Loop", "Between Nodes"]
        if col in [0, 1]:  # show red/blue colorbar
            rgba = [[x for x in colors427[col]], [x for x in colors427[col]]]
            rgba[1][3] = 1  # opaque
            cmap = LinearSegmentedColormap.from_list("custom edge color", rgba, N=nu)
            norm = Normalize(vmin=0, vmax=nu - 1)
            kwargs_cax = dict(cax=axcb, drawedges=False)
            cbr = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), **kwargs_cax)
            cbr.set_ticks([(nu - 1) * (2 * i + 1) / (2 * nu) for i in range(nu)])
            cbr.set_ticklabels([f"{x:.2f}" for x in np.linspace(0, max_prob, nu)])
            # cbr.ax.tick_params(labelsize=9)  # no effect
            cbrLabel427(axcb, f"Transition Probability ({tran_prob_title[col]})")
        else:
            axcb.set_axis_off()  # don't show 3rd colorbar
    else:
        # draw nodes; use default PMMM theme color
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
        # draw edges; edge level coloring
        cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:nu], N=nu)
        norm = Normalize(vmin=0, vmax=nu - 1)
        cbr = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axcb)
        cbr.set_ticks([(nu - 1) * (2 * i + 1) / (2 * nu) for i in range(nu)])
        cbr.set_ticklabels([f"{x:d}" for x in np.arange(1, nu + 1)])
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
            ax.set_title(title, fontsize=17)
        else:
            if annotate == -1:
                suffix = "Decimal)"
            else:
                suffix = f"Base-{base_p:d})"
            if bA is None:
                ax.set_title(title[20:-12] + suffix, fontsize=17)
            else:
                ax.set_title(title[:-1] + ", " + suffix, fontsize=17)
        ax.grid(False)
        # plt.legend(loc='upper left')
    else:
        axt.set_frame_on(False)
        axt.set_axis_off()
        kwargs_axt = dict(fontsize=17, transform=axt.transAxes)
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


def set_community_color(n_):
    """
    within level 2 community vs. between level 2 community
    color for w/in: #ef8a62 (239,138,98) (red) | between: #67a9cf (103,169,207) (blue)
    https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=3
    """
    RGBA_within = np.zeros((round(n_*(n_-1)/2), 4))
    RGBA_between = np.zeros((round(n_*(n_-1)/2), 4))
    RGBA_within[:, 0] = 239/255
    RGBA_within[:, 1] = 138/255
    RGBA_within[:, 2] = 98/255
    RGBA_between[:, 0] = 103/255
    RGBA_between[:, 1] = 169/255
    RGBA_between[:, 2] = 207/255
    return RGBA_within, RGBA_between


if __name__ == "__main__":
    CCS_main()
