"""
Empirical CCS plots (including histogram and others for experimental data)
Created: Wednesday, ‎November ‎24, ‎2021, ‏‎2:59:27 PM (EST)
@author: Xiaohuan (Pixel) X.
"""
import os, json
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf  # regressions

from utility427.helper427 import set_dir427, mkdir_p, get_params, partial_427_decorator
from utility427.math427 import log_b, findNearest, rank_eigvals, W_norm, np
from utility427.plt427 import plt, Normalize, LinearSegmentedColormap, GridSpec, Line2D, ticker
from utility427.plt427 import saveNclose427, colors_selector, cbrLabel427, get_violin_pw
from utility427.plt427 import load_CCS_stat, save_masks
from utility427.Sierpinski427 import make_Sierpinski427, p_ary, make_SierpinskiGraph427, W_CG

from CCS427v2 import ax_CCS, make_A_hat_beta


def ECCS_main():
    cwd = set_dir427()  # script dir
    colors = colors_selector(str="5-class Greens")

    MEM = get_ECCS_from_data(cwd=cwd + "/input/empirical_data/", plot_data=[3, 4, 5])
    kwargs = dict(cwd=cwd + "/input/empirical_data/", )
    for nback_idx in [0, 1]:  # 0 -> direct fit; 1 -> MLE
        for gof_thres in [0.80, 0.85, 0.90, 0.95]:
            get_process_beta_2src(nback_idx=nback_idx, gof_thres=gof_thres, **kwargs)


    beta_arr = np.geomspace(0.0001, 10, 400)  # for analytical curve only
    kwargs = dict(sub_folder_name="ECCS", colors=colors)
    plot_hists_ECCS(MEM, (3, 3, 3), **kwargs)
    return 0


def plot_hists_ECCS(
    MEM, params, raw_method=None, err_type="ste",
    colors=None, dpi=None, sub_folder_name=""
):
    """It produces histograms for three params and 1 ECCS plot

    display 2 rows: 3 histograms 1 ECCS plot

    Args
    ----
    - param (tuple): (regType,p,n)

    Kwargs
    ------
    - raw_method (str): if prefix="violin", assume CCS_stat has "raw" key
        - violin: vanilla violin plot
        - violin_median: show median in red, and only 2nd lv CCS as well (1st lv is as expected)
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - colors (list of color hex strings):
        e.g., plt.rcParams['axes.prop_cycle'].by_key()['color'] is default color in pyplot:
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    - sub_folder_name (str):
        the name of the folder in f"output/{whatnot}/" to store the plots
        where <whatnot> is defined in saveNclose427()

    Intermediary
    ------------
    - keys (dict): k:v -> keys in MEM's key:xlabel
    - ECCS_arr (2D nparr): count x (beta,ECCS) (2 levels)
        ECCS_arr[i, [0,1,2]]: [beta, ECCS_1, ECCS_2]

    """
    is_log = True
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=[10, 4])  # initialize

    ds = 0.2  # dummy axes for spacing between the visible plots
    hf = 9  # relative height of (sub)figure
    wf = 1  # relative width of (sub)figure's portion
    width_ratios = [wf] * 31
    height_ratios = [1, hf, ds, 7] + [1, hf, ds]
    kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
    kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
    gs = GridSpec(**kw1)

    axes = dict()
    axes["Hists"], axes["CCS"], axes["labels"] = [None] * 3, [None] * 2, [None] * 5

    fname = f"histECCS"

    keys = dict(r0=r"$r_0$", r1=r"$r_1$", beta=r"$\beta$")
    ibt = 2  # number of portions to space (sub)figures
    axes["Hists"][0] = fig.add_subplot(gs[0:3, 1 : 1 + 9])
    axes["Hists"][1] = fig.add_subplot(gs[0:3, 1 + 9 + ibt : 9 + ibt + 9])
    axes["Hists"][2] = fig.add_subplot(gs[0:3, 18 + ibt + ibt : 18 + ibt + ibt + 9])
    axes["CCS"][0] = fig.add_subplot(gs[4:7, 1 : 1 + 14])
    axes["CCS"][1] = fig.add_subplot(gs[4:7, 1 + 14 + ibt : 1 + 14 + ibt + 14])
    axes["labels"][0] = fig.add_subplot(gs[0:3, 0])
    axes["labels"][1] = fig.add_subplot(gs[0:3, 9 + ibt])
    axes["labels"][2] = fig.add_subplot(gs[0:3, 17 + ibt + ibt])
    axes["labels"][3] = fig.add_subplot(gs[4:7, 0])
    axes["labels"][4] = fig.add_subplot(gs[4:7, 14 + ibt])

    for i, k in enumerate(keys):
        arr = [v[k] for v in MEM.values()]
        ax_hist(axes["Hists"][i], arr, keys[k], density=False)

    kw_axl = dict(fontsize=17, horizontalalignment="center")
    text_labels = ["A", "B", "C", "D", "E"]  # panel label list
    label_rpos = [-0.5] * 5  # relative x position
    for i in range(len(text_labels)):
        # axlabel = fig.add_subplot(gs[0:3, 2*i])
        axes["labels"][i].set_frame_on(False)
        axes["labels"][i].set_axis_off()  # same as ax.axis('off')
        kw_axl.update(dict(transform=axes["labels"][i].transAxes))
        axes["labels"][i].text(label_rpos[i], 1.09, f"{text_labels[i]}", **kw_axl)

    t_test_ECCS(MEM, ccs_lv=2)
    ECCS_arr, num_exc = make_ECCS_arr(MEM, (1e-4, 999), (-5, 10))
    # print(num_exc, ECCS_arr)
    for i in range(2):
        kw_ECCS = dict(ax=axes["CCS"][i], x=ECCS_arr[:, 0], CCS_arrs=None, params=params)
        kw_ECCS.update(dict(raw_method=raw_method, err_type=err_type))
        kw_ECCS.update(dict(show_legend=False, is_log=is_log, colors=colors, dpi=dpi))
        kw_ECCS.update(dict(ECCS_arr=ECCS_arr, CG_lv=0, lv=i))
        ax_ECCS(**kw_ECCS)
    saveNclose427(fig, fname + " (log)"*is_log, dpi=dpi, sub_folder_name=sub_folder_name)


def t_test_ECCS(MEM, ccs_lv=2):
    """
    greater (one-sided) one-sample t-test per ccs_lv
    H0: ECCS = 1
    Ha: ECCS > 1
    """
    xlims = [(4.5e-3, 2), (2e-3, 2e-1)]
    # xlims = [(4.5e-3, 2), (4.5e-3, 2e-1)]
    ylims = [(-5, 10), (-5, 10)]
    kwargs = dict(axis=0, nan_policy="raise", alternative="greater")
    kwargs2 = dict(alternative="greater")
    for i in range(ccs_lv):
        ECCS_arr, num_exc = make_ECCS_arr(MEM, xlims[i], ylims[i])
        n_sample = ECCS_arr.shape[0]  # sample size
        the_mean = np.mean(ECCS_arr[:, i + 1])
        result = stats.ttest_1samp(ECCS_arr[:, i + 1], popmean=1, **kwargs)
        result2 = stats.wilcoxon(ECCS_arr[:, i + 1], y = [1] * n_sample, **kwargs2)
        print(f"ECCS lv={i + 1} | mean = {the_mean:.2f} | n={n_sample}")
        print(f"(t-test) pval = {result.pvalue:.4f}")
        print(f"(Wilcoxon) pval = {result2.pvalue:.4f}")


def make_ECCS_arr(MEM, xlim=None, ylim=None):
    """ exclude outliers
    Kwargs
    ------
    - xlim/ylim (tuple): include only those in [LB, UB] x -> beta; y -> ECCS
    """
    # below is default value
    # if ylim is None:
    #     ylim = (-5, 10)  # exclude if ECCS at any level is < -5 or > 10
    # if xlim is None:
    #     xlim = (0, 999)  # exclude if beta == 1e3 (because beta will not go beyond 1e3 or below 0)
    def nonce(arr, alim):  # arr is x["ECCS"] or [x["beta"]]
        if alim is None:  # no exclusion
            return True
        for a in arr:
            if a < alim[0] or a > alim[1]:
                return False
        return True
    a = np.array(
        [
            [x["beta"], *x["ECCS"]]
            for x in MEM.values()
            if nonce(x["ECCS"], ylim) and nonce([x["beta"]], xlim)
        ]
    )
    num_exc = len(MEM) - a.shape[0]
    return a, num_exc


def ax_hist(ax, arr, xlabel, density=True):
    """
    Intermediary
    ------------
    - stats_data (dict): simple statistics of beta; keys:
        - <stat>: stat (e.g., mean/median) among all 100 subjects
        - <stat>_mid: stat among the mid_range subjects (i.e., those s.t. 0 < beta < 1000)
        - 0 or 1000: number of subjects whose beta = 0 or 1000
    """
    kwargs = dict(density=density)
    if xlabel != r"$\beta$":
        scale_y = 10 ** 3 if xlabel == r"$r_0$" else 10 ** 4
        xlabel += " (ms)"
    else:  # beta
        arr = np.sort(arr)
        stats_data = dict()
        stats_data["mean"] = np.mean(arr)
        stats_data["median"] = np.median(arr)
        bool_0s = np.isclose(arr, 0, rtol=1e-05)
        bool_1ks = np.isclose(arr, 1000, rtol=1e-05)
        bool_mid = np.logical_not(bool_0s | bool_1ks)
        stats_data["0"] = sum(bool_0s)
        stats_data["1000"] = sum(bool_1ks)
        stats_data["sup2.5"] = sum(arr >= 2.5)
        arr = arr[bool_mid]
        stats_data["mean_mid"] = np.mean(arr)
        stats_data["median_mid"] = np.median(arr)
        print(stats_data)
        kwargs.update(dict(bins=10))
        # ticks_loc = np.logspace(10**(-4), 10, num=6)
        # ticks_label = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"]
        # ax.set_xticks(ticks_loc)
        # ax.set_xscale("log")  # set x to log scale
        # ax.set_xlim([10**(-3), 10])
        ax.set_xlim([0, 2.5])
    ax.hist(arr, **kwargs)
    # ax.set_title(title, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=11)
    styles_txt = dict(fontsize=11, horizontalalignment="center", transform=ax.transAxes)
    if xlabel == r"$r_1$ (ms)":
        ax.set_xlim([-10**4, 10**4])
        ticks_x = ticker.FuncFormatter(lambda x, pos: f"{x / 10**4:g}")
        ax.xaxis.set_major_formatter(ticks_x)
        ax.text(0.85, 0.15, r"$\times10^{4}$", **styles_txt)
    if density:
        if xlabel != r"$\beta$":
            ticks_y = ticker.FuncFormatter(lambda x, pos: f"{x * scale_y:g}")
            ax.yaxis.set_major_formatter(ticks_y)
            if xlabel == r"$r_0$ (ms)":
                ax.text(0, 1.05, r"$\times10^{-3}$", **styles_txt)
            else:
                ax.text(0, 1.05, r"$\times10^{-4}$", **styles_txt)
        ax.set_ylabel("Probability Density", fontsize=11)
    else:
        ax.set_ylabel("Frequency", fontsize=11)


def ax_ECCS(ax, x, CCS_arrs, params, raw_method=None, ECCS_arr=None, err_type='ste', CG_lv=0,
           show_legend=False, is_log=True, colors=None, dpi=None, lv=0):
    """
    Args
    ----
    - ax: axis object
    - x: a list of beta
    - params (tuple): (regType, p, n)
        - regType (int):
            0: default Sierpiński graph
            x: Sierpiński-like graph of type x regularization

    Kwargs
    ------
    - raw_method (str): if prefix="violin", assume CCS_stat has "raw" key
        - violin: vanilla violin plot; show mean and SD
        - violin_median: show median and ste_median instead
    - ECCS_arr (2D nparr): ECCS_arr[i, [0,1,2]]: [beta, ECCS_1, ECCS_2]
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - show_legend (bool): whether we show simulation parameters
    - is_log (bool): if True then use log scale on x axis.

    Intermediary
    ------------

    Return
    ------
    - xmaxs (list): list of beta that maximizes CCS at each level
        xmaxs[1]: beta that maximizes CCS at lv2/lv3
        xmaxs[i]: beta that maximizes CCS at lv(i+1)/lv(i+2)
    """
    # set up beta range (analytical) for plot
    if is_log:
        x_range = (min(x) * 0.55, max(x) * 1.55)
    else:
        delta = (max(x) - min(x)) * 0.05
        x_range = (min(x) - delta, max(x) + delta)
    ax.set_xlim(x_range)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    (regType, p, n) = params
    # CCS_arr = CCS_arrs["CCS"]  # analytical
    # n_level = CCS_arr.shape[2]
    n_level = n - 1  # for (3,3,3), max CCS level is 2
    if n_level == n - 1:  # normally there are only n-1* levels (since CCS is function of 2 lvs)
        pass
    elif n_level == n:  # *it could be violated, due to regularization
        n_level -= 1  # don't show higher level introduced by regularization
    else:
        raise Exception("something went wrong in <CCS_arr.shape[2]>")
    CCS_type_slice = 0
    cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:n_level], N=n_level)
    xlabel = r"Shuffling Parameter $\beta$"
    ylabel = "ECCS"
    if regType in [0, 1, 2, 3]:
        title = fr"ECCS of $^{regType}S_{p:d}^{n:d}$ (lv${lv + 1}$/lv${lv + 2}$)"
    else:
        raise NotImplementedError(f"<regType>={regType} is invalid")

    if show_legend:  # only show walk length
        styles_txt = dict(fontsize=11, horizontalalignment="center", transform=ax.transAxes)
        n_agents = 100  # num of subjects
        n_steps = 1500  # number of steps
        topy, s = 0.94, 0.06
        ax.text(0.80, topy - 1 * s * 0, f"walk length={n_steps:.0f}", **styles_txt)

    sty1 = dict(alpha= 0.74, linewidth= 2)

    def temp_a1(i):  # draw ECCS w/o analytical for now
        sty1.update(dict(color=cmap(i)))
        if show_legend:
            sty1.update(dict(label=f"lv{i+1}/lv{i+2}"))
        # ax.plot(x, CCS_arr[:, CCS_type_slice, i], **sty1)  # plot analytical curve
        # put scatter points of empirical results in
        sty2 = dict(linestyle="None", marker=".", s=11, color=cmap(i))
        sty2.update(dict(alpha = 0.74, linewidth = 2, zorder=2))
        # if show_legend:
        #     sty2.update(dict(label='Stochastic '+labels[i].replace('-','/lv')))
        kw_spt = dict(x=ECCS_arr[:, 0])
        kw_spt.update(dict(y=ECCS_arr[:, i + 1]))
        ax.scatter(**kw_spt, **sty2)
        if raw_method is None:
            return 0
    

    # def temp_b1(i):  # annotate CCS plot with maxima
    #     return 0  # no annotation for now
    #     xmaxs[i] = x[np.argmax(CCS_arr[:, CCS_type_slice, i])]
    #     ymaxs[i] = np.max(CCS_arr[:, CCS_type_slice, i])
    #     # texts[i] = f"({xmaxs[i]:.3f},{ymaxs[i]:.3f})"  # show both beta and CCS
    #     texts[i] = f"{xmaxs[i]:.3f}"  # show only beta
    #     kw_text.update(dict(color=cmap(i), xy=(xmaxs[i], ymaxs[i])))
    #     kw_text.update(dict(xytext=(0.85 - i * 0.2, 0.05)))
    #     ax.annotate(texts[i], **kw_text)

    # argmax = beta that maximizes CCS at different permissible levels
    arrowprops = dict(arrowstyle="simple", facecolor="grey", edgecolor="grey")
    arrowprops.update(dict(linewidth=1 / 3, alpha=0.74))
    kw_text = dict(textcoords="axes fraction", fontsize=11, arrowprops=arrowprops)
    kw_text.update(dict(ha="center", va="center"))

    xmaxs, ymaxs, texts = [None] * (n-1), [None] * (n-1), [None] * (n-1)
    temp_a1(lv)  # one level per plot (previously all levels per plot)
    # for i in range(n_level):  # n_level is always n-1; see earlier code for why this is true
    #     temp_a1(i)
        # temp_b1(i)  # put peak val on plot

    # ax.set_ylim((0.9, 1.3))  # for regType=3, p=3, n=3 max ECCS is <1.3
    ax.plot(ax.get_xlim(), (1, 1), "--", color="grey", zorder=0)  # draw y=1 line in grey

    ax.set_title(title, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if is_log:
        ax.set_xscale("log")  # set x to log scale
        loc="upper left"
    else:
        loc="center right"
    temp_li, temp_la = ax.get_legend_handles_labels()
    if show_legend:
        ax.legend(temp_li, temp_la, loc=loc)
    ax.grid(False)
    return xmaxs


def get_ECCS_from_data(cwd="", plot_data=None):
    """calculate ECCS for each subject
    process processed data (including rt_pred) and MEM results into dictionaries

    ECCS: for each subject
    anticipation = (regressed_out rt - r0) / r1 (for given edgelv)
    ECCS at level l = mean(anticipation at edgelv=l) - mean(anticipation at edgelv=l+1)
    NOTE however, mean() is two-fold:
        first mean across all instances of transitions on given edge
        then mean across edges at given edgelv (using results from above step)

    Return
    ------
    - MEM (dict): k:v -> "id":fit_dict
        fit_dict (dict): keys: "beta", "r0", "r1"; "ECCS" is arr of ECCS
    """
    temp = os.listdir(cwd)
    fname = "MEM.json"
    if fname in temp:
        with open(cwd + fname) as f:
            MEM = json.load(f)
        keys = list(MEM.keys())
        for k in keys:  # change key: str -> int
            MEM[int(k)] = MEM.pop(k)
    else:
        df = pd.read_csv(cwd + "MEM_results.csv")
        df = df[["id", "beta", "r0", "r1"]]
        df = df.set_index("id")
        MEM = df.to_dict("index")


    def ant(x):  # x.name is "id"
        return (x.loc["rt"] - x.loc["rt_pred"] - MEM[x.name]["r0"]) / MEM[x.name]["r1"]

    df = pd.read_csv(cwd + "data_sier.csv")
    df = df.set_index("id")
    df["ant"] = df.apply(ant, axis=1)
    pd_series = df.groupby(level="id").apply(ECCS)

    for id, val in pd_series.iteritems():
        MEM[id]["ECCS"] = val[0]
        MEM[id]["ants_arr"] = val[1]
        MEM[id]["rts_arr"] = val[2]
        MEM[id]["rts_edgelv"] = val[3]
        MEM[id]["rts_pred_arr"] = val[4]
        MEM[id]["rts_pred_edgelv"] = val[5]
        MEM[id]["edges_arr"] = val[6]

    if plot_data is None:  # plot none
        plot_data = [-1]
    else:  # set up some global params
        txt_size = 5
        txt_id_x, txt_id_y = (0.10, 0.94)
        txt_beta_x, txt_beta_y = (0.50, 0.94)
    if 1 in plot_data:  # plot emp. anticipation
        fig = plt.figure(figsize=[20, 20])
        for i, id in enumerate(MEM.keys()):
            ax = fig.add_subplot(10, 10, i + 1)
            x = [i for i in range(len(MEM[id]["ants_arr"]))]
            ax.scatter(x, MEM[id]["ants_arr"])
        saveNclose427(fig, "ants_arrs", dpi=300, sub_folder_name="100subs")

    if 2 in plot_data:  # plot rt & rt_ro for each edgelv
        fig2 = plt.figure(figsize=[14, 20])
        ylabels = [r"rt (raw)", r"rt (regressed-out)"]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:2], N=2)
        ax1 = fig2.add_subplot(1, 2, 1)
        ax2 = fig2.add_subplot(1, 2, 2)
        for i, id in enumerate(MEM.keys()):
            for j in range(2):  # 3 edgelv
                y = [MEM[id]["rts_edgelv"][j], MEM[id]["rts_edgelv"][j + 1]]
                color = cmap(0) if y[1] > y[0] else cmap(1)  # blue if rt increases
                ax1.plot([1 + j, 2 + j], y, color=color)
                y2 = [MEM[id]["rts_pred_edgelv"][j], MEM[id]["rts_pred_edgelv"][j + 1]]
                y2 = [y[x] - y2[x] for x in range(2)]
                color = cmap(0) if y2[1] > y2[0] else cmap(1)  # blue if rt_ro increases
                ax2.plot([1 + j, 2 + j], y2, color=color)

        for i, ax in enumerate([ax1, ax2]):
            ax.set_xticks([1, 2, 3])
            ax.set_xlabel(r"edgelv", fontsize=11)
            ax.set_ylabel(ylabels[i], fontsize=11)
        saveNclose427(fig2, "rts_edgelv", dpi=300, sub_folder_name="100subs")

    if 3 in plot_data:
        fig3 = plt.figure(figsize=[14, 14])  # plot rt for each theo. anticipation
        fig4 = plt.figure(figsize=[14, 14])  # plot rt_ro for each theo. anticipation
        A = make_SierpinskiGraph427(3, 3, norm=True, regType=3, use_set=False)["A"]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:2], N=2)
        kwargs_style = dict(s=7)
        for i, id in enumerate(MEM.keys()):
            Ahat = make_A_hat_beta(A, MEM[id]["beta"], CG_lv=0, tup=None)
            x = [Ahat[e[0], e[1]] for e in MEM[id]["edges_arr"]]

            ax1 = fig3.add_subplot(10, 10, 1 + i)
            ax1.scatter(x, MEM[id]["rts_arr"], **kwargs_style)

            ax2 = fig4.add_subplot(10, 10, 1 + i)
            y = [
                MEM[id]["rts_arr"][x] - MEM[id]["rts_pred_arr"][x]
                for x in range(len(MEM[id]["edges_arr"]))
            ]
            ax2.scatter(x, y, **kwargs_style)
            for ax in [ax1, ax2]:
                styles_txt = dict(fontsize=txt_size, horizontalalignment="center", transform=ax.transAxes)
                ax.text(txt_beta_x, txt_beta_y, fr"$\beta$: {MEM[id]['beta']:.3f}", **styles_txt)
                ax.text(txt_id_x, txt_id_y, f"id: {id:d}", **styles_txt)
                ax.tick_params(axis="both", which="major", labelsize=txt_size)

        saveNclose427(fig3, "rts_theo-ants", dpi=300, sub_folder_name="100subs")
    if 4 in plot_data:
        saveNclose427(fig4, "rts_ro_theo-ants", dpi=300, sub_folder_name="100subs")

    if 5 in plot_data:  # plot rt_ro for each theo. anticipation at different betas & linear reg
        sub_ids = [1, 2, 4, 5, 7, 10, 123]  # choose what subject to plot
        betas = np.geomspace(0.0001, 10, 40)
        A = make_SierpinskiGraph427(3, 3, norm=True, regType=3, use_set=False)["A"]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:2], N=2)
        kwargs_style = dict(s=7)
        for id in sub_ids:
            fig5 = plt.figure(figsize=[20, 7])
            y = [
                MEM[id]["rts_arr"][x] - MEM[id]["rts_pred_arr"][x]
                for x in range(len(MEM[id]["edges_arr"]))
            ]
            res_list = [None] * len(betas)
            for i, beta in enumerate(betas):
                if i == 0:  # first beta uses fitted beta instead
                    beta = MEM[id]["beta"]
                Ahat = make_A_hat_beta(A, beta, CG_lv=0, tup=None)
                x = [Ahat[e[0], e[1]] for e in MEM[id]["edges_arr"]]
                res = smf.ols("y ~ x", data=dict(x=x, y=y)).fit()  # simplest linear regression
                pred_ols = res.get_prediction()
                res_list[i] = (i, x, beta, res, pred_ols, np.sqrt(res.ssr/len(x)))
            res_list = sorted(res_list, key=lambda x:[x[-1], x[2]], reverse=True)
            for i, tup in enumerate(res_list):
                j, x, beta, res, pred_ols, gof = tup  # unpack
                ax = fig5.add_subplot(4, 10, 1 + i)
                styles_txt = dict(fontsize=txt_size, horizontalalignment="center", transform=ax.transAxes)
                ax.scatter(x, y, **kwargs_style)
                temp_str = "r" if j != 0 else "k"
                ax.plot(x, res.fittedvalues, f"{temp_str}-")
                ax.plot(x, pred_ols.summary_frame()["obs_ci_upper"], f"{temp_str}--")
                ax.plot(x, pred_ols.summary_frame()["obs_ci_lower"], f"{temp_str}--")
                ax.tick_params(axis="both", which="major", labelsize=txt_size)
                ax.text(txt_beta_x, txt_beta_y, fr"$\beta$: {beta:.3f}", **styles_txt)
                ax.text(txt_id_x, txt_id_y, f"id: {id:d}", **styles_txt)
                ax.text(0.85, txt_beta_y, fr"RMSE: {gof:.3f}", **styles_txt)
                ax.text(txt_id_x + 0.05, 0.12, fr"$p_0$: {res.pvalues['Intercept']:.3f}", **styles_txt)
                ax.text(txt_id_x + 0.05, 0.05, fr"$p_1$: {res.pvalues['x']:.3f}", **styles_txt)
            saveNclose427(fig5, f"rts_ro_theo-ants_betas_id={id}", dpi=300, sub_folder_name="100subs")
    


    with open(cwd + fname, "w") as f:
        json.dump(MEM, f, indent=4)
    return MEM


def get_process_beta_2src(cwd="", nback_idx=1, gof_thres=0.80):
    """
    This is to get beta from two sources: sier = serial response; nback = n-back
    and then process them:
    1) beta vs. beta
    2) Anderson-Darling normality test on both
    3) Spearman correlation

    Kwarg
    -----
    - nback_idx (int): estimate of nback beta; 0 -> direct fit; 1 -> MLE; 2 -> gof for 0
    - gof_thres (float): in analysis, only use those whose gof >= gof_thres

    Intermediary
    ------------
    beta_sier/beta_nback (dict): k:v -> id:beta (NOTE we convert id str -> int)
    beta_arr (2D nparr): row -> index for id; col -> beta for each source
    beta_arr_filt (2D nparr): keep non-extreme & good gof
        1) 0 <= beta <= 6 on both axes
        2) gof >= 0.8
    """
    temp = os.listdir(cwd)
    fname = "MEM.json"
    if fname in temp:
        with open(cwd + fname) as f:
            MEM = json.load(f)
    else:
        df = pd.read_csv(cwd + "MEM_results.csv")
        df = df[["id", "beta"]]
        df = df.set_index("id")
        MEM = df.to_dict("index")
    beta_sier = {int(k):v["beta"] for k,v in MEM.items()}

    fname = "beta_nback.json"
    if fname in temp:
        with open(cwd + fname) as f:
            beta_nback = json.load(f)
    else:
        df = pd.read_csv(cwd + "beta_nback.csv")
        df = df.set_index("id")
        beta_nback = df.to_dict("index")
        temp_keys = ["beta_nback", "beta_nback_MLE", "gof_nback"]  # gof -> r-squared adjusted
        beta_nback = {k:[v[key] for key in temp_keys] for k,v in beta_nback.items()}
        with open(cwd + fname, "w") as f:  # int -> str in keys automatically
            json.dump(beta_nback, f, indent=4)
    beta_nback = {int(k):v for k,v in beta_nback.items()}  # str -> int

    # just checking: nback should be contained in sier
    temp = set(beta_sier.keys()).difference(set(beta_nback.keys()))
    if temp:
        raise Exception("id for serial response and nback are not the same")

    # filtering
    key_set = set()  # subject ids that remain
    beta_arr = np.zeros((len(beta_sier), 2), dtype=float)
    for i, k in enumerate(beta_sier):
        beta_arr[i, 0] = beta_sier[k]
        beta_arr[i, 1] = beta_nback[k][nback_idx]
        if 0 <= beta_sier[k] <= 6 and 0 <= beta_nback[k][nback_idx] <= 6 and beta_nback[k][2] >= gof_thres:
            key_set.add(k)
    # print(f"DEBUG beta ids that remain: {key_set}")
    beta_arr_filt = np.zeros((len(key_set), 2), dtype=float)
    for i, k in enumerate(key_set):
        beta_arr_filt[i, 0] = beta_sier[k]
        beta_arr_filt[i, 1] = beta_nback[k][nback_idx]

    # visualization & analyses
    txt_beta_arr = "Direct-Fit" if nback_idx == 0 else "MLE"
    n_spearmanr = beta_arr_filt.shape[0]
    spearmanr = stats.spearmanr(beta_arr_filt, axis=0)
    ander_sier = stats.anderson(beta_arr_filt[:, 0], dist="norm")
    ander_nback = stats.anderson(beta_arr_filt[:, 1], dist="norm")
    print(f"\n nback {txt_beta_arr} β; gof thresholded at>={gof_thres:.2f}:")
    print(f"network β:{ander_sier.statistic:.3g}\n", ander_sier.critical_values, ander_sier.significance_level)
    print(f"nback β:{ander_nback.statistic:.3g}\n", ander_nback.critical_values, ander_nback.significance_level, end="\n")

    fig = plt.figure(figsize=[14, 6])  # initialize
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    axes = [ax1, ax2]
    ax1.scatter(beta_arr[:, 1], beta_arr[:, 0])
    ax1.set_title(r"$\beta$: network vs. n-back", fontsize=17)
    ax2.scatter(beta_arr_filt[:, 1], beta_arr_filt[:, 0])
    ax2.set_title(fr"$\beta$: network vs. n-back ({txt_beta_arr}; gof$\geq${gof_thres:.2f})", fontsize=17)
    styles_txt = dict(fontsize=11, horizontalalignment="center", transform=ax2.transAxes)
    ax2.text(0.8, 0.95, f"$r_s={spearmanr.correlation:.3f}$, $p={spearmanr.pvalue:.3f}$", **styles_txt)
    ax2.text(0.8, 0.90, f"$n={n_spearmanr:d}$", **styles_txt)

    for ax in axes:
        ax.set_xlabel(r"n-back $\beta$", fontsize=11)
        ax.set_ylabel(r"network $\beta$", fontsize=11)
    saveNclose427(fig, f"beta_arr_{txt_beta_arr}_gofgeq-{gof_thres:.2f}", dpi=300, sub_folder_name="100subs\\betas")


def ECCS(df):
    """df should come from groupby(level="id")
    df should have fields: "ant", "node", "node_prev", "edgelv", "rt", "rt_pred"

    Intermediary
    ------------
    - ants (dict): k:v -> edge:[n, x]; x is sum of anticipations; n is number of instances
        edge is a frozenset of two elements (or one, if self-loop): source and target nodes
        n += 1 each time we add a new ant to x
        becomes k:v -> edge:x_mean after first mean calculation
    - rts (dict): ditto but replace anticipation with rt
    - edgelvs (dict): k:v -> edgelv (int):set of edge (set of frozenset)

    after completing ants (iterating over all rows of df)
    we find mean for each edge; then mean for each edgelv; then ECCS

    Return
    ------
    eccs (list): eccs[i]: level i+1 ECCS
    ants_arr (list): mean ant for each edge (sorted by level, ascending order)
    rts_arr/rts_pred_arr (list): ditto but with rt/rt_pred instead
    rts_edgelv/rts_pred_edgelv (list): mean rt/rt_pred per edgelv
    edges_arr (list): list of len-2 lists; NOTE the list is ordered (frozenset -> list)
    """
    ants, rts, rts_pred = dict(), dict(), dict()
    edgelvs = dict()
    for row in df.itertuples(index=False):
        edge = frozenset([row.node, row.node_prev])
        if edge not in ants:  # add new entries to ants and edgelvs
            ants[edge] = [1, row.ant]
            rts[edge] = [1, row.rt]
            rts_pred[edge] = [1, row.rt_pred]
            if row.edgelv not in edgelvs:  # add only if never seen this edgelv
                edgelvs[row.edgelv] = set([edge])
            else:  # add new edge for this edgelv
                edgelvs[row.edgelv].add(edge)
        else:  # if we've seen that edge, we only need to update ants
            ants[edge][0] += 1  # update number of ant along this edge
            ants[edge][1] += row.ant
            rts[edge][0] += 1  # update number of rt along this edge
            rts[edge][1] += row.rt
            rts_pred[edge][0] += 1  # update number of rt_pred along this edge
            rts_pred[edge][1] += row.rt_pred

    edgelvs = {k:list(v) for k,v in edgelvs.items()}  # s.t. edge order is fixed

    # first mean for each edge
    for edge in ants:
        ants[edge] = ants[edge][1] / ants[edge][0]
        rts[edge] = rts[edge][1] / rts[edge][0]
        rts_pred[edge] = rts_pred[edge][1] / rts_pred[edge][0]
    # then mean for each edgelv
    lv_arr = sorted([x for x in edgelvs if x >= 1])  # 1,2,...,lv_max
    edges_arr = [list(e) for lv in lv_arr for e in edgelvs[lv]]  # s.t. edge is subscriptable
    mean_weights = [0.0 for _ in range(lv_arr[-1])]
    rts_edgelv = [0.0 for _ in range(lv_arr[-1])]
    rts_pred_edgelv = [0.0 for _ in range(lv_arr[-1])]
    for lv in lv_arr:
        mean_weights[lv - 1] = np.mean([ants[e] for e in edgelvs[lv]])
        rts_edgelv[lv - 1] = np.mean([rts[e] for e in edgelvs[lv]])
        rts_pred_edgelv[lv - 1] = np.mean([rts_pred[e] for e in edgelvs[lv]])

    eccs = list(np.divide(mean_weights[:-1], mean_weights[1:]))  # calculate ECCS
    # turn ants & rts into a sorted list
    ants_arr = [ants[e] for lv in lv_arr for e in edgelvs[lv]]
    rts_arr = [rts[e] for lv in lv_arr for e in edgelvs[lv]]
    rts_pred_arr = [rts_pred[e] for lv in lv_arr for e in edgelvs[lv]]

    return eccs, ants_arr, rts_arr, rts_edgelv, rts_pred_arr, rts_pred_edgelv, edges_arr


if __name__ == "__main__":
    ECCS_main()
