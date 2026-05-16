"""
Empirical CCS: paper Fig.6
Created: Wednesday, November 24, 2021, 2:59:27 PM (EST)
@author: Xiaohuan (Pixel) X.
"""
import os, json
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf  # regressions

from utility427.helper427 import set_dir427, mkdir_p, get_params, partial_427_decorator, print427
from utility427.math427 import log_b, findNearest, rank_eigvals, W_norm, np, pval_star
from utility427.plt427 import plt, Normalize, LinearSegmentedColormap, GridSpec, Line2D, ticker
from utility427.plt427 import saveNclose427, colors_selector, cbrLabel427, get_violin_pw
from utility427.plt427 import load_CCS_stat, save_masks
from utility427.Sierpinski427 import make_Sierpinski427, p_ary, make_SierpinskiGraph427, W_CG
from utility427.sim_params427 import make_beta_boundry

FIGNUM = 6  # which figure to generate; 6

FS_TICKLAB = 12
FS_LAB = 14
FS_MAIN = 17
FS_TICKLEN = 5.5
FS_TICKWID = 1.7
FS_MINOR_TICKLEN = 3
FS_MINOR_TICKWID = 1.4

TICK_PARAMS = dict(labelsize=FS_TICKLAB, length=FS_TICKLEN, width=FS_TICKWID)
TICK_PARAMS_MINOR = dict(length=FS_MINOR_TICKLEN, width=FS_MINOR_TICKWID)

ALT=False  # directly used inside ECCS2()

def ECCS_main():
    cwd = set_dir427()  # script dir
    colors = colors_selector(str="7-class Greens")
    ECCS_type = 2
    yrange = (-3, 6) if ECCS_type == 1 else None

    MEM = get_ECCS_from_data(cwd=cwd + "/input/empirical_data/", ECCS_type=ECCS_type)
    beta_arr = np.geomspace(0.0001, 10, 400)  # for analytical curve only
    kwargs = dict(sub_folder_name="ECCS", colors=colors, ECCS_type=ECCS_type)
    kwargs.update(dict(yrange=yrange, boxed=True))
    plot_hists_ECCS(MEM, (3, 3, 3), **kwargs)

    return 0

    kwargs = dict(cwd=cwd + "/input/empirical_data/", )
    for nback_idx in [0, 1]:  # 0 -> direct fit; 1 -> MLE
        for gof_thres in [None, 0, 0.80, 0.85, 0.90, 0.95]:
            get_process_beta_2src(nback_idx=nback_idx, gof_thres=gof_thres, **kwargs)



def plot_hists_ECCS(
    MEM, params, err_type="ste", ECCS_type=2, yrange=None, boxed=False,
    colors=None, dpi=None, sub_folder_name=""
):
    """It produces histograms for three params and 1 ECCS plot

    display 2 rows: 3 histograms 1 ECCS plot

    Args
    ----
    - param (tuple): (regType,p,n)

    Kwargs
    ------
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - colors (list of color hex strings):
        e.g., plt.rcParams['axes.prop_cycle'].by_key()['color'] is default color in pyplot:
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    - sub_folder_name (str):
        the name of the folder in f"output/{whatnot}/" to store the plots
        where <whatnot> is defined in saveNclose427()
    - ECCS_type (int):
        this only affects file name and the actual selection of ECCS type happens in make_ECCS_arr()
    - yrange (None or len-2 list-like):
        if not None, (a, b): a <= ECCS (any level) <= b
    - boxed (bool): whether we bin the ECCS

    Intermediary
    ------------
    - keys (dict): k:v -> keys in MEM's key:xlabel
    - ECCS_arr (2D nparr): count x (beta,ECCS) (2 levels)
        ECCS_arr[i, [0,1,2]]: [beta, ECCS_1, ECCS_2]

    """
    is_log = True
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=[20, 9])  # initialize

    ds = 0.2  # dummy axes for spacing between the visible plots
    hf = 9  # relative height of (sub)figure
    wf = 1  # relative width of (sub)figure's portion
    width_ratios = [wf] * 31
    height_ratios = [1, hf, ds, 3] + [1, hf + 4, ds]
    kw1 = {"nrows": len(height_ratios), "ncols": len(width_ratios)}
    kw1.update({"height_ratios": height_ratios, "width_ratios": width_ratios})
    gs = GridSpec(**kw1)

    axes = dict()
    axes["Hists"], axes["CCS"], axes["labels"] = [None] * 3, [None] * 2, [None] * 5


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
        ax_hist(axes["Hists"][i], arr, keys[k], density=False, logscale=i==2)

    kw_axl = dict(fontsize=FS_MAIN, horizontalalignment="center")
    text_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]  # panel label list
    label_rposx = [-0.3] * 5  # relative x position
    label_rposy = [1.04] * 5  # relative y position
    for i in range(len(text_labels)):
        # axlabel = fig.add_subplot(gs[0:3, 2*i])
        axes["labels"][i].set_frame_on(False)
        axes["labels"][i].set_axis_off()  # same as ax.axis('off')
        kw_axl.update(dict(transform=axes["labels"][i].transAxes))
        axes["labels"][i].text(label_rposx[i], label_rposy[i], f"{text_labels[i]}", **kw_axl)

    temp_txt = "full" if yrange is None else f"{yrange[0]:d}to{yrange[1]:d}"
    fname = f"Fig{FIGNUM}.histECCS{ECCS_type}_{temp_txt}"
    ECCS_arr, num_exc = make_ECCS_arr(MEM, (1e-4, 999), yrange, ECCS_type=ECCS_type)
    box_dict = sig_tests_ECCS(ECCS_arr, num_exc, ccs_lv=2, boxed=boxed)
    spearmanr = stats.spearmanr(ECCS_arr[:, [1, 2]], axis=0)
    temp_var = f"corr={spearmanr.correlation:.3f} pval={spearmanr.pvalue:.5f}"
    print427(f"corr: ECCS{ECCS_type} lv1 to lv2", var=temp_var)
    box_list = [box_dict.pop("bbdry"), None]
    box_list[1] = box_dict
    for i in range(2):
        kw_ECCS = dict(ax=axes["CCS"][i], ECCS_arr=ECCS_arr, params=params)
        kw_ECCS.update(dict(box_list=box_list, err_type=err_type))
        kw_ECCS.update(dict(show_legend=False, is_log=is_log, colors=colors, dpi=dpi))
        kw_ECCS.update(dict(CG_lv=0, lv=i, yrange=yrange, ECCS_type=ECCS_type))
        ax_ECCS(**kw_ECCS)
    fname += "_log" * is_log + "_boxed" * boxed
    saveNclose427(fig, fname, dpi=dpi, sub_folder_name=sub_folder_name)


def sig_tests_ECCS(ECCS_arr, num_exc, ccs_lv=None, boxed=False):
    """
    greater (one-sided) one-sample t-test / Wilcoxon per ccs_lv
    they test if mean/median is different from null's mean/median
    Wilcoxon signed-rank test tests if diff (one sample - null) is symmetric around 0 
    H0: ECCS = 1
    Ha: ECCS > 1

    also test diff (lv l - lv (l+1))

    - boxed (bool): if True, we return a dict `box_dict`
    the bin is left inclusive: [a, b) i.e., a <= beta < b (the last one obviously is <=b)
    "bbdry": bbdry from make_beta_boundry()
    b: b is box index, 0,...,9, val is another dict with following k:v pair
        "mask": (1D nparr) bool_mask of ECCS_arr in that box
        "n": (int) number of subjects in that bin
        "center": (float) center beta of the bin
        "Wilcoxon": (list) Wilcoxon pval for each CCS lv

    return arr, bbdry

    xlims = [(4.5e-3, 2), (2e-3, 2e-1)]
    xlims = [(4.5e-3, 2), (4.5e-3, 2e-1)]
    ylims = [(-5, 10), (-5, 10)]
    """
    kwargs = dict(axis=0, nan_policy="raise", alternative="greater")
    kwargs2 = dict(alternative="greater")
    n_sample = ECCS_arr.shape[0]  # sample size
    if ccs_lv is None:
        ccs_lv = ECCS_arr.shape[1] - 1  # infer number of CCS level
    print427("ECCS mean/median - 1 compared to 0", var=f"n={n_sample} | removed {num_exc} outliers")
    for i in range(ccs_lv):
        diff = ECCS_arr[:, i + 1] - 1  # compared to baseline of 1
        the_mean = np.mean(diff)
        result = stats.ttest_1samp(diff, popmean=0, **kwargs)
        result2 = stats.wilcoxon(diff, **kwargs2)
        temp_txt = f"lv={i + 1} | diff mean = {the_mean:.2f}"
        temp_txt += f" | (t-test) stat={result.statistic:.2f} pval={result.pvalue:.4f}"
        temp_txt += f" | (Wilcoxon) stat={result2.statistic:.2f} pval={result2.pvalue:.4f}"
        print(temp_txt)

    print427("ECCS mean/median: one lv to coarser lv", var=f"n={n_sample} | removed {num_exc} outliers")
    for i in range(ccs_lv-1):
        diff = ECCS_arr[:, i + 1] - ECCS_arr[:, i + 2]  # ECCS diff across 2 adjacent levels
        the_mean = np.mean(diff)
        result = stats.ttest_1samp(diff, popmean=0, **kwargs)
        result2 = stats.wilcoxon(diff, **kwargs2)
        temp_txt = f"lv={i + 1}-lv={i + 2} | diff mean = {the_mean:.2f}"
        temp_txt += f" | (t-test) stat={result.statistic:.2f} pval={result.pvalue:.4f}"
        temp_txt += f" | (Wilcoxon) stat={result2.statistic:.2f} pval={result2.pvalue:.4f}"
        print(temp_txt)

    if boxed:
        bbdry = make_beta_boundry(10, b=10)  # log-uniform; 10 bins/boxes
        box_dict = dict()
        box_dict["bbdry"] = bbdry
        # assume ECCS_arr is sorted in beta (ECCS_arr[:, 0])
        for i, beta in enumerate(bbdry[:, 0]):  # loop over centers
            box_dict[i] = dict()
            bool_mask = bbdry[i, 1] <= ECCS_arr[:, 0]
            if i != n_sample - 1:
                bool_mask = np.logical_and(bool_mask, ECCS_arr[:, 0] < bbdry[i, 2])
            else:
                bool_mask = np.logical_and(bool_mask, ECCS_arr[:, 0] <= bbdry[i, 2])
            box_dict[i]["mask"] = bool_mask
            box_dict[i]["n"] = sum(bool_mask)
            box_dict[i]["center"] = beta
            box_dict[i]["Wilcoxon"] = [None] * ccs_lv
            if box_dict[i]["n"] != 0:  # if == 0, undefined (None) pval since n=0
                for j in range(ccs_lv):
                    diff = ECCS_arr[bool_mask, j + 1] - 1  # compared to baseline of 1
                    box_dict[i]["Wilcoxon"][j] = stats.wilcoxon(diff, **kwargs2).pvalue
        return box_dict
    else:
        return None


def make_ECCS_arr(MEM, xlim=None, ylim=None, ECCS_type=1, sort_beta=True):
    """ exclude outliers
    Kwargs
    ------
    - xlim/ylim (tuple): include only those in [LB, UB] x -> beta; y -> ECCS
    - sort_beta (bool): whether we sort the entries of a by beta (a[:, 0])
    """
    temp_eccs = "ECCS" if ECCS_type == 1 else f"ECCS{ECCS_type}"
    def nonce(arr, alim):  # arr is x[temp_eccs] or [x["beta"]]
        if alim is None:  # no exclusion
            return True
        for a in arr:
            if a < alim[0] or a > alim[1]:
                return False
        return True
    a = np.array(
        [
            [x["beta"], *x[temp_eccs]]
            for x in MEM.values()
            if nonce(x[temp_eccs], ylim) and nonce([x["beta"]], xlim)
        ]
    )
    num_exc = len(MEM) - a.shape[0]
    if sort_beta:
        idx_sorted = np.argsort(a[:, 0])  # sort by beta
        a = a[idx_sorted, :]
    return a, num_exc


def ax_hist(ax, arr, xlabel, density=False, logscale=False):
    """
    Intermediary
    ------------
    - stats_data (dict): simple statistics of beta; keys:
        - <stat>: stat (e.g., mean/median) among all 100 subjects
        - <stat>_mid: stat among the mid_range subjects (i.e., those s.t. 0 < beta < 1000)
        - 0 or 1000: number of subjects whose beta = 0 or 1000
    """
    kwargs = dict(density=density, edgecolor="black", linewidth=0.7)
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
        stats_data["(2.5,1000)"] = sum((arr > 2.5) & (arr < 1000))
        stats_data["(0,1e-4)"] = sum((arr > 0) & (arr < 1e-4))
        arr = arr[bool_mid]
        stats_data["mean_mid"] = np.mean(arr)
        stats_data["median_mid"] = np.median(arr)
        print427("beta stats", var=stats_data)
        # ticks_loc = np.logspace(10**(-4), 10, num=6)
        # ticks_label = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"]
        # ax.set_xticks(ticks_loc)
        if logscale:
            ax.set_xscale("log")  # set x to log scale
            xlim = [1e-4, 1e1]
            kwargs.update(dict(bins=np.geomspace(xlim[0],xlim[1],20)))
            # https://stackoverflow.com/questions/45905135/matplotlib-missing-minor-ticks-on-y-axis-because-of-log-range-10-decades
            locmaj = ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
            ax.xaxis.set_major_locator(locmaj)
            locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            xlim = [1e-4, 3.1]
            kwargs.update(dict(bins=np.linspace(xlim[0],xlim[1],20)))
        ax.set_xlim(xlim)
    ax.hist(arr, **kwargs)
    # ax.set_title(title, fontsize=FS_MAIN)
    ax.set_xlabel(xlabel, fontsize=FS_LAB)
    ax.tick_params(axis="both", which="major", **TICK_PARAMS)
    ax.tick_params(axis="both", which="minor", **TICK_PARAMS_MINOR)
    styles_txt = dict(fontsize=FS_LAB, horizontalalignment="center", transform=ax.transAxes)
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
        ax.set_ylabel("Probability Density", fontsize=FS_LAB)
    else:
        ax.set_ylabel("Frequency", fontsize=FS_LAB)


def ax_ECCS(ax, ECCS_arr, params, box_list=None, err_type='ste', CG_lv=0,
           show_legend=False, is_log=True, colors=None, dpi=None, lv=0, yrange=None, ECCS_type=2):
    """
    Args
    ----
    - ax: axis object
    - ECCS_arr: generated from make_ECCS_arr()
    - params (tuple): (regType, p, n)
        - regType (int):
            0: default Sierpiński graph
            x: Sierpiński-like graph of type x regularization

    Kwargs
    ------
    - box_list (len-2 list): bbdry and nested dict
    - ECCS_arr (2D nparr): ECCS_arr[i, [0,1,2]]: [beta, ECCS_1, ECCS_2]
    - err_type (str): type to use as errorbar: 'std' or 'ste'
    - show_legend (bool): whether we show simulation parameters
    - is_log (bool): if True then use log scale on x axis
    - yrange & ECCS_type: both are only used for ax.set_ylim()

    Intermediary
    ------------

    Return
    ------
    - xmaxs (list): list of beta that maximizes CCS at each level
        xmaxs[1]: beta that maximizes CCS at lv2/lv3
        xmaxs[i]: beta that maximizes CCS at lv(i+1)/lv(i+2)
    """
    # set up beta range (analytical) for plot
    x = ECCS_arr[:, 0]  # beta
    if is_log:
        x_range = (min(x) * 0.55, max(x) * 1.55)
    else:
        delta = (max(x) - min(x)) * 0.05
        x_range = (min(x) - delta, max(x) + delta)
    ax.set_xlim(x_range)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    (regType, p, n) = params
    n_level = n - 1  # for (3,3,3), max CCS level is 2
    CCS_type_slice = 0
    cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:n_level], N=n_level)
    xlabel = r"Memory Error Parameter $\beta$"
    ylabel = "ECCS"
    if regType in [0, 1, 2, 3]:
        title = fr"ECCS of $^{regType}S_{p:d}^{n:d}$ (lv${lv + 1}$/lv${lv + 2}$)"
    else:
        raise NotImplementedError(f"<regType>={regType} is invalid")

    if show_legend:  # only show walk length
        styles_txt = dict(fontsize=FS_LAB, horizontalalignment="center", transform=ax.transAxes)
        n_agents = 100  # num of subjects
        n_steps = 1500  # number of steps
        topy, s = 0.94, 0.06
        ax.text(0.80, topy - 1 * s * 0, f"walk length={n_steps:.0f}", **styles_txt)

    sty1 = dict(alpha= 0.74, linewidth= 2)

    def temp_a1(i):  # draw ECCS
        sty1.update(dict(color=cmap(i)))
        if show_legend:
            sty1.update(dict(label=f"lv{i+1}/lv{i+2}"))
        # ax.plot(x, CCS_arr[:, CCS_type_slice, i], **sty1)  # plot analytical curve
        # put scatter points of empirical results in
        sty2 = dict(linestyle="None", marker=".", s=11, color=cmap(i))
        sty2.update(dict(alpha = 0.74, linewidth = 2, zorder=2))
        # if show_legend:
        #     sty2.update(dict(label='Stochastic '+labels[i].replace('-','/lv')))
        kw_spt = dict(x=x, y=ECCS_arr[:, i + 1])
        ax.scatter(**kw_spt, **sty2)
        styles_txt = dict(fontsize=FS_LAB, horizontalalignment="right", transform=ax.transAxes)
        ax.text(0.97, 0.88, f"n={kw_spt['y'].shape[0]:d}", **styles_txt)
        if box_list is not None:
            bbdry, box_dict = box_list  # unpack
            bdata = [ECCS_arr[:, i + 1][box_dict[b]["mask"]] for b in box_dict if box_dict[b]["n"] != 0]
            positions = [box_dict[b]["center"] for b in box_dict if box_dict[b]["n"] != 0]
            if is_log:  # create a dummy axis and plot boxes on that axis
                ax_bx = ax.twiny()  # instantiate a separate x-axis for equal spacing boxes
                # below position kwarg are crucial steps, turn exp to linear (log scale)
                # log width is the same for all rows, thus only using 1st row
                widths = log_b(bbdry[0, 2], 10) - log_b(bbdry[0, 1], 10)  # full width
                kwargs = dict(widths=widths, manage_ticks=False, whis=(2.5, 97.5))  # 95% range whisker
                ax_bx.set_xlim(log_b(ax.get_xlim(), 10))  # s.t. box matches the scatter
                parts = ax_bx.boxplot(bdata, positions=log_b(positions, 10), showfliers=False, **kwargs)
                ax_bx.xaxis.set_visible(False)  # hide twin axis, and doesn't take up space
            else:
                parts = ax.boxplot(bdata, positions=positions)  # incomplete but unused
            
            for pc in parts.values():  # match box colors to scatter
                for line in pc:
                    line.set_color(cmap(i))
                    line.set_alpha(1)

            caps_lower = [parts['caps'][c]._xy[0, 1] for c in range(0, len(parts['caps']), 2)]
            caps_upper = [parts['caps'][c]._xy[0, 1] for c in range(1, len(parts['caps']), 2)]

            # show Wilcoxon signed-rank tests (both per bin and global)
            # global
            diff = ECCS_arr[:, i + 1] - 1  # compared to baseline of 1
            result = stats.wilcoxon(diff, alternative="greater")
            wstat, pval = result.statistic, result.pvalue
            print427(f"ECCS median - 1 compared to 0 (Fig.6; lv={i + 1}): ", var=f"n={len(diff)}")
            temp_txt = f"lv={i + 1}"
            temp_txt += f" | (Wilcoxon) stat={wstat:.2f} pval={pval:.4f}"
            print(temp_txt)
            ax.text(0.97, 0.94, f"Wilcoxon {pval_star(pval, star=False)}", **styles_txt)
            # pval2 = 1 - sum(diff > 0) / diff.shape[0]
            # ax.text(0.99, 0.91, f"Proportion {pval_star(pval2, star=False)}", **styles_txt)
            # ax.text(0.99, 0.91, f"Proportion p={pval2:.3f}", **styles_txt)
            # per bin
            styles_txt.pop("transform", None)
            styles_txt.update(dict(horizontalalignment="center"))
            # show p-val (stars) and sample size per bin
            for j, a in enumerate(bdata):  # loop over each bin (beta)
                diff = a - 1  # compared to baseline of 1
                pval = stats.wilcoxon(diff, alternative="greater").pvalue
                ax.text(positions[j], caps_upper[j] - 0.010, pval_star(pval, star=True), **styles_txt)
                # pval2 = 1 - sum(diff > 0) / diff.shape[0]
                # ax.text(temp_betas[i], 0.92, pval_star(pval2, star=True), **styles_txt)
                ax.text(positions[j], caps_lower[j] - 0.060, f"{len(diff):d}", **styles_txt)
            
            if yrange is not None:  # only match both lvs to a fix val if yrange is explicitly set
                if ECCS_type == 1:
                    ax.set_ylim(yrange)
                else:
                    ax.set_ylim((0.6, 1.6))
            else:  # if None, set it to (0.6, 1.6) anyway
                ax.set_ylim((0.6, 1.6))
    


    # argmax = beta that maximizes CCS at different permissible levels
    arrowprops = dict(arrowstyle="simple", facecolor="grey", edgecolor="grey")
    arrowprops.update(dict(linewidth=1 / 3, alpha=0.74))
    kw_text = dict(textcoords="axes fraction", fontsize=FS_LAB, arrowprops=arrowprops)
    kw_text.update(dict(ha="center", va="center"))

    xmaxs, ymaxs, texts = [None] * (n-1), [None] * (n-1), [None] * (n-1)
    temp_a1(lv)  # one level per plot (previously all levels per plot)
    # for i in range(n_level):  # n_level is always n-1; see earlier code for why this is true
    #     temp_a1(i)
        # temp_b1(i)  # put peak val on plot

    # ax.set_ylim((0.9, 1.3))  # for regType=3, p=3, n=3 max ECCS is <1.3
    ax.plot(ax.get_xlim(), (1, 1), "--", color="grey", zorder=0)  # draw y=1 line in grey

    ax.set_title(title, fontsize=FS_MAIN)
    ax.set_xlabel(xlabel, fontsize=FS_LAB)
    ax.tick_params(axis="both", which="major", **TICK_PARAMS)
    ax.tick_params(axis="both", which="minor", **TICK_PARAMS_MINOR)
    if lv == 0:  # only show y label on lv 1 ECCS
        ax.set_ylabel(ylabel, fontsize=FS_LAB)
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


def get_ECCS_from_data(cwd="", ECCS_type=1):
    """calculate ECCS for each subject
    process processed data (including rt_pred) and MEM results into dictionaries

    ECCS: for each subject
    There are 2 flavors of ECCS that are complementary to each other:
    1) ECCS1 - rt/linear part: we explicitly only use r0 and r1 | ECCS_type==1
    anticipation = (regressed_out rt - r0) / r1 (for given edgelv)
    ECCS at level l = mean(anticipation at edgelv=l) - mean(anticipation at edgelv=l+1)
    NOTE however, mean() is two-fold:
        first mean across all instances of transitions on given edge
        then mean across edges at given edgelv (using results from above step)

    2) ECCS2 - beta part: we explicitly only use beta (fitting process) | ECCS_type==2
    anticipation from beta fitting process:
    one of the last steps is to calculate "belief" using mental counts
    in addition to outputting normalized belief, we can output raw mental counts (numerator) too
    ECCS2 calculation is very similar to how CCS is calculated from simulation results
    to get mental count matrix, we use last belief count for each edge

    Return
    ------
    - MEM (dict): k:v -> "id":fit_dict
        fit_dict (dict): keys: "beta", "r0", "r1"; "ECCS" is arr of ECCS
    """
    temp = os.listdir(cwd)
    fname = "MEM.json"  # TODO add all mental count columns to json as well
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

        # below add new entries to MEM.json
        df = pd.read_csv(cwd + "data_sier.csv")
        # ECCS calculation happens here
        if ECCS_type == 1:
            def ant(x):
                return (x.loc["rt"] - x.loc["rt_pred"] - MEM[x.id]["r0"]) / MEM[x.id]["r1"]
            df["ant"] = df.apply(ant, axis=1)
            pd_series = df.groupby(by="id").apply(ECCS1)
            for id, val in pd_series.iteritems():
                MEM[id]["ECCS"] = val[0]
                MEM[id]["ants_arr"] = val[1]
                MEM[id]["rts_arr"] = val[2]
                MEM[id]["rts_edgelv"] = val[3]
                MEM[id]["rts_pred_arr"] = val[4]
                MEM[id]["rts_pred_edgelv"] = val[5]
                MEM[id]["edges_arr"] = val[6]
        elif ECCS_type == 2:
            df_MC = pd.read_csv(cwd + "MEM_results.csv")  # need those 1000 columns for mental counts
            df_MC = df_MC.set_index("id")
            pd_series = df.groupby(by="id").apply(ECCS2, df_MC=df_MC)
            for id, val in pd_series.iteritems():
                MEM[id]["ECCS2"] = val[0]
                MEM[id]["ants2_arr"] = val[1]
                MEM[id]["edges_arr"] = val[2]

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
    - nback_idx (int): estimate of nback beta; 0 -> direct fit; 1 -> MLE; 2(3) -> gof for 0(1)
    - gof_thres (float): in analysis, only use those whose gof >= gof_thres; r-squared adjusted

    Intermediary
    ------------
    beta_sier/beta_nback (dict): k:v -> id:beta (NOTE we convert id str -> int)
    beta_arr (2D nparr): row -> index for id; col -> beta for each source
    beta_arr_filt (2D nparr): keep non-extreme & good gof
        1) 0 < beta < 1000 on both axes
        1*) 0 <= beta <= 6 on both axes
        1**) 0 < beta < 1000 on network beta only
        2) 0 <= beta <= 1000 on n-back beta only (there are -1 and inf)
        3) gof >= 0.8
    """
    show_full = False  # show all network vs. n-back or not
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
        temp_keys = ["beta_nback", "beta_nback_MLE", "gof_nback", "gof_nback_MLE"]
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
        cond_gof = True if gof_thres is None else (beta_nback[k][2 + nback_idx] >= gof_thres)
        # if 0 <= beta_sier[k] <= 6 and 0 <= beta_nback[k][nback_idx] <= 6 and cond_gof:
        # if 0 < beta_sier[k] < 1e3 and 0 < beta_nback[k][nback_idx] < 1e3 and cond_gof:
        if 0 <= beta_sier[k] <= 1e3 and 0 <= beta_nback[k][nback_idx] <= 1e3 and cond_gof:
            key_set.add(k)
    # print(f"DEBUG beta ids that remain: {key_set}")
    beta_arr_filt = np.zeros((len(key_set), 2), dtype=float)
    for i, k in enumerate(key_set):
        beta_arr_filt[i, 0] = beta_sier[k]
        beta_arr_filt[i, 1] = beta_nback[k][nback_idx]

    # visualization & analyses
    txt_beta_arr = "Direct-Fit" if nback_idx == 0 else "MLE"
    spearmanr = stats.spearmanr(beta_arr_filt, axis=0)
    ander_sier = stats.anderson(beta_arr_filt[:, 0], dist="norm")
    ander_nback = stats.anderson(beta_arr_filt[:, 1], dist="norm")
    text_gof1 = "all" if gof_thres is None else f"{gof_thres:.2f}"
    print(f"\n nback {txt_beta_arr} β; gof thresholded at {text_gof1} | n={beta_arr_filt.shape[0]}")
    print(f"network β:{ander_sier.statistic:.3g}\n", ander_sier.critical_values, ander_sier.significance_level)
    print(f"nback β:{ander_nback.statistic:.3g}\n", ander_nback.critical_values, ander_nback.significance_level, end="\n")

    fig = plt.figure(figsize=[14, 6])  # initialize
    axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    temp_median1 = np.median(beta_arr[:, 0])
    temp_median2 = np.median(beta_arr[:, 1])
    temp_mean1 = np.mean(beta_arr_filt[:, 0])
    temp_mean2 = np.mean(beta_arr_filt[:, 1])
    temp_median3 = np.median(beta_arr_filt[:, 0])
    temp_median4 = np.median(beta_arr_filt[:, 1])

    ax1, ax2 = axes  # unpack
    styles_txt = dict(fontsize=FS_LAB, horizontalalignment="center", transform=ax1.transAxes)
    if show_full:
        ax1.scatter(beta_arr[:, 1], beta_arr[:, 0])
        ax1.set_title(r"$\beta$: network vs. n-back", fontsize=FS_MAIN)
        ax1.text(0.8, 0.90, f"$n={beta_arr.shape[0]:d}$", **styles_txt)
        ax1.text(0.6, 0.85, f"median={temp_median1:.3f} (network) | {temp_median2:.3f} (n-back)", **styles_txt)
    else:
        beta_diff = beta_arr_filt[:, 1] - beta_arr_filt[:, 0]
        ax1.scatter(range(beta_arr_filt.shape[0]), beta_diff)
        ax1.plot(ax1.get_xlim(), [0] * 2, "--", zorder=0, color="grey")  # draw line
        text_gof2 = "" if gof_thres is None else fr"; gof$\geq${gof_thres:.2f}"
        ax1.set_title(fr"$\beta$: n-back - network ({txt_beta_arr}{text_gof2})", fontsize=FS_MAIN)
        # ax1.text(0.8, 0.95, f"$r_s={spearmanr.correlation:.3f}$, $p={spearmanr.pvalue:.3f}$", **styles_txt)
        ax1.text(0.8, 0.90, f"$n={beta_arr_filt.shape[0]:d}$", **styles_txt)
        ax1.text(0.8, 0.85, f"mean={np.mean(beta_diff):.3f} (diff)", **styles_txt)
        ax1.text(0.8, 0.80, f"median={np.median(beta_diff):.3f} (diff)", **styles_txt)

    ax2.scatter(beta_arr_filt[:, 1], beta_arr_filt[:, 0])
    ax2.set_title(fr"$\beta$: network vs. n-back ({txt_beta_arr}{text_gof2})", fontsize=FS_MAIN)
    styles_txt.update(dict(transform=ax2.transAxes))
    ax2.text(0.8, 0.95, f"$r_s={spearmanr.correlation:.3f}$, $p={spearmanr.pvalue:.3f}$", **styles_txt)
    ax2.text(0.8, 0.90, f"$n={beta_arr_filt.shape[0]:d}$", **styles_txt)
    ax2.text(0.6, 0.85, f"mean={temp_mean1:.3f} (network) | {temp_mean2:.3f} (n-back)", **styles_txt)
    ax2.text(0.6, 0.80, f"median={temp_median3:.3f} (network) | {temp_median4:.3f} (n-back)", **styles_txt)

    for ax in axes:
        ax.set_xlabel(r"n-back $\beta$", fontsize=FS_LAB)
        ax.set_ylabel(r"network $\beta$", fontsize=FS_LAB)
    if not show_full:
        axes[0].set_xlabel("index", fontsize=FS_LAB)
        axes[0].set_ylabel(r"difference in $\beta$", fontsize=FS_LAB)
    saveNclose427(fig, f"beta_arr_{txt_beta_arr}_gofgeq-{text_gof1}", dpi=300, sub_folder_name="100subs\\betas")


def ECCS1(df):
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


def ECCS2(df, df_MC, alt=None):
    """df should come from groupby(level="id")
    df should have fields: "node", "node_prev", "edgelv"
    df_MC should have fields: "M_1", ..., "M_1000"
    even though this requires two inputs, it actually is simpler than ECCS1,
    because heavy-lifting was done in the max entropy model param fitting process

    Kwargs
    ------
    - alt (bool): alternative ECCS definition
        only p=3, n=3, regularization shouldn't matter but self-loop is one we do on
        unlike all CCS/ECCS definitions we have thus far,
        this alt version removes some edges (as opposed to using all or almost all)
        it removes immediate neighboring edges for both x-1,x and x,x+1 CCS
        lv1 edges: for s in {0,1,2}, keep only edges w/ sij nodes where i=j
        lv2 edges: similar to lv1, but alternatively,
        we could remove edges w/ nodes whose digits are all unique (i.e., permutation of 0,1,2)
        commom pattern: remove edges whose both nodes do not have repeating digits w/in level
        the sub-str we consider the pattern in has length lv+1;
        for lv1 edges, we look at 2 digits from the right
        for lv2 edges, we look at 3 digits from the right
        e.g., 201-210, 202-220, 220-221, 201-202 (first two are lv2, second two are lv1)
        remove 201-210: both nodes contain non-repeating digits
        keep 202-220: both repeat "2"
        remove 220-221: "20" and "21" contain non-repeating digits
        remove 201-202: "01" and "02" contain non-repeating digits

    Intermediary
    ------------
    - ants (dict): k:v -> edge:[n, x]; x is last mental count; n is corresponding trial number
        edge is a frozenset of two elements (or one, if self-loop): source and target nodes
        we search from last trial backwards to first trial
        if we find the edge, we update it once and will not touch it again until normalization step
        becomes k:v -> edge:x_prob after row-normalization
    - edgelvs (dict): k:v -> edgelv (int):set of edge (set of frozenset)

    after completing ants (iterating over all rows of df)
    we find mean for each edge; then mean for each edgelv; then ECCS

    Return
    ------
    eccs (list): eccs[i]: level i+1 ECCS
    ants_arr (list): last mental count for each edge (sorted by level, ascending order)
    edges_arr (list): list of len-2 lists; NOTE the list is ordered (frozenset -> list)
    """
    if alt is None:
        alt = ALT  # global param, see top of the script
    ants = dict()
    edgelvs = dict()
    n = len(df)
    id = df["id"].iloc[0]
    if n > 1000:
        msg = f"oof, in data_sier.csv, id={id} has {n} trials, "
        msg += "> available mental counts in MEM_results.csv"
        raise Exception(msg)
    for i in range(-1, -1 - n, -1):  # a bit inefficient since it loops all trials in df
        row = df.iloc[i]
        edge = frozenset([int(row.node), int(row.node_prev)])  # np.int64 -> int; work w/ JSON
        trial_MC = row.trial - 500  # col for df_MC
        temp_MC = df_MC[f"M_{trial_MC}"].loc[id]  # mental count
        if pd.isnull(temp_MC):  # if any entry is empty, smth is wrong with df_MC, raise!
            raise Exception(f"oof, in MEM_results.csv, id={id} \"M_{trial_MC}\" column is empty")
        if edge not in ants:  # add new entries to ants and edgelvs; ONLY once per edge
            ants[edge] = [row.trial, temp_MC]
            if row.edgelv not in edgelvs:  # add only if never seen this edgelv
                edgelvs[row.edgelv] = set([edge])
            else:  # add new edge for this edgelv
                edgelvs[row.edgelv].add(edge)

    # row-normalized mental counts
    def temp_set2list(e):  # turn frozenset e into an edge (list)
        temp = list(e)
        if len(temp) == 1:  # self-loop; frozenset only has 1 element
            temp = [temp[0], temp[0]]
        return temp
    N = max([max(e) for e in ants]) + 1  # num of nodes
    W = np.zeros((N, N))
    for e in ants:
        temp = temp_set2list(e)
        W[temp[0],temp[1]] = ants[e][1]
        W[temp[1],temp[0]] = ants[e][1]
    W = W_norm(W)

    edgelvs = {k:list(v) for k,v in edgelvs.items()}  # s.t. edge order is fixed
    edgelvs_l = {k:[temp_set2list(x) for x in v] for k,v in edgelvs.items()}  # s.t. edge is subscriptable

    # find ECCS2
    for edge in ants:  # keep only mental counts
        ants[edge] = ants[edge][1]
    lv_arr = sorted([x for x in edgelvs if x >= 1])  # 1,2,...,lv_max
    edges_arr = [list(e) for lv in lv_arr for e in edgelvs[lv]]  # s.t. edge is subscriptable
    mean_weights = [0.0 for _ in range(lv_arr[-1])]
    if not alt:
        for lv in lv_arr:
            # mean_weights[lv - 1] = np.mean([ants[e] for e in edgelvs[lv]])  # unnormalized
            mean_weights[lv - 1] = np.mean([W[e[0],e[1]] for e in edgelvs_l[lv]])
    else:
        def temp_r(node, lv):  # check if node has repeating digits
            digits = p_ary(node, 3)[::-1][:(lv+1)]  # assume p=3; invert it and sub-str it
            return len(set(digits)) != len(digits)
        for lv in lv_arr:
            if lv in [1, 2]:  # currently only works for n=3, p=3, so only 1, 2 are used
                _edges = [
                    W[e[0], e[1]] for e in edgelvs_l[lv] if (temp_r(e[0], lv) or temp_r(e[1], lv))
                ]
            else:
                _edges = [W[e[0],e[1]] for e in edgelvs_l[lv]]
            mean_weights[lv - 1] = np.mean(_edges)

    eccs = list(np.divide(mean_weights[:-1], mean_weights[1:]))  # calculate ECCS2
    # turn ants into a sorted list
    ants_arr = [ants[e] for lv in lv_arr for e in edgelvs[lv]]

    return eccs, ants_arr, edges_arr


if __name__ == "__main__":
    ECCS_main()
