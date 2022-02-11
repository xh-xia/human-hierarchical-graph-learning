"""
This is to debug CCS_num.py and compare it with simulation (AKA sim)
Created: Monday, ‎January ‎24, ‎2022, ‏‎5:54:19 PM (EST)
@author: Xiaohuan (Pixel) X.
"""

from utility427.helper427 import set_dir427, mkdir_p, get_params, partial_427_decorator
from utility427.math427 import log_b, findNearest, rank_eigvals, W_norm, np
from utility427.plt427 import plt, Normalize, LinearSegmentedColormap, GridSpec, Line2D  # mpl
from utility427.plt427 import (
    saveNclose427,
    colors_selector,
    cbrLabel427,
    get_violin_pw,
)  # mpl helpers
from utility427.plt427 import load_CCS_stat, save_masks
from utility427.Sierpinski427 import make_Sierpinski427, p_ary, make_SierpinskiGraph427
from utility427.sim_params427 import make_sim_params, make_beta_boundry
from utility427.CCS_num import CCS_ep


def main():
    params = (3, 4, 3)
    (regType, p, n) = params
    GTDict = make_SierpinskiGraph427(p, n, norm=True, regType=regType)
    b = 5
    sub_fo_name = "reg_n_p_unbinned_beta_100perbeta"
    key_class = sub_fo_name
    beta_class = "constant_unbinned"
    n_agents = 100
    steps_arr = [1500, 3000, 4500, 6000, 7500]

    npy_sub_path = f"{sub_fo_name}\\CCS_stat_mean_{key_class}_{beta_class}_{n_agents}"
    CCS_stat = load_CCS_stat(sim_path="sim427\\output", fname=npy_sub_path)
    beta_arr = CCS_stat["mean"][params][0, 0, :]  # beta
    kwargs = dict(noise=CCS_stat, steps_arr=steps_arr[:3], beta_arr=beta_arr, b=b)
    kw = dict(mp=False)
    kw["T"] = kwargs["steps_arr"]
    A_hat_list2 = [make_A_hat_beta(GTDict["A"], beta_arr[b])]
    CCS_stat_ep = CCS_ep(GTDict, A_hat_list2, [beta_arr[b]], **kw)

    kwargs.update(dict(noise_ep=CCS_stat_ep))
    kwargs["colors"] = colors_selector(str="5-class Greens")

    print(f"DEBUG beta={beta_arr[b]}")
    plot_simvsep(params=params, **kwargs)


def plot_simvsep(params, colors, noise, noise_ep, steps_arr, beta_arr, b):
    (regType, p, n) = params
    n_level = n - 1
    cmap = LinearSegmentedColormap.from_list("custom edge color", colors[:n_level], N=n_level)
    for spl in range(len(steps_arr)):
        for i in range(n_level):
            fig, ax = plt.subplots(1, 1)
            title = fr"$^{regType}S_{p:d}^{n:d}$ CCS lv{i + 1}"
            title += fr" walk={steps_arr[spl]} β≈{beta_arr[b]:.3f}"
            y = noise["mean"][params][spl, 3 + i, b]
            yerr = noise["std"][params][spl, 3 + i, b]
            ax_sim(ax, y, yerr, cmap(i), title, noise["raw"][params][spl, i, b, :])
            x0 = noise_ep["ps"][spl, i, 0, :]
            rs = noise_ep["rs"][spl, i, 0, :]
            ax_ep(ax, noise_ep["mean"][spl, i, 0], noise_ep["std"][spl, i, 0], cmap(i), x0, rs)
            n_agents = round(noise["mean"][params][spl, 1, 0])
            n_steps = round(noise["mean"][params][spl, 2, 0])
            ax.set_ylim([0.8, 1.6])
            ax.set_xlim([0.7, 2.3])
            ax.plot(ax.get_xlim(), (1, 1), "--", color="grey", zorder=0)  # draw y=1 line in grey
            fname = "singleton"
            fname += f"_{n_agents}_{n_steps}_CCSlv={i+1}"
            saveNclose427(fig, fname, dpi=300, sub_folder_name="CCS_ep_DEBUG")


def ax_ep(ax, y, yerr, cmap, x0, rs):
    # scatter
    sty = dict(linestyle="None", capsize=4.0, marker=".", markersize=11)
    sty.update(dict(alpha=0.74, linewidth=2, zorder=2))
    sty.update(dict(markeredgecolor=cmap, markerfacecolor=cmap, ecolor=cmap))
    kw_erb = dict(x=[2], y=[y], yerr=[yerr])
    ax.errorbar(**kw_erb, **sty)  # python 3.5+ PEP 448 (Unpacking Generalizations)
    # violin
    kw_temp = dict(color=cmap, alpha=0.27, linewidth=0)
    # x0 = x0 / simpson(x0, rs)  # norm by area (should not use min-max for pdf)
    x0 /= 40  # TODO: need fixing
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
    kw_temp["y"] = rs[idx0:idx1]
    x0 = x0[idx0:idx1]
    kw_temp["x1"] = 1.5 + (1 - x0) * (2 - 1.5)
    kw_temp["x2"] = 2 + x0 * (2.5 - 2)
    ax.fill_betweenx(**kw_temp)


def ax_sim(ax, y, yerr, cmap, title, y0):
    # scatter
    sty = dict(linestyle="None", capsize=4.0, marker=".", markersize=11)
    sty.update(dict(alpha=0.74, linewidth=2, zorder=2))
    sty.update(dict(markeredgecolor=cmap, markerfacecolor=cmap, ecolor=cmap))
    kw_erb = dict(x=[1], y=[y], yerr=[yerr])
    ax.errorbar(**kw_erb, **sty)  # python 3.5+ PEP 448 (Unpacking Generalizations)
    # violin
    data_vl = [y0]
    pos_vl = [1]  # x-axis
    # pos_vl, widths = get_violin_pw(pos_vl, x_scale=None)
    kw_vl = dict(showmeans=False, showmedians=False, showextrema=False, widths=[0.5])
    parts = ax.violinplot(data_vl, positions=pos_vl, **kw_vl)
    for pc in parts["bodies"]:
        pc.set_facecolor(cmap)
        pc.set_alpha(0.27)
    ax.set_title(title, fontsize=17)
    ax.set_xlabel("type: sim/ep", fontsize=11)
    ax.set_ylabel("CCS", fontsize=11)


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


if __name__ == "__main__":
    main()
