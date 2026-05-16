"""
matplotlib helper functions

Created: Wednesday, June 9, 2021, 8:51:14 AM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import matplotlib.pyplot as plt

# custom colorbar (https://matplotlib.org/tutorials/colors/colormap-manipulation.html)
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec  # for subplots placement manipulation
from matplotlib.lines import Line2D  # for median legend in violin plots
import matplotlib.ticker as ticker  # for xticks/yticks

import sys, os, inspect
temp_cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, temp_cwd)
from helper427 import set_dir427, mkdir_p
from math427 import log_b, np
sys.path.pop(0)  # remove script dir from sys.path

"""
    ########################################
        matplotlib
    ########################################
"""


def saveNclose427(fig, fname, dpi=None, sub_folder_name="", makedir=True, transparent=False):
    set_dir427(depth=2)  # set cwd to CCS427.py dir
    if sub_folder_name:  # empty string is falsy
        sub_folder_name = f"\\{sub_folder_name}"

    # if dpi not provided, save both 300dpi and lossless; pdf or svg for lossless quality
    if dpi is None:
        dpi = 300
        dir_lossless = f"output\\pdf_lossless{sub_folder_name}"
        if makedir:
            mkdir_p(dir_lossless)
        # bbox_inches="tight": auto resize fig to fit titles and such
        fig.savefig(dir_lossless + f"\\{fname}.pdf", bbox_inches="tight", transparent=transparent)
    # if dpi is provided, only save lossy version
    dir_dpi = f"output\\png_dpi{dpi}{sub_folder_name}"
    if makedir:
        mkdir_p(dir_dpi)
    fig.savefig(dir_dpi + f"\\{fname}.png", bbox_inches="tight", dpi=dpi, transparent=transparent)

    fig.clf()  # clear figure
    plt.close(fig=fig)  # close figure


def colors_selector(str=None, reverse=True):
    """
    website of reference: https://colorbrewer2.org/
    I picked the color from top to bottom, so typically that's light to dark
    if reverse is True, I will reverse said order, going from bottom to top.
    """
    if str is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # default color in pyplot
    else:
        if str == "7-class Greys":
            colors = ["#f7f7f7", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525"]
        elif str == "7-class Purples":
            colors = ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"]
        elif str == "7-class Oranges":
            colors = ["#feedde", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#8c2d04"]
        elif str == "7-class Greens":
            colors = ["#edf8e9", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"]
        elif str == "7-class Blues":
            colors = ["#eff3ff", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
        elif str == "7-class Reds":
            colors = ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"]
        elif str == "5-class Greys":
            colors = ["#f7f7f7", "#cccccc", "#969696", "#636363", "#252525"]
        elif str == "5-class Purples":
            colors = ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"]
        elif str == "5-class Oranges":
            colors = ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"]
        elif str == "5-class Greens":
            colors = ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"]
        elif str == "5-class Blues":
            colors = ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"]
        elif str == "5-class Reds":
            colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
        elif str == "":
            colors = ["#", "#", "#", "#", "#", "#", "#"]
    return colors[::-1] if reverse else colors


def cbrLabel427(cax, title, fontsize=14):
    """
    set colorbar label to the left, vertically
    cbm.set_label() is the "vanilla"
    assume fig.colorbar(ticklocation='right') which should be default
    default fontsize plt.rcParams['font.size']=10.0
    my convention of fontsize: title 17, other titles 11.
    Args
    ----
    - cax: axis onto which colorbar is drawn
    """
    kwargs = {
        "transform": cax.transAxes,
        "verticalalignment": "center",
        "horizontalalignment": "center",
        "fontsize": fontsize,
        "rotation": "vertical",
    }
    cax.text(-0.9, 0.5, title, **kwargs)


def get_violin_pw(x, x_scale=None):
    """
    input 1D arr of x-axis values
    output positions and widths vector for ax.violinplot(positions, widths)
    such that even if the final plot is in log scale (assuming in that case x_scale is not None),
    the widths of the violin is still symmetric,
    and to achieve this we would output linear scale violin parameters,
    and later apply it on twin axis (e.g., twiny, a separate x-axis)

    Arg
    ---
    - x (arr-like): 1D arr
    - x_scale: the scale of arr x
        - None: linear (x[i+1] - x[i] = const)
        - base (int): exponential with base=`base` (`x[i+1]` / `x[i]` = `base`^const)
            meaning we assume x was obtained by `base`^x0
            where x0 is original linear scale arr with x0[i+1] - x0[i] = const

    Note
    ----
    from https://matplotlib.org/stable/_modules/matplotlib/axes/_axes.html#Axes.violinplot:
    # Calculate ranges for statistics lines
        pmins = -0.25 * np.array(widths) + positions
        pmaxes = 0.25 * np.array(widths) + positions
        where positions, pmins, and pmaxes are x-axis values
    """
    n = len(x)
    if n < 3:
        raise NotImplementedError("len(<x>) has to be >= 3")

    def temp_is_linear(arr):  # check if arr is linear
        return np.allclose(arr[:-1], arr[1:], rtol=1.e-07, atol=1.e-10)

    if x_scale is None:
        w = x[1] - x[0]  # since linear, spacing is the same
        return x, [w] * n
    else:
        x0 = log_b(x, x_scale)  # de-exponentiate it by logarithmizing it
        w = x0[1] - x0[0]  # since x0 is linear, spacing is the same
        return x0, [w] * n


"""
    ########################################
        CCS loading
    about stacks: e.g., for set_dir427() called here (AKA current script):
    depth=0: 1st call of set_dir427() in helper427
    depth=1: 2nd call of set_dir427() in plt427 (current script)
    depth=2: 3rd call of whatever function containing set_dir427() in whatever script importing plt427
    therefore for I/O in whatever script importing plt427.py, we need depth=2 here.
    ########################################
"""


def load_CCS_stat(sim_path=None, fname="CCS_stat"):

    if sim_path is not None:
        try:
            return np.load(f"{sim_path}\\{fname}.npy", allow_pickle=True).tolist()
        except OSError:  # couldn't find the file
            raise OSError(f'make sure {sim_path}\\{fname} exists')
        except Exception:
            raise
    else:
        set_dir427(depth=2)  # set cwd to CCS427.py dir
        fname = f"input\\{fname}.npy"
        try:
            return np.load(fname, allow_pickle=True).tolist()
        except OSError:  # couldn't find the file
            raise OSError('make sure .npy (stochastic sim results) is in "input" folder')
        except Exception:
            raise


def make_level_masks(GTDict):
    """
    Generate mask of adjacency matrix such that all edges belong to certain level.
    """
    masks = dict()
    # initialize mask for all levels (including -1, which is undefined lv)
    for l in set(GTDict["lvList"]):
        masks[f"lv{l}"] = np.zeros_like(GTDict["A"], dtype=int)
    for k, (i, j) in enumerate(GTDict["edgeList"]):
        masks[f"lv{GTDict['lvList'][k]}"][i, j] = 1
        masks[f"lv{GTDict['lvList'][k]}"][j, i] = 1  # undirected graph
    return masks


def save_masks(GTDict, regType, p, n):
    set_dir427(depth=2)  # set cwd to CCS427.py dir
    fname = "output\\npy_files\\"
    mkdir_p(fname)
    fname += f"Sierpinski(regType={regType},p={p},n={n})"
    Sierpinski_dict = dict()
    Sierpinski_dict["A"] = GTDict["A"]
    Sierpinski_dict["masks"] = make_level_masks(GTDict)
    np.save(fname, Sierpinski_dict)
