"""
This is for CCS numerical approximation using Andrei's Exposure Theory

it's very slow (5~10 sec per CCS point depending on the resolution)

Created: Monday, ‎January ‎10, ‎2022, ‏‎3:41:03 PM (EST)
@author: Xiaohuan (Pixel) Xia
"""

import numpy as np
from scipy.stats import poisson
from scipy.integrate import simpson as simpson
from concurrent.futures import ProcessPoolExecutor  # for multi-processing

import sys, os, inspect
temp_cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, temp_cwd)
from helper427 import partial_427_decorator
from Sierpinski427 import make_SierpinskiGraph427
from plt427 import make_level_masks
from math427 import W_norm
sys.path.pop(0)  # remove script dir from sys.path

NUM_CCS = 401  # update it according to CCS_CDF()


def main_DEBUG():
    beta_arr = np.array([0.061, 0.276, 1000])  # for analytical curve only
    GTDict = make_SierpinskiGraph427(3, 3, norm=True, regType=3)
    A_hat_list = [make_A_hat_beta(GTDict["A"], beta) for beta in beta_arr]
    data = CCS_ep(GTDict, A_hat_list, beta_arr, T=[1500, 3000], mp=False)
    print(data["ps"][0, 0, 0, :][85:105])
    print(data["ps"][1, 0, 0, :][85:105])


def CCS_ep(GTDict, A_hat_list, beta_arr, T=None, mp=False):
    knames = ["mean", "std", "rs", "ps"]
    if T is None:
        T = [1500]
    data = dict(beta=np.array(beta_arr))
    n = GTDict["n"]
    for k in knames[:2]:
        data[k] = np.zeros((len(T), n - 1, len(A_hat_list)), dtype=float)
    for k in knames[2:]:
        data[k] = np.zeros((len(T), n - 1, len(A_hat_list), NUM_CCS), dtype=float)
    masks = make_level_masks(GTDict=GTDict)
    # self-loop excluded; only one direction used
    masks = [np.triu(masks[f"lv{l}"], k=1) for l in range(1, n + 1)]

    kwargs = dict(masks=masks, n=n, T=T, params=None, get_pdf=True, res=3)
    kwargs["Ws"] = A_hat_list
    if mp:  # didn't work, as usual
        with ProcessPoolExecutor() as executor:
            dd_list = executor.map(temp_W(**kwargs), range(len(A_hat_list)))
    else:
        dd_list = map(temp_W(**kwargs), range(len(A_hat_list)))
    DD = dict()
    for k in knames:
        DD[k] = dict()
    for dd in dd_list:  # aggregate all keys of dd in dd_list into DD
        for k in DD:
            DD[k].update(dd[k])
    for k in DD["mean"]:  # beta idx
        for t in range(len(T)):
            for key in knames[:2]:
                data[key][t, :, k] = DD[key][k][t, :]
            for key in knames[2:]:
                data[key][t, :, k, :] = DD[key][k][t, :, :]

    return data


@partial_427_decorator
def temp_W(i, Ws, masks, n, T, params, get_pdf, res):
    """
    integrating using samples:
    simpson: https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
    or np.trapz

    Intermediary
    ------------
    idx explanation:
    [i][t, j]: i-th beta, T[t] RW length, j+1 CCS level
    for k in ["rs", "ps"]: .shape[-1] = NUM_CCS because of default params in CCS_CDF()
    """
    kw = dict(params=params, get_pdf=get_pdf, res=res)
    dd = dict()
    knames = ["mean", "std", "rs", "ps"]
    for k in knames:
        dd[k] = dict()
    for k in dd:
        if k in ["rs", "ps"]:
            dd[k][i] = np.zeros((len(T), n - 1, NUM_CCS))  # default params in CCS_CDF()
        else:
            dd[k][i] = np.zeros((len(T), n - 1))
    for t in range(len(T)):  # walk length loop
        rs, ps = CCS_CDF(*get_CCS_CDF_params(Ws[i], masks, n, T=T[t]), **kw)
        for j in range(n - 1):  # CCS level loop
            dd["rs"][i][t, j, :] = rs[j]
            dd["ps"][i][t, j, :] = ps[j]
            dd["mean"][i][t, j] = simpson(rs[j] * ps[j], rs[j])
            temp = simpson((rs[j] ** 2) * ps[j], rs[j]) - (dd["mean"][i][t, j]) ** 2
            dd["std"][i][t, j] = np.sqrt(temp)
    return dd


def get_CCS_CDF_params(W, masks, n, T=1500):
    """
    Kwargs
    ------
    - T (int): random walk size
    """
    n_arr = [0] * n
    la_arr = [0.0] * n
    for i in range(n):  # edge level loop
        n_arr[i] = np.sum(masks[i])  # number of edges at level l
        la_arr[i] = T * np.sum(W * masks[i])  # total exposure

    return n_arr, la_arr


def CCS_CDF(n_arr, la_arr, params=None, get_pdf=True, res=3):
    """it numerically approximates F(r) = int_0^r pr(x)dx where pr(x) is CCS_PDF

    Args
    ----
    - n_arr (list-like): n_arr[l-1]: number of edges at level l
    - la_arr (list-like): same shape as n_arr; la_arr[l-1]: total exposure at level l

    Kwargs
    ------
    - get_pdf (bool): whether we return pdf using np.gradient
    - res (float or int): 2nd factor of xym

    Intermediary
    ------------
    - dr (float): CCS value resolution
    - rm (list of float): max CCS considered; CCS values [0,0+dr,...,~(rm-dr), ~rm]
    - xym (list of int): xym[c]: middle value of x or y for edge level c+1
        a grid will be made centering at the middle values
        the size of the grid is 2*round(res*sqrt(la_arr)) + 1
        1st factor of 2 is the half width of the grid box
        2nd factor of res is arbitrary choice of coverage (in standard dev.) of poisson distribution
        +1 because we have the middle point as well

    at given CCS level:
    - xm (int): max value of x in ρ; for level 1,...,n-1
    - ym (int): max value of y in ρ; for level 2,...,n
    - ρ (2D nparr): xy_grid

    Return
    ------
    - rs (list of 1D nparr): rs[c]: arguments (r) for CDF of CCS at level c+1
    - Fs (list of 1D nparr): Fs[c]: values (F(r)) for CDF of CCS at level c+1
    - ps (list of 1D nparr): ps[c]: values (f(r)) for PDF of CCS at level c+1

    """
    n = len(n_arr)  # number of edge levels
    if params is None:
        params = dict(dr=0.01, xym=[round(x) for x in la_arr])
    params.update(dict(rm=[4 + params["dr"]] * (n - 1)))
    dr, rm, xym = params["dr"], params["rm"], params["xym"]
    rs = [np.arange(start=0, stop=rm[i], step=dr, dtype=float) for i in range(n - 1)]  # len=NUM_CCS
    Fs = [np.zeros_like(rs[i], dtype=float) for i in range(n - 1)]
    # pre-calculate xy components (both idx (i.e., arg for poisson) and val) of ρ
    width = np.around(res * np.sqrt(la_arr), decimals=0).astype(
        np.int32, casting="unsafe", copy=False
    )
    ρxyidx = [np.arange(lclamp(xym[i] - width[i]), xym[i] + width[i] + 1, 1) for i in range(n)]
    ρxy = [np.array([poisson.pmf(ρxyidx[i], la_arr[i])]) for i in range(n)]
    # ρxy = [np.array([[1,1,1,1]]) for i in range(n)]  # DEBUG
    coef = np.divide(n_arr[:-1], n_arr[1:])  # n_l : n_l+1
    for c in range(n - 1):  # CCS level loop; CCS level = c + 1
        ρ = ρxy[c].T @ ρxy[c + 1]  # calculate xy_grid from pre-calculated xy components
        xn = ρxyidx[c].shape[0]  # num of points on x
        yn = ρxyidx[c + 1].shape[0]  # num of points on y
        temp = 0
        s = [None] * xn  # value should be y idx, not y val in the grid
        t = [None] * xn  # value should be y idx, not y val in the grid
        for k, r in enumerate(rs[c]):
            rhat = r * coef[c]
            for i in range(xn):  # iterate rows (x; edge level c)
                if ρxyidx[c][i] > rhat * ρxyidx[c + 1][-1]:
                    break  # last point in row i does not cross the line
                else:
                    if t[i] is None:  # first time last point crosses the line
                        t[i] = -1  # t[i] will never be None from now on
                        # print(f"DEBUG i={i} | k={k} | r={rhat}")
                for j in range(t[i], -yn, -1):
                    if ρxyidx[c][i] <= rhat * ρxyidx[c + 1][j - 1]:
                        t[i] = j - 1
                    else:
                        break
                temp += np.sum(ρ[i, t[i] : s[i]])  # row i
                s[i] = t[i]
            Fs[c][k] = temp

    if get_pdf:
        ps = [np.gradient(y, x) for (y, x) in zip(Fs, rs)]
        return rs, ps
    else:
        return rs, Fs


def lclamp(val, minval=0):  # left clamp; general clamp can be max(minval, min(maxval, val))
    if val < minval:
        return minval
    else:
        return val


def make_A_hat_beta(A, beta):
    n = np.shape(A)[0]  # # of rows, but assuming symmetric, thus also cols (=nodes)
    A_ = W_norm(A)
    return (1 - np.exp(-beta)) * A_ @ np.linalg.inv(np.eye(n) - np.exp(-beta) * A_)


# rs, Fs = CCS_CDF([1,1], [4,4], T=1500)  # DEBUG
# for x in [25,50,75,100]:
#     ss = slice(x-1,x+1)
#     print(f"rs:{rs[0][ss]}")
#     print(f"Fs:{Fs[0][ss]}\n")

# print(f"\n\nDEBUG Fs:\n{rs[0]}")
# print(f"DEBUG Fs:\n{Fs[0]}")


if __name__ == "__main__":
    main_DEBUG()
