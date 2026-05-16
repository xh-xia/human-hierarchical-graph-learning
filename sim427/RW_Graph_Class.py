"""
This is library of classes for running Graph Learning (GL) simulations.
It has only basic functionalities for now.
Created: Monday, March 22, 2021, 9:35:55 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utility427.helper427 import set_dir427, mkdir_p
from utility427.math427 import step_funct, A2P, np, npdivide0
from utility427.plt427 import make_level_masks
from utility427.Sierpinski427 import p2ten, p_ary, make_SierpinskiGraph427
from utility427.sim_params427 import make_beta_mat  # for use in finite geometric only


def get_geom_dist(n, beta):
    """
    it's adapted from Chris's belief_db_sequence2.m matlab code
    NOTE we use python indexing convention throughout this function
    NOTE it takes ~ 4 sec for n=7500, so we don't want to do it for every agent

    Return
    ------
    - P (2D nparr): P[t,s] is prob that at time t=1,...,n-1,
    (we start at second step because for counting edges, at least two nodes have to be visited)
    mistaking node at time t-1 for node at time s where s = 0,1,...,t-1
    how to use it:
    we sample time index s with which we find the corresponding node to update mental count
    s = self.RNG.choice(t, p=P[t-1, 0:t])
    """
    E = np.tile(range(n), [n, 1])
    E = abs(E.T - E)
    LowT = np.tril(np.ones((n, n)))
    W = np.exp(-beta * E)
    Z = np.sum(W * LowT, 1)[:, np.newaxis]  # col vec
    P = W * LowT / np.tile(Z, n)  # repeat Z along col
    return P


def save_geom_dist(n, n_beta=10):
    beta_mat = make_beta_mat(n_beta, 0, binned=False)
    beta_arrs = beta_mat[:, 0]  # group beta; only one entry since n_agents per bin is 0
    set_dir427()
    prefix = "npy_files\\"
    mkdir_p(prefix)
    for i in range(len(beta_arrs)):
        G = get_geom_dist(n, beta_arrs[i])
        np.save(prefix + f"Geom_beta{n_beta:d}bins_len-{n:d}_idx-{i:d}.npy", G)


def load_Geom(beta_idx, n, n_beta=10, folder_str="npy_files\\"):
    set_dir427()
    folder_str += f"Geom_beta{n_beta:d}bins_len-{n:d}_idx-{beta_idx:d}.npy"
    try:
        G = np.load(folder_str, allow_pickle=True)
    except OSError:  # couldn't find the file
        raise OSError("Oof, doesn't have finite geometric dist. npy file to work with")
    return G


def load_Sier(regType, p, n, folder_str="npy_files\\", return_which="both"):
    """
    load transition probability matrix as well as masks
    onto the arguments <P> and <masks> respectively
    assume "npy_files" is contained in where this script is at
    the npy file contains a dictionary which has two keys:
    'A' (nparr): transition probability matrix
    'masks' (dict): [f"lv{l}"] is a nparr containing mask of level l edge

    Args:
    -----
    return_which (str): not implemented
        'both': returns both 'A' and 'masks'
        'A': returns 'A'
        'masks': returns 'masks'
    """
    set_dir427()
    folder_str += f"Sierpinski(regType={regType:d},p={p:d},n={n:d}).npy"
    try:
        Sierpinski_dict = np.load(folder_str, allow_pickle=True).tolist()  # convert nparr to dict
    except OSError:  # couldn't find the file
        raise OSError("Oof, doesn't have transition prob matrix to work with")
    return Sierpinski_dict["A"], Sierpinski_dict["masks"]


class GLsim:
    """
    main simulation object (discrete obviously, since we are working with a graph)

    Args
    ----
    regType, p, n: regularization type, power, level
    seed: seed for np.random.default_rng()
    steps_tot: total number of steps of the random walk
    sample_period: every <num of time steps> to record the related matrices from RW
    agentID: agent id for the current run (i.e., agent id within group which has same param)

    Intermediary
    ------------
    self.beta (num, list-like): shuffling parameter in the vanilla Max-Entropy model
        NOTE: this also applies to beta (arg) if kwargs['var_beta'] is False or KeyError
        np.random.default_rng.geometric():
        https://numpy.org/doc/stable/reference/random/generator.html
        which basically says f(k) = (1-p)^(k-1) * p for k=1,2,...
        if p is list-like, and we want n samples per p: geometric(p, size=(n, len(p)))
        but we will only implement p in geometric() one at a time given how walk() is written
        and p = 1 - exp(-beta)
    """

    def __init__(self, seed, steps_tot, sample_period, agentID, beta, regType, p, n, **kwargs):
        self.seed = seed
        self.steps_tot = steps_tot
        self.sample_period = sample_period
        self.agentID = agentID
        self.beta = beta
        """
        if variable beta case, <beta> can be a positive or negative float (but not list-like)
        convert codename beta (negative float) into list-like beta
        NOTE: <0 beta := number of different actual beta
        """
        if "var_beta" in kwargs:
            if beta < 0:
                if isinstance(kwargs["var_beta"], bool) and kwargs["var_beta"]:
                    self.beta = step_funct(kwargs["max_betas"], steps_tot)
                elif isinstance(kwargs["var_beta"], (list, np.ndarray)):
                    raise Exception(f"<kwargs['var_beta']> being non-bool is currently deprecated")
                    # NOTE: we unpack kwargs["var_beta"] (list), assuming it has only 2 numbers
                    # self.beta = step_funct(-round(beta), steps_tot, *kwargs["var_beta"])
        # load transition prob matrix & masks
        # self.P, self.masks = load_Sier(regType, p, n)
        self.P, _ = load_Sier(regType, p, n)

        # set up RNG
        self.RNG = np.random.default_rng(seed=self.seed)
        self.N = self.P.shape[0]  # get num of rows to be size of graph
        if isinstance(self.beta, str):
            raise TypeError("<self.beta> is a string, but valid type is num or list-like")
        if hasattr(self.beta, "__len__"):
            if len(self.beta) == steps_tot:  # if there is a beta for each step
                self.p = 1.0 - np.exp(-self.beta)
            else:
                raise ValueError("if <self.beta> is list-like, its size has to match steps_tot")
        else:  # assume what's left is either an int or float
            self.p = np.full(steps_tot, 1.0 - np.exp(-self.beta))  # repeat same p for all steps
        # bunch of initializations
        self.steps_now = 0  # initialize current num of steps traversed
        self.node_now = self.RNG.integers(self.N)  # starting node drawn at random
        self.path = np.full(self.steps_tot + 1, fill_value=-1, dtype=int)  # actual trajectory
        self.path_me = np.full(self.steps_tot + 1, fill_value=-1, dtype=int)  # mental trajectory
        self.path[self.steps_now] = self.node_now  # start at current node
        self.path_me[self.steps_now] = self.node_now  # start at current node mentally as well
        if self.sample_period > self.steps_tot:
            raise ValueError("<sample_period> larger than <steps_tot>.")
        self.n_sample = int(
            np.floor(self.steps_tot / self.sample_period)
        )  # tot num of samples (don't sample at the start)
        self.steps_sample = (
            np.arange(1, self.n_sample + 1) * self.sample_period
        )  # time stamps at the time of sampling
        self.count_ma = np.zeros((self.N, self.N), dtype=int)  # current count matrix
        self.count_ma_me = np.zeros((self.N, self.N), dtype=int)  # current mental count matrix
        self.counts = np.zeros(
            (self.n_sample, self.N, self.N), dtype=int
        )  # tensor: count_ma[i] is count matrix at sample i
        self.counts_me = np.zeros(
            (self.n_sample, self.N, self.N), dtype=int
        )  # ditto except this is mental count

    def walk(self):
        """
        walk one step
        """
        # get next node and walk onto it
        self.node_now = self.RNG.choice(self.N, p=self.P[self.node_now, :])
        self.steps_now += 1  # update num of steps walked
        idx = self.steps_now - 1  # index of steps walked, starting from 0 instead of 1
        # update path
        self.path[self.steps_now] = self.node_now
        # update mental path with shuffling (from actual path)
        # the current one is as it is, shuffle is about last step
        self.path_me[self.steps_now] = self.path[self.steps_now]
        # convention 1: if path does not yet have old enough history, shuffle to earliest possible
        self.path_me[idx] = self.path[max(0, self.steps_now - self.RNG.geometric(self.p[idx]))]
        """ convention 2: # if path does not yet have old enough history, no shuffling
        back_to_step = self.steps_now-self.RNG.geometric(self.p)
        if back_to_step < 0: # if path does not yet have old enough history, no shuffling
            self.path_me[idx] = self.path[idx]
        else:
            self.path_me[idx] = self.path[back_to_step]
        """
        # update count matrix and mental one too
        self.count_ma[self.path[idx], self.path[self.steps_now]] += 1
        self.count_ma_me[self.path_me[idx], self.path_me[self.steps_now]] += 1
        if self.steps_now % self.sample_period == 0:  # record count if at sampling point
            self.counts[self.steps_now // self.sample_period - 1, :, :] = self.count_ma
            self.counts_me[self.steps_now // self.sample_period - 1, :, :] = self.count_ma_me

    def walks(self):  # RW on full length
        for _ in range(self.steps_tot):
            self.walk()
        return self.output()

    def output(self):
        # return a dictionary
        return {
            "path": self.path,
            "path_me": self.path_me,
            "counts_me": self.counts_me,
            "steps_sample": self.steps_sample,
        }


class GLsim2:
    """
    main simulation object (discrete obviously, since we are working with a graph)

    Main difference from GLsim is that this uses finite geometric distribution
    but since creating one takes long time, only fix beta sequence is pre-created,
    so it doesn't have var_beta or beta grouping functionalities anymore
    Created: Monday, February 21, 2022, 8:51:09 PM (EST)

    Args
    ----
    regType, p, n: regularization type, power, level
    seed: seed for np.random.default_rng()
    steps_tot: total number of steps of the random walk
    sample_period: every <num of time steps> to record the related matrices from RW
    agentID: agent id for the current run (i.e., agent id within group which has same param)

    Intermediary
    ------------
    self.beta_idx (int): idx for shuffling parameter in the vanilla Max-Entropy model
        use finite geometric instead of previous infinite version
        this does not work with kwargs['var_beta'], yet
        beta will only be inferred from beta_idx
    """

    def __init__(self, seed, steps_tot, sample_period, agentID, beta_idx, regType, p, n, **kwargs):
        self.seed = seed
        self.steps_tot = steps_tot
        self.sample_period = sample_period
        self.agentID = agentID
        self.beta_idx = beta_idx  # NOTE no actual beta involved, only idx
        
        self.P, _ = load_Sier(regType, p, n)  # load transition prob matrix
        self.G = load_Geom(self.beta_idx, self.steps_tot)  # load finite Geometric dist.

        # set up RNG
        self.RNG = np.random.default_rng(seed=self.seed)
        self.N = self.P.shape[0]  # get num of rows to be size of graph
        # bunch of initializations
        self.steps_now = 0  # initialize current num of steps traversed
        self.node_now = self.RNG.integers(self.N)  # starting node drawn at random
        self.path = np.full(self.steps_tot + 1, fill_value=-1, dtype=int)  # actual trajectory
        self.path_me = np.full(self.steps_tot + 1, fill_value=-1, dtype=int)  # mental trajectory
        self.path[self.steps_now] = self.node_now  # start at current node
        self.path_me[self.steps_now] = self.node_now  # start at current node mentally as well
        if self.sample_period > self.steps_tot:
            raise ValueError("<sample_period> larger than <steps_tot>.")
        self.n_sample = int(
            np.floor(self.steps_tot / self.sample_period)
        )  # tot num of samples (don't sample at the start)
        self.steps_sample = (
            np.arange(1, self.n_sample + 1) * self.sample_period
        )  # time stamps at the time of sampling
        self.count_ma = np.zeros((self.N, self.N), dtype=int)  # current count matrix
        self.count_ma_me = np.zeros((self.N, self.N), dtype=int)  # current mental count matrix
        self.counts = np.zeros(
            (self.n_sample, self.N, self.N), dtype=int
        )  # tensor: count_ma[i] is count matrix at sample i
        self.counts_me = np.zeros(
            (self.n_sample, self.N, self.N), dtype=int
        )  # ditto except this is mental count

    def walk(self):
        """
        walk one step
        """
        # get next node and walk onto it
        self.node_now = self.RNG.choice(self.N, p=self.P[self.node_now, :])
        self.steps_now += 1  # update num of steps walked
        idx = self.steps_now - 1  # index of steps walked, starting from 0 instead of 1
        # update path
        self.path[self.steps_now] = self.node_now
        # update mental path with shuffling (from actual path)
        # the current one is as it is, shuffle is about last step
        self.path_me[self.steps_now] = self.path[self.steps_now]
        self.path_me[idx] = self.path[self.RNG.choice(idx + 1, p=self.G[idx, 0:(idx + 1)])]
        # update count matrix and mental one too
        self.count_ma[self.path[idx], self.path[self.steps_now]] += 1
        self.count_ma_me[self.path_me[idx], self.path_me[self.steps_now]] += 1
        if self.steps_now % self.sample_period == 0:  # record count if at sampling point
            self.counts[self.steps_now // self.sample_period - 1, :, :] = self.count_ma
            self.counts_me[self.steps_now // self.sample_period - 1, :, :] = self.count_ma_me

    def walks(self):  # RW on full length
        for _ in range(self.steps_tot):
            self.walk()
        return self.output()

    def output(self):
        # return a dictionary
        return {
            "path": self.path,
            "path_me": self.path_me,
            "counts_me": self.counts_me,
            "steps_sample": self.steps_sample,
        }


class GLsim3:
    """
    main simulation object (discrete obviously, since we are working with a graph)

    this is for the alternative model: two-point distribution on correct and most visited node

    for the purposes of testing and revision, I used the default values:
        error rate r = 0.74 (roughly beta=0.3)
        window size w = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144] (1 means 1 node before the correct one)
        (window size has 10 to match number of beta we considered,
        and it's a part of the Fibonacchi sequence because
        1. I want good coverage of smaller sizes
        2. but also some way larger sizes in case small ones can't learn the coarser hierarchy)
    Created: Wednesday, December 13, 2023, 7:33:11 PM (EST)

    Args
    ----
    regType, p, n: regularization type, power, level
    seed: seed for np.random.default_rng()
    steps_tot: total number of steps of the random walk
    sample_period: every <num of time steps> to record the related matrices from RW
    agentID: agent id for the current run (i.e., agent id within group which has same param)

    Intermediary
    ------------
    self.w_arr (arr): the index (i.e., w_idx) is used on this arr of actual window sizes
    """

    def __init__(self, seed, steps_tot, sample_period, agentID, w_idx, regType, p, n, **kwargs):
        self.seed = seed
        self.steps_tot = steps_tot
        self.sample_period = sample_period
        self.agentID = agentID
        self.w_idx = w_idx

        self.P, _ = load_Sier(regType, p, n)  # load transition prob matrix
        # if we want to change, we just need to specify in kwargs,
        # otherwise below are default values
        self.r = 0.74
        self.w_arr = np.array([2, 3, 5, 8, 13, 21, 34, 55, 89, 144])  # window size arr
        self.w = self.w_arr[self.w_idx]
        # below (until walk(self)) is the same as GLsim2

        # set up RNG
        self.RNG = np.random.default_rng(seed=self.seed)
        self.N = self.P.shape[0]  # get num of rows to be size of graph
        # bunch of initializations
        self.steps_now = 0  # initialize current num of steps traversed
        self.node_now = self.RNG.integers(self.N)  # starting node drawn at random
        self.path = np.full(self.steps_tot + 1, fill_value=-1, dtype=int)  # actual trajectory
        self.path_me = np.full(self.steps_tot + 1, fill_value=-1, dtype=int)  # mental trajectory
        self.path[self.steps_now] = self.node_now  # start at current node
        self.path_me[self.steps_now] = self.node_now  # start at current node mentally as well
        if self.sample_period > self.steps_tot:
            raise ValueError("<sample_period> larger than <steps_tot>.")
        self.n_sample = int(
            np.floor(self.steps_tot / self.sample_period)
        )  # tot num of samples (don't sample at the start)
        self.steps_sample = (
            np.arange(1, self.n_sample + 1) * self.sample_period
        )  # time stamps at the time of sampling
        self.count_ma = np.zeros((self.N, self.N), dtype=int)  # current count matrix
        self.count_ma_me = np.zeros((self.N, self.N), dtype=int)  # current mental count matrix
        self.counts = np.zeros(
            (self.n_sample, self.N, self.N), dtype=int
        )  # tensor: count_ma[i] is count matrix at sample i
        self.counts_me = np.zeros(
            (self.n_sample, self.N, self.N), dtype=int
        )  # ditto except this is mental count

    def walk(self):
        """
        walk one step
        """
        # get next node and walk onto it
        self.node_now = self.RNG.choice(self.N, p=self.P[self.node_now, :])
        self.steps_now += 1  # update num of steps walked
        idx = self.steps_now - 1  # index of steps walked, starting from 0 instead of 1
        # update path
        self.path[self.steps_now] = self.node_now
        # update mental path with shuffling (from actual path)
        # the current one is as it is, shuffle is about last step
        self.path_me[self.steps_now] = self.path[self.steps_now]
        # find most visited node
        if self.w < 3:  # window size of 1 or 2; it includes the correct node, thus min=1
            idx_err = idx  # the correct node, so it's always correct, no errors made
        else:
            temp_mv = self.path[max(0, self.steps_now - self.w) : self.steps_now]
            temp_mode = np.unique(temp_mv[::-1], return_index=True, return_counts=True)
            temp_idx_max = np.flatnonzero(temp_mode[2] == np.max(temp_mode[2]))  # most visited
            temp_idx_recent = np.min(temp_mode[1][temp_idx_max])  # most recent (idx from right of temp_mv)
            idx_err = idx - temp_idx_recent
        self.path_me[idx] = self.path[self.RNG.choice([idx_err, idx], p=[self.r, 1 - self.r])]
        # update count matrix and mental one too
        self.count_ma[self.path[idx], self.path[self.steps_now]] += 1
        self.count_ma_me[self.path_me[idx], self.path_me[self.steps_now]] += 1
        if self.steps_now % self.sample_period == 0:  # record count if at sampling point
            self.counts[self.steps_now // self.sample_period - 1, :, :] = self.count_ma
            self.counts_me[self.steps_now // self.sample_period - 1, :, :] = self.count_ma_me

    def walks(self):  # RW on full length
        for _ in range(self.steps_tot):
            self.walk()
        return self.output()

    def output(self):
        # return a dictionary
        return {
            "path": self.path,
            "path_me": self.path_me,
            "counts_me": self.counts_me,
            "steps_sample": self.steps_sample,
        }


def CCS(counts_me, regType, p, n, seed=0, analytic_comp=False):
    """simplified & modified heavily from CCS_analysis in CCS427.py
    This function finds CCS for given transition prob matrix
    It calculates CCS for all P_hat in count_ma_me

    2022.2.17 added new convention: when calculating CCS (ratio)
    we set the result to be np.nan if denom is close to 0
    Args
    ----
    - counts_me (3D np.arr): simulated result
    - regType, p, n: regularization type, power, level
    - masks (dict): masks[f"lv{l}"] is level-l mask for P

    Return
    ------
    - CCS_arr (3D nparr):
    since CCS is for every 2 consecutive lvs, we have only (lv-1) entries out of lv levels
        CCS_arr[s,0,l-1]: CCS of means at sample s for level l (f"lv{l}{'-'}{l+1}") for current agent
        CCS_arr[s,1,l-1]: CCS of stds at sample s for level l (f"lv{l}{'-'}{l+1}") for current agent
    """
    try:
        _, masks = load_Sier(regType, p, n)
    except OSError:  # generate on the fly
        masks = make_level_masks(make_SierpinskiGraph427(p, n, norm = True, regType = regType))
    if analytic_comp:  # assume counts_me is transition prob matrix
        Ps_me = counts_me[np.newaxis, ...]  # since it's analytic, there is no sample (=1)
    else:  # ↓ convert counts_me into transition prob matrix
        np.allclose(counts_me, 0)  # don't ask me why, but this fixes the Ps_me containing nan issue
        Ps_me = A2P(counts_me, axis=2)  # get transition probability matrix from counts

    list_lv = [
        int(k[2:]) for k in masks.keys()
    ]  # list of all levels (e.g., [1,2,3,-1] for load_Sier(3, 3, 3))
    lv = max(list_lv)  # (max) hierarchical level (also the coarsest level)

    CCS_arr = np.zeros((Ps_me.shape[0], 2, lv - 1))
    # ↓ calculation
    for s in range(Ps_me.shape[0]):  # s stand for sample
        mean_weights = [0.0 for i in range(lv)]
        std_weights = [0.0 for i in range(lv)]
        for l in range(1, lv + 1):
            mean_weights[l - 1] = np.mean(Ps_me[s][np.nonzero(masks[f"lv{l}"])])
            std_weights[l - 1] = np.std(Ps_me[s][np.nonzero(masks[f"lv{l}"])])
        # temp = -np.diff(mean_weights) # diff: all >0 if edge weights in finer level > coarser level
        # temp = np.exp(-np.diff(np.log(mean_weights))) # ratio: all >1 if edge weights in finer level > coarser level
        # print("DEBUG mean_weights: {} | std_weights: {}".format(mean_weights,std_weights))
        temp1 = npdivide0(mean_weights[:-1], mean_weights[1:], atol=1.e-14, filler=np.nan)  # ditto, but more explicit
        temp2 = npdivide0(std_weights[:-1], std_weights[1:], atol=1.e-14, filler=np.nan)
        # print("DEBUG: {} with seed {}".format(mean_weights,seed))
        for l in range(0, lv - 1):
            CCS_arr[s, 0, l] = temp1[l]
            CCS_arr[s, 1, l] = temp2[l]

    return CCS_arr


def CCPS(counts_me, regType, p, n, ccps_type=1, analytic_comp=False, scale=1):
    """
    similar to Cross-Community Surprisal (CCS),
    Cross-Community Pseudo-Surprisal (CCPS) is also defined per 2 consecutive levels
    but on spurious edges and across communities (instead of CCS's real edges and across nodes)
    for both CCS and CCPS, l=1,2,...,n-1
    level-l CCS: mean(W(lv=l)) / mean(W(lv=l+1)) | finer:coarser
    level-l CCPS: mean(W(w/in lv=l)) - mean(W(between lv=l)) | finer:coarser | spurious only
    level-l CCTS: mean(W(w/in lv=l)) - mean(W(between lv=l)) | finer:coarser | all edges
    NOTE: W(w/in lv=l+1) = W(between lv=l)

    Kwargs
    ------
    - ccps_type (int):
        1: spurious edges only
        2: both spurious and real edges; this we can do divide since CCS works
            however: scaling is a problem in terms of interpretation
            1) mean(), the curve = 1 at beta->0; but curve is some weird large number at beta->infty
            because as lv increases, the edge becomes sparser in ground truth, diluting the mean()
            2) divide sum by num of real edges in groud truth,
            the curve is some weird small number (~0) at beta->0; but curve = 1 at beta->infty:
            assume we only count each edge once: (only `mean_weights` is modified)
            regType in [0, 3]: mean_weights[l - 1] := sum(w/in lv=l) / p^(n-l+1)
            regType in [1]: mean_weights[l - 1] := sum(w/in lv=l) / p^(n-l+1) if l<n+1
                            mean_weights[l - 1] := sum(w/in lv=l) / p if l=n+1
            intuition behind results (regardless of scaling):
            at ground truth (beta=infty), max cross-community surprisal (different from CCS result)
            but as beta decreases, it turns out higher lv CCTS reaches max surprisal earlier;
            sub-max surprisal means "leakage" biases towards higher level;
            earlier means difference of transition prob "leakage" between two consecutive levels
            is more pronounced at lower level (leakage is more obvious at lower levels)
            from another perspective:
            learned graph is losing community structure faster at lower level as beta decreases
    - scale (int): ccps_type=2 only; 1 -> mean() 2-> divide sum by num of real edges in groud truth
    """
    try:
        A, _ = load_Sier(regType, p, n)
    except OSError:  # generate on the fly
        A = make_SierpinskiGraph427(p, n, norm = True, regType = regType)["A"]
    A[A>0] = 1  # adjacency matrix of ground truth
    masks = make_masks(regType, p, n, A.astype(int), ccps_type)
    if analytic_comp:  # assume counts_me is transition prob matrix
        Ps_me = counts_me[np.newaxis, ...]  # since it's analytic, there is no sample (=1)
    else:  # ↓ convert counts_me into transition prob matrix
        np.allclose(counts_me, 0)  # don't ask me why, but this fixes the Ps_me containing nan issue
        Ps_me = A2P(counts_me, axis=2)  # get transition probability matrix from counts

    if regType in [0, 3]:
        lv = n  # (max) hierarchical level (also the coarsest level) for node community
    elif regType in [1]:
        lv = n + 1
    else:
        raise NotImplementedError(f"regType={regType} is not implemented yet")

    CCPS_arr = np.zeros((Ps_me.shape[0], 2, lv - 1))
    # ↓ calculation
    for s in range(Ps_me.shape[0]):  # s stand for sample
        mean_weights = [0.0 for i in range(lv)]
        std_weights = [0.0 for i in range(lv)]
        for l in range(1, lv + 1):
            if ccps_type == 1 and l == 1:  # there is no spurious edge in level 1 community
                mean_weights[l - 1] = 0
                std_weights[l - 1] = 0
            else:
                if scale == 1 or ccps_type == 1:
                    mean_weights[l - 1] = np.mean(Ps_me[s][masks[f"lv{l}"]])
                elif scale == 2 and ccps_type == 2:
                    if l < n + 1:
                        mean_weights[l - 1] = np.sum(Ps_me[s][masks[f"lv{l}"]]) / round(p ** (n - l + 1))
                    else:  # l=n+1 i.e., regType in [1]; since otherwise l can only reach n
                        mean_weights[l - 1] = np.sum(Ps_me[s][masks[f"lv{l}"]]) / p
                else:
                    raise NotImplementedError(f"scale={scale} is invalid")
                std_weights[l - 1] = np.std(Ps_me[s][masks[f"lv{l}"]])
        if ccps_type == 1:  # divide -> unsteady since both numerator and denominator can be small
            temp1 = np.subtract(mean_weights[:-1], mean_weights[1:])
            temp2 = np.subtract(std_weights[:-1], std_weights[1:])
        elif ccps_type == 2:  # divide -> steady since CCS is steady
            temp1 = np.divide(mean_weights[:-1], mean_weights[1:])
            temp2 = np.divide(std_weights[:-1], std_weights[1:])
        else:
            raise NotImplementedError("currently only ccps_type=1 or 2 is implemented")
        for l in range(0, lv - 1):
            CCPS_arr[s, 0, l] = temp1[l]
            CCPS_arr[s, 1, l] = temp2[l]

    return CCPS_arr


def make_masks(regType, p, n, A_real, ccps_type):
    """
    calculate on the fly instead of loading local files
    this is to find spurious edges on multiple hierarchical levels
    given how Sierpiński family is defined, level 1 community is the basic motif, fully connected
    which means w/in lv=1 community there are no spurious edges
    the way I define w/in or between for spurious edges is as follows:
    w/in lv=l+1 community := between lv=l community
    in the script I will use w/in consistently,
    for intuition it's more convenient to use both w/in and between when defining CCPS at lv=l:
        w/in lv=l - between lv=l
    AKA w/in lv=l - w/in lv=l+1

    Args
    ----
    - A_real (np.arr): adjacency matrix (assume int entry) for real edges
    - ccps_type (int):
        1: spurious edges only
        2: both spurious and real edges; this we can do divide since CCS works
    """
    # for now we only implement it for regType=0,1,3 (no reg, 1-node, and self-loop)
    if regType in [0, 3]:
        N = round(p**n)
        nodeidx_p = [p_ary(x, p=p, L=n) for x in range(N)]
        masks = dict()
        cum_mask = np.zeros((N, N), dtype=bool)  # cumulative via OR(entry_i==1,i=1,...)
        for l in range(1, n + 1):
            asm = [p2ten(pstr[0 : n - l], p=p) for pstr in nodeidx_p]  # community assignments
            masks[f"lv{l}"] = np.zeros((N, N), dtype=bool)
            if ccps_type == 1:
                if l == 1:
                    continue  # because there are no spurious edges w/in lv=1 community
                for i in range(N):  # no self-loop; assume undirected
                    for j in range(i + 1, N):
                        b = (A_real[i, j]==0) and (asm[i] == asm[j]) and (not cum_mask[i, j])
                        masks[f"lv{l}"][i, j] = b
                        masks[f"lv{l}"][j, i] = b
                        if masks[f"lv{l}"][i, j]:
                            cum_mask[i, j] = True
                            cum_mask[j, i] = True
            elif ccps_type == 2:
                for i in range(N):  # no self-loop; assume undirected
                    for j in range(i + 1, N):
                        b = (asm[i] == asm[j]) and (not cum_mask[i, j])
                        masks[f"lv{l}"][i, j] = b
                        masks[f"lv{l}"][j, i] = b
                        if masks[f"lv{l}"][i, j]:
                            cum_mask[i, j] = True
                            cum_mask[j, i] = True
            else:
                raise NotImplementedError("currently only ccps_type=1 or 2 is implemented")
    elif regType in [1]:
        N = round(p**n) + 1
        nodeidx_p = [p_ary(x, p=p, L=n + 1) for x in range(N)]
        masks = dict()
        cum_mask = np.zeros((N, N), dtype=bool)  # cumulative via OR(entry_i==1,i=1,...)
        for l in range(1, n + 2):
            asm = [p2ten(pstr[0 : n - l + 1], p=p) for pstr in nodeidx_p]  # community assignments
            masks[f"lv{l}"] = np.zeros((N, N), dtype=bool)
            if ccps_type == 1:
                if l == 1:
                    continue  # because there are no spurious edges w/in lv=1 community
                for i in range(N):  # no self-loop; assume undirected
                    for j in range(i + 1, N):
                        b = (A_real[i, j]==0) and (asm[i] == asm[j]) and (not cum_mask[i, j])
                        masks[f"lv{l}"][i, j] = b
                        masks[f"lv{l}"][j, i] = b
                        if masks[f"lv{l}"][i, j]:
                            cum_mask[i, j] = True
                            cum_mask[j, i] = True
            elif ccps_type == 2:
                for i in range(N):  # no self-loop; assume undirected
                    for j in range(i + 1, N):
                        b = (asm[i] == asm[j]) and (not cum_mask[i, j])
                        masks[f"lv{l}"][i, j] = b
                        masks[f"lv{l}"][j, i] = b
                        if masks[f"lv{l}"][i, j]:
                            cum_mask[i, j] = True
                            cum_mask[j, i] = True
            else:
                raise NotImplementedError("currently only ccps_type=1 or 2 is implemented")
    else:
        raise NotImplementedError(f"regType={regType} is not implemented yet")

    return masks

