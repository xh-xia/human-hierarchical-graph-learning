"""
This is library of classes for running Graph Learning (GL) simulations.
It has only basic functionalities for now.
Created: Monday, ‎March ‎22, ‎2021, ‏‎9:35:55 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import numpy as np

class GLsim:
    """
    main simulation object (discrete obviously, since we are working with a graph)
    Args
    -------
    params (dict): dict of parameters
        'seed': seed for np.random.default_rng
        'steps_tot': total number of steps of the random walk
        'sample_period': every <num of time steps> to record the related matrices from RW
        'A' (nparr): adjacency matrix (if A present, don't use P)
        'P' (nparr): transition prob matrix (if P present, don't use A)
        'agentID': agent id for the current simulation run (i.e., agent id within group which has same param)
        'beta': shuffling parameter in the vanilla Max-Entropy model
    """
    def __init__(self, params):
        self.seed = params['seed']
        self.steps_tot = params['steps_tot']
        self.sample_period = params['sample_period']
        self.agentID = params['agentID']
        self.beta = params['beta']

        # set up RNG
        self.RNG=np.random.default_rng(seed=self.seed)
        if params.has_key('A'):
            self.P=params['A']/np.sum(params['A'],1,keepdims=True) # row-normalize adjacency matrix
        else:
            self.P=params['P']
        self.N = self.P.shape[0] # get num of rows to be size of graph
        self.p = 1.0 - np.exp(-self.beta) # p parameter for geometric distribution starting at k=1
        # bunch of initializations
        self.steps_now = 0 # initialize current num of steps traversed
        self.node_now = self.RNG.integers(self.N) # starting node drawn at random
        self.path = np.full(self.steps_tot+1, fill_value=-1, dtype=int) # actual trajectory
        self.path_me = np.full(self.steps_tot+1, fill_value=-1, dtype=int) # mental trajectory
        self.path[self.steps_now] = self.node_now # start at current node
        self.path_me[self.steps_now] = self.node_now # start at current node mentally as well
        if self.sample_period > self.steps_tot:
            raise ValueError('<sample_period> larger than <steps_tot>.')
        self.n_sample = np.floor(self.steps_tot/self.sample_period) # tot num of samples (don't sample at the start)
        self.steps_sample = np.arange(1,self.n_sample+1) * self.sample_period # time stamps at the time of sampling
        self.count_ma = np.zeros((self.N,self.N),dtype=int) # current count matrix
        self.count_ma_me = np.zeros((self.N,self.N),dtype=int) # current mental count matrix
        self.counts = np.zeros((self.n_sample,self.N,self.N),dtype=int) # tensor: count_ma[i] is count matrix at sample i
        self.counts_me = np.zeros((self.n_sample,self.N,self.N),dtype=int) # ditto except this is mental count
    def walk(self):
        """
        walk one step
        """
        # get next node and walk onto it
        self.node_now = self.RNG.choice(self.N,p=self.P[self.node_now,:])
        self.steps_now += 1 # update num of steps walked
        # update path
        self.path[self.steps_now] = self.node_now
        # update mental path with shuffling (from actual path)
        self.path_me[self.steps_now] = self.path[self.steps_now] # the current one is fine, shuffle is about last step
        # convention 1: if path does not yet have old enough history, shuffle to earliest possible
        self.path_me[self.steps_now-1] = self.path[max(0,self.steps_now-self.RNG.geometric(self.p))]
        """ convention 2: # if path does not yet have old enough history, no shuffling
        back_to_step = self.steps_now-self.RNG.geometric(self.p)
        if back_to_step < 0: # if path does not yet have old enough history, no shuffling
            self.path_me[self.steps_now-1] = self.path[self.steps_now-1]
        else:
            self.path_me[self.steps_now-1] = self.path[back_to_step]
        """
        # update count matrix and mental one too
        self.count_ma[self.path[self.steps_now-1],self.path[self.steps_now]] += 1
        self.count_ma_me[self.path_me[self.steps_now-1],self.path_me[self.steps_now]] += 1
        if self.steps_now % self.sample_period == 0: # record count if at sampling point
            self.counts[self.steps_now/self.sample_period-1,:,:] = self.count_ma
            self.counts_me[self.steps_now/self.sample_period-1,:,:] = self.count_ma_me
    def walks(self): # RW on full length
        for k in range(self.steps_tot):
            self.walk()
