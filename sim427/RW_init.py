"""
This is to add state points to signac database.
It defines ranges of params over which one wants to sweep.

Created: Tuesday, ‎March ‎23, ‎2021, ‏‎6:49:54 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import signac as sn
import numpy as np
from itertools import product

project=sn.get_project()
SEED = 427
RNG=np.random.default_rng(seed=SEED)

hierDict = dict()
#hierDict['n'] = [[0],[3],[3,4,5]]
#hierDict['p'] = [[3],[3,4,5],[3]]
#hierDict['reg_n'] = [[0,1,2,3],[3],[3,4,5]]
#hierDict['reg_p'] = [[0,1,2,3],[3,4,5],[3]]
hierDict['r'] = [[0,1,2],[3],[3]]

key_classes = hierDict.keys()
n_agents = 10 # num of agents per param (including beta)
int_max = np.iinfo(int).max # 2147483647 for int (int32 really)

beta_arr = 10**np.linspace(-3,1,13,endpoint=True)
range_agents = range(n_agents)

for key_class in key_classes:
    hierLists = hierDict[key_class]
    for regType, p, n, beta, agentID in product(hierLists[0],hierLists[1],hierLists[2],beta_arr,range_agents):
        fname = 'output/npy_files/'
        fname += 'Sierpinski(regType={:d},p={:d},n={:d}).npy'.format(regType,p,n)
        seed = RNG.integers(int_max)
        project.open_job({
            'key_class':key_class, 'agentID':agentID, 'seed':seed,
            'steps_tot':steps_tot, 'sample_period':sample_period,
            'regType':regType, 'p':p, 'n':n, 'beta':beta
            }).init()
