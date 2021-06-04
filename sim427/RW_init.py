"""
This is to add state points to signac database.
It defines ranges of params over which one wants to sweep.

Created: Tuesday, ‎March ‎23, ‎2021, ‏‎6:49:54 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import signac as sn
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utility427.helper427 import get_params, set_dir427

set_dir427() # change cwd to dir that the script is in

project=sn.get_project()

params = get_params(n_agents=100, key_class='reg_n_p')

RNG=np.random.default_rng(seed=params['SEED'])

for key_class in params['key_classes']:
    for regType, p, n in params['pd']:
        fname = 'output/npy_files/'
        fname += 'Sierpinski(regType={:d},p={:d},n={:d}).npy'.format(regType,p,n)
        for beta_idx in range(len(params['beta_arr'])):
            for agentID in params['range_agents']:
                seed = RNG.integers(params['int_max'])
                project.open_job({
                    'key_class':key_class, 'seed':seed,
                    'agentID':agentID, 'n_agents':params['n_agents'],
                    'steps_tot':params['steps_tot'], 'sample_period':params['sample_period'],
                    'regType':regType, 'p':p, 'n':n,
                    'beta':params['beta_arr'][beta_idx], 'beta_idx':beta_idx
                    }).init()
