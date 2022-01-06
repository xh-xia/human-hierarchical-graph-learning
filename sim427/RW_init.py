"""
This is to add state points to signac database.
It defines ranges of params over which one wants to sweep.

Created: Tuesday, ‎March ‎23, ‎2021, ‏‎6:49:54 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import signac as sn

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utility427.helper427 import set_dir427, get_params
from utility427.math427 import np
from utility427.sim_params427 import make_sim_params, get_max_betas

set_dir427()  # change cwd to script dir

project = sn.get_project()

"""
reg_n_p: 20 cases of combination from:
    hierDict["reg_n"] = [[0, 1, 2, 3], [3], [3, 4, 5]] = 12
    hierDict["reg_p"] = [[0, 1, 2, 3], [3, 4, 5], [3]] = 8 (excluding those in reg_n)
implemented variable beta case: with num of changes = [1, 2, 3, 4, 5, 6, 8, 10, 12]
"""
params = make_sim_params(get_params())  # load params.json as dict, use it to create sim params
max_beta_dict = get_params(fname="params_max_beta")  # from hi to low

reverse = True  # default is reversed, meaning lo2hi
if "hi2lo" in params["key_class"]:  # if key_class contains hi2lo, don't reverse
    reverse = False

RNG = np.random.default_rng(seed=params["SEED"])

"""NOTE
"var_betas", "beta_classes", "beta_arrs" have same length
that's why they are all sliced by [i]
time it took:
- 621 sec (10.35 min) to make 132,000 (3*20*(13+9)*100) state points (36.7 MB)
- 1 min 59 sec + 58 sec (~3 min) to check status (python RW_jobs.py status)
- 2 hr 51 min to run all jobs (python RW_jobs.py run --ignore-conditions none --progress)
- 311 sec to run RW_CCS_stat.py
"""

counter = 0  # to report number of sp created
for i in range(len(params["beta_classes"])):
    for regType, p, n in params["pd"]:
        fname = "output/npy_files/"
        fname += f"Sierpinski(regType={regType:d},p={p:d},n={n:d}).npy"
        for beta_idx in range(len(params["beta_arrs"][i])):  # loop over groups of beta: group beta
            for agentID in params["range_agents"]:  # loop over each group: actual beta
                seed = RNG.integers(params["int_max"])
                project.open_job(
                    {
                        "key_class": params["key_class"],
                        "beta_class": params["beta_classes"][i],
                        "seed": seed,
                        "agentID": agentID,
                        "n_agents": params["n_agents"],
                        "steps_tot": params["steps_tot"],
                        "sample_period": params["sample_period"],
                        "regType": regType,
                        "p": p,
                        "n": n,
                        "beta_grp": params["beta_arrs"][i][beta_idx],  # beta shared in grp
                        "beta": params["beta_acts"][i][beta_idx, agentID],  # beta used in sim
                        "beta_idx": beta_idx,  # group beta idx; actual beta idx = agentID
                        "var_beta": params["var_betas"][i],
                        "max_betas": get_max_betas(max_beta_dict, regType, p, n, reverse=reverse),
                    }
                ).init()
                counter += 1
                if counter % 200 == 0:
                    print(f"Progresss: {counter} state points created")

print(f"Total Progresss: {counter} state points created")
