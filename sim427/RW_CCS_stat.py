"""
This is to post-process the data from completed jobs in signac database
Specifically, those whose CCS_done(job) is True
what it does is simply find basic statistics for the CCS:
mean, standard deviation, standard error (std/sqrt(sample size))
Created: Thursday, ‎March ‎25, ‎2021, ‏‎10:30:09 AM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import signac as sn

from RW_Graph_Class import CCS, CCPS

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utility427.helper427 import set_dir427, mkdir_p, get_params
from utility427.math427 import np, bootstrap427
from utility427.sim_params427 import make_sim_params


"""
CCS_stat (dict): 3 keys each corresponding to one statistic of CCS:
"mean", "std", etc., the value (dict) of them is of the same structure:
value[(regType,p,n)] (3D nparr): "[slice]: meaning"
    s stands for the s-th sample in counts tensor defined in RW_Graph_Class.py
    (2nd dim specifies type of value in the cell; 3rd dim corresponds to different beta)
    [s,0,:]: group beta
    [s,1,:]: group size (each one in the group has different beta and aggregated via statistics)
    [s,2,:]: steps_sample (time stamps at the time of sampling = walk length)
    [s,3,:]: stat of the group having that beta for CCS at level 1
    ...
    [s,3+n-2,:]: stat of the group having that beta for CCS at level n-1
"raw" = stat_arr, has value[(regType,p,n)] (4D nparr):
    [s,l,b,i]: lv l CCS at s-th sample, b-th beta, for i-th agent
"""

is_operation = True  # whether CCS_compute is done as @operation in signac
sub_folder_name = "reg_n_p_binned_beta" # folder inside output folder to store all CCS_stat
save_raw = True  # whether save all agents' CCS; this makes the .np around 500kb instead of 30kb
CCS_key = "CCS"  # options: CCS, CCPS, CCPS2

KEYS = ["mean", "ste", "median", "ste_median"]  # keys of CCS_stat
if save_raw:
    KEYS += ["raw"]


def print_progress(counter, tot=28000):
    if counter % 1000 == 0:
        print(f"Progresss: {counter}/{tot}")


def main_CCS_stat(CCS_key):
    # np.seterr(all='raise') # set all runtime warning to raise errors
    # load parameters from json
    temp = get_params()
    # change some parameters
    # temp["n_agents"], temp["key_class"] = 10, "r"  # ~ 6 sec (may be inaccurate)
    # temp["n_agents"], temp["key_class"] = 10, "reg_n_p"  # ~ 7 sec (may be inaccurate)
    # temp["n_agents"], temp["key_class"] = 100, "max_beta"  # 28000 sp ~ 200 sec
    params = make_sim_params(temp)
    CCS_type = "mean"  # 'mean' or 'std'
    if CCS_type == "mean":
        CCS_type_slice = 0
    else:
        CCS_type_slice = 1
    CCS_stat = dict()

    for k in KEYS:
        CCS_stat[k] = dict()
    project = sn.get_project()
    n_sample = int(np.floor(params["steps_tot"] / params["sample_period"]))
    for i in range(len(params["beta_classes"])):
        counter = 0  # to report sub-progress
        for regType, p, n in params["pd"]:
            if regType == 2:  # this is never used in CCS, CCPS, or CCTS
                continue  # skip the rest and go back to the current loop of regType, p, n
            job_criteria = {
                "key_class": params["key_class"],
                "beta_class": params["beta_classes"][i],
                "regType": regType,
                "p": p,
                "n": n,
                "n_agents": params["n_agents"],
            }
            nparr = np.zeros((n_sample, 3 + n - 1, len(params["beta_arrs"][i])))
            stat_arr = np.zeros((n_sample, n - 1, len(params["beta_arrs"][i]), params["n_agents"]))
            for job in project.find_jobs(job_criteria):
                counter += 1
                print_progress(counter)
                nparr[:, 0, job.sp.beta_idx] = job.sp.beta_grp  # not the actual beta used in sim
                nparr[:, 1, job.sp.beta_idx] = params["n_agents"]
                with job.data:
                    nparr[:, 2, job.sp.beta_idx] = job.data["GLsim_data"]["steps_sample"][:]
                    if is_operation:
                        temp = job.data[CCS_key][:]
                    else:
                        # CCS_compute here (not as @operation)
                        kw_CCS = dict(counts_me=job.data["GLsim_data"]["counts_me"][:])
                        kw_CCS.update(dict(regType=job.sp.regType, p=job.sp.p, n=job.sp.n))
                        if CCS_key == "CCS":
                            temp = CCS(**kw_CCS, seed=job.sp.seed)
                        elif CCS_key == "CCPS":
                            temp = CCPS(**kw_CCS, ccps_type=1)
                        elif CCS_key == "CCPS2":
                            temp = CCPS(**kw_CCS, ccps_type=2)

                        else:
                            raise NotImplementedError("currently only CCS and CCPS are valid")
                for l in range(n - 1):  # CCS level index; only up to n-2 (i.e., CCS level n-1)
                    stat_arr[:, l, job.sp.beta_idx, job.sp.agentID] = temp[:, CCS_type_slice, l]
                    # print('DEBUG: regType={},p={},n={},agentID={},beta_idx={},seed={}'\
                    #       .format(regType, p, n, job.sp.agentID, job.sp.beta_idx, job.sp.seed))
            if save_raw:
                CCS_stat["raw"][(regType, p, n)] = stat_arr

            nparr[:, 3:, :] = np.nanmean(stat_arr, axis=3)
            CCS_stat["mean"][(regType, p, n)] = nparr.copy()
            nparr[:, 3:, :] = np.nanstd(stat_arr, axis=3) / np.sqrt(params["n_agents"])
            CCS_stat["ste"][(regType, p, n)] = nparr.copy()

            nparr[:, 3:, :] = np.nanmedian(stat_arr, axis=3)
            CCS_stat["median"][(regType, p, n)] = nparr.copy()
            kwargs = dict(n_sample=params["n_agents"], statistic0="median", statistic1="std")
            kwargs.update(dict(repeat=False))
            nparr[:, 3:, :] = bootstrap427(stat_arr, axis=3, **kwargs)
            CCS_stat["ste_median"][(regType, p, n)] = nparr.copy()
            # median = np.nanmedian(stat_arr, axis=3)[..., np.newaxis]
            # median = np.repeat(median, repeats=np.shape(stat_arr)[3], axis=3)
            # nparr[:, 3:, :] = np.nanmedian(np.abs(stat_arr - median), axis=3)
            # CCS_stat["mad"][(regType, p, n)] = nparr.copy()

        print(f"loop {i}: {params['beta_classes'][i]} has {counter} jobs")
        fname = set_dir427() + f"\\output\\{sub_folder_name}\\"
        mkdir_p(fname)
        fname += f"{CCS_key}_stat_{CCS_type}_{params['key_class']}_{params['beta_classes'][i]}"
        fname += f"_{params['n_agents']:d}"
        np.save(fname, CCS_stat)

    return 0


if __name__ == "__main__":
    main_CCS_stat(CCS_key)
