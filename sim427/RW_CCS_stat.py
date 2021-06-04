"""
This is to post-process the data from completed jobs in signac database
Specifically, those whose CCS_done(job) is True
what it does is simply find basic statistics for the CCS:
mean, standard deviation, standard error (std/sqrt(sample size))
Created: Thursday, ‎March ‎25, ‎2021, ‏‎10:30:09 AM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import signac as sn
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utility427.helper427 import get_params, set_dir427, mkdir_p
from RW_Graph_Class import CCS
set_dir427(add_parent_to_path=True)

"""
CCS_stat (dict): 3 keys each corresponding to one statistic
'mean','std','ste', the value (dict) of them is of the same structure:
value[(regType,p,n)] (3D nparr): "[slice]: meaning"
    s stands for the s-th sample in counts tensor defined in RW_Graph_Class.py
    [s,0,:]: beta
    [s,1,:]: group size
    [s,2,:]: steps_sample (time stamps at the time of sampling = walk length)
    [s,3,:]: stat of the group having that beta for CCS at level 1
    ...
    [s,3+n-2,:]: stat of the group having that beta for CCS at level n-1
"""

def print_progress(counter,tot=26000):
    if counter % 200 == 0: print('Progresss: {}/{}'.format(counter,tot))

def main_CCS_stat():
    #np.seterr(all='raise') # set all runtime warning to raise errors
    is_operation = True
    #n_agents, key_class = 10, 'r' # ~ 6 sec
    n_agents, key_class = 100, 'reg_n_p' # ~ 160 sec
    CCS_type = 'mean' # 'mean' or 'std'
    if CCS_type=='mean':
        CCS_type_slice = 0
    else:
        CCS_type_slice = 1
    CCS_stat = dict()
    CCS_stat['mean'], CCS_stat['std'], CCS_stat['ste'] = dict(), dict(), dict()
    project = sn.get_project()
    params = get_params(n_agents=n_agents, key_class=key_class)
    n_sample = int(np.floor(params['steps_tot']/params['sample_period'])) # copied from RW_Graph_Class.py
    counter = 0 # to report progress
    for regType, p, n in params['pd']:
        job_criteria = {'key_class':params['key_classes'][0],'regType':regType,'p':p,'n':n,'n_agents':params['n_agents']}
        nparr = np.zeros((n_sample,3+n-1,len(params['beta_arr'])))
        stat_arr = np.zeros((n_sample,n-1,len(params['beta_arr']),params['n_agents']))
        for job in project.find_jobs(job_criteria):
            counter+=1
            print_progress(counter)
            nparr[:,0,job.sp.beta_idx] = job.sp.beta
            nparr[:,1,job.sp.beta_idx] = params['n_agents']
            with job.data:
                nparr[:,2,job.sp.beta_idx] = job.data['GLsim_data']['steps_sample'][:]
                if is_operation:
                    temp = job.data['CCS'][:]
                else:
                    # CCS_compute here (not as @operation)
                    temp = CCS(job.data['GLsim_data']['counts_me'][:],job.sp.regType,job.sp.p,job.sp.n, job.sp.seed)
            for l in range(n-1): # CCS level index; only up to n-2 (i.e., CCS level n-1)
                stat_arr[:,l,job.sp.beta_idx,job.sp.agentID] = temp[:,CCS_type_slice,l]
                # print('DEBUG: regType={},p={},n={},agentID={},beta_idx={},seed={}'\
                #       .format(regType, p, n, job.sp.agentID, job.sp.beta_idx, job.sp.seed))
        nparr[:,3:,:] = np.nanmean(stat_arr,axis=3)
        CCS_stat['mean'][(regType,p,n)] = nparr.copy()
        nparr[:,3:,:] = np.nanstd(stat_arr,axis=3)
        CCS_stat['std'][(regType,p,n)] = nparr.copy()
        nparr[:,3:,:] = np.nanstd(stat_arr,axis=3) / np.sqrt(params['n_agents'])
        CCS_stat['ste'][(regType,p,n)] = nparr.copy()
    print('Total number of jobs: {:d}'.format(counter))
    fname = 'output/'
    mkdir_p(fname)
    np.save(fname+'CCS_stat_{}_{}_{:d}'.format(CCS_type,key_class,n_agents), CCS_stat)

    return 0


if __name__=="__main__":
    main_CCS_stat()
    #main_CCS_stat_mp()




def CCS_stat_mp_Gwise(CCS_type_slice, project, params, n_sample, graph_type):
    """ assume is_operation=True
    calculate for a given graph (for multiprocessing purposes)
    since we are going for multiprocessing, the global counter won't work here
    note: this function returns a dictionary of dict as well, but reversed:
    CCS_stat only has 1 key, with 1 value which is a dict with 3 keys: 'mean', 'std', 'ste'
    e.g., CCS_stat[(regType,p,n)]['std']
    hence the key for CCS_stat is unique for each CCS_stat returned by the function
    """
    regType, p, n = graph_type # unpacking
    CCS_stat = dict()
    CCS_stat[(regType,p,n)] = dict()
    job_criteria = {'key_class':params['key_classes'][0],'regType':regType,'p':p,'n':n,'n_agents':params['n_agents']}
    nparr = np.zeros((n_sample,3+n-1,len(params['beta_arr'])))
    stat_arr = np.zeros((n_sample,n-1,len(params['beta_arr']),params['n_agents']))
    for job in project.find_jobs(job_criteria):
        nparr[:,0,job.sp.beta_idx] = job.sp.beta
        nparr[:,1,job.sp.beta_idx] = params['n_agents']
        with job.data:
            nparr[:,2,job.sp.beta_idx] = job.data['GLsim_data']['steps_sample'][:]
            temp = job.data['CCS'][:] # assume @operation
        for l in range(n-1): # CCS level index; only up to n-2 (i.e., CCS level n-1)
            stat_arr[:,l,job.sp.beta_idx,job.sp.agentID] = temp[:,CCS_type_slice,l]
    nparr[:,3:,:] = np.nanmean(stat_arr,axis=3)
    CCS_stat[(regType,p,n)]['mean'] = nparr.copy()
    nparr[:,3:,:] = np.nanstd(stat_arr,axis=3)
    CCS_stat[(regType,p,n)]['std'] = nparr.copy()
    nparr[:,3:,:] = np.nanstd(stat_arr,axis=3) / np.sqrt(params['n_agents'])
    CCS_stat[(regType,p,n)]['ste'] = nparr.copy()

    return CCS_stat

def main_CCS_stat_mp():
    """
    I consider this multiprocessing a failure.
    Not that it doesn't run.
    It runs, and seems to produce exactly same result (checked by plotting).
    It's a failure in terms of speed improvement because it makes it slower!!
    My test code did run significantly faster than single-processing.
    But in the case of CCS_stat? No, it's actually the same effect but reversed.
    Why?
    It's taking so long I don't even want to wait...
    I canceled it, it did not seem to finish within reasonable time...
    """
    #n_agents, key_class = 10, 'r' # much slower (~ 35 sec) than sp (single-processing)
    n_agents, key_class = 100, 'reg_n_p'
    CCS_type = 'mean' # 'mean' or 'std'
    if CCS_type=='mean':
        CCS_type_slice = 0
    else:
        CCS_type_slice = 1
    CCS_stat = dict()
    CCS_stat['mean'], CCS_stat['std'], CCS_stat['ste'] = dict(), dict(), dict()
    project = sn.get_project()
    params = get_params(n_agents=n_agents, key_class=key_class)
    n_sample = int(np.floor(params['steps_tot']/params['sample_period'])) # copied from RW_Graph_Class.py
    with concurrent.futures.ProcessPoolExecutor() as executor:
        dict_list = list(executor.map(CCS_stat_mp_Gwise, \
        repeat(CCS_type_slice),repeat(project),repeat(params),repeat(n_sample),params['pd']))
    # ↓ reverse it back
    for key in CCS_stat.keys():
        for item in dict_list: # the output after mp is a list of dict, so item is dict
            CCS_stat[key][list(item.keys())[0]] = item[list(item.keys())[0]][key]
    fname = 'output/'
    mkdir_p(fname)
    np.save(fname+'CCS_stat_{}_{}_{:d}'.format(CCS_type,key_class,n_agents), CCS_stat)

    return 0
