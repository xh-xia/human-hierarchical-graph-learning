"""
This is to post-process the data from completed jobs in signac database.
Specifically, those whose CCS_done(job) is True.
what it does is simply find basic statistics for the CCS:
mean, standard deviation, standard error (std/sqrt(sample size))
Created: Thursday, ‎March ‎25, ‎2021, ‏‎10:30:09 AM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import signac as sn
from RW_utilities import *
from RW_Graph_Class import CCS
set_dir427(add_parent_to_path=True)
from CCS427 import mkdir_p

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
    #n_agents, key_class = 10, 'r'
    n_agents, key_class = 100, 'reg_n_p'
    project = sn.get_project()
    params = get_params(n_agents=n_agents, key_class=key_class)
    CCS_stat = dict()
    CCS_stat['mean'], CCS_stat['std'], CCS_stat['ste'] = dict(), dict(), dict()
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
                stat_arr[:,l,job.sp.beta_idx,job.sp.agentID] = temp[:,l]
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
    np.save(fname+'CCS_stat_{}_{:d}'.format(key_class,n_agents), CCS_stat)

    return 0


if __name__=="__main__":
    main_CCS_stat()
