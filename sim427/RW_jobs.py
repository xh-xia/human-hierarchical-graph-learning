"""
This is to communicate between the simulator (RW_Graph_Class) and signac
Created: Wednesday, ‎March ‎24, ‎2021, ‏‎2:49:32 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import RW_Graph_Class
from flow import FlowProject # signac-flow library

class Project(FlowProject):
    pass

# defining labels to indicate what jobs (and hence data) the state point data container has
@Project.label
def RW_done(job):
    return ('GLsim_data' in job.data)

@Project.label
def CCS_done(job):
    return ('CCS' in job.data)

# @Project.label
# def CCS_stat_done(job):
#     return ('CCS_stat' in job.data)


@Project.operation # for running the main simulations (outputing count matrices)
@Project.post(RW_done)
def RW_run(job):
    GLsim_object=RW_Graph_Class.GLsim(**job.sp)
    job.data['GLsim_data']=GLsim_object.walks()

#comment out CCS_compute because the np.allclose doesn't work in here for some reason
@Project.operation
@Project.pre(RW_done)
@Project.post(CCS_done)
def CCS_compute(job):
    with job.data:
        counts_me = job.data['GLsim_data']['counts_me'][:,:,:] # load this 3D nparr into memory
    job.data['CCS'] = RW_Graph_Class.CCS(counts_me,job.sp.regType,job.sp.p,job.sp.n,job.sp.seed)

# @Project.operation
# @Project.pre(CCS_done)
# @Project.post(CCS_stat_done)
# def CCS_stat_compute(job):
#     job.data

if __name__ == '__main__':
    Project().main()
