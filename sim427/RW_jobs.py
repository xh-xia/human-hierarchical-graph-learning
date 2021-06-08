"""
This is to communicate between the simulator (RW_Graph_Class) and signac
Created: Wednesday, ‎March ‎24, ‎2021, ‏‎2:49:32 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

from flow import FlowProject  # signac-flow library

import RW_Graph_Class

"""in command line (in sim427 dir):
[check status]
    python RW_jobs.py status

[to run jobs]
    :: -o RW_run           | run only RW_run operation
    :: --ignore-conditions all  | {none,pre,post,all} config eligibility of jobs to run
    :: --progress               | show progress bar
    :: CCS_compute has pre cond of RW_done (i.e., RW_run operation has to be done first)
    python RW_jobs.py run -o RW_run --ignore-conditions none --progress
    python RW_jobs.py run -o CCS_compute --ignore-conditions none --progress
    :: or run both in one line
    python RW_jobs.py run --ignore-conditions none --progress
"""

class Project(FlowProject):
    pass


# defining labels to indicate what jobs (and hence data) the state point data container has
@Project.label
def RW_done(job):
    return "GLsim_data" in job.data


@Project.label
def CCS_done(job):
    return "CCS" in job.data


# @Project.label
# def CCS_stat_done(job):
#     return ('CCS_stat' in job.data)


@Project.operation  # for running the main simulations (outputing count matrices)
@Project.post(RW_done)
def RW_run(job):
    GLsim_object = RW_Graph_Class.GLsim(**job.sp)
    job.data["GLsim_data"] = GLsim_object.walks()


# comment out CCS_compute because the np.allclose doesn't work in here for some reason
@Project.operation
@Project.pre(RW_done)
@Project.post(CCS_done)
def CCS_compute(job):
    with job.data:
        counts_me = job.data["GLsim_data"]["counts_me"][:, :, :]  # load this 3D nparr into memory
    job.data["CCS"] = RW_Graph_Class.CCS(counts_me, job.sp.regType, job.sp.p, job.sp.n, job.sp.seed)


# @Project.operation
# @Project.pre(CCS_done)
# @Project.post(CCS_stat_done)
# def CCS_stat_compute(job):
#     job.data

if __name__ == "__main__":
    Project().main()
