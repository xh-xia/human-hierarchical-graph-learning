"""
This is to communicate between the simulator (RW_Graph_Class) and signac
Created: Wednesday, March 24, 2021, 2:49:32 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

from flow import FlowProject  # signac-flow library

import RW_Graph_Class

"""
pre/post conditions:
https://docs.signac.io/en/latest/flow-project.html#conditions

in command line (in sim427 dir):
[check status]
python RW_jobs.py status

[to run jobs]
:: -o RW_run           | run only RW_run operation
:: --ignore-conditions all  | {none,pre,post,all} config eligibility of jobs to run
:: --progress               | show progress bar
:: CCS_compute has pre cond of RW_done (i.e., RW_run operation has to be done first)
python RW_jobs.py run -o RW_run --ignore-conditions none --progress
python RW_jobs.py run -o CCS_compute --ignore-conditions none --progress
python RW_jobs.py run -o CCPS_compute --ignore-conditions post --progress  # rerun/overwrite
python RW_jobs.py run -o CCPS_compute --ignore-conditions post -p 8 --progress  # rerun/overwrite
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


@Project.label
def CCPS_done(job):
    return "CCPS" in job.data


@Project.label
def CCPS2_done(job):
    return "CCPS2" in job.data


@Project.label
def regType_is_0_1_3(job):
    return job.sp.regType in [0, 1, 3]


# @Project.label
# def CCS_stat_done(job):
#     return ('CCS_stat' in job.data)


@Project.operation  # for running the main simulations (outputing count matrices)
@Project.post(RW_done)
def RW_run(job):
    GLsim_object = RW_Graph_Class.GLsim2(**job.sp)
    job.data["GLsim_data"] = GLsim_object.walks()


@Project.operation
@Project.post(RW_done)
def RW_run_alt(job):  # alt model: two-point error distribution
    GLsim_object = RW_Graph_Class.GLsim3(**job.sp)
    job.data["GLsim_data"] = GLsim_object.walks()


@Project.operation
@Project.pre(RW_done)
@Project.post(CCS_done)
def CCS_compute(job):
    with job.data:
        counts_me = job.data["GLsim_data"]["counts_me"][:, :, :]  # load this 3D nparr into memory
    job.data["CCS"] = RW_Graph_Class.CCS(counts_me, job.sp.regType, job.sp.p, job.sp.n, job.sp.seed)


@Project.operation
@Project.pre(RW_done)
@Project.pre(regType_is_0_1_3)
@Project.post(CCPS_done)
def CCPS_compute(job):
    with job.data:
        counts_me = job.data["GLsim_data"]["counts_me"][:, :, :]  # load this 3D nparr into memory
    kwargs = dict(ccps_type=1, analytic_comp=False, scale=1)
    job.data["CCPS"] = RW_Graph_Class.CCPS(counts_me, job.sp.regType, job.sp.p, job.sp.n, **kwargs)


@Project.operation
@Project.pre(RW_done)
@Project.pre(regType_is_0_1_3)
@Project.post(CCPS2_done)
def CCPS2_compute(job):
    with job.data:
        counts_me = job.data["GLsim_data"]["counts_me"][:, :, :]  # load this 3D nparr into memory
    kwargs = dict(ccps_type=2, analytic_comp=False, scale=1)
    job.data["CCPS2"] = RW_Graph_Class.CCPS(counts_me, job.sp.regType, job.sp.p, job.sp.n, **kwargs)


# @Project.operation
# @Project.pre(CCS_done)
# @Project.post(CCS_stat_done)
# def CCS_stat_compute(job):
#     job.data

if __name__ == "__main__":
    Project().main()
