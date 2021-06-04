"""
process csv file downloaded from Heroku server.
Created: Tuesday, ‎May ‎4, ‎2021, ‏‎11:14:58 AM (EDT)
@author: Xiaohuan (Pixel) X.
"""
import pandas as pd
from math import log as log_e  # base e logarithm
from math import isnan
import base64, zlib, json  # to parse data string
from sys import exit
from utility427.helper427 import set_dir427, mkdir_p  # import helper functions from my own script

# import numpy as np


def main():
    """
    which_db:
    1: full n-back data; ver = "22.427"
    2: full 1500 step walkdata; ver = "22.427"
    8,9: test on demographic data; ver = "23.427"
    10: with break, and data collection should be fixed; ver = "24.427"
    11: full n-back with break, demographics; ver = "24.427"
    12: have edgelv & hamiltonian in db now; should be about complete; ver = "25.427"
        when we change tablename, only change SUFFIX in custom_models.py and config.txt
    """
    which_db = 12  # which database we want to parse
    ver = "25.427"
    cwd = set_dir427(return_cwd=True)
    folder_path = "\\input\\database\\"
    csvs = ["experiment", "participants", "walkdata"]
    df_dict = dict()
    csv_dict = get_csv_name(cwd + folder_path, csvs, which_test=which_db)
    for name in csvs:
        df_dict[name] = pd.read_csv(csv_dict[name], index_col=0)
        # print(df_dict[name].columns.values)

    get_subjects_data(df_dict, db_ver=[which_db, ver], live_only=False, complete=False)


def get_subjects_data(df_dict, db_ver, live_only=True, complete=True):
    """
    Args:
    --------------
    df_dict (dict): created in main, its value is pd.df:
        df_participants (pd.df):
            participant df: df_dict["participants"] (from participants.db, 1 of 3 db files)
        df_dict["walkdata"] (pd.df): contains walk data indexed by walk_id (1,2,3,...)
            walk_one: node index of the graph
            edgelv: 0 is starting node; -1 is self-loop; 1 is finest level
            is_hamiltonian: if the node is part of the hamiltonian walk
        df_dict["experiment"] (pd.df):
            4 fields: ['uniqueId' 'finger_mapping' 'walk_id' 'bonus_info']
                uniqueId (str) = UID = workerid:assignmentid
                finger_mapping (str):
                    will be converted to nested list of int (1=True, 0=False)
                    index maps to node index of the graph
    db_ver (list): db_ver = [which_db, ver]
        which_db (int): which database we want to parse
        ver (string):
            version of experiment, used to distinguish what rows are from current experiment
    live_only (bool):
        True: only consider data from live (HIT on mturk) data
    complete (bool):
        True: only write preprocessed data of subjects who completed the whole experiment

    Return:
    --------------
    data_dict (nested dict):
        key: string of UID (index of df_participants)
        val: dict:
            key:
                from "hitid" to "mode" with corresponding val in df_participants
                (note: if there is same key inside json_dict, use val from there)
    Output:
    -------
    task_all (csv file):
        name convention: task_all_dbver{dbver}
            dbver: {which_db}-{ver}
    """
    """ criterion for current data
    if codeversion == ver:
        it is our current data
    """
    which_db, ver = db_ver  # unpack list
    df_participants = df_dict["participants"]
    df_walkdata = df_dict["walkdata"]
    print(f"DEBUG df_walkdata['walk_one']: {df_walkdata['walk_one']}")
    is_current = df_participants["codeversion"] == ver
    df = df_participants[is_current]
    """ from here on we work with modified copy of df_participants """
    if live_only:
        df = df[df["mode"] == "live"]
    # there will be SettingWithCopyWarning, e.g., below line
    # ignore it, since we deliberately want to modify this copy (i.e., df)
    df.rename(columns={"cond": "condition"}, inplace=True)  # change it to match that in datastring
    fieldnames = list(df.keys())
    # print(f"DEBUG\n original fieldnames (w/ cond renamed): {fieldnames}")
    fieldnames = fieldnames[2:-1]
    fieldnames += ["currenttrial"]  # unique to datastring
    data_dict = dict()
    for UID in df.index:  # UID = workerid:assignmentid; loop through each participant
        tempdf = df_dict["experiment"].loc[df_dict["experiment"]["uniqueId"] == UID]
        if tempdf.empty:
            raise Exception(f"couldn't find UID={UID} in experiment.db.")
        if len(tempdf.index) > 1:
            raise Exception(f"UID={UID} has more than 1 row in experiment.db.")
        walk_id = tempdf["walk_id"].iloc[0]  # np.int64; a diff walk_id should mean diff subject
        bonus_info = tempdf["bonus_info"].iloc[0]
        finger_mapping = tempdf["finger_mapping"].iloc[0][2:-2]  # str; get rid of outer "[[]]"
        # convert str to nested list
        finger_mapping = finger_mapping.split("], [")
        for i in range(len(finger_mapping)):
            finger_mapping[i] = finger_mapping[i].split(", ")
            finger_mapping[i] = [int(x == "true") for x in finger_mapping[i]]  # convert str to int
        # print(f"DEBUG finger_mapping: {finger_mapping}")
        print(f"DEBUG bonus_info: {bonus_info}")

        if isinstance(df.loc[UID]["datastring"], float):  # assume if it's float, it's np.nan
            print(f"{UID}'s datastring field is NaN (np.nan).")
            continue
        # print(df.loc[UID]["hitid":"mode"])  # DEBUG: look at row outside of datastring field
        # get datastring (i.e., core experimental data we collected from the participants)
        json_dict = json.loads(df.loc[UID]["datastring"])  # 'tis a dict
        lv1_keys = list(json_dict.keys())  # list of keys (non-recursive) in json_dict
        data_dict[UID] = dict()
        for key in fieldnames:
            if key in lv1_keys:  # use val in json_dict if it has the key
                data_dict[UID][key] = json_dict[key]
            else:  # unique to outside of datastring, use outside val
                data_dict[UID][key] = df.loc[UID][key]
        """ 3 keys in json_dict:
        questiondata (dict):
            it contains most of the data we want. 4 big fields:
            'compressed_task_data': actual walk data
                phase, stage, trial, node, correct, nTries, rt, response, target, keyCode, event, query
            'compressed_quiz_data': not very useful, only quiz data as indicated in the name
            'completed_demo', 'completed_walk_one' (bool)
        data (list of dict): generally useless info on instruction pages
            has 4 fields (in each dict): current_trial, dateTime, trialdata, uniqueid
            it contains viewTime and information on viewing instruction pages
            two rows has phase="postquestionnaire" and status="begin" / "submit"
            trialdata is a heterogeneous dict
                meaning each row of trialdata has slightly different keys
        eventdata (list of dict): generally useless
            has 4 fields (eventtype, value, timestamp, interval), but all useless
            like eventtype is either initialized or window_resize
            it only has 5 rows (i.e., 5 items in the list)
        """
        questiondata = json_dict["questiondata"]  # 'tis a dict

        pd_set_max_display(400, 8)
        # DEBUG_show_fields(json_dict)
        # demographics & free response
        data_dict[UID]["Resp"] = parse_response_data(questiondata)

        completion = get_completion(questiondata)
        data_dict[UID]["df_n_back"] = None
        data_dict[UID]["df_task"] = None
        if completion["n_back_DONE"]:  # n-back
            df_n_back = make_df_n_back(questiondata["n-back"])
            data_dict[UID]["df_n_back"] = parse_n_back(df_n_back, walk_id, start_idx=1)

        if completion["walk_b_DONE"]:  # serial response
            data_dict[UID]["df_task"], data_dict[UID]["stat_dict"] = parse_task_data(
                questiondata,
                walk_id,
                first_trial_num=1,
                start_trial=1 / 3,
                first_trial_is_zero=False,
            )

    if complete:  # remove subjects who did not complete the whole experiment
        for k in data_dict:
            if (data_dict[k]["df_n_back"] is None) or (data_dict[k]["df_task"] is None):
                del data_dict[k]
        n_back_all = pd.concat([data_dict[k]["df_n_back"] for k in data_dict], ignore_index=True)
        task_all = pd.concat([data_dict[k]["df_task"] for k in data_dict], ignore_index=True)
    else:  # subjects in n_back_all are not necessarily the same as those in task_all
        n_back_all = pd.concat(
            [data_dict[k]["df_n_back"] for k in data_dict if data_dict[k]["df_n_back"] is not None],
            ignore_index=True,
        )
        task_all = pd.concat(
            [data_dict[k]["df_task"] for k in data_dict if data_dict[k]["df_task"] is not None],
            ignore_index=True,
        )
    # n-back: save n_back_all as csv
    save_df_to_csv(n_back_all, "n_back", f"n_back_all_dbver{which_db}-{ver}", show_df=False)
    # serial response: save task_all as csv
    save_df_to_csv(task_all, "serial_response", f"task_all_dbver{which_db}-{ver}", show_df=False)

    return data_dict


# region: parser functions
"""
parser functions
"""


def get_completion(questiondata):
    """find completion status in questiondata (i.e., json_dict["questiondata"])
    for walk is really straightforward: completed_walk_one_b==True means completion basically

    Return:
    -------
    completion (dict): n_back_DONE, walk_a_DONE, walk_b_DONE
        note: only walk_b_DONE==True means they completed the n-back and all graph learning tasks
        if n_back itself is completed, pd.df make_df_n_back() creates should have 390 rows.
    """
    completion = {"n_back_DONE": False, "walk_a_DONE": False, "walk_b_DONE": False}
    if "n-back" in questiondata:  # not the most efficient way
        completion["n_back_DONE"] = len(make_df_n_back(questiondata["n-back"]).index) == 390
    if "completed_walk_one_a" in questiondata:
        completion["walk_a_DONE"] = questiondata["completed_walk_one_a"]
    if "completed_walk_one_b" in questiondata:
        completion["walk_b_DONE"] = questiondata["completed_walk_one_b"]
    return completion


def make_df_n_back(n_back_list):
    """create n_back df
    it's not the whole n_back data,
    we will only use the filtered one (i.e., trial_id=="stim"),
    one used for beta estimation for the max entropy model
    some info (learned by looking at data) about fieldnames:
    ----------
    trial_id: type of row: instruction/stim/*_intro/delay_text/questions/end
    trial_type: type of actual trial: poldrack-categorize/single-stim/text
    trial_index: same as index of df_n_back
    exp_id: "n-back"
    correct: whether one got it right: True/False/NaN
    stimulus: html text string
    exp_stage: stage of exp: NaN/practice/test
        (for test, it appears in both ctrl_intro and test_intro groups)
    stim: letter one sees
    target: target response (that will yield correct)
    """
    n_back_data = decompress_pako(n_back_list)  # 'tis a list of dict
    df_n_back = pd.DataFrame(n_back_data)
    print("DEBUG: n_back keys: \n", df_n_back.columns.values)
    # is_stim = df_n_back["trial_id"] == "stim"
    # df_n_back = df_n_back[is_stim]
    # print(df_n_back[["trial_id", "trial_type", "correct", "stim", "target"]])
    return df_n_back


def parse_n_back(df_n_back, walk_id, start_idx=0):
    """
    each section
        trial_index: unique index starting from 0, basically count of all rows
        correct (bool): correctness of response; NaN if it is not an n-back trial
        trial_id: starts with "*_intro" and ends with the last "stim"
        key_press (float): 37 (left; match/positive trial) or 40 (down; no match/negative trial)
        stim: current letter
        target: letter shown n-step back
        exp_stage: NaN, practice, test
        delta_t: as defined in [MentalErrors](https://doi.org/10.1038/s41467-020-15146-7) Fig. 5a

    Return:
    -------
    processed df_n_back with a subset of fields

    Data Processing Rules:
    ----------------------
    1) add "id"=walk_id field
    2) leave only index "section"=test_# rows
    3) calculate delta_t
    4) remove negative trials (i.e., leave only field "key_press"=37)
    5) remove delta_t < 0 (those are trials that has letter not seen n-back before)
    """
    df_n_back["target"].replace("", None, inplace=True)
    is_na = df_n_back["target"].isna()
    df_n_back["target_na"] = is_na  # add a field to show if there is no target
    is_section_start = df_n_back["trial_id"].str.contains("_intro")
    df_n_back["section_start"] = is_section_start  # add a field to show if it is section start
    df_n_back.rename_axis("idx", axis="rows", inplace=True)  # rename index title

    df_n_back.set_index(make_MultiIndex(df_n_back, start_idx), inplace=True)
    print(f"DEBUG df_n_back.columns.values: {df_n_back.columns.values}")
    # 1) add walk_id
    df_n_back["id"] = [walk_id] * len(df_n_back.index)
    # 2) filter according to section index to have only test_# groups left
    df_n_back = df_n_back[df_n_back.index.get_level_values("section").str.contains("test_")]
    # 3) apply get_recency per group
    def nonce1(df):  # function apply to groupby.apply() has to take in df as 1st arg
        n = df.index.get_level_values("n").values[0]  # all rows have same index
        df["delta_t"] = get_recency(df["stim"].values, n=int(n), ignore_case=True)
        return df

    df_n_back = df_n_back.groupby(level="section").apply(nonce1)
    # 4) only use positive trials
    def nonce2(flt):
        if isnan(flt):
            return False
        else:
            return int(flt) == 37

    df_n_back = df_n_back[df_n_back["key_press"].apply(nonce2)]
    # 5) remove delta_t < 0 trials
    df_n_back = df_n_back[df_n_back["delta_t"] >= 0]
    # get relevant fields to show
    fieldnames = ["trial_index", "correct", "stim", "delta_t", "id"]
    # print(f"DEBUG df_n_back[fieldnames]:\n{df_n_back[fieldnames]}")
    return df_n_back[fieldnames]


def make_MultiIndex(df_n_back, start_idx):
    """
    NOTE in pandas, leftmost index has level 0, and then 1, and so on
    NOTE in my convention, leftmost is coarsest hierarchy, hence has highest level number
    returns a MultiIndex object
    lv1 idx: unique up to section index; start from <start_idx>
    lv2 idx: section index, grouping trials based on trial_id/section_start
    lv3 idx: n in n-back (for test and ctrl, the rest is -1)
    for those whose trial_id != "stim", lv2 idx is "nah"
    I keep generality of flag here, in that I could just use bool type parity
    but keeping generality means it is essentially a "switch" flag
    """
    lv1_idx = [start_idx] * len(df_n_back.index)
    lv2_idx = ["nah"] * len(df_n_back.index)
    lv3_idx = [-1] * len(df_n_back.index)
    flag = 0  # 1: in a section; 0: not in a section
    count_ctrl = 0  # number of control_intro
    count_test = 0  # number of test_intro
    temp = df_n_back.iloc[0]["trial_id"]  # initialize current lv2 idx
    n = 0  # initialize current n of n-back
    flag_count_n = True  # if we are counting n of n-back
    for i in range(1, len(df_n_back.index)):  # start from second item in the list
        if df_n_back.iloc[i]["trial_id"] != df_n_back.iloc[i - 1]["trial_id"]:  # trial_id changes
            if "stim" in df_n_back.iloc[i - 1 : i + 1]["trial_id"].values:
                flag = (flag + 1) % 2  # flag changes (flips in this case)
                flag_count_n = True  # reset whenever <flag> changes
                n = 0  # ditto
        else:
            lv1_idx[i] = lv1_idx[i - 1] + 1  # trial_id unchanged, hence +1 in index
        if flag == 1:
            if temp == "control_intro":
                count_ctrl += 1
                temp = "ctrl_" + str(count_ctrl)
            elif temp == "test_intro":
                count_test += 1
                temp = "test_" + str(count_test)
            lv2_idx[i] = temp
            if flag_count_n:
                if not isinstance(
                    df_n_back.iloc[i]["target"], str
                ):  # NaN (float) or None (NoneType)
                    n += 1
                else:  # letter (str); done counting
                    flag_count_n = False
                    for j in range(0, n + 1):  # update n for trial i as well as trials counting n
                        lv3_idx[i - j] = n
            else:  # not in counting phase
                lv3_idx[i] = n  # so we just update using updated n
        elif flag == 0:
            temp = df_n_back.iloc[i]["trial_id"]
        else:
            raise Exception("<flag> not in [0,1] is not implemented.")
    idx_list = [lv3_idx, lv2_idx, lv1_idx]
    MultiIndex = pd.MultiIndex.from_arrays(idx_list, names=("n", "section", "idx"))
    return MultiIndex


def parse_response_data(questiondata):
    """this is to parse free_response data and such
    excluding random walk data, which is processed by parse_task_data()

    Arg:
    ----
    questiondata (dict):
        it contains psychological test battery and demographic info and such
        free_response data should be contained in below keys:
        "feedback", "nature", etc.

    Return:
    -------
    Resp (dict): nested
        1st lv key is the type of response
        2nd lv key (in each 1st lv val):
            val is just what it has in questiondata
    """
    # ignore keys with below prefixes
    prefixes = ("compressed_", "completed_", "n-back")
    feedback_keys = ("engagement", "difficulty", "feedback", "nature")
    Resp = dict()
    Resp["demographics"] = dict()
    Resp["feedback"] = dict()  # engagement, difficulty, feedback, nature
    for key, val in questiondata.items():
        if not key.startswith(prefixes):
            if key in feedback_keys:
                Resp["feedback"][key] = val
            else:
                Resp["demographics"][key] = val
    return Resp


def parse_task_data(
    questiondata, walk_id, first_trial_num=0, start_trial=None, first_trial_is_zero=True
):
    """
    Args:
    ----
    questiondata (dict):
        use only below keys (for random walk data):
        "compressed_task_data", "completed_walk_one_a", "completed_walk_one_b"
    walk_id (int): walk_id of current UID
    first_trial_num (int): convention of the number of first trial
    start_trial (int/float): where we start the trial
        if >=1: subset trials start_trial, start_trial+1, ..., end
        if <1: subset trials from trial round(tot_trial*start_trial) + x
        where tot_trial is total number of trials; x is the first trial number in the walk
    first_trial_is_zero (bool):
        if True, will throw error if the first trial number in the walk is not 0

    Return:
    -------
    df_task (df): modified df_task
        if there is an incorrect trial,
        there has to be a correct trial following it with same trial number
        since they have to press the correct key to advance to the next trial
    stat_dict (dict): statistics


    Data Processing Rules:
    ----------------------
    1) remove demo trials
    2) reset trial number using <first_trial_num>; assume the first row after step 1 is first trial
    3) remove correct trial that follows from incorrect one, this may make rt smaller
    i.e., remove rows with nTries>=2
    e.g., trial 10 incorrect followed by trial 10 correct, we ignore the second trial.
    4) add recency (if no occurrence, recency=0)
    e.g., if trial 5 has node 10, trial 6 has node 11, trial 7 has node 10
    then recency at trial 7 is 2.
    5) add log_trial (essentially linearizing rt~trial)
    6) subset trials starting from <start_trial>
    assume there is no missing trial number
    7) add walk_id (id) field
    """
    ### get random walk data
    task_data = decompress_pako(questiondata["compressed_task_data"])  # 'tis a list of dict
    df_task = pd.DataFrame(task_data)
    stat_dict = dict()
    # 1) remove demo trials
    df_task = df_task[df_task["stage"] != "demo"]
    # print(f"DEBUG df_task.iloc[0]:\n{df_task.iloc[2]}")
    # 2) reset trial number
    if df_task.iloc[0]["trial"] != first_trial_num:
        diff = first_trial_num - df_task.iloc[0]["trial"]
        df_task["trial"] = df_task["trial"] + diff
    # 3) remove correct trials following incorrect ones
    df_task = df_task[df_task["nTries"] == 1]
    stat_dict["accuracy"] = df_task["correct"].mean()  # get accuracy after step 2
    # 4) add recency field
    df_task["recency"] = get_recency(df_task["node"].values, n=0)
    # 5) add log_trial field
    df_task["log_trial"] = df_task["trial"].apply(log_e)
    # 6) subset trials
    first_trial = df_task.iloc[0]["trial"]  # first trial number in the walk
    last_trial = df_task.iloc[-1]["trial"]  # last trial number in the walk
    if first_trial_is_zero:
        if first_trial != 0:
            raise Exception("trial number doesn't start from 0.")
    if start_trial is None:
        start_trial = first_trial
    if start_trial >= 1:
        if start_trial < first_trial or start_trial > last_trial:
            raise ValueError(
                "<start_trial> is starting trial number, which has to be included in trial number of data."
            )
        filt = df_task["trial"] >= start_trial
    elif 0 <= start_trial < 1:
        start_trial = round((last_trial - first_trial + 1) * start_trial) + first_trial
        filt = df_task["trial"] >= start_trial
    else:
        raise ValueError("<start_trial> is not a number that is >=0.")
    df_task = df_task[filt]
    # 7) add walk_id (id) field
    df_task["id"] = [walk_id for _ in range(len(df_task.index))]

    fieldnames = ["id", "stage", "trial", "correct", "node", "recency", "nTries", "rt"]
    fieldnames += ["target", "is_hamiltonian", "edgelv", "log_trial"]
    # print(f"DEBUG df_task:\n{df_task[fieldnames]}")
    # print(f"DEBUG stat_dict: {stat_dict}")
    return df_task[fieldnames], stat_dict


# endregion: parser functions

# region: utility functions
"""
utility functions
"""


def get_recency(stims, n=0, ignore_case=True):
    """this is generalized recency
        n=0: normal recency
        n>0: n-back delta t (i.e., target(at n-back) - memory)

    Arg:
    ----
    stims (list-like): arr of node indices (doesn't need to be node indices, can be char for n-back)
    n (int): n in n-back
    ignore_case (bool): if True, convert char in nodes to lowercase
        only convert if element is a str

    Return:
    -------
    recency (list-like): arr of (lagged) recency
        recency:= number of trials since it last (lagged by n) occurs
            -n: if no occurrence (n>=0)
            =0: exactly matched (target=memory; n>0)
            >0: normal recency (n=0)
            >0: lagged matched (memory lagged behind target by recency; n>0)
        nodes: lagged nodes
        shift: current nodes; used to find lagged recency
        e.g., nodes = [ 0, 0, 4, 2, 7, 4,0] | n=1
            shift = [ 0,0, 4, 2, 7, 4, 0]
          recency = [-1,0,-1,-1,-1, 2, 4]
                  nodes = ["a", "c", "c", "b", "a", "a", "a"] | n=2
        shift = ["a", "c", "c", "b", "a", "a", "a"]
      recency = [ -2,  -2,  -2,  -2,   2,   3,   0]
                  nodes = ["a", "c", "c", "b", "a", "a", "a"] | n=1
             shift = ["a", "c", "c", "b", "a", "a", "a"]
            recency = [-1,  -1,   0,  -1,   3,   0,   0]
                  nodes = ["a", "c", "c", "b", "a", "a", "a"] | n=0
                  shift = ["a", "c", "c", "b", "a", "a", "a"]
                recency = [  0,   0,   1,   0,   4,   1,   1]
    difference between n=0 and n>0 lies in the position of "counter[nodes[i - n]] = 0"
    Performance:
    ------------
    TL;DR:
        int or char entry doesn't matter, number of unique element makes a difference
        ignore_case=True will make it a bit slower
    stims = [0, 2, 2, 1, 0, 0, 0] * 1000000/2000000
    stims = ["a", "c", "c", "b", "a", "a", "a"] * 1000000/2000000
    ignore_case=False
    for len(stims)=7,000,000 | n=0 | n=1
        int entry: takes ~ 2.6 sec | ~ 3.0 sec
        char entry: takes ~ 2.7 sec | ~ 3.0 sec
    for len(stims)=14,000,000 | n=0 | n=1
        int entry: takes ~ 4.8 sec | ~ 5.6 sec
        char entry: takes ~ 5.1 sec | ~ 5.6 sec

    stims = ["A", "c", "C", "b", "a", "A", "a"] * 1000000
    for len(stims)=7,000,000 | n=0 | n=1
        ignore_case=False: takes ~ 3.7 sec | ~ 3.9 sec
        ignore_case=True: takes ~ 3.9 sec | ~ 4.2 sec
    """
    if not isinstance(n, int):
        raise ValueError(f"<n>={n} is not an integer")
    if n < 0:
        raise ValueError(f"<n>={n} is negative, which is not supported")
    if ignore_case:
        nodes = [None] * len(stims)
        for i in range(len(stims)):
            if isinstance(stims[i], str):
                nodes[i] = stims[i].lower()
            else:
                nodes[i] = stims[i]
    else:
        nodes = stims  # we won't change the value in nodes; so not a copy is fine

    counter = dict()  # initialize recency counter for each node; key is node, val is count
    recency = [-n] * len(nodes)
    if n == 0:
        for i in range(len(nodes)):
            for k in counter:  # increment all seen nodes' counters
                counter[k] += 1
            if nodes[i] in counter:  # current node is seen
                recency[i] = counter[nodes[i]]  # update recency
            counter[nodes[i]] = 0  # either create node count in dict or reset it
    else:
        for i in range(n, len(nodes)):
            # nodes[i] is essentially the <shift> in example above
            # hence nodes[i-n] is <nodes> in example above; which is called lagged node
            for k in counter:  # increment all seen nodes' counters
                counter[k] += 1
            counter[nodes[i - n]] = 0  # either create lagged node count in dict or reset it
            if nodes[i] in counter:  # current node is seen in lagged nodes counter
                recency[i] = counter[nodes[i]]  # update recency

    return recency


def DEBUG_show_fields(json_dict):
    """
    json_dict = json.loads(df.loc[UID]["datastring"])  # 'tis a dict
    """
    print(end="\n")
    print(f"DEBUG: keys in questiondata:\n {list(json_dict['questiondata'].keys())}", end="\n\n")
    print(f"DEBUG: keys in data:\n {list(json_dict['data'][0].keys())}", end="\n\n")
    print(f"DEBUG: keys in eventdata:\n {list(json_dict['eventdata'][0].keys())}", end="\n\n")


def decompress_pako(datastring):
    """this is modified from decompress_pako() in custom.py
    Decompress json data that we compressed in the browser with paco.

    Assumes data was then base64-encoded:

    btoa(pako.deflate(JSON.stringify(data), { to: 'string' }));

    Parameters
    ----------
    datastring : string
        base64-encoded json data to decompress

    Returns
    -------
    dict
        JSON-decoded and decompressed data

    """
    return json.loads(zlib.decompress(base64.decodebytes(datastring.encode())))


def pd_set_max_display(row=15, col=4):
    pd.set_option("display.max_rows", row)
    pd.set_option("display.max_columns", col)


def get_csv_name(cwd, csvs, which_test=0):
    """
    Args:
        csvs (list): e.g., ["experiment", "participants", "walkdata"]
        which_test (int): int val leading the name of the folder which contains .csv files
    """
    import os

    temp = os.listdir(cwd)
    for name in temp:
        if int(name.split(" ")[0]) == which_test:
            temp = name
            break
    else:  # run if nobreak
        raise Exception("<which_test> index is not found in given cwd.")
    # we found the csv folder, now get the csv dir
    cwd += temp + "\\"
    temp = os.listdir(cwd)
    csv_dict = dict()
    for name in temp:
        if name.split("_")[0] in csvs:
            # attach leading dir onto the name
            csv_dict[name.split("_")[0]] = cwd + name

    return csv_dict


def save_df_to_csv(df, folder_name, fname, show_df=False):
    """
    Args:
    -----
    df (pd.df): pandas DataFrame
    folder_name (str): name of folder containing the csv file
    fname (str): name of the csv file; don't need to contain "csv" extension
    show_df (bool): if True, show df via DEBUG
    """
    if show_df:
        print(f"DEBUG df:\n{df}")
    dir_ = set_dir427(return_cwd=True) + f"\\output\\{folder_name}\\"
    mkdir_p(dir_)  # creates output\<folder_name> folder if it doesn't exist
    dir_ += fname + ".csv"
    df.to_csv(dir_, sep=",", quotechar='"')  # save to csv file


# endregion: utility functions

if __name__ == "__main__":
    main()