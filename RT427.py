import numpy as np
import numpy.linalg as npL
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm # Linear Mixed (Effects) Model (or LMM)
import pandas as pd
from scipy.signal import savgol_filter # Savitzky–Golay filter (aesthetic purposes)

def main_RT427():
    cd_2_Curio(show=1)
    dir_ME = 'Misc/' + 'data_Fig1' # mental error paper's data
    #dir_test = 'Misc/' + 'testt' + '.csv'
    MEstructArr = readNload427(dir_ME, vanilla = False)
    print(MEstructArr.dtype.names)
    #print(len(set(MEstructArr['ID']))) # 103 participants

    # ↓ is correct ↓
    bArr = np.invert(MEstructArr['error'].astype('?'))
    # ↓ is Modular ↓
    bArrM = np.logical_and(bArr,MEstructArr['graphType']=='modular')
    # ↓ is Lattice ↓
    bArrL = np.logical_and(bArr,MEstructArr['graphType']=='lattice')

    RTplots427(MEstructArr[bArr], graphType = '(modular & lattice)')
    RTplots427(MEstructArr[bArrM], graphType = '(modular)')
    RTplots427(MEstructArr[bArrL], graphType = '(lattice)')

    # ↓ is correct ↓
    bArr = np.invert(MEstructArr['error'].astype('?'))
    # ↓ exclude rt>2000ms (2s) ↓
    bArr = np.logical_and(bArr,MEstructArr['rt']<=2000)
    # ↓ is Modular ↓
    bArrM = np.logical_and(bArr,MEstructArr['graphType']=='modular')
    # ↓ is Lattice ↓
    bArrL = np.logical_and(bArr,MEstructArr['graphType']=='lattice')

    RTplots427(MEstructArr[bArr], graphType = '(modular & lattice; rt leq 2000ms)')
    RTplots427(MEstructArr[bArrM], graphType = '(modular; rt leq 2000ms)')
    RTplots427(MEstructArr[bArrL], graphType = '(lattice; rt leq 2000ms)')

    #MEdf = pd.DataFrame(MEstructArr)
    #print(MEdf.info())
    #result = LMM427(MEdf)
    #print(result)



def LMM427(df):
    ''' This is for Cross-Cluster Surprisal (CCS)
    Args:
        df: pandas DataFrame object
    Return:


    fixed effects: (reference see page 10 of Mental Errors paper)
    This is a global distribution
        1) natural quickening or reactions with time [trial*stage]
        2) biomechanical dependencies on target button combo [target]
        3) effects of recency on reaction times [recency]
        4) CCS [edgeType]
    group: each individual is a group [id]
    random effects (those that varies across groups):
    This is a group-wise variation
        1) [trial*stage]
        2) [recency]
        3) [edgeType]
        *4) [target]
    '''
    #formula = 'rt ~ np.log(trial)*C(stage) + recency + C(edgeType) + C(target)' # fixed effects
    #re_formula = '~ np.log(trial)*C(stage) + recency + C(edgeType)' # random effects
    formula = 'rt ~ np.log(trial)'
    re_formula = '~ 1 + recency'

    return mixedlm(formula, df, re_formula=re_formula, groups='ID').fit().summary()

def DB_empirical_adjacency(structArr, num=200):
    A = np.zeros((15,15),dtype=np.int32)
    for i in range(num):
        A[structArr['lastnode']-1,structArr['node']-1] = 1
    print(A)

def csv2dict(dir, dtypeArr = None):
    kargs = dict(encoding='utf-8', delimiter=',', names=True, dtype=dtypeArr)
    structArr = np.genfromtxt(dir, **kargs)
    return structArr

def MEedgeType(sA, i):
    '''
    sA = structArr
    Find edgeType using i-1 data:
    1) make sure structArr[i] and structArr[i-1] has the same:
        [experiment], [stage], [id] (subject id)
    2) structArr['trial'][i-1] = structArr['trial'][i] - 1
    within: edgeType=2 | between: edgeType=1 (Coarsest)
    otherwise use -1
    '''
    if i==0:
        return [-1, sA['node'][i], -1] # otherwise no last node, no edgeType

    lv1nodepair = [(1,15),(15,1),(5,6),(6,5),(10,11),(11,10)] # both directions
    if (sA['experiment'][i-1] == sA['experiment'][i])\
    and (sA['stage'][i-1] == sA['stage'][i])\
    and (sA['id'][i-1] == sA['id'][i]):
        if sA['trial'][i-1] == (sA['trial'][i]-1):
            # return node-node pair (v_(i-1),v_i) and edgeType
            nodepair = (sA['node'][i-1], sA['node'][i])
            if nodepair in lv1nodepair: # between
                return [nodepair[0], nodepair[1], 1]
            else: # within
                return [nodepair[0], nodepair[1], 2]
    return [-1, sA['node'][i], -1] # otherwise no last node, no edgeType

def get_new_structArr(structArr):
    ''' This is for Cross-Cluster Surprisal (CCS); it trims structArr and make categorical variables
    Args:
        structArr is numpy structured array, with field names:
          'experiment', 'stage', 'graph', 'id', 'trial', 'rt', 'node'
        , 'isHamiltonian', 'target_1', 'target_2', 'target_3', 'target_4', 'target_5'
        , 'error', 'recency'
    Return:
        a different structArr that serves our purposes, with fields:
        'rt': numerical ('<i4')
        'trial': numerical ('<i4')
        'recency': numerical ('<i4')
        ---------- ---------- ---------- ----------
        'stage': categorical ('<i4')
        'ID': categorical ('<i4')
        'target': categorical ('<U40')
        'graphType': categorical ('<U40')
        'lastnode': categorical ('<i4')
        'node': categorical ('<i4')
        'edgeType': categorical ('<i4')
        'error': binary ('<i4')

    '''
    # ↓ is Random Walk ↓
    bArr = np.invert(structArr['isHamiltonian'].astype('?')) # cast to boolean (data_Fig1 is Eulerian only)
    # ↓ only stage 1 ↓
    bArr = np.logical_and(bArr,structArr['stage']==1)
    # ↓ is also correct ↓
    #bArr = np.logical_and(bArr, np.invert(structArr['error'].astype('?')))
    # ↓ is also Modular ↓
    #bArr = np.logical_and(bArr, structArr['graph']=='modular')
    # ↓ exclude rt>2000ms (2s) ↓
    #bArr = np.logical_and(bArr,structArr['rt']<=2000)

    structArr_ = structArr[bArr] # subset of interest of structArr (all fields, choice rows)
    # build a new structArr_ that has different fields
    dtypeList = [['rt','<i4'], ['trial','<i4'], ['recency','<i4'] \
               , ['stage','<i4'], ['ID','<i4'], ['target','<U40'], ['graphType','<U40']\
               , ['lastnode','<i4'], ['node','<i4'], ['edgeType','<i4'], ['error','<i4']]
    dtype = np.dtype([tuple(x) for x in dtypeList])
    structArr_ = np.array([(\
    structArr_['rt'][i],
    structArr_['trial'][i],
    structArr_['recency'][i],
    structArr_['stage'][i],
    structArr_['id'][i],
    str((structArr_['target_1'][i], structArr_['target_2'][i], structArr_['target_3'][i]\
    , structArr_['target_4'][i], structArr_['target_5'][i])),
    structArr_['graph'][i],
    MEedgeType(structArr_, i)[0],
    MEedgeType(structArr_, i)[1],
    MEedgeType(structArr_, i)[2],
    structArr_['error'][i]
    ) for i in range(len(structArr_))], dtype=dtype)

    return structArr_


'''
    ########################################
        Figure Generation Function 427
    ########################################
'''
def mean3Across(structArr, yfield, xfield0, xfield1):
    '''
    average yfield per (xfield0, xfield1)
    🔴 assume structArr[yfield].dtype = float32
    return a np structured Arr which has the same 3 fields
    '''
    x0 = list(set(structArr[xfield0]))
    x1 = list(set(structArr[xfield1]))
    dtype = [(yfield,structArr[yfield].dtype),(xfield0,structArr[xfield0].dtype),(xfield1,structArr[xfield1].dtype)]
    list_ = [None] * len(x0)*len(x1)
    counter = 0
    for i in x0:
        for j in x1:
            bArr = np.logical_and(structArr[xfield0]==i, structArr[xfield1]==j)
            temp = structArr[yfield][bArr]
            if len(temp) == 0:
                continue # skip if the (i,j) pair is not found
            else:
                list_[counter] = (temp.mean(), i, j)
                counter += 1
    # ↓ get rid of the trailing None object ↓
    try:
        return np.array(list_[:list_.index(None)],dtype=dtype)
    except ValueError: # "ValueError: None is not in list"
        return np.array(list_,dtype=dtype)
def mean2Across(structArr, yfield, xfield):
    '''
    average yfield per xfield
    return a np structured Arr which has only yfield, xfield
    '''
    dtype = [(yfield,structArr[yfield].dtype),(xfield,structArr[xfield].dtype)]
    list_ = [None] * structArr.size # over-estimation
    for counter,i in enumerate(list(set(structArr[xfield]))):
        list_[counter] = (structArr[yfield][structArr[xfield]==i].mean(), i)
    # ↓ get rid of the trailing None object ↓
    try:
        return np.array(list_[:list_.index(None)],dtype=dtype)
    except ValueError: # "ValueError: None is not in list"
        return np.array(list_,dtype=dtype)



def RTplots427(structArr, dpi=None, graphType = '(Modular)'):
    '''
    y axis: always reaction time
    x: can be trial, target, recency, edgeType, ID
    will show aggregated (mean) in the same plot
    '''
    ChrisGreen = "#80a680" # the green used in Fig1.f in Mental Error
    fname = 'ME_raw_rt-'
    fields = ['trial']#, 'recency'] # x axis fields
    marker_style = dict(facecolor='#f48ea5',edgecolor='#7f7596', marker='o'\
                       ,alpha=1,s=5) # previous CSS colors: lightcoral, cornflowerblue
    marker_style['facecolor'], marker_style['edgecolor'] = ChrisGreen, ChrisGreen
    for field in fields:
        s2Arr = mean2Across(structArr, 'rt', field)
        s2Arr_ascending = s2Arr[s2Arr[field].argsort()]
        s2Arr_filtered = savgol_filter(s2Arr_ascending['rt'], 71, 4)
        fig = plt.figure(figsize=[10,5])
        ax = fig.add_subplot()
        ax.scatter(s2Arr_ascending[field], s2Arr_ascending['rt'],**marker_style)
        ax.plot(s2Arr[field], s2Arr_filtered, color ='r', label='Savitzky–Golay filter (size=71, order=4)')
        ax.legend(loc='upper right')
        ax.set_xticks(np.linspace(0,1500,16))
        ax.set_yticks(np.arange(800,ax.get_ylim()[1],50))
        #ax.locator_params(axis='x', nbins=16)
        #ax.locator_params(axis='y', nbins=10)
        #ax.set_xlim(s3Arr[field].min(),s3Arr[field].max())
        #ax.set_ylim(300,2000)
        ax.grid(True)
        ax.set_xlabel(field,fontsize=17)
        ax.set_ylabel(r'Reaction Time ($ms$)',fontsize=17)
        ax.set_title('Reaction Time vs. '+field+' '+graphType,fontsize=17)
        saveNclose427(fig,fname+field+' '+graphType,dpi)

def readNload427(fname, vanilla = True):
    '''
    assume fname is always vanilla name
    two versions: vanilla and modified (vanilla_m)
    e.g., vanilla = 'data_Fig1' | modified = vanilla + '_m' = 'data_Fig1_m'
    if vanilla:
        find if vanilla.npy is in the current directory
            1: load vanilla.npy
            0: load vanilla.csv, save as vanilla.npy (because quicker for future loads) and load
    else:
        find if modified.npy is in the current directory
            1: load modified.npy
            0: try to find if vanilla.npy is in the current directory
                yes: load vanilla.npy, process it and save as modified.npy&.csv and load .npy
                no: load vanilla.csv, save as vanilla.npy & last yes step



    fieldNames = structArr.dtype.names
    field_dtype = [structArr.dtype[i] for i in fieldNames]
    '''
    if vanilla:
        try:
            return np.load(fname+'.npy')
        except:
            dtypeArr_ME = ['<i4','<i4','<U7'] + ['<i4'] * 12
            structArr = csv2dict(fname+'.csv', dtypeArr = dtypeArr_ME)
            np.save(fname, structArr)
            return np.load(fname+'.npy')
    else:
        try:
            return np.load(fname+'_m.npy')
        except:
            try: # first try to find if vanilla.npy exists
                structArr = np.load(fname+'.npy')
            except:
                dtypeArr_ME = ['<i4','<i4','<U7'] + ['<i4'] * 12
                np.save(fname, csv2dict(fname+'.csv', dtypeArr = dtypeArr_ME))
                structArr = np.load(fname+'.npy')
            structArr = get_new_structArr(structArr)
            np.save(fname+'_m', structArr) # save modified as .npy
            #np.savetxt(fname+'_m.csv', structArr, delimiter='.') # save modified as .csv
            return np.load(fname+'_m.npy')



def saveNclose427(fig, fname, dpi, makedir = True):
    if makedir:
        if dpi is not None:
            mkdir_p(f"{'png_dpi'}{dpi}")
        else: # save both
            mkdir_p('png_dpi300')
            mkdir_p('pdf_lossless')
    if dpi is not None:
        fig.savefig(f"{'png_dpi'}{dpi}{'/'}"+fname+'.png', bbox_inches='tight',dpi=dpi) # auto resize fig to fit titles and such
    else: # save both
        fig.savefig('png_dpi300/'+fname+'.png', bbox_inches='tight',dpi=300) # auto resize fig to fit titles and such
        # pdf or svg for lossless quality
        fig.savefig('pdf_lossless/'+fname+'.pdf', bbox_inches='tight') # auto resize fig to fit titles and such
    fig.clf() # clear figure
    plt.close(fig=fig) # close figure

'''
    ########################################
        NetworkScience.py
        But I changed the functions a bit here.
    ########################################
'''

def W_norm(W,axis=1,self_loop=0):
    # W: undirected/directed Weight Matrix, self-loop matters, be cautious
    # when "completely isolated", the vertex in W have self-loop value of 0 and degree 0
    # not completely isolated if self-loop!=0 even if the edges to other vertices have 0 weight
    # self_loop: value to replace self_loop when vertex is completely isolated
    # axis=0/1: normalize W along columns/rows (by default 1 because i->j: W[i,j])
    denom=np.sum(W,axis=axis)
    bit_arr=denom==0 # bit arr, True if completely isolated
    denom[bit_arr]=1 # if completely isolated, divide by 1 instead of 0
    denom=np.repeat([denom],len(denom),axis=0)
    normed=np.divide(W,denom)
    if self_loop!=0:
        ind_arr=np.nonzero(bit_arr) # node indices that are completely isolated
        normed[ind_arr,ind_arr]=self_loop
    return normed


def cd_2_Curio(str="E:/Lune/Study/Coding/Python3/GL", show=0):
    import os # working directory management
    if show==0:
        os.chdir(str)
    elif show==1:
        print("Previously: ",os.getcwd())
        os.chdir(str)
        print("Currently: ",os.getcwd())
    else:
        os.chdir(str)
'''
    ########################################
        Miscellanious
    ########################################
'''

def mkdir_p(path_):
    '''creates a directory. equivalent to using mkdir -p on the command line'''
    from errno import EEXIST
    from os import makedirs,path # create new directories (i.e., folders)
    try:
        makedirs(path_)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(path_): # if existed, pass
            pass
        else: raise


if __name__=="__main__":
    main_RT427()
