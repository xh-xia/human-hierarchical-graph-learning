import numpy as np
from utility427.Sierpinski427 import *
from utility427.helper427 import set_dir427, mkdir_p
from stims427 import Hamiltonian_cycle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap # custom colorbar (https://matplotlib.org/tutorials/colors/colormap-manipulation.html)
from matplotlib.gridspec import GridSpec # for subplots placement manipulation

def main_Sierpinski427():
    set_dir427() # make sure cwd is the one this script is in
    colors = colors_selector(str='5-class Greens')
    beta_arr = np.geomspace(0.0001,10,400)
    hierDict = dict()
    #hierDict['n'] = [[0],[3],[3,4,5]]
    #hierDict['p'] = [[3],[3,4,5],[3]]
    #hierDict['reg_n'] = [[0,1,2,3],[3],[3,4,5]]
    #hierDict['reg_p'] = [[0,1,2,3],[3,4,5],[3]]
    hierDict['r'] = [[0,1,3],[3],[3]]
    err_type = 'std'
    CCS_type = 'mean' # 'mean' or 'std'
    CCS_stat = load_CCS_stat(fname='CCS_stat_{}_{}_{:d}'.format(CCS_type,'reg_n_p',100)) # load simulation results
    for key in hierDict.keys():
        DD = dict() # = DataDict = {(regType,p,lv):{'GTDict'=GTDict,etc.}}
        hierLists = hierDict[key]
        for regType in hierLists[0]:
            for p in hierLists[1]:
                for lv in hierLists[2]:
                    DD[(regType,p,lv)] = dict()
                    DD[(regType,p,lv)]['GTDict'] = make_SierpinskiGraph427(p, lv, norm = True, regType = regType)
                    save_Masks(DD[(regType,p,lv)]['GTDict'], p, lv, regType)
                    DD[(regType,p,lv)]['A_hat_list'] = [make_A_hat_beta(DD[(regType,p,lv)]['GTDict']['A'], beta) for beta in beta_arr]
                    DD[(regType,p,lv)]['CCS_arr'] = CCS_analysis(DD[(regType,p,lv)]['GTDict'], beta_arr, DD[(regType,p,lv)]['A_hat_list'])
                    Sier = make_Sierpinski427(p, lv, x0 = [0.0,0.0], s0=1.0 , c=1.0, regType = regType)
                    Sier.Layout_Sierpinski427()
                    DD[(regType,p,lv)]['Sier'] = Sier
        plot_Graph_CCS(DD, beta_arr, key, \
                       CCS_stat=CCS_stat, err_type=err_type, CCS_type=CCS_type, \
                       colors=colors, regCCS=len(key)>1)


def make_A_hat_beta(A, beta):
    '''
    Arg:
        A (2D nparr; symmetric): adjacency/weight matrix
        beta (any number): complexity-accuracy trade-off param
    Return:
        A_hat (2D nparr, shape = np.shape (A)):
            assuming infinite walks on A, this is the resulting A_hat learned based on beta
            A_hat = (1-e^(-β)) * A * (I - (e^(-β))A)^(-1)
            undirected, weighted 3-regular graph with:
            lv hierarchies:
            level lv: base level; smallest communities/clusters of (3) nodes
            ...
            level 2: 3 clusters of (3) level-3 units
            level 1: 1 cluster of (3) level-2 unit (coarsest level)
    '''
    n = np.shape(A)[0] # # of rows, but assuming symmetric, thus also cols (=nodes)
    A_ = W_norm(A)
    return (1 - np.exp(-beta)) * A_ @ np.linalg.inv(np.eye(n) - np.exp(-beta) * A_)

def get_S_kl(n, A, beta_arr, edgeList, lvList, pList=[1,2,3,4], nList=np.arange(2,47)):
    """
    Args:
        n: power, whereas p is base
        A: GroundTruth Transition Prob matrix.
        beta_arr: list of β
        pList: list of p in L_p(l)
        nList: list of n in I_n(l)
    get L_pl ≡ L_p(l) = sum over all eigenvalues: (λ/(1-λ))**p * S_kl
    where (λ_k/(1-λ_k))**p = pk[p,k]
    where we also need to compute S_kl first:
    "structure factor":
    S_kl: S_kl[k,l] = mean_ij(v[i]*v[j]) for k-th eigvec v; v[i],v[j] ∈ level-(l+1) community (edge is level-(l+1) only)
    with corresponding eigvals
    eigval_kβ: eigval_kβ[k,β] = λ_A_hat[k]
               eigval_kβ[k,-1] = λ_A[k]
    "finite-step surprisal":
    ΔI_n(l1,l2) = sum_k (λ_k^n*(S_kl1-S_kl2))
    I_n(l) = sum_k (λ_k^n*S_kl)
    """
    res = dict() # results of the calculations

    res['eigvals'], res['eigvecs'], res['eigvals_rank'] = rank_eigvals(A) # GroundTruth
    res['num_eigval'] = max(res['eigvals_rank']) + 1 # num of unique eigvals
    res['S_kl'] = np.zeros((res['num_eigval'], max(lvList)))
    res['pk'] = np.zeros((len(pList), res['num_eigval'])) # largest eigval will not be calculated, thus always zeroes in here
    res['nk'] = np.zeros((len(nList), res['num_eigval'])) # ditto
    res['eigval_kβ'] = np.zeros((res['num_eigval'], len(beta_arr)+1))
    #res['L_pl'] = np.zeros((len(pList), max(lvList)))
    #res['I_nl'] = np.zeros((len(nList), max(lvList)))
    #res['ΔI_n'] = np.zeros((len(nList), max(lvList)-1))

    '''
    S_kl calculation (no β involved) & eigval_kβ calculation (no group assignment involved)
    note: (algebraic=geometric in our case) multiplicity should be the same for any β
    '''
    for eig_k in range(res['num_eigval']):
        idx = res['eigvals_rank'].index(eig_k) # first one that matches
        res['eigval_kβ'][eig_k,-1] = res['eigvals'][idx]
        res['eigval_kβ'][eig_k,:-1] = (1-np.exp(-beta_arr)) * res['eigvals'][idx] / (1-(np.exp(-beta_arr))*res['eigvals'][idx])
        if eig_k>=1: # excluding largest eigval (i.e., eig_k = 0)
            for i in range(len(pList)):
                res['pk'][i,eig_k] = (res['eigvals'][idx] / (1-res['eigvals'][idx]))**pList[i]
            for i in range(len(nList)):
                res['nk'][i,eig_k] = (res['eigvals'][idx])**nList[i]
        kth_eigvec = res['eigvecs'][:,idx] # nparr
        for l in range(1,max(lvList)+1):
            b_ = [x==l for x in lvList] # boolean mask
            b_edgeList = [e for (e, v) in zip(edgeList, b_) if v] # edges in level l
            res['S_kl'][eig_k,l-1] = np.mean([kth_eigvec[v_i] * kth_eigvec[v_j] for (v_i,v_j) in b_edgeList])
    res['L_pl'] = res['pk'] @ res['S_kl']
    res['I_nl'] = res['nk'] @ res['S_kl']
    res['ΔI_n'] = np.diff(res['I_nl'][:,::-1], axis=1)[:,::-1]
    return res

def CCS_analysis(GTDict, beta_arr, A_hat_list = None): # need to change the awkward dict output into nparr
    '''
    This function finds CCS for all beta in beta_arr.
    But it also finds CCS analytical approximation.
    Args:
        GroundTruthOnly (bool): not implemented ∵ those zero in original don't have well-defined hierarchies
            True: only calculate mean over the edges that are non-zero in original Sierpiński
            False: calculate mean over all appropriate edges
        A_hat_list (np.arr):
            None: analytical prediction from Eigen-decomposition (requires beta_arr)
            np.arr: simulated result (vanilla method; doesn't require beta_arr)
    Return:
        CCS_arr (3D nparr):
        since CCS is for every 2 consecutive lvs, we have only (lv-1) entries out of lv levels
            CCS_arr[s,0,l-1]: CCS of means at beta s for level l (f"{'lv'}{l}{'-'}{l+1}")
            CCS_arr[s,1,l-1]: CCS of stds at beta s for level l (f"{'lv'}{l}{'-'}{l+1}")

    Copy Paste from make_SierpinskiGraph427() documentation:
    edgeList (a list of size-2 tuples (v_i,v_j))
        node index in edgeList is simply p2ten(s, p=p)
        where s is nodel p-ary string label
    lvList (a list of hierarchy labels): finest level is 1
    '''
    edgeList, lvList = GTDict['edgeList'], GTDict['lvList']
    n = len(beta_arr) # number of beta (which is also number of graphs)
    lv = max(lvList) # (max) hierarchical level (also the coarsest level)
    CCS_arr = np.zeros((n,2,lv-1)) # since CCS is for every 2 consecutive lvs, we have only (lv-1) entries out of lv levels
    # ↓ calculation
    if A_hat_list is not None:
        for i in range(n):
            mean_weights = [0.0 for i in range(lv)]
            std_weights = [0.0 for i in range(lv)]
            for l in range(1,lv+1):
                b_ = [x==l for x in lvList] # boolean mask
                b_edgeList = [e for (e, v) in zip(edgeList, b_) if v] # edges in level l
                temp_list = [A_hat_list[i][v_i,v_j] for (v_i,v_j) in b_edgeList]
                mean_weights[l-1] = np.mean(temp_list)
                std_weights[l-1] = np.std(temp_list)
            #temp = -np.diff(mean_weights) # diff: all >0 if edge weights in finer level > coarser level
            #temp = np.exp(-np.diff(np.log(mean_weights))) # ratio: all >1 if edge weights in finer level > coarser level
            # if i == findNearest(beta_arr, 0.3, is_arg = True):
            #     print("DEBUG [analytical] mean_weights: {} | std_weights: {}".format(mean_weights,std_weights))
            #     exit()
            temp1 = np.divide(mean_weights[:-1], mean_weights[1:]) # ditto, but more explicit
            temp2 = np.divide(std_weights[:-1], std_weights[1:]) # ditto, but more explicit
            for l in range(0,lv-1):
                CCS_arr[i,0,l] = temp1[l]
                CCS_arr[i,1,l] = temp2[l]
    else:
        eigvals, eigvecs = np.linalg.eigh(GTDict['A'])
        eigvals = np.diag(eigvals) # 𝚲
        for i in range(n):
            mean_weights = [0.0 for i in range(lv)]
            std_weights = [0.0 for i in range(lv)]
            EB = np.exp(-beta_arr[i]) # coefficient (e^-β) to find the eigenvalue of learned matrix A_hat
            𝚲_ = (1-EB)*eigvals/(1-EB*eigvals)
            A_hat = eigvecs @ 𝚲_ @ (eigvecs.T) # because A is symmetric (regularized) # for some reason it is all nan when p=3, lv=4
            #A_hat = eigvecs @ 𝚲_ @ (np.linalg.inv(eigvecs))
            for l in range(1,lv+1):
                b_ = [x==l for x in lvList] # boolean mask
                b_edgeList = [e for (e, v) in zip(edgeList, b_) if v] # edges in level l
                temp_list = [A_hat[v_i,v_j] for (v_i,v_j) in b_edgeList]
                mean_weights[l-1] = np.mean(temp_list)
                std_weights[l-1] = np.std(temp_list)
            #temp = -np.diff(mean_weights) # diff: all >0 if edge weights in finer level > coarser level
            #temp = np.exp(-np.diff(np.log(mean_weights))) # ratio: all >1 if edge weights in finer level > coarser level
            temp1 = np.divide(mean_weights[:-1], mean_weights[1:]) # ditto, but more explicit
            temp2 = np.divide(std_weights[:-1], std_weights[1:]) # ditto, but more explicit
            for l in range(0,lv-1):
                CCS_arr[i,0,l] = temp1[l]
                CCS_arr[i,1,l] = temp2[l]

    return CCS_arr

def plot_Graph_CCS(DD, beta_arr, key,
                   CCS_stat=None, CCS_type='mean',
                   err_type='ste', colors=None, regCCS=False):
    """
    Args
    --------------
    DD (dict):
        DD.keys (tuple): (regType,p,n)
    key (str): those in hierDict.keys()
    CCS_stat (dict): CCS_stat['mean'], CCS_stat['std'], and CCS_stat['ste'] have:
        same keys as DD; value is 3D nparr (see RW_CCS_stat.py for description)
    err_type (str): type to use as errorbar: 'std' or 'ste'
    CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    colors (list of color hex strings):
        e.g., plt.rcParams['axes.prop_cycle'].by_key()['color'] is default color in pyplot:
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    regCCS (bool):
        whether we will display 4 rows of CCS with no graphs
        assume:
            hierDict['reg_n'] = [[0,1,2,3],[3],[3,4,5]]
            or
            hierDict['reg_p'] = [[0,1,2,3],[3,4,5],[3]]
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if regCCS:
        fname='CCS_' + key
        fig = plt.figure(figsize=[20,18]) # initialize
    else:
        fname='CCS_' + key
        fig = plt.figure(figsize=[20,9]) # initialize

    ds = 0.2 # dummy axes for spacing between the visible plots
    cbW = 1 # colorbar width
    width_ratios = [ds,19,cbW]*3 # ≡[ds,19,cbW,ds,19,cbW,ds,19,cbW]
    height_ratios = [1,9,ds,1,1,9,ds]
    if regCCS: height_ratios = [1,9,ds,1,1,9,ds,1]+height_ratios
    gs = GridSpec(nrows=len(height_ratios), ncols=len(width_ratios),\
                  height_ratios=height_ratios, width_ratios=width_ratios)
    axes = dict()
    if not regCCS:
        axes['Graph'], axes['CCS'], axes['Colorbar'] = [None] * 3, [None] * 3, [None] * 3
    else:
        for regType in [0,1,2,3]:
            axes['CCS_reg{}'.format(regType)] = [None] * 3
    axes_cb = [None] * 3
    DD_keys = sorted(DD.keys(), reverse=False) # ascending (default)

    if CCS_stat is not None:
        n_agents = CCS_stat['mean'][DD_keys[0]][-1,1,0] # since all β have same group size, take 0
        n_steps = CCS_stat['mean'][DD_keys[0]][-1,2,0] # ditto but w/ walk length
        fname='CCS_' + key + '_{:.0f}_{:.0f}_{}_{}'.format(n_agents,n_steps,err_type,CCS_type)

    if regCCS: # assuming DD_keys has 12 entries
        for i in range(3):
            for regType in [0,1,2,3]:
                axes['CCS_reg{}'.format(regType)][i] = fig.add_subplot(gs[regType*4:regType*4+3,i*3+1])
                params = DD_keys[i+3*regType]
                ax_CCS(axes['CCS_reg{}'.format(regType)][i], beta_arr, DD[params]['CCS_arr'], params, key,\
                noise=CCS_stat, err_type=err_type, CCS_type=CCS_type, \
                is_log=True, colors=colors, regCCS=regType)
    else:
        for i in range(3):
            axes['Graph'][i] = fig.add_subplot(gs[0:3,i*3+1])
            axes['CCS'][i] = fig.add_subplot(gs[4:7,i*3+1])
            params = DD_keys[i]
            axes['Colorbar'][i] = fig.add_subplot(gs[0:3,i*3+2]) # Edge Type colorbar
            ax_Graph(axes['Graph'][i], axes['Colorbar'][i], fig, params, DD[params]['Sier'].nodeList, DD[params]['GTDict'], colors=colors)
            ax_CCS(axes['CCS'][i], beta_arr, DD[params]['CCS_arr'], params, key,\
                   noise=CCS_stat, err_type=err_type, CCS_type=CCS_type, show_sim_param=i==1,\
                   is_log=True, colors=colors, regCCS=3)
    # panel label list
    text_labels = ['A', 'B', 'C', 'D'] if regCCS else ['A', 'B']
    for i in range(len(text_labels)):
        axlabel = fig.add_subplot(gs[i*4:i*4+3,0])
        axlabel.set_frame_on(False)
        axlabel.set_axis_off() # same as ax.axis('off')
        axlabel.text(-2.4,1.05,'{}'.format(text_labels[i]),fontsize=17,\
                     horizontalalignment='center',transform=axlabel.transAxes)
    saveNclose427(fig, fname, dpi = None)

def ax_CCS(ax, x, CCS_arr, params, key,
           noise=None, err_type='ste', CCS_type='mean', show_sim_param=False,
           is_log=True, colors=None, dpi=None, regCCS=None):
    '''
    Args
    --------------
    ax: axis object
    params (tuple): (regType, p, n)
    x: a list of beta
    regType:
        0: default Sierpiński graph
        x: Sierpiński-like graph of type x regularization
    key (str): 'n','p','reg_n', or 'reg_p', this only affects ax.set_ylim() line
    noise (dict of 3D nparr): noise['mean'][params][s,i,beta]
    err_type (str): type to use as errorbar: 'std' or 'ste'
    CCS_type (str): type of edge stat for CCS: 'mean' or 'std'
    show_sim_param (bool): whether we show simulation parameters
    is_log (bool): if True then use log scale on x axis.
    regCCS (int): reusing same var name,
        but in this function = (regCCS+1)th row.
    '''
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    (regType, p, n) = params
    n_level = CCS_arr.shape[2]
    if CCS_type=='mean':
        CCS_type_slice = 0
    else:
        CCS_type_slice = 1
    if n_level>n-1:
        n_level -= 1 # don't show higher level introduced by regularization
    cmap = LinearSegmentedColormap.from_list('custom edge color',colors[:n_level],N=n_level)
    xlabel = r'Shuffling Parameter $\beta$'
    ylabel = 'Ratio of Means of Two Consecutive Levels'
    if regType in [0,1,2,3]:
        title = 'Cross-Cluster Surprisal of ' +\
                r'$^{}$'.format(regType) + r'$S_{:d}^{:d}$'.format(p,n)
    else:
        raise ValueError('<regType> is unclear.')

    styles = dict(alpha=0.74, linewidth=2)
    if noise is not None and show_sim_param:
            styles_txt = dict(fontsize=11,\
                         horizontalalignment='center',transform=ax.transAxes)
            n_agents = noise['mean'][params][-1,1,0] # since all β have same group size, take 0
            n_steps = noise['mean'][params][-1,2,0] # ditto but w/ walk length
            ax.text(0.5,0.92,'n_agents={:.0f}'.format(n_agents), **styles_txt)
            ax.text(0.5,0.85,'walk_length={:.0f}'.format(n_steps), **styles_txt)
            ax.text(0.5,0.78,'errorbar={:s}'.format(err_type), **styles_txt)
            ax.text(0.5,0.71,'CCS_type={:s}'.format(CCS_type), **styles_txt)
    for i in range(n_level): # ↓ first plot analytical curve
        # print('DEBUG: CCS_arr_lv1 - mean {}'.format(CCS_arr[:,0,i]))
        # print('DEBUG: CCS_arr_lv1 - std {}'.format(CCS_arr[:,1,i]))
        # exit()
        ax.plot(x,CCS_arr[:,CCS_type_slice,i],label=f"{'lv'}{i+1}{'/lv'}{i+2}",color=cmap(i),**styles)
        if noise is not None:
            ax.errorbar(noise['mean'][params][-1,0,:], noise['mean'][params][-1,3+i,:], \
                        yerr=noise[err_type][params][-1,3+i,:], \
                        #label='Stochastic '+labels[i].replace('-','/lv'), \
                        linestyle='None', capsize=4.0, marker=".", markersize=11, \
                        markeredgecolor=cmap(i), markerfacecolor=cmap(i), \
                        ecolor=cmap(i), **styles)

    # argmax = beta that maximizes bottom 3 level diffs (2 diffs)
    xmax3 = x[np.argmax(CCS_arr[:,CCS_type_slice,0])]; ymax3 = np.max(CCS_arr[:,CCS_type_slice,0])
    xmax2 = x[np.argmax(CCS_arr[:,CCS_type_slice,1])]; ymax2 = np.max(CCS_arr[:,CCS_type_slice,1])
    text3= '({:.3f},{:.3f})'.format(xmax3,ymax3)
    text2= '({:.3f},{:.3f})'.format(xmax2,ymax2)
    arrowprops=dict(arrowstyle='simple', facecolor='grey', edgecolor='grey', linewidth=1/3, alpha=0.74)
    kw = dict(textcoords='axes fraction', fontsize = 11,
              arrowprops=arrowprops, ha='center', va='center')
    ax.annotate(text3, color=colors[0], xy=(xmax3, ymax3), xytext=(0.85, 0.95), **kw)
    ax.annotate(text2, color=colors[1], xy=(xmax2, ymax2), xytext=(0.15, 0.65), **kw)
    if regCCS==3: ax.set_xlabel(xlabel,fontsize=11) # only have xlabel if bottom row
    ax.set_ylabel(ylabel,fontsize=11)
    if key in ['n','reg_n']:
        ax.set_ylim((0.9,1.3)) # for regType=3, p=3, n=3 max CCS is <1.3
    else:
        ax.set_ylim((0.9,1.63)) # for regType=3, p=5, n=3 max CCS is
    ax.plot(ax.get_xlim(),(1,1), '--', color = 'grey', zorder=0) # y=1 line
    ax.set_title(title,fontsize=17)
    if is_log:
        ax.set_xscale('log') # set x to log scale
        ax.legend(loc='upper left')
    else:
        ax.legend(loc='center right')
    ax.grid(False)
    return xmax2,xmax3

def ax_Graph(ax, axcb, fig, params, nodeList, GTDict, colors=None, dpi=None, annotate=None):
    '''
    Args:
        ax/axcb: axis object
        params (tuple): (regType, p, n)
        nodeList: [(i,x,y),...] (x,y) is coordinate
        GTDict: dictionary containg 'A', 'edgeList', 'lvList' (all GroundTruth)
        annotate (int):
            None: we don't label the nodes
            -1: Decimal
            p (0<p<10): base-p expansion
    '''
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    (regType, p, n) = params
    A = GTDict['A']
    # n is used here to scale node size properly
    #scale = 200 * (5*n**2+4*n)/(2.7**(1.46*n*p/3)-(n*2*(p/3))**2)
    #scale = 240 / np.log(0.1*(p+n)**n)
    scale = 240 / np.log(0.1*(p+n)**n) - n**2/2
    num_nodes = round(p**n)
    all_levels = [x for x in range(1,n+1)] # get all levels, starting from 1, but may end at n+1
    nu = len(all_levels) # num of all levels minus -1 level
    if regType==1: # regularized edges are level n+1
        all_levels.append(n+1)
        nu += 1
    elif regType==3: # regularized edges are level -1
        all_levels.insert(0,-1)


    if regType in [0,1,2,3]:
        title = 'Sierpiński Graph of ' +\
                r'$^{}$'.format(regType) + r'$S_{:d}^{:d}$'.format(p,n)
    else:
        raise ValueError('<regType> is unclear.')

    # use default PMMM theme color
    # draw nodes
    if annotate is not None: scale *= 4.7 # make node larger to fit annotation
    marker_style = dict(facecolor='#f48ea5',edgecolor='#7f7596', marker='o'\
                       ,alpha=1,s=scale) # previous CSS colors: lightcoral, cornflowerblue
    annokw = dict(horizontalalignment='center', verticalalignment='center'\
                 ,color='b', fontsize = 10)
    if annotate is None:
        for i,x,y in nodeList:
            ax.scatter(x,y,zorder=2,**marker_style)
    else:
        for i,x,y in nodeList:
            ax.scatter(x,y,zorder=2,**marker_style)
            ax.annotate(str(p_ary(i,p=p,L=n)), xy=(x,y), xytext=(x,y), **annokw)
    # draw edges & Edge Weight Coloring
    n_ = np.shape(A)[0]
    ''' this is for transition prob edge drawing, which is not used anymore
    RGBA = np.zeros((round(n_*(n_-1)/2),4))
    RGBA[:,1] = 0.5 # for green (not sure if this is color 'g')
    counter = 0
    for i in range(0,n_):
        for j in range(i+1,n_): # undirected (A is symmetric)
            # 🔴 assuming nodeList[i][0] = i
            x = [nodeList[i][1],nodeList[j][1]]
            y = [nodeList[i][2],nodeList[j][2]]
            RGBA[counter,3] = 1 if A[i,j]>0 else 0 # set alpha to 1 (opaque) if edge exists
            ax.plot(x,y,color=RGBA[counter,:],zorder=1) # lower int means drawn on the canvas earlier
            counter += 1
    '''

    cmap = LinearSegmentedColormap.from_list('custom edge color',colors[:nu],N=nu)
    cbr = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0,vmax=nu-1), cmap=cmap)\
               , cax=axcb, format='%.2f')
    cbr.set_ticks([(nu-1)*(2*i+1)/(2*nu) for i in range(nu)])
    cbr.set_ticklabels(np.arange(1,nu+1))
    cbrLabel427(axcb, 'Edge Level')
    for lv in all_levels:
        b_ = [x==lv for x in GTDict['lvList']] # boolean mask
        b_edgeList = [e for (e,v) in zip(GTDict['edgeList'], b_) if v] # edges in level lv
        xcoords,ycoords = np.zeros((2,len(b_edgeList))), np.zeros((2,len(b_edgeList)))
        for i,(v_i,v_j) in enumerate(b_edgeList):
            xcoords[:,i] = [nodeList[v_i][1],nodeList[v_j][1]]
            ycoords[:,i] = [nodeList[v_i][2],nodeList[v_j][2]]
        ax.plot(xcoords,ycoords,color=cmap(lv-1),zorder=1) # lower int means drawn on the canvas earlier
    # Grid setting and save
    axcb.set_frame_on(False)
    axcb.set_axis_off() # same as ax.axis('off')
    ax.set_frame_on(False)
    ax.set_axis_off() # same as ax.axis('off')
    ax.axis('equal') # so that regular polygons appear to be regular as well
    if annotate is None:
        ax.set_title(title, fontsize=17)
    elif annotate==-1:
        ax.set_title(title[20:-12]+'Decimal Representation)', fontsize=17)
    else: # since axET is always GroundTruthOnly, title will just be GroundTruth
        ax.set_title(title[20:-12]+'Base {:d} Representation)'.format(p), fontsize=17)
    ax.grid(False)
    #plt.legend(loc='upper left')


def findNearest(arr, val, is_arg = True):
    '''
    Arg:
        arr (any list like object)
        val (any number): the value to which one wants to find in arr that is nearest
        is_arg (bool): if False, return the value instead of argument/index
    Return:
        index or value depending on is_arg
    '''
    arr_ = np.array(arr) # convert to nparr, make a copy by default
    ind = np.abs(arr_ - val).argmin()
    if is_arg:
        return ind
    else:
        return arr[ind]

def getUpperTriangle(W, diag = True, up = True):
    '''
    Arg:
        W (np.arr; (n,n)): if not square matrix, raise ValueError
        diag (bool): if False, then exclude diagonals (self-loop)
        up (bool): if False, then use lower triangle instead
    Return:
        (Flattened list): e.g., 1,2,3,4,5,6...
        (including diagonals)
        up: 1 2 3   down: 1
              4 5         2 4
                6         3 5 6
    '''
    nrow,ncol = np.shape(W)
    if nrow != ncol:
        raise ValueError('<W> is not square.')
    if up:
        return [e for list_ in [W[i,i+int(not diag):] for i in range(nrow)] for e in list_]
    else:
        return [e for list_ in [W[i+int(not diag):,i] for i in range(nrow)] for e in list_]


def rank_eigvals(A, eig_k = None, rtol = 1e-05):
    '''
    Return:
        eigtup = (eigvals, eigvecs, l, kth_eigval, kth_eigvec)
        ~ or ~
        eigtup = (eigvals, eigvecs, l)
        l: ranking of the eigvals (e.g., eigvals=[1,0.7,0.7,0.6] -> [0,1,1,2])
    '''
    # per (numpy v1.19) https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh
    # "The eigenvalues in ascending order, each repeated according to its multiplicity."
    np.linalg.eigh(A)
    ''' ↑ ❗ if I run this twice, the 2nd time it will be fine;
    otherwise will not function if ~ draw 2-panel plot ~ is ran in plotGraph()'''
    eigvals, eigvecs = np.linalg.eigh(A) # eigenvalues could be repeated (especially for symmetric Graph)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1] # descending order
    n = len(eigvals)
    l = [0 for i in range(n)]
    start_idx = 0 # starting index of the repeated value
    last_rank = 0 # self-explanatory
    for i in range(1, n):
        if eigvals[i] < eigvals[i-1] * (1-rtol):
            l[start_idx:i] = [last_rank] * (i-start_idx)
            start_idx = i # update starting index after finishing with last largest value
            last_rank += 1 # update rank to be the current one
    l[start_idx:] = [last_rank] * (n-start_idx)

    if eig_k is not None:
        try:
            idx = l.index(eig_k) # first one that matches
        except: # can't find eig_k
            print("Can't find eig_k.")
            print('is A symmetric? ', np.allclose(A, A.T, rtol=1e-05, atol=1e-08))
            idx = 0
        kth_eigval = eigvals[idx]
        kth_eigvec = eigvecs[:,idx] # nparr; ↓ min-max normalization ↓
        #kth_eigvec = (kth_eigvec - kth_eigvec.min()) / (kth_eigvec.max() - kth_eigvec.min())
        return (eigvals, eigvecs, l, kth_eigval, kth_eigvec)
    else:
        return (eigvals, eigvecs, l)

def findNearest(arr, val, is_arg = True):
    ''' # modified from one in Sierpinski_Graph.py
    Arg:
        arr (any list like object)
        val (any number): the value to which one wants to find in arr that is nearest
        is_arg (bool): if False, return the value instead of argument/index
    Return:
        index or value depending on is_arg
    '''
    arr_ = np.array(arr) # convert to nparr, make a copy by default
    ind = np.abs(arr_ - val).argmin()
    if is_arg:
        return ind
    else:
        return arr[ind]

def saveNclose427(fig, fname, dpi, makedir = True):
    if makedir:
        if dpi is not None:
            mkdir_p(f"{'output/png_dpi'}{dpi}")
        else: # save both
            mkdir_p('output/png_dpi300')
            mkdir_p('output/pdf_lossless')
    if dpi is not None:
        fig.savefig(f"{'output/png_dpi'}{dpi}{'/'}"+fname+'.png', bbox_inches='tight',dpi=dpi) # auto resize fig to fit titles and such
    else: # save both
        fig.savefig('output/png_dpi300/'+fname+'.png', bbox_inches='tight',dpi=300) # auto resize fig to fit titles and such
        # pdf or svg for lossless quality
        fig.savefig('output/pdf_lossless/'+fname+'.pdf', bbox_inches='tight') # auto resize fig to fit titles and such
    fig.clf() # clear figure
    plt.close(fig=fig) # close figure



'''
    ########################################
        Miscellanious
    ########################################
'''

def colors_selector(str=None, reverse=True):
    """
    website of reference: https://colorbrewer2.org/
    I picked the color from top to bottom, so typically that's light to dark
    if reverse is True, I will reverse said order, going from bottom to top.
    """
    if str is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color in pyplot
    else:
        if str=='7-class Greys':
            colors = ['#f7f7f7','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']
        elif str=='7-class Purples':
            colors = ['#f2f0f7','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#4a1486']
        elif str=='7-class Oranges':
            colors = ['#feedde','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#8c2d04']
        elif str=='7-class Greens':
            colors = ['#edf8e9','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#005a32']
        elif str=='5-class Greys':
            colors = ['#f7f7f7','#cccccc','#969696','#636363','#252525']
        elif str=='5-class Purples':
            colors = ['#f2f0f7','#cbc9e2','#9e9ac8','#756bb1','#54278f']
        elif str=='5-class Oranges':
            colors = ['#feedde','#fdbe85','#fd8d3c','#e6550d','#a63603']
        elif str=='5-class Greens':
            colors = ['#edf8e9','#bae4b3','#74c476','#31a354','#006d2c']
        elif str=='':
            colors = ['#','#','#','#','#','#','#']
    return colors[::-1] if reverse else colors

def load_CCS_stat(fname='CCS_stat'):
    set_dir427() # make sure cwd is the one this script is in
    fname = 'input/' + fname + '.npy'
    try:
        return np.load(fname, allow_pickle=True).tolist()
    except OSError: # couldn't find the file
        raise OSError("Where is the result from stochastic simulations? (make sure .npy is in 'input' folder)")

def make_level_masks(GTDict):
    """
    Generate mask of adjacency matrix such that all edges belong to certain level.
    """
    masks = dict()
    for l in set(GTDict['lvList']): # initialize mask for all levels (including -1, which is undefined lv)
        masks[f"{'lv'}{l}"] = np.zeros_like(GTDict['A'])
    for k,(i,j) in enumerate(GTDict['edgeList']):
        masks[f"{'lv'}{GTDict['lvList'][k]}"][i,j]=1
        masks[f"{'lv'}{GTDict['lvList'][k]}"][j,i]=1 # undirected graph
    return masks

def save_Masks(GTDict, p, n, regType):
    Sierpinski_dict = dict()
    Sierpinski_dict['A'] = GTDict['A']
    Sierpinski_dict['masks'] = make_level_masks(GTDict)
    fname = 'output/npy_files/'
    mkdir_p(fname)
    fname += 'Sierpinski(regType={:d},p={:d},n={:d})'.format(regType,p,n)
    np.save(fname, Sierpinski_dict)

def cbrLabel427(cax, title):
    '''
    set colorbar label to the left, vertically
    cbm.set_label() is the "vanilla"
    assume fig.colorbar(ticklocation='right') which should be default
    default fontsize plt.rcParams['font.size']=10.0
    my convention of fontsize: title 17, other titles 11.
    Args:
        cax: axis onto which colorbar is drawn
    '''
    cax.text(-0.9,0.5,title\
           , transform=cax.transAxes\
           , verticalalignment='center', horizontalalignment='center'\
           , fontsize=11\
           , rotation='vertical')


if __name__=="__main__":
    main_Sierpinski427()
