import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap # custom colorbar (https://matplotlib.org/tutorials/colors/colormap-manipulation.html)
from matplotlib.gridspec import GridSpec # for subplots placement manipulation

from utility427.Sierpinski427 import *
from stims427 import Hamiltonian_cycle

def main_Sierpinski427():
    cd427(show=1)

    hierLists = [[0],[3],[3,4,5]]
    #hierLists = [[0,1,2],[3,4],[3,4]]
    #hierLists = [[0,1],[3,4,5],[3]]
    plot_hamiltonian = False
    beta_arr = np.geomspace(0.0001,10,400)
    #beta_arr_ = np.array([2**(-3) * 2**i for i in range(7)])
    beta_arr_ = np.array([4**(-2) * 4**i for i in range(5)]) # for eigenPlot427 only

    for regType in hierLists[0]:
        for p in hierLists[1]:
            for lv in hierLists[2]:
                GTDict = make_SierpinskiGraph427(p, lv, norm = True, regType = regType)
                save_Masks(GTDict, p, lv, regType)
                #continue
                A_hat_list = [make_A_hat_beta(GTDict['A'], beta) for beta in beta_arr]
                CCS_dict = CCS_analysis(A_hat_list, GTDict, beta_arr, sim = True, approx = True)
                beta1,beta3 = plot_CCS(beta_arr,CCS_dict,p,lv,y2=1,regType=regType,is_log=True,dpi=None)
                #plot_deltaI_n(CCS_dict, p, lv, regType=regType, dpi = None)

                #continue
                #eigenPlot427(regType, p, lv, GTDict['A'], beta_arr_, GTDict['edgeList'], GTDict['lvList'])

                Sier = make_Sierpinski427(p, lv, x0 = [0.0,0.0], s0=1.0 , c=1.0, regType = regType)
                Sier.Layout_Sierpinski427()

                #_,eigvecs = np.linalg.eig(GTDict['A']) # all betas share same eigvecs
                eig_ks = [None,1,2,3,9,10,11]
                if plot_hamiltonian:
                    _, A_Hamiltonian = Hamiltonian_cycle(p, lv, regType=0, adjacency=True)
                    plotGraph(regType, p, lv, Sier.nodeList, GTDict, A=A_Hamiltonian, beta=None\
                    , layoutInd = 427, eig_ks = eig_ks, pivots=None, dpi=None, annotate=True, GroundTruthOnly=True)
                else:
                    plotGraph(regType, p, lv, Sier.nodeList, GTDict, A=GTDict['A'], beta=None\
                    , layoutInd = 427, eig_ks = eig_ks, pivots=None, dpi=None, annotate=False, GroundTruthOnly=True)
                continue
                for beta in [beta1]:#,0.33,beta3]:
                    A_ptsmth = A_hat_list[findNearest(beta_arr,beta)]
                    #hiss(getUpperTriangle(A_ptsmth, diag = True, up = True), lv\
                    #, 17, text='', beta=beta, dpi=300)
                    plotGraph(regType, p, lv, Sier.nodeList, GTDict, A=A_ptsmth, beta=beta\
                    , layoutInd = 427, eig_ks = eig_ks, pivots=None, dpi=None)

def make_level_masks(GTDict):
    """
    Generate mask of adjacency matrix such that all edges belong to certain level.
    """
    masks = dict()
    for l in set(GTDict['lvList']): # initialize mask for all levels (including -427, which is undefined lv)
        masks[f"{'lv'}{l}"] = np.zeros_like(GTDict['A'])
    for k,(i,j) in enumerate(GTDict['edgeList']):
        masks[f"{'lv'}{GTDict['lvList'][k]}"][i,j]=1
        masks[f"{'lv'}{GTDict['lvList'][k]}"][j,i]=1 # undirected graph
    return masks

def save_Masks(GTDict, p, n, regType):
    Sierpinski_dict = dict()
    Sierpinski_dict['A'] = GTDict['A']
    Sierpinski_dict['masks'] = make_level_masks(GTDict)
    fname = 'npy_files/'
    mkdir_p(fname)
    fname += 'Sierpinski(regType={:d},p={:d},n={:d})'.format(regType,p,n)
    np.save(fname, Sierpinski_dict)

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

def CCS_analysis(A_hat_list, GTDict, beta_arr, sim = True, approx = True):
    '''
    This function finds CCS for all beta in beta_arr.
    But it also finds CCS analytical approximation.
    Args:
        GroundTruthOnly (bool):
            True: only calculate mean over the edges that are non-zero in original Sierpiński
            False: calculate mean over all appropriate edges
        sim (bool):
            True: simulated result (vanilla method)
            False: analytical prediction from Eigen-decomposition
        approx (bool):
            False: simulated/exact CCS (use ratio instead of diff if False)
            True: calculate appoximated MEANS: MEANS2, MEANS3 (∵ approx, use diff instead of ratio)
                2: geometric expansions
                3: first principle approach (eq. right above eq.6 in Mental Errors paper)
    Return:
        CCS_dict: a dictionary of dictionaries:
        (both MEANS are CCS)
        'MEANS': be it simulated or analytical, they should be the same and exact
        'MEANS2': this is analytical approximation of MEANS, provided beta_arr is present
        'MEANS3': ditto except using 1st principle approach
        'CCS2_extrema': extrema (a tuple (β,CCS)) for MEANS two, one per label
        'labels': keys for the MEANS dict and extrema dict

    Copy Paste from make_SierpinskiGraph427() documentation:
    edgeList (a list of size-2 tuples (v_i,v_j))
        node index in edgeList is simply p2ten(s, p=p)
        where s is nodel p-ary string label
    lvList (a list of hierarchy labels): finest level is 1
    '''
    edgeList, lvList = GTDict['edgeList'], GTDict['lvList']
    n = len(beta_arr) # number of beta (which is also number of graphs)
    lv = max(lvList) # (max) hierarchical level (also the coarsest level)
    CCS_dict = dict()
    MEANS, MEANS2, MEANS3 = dict(), dict(), dict()
    labels = [f"{'lv'}{l}{'-'}{l+1}" for l in range(1,lv)] # key for MEANS
    # ↓ initialization
    for l in range(0,lv-1): # since CCS is difference, we will have only (lv-1) entries out of lv levels
        MEANS[labels[l]] = [0.0 for i in range(n)]
        MEANS2[labels[l]] = [0.0 for i in range(n)]
        MEANS3[labels[l]] = [0.0 for i in range(n)]
    mean_weights=[0.0 for i in range(lv)]
    # ↓ calculation
    if sim:
        for i in range(n):
            for l in range(1,lv+1):
                b_ = [x==l for x in lvList] # boolean mask
                b_edgeList = [e for (e, v) in zip(edgeList, b_) if v] # edges in level l
                mean_weights[l-1] = np.mean([A_hat_list[i][v_i,v_j] for (v_i,v_j) in b_edgeList])
            #temp = -np.diff(mean_weights) # diff: all >0 if edge weights in finer level > coarser level
            #temp = np.exp(-np.diff(np.log(mean_weights))) # ratio: all >1 if edge weights in finer level > coarser level
            temp = np.divide(mean_weights[:-1], mean_weights[1:]) # ditto, but more explicit
            for l in range(0,lv-1):
                MEANS[labels[l]][i] = temp[l]
    else:
        eigvals, eigvecs = np.linalg.eigh(GTDict['A'])
        eigvals = np.diag(eigvals) # 𝚲
        for i in range(n):
            EB = np.exp(-beta_arr[i]) # coefficient (e^-β) to find the eigenvalue of learned matrix A_hat
            Lambda = (1-EB)*eigvals/(1-EB*eigvals)
            A_hat = eigvecs @ Lambda @ (eigvecs.T) # because A is symmetric (regularized) # for some reason it is all nan when p=3, lv=4
            #A_hat = eigvecs @ Lambda @ (np.linalg.inv(eigvecs))
            for l in range(1,lv+1):
                b_ = [x==l for x in lvList] # boolean mask
                b_edgeList = [e for (e, v) in zip(edgeList, b_) if v] # edges in level l
                mean_weights[l-1] = np.mean([A_hat[v_i,v_j] for (v_i,v_j) in b_edgeList])
            #temp = -np.diff(mean_weights) # diff: all >0 if edge weights in finer level > coarser level
            #temp = np.exp(-np.diff(np.log(mean_weights))) # ratio: all >1 if edge weights in finer level > coarser level
            temp = np.divide(mean_weights[:-1], mean_weights[1:]) # ditto, but more explicit
            for l in range(0,lv-1):
                MEANS[labels[l]][i] = temp[l]

    if approx:
        n_Choice_per_Label = [2, 10, 20, 40]
        CCS2_extrema = dict()
        res = get_S_kl(GTDict['n'], GTDict['A'], beta_arr, GTDict['edgeList'], GTDict['lvList'])
        for i in range(n):
            for l in range(0,lv-1):
                ΔL1 = res['L_pl'][0,l]-res['L_pl'][0,l+1]
                ΔL2 = res['L_pl'][1,l]-res['L_pl'][1,l+1]
                if i==0: # any i, doesn't matter, only calculate once per l
                    CCS2_extrema[labels[l]] = (ΔL1/(ΔL1+2*ΔL2), ΔL1**2/(2*(ΔL1+2*ΔL2)))
                β = beta_arr[i]
                E = 1 - np.exp(-β)
                #MEANS2[labels[l]][i] = ΔL1*β - (ΔL1/2 + ΔL2) * β**2
                MEANS2[labels[l]][i] = ΔL1 * E - ΔL2 * E**2 # up to pow=2 term
                ΔIn = res['ΔI_n'][n_Choice_per_Label[l]-2, l]
                MEANS3[labels[l]][i] = E*(np.exp(-(n_Choice_per_Label[l]-1)*β))*ΔIn # n=n_Choice_per_Label[l]-1 term
                #MEANS3[labels[l]][i] += E*(np.exp(-2*β))*ΔI3 # + n=2 term
                #MEANS3[labels[l]][i] += E*(np.exp(-3*β))*ΔI4 # + n=3 term
                #MEANS3[labels[l]][i] += E*(np.exp(-4*β))*ΔI5 # + n=4 term
                '''
                for o in range(2,10): # if end=10, make sure pList includes 10
                    ΔL = res['L_pl'][o,l+1]-res['L_pl'][o,l]
                    MEANS2[labels[l]][i] += (-1)**o * ΔL * E**(o+1)'''
        CCS_dict['CCS2_extrema'] = CCS2_extrema
        CCS_dict['ΔI_n'] = res['ΔI_n']
    CCS_dict['MEANS'], CCS_dict['MEANS2'], CCS_dict['MEANS3'] = MEANS, MEANS2, MEANS3
    CCS_dict['labels'] = labels
    return CCS_dict

def plot_deltaI_n(CCS_dict, p, n, regType=0, dpi = None):
    labels = CCS_dict['labels']
    x = np.arange(2,47); y = CCS_dict['ΔI_n']
    xlabel = r'$n$'
    ylabel = r'$\Delta I_{n}$'
    if regType == 0:
        title = 'Finite-Step Surprisal of ' +\
                r'$S_{:d}^{:d}$'.format(p,n)
    elif regType == 1:
        title = 'Finite-Step Surprisal of ' +\
                r'$^{+}$' + r'$S_{:d}^{:d}$'.format(p,n)
    elif regType == 2:
        title = 'Finite-Step Surprisal of ' +\
                r'$^{++}$' + r'$S_{:d}^{:d}$'.format(p,n)
    else:
        raise ValueError('<regType> is unclear.')
    styles = dict(alpha=0.74, linewidth=1.4)
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.plot(x,y[:,i],label=labels[i],**styles)
    # maxima annotations
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default colors in pyplot
    arrowprops=dict(arrowstyle='simple', facecolor='grey', edgecolor='grey', linewidth=1, alpha=0.74)
    kw = dict(textcoords='axes fraction', fontsize = 11,
              arrowprops=arrowprops, ha='center', va='center')
    for i in range(n-1):
        xmax = x[np.argmax(y[:,i])]; ymax = np.max(y[:,i])
        text= r'$n={:.0f}$'.format(xmax)
        ax.annotate(text, color=colors[i], xy=(xmax, ymax), xytext=(0.15+0.10*i, 0.95-0.10*i), **kw)
    ax.set_xlabel(xlabel,fontsize=11);ax.set_ylabel(ylabel,fontsize=11)
    ax.set_title(title,fontsize=17)
    ax.legend(loc='best')
    ax.grid(False)

    fname='FSS(regType={:d},p={:d},n={:d})'.format(regType,p,n)
    saveNclose427(fig, fname, dpi = dpi)

def plot_CCS(x, CCS_dict, p, n, y2=1, regType=0, is_log=True, text='', dpi=None):
    '''
    Args:
        x: a list of beta
        y: a dict with key in <labels>
        y2 (int): alternative way of obtaining (e.g., approximation) y
            1: don't use approximation at all.
            2: MEANS2
            3: MEANS3
        regType:
            0: default Sierpiński graph
            1: Sierpiński-like graph of type 1 regularization
            2: Sierpiński-like graph of type 2 regularization
        is_log (bool): if True then use log scale on x axis.
    '''
    y = CCS_dict['MEANS']
    labels = CCS_dict['labels']
    labels.remove(f"{'lv'}{n}{'-'}{n+1}") if len(labels)>n-1 else None # don't show higher level introduced by regularization
    xlabel = r'$\beta$'
    ylabel = 'Ratio of the Means Between Consecutive Levels'
    if regType == 0:
        title = 'Cross-Cluster Surprisal of ' +\
                r'$S_{:d}^{:d}$'.format(p,n)
    elif regType == 1:
        title = 'Cross-Cluster Surprisal of ' +\
                r'$^{+}$' + r'$S_{:d}^{:d}$'.format(p,n)
    elif regType == 2:
        title = 'Cross-Cluster Surprisal of ' +\
                r'$^{++}$' + r'$S_{:d}^{:d}$'.format(p,n)
    else:
        raise ValueError('<regType> is unclear.')

    scale = 30
    styles = dict(alpha=0.74, linewidth=scale/15)
    fig, ax = plt.subplots()
    #fig.scatter(x,y,facecolor='g',alpha=0.74,s=scale,label=labels[0]) # for scatter plot reference
    for i in range(len(labels)):
        ax.plot(x,y[labels[i]],label=labels[i],**styles)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default colors in pyplot
    flag_y2=False
    if y2>1:
        flag_y2=True
        y2 = CCS_dict['MEANS'+str(y2)]
        counter=0
        for i in range(len(labels)):
            y2_ = np.array(y2[labels[i]])
            # don't plot out of bounds (in exact plot)
            b_ =  np.logical_and(y2_ >= ax.get_ylim()[0], y2_ <= ax.get_ylim()[1]) # boolean mask
            ax.plot(x[b_],y2_[b_],label=labels[i]+' (approx.)',linestyle='--',color=colors[counter], **styles)
            counter += 1

    # argmax = beta that maximizes bottom 3 level diffs (2 diffs)
    xmax3 = x[np.argmax(y[labels[0]])]; ymax3 = np.max(y[labels[0]])
    xmax2 = x[np.argmax(y[labels[1]])]; ymax2 = np.max(y[labels[1]])
    text3= r'$\beta={:.3f}$'.format(xmax3)
    text2= r'$\beta={:.3f}$'.format(xmax2)
    arrowprops=dict(arrowstyle='simple', facecolor='grey', edgecolor='grey', linewidth=scale/90, alpha=0.74)
    kw = dict(textcoords='axes fraction', fontsize = 11,
              arrowprops=arrowprops, ha='center', va='center')
    ax.annotate(text3, color=colors[0], xy=(xmax3, ymax3), xytext=(0.90, 0.95), **kw)
    ax.annotate(text2, color=colors[1], xy=(xmax2, ymax2), xytext=(0.15, 0.65), **kw)
    if flag_y2:
        def is_in_frame(ex,ax):
            return False # set to always False for now because otherwise error: image size too large
            if ax.get_xlim()[0] <= ex[0] <= ax.get_xlim()[1]:
                if ax.get_ylim()[0] <= ex[1] <= ax.get_ylim()[1]:
                    return True
            return False
        text3_= r'$\beta={:.3f}$'.format(CCS_dict['CCS2_extrema'][labels[0]][0])
        text2_= r'$\beta={:.3f}$'.format(CCS_dict['CCS2_extrema'][labels[1]][0])
        if is_in_frame(CCS_dict['CCS2_extrema'][labels[0]],ax):
            ax.annotate(text3_, color=colors[0], xy=CCS_dict['CCS2_extrema'][labels[0]], xytext=(0.90, 0.40), **kw)
        if is_in_frame(CCS_dict['CCS2_extrema'][labels[1]],ax):
            ax.annotate(text2_, color=colors[1], xy=CCS_dict['CCS2_extrema'][labels[1]], xytext=(0.15, 0.20), **kw)
    ax.set_xlabel(xlabel,fontsize=11);ax.set_ylabel(ylabel,fontsize=11)
    #ax.set_ylim((0.9,1.67)) # used to be ax.set_ylim((0,ymax3*1.04))
    ax.set_ylim((0.9,1.4)) # for p=3 highest CCS is <1.4
    ax.plot(ax.get_xlim(),(1,1), '--', color = 'grey', zorder=0) # y=1 line
    ax.set_title(title,fontsize=17)
    if is_log:
        ax.set_xscale('log') # set x to log scale
        ax.legend(loc='upper left')
    else:
        ax.legend(loc='center right')
    ax.grid(False)
    #text_x=(max(x)-min(x))*8/14+min(x);text_y=(max(y)-min(y))*1/10+min(y)
    #ax.text(text_x,text_y,text,fontsize='x-large')

    fname='CCS(regType={:d},p={:d},n={:d})'.format(regType,p,n)
    saveNclose427(fig, fname, dpi = dpi)
    return xmax2,xmax3


def eigenPlot427(regType, p, n, A, beta_arr, edgeList, lvList, dpi=None):
    ''' This one uses nparr
    🔴 assume A is GroundTruth & beta_arr is np.arr
    A_hat = (1-e^(-β)) * A * (I - (e^(-β))A)^(-1)
    λ_A_hat = (1-e^(-β)) * λ_A / (1-(e^(-β))*λ_A)
    '''
    res = get_S_kl(n, A, beta_arr, edgeList, lvList)

    ''' '''
    for eig_k in range(res['num_eigval']):
        idx = res['eigvals_rank'].index(eig_k) # first one that matches
    title = 'Structure Factors vs. Eigenvalues of All-Level Groups of '
    if regType == 0:
        title += r'$S_{:d}^{:d}$'.format(p,n)
    elif regType == 1:
        title += r'$^{+}$' + r'$S_{:d}^{:d}$'.format(p,n)
    elif regType == 2:
        title += r'$^{++}$' + r'$S_{:d}^{:d}$'.format(p,n)
    else:
        raise ValueError('<regType> is unclear.')

    title += ' (GroundTruth)'
    fname='StructureScat(regType={:d},p={:d},n={:d},GroundTruth)'.format(regType,p,n)

    num_beta = 7 # number of different βs to plot
    if len(beta_arr) < 14:
        num_beta = len(beta_arr) # override with len of beta_arr
        betaidx = [i for i in range(num_beta)]
    else:
        betaidx = [round((2*i+1)*len(beta_arr)/(2*num_beta)) for i in range(num_beta)] # set 7 β to plot (similar to cbr.set_ticks)
    fig = plt.figure(figsize=[40,7]) # initialize
    #ds = 0.2 # dummy axes for spacing between the visible plots
    ywdth = 10 # width for master y-axis for all num_beta plots
    bw = 17 # beta plot width
    gs = GridSpec(nrows=3, ncols=2+num_beta, height_ratios=[1,7,1]\
    , width_ratios=[24,ywdth] + [bw]*num_beta, wspace=0)
    ax = fig.add_subplot(gs[:,0]) # Structure Factor
    axes = [None] * num_beta
    for i in range(num_beta):
        axes[i] = fig.add_subplot(gs[:,i+2]) # y axis is now eigval(β)*Structure Factor (7 β)
        axes[i].sharex(ax)
        if i>=1:
            axes[i].axes.yaxis.set_visible(False)
    for i in range(n):
        s = ax.scatter(res['eigval_kβ'][:,-1],res['S_kl'][:,i],alpha=0.74)
        ax.plot(res['eigval_kβ'][:,-1],res['S_kl'][:,i],alpha=0.74) # add line plot to connect the points
        for j in range(num_beta):
            axes[j].scatter(res['eigval_kβ'][:,-1],res['eigval_kβ'][:,betaidx[j]]*res['S_kl'][:,i]\
            ,alpha=0.74)
            axes[j].plot(res['eigval_kβ'][:,-1],res['eigval_kβ'][:,betaidx[j]]*res['S_kl'][:,i]\
            ,alpha=0.74)
        if i==0:
            s.set_label('Level-{:d}'.format(i+1)+' (Coarsest)')
        elif i==n-1:
            s.set_label('Level-{:d}'.format(i+1)+' (Finest)')
        else:
            s.set_label('Level-{:d}'.format(i+1))
    ax.set_xlim((-np.abs(ax.get_xlim()).max(),np.abs(ax.get_xlim()).max()))
    ax.set_ylim((-np.abs(ax.get_ylim()).max(),np.abs(ax.get_ylim()).max()))
    ax.plot((0,0),ax.get_ylim(), '--', color = 'grey', zorder=0) # x=0 line
    ax.plot(ax.get_xlim(),(0,0), '--', color = 'grey', zorder=0) # y=0 line
    for i in range(num_beta):
        axes[i].set_ylim(ax.get_ylim())
        axes[i].set_xlabel('Eigenvalue')
        axes[i].text(0.25,0.1,r'$\beta={:.4f}$'.format(beta_arr[betaidx[i]])\
        ,fontsize=11,horizontalalignment='center',transform=axes[i].transAxes)
        axes[i].plot((0,0),ax.get_ylim(), '--', color = 'grey', zorder=0) # x=0 line
        axes[i].plot(ax.get_xlim(),(0,0), '--', color = 'grey', zorder=0) # y=0 line
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Structure Factor')
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1,0))
    axes[0].set_ylabel(r'Kernel $\cdot$ Structure Factor')

    saveNclose427(fig,fname,dpi,makedir=True)

def plotGraph(regType, p, n, nodeList, GTDict, A=None, beta=None, layoutInd=0\
            , eig_ks=None, pivots=None, dpi=None, annotate=False, GroundTruthOnly=False):
    '''
    Args:
        nodeList: [(i,x,y),...] (x,y) is coordinate
        GTDict: dictionary containg 'A', 'edgeList', 'lvList' (all GroundTruth)
        A: weight matrix to draw edge
        eig_ks: indices for eigenvec, 1 corresponds to 2nd largest eigval (eig(A))
        annotate (bool): whether we label the nodes
        GroundTruthOnly (bool): if True, will only have one panel
    '''
    # n is used here to scale node size properly
    #scale = 200 * (5*n**2+4*n)/(2.7**(1.46*n*p/3)-(n*2*(p/3))**2)
    scale = 240 / np.log(0.1*(p+n)**n)
    num_nodes = round(p**n)
    all_levels = list(set(GTDict['lvList'])) # get all levels, starting from 1, but may end at n+1
    nu = len(all_levels)

    if annotate: beta=None # annotation, we always use GroundTruth, so ignore beta input

    if regType == 0:
        title = 'Sierpiński Graph of ' +\
                r'$S_{:d}^{:d}$'.format(p,n)
    elif regType == 1:
        title = 'Sierpiński Graph of ' +\
                r'$^{+}$' + r'$S_{:d}^{:d}$'.format(p,n)
        num_nodes+=1
    elif regType == 2:
        title = 'Sierpiński Graph of ' +\
                r'$^{++}$' + r'$S_{:d}^{:d}$'.format(p,n)
        num_nodes+=round(p**(n-1))
    else:
        raise ValueError('<regType> is unclear.')
    if GroundTruthOnly:
        fname = 'OnePanel-'
    else:
        fname = '{:d}-'.format(layoutInd)
    if beta is not None:
        title += r' ($\beta={:.3f}$)'.format(beta)
        fname+='Visual_S(regType={:d},p={:d},n={:d},beta={:.3f})'.format(regType,p,n,beta)
    else:
        if not GroundTruthOnly:
            title += ' (GroundTruth)'
        fname+='Visual_S(regType={:d},p={:d},n={:d},beta=GroundTruth)'.format(regType,p,n)

    ''' draw eigenvalues '''
    if A is not None and not GroundTruthOnly:
        eigvals,_,_ = rank_eigvals(A)
        fig2, ax2 = plt.subplots()
        ax2.scatter(np.arange(len(eigvals)),eigvals,facecolor='g',alpha=0.74)
        ax2.set_xlabel('Eigenvalue Index')
        ax2.set_ylabel('Eigenvalue')
        ax2.set_title('Eigenvalues (of Weight Matrix)' + title[len('Sierpiński Graph'):])
        saveNclose427(fig2,fname.replace('Visual_S','eigval'),dpi)

    ''' draw 2-panel plot: left is Sierpiński Graph, edge weight; right is also Sierpiński, but with edgeType
        hence nodes are the same for both axTP & axET
        If annotate is true, we also have two panels, but they are all GroundTruth and no other information.
        If GroundTruthOnly is true, only one panel.
    '''
    temp_len = len(fname)
    if GroundTruthOnly: # use default PMMM theme color; also don't generate any eig_k plots; no transition plots
        # make figure using GridSpec
        fig = plt.figure(figsize=[6,5]) # initialize
        ds = 0.2 # dummy axes for spacing between the visible plots
        cbW = 1 # colorbar width
        gs = GridSpec(nrows=3, ncols=4, height_ratios=[1,7,1]\
        , width_ratios=[ds,17,ds,cbW])
        axET = fig.add_subplot(gs[:,1]) # same Sierpiński graph but for Edge Type
        axcbr = fig.add_subplot(gs[1,3]) # Edge Type colorbar
        # draw nodes
        marker_style = dict(facecolor='#f48ea5',edgecolor='#7f7596', marker='o'\
                           ,alpha=1,s=scale) # previous CSS colors: lightcoral, cornflowerblue
        annokw = dict(horizontalalignment='center', verticalalignment='center'\
                     ,color='b', fontsize = 10)
        for i,x,y in nodeList:
            axET.scatter(x,y,zorder=2,**marker_style)
        # draw edges
        if A is not None:
            # Edge Weight Coloring
            n_ = np.shape(A)[0]
            RGBA = np.zeros((round(n_*(n_-1)/2),4))
            RGBA[:,1] = 0.5 # for green (not sure if this is color 'g')
            counter = 0
            for i in range(0,n_):
                for j in range(i+1,n_): # undirected (A is symmetric)
                    # 🔴 assuming nodeList[i][0] = i
                    x = [nodeList[i][1],nodeList[j][1]]
                    y = [nodeList[i][2],nodeList[j][2]]
                    RGBA[counter,3] = 1 if A[i,j]>0 else 0 # set alpha to 1 (opaque) if edge exists
                    axET.plot(x,y,color=RGBA[counter,:],zorder=1) # lower int means drawn on the canvas earlier
                    counter += 1
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color in pyplot
            cmap = LinearSegmentedColormap.from_list('custom edge color',colors[:nu],N=nu)
            cbr = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0,vmax=nu-1), cmap=cmap)\
                       , cax=axcbr, format='%.2f')
            cbr.set_ticks([(nu-1)*(2*i+1)/(2*nu) for i in range(nu)])
            cbr.set_ticklabels(np.arange(1,nu+1))
            cbrLabel427(axcbr, 'Edge Level')
            for lv in all_levels:
                b_ = [x==lv for x in GTDict['lvList']] # boolean mask
                b_edgeList = [e for (e,v) in zip(GTDict['edgeList'], b_) if v] # edges in level lv
                xcoords,ycoords = np.zeros((2,len(b_edgeList))), np.zeros((2,len(b_edgeList)))
                for i,(v_i,v_j) in enumerate(b_edgeList):
                    xcoords[:,i] = [nodeList[v_i][1],nodeList[v_j][1]]
                    ycoords[:,i] = [nodeList[v_i][2],nodeList[v_j][2]]
                axET.plot(xcoords,ycoords,color=cmap(lv-1),zorder=1) # lower int means drawn on the canvas earlier
        # Grid setting and save
        for ax in [axcbr]: # does not display empty colorbar
            ax.set_frame_on(False)
            ax.set_axis_off() # same as ax.axis('off')
        for i, ax in enumerate([axET]):
            ax.set_frame_on(False)
            ax.set_axis_off() # same as ax.axis('off')
            ax.axis('equal') # so that regular polygons appear to be regular as well
            ax.set_title(title, fontsize=17)
            ax.grid(False)
        #plt.legend(loc='upper left')
        saveNclose427(fig,fname,dpi)
    elif annotate: # use default PMMM theme color; also don't generate any eig_k plots
        fig = plt.figure(figsize=[14,5]) # initialize
        ds = 0.2 # dummy axes for spacing between the visible plots
        cbW = 1 # colorbar width
        gs = GridSpec(nrows=3, ncols=9, height_ratios=[1,7,1]\
        , width_ratios=[cbW,ds,17,ds,cbW,ds,17,ds,cbW])
        axTP = fig.add_subplot(gs[:,2]) # main Sierpiński graph for Transition Probability
        axET = fig.add_subplot(gs[:,6]) # same Sierpiński graph but for Edge Type
        axcbl = fig.add_subplot(gs[1,0]) # Transition Probability colorbar
        axcbm = fig.add_subplot(gs[1,4]) # Node Type colorbar
        axcbr = fig.add_subplot(gs[1,8]) # Edge Type colorbar
        # draw nodes
        marker_style = dict(facecolor='#f48ea5',edgecolor='#7f7596', marker='o'\
                           ,alpha=1,s=scale*4.7) # previous CSS colors: lightcoral, cornflowerblue
        annokw = dict(horizontalalignment='center', verticalalignment='center'\
                     ,color='b', fontsize = 10)
        for i,x,y in nodeList:
            axTP.scatter(x,y,zorder=2,**marker_style)
            axTP.annotate(str(i), xy=(x,y), xytext=(x,y), **annokw)
            axET.scatter(x,y,zorder=2,**marker_style)
            axET.annotate(str(p_ary(i,p=p,L=n)), xy=(x,y), xytext=(x,y), **annokw)
        # draw edges
        if A is not None:
            # Edge Weight Coloring
            n_ = np.shape(A)[0]
            RGBA = np.zeros((round(n_*(n_-1)/2),4))
            RGBA[:,1] = 0.5 # for green (not sure if this is color 'g')
            counter = 0
            for i in range(0,n_):
                for j in range(i+1,n_): # undirected (A is symmetric)
                    # 🔴 assuming nodeList[i][0] = i
                    x = [nodeList[i][1],nodeList[j][1]]
                    y = [nodeList[i][2],nodeList[j][2]]
                    RGBA[counter,3] = 1 if A[i,j]>0 else 0 # set alpha to 1 (opaque) if edge exists
                    axTP.plot(x,y,color=RGBA[counter,:],zorder=1) # lower int means drawn on the canvas earlier
                    axET.plot(x,y,color=RGBA[counter,:],zorder=1) # lower int means drawn on the canvas earlier
                    counter += 1
        # Grid setting and save
        for ax in [axcbl,axcbm,axcbr]: # does not display empty colorbar
            ax.set_frame_on(False)
            ax.set_axis_off() # same as ax.axis('off')
        for i, ax in enumerate([axTP, axET]):
            ax.set_frame_on(False)
            ax.set_axis_off() # same as ax.axis('off')
            ax.axis('equal') # so that regular polygons appear to be regular as well
            if i < 1:
                ax.set_title(title[20:-12]+'Decimal Representation)', fontsize=17)
            else: # since axET is always GroundTruthOnly, title will just be GroundTruth
                ax.set_title(title[20:-12]+'Base {:d} Representation)'.format(p), fontsize=17)
            ax.grid(False)
        #plt.legend(loc='upper left')
        saveNclose427(fig,fname,dpi)
    else:
        for eig_k in eig_ks:
            # make figure using GridSpec (identical to the annotate one above)
            fig = plt.figure(figsize=[14,5]) # initialize
            ds = 0.2 # dummy axes for spacing between the visible plots
            cbW = 1 # colorbar width
            gs = GridSpec(nrows=3, ncols=9, height_ratios=[1,7,1]\
            , width_ratios=[cbW,ds,17,ds,cbW,ds,17,ds,cbW])
            axTP = fig.add_subplot(gs[:,2]) # main Sierpiński graph for Transition Probability
            axET = fig.add_subplot(gs[:,6]) # same Sierpiński graph but for Edge Type
            axcbl = fig.add_subplot(gs[1,0]) # Transition Probability colorbar
            axcbm = fig.add_subplot(gs[1,4]) # Node Type colorbar
            axcbr = fig.add_subplot(gs[1,8]) # Edge Type colorbar
            # eig_k specifics
            if eig_k is None:
                fname = fname[:temp_len] + ' (Vanilla)'
            else:
                eigtup = rank_eigvals(A, eig_k)
                fname = fname[:temp_len] + ' (#{:d} Eigvec)'.format(eig_k)
            # draw nodes
            for ax in [axTP, axET]:
                if eig_k is None: # use default PMMM theme color
                    marker_style = dict(facecolor='#f48ea5',edgecolor='#7f7596', marker='o'\
                                       ,alpha=1,s=scale) # previous CSS colors: lightcoral, cornflowerblue
                    ax.scatter([x for (_,x,_) in nodeList],[y for (_,_,y) in nodeList],zorder=2,**marker_style)
                else: # gradient of colors based on eigvec entry values
                    cmap = plt.cm.get_cmap('coolwarm', None)
                    scatt = ax.scatter([x for (_,x,_) in nodeList],[y for (_,_,y) in nodeList],zorder=2\
                             , c=eigtup[4], cmap=cmap, vmin=-1/np.sqrt(num_nodes), vmax=1/np.sqrt(num_nodes)\
                             , marker='o', alpha=1, s=scale)
                    cbm = fig.colorbar(scatt, cax=axcbm, extend='both')
                    cbrLabel427(axcbm, '#{:d} Eigenvector Entry Values'.format(eig_k))

                if pivots is not None:
                    for x,y in pivots:
                        ax.scatter(x,y,zorder=2,facecolor='none',edgecolor='b', marker='^', s=70)

            # draw edges
            if A is not None:
                # Edge Weight Coloring
                n_ = np.shape(A)[0]
                maxweight = 1.0/p
                if regType == 0: maxweight = 1
                RGBA = np.zeros((round(n_*(n_-1)/2),4))
                RGBA[:,1] = 0.5 # for green (not sure if this is color 'g')
                counter = 0
                for i in range(0,n_):
                    for j in range(i+1,n_): # undirected (A is symmetric)
                        # 🔴 assuming nodeList[i][0] = i
                        x = [nodeList[i][1],nodeList[j][1]]
                        y = [nodeList[i][2],nodeList[j][2]]
                        RGBA[counter,3] = A[i,j]/maxweight # set alpha
                        axTP.plot(x,y,color=RGBA[counter,:],zorder=1) # lower int means drawn on the canvas earlier
                        counter += 1
                cmap = LinearSegmentedColormap.from_list('custom green',[(1,1,1),(0,0.5,0)]) # from white to green [0,0.5,0]
                cbl = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0,vmax=maxweight), cmap=cmap)\
                           , cax=axcbl, format='%.2f')
                cbrLabel427(axcbl, 'Transition Probability')
                # Edge Type Coloring
                #colors = [(0.5, 0, 1), (0, 0.5, 0)]  # Purple -> (White) -> Green
                #colors = [(0.922, 0.420, 0.906), (0.110, 0.831, 0.314)]
                #colors = [(244/255, 142/255, 165/255), (127/255, 117/255, 150/255)] # same PMMM theme color (Madoka pink to Homura purple)
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color in pyplot
                cmap = LinearSegmentedColormap.from_list('custom edge color',colors[:nu],N=nu)
                cbr = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0,vmax=nu-1), cmap=cmap)\
                           , cax=axcbr, format='%.2f')
                cbr.set_ticks([(nu-1)*(2*i+1)/(2*nu) for i in range(nu)])
                cbr.set_ticklabels(np.arange(1,nu+1))
                cbrLabel427(axcbr, 'Edge Level')
                for lv in all_levels:
                    b_ = [x==lv for x in GTDict['lvList']] # boolean mask
                    b_edgeList = [e for (e,v) in zip(GTDict['edgeList'], b_) if v] # edges in level lv
                    xcoords,ycoords = np.zeros((2,len(b_edgeList))), np.zeros((2,len(b_edgeList)))
                    for i,(v_i,v_j) in enumerate(b_edgeList):
                        xcoords[:,i] = [nodeList[v_i][1],nodeList[v_j][1]]
                        ycoords[:,i] = [nodeList[v_i][2],nodeList[v_j][2]]
                    axET.plot(xcoords,ycoords,color=cmap(lv-1),zorder=1) # lower int means drawn on the canvas earlier

            if eig_k is None:
                for ax in [axcbm]: # does not display empty colorbar
                    ax.set_frame_on(False)
                    ax.set_axis_off() # same as ax.axis('off')
            for i, ax in enumerate([axTP, axET]):
                ax.set_frame_on(False)
                ax.set_axis_off() # same as ax.axis('off')
                ax.axis('equal') # so that regular polygons appear to be regular as well
                if i < 1:
                    ax.set_title(title, fontsize=17)
                else: # since axET is always GroundTruthOnly, title will just be GroundTruth
                    if beta is not None:
                        ax.set_title(title[:-14]+'GroundTruth)', fontsize=17)
                    else:
                        ax.set_title(title[:-12]+'GroundTruth)', fontsize=17)
                ax.grid(False)
            #plt.legend(loc='upper left')
            saveNclose427(fig,fname,dpi)


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


def cd427(dir_="E:/Lune/Study/Coding/Python3/GL/main427", show=False):
    import os # working directory management

    if dir_ is not None:
        if show:
            print("Previous cwd: ", os.getcwd())
            os.chdir(dir_)
            print("Current cwd after cd: ", os.getcwd())
        else:
            os.chdir(dir_)
    else: # use the dir_ the script is in
        dir_ = os.path.abspath(__file__)
        dir_ = dir_[0:dir_.rfind('\\')] # cut off the '\'+fname to get dir instead of path
        if show:
            print("Previous cwd: ", os.getcwd())
            os.chdir(dir_)
            print("Current cwd after cd: ", os.getcwd())
        else:
            os.chdir(dir_)

'''
    ########################################
        Miscellanious
    ########################################
'''

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
    main_Sierpinski427()
