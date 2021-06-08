"""
This script is to generate GroundTruth of Sierpiński for Sierpinski_Graph.py.
It was originally in Sierpinski_Graph.py (up to folder 7-6).
On 2021.1.13 at nearly 5 PM (EST) I moved it here.
"""

import numpy as np
from copy import deepcopy

import sys, os, inspect
temp_cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, temp_cwd)
from math427 import W_norm


'''
    ########################################
        Helper Classes 427
    ########################################
'''
class hierList:
    # only the bottom level list of the resursion is list of actual coordinates (x,y)
    # every other level list is list of objects hierList
    def __init__(self): # constructor
        ''' About index_start & index_call (both are starting indices)
        [O4,O5,O6,O7,O8,O9] # doesn't matter if Ox is object or numbers
        index_start = 2:
        self.Elements()=[O6,O7,O8,O9,O4,O5]
        index_call = 4 = (self.length - self.index_start) % self.length
        '''
        self.length=0 # length of this hierList
        self.index_start=0 # when we index hlist, we start at this index
        self.index_call=0 # this is where orginal label is
        self.hlist=list() # list of objects or coordinates

    def add_nodes(self, ini_list): # finest level nodes; smallest regular graph
        ''' e.g. for simple S(p=3,n=1)
        ini_list = [[0,0], [1/2,1], [1,0]] # notice index is correct: 0,1,2
        '''
        self.hlist.extend(ini_list)
        self.length += len(ini_list)
    def add_hierList(self, hierList_): # any higher level grouping (AKA object)
        self.hlist.append(hierList_)
        self.length += 1

    def Element_i(self, i): # this is one of the key of list rotation: through indexing
        return self.hlist[(self.index_start+i) % self.length]
    def Elements(self):
        return [self.hlist[(self.index_start+i) % self.length] for i in range(self.length)]
    def CallElement_i(self, i): # this is one of the key of list rotation: through indexing
        return self.hlist[(self.index_call+i) % self.length]
    def CallElements(self):
        return [self.hlist[(self.index_call+i) % self.length] for i in range(self.length)]
    def Coordinates(self): # retrieve the innermost nested coordinates; recursively update coorlist
        coorlist = list()
        if isinstance(self.hlist[0],hierList): # check if any element of hlist is also hierList
            for i in range(self.length):
                coorlist.extend(self.CallElement_i(i).Coordinates())
        else:
            return self.CallElements()
        return coorlist

    ''' # this is not deep, troublesome if the elements are themselves objects
    def copyElement_i(self, i): # copy i-th element and append to hlist
        self.hlist.append([self.Element_i(i)[0], self.Element_i(i)[1]])
        self.length += 1
    '''
    ''' Below override is not necessary... ?
    def __deepcopy__(self, memo): # deepcopy this class itself
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    '''
    def copyElement_i(self, i): # deepcopy i-th element and append to hlist
        self.hlist.append(deepcopy(self.Element_i(i)))
        self.length += 1

    def hlist_Rotate(self, stepsize, i_=None): # list index rotation (not coordinates rotation)
        ''' default LEFT rotate (LEFT shift) n times
        say self.length = 4:
        self.hlist=[a b c d]
        stepsize=1:  [b c d a]
        stepsize=2:  [c d a b]
        stepsize=5:  [b c d a]
        say if a,b,c,d are also hierList objects (continue):
        stepsize=1: b.hlist_Rotate(4)

        BUT when we retrieve them, the starting index should be at "a"
        '''
        if i_ is None: # rotate all
            # 1st rotate the higher level hlist
            self.index_start = (self.index_start+stepsize) % self.length
            self.index_call = (self.length - self.index_start) % self.length
            # 2nd check if any element of hlist is also hierList; if so, element.hlist_Rotate(stepsize)
            if isinstance(self.hlist[0],hierList):
                for i in range(self.length):
                    self.hlist[i].hlist_Rotate(stepsize) # rotate element, but this is resursive
        else: # rotate i_-th member only
            i = (self.index_start+i_) % self.length
            # nothing to rotate if it is not an hierList (i.e., self.hlist[i]=4)
            if isinstance(self.hlist[i],hierList):
                self.hlist[i].hlist_Rotate(stepsize) # rotate element, but this is resursive

    def Rotate(self, theta, pivot, i_=None): # coordinates rotation (only rotate if elements are coor)
        ''' RIGHT rotation
        after making a copy (in another class), we can now rotate around pivot by theta
        pivot is a vector [xj,yj]
        if i_∈[0,...,self.length-1], rotate that element only, and rotation order matters in this case

        🔴 assume elements/members of the hlist is of homogeneous type
        '''
        if i_ is None: # rotate all
            if isinstance(self.hlist[0],hierList): # check if any element of hlist is also hierList
                for i in range(self.length):
                    self.hlist[i].Rotate(theta, pivot) # we don't care about order of rotation
            else: # hlist is bottom most list (i.e., self.hlist=[[x,y],...])
                SO = np.sin(theta)
                CO = np.cos(theta)
                # again we don't care about order of rotation
                for i in range(self.length):
                    self.hlist[i] = [CO*(self.hlist[i][0]-pivot[0]) + SO*(self.hlist[i][1]-pivot[1]) + pivot[0]\
                                  ,- SO*(self.hlist[i][0]-pivot[0]) + CO*(self.hlist[i][1]-pivot[1]) + pivot[1]]
        else: # rotate i_-th member only
            i = (self.index_start+i_) % self.length
            if isinstance(self.hlist[i],hierList): # check if that element of hlist is also hierList
                self.hlist[i].Rotate(theta, pivot)
            else:
                SO = np.sin(theta)
                CO = np.cos(theta)
                self.hlist[i] = [CO*(self.hlist[i][0]-pivot[0]) + SO*(self.hlist[i][1]-pivot[1]) + pivot[0]\
                              ,- SO*(self.hlist[i][0]-pivot[0]) + CO*(self.hlist[i][1]-pivot[1]) + pivot[1]]

class make_Sierpinski427:
    def __init__(self, p, n, x0 = [0.0,0.0], s0=1.0 , c=1.0, regType = 0): # constructor; n is vertex name/index
        '''
        regType (int):
            Note: there are p nodes with deg=p-1 that may need to be regularized (i.e., homogenerous degree)
                  these nodes have string like so: 'k'*n (k in [0,...,p-1])
            0: no regularization of boundary nodes
            1: type 1 regularization (add 1 node)
            2: type 2 regularization (add 1 make_SierpinskiGraph427(p, n-1, use_set=use_set))
            3: type 3 regularization (add self-loop to extreme nodes)
        '''
        self.p, self.n = p, n # base & power
        self.x0=x0 # the coordinate of bottom left node
        self.s0=s0 # stage 0 shift (this stage does not have offset)
        self.c=c # constant offset value that adds to the previous radius to make the new shift length
        self.regType=regType
        #self.hierList = None
        self.nodeList = None # [(i,x,y)]
        self.c_arr=None # variable offset value that adds to the previous radius to make the new shift length
        self.radii=None
        self.shift_len=None
        self.pivots = None
        ''' ↓ calculates angles (constant) ↓
        theta_s: shift angle, a constant scalar
        theta_r: rotate angle, a list of angles for rotation at any stage, len=p-1
        '''
        self.theta_s = (p-2)/(2*p)*np.pi
        self.theta_r = [m*2*np.pi/p for m in range(1,p)]

    def set_params(self):
        ''' ↓ calculates c at all stages j∈[0,...,n-1] ↓ '''
        if self.p <=3:
            self.c_arr = [self.c for i in range(1,self.n)]
        else:
            self.c_arr = [self.c+(i**2-0.7*i+0.24)*0.618*(self.p-4)**2 for i in range(1,self.n-1)]
        self.c_arr.insert(0,0) # stage 0 has 0 c
        self.c_arr.append(self.c_arr[-1]*self.n*0.74) # last stage has large c
        ''' ↓ calculates radius & shift length at all stages j∈[0,...,n-1] ↓ '''
        self.radii = [2**j*self.s0 \
        + sum([2**(j-i)*self.c_arr[i] for i in range(1,j+1)]) for j in range(0,self.n)]
        self.shift_len = [self.s0] + [self.radii[j-1] + self.c_arr[j] for j in range(1,self.n)] # never used radii[n-1]
        ''' ↓ calculates pivot at all stages j∈[0,...,n-1] ↓ '''
        p0 = (self.x0[0]+self.shift_len[0]*np.cos(self.theta_s)\
             ,self.x0[1]+self.shift_len[0]*np.sin(self.theta_s))
        self.pivots = [p0 for j in range(self.n)]
        for j in range(1,self.n):
            self.pivots[j] = (self.pivots[j-1][0]+self.shift_len[j]*np.cos(self.theta_s)\
                            , self.pivots[j-1][1]+self.shift_len[j]*np.sin(self.theta_s))

    def Layout_Sierpinski427(self):
        self.set_params()
        LO_hier = hierList()
        LO_hier.add_nodes([self.x0])
        for j in range(0, self.n): # stage j
            LO = deepcopy(LO_hier)
            LO_hier = hierList()
            LO_hier.add_hierList(LO)
            if j==self.n-1 and self.regType == 2: # recalculate some variables
                temp_theta_r = [m_*2*np.pi/(self.p+1) for m_ in range(1,self.p+1)]
                temp_theta_s = (self.p+1-2)/(2*(self.p+1))*np.pi
                temp_pivot = (self.pivots[j-1][0]+0.9*self.radii[j]*np.cos(temp_theta_s)\
                            , self.pivots[j-1][1]+0.9*self.radii[j]*np.sin(temp_theta_s))
                for m in range(1, self.p+1): # m-th copy
                    LO_hier.copyElement_i(0)
                    LO_hier.Rotate(temp_theta_r[m-1], temp_pivot, i_ = m)
                    LO_hier.hlist_Rotate(m, i_=m)
                break
            for m in range(1, self.p): # m-th copy
                LO_hier.copyElement_i(0)
                LO_hier.Rotate(self.theta_r[m-1], self.pivots[j], i_ = m)
                LO_hier.hlist_Rotate(m, i_=m)

        self.nodeList = [(i,x,y) for i,(x,y) in enumerate(LO_hier.Coordinates())]
        if self.regType == 1:
            self.nodeList.append((len(self.nodeList)\
                                , self.pivots[self.n-1][0], self.pivots[self.n-1][1]))

'''
    ########################################
        Helper Functions 427
    ########################################
'''

def p_ary(n, p=2, L=None):
    ''' 🔴 assume 2<=p<=10
    Args
    -------
    n (int): base 10 integer
    p (int): string base (base; n ∈ {2,3,...})
    L (int): pad leading zeros s.t. the length is L (if L<=len(l_), keep l_)
    Return
    -------
    l_ (string): p-ary string
    '''
    if n == 0:
        if L is None:
            return '0'
        else:
            leading_0s = ['0' for i in range(L)]
            return ''.join(leading_0s)
    q=n # quotient
    l_=[] # will be converted to a string at the end
    i=0 # index of l_ (note: starts from the right and append to the left!)
    while q!=0:
        l_ = [str(q%p)] + l_ # remainder is the value in ith place from right
        q = q // p # the next integer to expand is the last quotient
        i+=1
    if L is None:
        return ''.join(l_)
    else:
        leading_0s = ['0' for i in range(L-len(l_))]
        return ''.join(leading_0s+l_)

def p2ten(str_, p=2):
    ''' 🔴 assume 2<=p<=10, this converts to decimal expansion
    Arg:
        str_ (str): p-ary string
        p (int): string base (base; n ∈ {2,3,...})
    Return:
        n (int): base 10 integer of str_
    '''
    n=0
    for k in range(len(str_)):
        n+=int(str_[-k-1]) * p**k
    return n

def find_CC_node(str_):
    """ CC=Cross-Cluster
    This is to find the next node that is connected by
    the level lv edge (lv!=1) of a given node represented in base p.
    Arg:
    -------
    str_ (str): sij^k

    Return:
    -------
    sji^k

    """
    if int(str_)==0: return str_ # return itself if str_ is just 0
    s,i,j,k = str(0), str(0), str_[-1], 1
    for n in range(1, len(str_)):
        s = str_[0:-1-n] # -1-n char is excluded, as python does with end point
        if str_[-1-n]!=j:
            i = str_[-1-n]
            break
        k += 1
    return s+j+i*k

def make_SierpinskiGraph427(p, n, norm = False, regType = 0, use_set=False):
    '''
    Arg:
        p (int): string base (base; n ∈ {3,4,...})
        n (int): length of p-ary string (exponent; n ∈ {0,1,...})
        norm (bool): if True, uses probability as weights
        regType (int):
            Note: there are p nodes with deg=p-1 that may need to be regularized
                  these nodes have string like so: 'k'*n (k in [0,...,p-1])
            0: no regularization of boundary nodes
            1: type 1 regularization (add 1 node)
            2: type 2 regularization (add 1 make_SierpinskiGraph427(p, n-1, use_set=use_set))
            3: type 3 regularization (add self-loop to extreme nodes)
        use_set (bool):
            if True, use set ({i,j}) as item in edgeList instead of tuple.
    N := p^n
    edgeList (a list of size-2 tuples (v_i,v_j))
        node index in edgeList is simply p2ten(s, p=p)
        where s is nodel p-ary string label
    lvList (a list of hierarchy labels): finest level is 1 (🔴 New Convention! Previously n is finest)
    Return:
        A (np.array; (N,N) for unregularized): adjacency matrix
    '''
    N = round(p**n)
    A = np.zeros((N,N),dtype=int)
    lvs = list(range(1,n+1))
    edgeList = []
    lvList = []
    # num of lööps: 1/2 * p(p^n-1)
    for k in range(n):
        for s_10 in range(round(p**(n-k-1))):
            for i in range(0,p-1):
                for j in range(i+1,p):
                    v_i = p2ten(p_ary(s_10,p=p)+str(i)+str(j)*k,p=p)
                    v_j = p2ten(p_ary(s_10,p=p)+str(j)+str(i)*k,p=p)
                    A[v_i,v_j] = 1
                    A[v_j,v_i] = 1 # undirected graph
                    if use_set:
                        edgeList.append({v_i,v_j})
                    else:
                        edgeList.append((v_i,v_j))
                    lvList.append(lvs[k])
    if regType == 0:
        pass
    elif regType == 1:
        A = np.block([[A, np.zeros((N,1),dtype=int)],
                      [np.zeros((1,N),dtype=int), np.zeros((1,1),dtype=int)]])
        for k in range(p):
            temp=p2ten(str(k)*n,p=p)
            A[temp,-1] = 1
            A[-1,temp] = 1 # undirected
            if use_set:
                edgeList.append({temp,N})
            else:
                edgeList.append((temp,N))
            lvList.append(n+1) # level is n+1 (coarsest level edges) for extra p edges in regType=1 (previously -427)
    elif regType == 2:
        N_ = round(p**(n-1))
        GTDict = make_SierpinskiGraph427(p, n-1, use_set=use_set)
        A = np.block([[A, np.zeros((N,N_),dtype=int)],
                      [np.zeros((N_,N),dtype=int), GTDict['A']]])
        if use_set:
            GTDict['edgeList'] = [{x+N,y+N} for (x,y) in GTDict['edgeList']] # shift index by N
        else:
            GTDict['edgeList'] = [(x+N,y+N) for (x,y) in GTDict['edgeList']] # shift index by N
        edgeList.extend(GTDict['edgeList'])
        lvList.extend(GTDict['lvList'])
        for k in range(p): # connect this new level-1 community with previous level-1 communities
            v_i, v_j = p2ten(str(k)*n,p=p), p2ten('10'+str(k)*(n-1),p=p)
            A[v_i,v_j] = 1
            A[v_j,v_i] = 1 # undirected
            if use_set:
                edgeList.append({v_i,v_j})
            else:
                edgeList.append((v_i,v_j))
            lvList.append(n) # these new edges are coarsest level edges
    elif regType == 3:
        for k in range(p):
            temp=p2ten(str(k)*n,p=p)
            A[temp,temp] = 1 # self_loop
            if use_set:
                edgeList.append({temp,temp})
            else:
                edgeList.append((temp,temp))
            lvList.append(-1) # level is -1 (undefined)
    else:
        raise ValueError('<regType> is unclear.')

    if norm: # if true, we normalize the nparr
        A = W_norm(A)
    return dict(A=A, edgeList=edgeList, lvList=lvList, n=n)


