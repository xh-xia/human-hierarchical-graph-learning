"""
This script is to generate .csv files in experiment/stims
"""
import numpy as np
import random
from utilities427.Sierpinski427 import p_ary, p2ten, find_CC_node, make_SierpinskiGraph427, W_norm

def main():
    cd427(show=1)
    for regType in [3]:
        GTDict = make_SierpinskiGraph427(3, 3, norm = False, regType = regType, use_set=True)
        make_Hamiltoniantasks(GTDict, regType = regType)

def make_Hamiltoniantasks(GTDict, regType=3):
    """
    This generates several csv files:
    1. nodes_Hamiltonian.csv: 500x1500
        entry: node label in decimal expansion for make_interspersed_walks
        row: a sequence of len=1500 walk
    2. isHamiltonian.csv: 500x1500
        entry: 0 if RW, 1 if Hamiltonian.
    3. crossCluster_Hamiltonian.csv: 500x1500
        entry: Hierarchical level: 1: coarsest, 2: higher level, etc.
    4. nodes_nBack_1.csv: 500x500
        entry: node label in decimal expansion for make_interspersed_walks
    5. queries_nBack_1.csv: 500x500
        entry: 0: no recall, 1/2: after this node we ask a 1/2-back query
        after every 20th trial, instead of seeing the 21th, subject
        will see n-back query (n=1,2), 10 1-back, 15 2-back, shuffled.
    """
    num_row, walk_length = 500, 1500 # if change 1500, also change make_interspersed_walks()
    arr1 = np.zeros((num_row,walk_length), dtype=int)
    arr2 = np.zeros((num_row,walk_length), dtype=int)
    arr3 = np.zeros((num_row,walk_length), dtype=int)
    arr4, arr5 = np.zeros((num_row,500), dtype=int), np.zeros((num_row,500), dtype=int)
    for row in range(num_row):
        arr1[row,], arr2[row,] = make_interspersed_walks(GTDict['A'], walk_length, regType=regType)
        arr3[row,] = make_HierLabels(arr1[row,], GTDict)
        arr4[row,] = random_walk(GTDict['A'], 500)
        arr5[row,] = make_nbackqueries(500)
    prefix = 'output/csv_files (regType={:d})'.format(regType)
    mkdir_p(prefix)
    np.savetxt(prefix+'/nodes_Hamiltonian'+'.csv', arr1, fmt='%d', delimiter=',')
    np.savetxt(prefix+'/isHamiltonian'+'.csv', arr2, fmt='%d', delimiter=',')
    np.savetxt(prefix+'/crossCluster_Hamiltonian'+'.csv', arr3, fmt='%d', delimiter=',')
    np.savetxt(prefix+'/nodes_nBack_1'+'.csv', arr4, fmt='%d', delimiter=',')
    np.savetxt(prefix+'/queries_nBack_1'+'.csv', arr5, fmt='%d', delimiter=',')

def make_nbackqueries(n, nList=[1,2], countList=[10,15]):
    """
    n: total length of the query
    """
    queries = np.repeat(nList,countList)
    random.shuffle(queries)
    query = np.zeros(n, dtype=int)
    for i in range(1,len(queries)+1):
        query[i*20-1] = queries[i-1]
    return query

def make_HierLabels(node_list, GTDict): # np.arr
    """
    This is kinda slow I think, since each time it will search through all (sparse) edges.
    """
    HierLabels = np.zeros(len(node_list), dtype=int)
    HierLabels[0] = 0 # convention, the edge before first node is level 0 (undefined level)
    for i in range(len(node_list)-1):
        HierLabels[i+1] = GTDict['lvList'][GTDict['edgeList'].index({node_list[i],node_list[i+1]})]
    return HierLabels

def make_interspersed_walks(A, walk_length, regType=3): # np.arr
    """
    5x244 RW + 5x56(2x28) = 1500 nodes RW + Hamiltonian Walk (regType=1)
    5x246 RW + 5x54(2x27) = 1500 nodes RW + Hamiltonian Walk (regType=0,3)
    """
    isHamiltonian = True
    len_RW = 244 if regType==1 else 246
    len_HT = 56 if regType==1 else 54
    walk = np.zeros(walk_length, dtype=int)
    isHamiltonian_arr = np.zeros(walk_length, dtype=int)
    walk[:len_RW] = random_walk(A, len_RW) # start at random
    i = len_RW # current index
    while i != walk_length:
        if isHamiltonian:
            walk[i-1:i+len_HT] = Hamiltonian_walk(len_HT+1, start_node = walk[i-1], regType=regType)
            isHamiltonian_arr[i:i+len_HT] = 1
            isHamiltonian = False
            i += len_HT
            continue
        else:
            walk[i-1:i+len_RW] = random_walk(A, len_RW+1, start_node = walk[i-1])
            isHamiltonian_arr[i:i+len_RW] = 0
            isHamiltonian = True
            i += len_RW
    return walk, isHamiltonian_arr

def random_walk(A, n, start_node=None): # node_list
    """
    Generate a random walk sequence starting at random.
    Args:
        A (nparr): adjacency matrix
        n (int): walk length
        start_node (int): starting node index
    Return:
        node_list: a list of nodes (a random walk sequence)
    """
    N = A.shape[0] # total number of nodes
    if start_node is None:
        start_node = random.randrange(0,N) # pick a starting node (uniformly) at random
    node_list = [start_node] # add to the walk sequence
    for _ in range(n-1):
        start_node = random.choice([i for i in range(N) if A[start_node,i]==1])
        node_list.append(start_node)
    return node_list

def Hamiltonian_walk(n, start_node, regType=1): # node_list
    cycle,_ = Hamiltonian_cycle(3, 3, regType=regType, adjacency=False)
    cycle.pop() # get rid of end point
    l = len(cycle)
    node_list = [start_node] # add start_node to the walk sequence
    ind_shift = cycle.index(start_node) # get the cycle index of start_node
    for i in range(1,n):
        node_list.append(cycle[(ind_shift+i) % l])
    return node_list


def Hamiltonian_cycle(p, n, regType = 3, adjacency=False): # node_list
    """ [WIP] also Hamiltonian_cycle on Sierpiński graph is non-trivial. Hence WIP.
    This for now is for Sierpiński graph only, and actually for p=3 only (and regType=0,1,3).
    I figured out the cycle myslef.
    Generate a Hamiltonian cycle sequence.
    This is to control for recency
    since Hamiltonian cycle/path on a Hamiltonian graph visits every node exactly once.
    (hence recency is NA since there is no last time visit.)
    Only those with Hamiltonian cycle (visit node exactly once) meet the recency NA need.
    S_p^n is Hamiltonian (i.e., has Hamiltonian cycle; p=3 the cycle is unique);
    not sure about regularized versions though.
    Args:
        p (int): base of Sierpiński graph (currently only p=3 is implemented)
        n (int): power of Sierpiński graph
        regType (int):
            Note: there are p nodes with deg=p-1 that may need to be regularized
                  these nodes have string like so: 'k'*n (k in [0,...,p-1])
            0: no regularization of boundary nodes
            1: type 1 regularization (add 1 node)
            2: type 2 regularization (add 1 make_SierpinskiGraph427(p, n-1))
            3: type 3 regularization (add self-loops to extreme nodes)
        adjacency (bool):
            if true, generate the underlying adjacency matrix
    Return:
        node_list: a list of nodes (a Hamiltonian cycle sequence)
    """
    if p==3:
        A = None
        node_idx = str(0)*n # start at 0^n
        node_list = list() # initialize walk sequence (decimal expansion)
        clockwise = (-1)**(n-1) # clockwise if +1, counter-clockwise if -1
        if regType==1:
            while node_idx != str(p-1)*n: # the last node in S_p^n (base10 index=p^n-1)
                node_idx = find_CC_node(node_idx) # go to next level-1 cluster
                node_list.append(p2ten(node_idx, p=p))
                node_idx = node_idx[0:-1] + str((int(node_idx[-1])+clockwise) % p)
                node_list.append(p2ten(node_idx, p=p))
                node_idx = node_idx[0:-1] + str((int(node_idx[-1])+clockwise) % p)
                node_list.append(p2ten(node_idx, p=p))
                clockwise *= -1 # reverse rotation direction
            node_list.append(p2ten(str(1)+str(0)*n, p=p)) # regType=1 extra node
            node_list.append(node_list[0]) # starting node
        elif regType in [0,3] and n==3: # p=3, n=3, a special case indeed
            # 000 & 001 (002 is for the second last in the cycle)
            node_list.append(p2ten(node_idx, p=p)) # 000
            node_idx = node_idx[0:-1] + str((int(node_idx[-1])+clockwise) % p)
            node_list.append(p2ten(node_idx, p=p)) # 001
            clockwise *= -1 # reverse rotation direction (specifically, now -1)
            for i in range(1,9):
                if i % 3 == 0: # extreme nodes cluster
                    clockwise *= -1 # reverse rotation direction to +1
                node_idx = find_CC_node(node_idx) # go to next level-1 cluster
                node_list.append(p2ten(node_idx, p=p))
                node_idx = node_idx[0:-1] + str((int(node_idx[-1])+clockwise) % p)
                node_list.append(p2ten(node_idx, p=p))
                node_idx = node_idx[0:-1] + str((int(node_idx[-1])+clockwise) % p)
                node_list.append(p2ten(node_idx, p=p))
                if i % 3 == 0: # extreme nodes cluster
                    clockwise *= -1 # reverse rotation direction back to -1
            node_list.append(2) # 002
            node_list.append(node_list[0]) # starting node 000
        else:
            raise ValueError("<regType> must be 0, 1, or 3. If 0 or 3, <n> has to be 3.")
        if adjacency:
            A = np.zeros((round(p**n+1),round(p**n+1)), dtype=int)
            for i in range(len(node_list)-1):
                A[node_list[i],node_list[i+1]] = 1
                A[node_list[i+1],node_list[i]] = 1 # undirected graph
        return node_list, A
    else:
        raise ValueError('p!=3 or regType not in [0,1,3].')



def cd427(dir_=None, show=False):
    import os, inspect # working directory management
    _cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    if dir_ is not None:
        if show:
            print("Previous cwd: ", os.getcwd())
            os.chdir(dir_)
            print("Current cwd after cd: ", os.getcwd())
        else:
            os.chdir(dir_)
    else: # change current working dir to where the file is
        if show:
            print("Previous cwd: ", os.getcwd())
            os.chdir(_cwd)
            print("Current cwd after cd: ", os.getcwd())
        else:
            os.chdir(_cwd)

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
    main()
