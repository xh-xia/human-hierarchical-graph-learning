"""
This generates a set of random walks.
"""
import random
import json
from collections import Counter
import networkx as nx
import numpy as np

N_SUBJECTS = 500
N_WALKS = 6
N_TESTS = 4
BLOCK_LENGTH = 250

_modular_list = {
    0: [1, 2, 3, 14],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [1, 2, 3, 5],
    5: [4, 6, 7, 8],
    6: [5, 7, 8, 9],
    7: [5, 6, 8, 9],
    8: [5, 6, 7, 9],
    9: [6, 7, 8, 10],
    10: [9, 11, 12, 13],
    11: [10, 12, 13, 14],
    12: [10, 11, 13, 14],
    13: [10, 11, 12, 14],
    14: [11, 12, 13, 0],
}
modular = nx.from_dict_of_lists(_modular_list)

# def random_walk_with_variations(G, n):
#     # Generate a shuffled list of image variations
#     variations_left = []
#     for i in range(15):
#         order = [0,1,2,3,4]*int(np.ceil(n/5))
#         np.random.shuffle(order)
#         variations_left.append(order)

#     walk = np.zeros(n)
#     variations = np.zeros(n)
#     node = random.choice(list(G.nodes()))
#     walk[0] = node
#     variations[0] = variations_left[node].pop()
#     for i in range(1,n):
#         node = random.choice(list(G[node].keys()))
#         walk[i] = node
#         variations[i] = variations_left[node].pop()
#     return walk, variations

def get_variations(walk):
    """
    Returns an array of variations for a walk

    Parameters
    ----------
    walk : array
        array of nodes

    Returns
    -------
    array
        Variations
    """

    # Make a shuffled list of variations for each node
    node_counter = Counter(walk)
    variations_left = {}
    for node, count in node_counter.iteritems():
        order = [0, 1, 2, 3, 4] * int(np.ceil(count / 5.0))
        np.random.shuffle(order)
        variations_left[node] = order

    # Iterate through the walk
    variations = np.zeros_like(walk)
    for i, node in enumerate(walk):
        variations[i] = variations_left[node].pop()

    return variations



def is_valid_walk(nodes, min_visits):
    """
    Make sure all 15 nodes are visited at least `min_visits` times
    """
    for i in range(15):
        if (nodes == i).sum() < min_visits:
            return False
    return True

def find_min_visits(nodes):
    """
    Find the minimum number of times any given node is visited
    """
    current_min = len(nodes)
    for i in range(15):
        num_visits = (nodes == i).sum()
        current_min = min(current_min, num_visits)
    return current_min

def find_min_distribution():
    """
    Find the distribution of minimum visits for walks of length 250
    """
    mins = []
    for _i in range(1000):
        starting_node = random.choice(list(modular.nodes()))
        walk = make_random_traversal(starting_node, BLOCK_LENGTH)
        mins.append(find_min_visits(walk))
    return mins

# With 250 trials, on average, each node should be visited 16.6 times
# 75% will visit every node at least 6 times, can be a decent baseline

def get_valid_walk():
    """
    Make a random traversal of `BLOCK_LENGTH`, and make sure it visits each node 6 times
    """
    valid_walk = False
    while not valid_walk:
        starting_node = random.choice(list(modular.nodes()))
        walk = make_random_traversal(starting_node, BLOCK_LENGTH)
        valid_walk = is_valid_walk(walk, 6)
    return walk

def make_hamiltonian_traversal(base_node, reverse):
    """
    Make a Hamiltonian traversal that starts at a node adjacent to `base_node`
    and ends at `base_node`

    Parameters
    ----------
    base_node : int
        Node prior to the hamiltonian walk
    reverse : boolean
        If true, nodes go in descending order, otherwise ascending

    Returns
    -------
    array
        Walk
    """

    traversal = [0]

    block = [1, 2, 3]
    random.shuffle(block)
    traversal += block
    traversal += [4, 5]

    block = [6, 7, 8]
    random.shuffle(block)
    traversal += block
    traversal += [9, 10]

    block = [11, 12, 13]
    random.shuffle(block)
    traversal += block
    traversal += [14]

    # reverse traversal
    if reverse:
        traversal = traversal[::-1]

    # set starting node to be one after `base_node`
    ind = np.where([x == base_node for x in traversal])[0][0]
    traversal = traversal[ind + 1:] + traversal[:ind + 1]

    return traversal

def make_random_traversal(base_node, n_steps):
    """
    Make a random walk whose prior node was `base_node`

    Parameters
    ----------
    base_node : int
        Prior node. First node of the walk will be adjacent to this.
    n_steps : int
        Length of the walk

    Returns
    -------
    array
        Walk
    """

    walk = np.zeros(n_steps, dtype=np.int)
    node = base_node
    for i in range(n_steps):
        node = random.choice(list(modular[node].keys()))
        walk[i] = node
    return walk

def get_mixed_hamiltonian_walk(reverse):
    """
    Make a walk of:
    30 trials random
    15 trials hamiltonian
    15 trials random

    Make sure we visit eacah node at least twice

    Parameters
    ----------
    forward : bool
        Whether the first hamiltonian walk is forward, or the second is.
    """
    good_walk = False
    while not good_walk:
        node = random.choice(list(modular.nodes()))
        walk = make_random_traversal(node, 30)
        # 15
        walk = np.concatenate([walk, make_hamiltonian_traversal(walk[-1], reverse)])
        # 30
        walk = np.concatenate([walk, make_random_traversal(walk[-1], 15)])
        counts = Counter(walk)
        good_walk = min(counts.values()) >= 2
    return walk

class NumpyEncoder(json.JSONEncoder):
    """
    Custom class to allow json encoding of a numpy object
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def gen_walks():
    """
    Main method to generate all walks.
    """
    np.random.seed(10)
    random.seed(10)

    all_walks = []
    for _subject in range(N_SUBJECTS):
        walks = []
        shape_checks = []
        tests = []
        for i in range(N_WALKS):
            # Main Walks
            nodes = get_valid_walk()
            variations = get_variations(nodes)
            walks.append(dict(nodes=nodes,
                              variations=variations,
                              ITI=0,
                              is_hamiltonian=np.array([False]*len(nodes))))

        for i in range(N_WALKS):
            # Shape Checks
            starting_node = random.choice(list(modular.nodes()))
            nodes = make_hamiltonian_traversal(starting_node, i % 2)
            variations = get_variations(nodes)
            # Generate a random sequence of 15 ITIs
            ITIs = np.array([2000, 3000, 4000] * 5)
            np.random.shuffle(ITIs)
            ITIs[0] = 1000
            shape_checks.append(dict(nodes=nodes,
                                     variations=variations,
                                     ITI=ITIs,
                                     trialLength=2000,
                                     is_hamiltonian=np.array([False]*len(nodes))))

        for i in range(N_TESTS):
            nodes = get_mixed_hamiltonian_walk(i % 2)
            variations = get_variations(nodes)
            # Generate a random sequence of 60 ITIs
            ITIs = np.array([2000, 3000, 4000] * 20)
            np.random.shuffle(ITIs)
            ITIs[0] = 1000
            is_hamiltonian = np.array([False] * 30 + [True] * 15 + [False] * 15)
            tests.append(dict(nodes=nodes,
                              variations=variations,
                              is_hamiltonian=is_hamiltonian,
                              trialLength=2000,
                              ITI=ITIs))
        all_walks.append(dict(walks=walks, tests=tests, shape_checks=shape_checks))
    with open('stims/shape_recall_walks.json', 'w') as fileobj:
        json.dump(all_walks, fileobj, cls=NumpyEncoder)

gen_walks()
