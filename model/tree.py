from utils import DEPREL_TO_ID
import numpy as np


class Tree(object):
    def __init__(self):
        self.idx = None
        self.dep_rel = None
        self.dist = None
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head, len_, dep_rel):
    # Convert a sequence of head indexes into a tree object.
    assert len(head) == len_
    # head = head[:len_].tolist()
    root = None
    nodes = [Tree() for _ in head]
    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1  # just a filler
        nodes[i].dep_rel = dep_rel[i]
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])
    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=True, self_loop=False, identity_rel=DEPREL_TO_ID['special_rel']):
    #  Convert a tree object to an (numpy) adjacency matrix.
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    dep_rel_mat = np.zeros((sent_len, sent_len), dtype=np.int64)
    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]
        idx += [t.idx]
        for c in t.children:
            ret[t.idx, c.idx] = 1
            dep_rel_mat[t.idx, c.idx] = c.dep_rel
        queue += t.children
    if not directed:
        ret = ret + ret.T
        dep_rel_mat = dep_rel_mat + dep_rel_mat.T
    if self_loop:
        for i in idx:
            ret[i, i] = 1
            dep_rel_mat[i, i] = identity_rel
    return ret, dep_rel_mat