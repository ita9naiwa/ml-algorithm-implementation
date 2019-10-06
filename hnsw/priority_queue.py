import copy
import heapq
from util import *
from node import Node


class priority_queue():
    def __init__(self, query_vec, farthest=False):
        self.heap = []
        self.set = set()
        self.query_vec = query_vec
        self.farthest = farthest
    def __repr__(self):
        return str(self.set)
    def append(self, node):
        if node not in self.set:
            dist = l2_dist(self.query_vec, node.vec)
            if self.farthest is True:
                dist = -dist
            heapq.heappush(self.heap, (dist, node))
            self.set.add(node)


    def top(self):
        return self.heap[0][1]

    def pop(self):
        ret = heapq.heappop(self.heap)[1]
        self.set.remove(ret)
        return ret

    def __len__(self):
        return len(self.set)

    def as_set(self):
        return copy.copy(self.set)

    def as_list(self, _sorted=False):
        ret = copy.deepcopy(self.set)
        if _sorted:
            sorted(ret)
        return ret

    def rebase(self, query_vec):
        """
        self.heap = []
        for node in self.set:
            dist = l2_dist(query_vec, node.vec)
            if self.farthest is True:
                dist = -dist
            heapq.heappush(self.heap, (dist, node))
        """
        ret_pq = priority_queue(query_vec, farthest=self.farthest)
        for node in self.set:
            ret_pq.append(node)
        return ret_pq

    def inverse(self):
        ret_pq = priority_queue(self.query_vec, farthest=not self.farthest)
        for node in self.set:
            ret_pq.append(node)
        return ret_pq

def list_as_pq(query_node, node_list):
    pq = priority_queue(query_node.vec)
    for node in node_list:
        pq.append(node)
    return pq
