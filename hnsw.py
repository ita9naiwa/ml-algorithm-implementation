from collections import defaultdict
import copy

import numpy as np

from priority_queue import priority_queue, list_as_pq
from node import Node
from util import *

class HNSW():
    def __init__(self):
        self.init_first = True
        self.enter_point = None
        self.m_l = 0.97
        self.max_level = -1
        self.ef_construction = 10
        self.M = 5
        self.M0 = 30
        self.nodes = []
    def insert(self, query_name, query_vec):
        if self.enter_point is None:
            node_level = self._draw_level()  # new element level, also top layer of the graph
            self.max_level = node_level
            query_node = Node(query_name, query_vec, node_level)
            self.enter_point = query_node
            if node_level > self.max_level:
                self.max_level = node_level
            self.nodes.append(query_node)
            return
        else:
            node_level = self._draw_level()  # new element level
            query_node = Node(query_name, query_vec, node_level)
        self.nodes.append(query_node)
        curr_level = node_level
        enter_point = self.enter_point
        visited = set()
        while curr_level > self.max_level:
            changed = True
            while changed is True:
                changed = False
                for neighbor in enter_point.get_neighbors(curr_level):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if node_dist(query_node, neighbor) < node_dist(query_node, enter_point):
                        enter_point = neighbor
                        changed = True
                        break
            curr_level -= 1

        enter_points = priority_queue(query_node.vec, False)
        enter_points.append(enter_point)

        for curr_level in reversed(range(1 + min(node_level, self.max_level))):
            if curr_level > 0:
                M = self.M
            else:
                M = self.M0
            W = self._search_layer(self.ef_construction, query_node, enter_points, curr_level)
            neighbors = self._select_neighbors(query_node, W.rebase(query_vec), curr_level)
            for neighbor in neighbors:
                if query_node != neighbor:
                    query_node.add_neighbor(neighbor, curr_level)
                    neighbor.add_neighbor(query_node, curr_level)
                    if len(neighbor.neighbors[curr_level]) > M:
                        newneighbors = self._select_neighbors(neighbor, neighbor.neighbors[curr_level], curr_level)
                        oldneighbors = copy.copy(neighbor.neighbors[curr_level])
                        for on in oldneighbors:
                            on.del_neighbor(neighbor, curr_level)
                            neighbor.del_neighbor(neighbor, curr_level)
                        for nn in newneighbors:
                            nn.add_neighbor(neighbor, curr_level)
                            neighbor.add_neighbor(nn, curr_level)

            enter_points = W

        if node_level > self.max_level:
            self.max_level = node_level
            self.enter_point = query_node

    def most_similar(self, query_vec, ef=50):
        query_node = Node("query", query_vec, 0)
        W = priority_queue(query_vec)
        ep = self.enter_point
        curr_level = ep.level
        enter_point = self.enter_point
        visited = set()
        while curr_level > 0:
            changed = True
            while changed is True:
                changed = False
                for neighbor in enter_point.get_neighbors(curr_level):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if node_dist(query_node, neighbor) < node_dist(query_node, enter_point):
                        enter_point = neighbor
                        changed = True
                        break
            curr_level -= 1
        enter_points = priority_queue(query_node.vec)
        enter_points.append(enter_point)

        W = self._search_layer(ef, query_node, enter_points, 0)
        neighbors = self._select_neighbors(query_node, W, 0)
        return [(node_dist(query_node, neighbor), neighbor) for neighbor in neighbors]


    def _select_neighbors(self, query_node, W, layer):
        if layer > 0:
            M = self.M
        else:
            M = self.M0
        if isinstance(W, list):
            _W = list_as_pq(query_node, W)
        else:
            _W = copy.copy(W)

        ret = []
        while len(_W) > 0 and len(ret) < M:
            p = _W.pop()
            if p != query_node:
                ret.append(p)
        return ret

    def _search_layer(self, ef, query_node, entry_points, layer):
        assert len(entry_points) > 0
        visited = entry_points.as_set()
        C = entry_points.rebase(query_node.vec)
        W = C.inverse()

        visited = set()
        while len(C)> 0:
            c = C.pop()
            f = W.top()

            if node_dist(c, query_node) > node_dist(f, query_node):
                break

            for neighbor in c.get_neighbors(layer):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                f = W.top()
                if len(W) < ef or node_dist(neighbor, query_node) < node_dist(f, query_node):
                    C.append(neighbor)
                    W.append(neighbor)
                    if len(W) > ef:
                        W.pop()

        return copy.copy(W)

    def _draw_level(self):
        return int(-np.log(np.random.uniform(0, 1)) * self.m_l)
        #return 2
        return 0



if __name__ == "__main__":
    h = HNSW()
    temp = []
    n_items = 1000
    for i in range(n_items):
        v = np.random.random(size=(10, ))
        temp.append(v)
        h.insert("%d" % i, v)

    ret = np.sum((np.array(temp) - v) **2, axis=1)
    print(sorted(enumerate(ret.tolist()), key=lambda x:x[1])[:5])
    print(sorted(h.most_similar(v, 10))[:5])

    #for i in range(n_items):
    #    print(i, dict(h.nodes[i].neighbors))
