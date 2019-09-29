from collections import defaultdict


class Node():
    def __init__(self, name, vec, level):
        self.name = name
        self.vec = vec
        self.neighbors = defaultdict(lambda: set())
        self.level = level

    def __repr__(self):
        return "node_" + self.name

    #def __eq__(self, other):
    #    return self.name == other.name

    def add_neighbor(self, in_node, layer):
        if in_node not in self.neighbors[layer]:
            self.neighbors[layer].add(in_node)

    def del_neighbor(self, out_node, layer):
        if out_node in self.neighbors[layer]:
            self.neighbors[layer].remove(out_node)

    def get_neighbors(self, layer):
        return self.neighbors[layer]

    def clear_neighbors(self, layer):
        self.neighbors[layer] = []
