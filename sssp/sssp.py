# implementation of https://arxiv.org/pdf/2504.17033 using networkx
#
# pseudocode and header comments are verbatim from the paper for the most part
import copy
import heapq
import itertools
import math
from collections import defaultdict
from typing import List, Tuple

import networkx as nx

from .pathstore import PathStore

REMOVED = "<removed-node>"


class ShortestPath:

    def __init__(self, graph: nx.Graph, weight_key="weight"):
        self._weight_key = weight_key
        self._node_mapping = {}
        # store the edges from the original graph so we don't iterate over
        # empty nodes in the normalized graph
        self._edge_graph = nx.DiGraph()
        self.graph = self._normalize(graph)
        self._vertex_count = self.graph.number_of_nodes()
        self.steps = math.floor(math.pow(math.log(self._vertex_count), 1/3))
        # XXX: need a better name for this value
        self._t = math.floor(math.pow(math.log(self._vertex_count), 2/3))
        self._distances = defaultdict(lambda: math.inf)
        self._predecessors = {}

    def shortest_path(self, u, v):
        u = self._node_mapping[u]
        v = self._node_mapping[v]

        self._distances[u] = 0
        level = math.ceil(math.log(self._vertex_count) / self._t)
        # l = ⌈(log n)/t⌉, B = ∞, S = {s}
        self.bmssp(level, math.inf, {u})
        path = [v]
        next_node = v

        while True:
            next_node = self._predecessors[next_node]
            path.append(next_node)
            if next_node == u:
                break

        return self._distances[v], path[::-1]

    def _normalize(self, graph: nx.Graph) -> nx.Graph:
        G = nx.DiGraph()
        counter = itertools.count()
        max_degree = max(graph.degree, key=lambda x: x[1])[1] + 1
        x = defaultdict(lambda: next(counter))

        # Create a new graph composed of a constant-degree strongly-connected
        #   cycle for each vertex v in G and its neighbours
        for u in graph.nodes:
            nodes = [x[u]]
            for v in graph.neighbors(u):
                nodes.append(x[v])
            # pad node list so all nodes have matching in/out degrees
            while len(nodes) < max_degree:
                nodes.append(next(counter))
            for u in nodes:
                for v in nodes:
                    if u != v:
                        G.add_edge(u, v)
                        G.add_edge(v, u)

        # For every edge (u, v) in G, add a directed edge from vertex x(u, v)
        #   to x(v, u) with weight w(u, v)
        for (u, v, w) in graph.edges.data(self._weight_key):
            u, v = x[u], x[v]
            G.add_edge(u, v, **{self._weight_key: w})
            self._edge_graph.add_edge(u, v, **{self._weight_key: w})

        self._node_mapping = x
        return G

    # function FindPivots(B, S)
    # • requirement: for every incomplete vertex v with d(v) < B, the shortest
    #   path to v visits some complete vertex in S
    # • returns: sets P, W
    # W ← S
    # W[0] ← S
    # for i ← 1 to k do                         ▷ Relax for k steps
    #   W[i] ← ∅
    #   for all edges (u, v) with u ∈ W[i−1] do
    #       if d[u] + w(u,v) ≤ d[v] then
    #           d[v] ← d[u] + w(u,v)
    #           if d[u] + w(u,v) < B then
    #               W[i] ← W[i] ∪ {v}
    #   W ← W ∪ W[i]
    #   if |W| > k|S| then
    #       P ← S
    #       return P, W
    # F ← {(u, v) ∈ E : u, v ∈ W, d[v] = d[u] + w(u,v)}
    # P ← {u ∈ S : u is a root of a tree with ≥ k vertices in F}
    # return P, W
    def find_pivots(self, upper_bound: float, vertices: set) -> Tuple[set, set]:
        potential_vertices = vertices
        step_vertices: List[set] = [vertices]
        forest_trees = defaultdict(set)

        # relax edges for k steps
        for i in range(1, self.steps + 1):
            step_vertices.append(set())
            # find edges in the graph such that the source vertex is in the
            # subset of potential vertices from the previous step
            edges = self._edge_graph.edges(step_vertices[i-1], data=self._weight_key)

            # walk the edges we found and relax if we find a shorter path
            for (u, v, w) in edges:
                if w is None:
                    continue

                path_length = self._distances[u] + w
                if path_length <= self._distances[v]:
                    self._distances[v] = path_length
                    # update shortest path tree
                    self._predecessors[v] = u
                    if path_length < upper_bound:
                        step_vertices[i].add(v)
                    # rather than creating a separate forest graph, just cache
                    # subgraph trees as sets for when we determine pivots
                    forest_trees[u].add(v), forest_trees[v].add(u)
                    forest_trees[u] |= forest_trees[v]
                    forest_trees[v] |= forest_trees[u]

            # add all potential vertices (destination nodes of shortest paths)
            # from this step to the return set. then, if the count of potential
            # vertices exceeds steps * |W| > k|S|
            potential_vertices |= step_vertices[i]
            if len(potential_vertices) > (self.steps * len(vertices)):
                return vertices, potential_vertices
        pivots = {u for u in vertices if len(forest_trees[u]) >= self.steps}
        return pivots, potential_vertices


    # function BaseCase(B, S)
    # • requirement 1: S = {x} is a singleton, and x is complete
    # • requirement 2: for every incomplete vertex v with d(v) < B, the
    #   shortest path to v visits x
    # • returns 1: a boundary B′ ≤ B
    # • returns 2: a set U
    #
    # U[0] ← S
    # initialize a binary heap H with a single element ⟨x, d[x]⟩
    # while H is non-empty and |U[0]| < k + 1 do
    #   ⟨u, d[u]⟩ ← H.ExtractMin()
    #   U[0] ← U[0] ∪ {u}
    #   for edge e = (u, v) do
    #       if d[u] + w(u,v) ≤ d[v] and d[u] + w(u,v) < B then
    #           d[v] ← d[u] + w(u,v)
    #           if v is not in H then
    #               H.Insert(⟨v, d[v]⟩)
    #           else
    #               H.DecreaseKey(⟨v, d[v]⟩)
    #   if |U[0]| ≤ k then
    #       return B′ ← B, U ← U[0]
    #   else
    #       return B′ ← max v ∈ U[0] d[v], U ← {v ∈ U[0] : d[v] < B′}
    def _bmssp_base_case(self, upper_bound: float, vertices: set) -> Tuple[float, set]:
        updated_upper_bound = 0

        # initialize a minheap q to store the frontier nodes and distances
        q = []
        source_node = vertices.pop()

        # use a simple counter to track insert order to ensure priority "key"
        # uniqueness for equidistant paths
        counter = itertools.count()

        # we use [priority, count, node] list entries for the heap instead of
        # tuples as we need them to be mutable for the priority queue decrease
        # key operation
        #
        # for this to work with python's heapq implementation, we create a
        # lookup to map each node to its corresponding entry in the heap
        root = [self._distances[source_node], next(counter), source_node]
        node_index = {source_node: root}
        heapq.heappush(q, root)

        # frontier is the set of nodes with path lengths less than the
        # determined upper bound to be returned to the bounded
        # multi-source single path function for the next step
        frontier = {source_node}

        # run a miniature Dijkstra's on the first k steps of the search
        # to build a subset of frontier nodes
        while q and len(frontier) < self.steps + 1:
            d, _, u = heapq.heappop(q)
            if u == REMOVED:
                continue

            frontier.add(u)

            for (u, v, w) in self._edge_graph.edges(u, data=self._weight_key):
                if w is None:
                    continue
                path_length = d + w
                if path_length <= self._distances[v]:
                    self._distances[v] = path_length
                    # update shortest path tree
                    self._predecessors[v] = u

                entry = [path_length, next(counter), v]

                if v in node_index:
                    # "decrease" the priority value. due to python's heapq
                    # implementation not supporting priority modification,
                    # use lazy deletion to avoid breaking the heap invariant
                    # https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
                    node_index[v][-1] = REMOVED
                heapq.heappush(q, entry)
                node_index[v] = entry

        if len(frontier) <= self.steps:
            return upper_bound, frontier

        updated_upper_bound = 0
        for v in frontier:
            updated_upper_bound = max(updated_upper_bound, self._distances[v])

        return updated_upper_bound, {
            v for v in frontier if self._distances[v] < updated_upper_bound
        }

    # function BMSSP(l, B, S)
    # • requirement 1: |S| ≤ 2lt
    # • requirement 2: for every incomplete vertex x with d(x) < B, the
    # shortest path to x visits some complete vertex y ∈ S
    # • returns 1: a boundary B′ ≤ B
    # • returns 2: a set U
    #
    # if l = 0 then
    #   return B′, U ← BaseCase(B, S)
    # P, W ← FindPivots(B, S)
    # D.Initialize(M, B) with M = 2^(l−1)t
    # D.Insert(⟨x, d[x]⟩) for x ∈ P
    # i ← 0; B′0 ← minx∈P d[x]; U ← ∅          ▷ If P = ∅ set B′0 ← B
    # while |U| < k2^lt and D is non-empty do
    #   i ← i + 1
    #   Bi, Si ← D.Pull()
    #   B′i, Ui ← BMSSP(l − 1, Bi, Si)
    #   U ← U ∪ Ui
    #   K ← ∅
    #   for edge e = (u, v) where u ∈ Ui do
    #       if d[u] + w(u,v) ≤ d[v] then
    #           d[v] ← d[u] + w(u,v)
    #           if d[u] + w(u,v) ∈ [Bi, B) then
    #               D.Insert(⟨v, d[u] + w(u,v)⟩)
    #           else if d[u] + w(u,v) ∈ [B′i, Bi) then
    #               K ← K ∪ {⟨v, d[u] + w(u,v)⟩}
    #   D.BatchPrepend(K ∪ {⟨x, d[x]⟩ : x ∈ Si and d[x] ∈ [B′i, Bi)})
    # return B′ ← min{B′i, B}; U ← U ∪ {x ∈ W : d[x] < B′}
    def bmssp(self, level: int, upper_bound: float, vertices: set) -> Tuple[float, set]:
        if level == 0:
            return self._bmssp_base_case(upper_bound, copy.copy(vertices))
        pivots, dests = self.find_pivots(upper_bound, copy.copy(vertices))
        block_size = int(math.pow(2, (level-1)*self._t))
        store = PathStore(upper_bound, block_size)

        next_bound = upper_bound
        for node in pivots:
            store.insert(node, self._distances[node])
            next_bound = min(self._distances[node], next_bound)

        paths = set()
        size_bound = self.steps * math.pow(2, level * self._t)
        store_min = upper_bound

        while len(paths) < size_bound and not store.is_empty():
            store_min, shortest_paths = store.pull()
            if store_min == math.inf:
                store_min = upper_bound
            next_bound, s = self.bmssp(level - 1, store_min, shortest_paths)
            paths |= s
            batch = []

            for node in s:
                for (u, v, w) in self._edge_graph.edges(node, data=self._weight_key):
                    if w is None:
                        continue
                    path_length = self._distances[u] + w
                    if path_length <= self._distances[v]:
                        self._distances[v] = path_length
                        # update shortest path tree
                        self._predecessors[v] = u

                        if store_min <= path_length < upper_bound:
                            store.insert(v, path_length)
                        elif next_bound <= path_length < store_min:
                            batch.append((v, path_length))

            for node in shortest_paths:
                path_length = self._distances[node]
                # minor but important correction here:
                #
                # D.BatchPrepend(K ∪ {⟨x, d[x]⟩ : x ∈ Si and d[x] ∈ [B′i, Bi)})
                #
                # the bounds for d[x] need to be (B′i, Bi) to avoid an infinite
                # loop when d[x] == B′i
                if next_bound < path_length < store_min:
                    batch.append((node, path_length))
            store.batch_prepend(batch)

        bound = min(next_bound, upper_bound)
        paths |= {x for x in dests if self._distances[x] < bound}
        return bound, paths


def shortest_path(graph: nx.Graph, src, dest, weight_key="weight"):
    return ShortestPath(graph, weight_key).shortest_path(src, dest)
