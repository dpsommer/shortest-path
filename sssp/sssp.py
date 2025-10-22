# implementation of https://arxiv.org/pdf/2504.17033 using networkx
#
# pseudocode and header comments are verbatim from the paper for the most part
import heapq
import itertools
import math
from collections import defaultdict
from typing import List, Tuple

import networkx as nx

from .pathstore import PathStore

REMOVED = '<removed-node>'


class UnsortedShortestPaths:

    def __init__(self, graph: nx.Graph, degree: int, weight_key='weight'):
        self.graph = graph
        self.steps = math.floor(math.pow(math.log2(degree), 1/3))
        self._t = math.floor(math.pow(math.log2(degree), 2/3))
        self._weight_key = weight_key
        self._distances = defaultdict(math.inf)
        self._predecessors = {}

    def shortest_path(self, u, v):
        self._distances[u] = 0
        # TODO

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
    # P ← {u ∈ S : u is a root of a tree with ≥ k vertices in F }
    # return P, W
    def find_pivots(self, upper_bound: float, vertices: set) -> Tuple[set, set]:
        potential_vertices = set()
        step_vertices: List[set] = [vertices]
        subgraph_forest = nx.Graph()

        # relax edges for k steps
        for i in range(1, self.steps + 1):
            step_vertices.append(set())
            # find edges in the graph such that the source vertex is in the
            # subset of potential vertices from the previous step
            edges = self.graph.edges(step_vertices[i-1], data=self._weight_key)

            # walk the edges we found and relax if we find a shorter path
            for (u, v, w) in edges:
                path_distance = self._distances[u] + w
                if path_distance <= self._distances[v]:
                    self._distances[v] = path_distance
                    # TODO: currently not doing anything with this
                    self._predecessors[v] = u
                    if path_distance < upper_bound:
                        step_vertices[i].add(v)
                    # for each edge (u, v) in the broader graph, we create a forest
                    # of nodes such that u, v ∈ potential_vertices and
                    # distances[v] = distances[u] + weight(u, v) <- relaxed edge
                    # XXX: should this be global to the algorithm run?
                    subgraph_forest.add_edge(u, v, weight=w)

            # add all potential vertices (destination nodes of shortest paths) from
            # this step to the return set. then, if the count of potential vertices
            # exceeds steps * |W| > k|S|
            potential_vertices.union(step_vertices[i])
            if len(potential_vertices) > (self.steps * len(vertices)):
                return vertices, potential_vertices
        # FIXME: need a way to maintain a lookup of tree sizes rather than having
        # to run dfs on every subtree each call
        pivots = {u for u in vertices if len(nx.dfs_tree(subgraph_forest, u).nodes) >= self.steps}
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
    #       return B′ ← maxv∈U0 d[v], U ← {v ∈ U[0] : d[v] < B′}
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
        frontier = set(source_node)

        # run a miniature Dijkstra's on the first k steps of the search
        # to build a subset of frontier nodes
        while q and len(frontier) < self.steps + 1:
            d, _, u = heapq.heappop(q)
            if u == REMOVED:
                continue

            frontier.add(u)
            updated_upper_bound = max(updated_upper_bound, self._distances[u])

            for (u, v, w) in self.graph.edges(u, data=self._weight_key):
                path_length = d + w
                if path_length <= self._distances[v]:
                    self._distances[v] = path_length
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
        return updated_upper_bound, {v for v in frontier if self._distances[v] < updated_upper_bound}

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
            return self._bmssp_base_case(upper_bound, vertices)
        pivots, dests = self.find_pivots(upper_bound, vertices)
        store = PathStore(upper_bound, math.pow(2, (level-1)*self._t))

        next_bound = upper_bound
        for k, v in pivots:
            store.insert(k, v)
            next_bound = min(v, next_bound)

        i = 0
        paths = set()
        size_bound = self.steps * math.pow(2, level * self._t)
        store_min = upper_bound

        while len(paths) < size_bound and not store.is_empty():
            i += 1
            store_min, shortest_paths = store.pull()
            next_bound, s = self.bmssp(level - 1, store_min, shortest_paths)
            paths |= s
            batch = []

            for node in s:
                for (u, v, w) in self.graph.edges(node, data=self._weight_key):
                    path_length = self._distances[u] + w
                    if path_length <= self._distances[v]:
                        self._distances[v] = path_length
                        if next_bound >= path_length < upper_bound:
                            store.insert(v, path_length)
                        elif store_min >= path_length < next_bound:
                            batch.append((path_length, v))
            store.batch_prepend(batch)

        bound = min(store_min, upper_bound)
        paths |= {x for x in dests if self._distances[x] < bound}
        return bound, paths


def shortest_path(graph: nx.Graph, s, d, n, weight_key='weight'):
    return UnsortedShortestPaths(graph, n, weight_key).shortest_path(s, d)
