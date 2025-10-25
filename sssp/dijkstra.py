import heapq
import math
import networkx as nx
from collections import defaultdict
from itertools import count

REMOVED = "<removed-node>"


def dijkstra(graph: nx.Graph, src, dest):
    frontier = []
    counter = count()
    dist = defaultdict(lambda: math.inf)
    dist[src] = 0

    pred = {src: None}

    src_entry = [0, next(counter), src]
    heapq.heappush(frontier, src_entry)
    lookup = {src: src_entry}

    while frontier:
        entry = heapq.heappop(frontier)
        node = entry[2]
        if node == REMOVED:
            continue

        for (u, v, w) in graph.edges(node, data="weight"):
            path_length = dist[u] + w
            if path_length < dist[v]:
                dist[v] = path_length
                pred[v] = u
                if v in lookup:
                    lookup[v][2] = REMOVED
                entry = [path_length, next(counter), v]
                lookup[v] = entry
                heapq.heappush(frontier, entry)

    path = []
    next_node = dest
    while True:
        next_node = pred[next_node]
        path.append(next_node)
        if next_node == src:
            break

    return dist[dest], path[::-1]
