import heapq
import math
import networkx as nx
from collections import defaultdict
from itertools import count

REMOVED = "<removed-node>"

#   function Dijkstra(Graph, source):
#       Q ← Queue storing vertex priority
#
#       dist[source] ← 0                          // Initialization
#       Q.add_with_priority(source, 0)            // associated priority equals dist[·]
#
#       for each vertex v in Graph.Vertices:
#           if v ≠ source
#               prev[v] ← UNDEFINED               // Predecessor of v
#              dist[v] ← INFINITY                // Unknown distance from source to v
#              Q.add_with_priority(v, INFINITY)
#
#
#      while Q is not empty:                     // The main loop
#          u ← Q.extract_min()                   // Remove and return best vertex
#          for each arc (u, v) :                 // Go through all v neighbors of u
#              alt ← dist[u] + Graph.Edges(u, v)
#              if alt < dist[v]:
#                  prev[v] ← u
#                  dist[v] ← alt
#                  Q.decrease_priority(v, alt)
#
#      return (dist, prev)
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
