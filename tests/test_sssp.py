import networkx as nx
import pytest

from sssp import ShortestPath
from sssp.dijkstra import dijkstra

CITIES_DATASET = pytest.DATA_DIR / "cities.csv"


@pytest.fixture(scope="session")
def cities():
    with open(CITIES_DATASET, "r") as f:
        # skip the csv header
        next(f, None)
        yield nx.parse_edgelist(
            f, delimiter=",",
            nodetype=str,
            data=(("weight", int),)
        )


@pytest.fixture(scope="session")
def shortest_path(cities):
    # this normalizes the graph before we benchmark so we get a better
    # comparison - in practice graphs would be normalized beforehand
    yield ShortestPath(cities)


@pytest.mark.benchmark
@pytest.mark.parametrize(
        "src,dest", [
            ("City1", "City123")
        ]
)
def test_sssp(benchmark, shortest_path: ShortestPath, src, dest):
    benchmark(shortest_path.shortest_path, src, dest)


@pytest.mark.benchmark
@pytest.mark.parametrize(
        "src,dest", [
            ("City1", "City123")
        ]
)
def test_dijkstra(benchmark, cities, src, dest):
    benchmark(dijkstra, graph=cities, src=src, dest=dest)
