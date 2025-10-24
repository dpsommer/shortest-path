import networkx as nx
import pytest

from sssp import shortest_path
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


@pytest.mark.benchmark
@pytest.mark.parametrize(
        "src,dest", [
            ("City1", "City123")
        ]
)
def test_sssp(benchmark, cities, src, dest):
    benchmark(shortest_path, graph=cities, src=src, dest=dest)


@pytest.mark.benchmark
@pytest.mark.parametrize(
        "src,dest", [
            ("City1", "City123")
        ]
)
def test_dijkstra(benchmark, cities, src, dest):
    benchmark(dijkstra, graph=cities, src=src, dest=dest)
