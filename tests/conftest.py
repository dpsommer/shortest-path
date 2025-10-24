from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent.absolute()


def pytest_addoption(parser):
    parser.addoption("--benchmark", action="store_true", default=False, help="run benchmark tests")


def pytest_configure(config):
    pytest.DATA_DIR = TEST_DIR / "data"
    config.addinivalue_line("markers", "benchmark: mark benchmark tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--benchmark"):
        return
    benchmark_skip_marker = pytest.mark.skip(reason="use --benchmark marker to run")
    for item in items:
        filename = Path(str(item.fspath)).name
        if "benchmark" in item.keywords or filename.startswith('test_benchmark'):
            item.add_marker(benchmark_skip_marker)