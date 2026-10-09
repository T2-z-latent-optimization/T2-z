from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-native", action="store_true", help="Run original seven-task simulator workers")
    parser.addoption("--native-output", default=None, help="Keep native simulator audit artifacts here")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-native"):
        for item in items:
            if "test_native.py" in str(item.path):
                item.add_marker(pytest.mark.skip(reason="Use --run-native to execute real simulators"))


@pytest.fixture
def native_output(request, tmp_path):
    supplied = request.config.getoption("--native-output")
    path = Path(supplied) / request.node.name if supplied else tmp_path
    path.mkdir(parents=True, exist_ok=True)
    return path
