import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="also run the tests that download the real example data",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "network: downloads real data (run with --network)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="use --network to test the real download")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
