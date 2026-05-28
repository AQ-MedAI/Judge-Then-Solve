"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runapi", action="store_true", default=False, help="run tests that call remote APIs"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "api: mark test as requiring a remote API call")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_api = pytest.mark.skip(reason="need --runapi option to run")
    for item in items:
        if "slow" in item.keywords and not config.getoption('--runslow'):
            item.add_marker(skip_slow)
        if "api" in item.keywords and not config.getoption('--runapi'):
            item.add_marker(skip_api)
