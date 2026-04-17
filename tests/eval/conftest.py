"""Pytest configuration for the eval harness.

Markers keep eval runs discoverable from the standard pytest collection while
letting callers run `pytest -m eval` for a fast, targeted slice.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "eval: MVP.1 deterministic evaluation harness fixtures",
    )
