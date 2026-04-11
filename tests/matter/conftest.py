"""Shared fixtures for matter API tests."""

import pytest


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from irys.service.api import app, _active_matter_models
    _active_matter_models.clear()
    return TestClient(app), _active_matter_models
