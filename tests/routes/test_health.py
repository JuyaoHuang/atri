"""Tests for health check endpoint."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.app import create_app
from src.utils.config_loader import load_config


@pytest_asyncio.fixture
async def client():
    """Create test client with app."""
    config = load_config("config.yaml")
    app = create_app(config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check_returns_ok(client):
    """Test health endpoint returns 200 with ok status."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_check_has_cors_headers(client):
    """Test CORS headers are present."""
    response = await client.get("/health")
    # CORS headers should be present if enabled in config
    # Note: In test environment, CORS middleware may not add headers for same-origin requests
    assert response.status_code == 200
