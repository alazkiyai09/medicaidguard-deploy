import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.rate_limit import rate_limiter


@pytest.fixture()
def client() -> TestClient:
    get_settings.cache_clear()
    rate_limiter.clear()
    with TestClient(app) as test_client:
        yield test_client
    rate_limiter.clear()
    get_settings.cache_clear()


@pytest.fixture()
def sample_transactions() -> list[dict]:
    sample_path = Path("tests/sample_input.json")
    return json.loads(sample_path.read_text(encoding="utf-8"))
