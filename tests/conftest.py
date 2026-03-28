import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_transactions() -> list[dict]:
    sample_path = Path("tests/sample_input.json")
    return json.loads(sample_path.read_text(encoding="utf-8"))
