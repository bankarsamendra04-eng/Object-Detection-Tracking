import sys
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.core.config import settings


@pytest.fixture(scope="session")
def client():
    """Provides a TestClient fixture for API testing."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_frame():
    """Provides a synthetic 640x480 RGB image for vision testing."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a rectangle pattern
    img[100:300, 150:400] = [200, 100, 50]
    return img
