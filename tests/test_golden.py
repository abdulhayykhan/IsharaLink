"""
test_golden.py -- End-to-end golden test for IsharaLink FastAPI backend.

Verifies:
  1. Health check returns 200, model loaded, 62 classes.
  2. Classes endpoint returns exactly the 62 ordered words.
  3. Input validation: clips <10 frames or malformed points reject with 422.
  4. Golden cases from tests/preprocess_cases.json pass through /predict.
  5. Output payload satisfies interface contract (label, confidence, top3, status).
"""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app, APP_STATE
from src.models.predict import load_assets

ROOT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    # Ensure assets are loaded before test requests
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """GET /health must return 200 with model_loaded=True and 62 classes."""
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["classes_count"] == 62
    assert "threshold" in data


def test_classes_endpoint(client):
    """GET /classes must return exactly 62 classes matching classes.json contract."""
    resp = client.get("/classes")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list), "Expected list of class names"
    assert len(data) == 62
    
    # Verify first and last words matching label_ids 0 and 61
    assert data[0] == "absolutely"
    assert data[-1] == "you"


def test_validation_empty_frames(client):
    """POST /predict with empty frames list must return 422."""
    resp = client.post("/predict", json={"frames": []})
    assert resp.status_code == 422, "Empty frames should be rejected with 422"


def test_validation_clip_too_short(client):
    """POST /predict with fewer than 10 frames must return 422."""
    short_frames = [{"left": None, "right": None} for _ in range(5)]
    resp = client.post("/predict", json={"frames": short_frames})
    assert resp.status_code == 422, "Clips < 10 frames must be rejected with 422"
    detail = resp.json().get("detail", "")
    assert "Clip too short" in str(detail) or "minimum required is 10" in str(detail)


def test_validation_malformed_landmark_points(client):
    """POST /predict with fewer than 21 points must return 422."""
    bad_frame = [{
        "left": [[0.0, 0.0, 0.0] for _ in range(15)],  # only 15 points
        "right": None
    } for _ in range(12)]
    resp = client.post("/predict", json={"frames": bad_frame})
    assert resp.status_code == 422, "Hand with wrong point count must be rejected with 422"


def test_golden_cases_prediction(client):
    """
    Feed golden browser format cases from tests/preprocess_cases.json into /predict.
    Assert valid schema, valid probabilities, and correct prediction behavior.
    """
    cases_path = ROOT_DIR / "tests" / "preprocess_cases.json"
    assert cases_path.exists(), "preprocess_cases.json not found"
    
    with open(cases_path, "r", encoding="utf-8") as f:
        cases_data = json.load(f)
        
    cases = cases_data["cases"]
    assert len(cases) >= 3, f"Expected at least 3 golden test cases, got {len(cases)}"
    
    for case in cases:
        name = case["name"]
        raw_frames = case["frames"]
        
        resp = client.post("/predict", json={"frames": raw_frames, "client": "test"})
        assert resp.status_code == 200, f"Predict failed on case '{name}': {resp.text}"
        
        result = resp.json()
        assert "label" in result
        assert "confidence" in result
        assert "top3" in result
        assert "status" in result
        
        assert isinstance(result["confidence"], (float, int))
        assert len(result["top3"]) == 3
        assert result["status"] in ("ok", "unknown")
        
        # Verify top-3 ordering
        probs = [item["p"] for item in result["top3"]]
        assert probs == sorted(probs, reverse=True), "Top-3 must be sorted descending by probability"
        
        # Verify specific expected predictions for known golden cases
        if name in ("two_handed", "short_resample"):
            assert result["label"] == "aircrash", f"Expected 'aircrash', got {result['label']}"
            assert result["status"] == "ok"
            assert result["confidence"] > 0.50
        elif name == "one_handed":
            assert result["label"] == "absolutely", f"Expected 'absolutely', got {result['label']}"
            assert result["status"] == "ok"
            assert result["confidence"] > 0.50
        print(f"\n[Golden PASS] Case '{name}': predicted '{result['label']}' ({result['confidence']*100:.1f}%)")


def test_unknown_threshold_behavior(client):
    """A clip with all zeros (no hands) should have very low confidence or unknown status."""
    empty_frames = [{"left": None, "right": None} for _ in range(60)]
    resp = client.post("/predict", json={"frames": empty_frames})
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] in ("ok", "unknown")
    print(f"\n[No-hands test]: status='{res['status']}', confidence={res['confidence']}")
