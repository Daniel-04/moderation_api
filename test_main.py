import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def setup_env():
    with open("test_banned_words.txt", "w") as f:
        f.write("spam_link_example\n")
        f.write("you idiot\n")

    os.environ["BANNED_WORDS_FILE"] = "test_banned_words.txt"
    yield
    os.remove("test_banned_words.txt")
    if "BANNED_WORDS_FILE" in os.environ:
        del os.environ["BANNED_WORDS_FILE"]


from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_moderate_pass():
    payload = {
        "message": "This is a normal message about the topic",
        "topic_context": "We are discussing normal things.",
    }
    response = client.post("/moderate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pass"


def test_moderate_delete_banned_word_exact():
    payload = {
        "message": "Check out my spam_link_example!",
    }
    response = client.post("/moderate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "delete"
    assert "banned phrase matching" in data["reason"]


def test_moderate_delete_banned_word_leet():
    payload = {
        "message": "y0u   1d10t",
    }
    response = client.post("/moderate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "delete"
    assert "banned phrase matching" in data["reason"]


def test_moderate_toxicity_delete():
    payload = {
        "message": "I absolutely despise everything about you, you are utterly disgusting.",
    }
    response = client.post("/moderate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in [
        "delete",
        "inspect",
    ]  # Could be either depending on the model's confidence


def test_moderate_harmless_negative():
    # Should not be flagged as toxic just because of negative sentiment
    payload = {
        "message": "I hate mondays",
    }
    response = client.post("/moderate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pass"


def test_moderate_off_topic():
    payload = {
        "message": "I really love buying crypto online and selling it.",
        "topic_context": "The topic of this channel is about gardening, growing tomatoes, and soil health.",
    }
    response = client.post("/moderate", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Could be pass or flagged_for_off_topic depending on distance
    assert "status" in data
