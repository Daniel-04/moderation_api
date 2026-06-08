import os
import pytest
import torch
from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)


class FakeClassification:
    def __call__(self, text):
        return [{"label": "toxic", "score": 0.95}] if "despise" in text else [{"label": "normal", "score": 0.02}]


class FakeEmbedding:
    def encode(self, text, convert_to_tensor=False):
        val = hash(text) % 1000 / 1000.0
        emb = torch.tensor([val] * 384)
        return emb


@pytest.fixture(autouse=True)
def setup_env_and_mocks():
    with open("test_banned_words.txt", "w") as f:
        f.write("spam_link_example\n")
        f.write("you idiot\n")

    main.settings.banned_words_file = "test_banned_words.txt"
    main._banned_regex_cache = None
    main.ready = True
    main.loading_failed = False
    main.classification_pipeline = FakeClassification()
    main.embedding_model = FakeEmbedding()

    yield

    os.remove("test_banned_words.txt")
    main.settings.banned_words_file = ""


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
    assert data["status"] in ["delete", "inspect"]


def test_moderate_harmless_negative():
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
    assert "status" in response.json()
