import os
import re
import torch
from threading import Thread
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Everything good and models ready?
ready = False


def load_models():
    global embedding_model, classification_pipeline, ready
    embedding_model = load_embedding_model()
    classification_pipeline = load_classification_pipeline()
    ready = True
    print("Models loaded successfully.")


class Settings(BaseSettings):
    model_embedding: str = "all-MiniLM-L6-v2"
    model_classification: str = "unitary/toxic-bert"
    banned_words_file: str = ""

    # Toxicity thresholds
    threshold_pass: float = 0.70
    threshold_delete: float = 0.90

    # Topic mismatch threshold
    threshold_similarity_flag: float = 0.15

    class Config:
        env_file = ".env"


settings = Settings()


def build_banned_regex(banned_words: List[str]):
    if not banned_words:
        return None

    leet_map = {
        "i": "[i1l!]",
        "o": "[o0]",
        "e": "[e3]",
        "a": "[a4@]",
        "s": "[s5$]",
        "t": "[t7+]",
        "g": "[g9]",
        "b": "[b8]",
    }
    patterns = []
    for word in banned_words:
        word = word.lower().replace(" ", "")
        if not word:
            continue
        pattern_chars = []
        for char in word:
            c_pattern = leet_map.get(char, re.escape(char))
            pattern_chars.append(c_pattern)
        pattern = r"[\s\W_]*".join(pattern_chars)
        patterns.append(pattern)

    if not patterns:
        return None

    combined = "|".join(patterns)
    return re.compile(f"({combined})", re.IGNORECASE)


banned_words_list = []
if settings.banned_words_file and os.path.exists(settings.banned_words_file):
    with open(settings.banned_words_file, "r", encoding="utf-8") as f:
        banned_words_list = [line.strip() for line in f if line.strip()]

banned_regex = build_banned_regex(banned_words_list)


def get_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # tensorflow will literally lie about availability and throw an
        # exception if the GPU doesnt have the needed compute capabilities.
        t = torch.tensor([1.0], device="cuda:0")
        _ = t * 2
        return "cuda:0"
    except Exception as e:
        print(f"Cant use CUDA ({e}), falling back to CPU.")
        return "cpu"


def load_embedding_model():
    device = get_device()
    try:
        print(f"Loading embedding model {settings.model_embedding} on {device}...")
        return SentenceTransformer(settings.model_embedding, device=device)
    except RuntimeError as e:
        if "CUDA" in str(e) or "capability" in str(e).lower() or "sm_" in str(e):
            print(
                f"Failed to load embedding model on {device} ({e}). Falling back to CPU..."
            )
            return SentenceTransformer(settings.model_embedding, device="cpu")
        raise


def load_classification_pipeline():
    device = get_device()
    device_arg = 0 if device.startswith("cuda") else -1
    try:
        print(
            f"Loading classification model {settings.model_classification} on {device}..."
        )
        return pipeline(
            "text-classification",
            model=settings.model_classification,
            device=device_arg,
        )
    except RuntimeError as e:
        if "CUDA" in str(e) or "capability" in str(e).lower() or "sm_" in str(e):
            print(
                f"Failed to load classification model on {device} ({e}). Falling back to CPU..."
            )
            return pipeline(
                "text-classification", model=settings.model_classification, device=-1
            )
        raise


app = FastAPI(title="Moderation API Microservice", version="1.1.0")


class MessagePayload(BaseModel):
    message: str
    recent_context_messages: Optional[List[str]] = []
    topic_context: Optional[str] = None


class ModerationResult(BaseModel):
    status: str  # 'pass', 'inspect', 'delete', 'flagged_for_off_topic', 'flagged_for_toxicity'
    reason: Optional[str] = None
    similarity_score: Optional[float] = None
    classification_score: Optional[float] = None


@app.get("/health")
def health_check():
    if not ready:
        return Response(
            content='{"status":"loading"}',
            media_type="application/json",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return {"status": "ok"}


@app.post("/moderate", response_model=ModerationResult)
def moderate_message(payload: MessagePayload):
    if not ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    if banned_regex:
        match = banned_regex.search(payload.message)
        if match:
            return ModerationResult(
                status="delete",
                reason=f"Contains banned phrase matching: {match.group(0)}",
            )

    # Toxicity check
    classification = classification_pipeline(payload.message)[0]
    label = classification["label"].lower()
    score = classification["score"]

    if label in [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
        "label_1",
    ]:
        if score >= settings.threshold_delete:
            return ModerationResult(
                status="delete",
                reason="High probability of toxic content",
                classification_score=score,
            )
        elif score >= settings.threshold_pass:
            # Between pass threshold and delete threshold
            return ModerationResult(
                status="inspect",
                reason="Suspicious or mildy toxic content detected",
                classification_score=score,
            )

    # Embeddings
    similarity_score = None
    if payload.topic_context:
        topic_emb = embedding_model.encode(
            payload.topic_context, convert_to_tensor=True
        )
        msg_emb = embedding_model.encode(payload.message, convert_to_tensor=True)

        cosine_score = util.cos_sim(topic_emb, msg_emb).item()
        similarity_score = cosine_score

        if cosine_score < settings.threshold_similarity_flag:
            return ModerationResult(
                status="flagged_for_off_topic",
                reason="Low relevance to the provided topic_context",
                similarity_score=cosine_score,
            )

    # all checks pass
    return ModerationResult(status="pass", similarity_score=similarity_score)


Thread(target=load_models).start()
