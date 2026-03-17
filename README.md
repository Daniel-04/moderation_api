# Moderation API Microservice

A microservice for moderating chat messages using local NLP models. Handles wordlist-based filtering (including leetspeak detection), toxicity classification, and topic relevance checking using vector embeddings.

## Filtering
- *Toxicity Classification*: Uses `unitary/toxic-bert` by default to score toxicity, insults, threats, etc.
- *Topic Relevance*: Uses `sentence-transformers/all-MiniLM-L6-v2` to compute the cosine similarity between the current message and the topic context, returning flags for off-topic messages.
- *Leetspeak filtering*: A regex is built from wordlist to match banned words fast.

## Installation

1. Initialize a virtual environment and install dependencies:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt  # (or install fastapi, uvicorn, transformers, sentence-transformers, torch, pydantic-settings)
```

## Configuration

Configuration is handled via environment variables.

| Variable | Default Value | Description |
|---|---|---|
| `MODEL_EMBEDDING` | `all-MiniLM-L6-v2` | Sentence-transformer model for topic relevance. |
| `MODEL_CLASSIFICATION` | `unitary/toxic-bert` | Huggingface model for toxicity detection. |
| `BANNED_WORDS_FILE` | `""` | Path to a text file containing one banned expression per line. |
| `THRESHOLD_PASS` | `0.70` | Score below which toxicity is ignored. |
| `THRESHOLD_DELETE` | `0.90` | Score above which message is explicitly flagged for deletion. |
| `THRESHOLD_SIMILARITY_FLAG`| `0.15` | Cosine similarity below which a message is considered off-topic. |

## Usage
Start the development server using Uvicorn:

```bash
uvicorn main:app --reload
```
By default, the API runs on `http://127.0.0.1:8000`.

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Example Requests
Send a POST request to `/moderate` with the JSON payload.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/moderate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "I really enjoy learning about Django, its a great framework.",
  "topic_context": "We discuss Python web development and frameworks like Django or Flask."
}'
```

```bash
# Toxic Example
curl -X 'POST' \
  'http://127.0.0.1:8000/moderate' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "You are a complete idiot."
}'
```

### Example Responses
```json
{
  "status": "pass",
  "reason": null,
  "similarity_score": 0.548,
  "classification_score": null
}
```

``` json
{
  "status": "delete",
  "reason": "High probability of toxic content",
  "classification_score": 0.986
}
```
