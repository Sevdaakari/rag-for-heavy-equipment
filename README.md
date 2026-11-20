# RAG Agent for Heavy Equipment

A retrieval-augmented generation (RAG) agent that answers questions
about heavy mining equipment, built from scratch in Python — runs fully
offline via [Ollama](https://ollama.com), no cloud, no API keys.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com), with `qwen2:7b` and `nomic-embed-text` pulled

## Setup

```bash
# 1. Install Ollama: https://ollama.com/download

# 2. Pull the required models
ollama pull qwen2:7b
ollama pull nomic-embed-text

# 3. Set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Run it
python3 main.py
```

## Usage

Place your knowledge base documents (`.txt` or `.pdf`) in `docs/`, or
drop new files into `docs_upload/` and run the update step (details once
`docs_validator.py` is built). Then run `main.py` and ask questions in
the terminal.