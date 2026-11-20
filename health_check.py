import os
import json
import sqlite3
import ollama
from embeddings_cache import CACHE_PATH
from history import DB_PATH as HISTORY_DB_PATH

DOCS_FOLDER = "docs"
REQUIRED_MODELS = ["qwen2:7b", "nomic-embed-text"]


def check_ollama_running():
    try:
        ollama.list()
        return True, "Ollama is reachable"
    except Exception as e:
        return False, f"Cannot reach Ollama: {e}"


def check_models_pulled():
    try:
        installed = [m["model"] for m in ollama.list().get("models", [])]
    except Exception as e:
        return False, f"Could not list models: {e}"

    missing = [m for m in REQUIRED_MODELS if not any(m in name for name in installed)]
    if missing:
        return False, f"Missing models: {missing}"
    return True, "All required models are pulled"


def check_docs_folder():
    if not os.path.isdir(DOCS_FOLDER):
        return False, f"'{DOCS_FOLDER}' folder does not exist"
    if not os.listdir(DOCS_FOLDER):
        return False, f"'{DOCS_FOLDER}' folder is empty"
    return True, f"{len(os.listdir(DOCS_FOLDER))} document(s) found"


def check_embeddings_cache():
    if not os.path.exists(CACHE_PATH):
        return False, "No embeddings cache yet (will be built on first run)"
    try:
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
        return True, f"Cache readable, {len(data.get('chunks', []))} chunks"
    except Exception as e:
        return False, f"Cache file is corrupted: {e}"


def check_history_db():
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        conn.execute("SELECT COUNT(*) FROM messages")
        conn.close()
        return True, "history.db is readable"
    except sqlite3.OperationalError:
        return True, "history.db will be created on first run"
    except Exception as e:
        return False, f"history.db error: {e}"


def run_health_check():
    checks = [
        ("Ollama running", check_ollama_running),
        ("Required models pulled", check_models_pulled),
        ("docs/ folder", check_docs_folder),
        ("Embeddings cache", check_embeddings_cache),
        ("History database", check_history_db),
    ]

    all_ok = True
    print("=== Health Check ===")
    for name, check_fn in checks:
        ok, message = check_fn()
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {message}")
        all_ok = all_ok and ok

    print("\nAll checks passed." if all_ok else "\nSome checks failed — see above.")
    return all_ok


if __name__ == "__main__":
    run_health_check()