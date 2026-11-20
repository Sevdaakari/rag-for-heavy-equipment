import os
import shutil
from pypdf import PdfReader

ALLOWED_EXTENSIONS = {".txt", ".pdf"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_DOCS_SIZE_BYTES = 200 * 1024 * 1024  # placeholder, не измерено
MIN_EXTRACTED_CHARS = 50
LOG_PATH = "docs_validation.log"


def _log(message):
    with open(LOG_PATH, "a") as f:
        f.write(message + "\n")
    print(message)


def extract_text(filepath):
    """Достать текст из .txt или .pdf файла."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        with open(filepath, "r", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


def validate_file(filepath, docs_folder):
    """Проверить ОДИН файл по всем правилам. Возвращает (валиден?, причина)."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return False, "extension not allowed"

    if os.path.getsize(filepath) > MAX_FILE_SIZE_BYTES:
        return False, "file too large"

    docs_total = sum(
        os.path.getsize(os.path.join(docs_folder, f))
        for f in os.listdir(docs_folder)
        if os.path.isfile(os.path.join(docs_folder, f))
    )
    if docs_total + os.path.getsize(filepath) > MAX_TOTAL_DOCS_SIZE_BYTES:
        return False, "would exceed total docs/ size limit"

    if len(extract_text(filepath).strip()) < MIN_EXTRACTED_CHARS:
        return False, "no usable text extracted"

    return True, "ok"


def process_upload_folder(upload_folder="docs_upload", docs_folder="docs"):
    """Переместить валидные файлы из upload_folder в docs_folder, остальные — залогировать."""
    if not os.path.isdir(upload_folder):
        return

    for filename in os.listdir(upload_folder):
        filepath = os.path.join(upload_folder, filename)
        if not os.path.isfile(filepath):
            continue

        is_valid, reason = validate_file(filepath, docs_folder)
        if is_valid:
            shutil.move(filepath, os.path.join(docs_folder, filename))
            _log(f"ACCEPTED: {filename}")
        else:
            _log(f"REJECTED: {filename} -> {reason}")


if __name__ == "__main__":
    process_upload_folder()