import sqlite3

DB_PATH = "history.db"


def init_db(db_path=DB_PATH):
    """Create the messages table if it doesn't exist yet."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_turn(user_message, assistant_message, db_path=DB_PATH):
    """
    Save a user+assistant exchange as ONE atomic operation — either
    BOTH messages get saved, or NEITHER does. Avoids orphaned,
    half-saved turns if the app crashes mid-exchange.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        ("user", user_message)
    )
    conn.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        ("assistant", assistant_message)
    )
    conn.commit()
    conn.close()


def load_history(db_path=DB_PATH):
    """Load the whole conversation history, in the order it happened."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT role, content FROM messages ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]