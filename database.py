import sqlite3
import json
from typing import Optional, List, Dict

DB_PATH = "clients.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_channel_id INTEGER NOT NULL UNIQUE,
            target_channel_id INTEGER NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            settings TEXT DEFAULT '{}'
        )
    ''')
    conn.commit()
    conn.close()

def add_client(source_channel_id: int, target_channel_id: int) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO clients (source_channel_id, target_channel_id) VALUES (?, ?)",
            (source_channel_id, target_channel_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"DB Error: {e}")
        return False

def get_all_clients() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT source_channel_id, target_channel_id, settings FROM clients WHERE is_active=1")
    rows = c.fetchall()
    conn.close()
    return [{"source": row[0], "target": row[1], "settings": json.loads(row[2] or "{}")} for row in rows]

def get_target_for_source(source_channel_id: int) -> Optional[int]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT target_channel_id FROM clients WHERE source_channel_id=? AND is_active=1", (source_channel_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# Initialize DB when module loads
init_db()