import sqlite3
import json
from typing import Optional, List, Dict

DB_PATH = "clients.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():
    conn = get_connection()
    c = conn.cursor()

    # Client/source/target configuration
    c.execute("""
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_channel_id INTEGER NOT NULL UNIQUE,
            target_channel_id INTEGER NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            settings TEXT DEFAULT '{}'
        )
    """)

    # Last processed Telegram message per source channel
    c.execute("""
        CREATE TABLE IF NOT EXISTS processing_state (
            source_channel_id INTEGER PRIMARY KEY,
            last_processed_message_id INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# CLIENTS
# ============================================================

def add_client(
    source_channel_id: int,
    target_channel_id: int,
) -> bool:
    try:
        conn = get_connection()
        c = conn.cursor()

        c.execute(
            """
            INSERT OR REPLACE INTO clients
            (
                source_channel_id,
                target_channel_id,
                is_active,
                settings
            )
            VALUES (?, ?, 1, '{}')
            """,
            (
                source_channel_id,
                target_channel_id,
            ),
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"DB Error adding client: {e}")
        return False


def get_all_clients() -> List[Dict]:
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        """
        SELECT
            source_channel_id,
            target_channel_id,
            settings
        FROM clients
        WHERE is_active = 1
        """
    )

    rows = c.fetchall()
    conn.close()

    result = []

    for row in rows:
        try:
            settings = json.loads(row[2] or "{}")
        except Exception:
            settings = {}

        result.append(
            {
                "source": row[0],
                "target": row[1],
                "settings": settings,
            }
        )

    return result


def get_target_for_source(
    source_channel_id: int,
) -> Optional[int]:
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        """
        SELECT target_channel_id
        FROM clients
        WHERE source_channel_id = ?
        AND is_active = 1
        """,
        (source_channel_id,),
    )

    row = c.fetchone()
    conn.close()

    return row[0] if row else None


# ============================================================
# MESSAGE PROCESSING STATE
# ============================================================

def get_last_processed(
    source_channel_id: int,
) -> Optional[int]:
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        """
        SELECT last_processed_message_id
        FROM processing_state
        WHERE source_channel_id = ?
        """,
        (source_channel_id,),
    )

    row = c.fetchone()
    conn.close()

    if row is None:
        return None

    return int(row[0])


def set_last_processed(
    source_channel_id: int,
    message_id: int,
) -> bool:
    try:
        conn = get_connection()
        c = conn.cursor()

        c.execute(
            """
            INSERT INTO processing_state
            (
                source_channel_id,
                last_processed_message_id
            )
            VALUES (?, ?)
            ON CONFLICT(source_channel_id)
            DO UPDATE SET
                last_processed_message_id =
                    excluded.last_processed_message_id
            """,
            (
                source_channel_id,
                message_id,
            ),
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"DB Error saving processing state: {e}")
        return False


# ============================================================
# OPTIONAL RESET
# ============================================================

def reset_last_processed(
    source_channel_id: int,
) -> bool:
    try:
        conn = get_connection()
        c = conn.cursor()

        c.execute(
            """
            DELETE FROM processing_state
            WHERE source_channel_id = ?
            """,
            (source_channel_id,),
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"DB Error resetting processing state: {e}")
        return False


# ============================================================
# INITIALIZE ON IMPORT
# ============================================================

init_db()
