import os
import logging
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# === Logging Config ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Environment Variables ===
load_dotenv()
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DBNAME = os.getenv("DB_NAME")

if not all([USER, PASSWORD, HOST, PORT, DBNAME]):
    raise ValueError("Database connection variables DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME must be set in .env")

# === SQL to ensure table exists ===
CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE TABLE IF NOT EXISTS user_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    user_id TEXT,
    user_name TEXT,
    user_token TEXT,
    user_message TEXT NOT NULL,
    bot_message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

class DBManager:
    def __init__(self):
        self.conn = None

    def connect(self):
        logger.info("🔌 Connecting to PostgreSQL via psycopg2...")
        self.conn = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        self.conn.autocommit = True
        logger.info("✅ Connection established.")
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        logger.info("✅ Table user_conversations ensured.")

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("🔒 Connection closed.")

    def insert_conversation(
        self,
        session_id: str,
        user_id: str,
        user_name: str,
        user_token: str,
        user_message: str,
        bot_message: str
    ) -> str:
        sql = """
        INSERT INTO user_conversations
            (session_id, user_id, user_name, user_token, user_message, bot_message)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql,
                (session_id, user_id or "", user_name or "", user_token or "", user_message, bot_message)
            )
            rec = cur.fetchone()
            conv_id = rec["id"]
            logger.info(f"🆕 Inserted conversation {conv_id}")
            return conv_id

    def get_conversations(self, session_id: str = None):
        if session_id:
            sql = "SELECT * FROM user_conversations WHERE session_id = %s ORDER BY created_at;"
            params = (session_id,)
        else:
            sql = "SELECT * FROM user_conversations ORDER BY created_at;"
            params = None
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            logger.info(f"💬 Fetched {len(rows)} conversations")
            return rows

    def update_conversation(self, conv_id: str, bot_message: str) -> bool:
        sql = "UPDATE user_conversations SET bot_message = %s WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(sql, (bot_message, conv_id))
            updated = cur.rowcount == 1
            logger.info(f"✏️ Update {conv_id}: {updated}")
            return updated

    def delete_conversation(self, conv_id: str) -> bool:
        sql = "DELETE FROM user_conversations WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(sql, (conv_id,))
            deleted = cur.rowcount == 1
            logger.info(f"🗑️ Delete {conv_id}: {deleted}")
            return deleted

# === Test Harness ===
if __name__ == "__main__":
    db = DBManager()
    db.connect()

    # 1) Insert
    conv_id = db.insert_conversation(
        session_id="test-session-1",
        user_id="user-123",
        user_name="Nawab",
        user_token="tok_xyz",
        user_message="Where is my order?",
        bot_message="Your order is being processed."
    )

    # 2) Fetch
    convs = db.get_conversations("test-session-1")
    print(convs)

    # 3) Update
    updated = db.update_conversation(conv_id, "Your order has shipped!")
    print("Updated:", updated)

    # 4) Delete
    deleted = db.delete_conversation(conv_id)
    print("Deleted:", deleted)

    db.close()
