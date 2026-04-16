import aiosqlite
import json
from datetime import datetime
from config import DB_PATH


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                duration_sec REAL,
                mode TEXT DEFAULT 'live'
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS utterances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                start_time REAL,
                end_time REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                embedding_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_owner INTEGER DEFAULT 0
            )
        """)
        await db.commit()


async def create_conversation(mode="live"):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO conversations (started_at, mode) VALUES (?, ?)",
            (datetime.now().isoformat(), mode)
        )
        await db.commit()
        return cursor.lastrowid


async def save_utterance(conv_id, speaker, text, start_time=None, end_time=None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO utterances (conversation_id, speaker, text, start_time, end_time, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, speaker, text, start_time, end_time, datetime.now().isoformat())
        )
        await db.commit()


async def end_conversation(conv_id, duration_sec):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE conversations SET duration_sec = ? WHERE id = ?",
            (duration_sec, conv_id)
        )
        await db.commit()


async def get_conversation(conv_id):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM utterances WHERE conversation_id = ? ORDER BY start_time, id",
            (conv_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_recent_conversations(limit=20):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM conversations ORDER BY started_at DESC LIMIT ?",
            (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def save_voice_profile(name, embedding_path, is_owner=False):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO voice_profiles (name, embedding_path, created_at, is_owner)
               VALUES (?, ?, ?, ?)""",
            (name, embedding_path, datetime.now().isoformat(), int(is_owner))
        )
        await db.commit()


async def get_voice_profiles():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM voice_profiles")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
