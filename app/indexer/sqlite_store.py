import sqlite3
import re
import logging
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
import unicodedata
from cryptography.fernet import Fernet

# Make imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config

# Logging with privacy
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    """Thread-safe SQLite connection manager."""
    
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.timeout = 120
        self.max_retries = 5
        self.retry_delay = 0.5
        # Ensure DB exists locally
        if not Path(db_path).exists():
            logger.info(f"Creating local SQLite database at {db_path}")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_conn(self):
        for attempt in range(self.max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=self.timeout)
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = -64000")
                yield conn
                conn.commit()
                conn.close()
                return
            except sqlite3.OperationalError as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise

class SQLiteStore:
    """SQLite storage with encryption and offline support."""
    
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.conn_manager = ConnectionManager(db_path)
        self.cipher = Fernet(Fernet.generate_key())
        self._init_db()
        # NLTK Bengali support
        try:
            import nltk
            self.nltk = nltk
            self.bengali_tokenizer = nltk.tokenize.word_tokenize
        except Exception:
            self.nltk = None
            self.bengali_tokenizer = None

    def _init_db(self):
        """Initialize database with multilang tokenizer."""
        try:
            with self.conn_manager.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        title TEXT,
                        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        doc_id TEXT,
                        page INTEGER,
                        text BLOB,  -- Store encrypted text
                        FOREIGN KEY (doc_id) REFERENCES documents(id)
                    )
                """)
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        text,
                        content='chunks',
                        content_rowid='rowid',
                        tokenize='porter unicode61'
                    )
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_after_insert
                    AFTER INSERT ON chunks
                    BEGIN
                        INSERT INTO chunks_fts(rowid, text)
                        VALUES (new.rowid, new.text);
                    END
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_after_update
                    AFTER UPDATE ON chunks
                    BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text)
                        VALUES ('delete', old.rowid, old.text);
                        INSERT INTO chunks_fts(rowid, text)
                        VALUES (new.rowid, new.text);
                    END
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_after_delete
                    AFTER DELETE ON chunks
                    BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text)
                        VALUES ('delete', old.rowid, old.text);
                    END
                """)
                logger.info(f"Initialized SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def _encrypt_text(self, text: str) -> bytes:
        return self.cipher.encrypt(text.encode())

    def _decrypt_text(self, encrypted_text: bytes) -> str:
        return self.cipher.decrypt(encrypted_text).decode()

    def add_document(self, doc_id: str, path: str, title: str) -> None:
        try:
            with self.conn_manager.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (id, path, title, added_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (doc_id, path, title))
                logger.info(f"Added document: {doc_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to add document {doc_id}: {str(e)}")
            raise

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        try:
            with self.conn_manager.get_conn() as conn:
                cursor = conn.cursor()
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    for chunk in batch:
                        chunk["text"] = unicodedata.normalize('NFKC', chunk["text"])
                        encrypted_text = self._encrypt_text(chunk["text"])
                        cursor.execute("SELECT id FROM chunks WHERE id = ?", (chunk["id"],))
                        if cursor.fetchone():
                            logger.debug(f"Skipping duplicate chunk: {chunk['id']}")
                            continue
                        cursor.execute("""
                            INSERT INTO chunks (id, doc_id, page, text)
                            VALUES (?, ?, ?, ?)
                        """, (chunk["id"], chunk["doc_id"], chunk.get("page", 0), encrypted_text))
                logger.info(f"Added {len(chunks)} chunks to database")
        except sqlite3.Error as e:
            logger.error(f"Failed to add chunks: {str(e)}")
            raise

    def _normalize_query(self, query: str, lang: str = 'multi') -> str:
        norm = unicodedata.normalize('NFKC', query.lower())
        # Use NLTK for Bengali tokenization if available
        if lang.startswith('bn') or lang.startswith('ben'):
            if self.bengali_tokenizer:
                try:
                    tokens = self.bengali_tokenizer(norm, language='bengali')
                    return ' '.join(tokens)
                except Exception:
                    pass
        return norm

    def keyword_search(self, query: str, limit: int = 10, bm25_weights: Optional[list] = None, recency_boost: bool = True, lang: str = 'multi') -> List[Dict[str, Any]]:
        try:
            config = Config()
            # BM25 weights from config or argument
            if bm25_weights is None:
                bm25_weights = [getattr(config, 'bm25_weight_text', 10.0), 0.0, getattr(config, 'bm25_weight_other', 5.0)]
            normalized_query = self._normalize_query(query, lang=lang)
            # FTS5: escape single quotes for SQL
            safe_query = normalized_query.replace("'", "''")
            with self.conn_manager.get_conn() as conn:
                cursor = conn.cursor()
                # Recency boost: gamma * (now - added_at in days)
                gamma = getattr(config, 'search_gamma', 0.1)
                now = int(time.time())
                sql = f"""
                    SELECT c.id, c.doc_id, c.page, c.text,
                        bm25(chunks_fts, {bm25_weights[0]}, {bm25_weights[1]}, {bm25_weights[2]}) as bm25_score,
                        d.added_at
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.rowid
                    JOIN documents d ON c.doc_id = d.id
                    WHERE chunks_fts MATCH '{safe_query}'
                """
                # Compute recency score in Python (since SQLite lacks date math on CURRENT_TIMESTAMP)
                cursor.execute(sql)
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    bm25_score = row[4]
                    added_at = row[5]
                    try:
                        # Parse added_at as timestamp
                        if isinstance(added_at, str):
                            # Try to parse as ISO or fallback
                            try:
                                ts = int(time.mktime(time.strptime(added_at, "%Y-%m-%d %H:%M:%S")))
                            except Exception:
                                ts = now
                        else:
                            ts = int(added_at)
                        days_ago = max(0, (now - ts) / 86400)
                    except Exception:
                        days_ago = 0
                    recency_score = gamma * (1.0 / (1.0 + days_ago)) if recency_boost else 0.0
                    total_score = bm25_score + recency_score
                    results.append({
                        "id": row[0],
                        "doc_id": row[1],
                        "page": row[2],
                        "text": self._decrypt_text(row[3]),
                        "score": total_score,
                        "bm25": bm25_score,
                        "recency": recency_score
                    })
                # Sort by total score descending
                results.sort(key=lambda x: x['score'], reverse=True)
                logger.info(f"Keyword search returned {len(results)} results (recency_boost={recency_boost})")
                return results[:limit]
        except sqlite3.Error as e:
            logger.error(f"Keyword search failed: {str(e)}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.conn_manager.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, path, title, added_at FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                if row:
                    return {"id": row[0], "path": row[1], "title": row[2], "added_at": row[3]}
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve document {doc_id}: {str(e)}")
            raise

    def close(self):
        logger.info("Closing SQLite database connections")

    def test_offline_search(self, query: str = "test query") -> List[Dict[str, Any]]:
        try:
            results = self.keyword_search(query, limit=5)
            logger.info(f"Offline search test returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Offline search test failed: {str(e)}")
            raise

if __name__ == "__main__":
    config = Config()
    store = SQLiteStore(config.db_path)
    
    # Test adding a document
    doc_id = "test_doc"
    store.add_document(doc_id, "test.pdf", "Test Document")
    
    # Test adding chunks
    chunks = [
        {"id": "chunk1", "doc_id": doc_id, "page": 1, "text": "This is a test chunk."},
        {"id": "chunk2", "doc_id": doc_id, "page": 2, "text": "এটি একটি পরীক্ষা।"}
    ]
    store.add_chunks(chunks)
    
    # Test offline search
    results = store.test_offline_search()
    for result in results:
        logger.info(f"Result: {result['text'][:50]}... (Score: {result['score']})")
    
    store.close()