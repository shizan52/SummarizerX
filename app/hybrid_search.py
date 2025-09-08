from app.qa_answerer import QAAnswerer
import logging
from typing import List, Dict, Any
import numpy as np
import nltk
from app.indexer.sqlite_store import SQLiteStore
from app.indexer.vector_store import VectorStore
from app.embeddings.embedder import Embedder
from app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSearch:
    """Hybrid keyword + semantic search with answer extraction."""

    def __init__(self, sqlite_store: SQLiteStore, vector_store: VectorStore, embedder: Embedder):
        self.sqlite_store = sqlite_store
        self.vector_store = vector_store
        self.embedder = embedder
        config = Config()
        self.alpha = getattr(config, 'search_alpha', 0.7)  # Semantic weight
        self.beta = getattr(config, 'search_beta', 0.25)   # Lexical weight
        self.gamma = getattr(config, 'search_gamma', 0.05) # Recency weight
        self.qa_answerer = QAAnswerer()

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search with re-ranking and improved answer extraction."""
        # Step 1: Keyword search
        keyword_results = self.sqlite_store.keyword_search(query, limit=k*2)

        # Step 2: Semantic search
        query_emb = self.embedder.embed([query])[0]
        semantic_results = self.vector_store.search(query_emb, k=k*2)

        # Step 3: Merge and re-rank
        merged = self._merge_results(keyword_results, semantic_results)
        ranked = sorted(merged, key=lambda x: self._compute_score(x), reverse=True)[:k]

        # Step 4: Merge top-3 chunk texts for QA context
        top_chunks = ranked[:3]
        merged_context = "\n".join([r.get('text', '') for r in top_chunks if r.get('text')])
        # Run QA on merged context
        try:
            answers = self.qa_answerer.answer(query, merged_context, top_k=1)
            qa_answer = answers[0] if answers else ""
        except Exception as e:
            qa_answer = ""

        # If QA answer is empty or too short, fallback to top-3 relevant sentences
        if not qa_answer or len(qa_answer.strip()) < 2:
            # Fallback: extract top-3 relevant sentences from merged context
            import nltk
            sentences = nltk.sent_tokenize(merged_context)
            if sentences:
                sent_embs = self.embedder.embed(sentences)
                query_emb = self.embedder.embed([query])[0]
                import numpy as np
                similarities = np.dot(sent_embs, query_emb) / (np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(query_emb))
                top_idx = np.argsort(similarities)[-3:][::-1]
                fallback_answers = [sentences[i] for i in top_idx]
                qa_answer = "\n".join(fallback_answers)
            else:
                qa_answer = merged_context[:200]

        # Attach answer to all top chunks for UI
        for result in ranked[:3]:
            result['extracted_answer'] = qa_answer
        for result in ranked[3:]:
            result['extracted_answer'] = self._extract_answer(query, result['text'])
        return ranked

    def _merge_results(self, keyword: List[Dict], semantic: List[Dict]) -> List[Dict]:
        """Merge results from both searches."""
        merged_dict = {r['id']: r for r in keyword}
        for s in semantic:
            if s['id'] in merged_dict:
                merged_dict[s['id']]['semantic_score'] = s['score']
            else:
                merged_dict[s['id']] = s
        return list(merged_dict.values())

    def _compute_score(self, result: Dict) -> float:
        """Compute hybrid score."""
        semantic = result.get('semantic_score', 0.0)
        lexical = result.get('bm25', 0.0)
        recency = result.get('recency', 0.0)
        return self.alpha * semantic + self.beta * lexical + self.gamma * recency

    def _extract_answer(self, query: str, text: str) -> str:
        """Extract most relevant part using embedding similarity."""
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return text[:200]
        sent_embs = self.embedder.embed(sentences)
        query_emb = self.embedder.embed([query])[0]
        similarities = np.dot(sent_embs, query_emb) / (np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(query_emb))
        top_idx = np.argmax(similarities)
        return sentences[top_idx]

if __name__ == "__main__":
    # Test code
    pass
from typing import List, Dict, Optional, Any
import sqlite3
import numpy as np
import importlib
import logging
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearch:
    def __init__(self, sqlite_store_or_path: Any, vector_store_or_index: Any, embedder_or_model: Any = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialize hybrid search.

        Accepts either:
        - sqlite_store_or_path: path to sqlite DB (str) or a SQLiteStore instance
        - vector_store_or_index: path to faiss index (str) or a VectorStore instance
        - embedder_or_model: Embedder instance or model_name string
        """
        try:
            # SQLite: accept SQLiteStore instance or path
            if hasattr(sqlite_store_or_path, 'conn_manager'):
                self.sqlite_store = sqlite_store_or_path
                try:
                    self.conn = self.sqlite_store.conn_manager.get_conn()
                except Exception:
                    # fallback: try to get a raw connection
                    self.conn = None
            else:
                self.sqlite_store = None
                self.conn = sqlite3.connect(str(sqlite_store_or_path))

            # Vector store: accept VectorStore instance or faiss index path
            if hasattr(vector_store_or_index, 'search'):
                self.vector_store = vector_store_or_index
                self.faiss_index = None
            else:
                # try to read faiss index file if faiss available
                try:
                    import faiss
                    self.faiss_index = faiss.read_index(str(vector_store_or_index))
                    self.vector_store = None
                except Exception:
                    self.faiss_index = None
                    self.vector_store = None

            # Embedder: accept Embedder instance (with encode) or a model name
            if hasattr(embedder_or_model, 'encode'):
                self.embedder = embedder_or_model
            else:
                # try to load sentence-transformers
                try:
                    from sentence_transformers import SentenceTransformer
                    self.embedder = SentenceTransformer(str(embedder_or_model), cache_folder='./models')
                except Exception:
                    self.embedder = None

            self.alpha = 0.6
            self.beta = 0.4
            logger.info("HybridSearch initialized")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid search: {e}")
            raise

    def _normalize_query(self, query: str) -> str:
        return unicodedata.normalize('NFKC', query.lower())

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """Perform hybrid search with multilang normalization."""
        try:
            normalized_query = self._normalize_query(query)
            # Keyword search against chunks FTS table (more relevant)
            keyword_results = []
            try:
                if self.sqlite_store is not None:
                    # try SQLiteStore provided API
                    try:
                        # expect a method get_fts_matches(query, limit)
                        fts = getattr(self.sqlite_store, 'keyword_search', None)
                        if callable(fts):
                            rows = fts(normalized_query, top_k)
                            for r in rows:
                                keyword_results.append({"id": r.get('id'), "text": r.get('text'), "keyword_score": r.get('score', 0.0)})
                        else:
                            # fallback to raw connection if available
                            conn = self.sqlite_store.conn_manager.get_conn()
                            cursor = conn.cursor()
                            cursor.execute("SELECT c.id, c.text, 0 as score FROM chunks c LIMIT ?", (top_k,))
                            for row in cursor.fetchall():
                                keyword_results.append({"id": row[0], "text": row[1], "keyword_score": row[2]})
                    except Exception:
                        pass
                elif self.conn is not None:
                    cursor = self.conn.cursor()
                    try:
                        cursor.execute("""
                            SELECT c.id, c.text, bm25(chunks_fts) as score
                            FROM chunks_fts
                            JOIN chunks c ON chunks_fts.rowid = c.rowid
                            WHERE chunks_fts MATCH ?
                            ORDER BY score DESC
                            LIMIT ?
                        """, (normalized_query, top_k))
                    except Exception:
                        cursor.execute("SELECT id, text, 0 as score FROM documents WHERE text MATCH ? LIMIT ?", (normalized_query, top_k))
                    for row in cursor.fetchall():
                        keyword_results.append({"id": row[0], "text": row[1], "keyword_score": row[2]})
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")

            # Semantic search
            # Semantic search
            semantic_results = []
            try:
                if self.embedder is not None and hasattr(self.embedder, 'encode'):
                    q_emb = self.embedder.encode([query], batch_size=1, convert_to_numpy=True)
                    q_emb = np.asarray(q_emb, dtype=np.float32)
                else:
                    # try to import SentenceTransformer as fallback
                    from sentence_transformers import SentenceTransformer
                    tmp_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='./models')
                    q_emb = tmp_model.encode([query], batch_size=1, convert_to_numpy=True)
                    q_emb = np.asarray(q_emb, dtype=np.float32)

                # Use VectorStore API if present
                if self.vector_store is not None:
                    vs_results = self.vector_store.search(q_emb, k=top_k)
                    # vs_results is a list of rows; take first row
                    if vs_results and len(vs_results) > 0:
                        for meta in vs_results[0]:
                            semantic_results.append({"id": str(meta.get('id', meta.get('chunk_id', ''))), "semantic_score": float(1.0 / (1.0 + float(meta.get('distance', 0.0))))})
                elif self.faiss_index is not None:
                    import faiss
                    if q_emb.ndim == 1:
                        q_emb = q_emb.reshape(1, -1)
                    D, I = self.faiss_index.search(q_emb, top_k)
                    for idx, dist in zip(I[0], D[0]):
                        if int(idx) < 0:
                            continue
                        semantic_results.append({"id": str(idx), "semantic_score": float(1.0 / (1.0 + float(dist)))})
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

            # Combine results
            combined_results = {}
            for res in keyword_results:
                combined_results[str(res["id"])] = {
                    "text": res.get("text", ""),
                    "keyword_score": float(res.get("keyword_score", 0.0)),
                    "semantic_score": 0.0
                }
            for res in semantic_results:
                if res["id"] in combined_results:
                    combined_results[res["id"]]["semantic_score"] = res["semantic_score"]
                else:
                    # Try to fetch text from chunks or documents
                    text_row = ""
                    try:
                        if self.sqlite_store is not None:
                            # try sqlite_store API
                            get_chunk = getattr(self.sqlite_store, 'get_chunk_by_id', None)
                            if callable(get_chunk):
                                rec = get_chunk(res["id"])
                                if rec:
                                    text_row = rec.get('text', '')
                        elif self.conn is not None:
                            cur = self.conn.cursor()
                            cur.execute("SELECT text FROM chunks WHERE id = ?", (res["id"],))
                            row = cur.fetchone()
                            if row:
                                text_row = row[0]
                    except Exception:
                        text_row = ""
                    combined_results[res["id"]] = {
                        "text": text_row,
                        "keyword_score": 0.0,
                        "semantic_score": res["semantic_score"]
                    }

            # Rerank
            final_results = []
            for id, res in combined_results.items():
                score = self.alpha * res["keyword_score"] + self.beta * res["semantic_score"]
                final_results.append({"id": id, "text": res["text"], "score": score})
            
            final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k]
            logger.info(f"Hybrid search returned {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            return []

    def __del__(self):
        """Close database connection."""
        try:
            if getattr(self, 'conn', None) is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
        except Exception:
            pass

if __name__ == "__main__":
    # Test the hybrid search with multilang query
    searcher = HybridSearch("documents.db", "faiss_index.bin")
    multilang_query = "বাংলা ডকুমেন্ট সামারাইজেশন This is English यह हिंदी में है"
    results = searcher.search(multilang_query, top_k=3)
    print("Multilang Search Results:", results)