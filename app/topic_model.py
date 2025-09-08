import logging
logger = logging.getLogger(__name__)
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import gc
import numpy as np
from collections import Counter

# scikit-learn
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available")

# BERTopic
try:
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTOPIC_AVAILABLE = False
    logger.warning("BERTopic not available - falling back to KMeans")

# NLTK multilang
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except Exception:
    def word_tokenize(text: str):
        return text.split()
    class _EmptyStopwords:
        def words(self, lang=None):
            return set()
    stopwords = _EmptyStopwords()
    NLTK_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.indexer.sqlite_store import SQLiteStore
from app.indexer.vector_store import VectorStore
from app.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TopicModel:
    """
    Clustering and topic modeling with multilang support.
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        language: str = "multi",  # 'multi' for Bengali/English/Hindi etc.
        sqlite_store: Optional[SQLiteStore] = None,
        vector_store: Optional[VectorStore] = None,
        db_path: Optional[str] = None,
        index_dir: Optional[str] = None
    ):
        if not SKLEARN_AVAILABLE and not BERTOPIC_AVAILABLE:
            raise RuntimeError("Clustering libraries not installed")
        
        self.n_clusters = max(2, int(n_clusters))
        self.language = language
        self.sqlite_store = sqlite_store or SQLiteStore(db_path=Config().db_path if db_path is None else db_path)
        self.vector_store = vector_store or VectorStore(index_dir=Config().index_dir if index_dir is None else index_dir, embedding_dim=Config().embedding_dim)
        
        # Multilang stopwords (combine NLTK where possible, plus curated Bengali list)
        self.stop_words = set()
        if 'eng' in language or language == 'multi':
            try:
                self.stop_words.update(stopwords.words('english'))
            except Exception:
                # NLTK english not available - best-effort minimal list
                self.stop_words.update({"the", "and", "is", "in", "it", "of", "to"})

        # Curated Bengali stopwords (expanded)
        bengali_core = {
            "এবং", "যে", "যা", "আমি", "তুমি", "তুমি", "তিনি", "এই", "ও", "কিন্তু",
            "হয়", "ছিল", "ছিলো", "হলে", "অনেক", "কিছু", "কারণ", "তাদের", "তারা", "আমরা",
            "করে", "করা", "হবে", "আর", "যখন", "সাথে", "বলে", "হয়েছে", "পর", "মধ্যে"
        }
        if 'ben' in language or language == 'multi':
            self.stop_words.update(bengali_core)

        if 'hin' in language or language == 'multi':
            try:
                self.stop_words.update(stopwords.words('hindi'))
            except Exception:
                self.stop_words.update({"और", "कि", "जो", "मैं", "तुम", "यह", "है", "पर"})

        self.use_bertopic = BERTOPIC_AVAILABLE
        if self.use_bertopic:
            # Initialize UMAP/HDBSCAN/vectorizer for BERTopic
            try:
                umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
                hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)
                vectorizer_model = ClassTfidfVectorizer(stop_words=list(self.stop_words))  # Multilang vectorizer
                self.topic_model = BERTopic(
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    nr_topics=self.n_clusters,
                    calculate_probabilities=True
                )
                logger.info("BERTopic initialized with multilang vectorizer and clustering models")
            except Exception as e:
                # If any of the optional BERTopic components fail to initialize, fall back
                logger.warning(f"BERTopic components failed to initialize ({e}), falling back to KMeans")
                self.use_bertopic = False
        else:
            logger.info("Falling back to KMeans")
    
    def cluster_chunks(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.sqlite_store or not self.vector_store:
            return {"chunk_ids": [], "topics": [], "error": "Stores not initialized"}

        try:
            chunks = self.sqlite_store.get_chunks(doc_id=doc_id)
            if not chunks:
                return {"chunk_ids": [], "topics": [], "error": "No chunks found"}

            chunk_ids = [chunk["id"] for chunk in chunks]
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.vector_store.get_embeddings(chunk_ids)
            if embeddings is None or len(embeddings) != len(chunk_ids):
                return {"chunk_ids": [], "topics": [], "error": "Embeddings retrieval failed"}

            topics = []
            if self.use_bertopic:
                # Fit BERTopic once on the full corpus so topic ids are consistent across all chunks
                try:
                    topic_ids, probs = self.topic_model.fit_transform(texts, embeddings=embeddings)
                except TypeError:
                    # Older BERTopic versions may expect embeddings as positional arg
                    topic_ids, probs = self.topic_model.fit_transform(texts, embeddings)

                # Build labels (c-TF-IDF style) by taking top words for each topic
                def make_label(tid: int, top_n: int = 3) -> str:
                    if tid == -1:
                        return "Other"
                    try:
                        topic_words = self.topic_model.get_topic(int(tid))
                        if not topic_words:
                            return f"Topic_{tid}"
                        words = [w for w, _ in topic_words[:top_n]]
                        return " ".join(words)
                    except Exception:
                        return f"Topic_{tid}"

                topic_labels = [make_label(int(tid)) for tid in topic_ids]
            else:
                # KMeans fallback - operate on all embeddings at once for consistency
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                topic_ids = kmeans.fit_predict(embeddings)
                topic_labels = [f"Topic_{label}" for label in topic_ids]

            for idx, (tid, label) in enumerate(zip(topic_ids, topic_labels)):
                topics.append({"chunk_id": chunk_ids[idx], "topic_id": int(tid), "topic_label": label})
            gc.collect()
            
            # Store topics
            with self.sqlite_store.conn_manager.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM topics WHERE chunk_id IN ({})".format(','.join('?' for _ in chunk_ids)), chunk_ids)
                cursor.executemany("INSERT INTO topics (chunk_id, topic_id, topic_label) VALUES (?, ?, ?)", 
                                   [(t["chunk_id"], t["topic_id"], t["topic_label"]) for t in topics])
            
            logger.info(f"Clustered {len(chunk_ids)} chunks into {self.n_clusters} topics")
            return {"chunk_ids": chunk_ids, "topics": topics, "error": None}
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return {"chunk_ids": [], "topics": [], "error": str(e)}

    def get_topics(self, doc_id: Optional[str] = None, persistent: bool = False) -> List[Dict[str, Any]]:
        conn = self.sqlite_store.conn_manager.get_conn() if persistent else self.sqlite_store.conn_manager.get_conn().__enter__()
        try:
            cursor = conn.cursor()
            if doc_id:
                cursor.execute("""
                    SELECT chunk_id, topic_id, topic_label FROM topics t JOIN chunks c ON t.chunk_id = c.id
                    WHERE c.doc_id = ?
                    """, (doc_id,))
            else:
                cursor.execute("SELECT chunk_id, topic_id, topic_label FROM topics")

            topics = [
                {"chunk_id": row[0], "topic_id": row[1], "topic_label": row[2]}
                for row in cursor.fetchall()
            ]
            if not persistent:
                conn.close()
            logger.info(f"Retrieved {len(topics)} topic assignments")
            return topics
        except Exception as e:
            logger.error(f"Failed to retrieve topics: {str(e)}")
            if not persistent:
                conn.close()
            return []

    def __del__(self):
        if self.sqlite_store:
            self.sqlite_store.close()
        if self.vector_store:
            self.vector_store.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Topic Modeler for Document Chunks")
    parser.add_argument("--doc-id", help="Document ID to cluster")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--language", default="multi", help="Language: multi, ben, eng, hin")
    args = parser.parse_args()

    try:
        config = Config()
        topic_model = TopicModel(
            n_clusters=args.n_clusters,
            language=args.language,
            db_path=config.db_path,
            index_dir=config.index_dir
        )

        result = topic_model.cluster_chunks(doc_id=args.doc_id)

        print(f"\n--- Clustering Results ---")
        if result["error"]:
            print(f"Error: {result['error']}")
        else:
            print(f"Clustered {len(result['chunk_ids'])} chunks into {args.n_clusters} topics")
            for assignment in result["topics"]:
                print(f"Chunk ID: {assignment['chunk_id']}, Topic ID: {assignment['topic_id']}, Label: {assignment['topic_label']}")

        print(f"\n--- All Topics ---")
        topics = topic_model.get_topics(doc_id=args.doc_id)
        for topic in topics:
            print(f"Chunk ID: {topic['chunk_id']}, Topic ID: {topic['topic_id']}, Label: {topic['topic_label']}")

    except Exception as e:
        logger.error(f"Topic modeling failed: {str(e)}")
        sys.exit(1)