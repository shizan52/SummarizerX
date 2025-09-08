import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pickle
import math
import gc

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class VectorStore:
    """Vector storage for embeddings using FAISS or HNSWlib with IVF-PQ indexing."""
    
    def __init__(self, index_dir: str, embedding_dim: int, index_type: str = "ivf_pq", use_pq: bool = True):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.index_type = index_type.lower()
        self.use_pq = use_pq
        self.index = None
        self.index_to_id = {}  
        self.faiss_available = False
        self.hnswlib_available = False
        
        # Validate parameters
        if self.index_type not in ["flat", "ivf_pq", "hnsw"]:
            raise ValueError(f"Invalid index_type: {self.index_type}.")
        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding_dim: {self.embedding_dim}.")
        
        # Try FAISS
        try:
            import faiss
            self.faiss_available = True
            logger.info("FAISS is available")
        except ImportError:
            logger.warning("FAISS not installed")
        
        # Try HNSWlib
        try:
            import hnswlib
            self.hnswlib_available = True
            logger.info("HNSWlib is available")
        except ImportError:
            logger.warning("HNSWlib not installed")
        
        if not (self.faiss_available or self.hnswlib_available):
            raise ImportError("Neither FAISS nor HNSWlib is installed.")
        
        # Initialize index
        self._init_index()

    def _init_index(self):
        """Initialize the vector index based on index_type."""
        try:
            if self.faiss_available and self.index_type in ["flat", "ivf_pq"]:
                import faiss
                if self.index_type == "flat":
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                elif self.index_type == "ivf_pq":
                    nlist = 100
                    quantizer = faiss.IndexFlatL2(self.embedding_dim)
                    self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 8, 8)
                    self.index.nprobe = 20
                logger.info(f"Initialized FAISS {self.index_type} index")
            elif self.hnswlib_available and self.index_type == "hnsw":
                import hnswlib
                self.index = hnswlib.Index(space='l2', dim=self.embedding_dim)
                self.index.init_index(max_elements=1000000, ef_construction=200, M=16)  # Much larger default
                self.index.set_ef(100)
                logger.info("Initialized HNSWlib index")
            else:
                raise ValueError(f"Unsupported index_type: {self.index_type}")
        except Exception as e:
            logger.error(f"Failed to initialize index: {str(e)}")
            raise

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], allow_fallback: bool = False) -> None:
        """Add embeddings and metadata to the index incrementally. If IVF-PQ training fails, do not fallback to flat unless allow_fallback=True."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings and metadata mismatch")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]}")
        try:
            emb = embeddings.astype(np.float32)
            train_vecs = emb.copy(order='C')
            batch_size = 1000
            for i in range(0, len(embeddings), batch_size):
                batch_emb = train_vecs[i:i + batch_size]
                batch_meta = metadata[i:i + batch_size]
                if self.faiss_available and self.index_type in ["flat", "ivf_pq"]:
                    import faiss
                    if self.index_type == "ivf_pq":
                        if not self.index.is_trained:
                            nlist = max(1, int(math.sqrt(len(batch_emb))))
                            quantizer = faiss.IndexFlatL2(self.embedding_dim)
                            new_index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 8, 8)
                            new_index.nprobe = 20
                            try:
                                new_index.train(batch_emb)
                                self.index = new_index
                            except Exception as e:
                                logger.error(f"IVFPQ training failed: {e}")
                                if allow_fallback:
                                    logger.warning("Falling back to flat index due to training failure.")
                                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                                else:
                                    raise RuntimeError("IVFPQ training failed and fallback is not allowed.")
                    pre = int(self.index.ntotal)
                    self.index.add(batch_emb)
                    for j, meta in enumerate(batch_meta):
                        self.index_to_id[pre + j] = meta
                elif self.hnswlib_available and self.index_type == "hnsw":
                    current_elements = self.index.get_current_count()
                    if current_elements + len(batch_emb) > self.index.max_elements:
                        new_max = (current_elements + len(batch_emb)) * 2
                        logger.warning(f"HNSWlib index resize: {self.index.max_elements} -> {new_max}")
                        self.index.resize_index(new_max)
                    labels = list(range(current_elements, current_elements + len(batch_emb)))
                    self.index.add_items(batch_emb, labels)
                    for j, meta in enumerate(batch_meta):
                        self.index_to_id[current_elements + j] = meta
                gc.collect()
            logger.info(f"Added {embeddings.shape[0]} embeddings")
        except Exception as e:
            logger.error(f"Failed to add embeddings: {str(e)}")
            raise

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for k-nearest neighbors."""
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            query = q.reshape(1, -1)
        elif q.ndim == 2:
            query = q
        else:
            raise ValueError(f"Query ndim={q.ndim}")

        if query.shape[1] != self.embedding_dim:
            raise ValueError(f"Query dimension mismatch: {query.shape[1]}")

        try:
            results = []
            if self.faiss_available and self.index_type in ["flat", "ivf_pq"]:
                D, I = self.index.search(query, k)
                for r in range(I.shape[0]):
                    row = []
                    for idx_pos in range(I.shape[1]):
                        idx = int(I[r, idx_pos])
                        if idx in self.index_to_id:
                            meta = dict(self.index_to_id[idx])
                            meta["distance"] = float(D[r, idx_pos])
                            row.append(meta)
                    results.append(row)
            elif self.hnswlib_available and self.index_type == "hnsw":
                labels, distances = self.index.knn_query(query, k=k)
                for r in range(labels.shape[0]):
                    row = []
                    for idx_pos in range(labels.shape[1]):
                        idx = int(labels[r, idx_pos])
                        if idx in self.index_to_id:
                            meta = dict(self.index_to_id[idx])
                            meta["distance"] = float(distances[r, idx_pos])
                            row.append(meta)
                    results.append(row)
            logger.info(f"Search returned {len(results)} result rows")
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def save_index(self):
        """Save the index and metadata (JSONL for metadata)."""
        import json
        try:
            if self.faiss_available and self.index_type in ["flat", "ivf_pq"]:
                import faiss
                faiss.write_index(self.index, str(self.index_dir / f"{self.index_type}.index"))
            elif self.hnswlib_available and self.index_type == "hnsw":
                self.index.save_index(str(self.index_dir / "hnsw.index"))
            # Save metadata as JSONL
            meta_path = self.index_dir / "index_to_id.jsonl"
            with meta_path.open("w", encoding="utf-8") as f:
                for idx, meta in self.index_to_id.items():
                    rec = dict(meta)
                    rec["_idx"] = idx
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Saved index to {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise

    def load_index(self):
        """Load the index and metadata (JSONL for metadata)."""
        import json
        try:
            if self.faiss_available and self.index_type in ["flat", "ivf_pq"]:
                import faiss
                index_path = self.index_dir / f"{self.index_type}.index"
                if not index_path.exists():
                    raise FileNotFoundError(f"Index file not found: {index_path}")
                self.index = faiss.read_index(str(index_path))
            elif self.hnswlib_available and self.index_type == "hnsw":
                import hnswlib
                index_path = self.index_dir / "hnsw.index"
                if not index_path.exists():
                    raise FileNotFoundError(f"Index file not found: {index_path}")
                self.index = hnswlib.Index(space='l2', dim=self.embedding_dim)
                self.index.load_index(str(index_path))
                self.index.set_ef(100)
            # Load metadata from JSONL
            meta_path = self.index_dir / "index_to_id.jsonl"
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            self.index_to_id = {}
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    idx = rec.pop("_idx")
                    self.index_to_id[int(idx)] = rec
            if not isinstance(self.index_to_id, dict):
                raise ValueError("Invalid metadata format")
            logger.info(f"Loaded index from {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
    def rerank(self, candidates: List[Dict[str, Any]], query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid reranking stub for integration with hybrid_search.py.
        This should combine semantic and keyword scores for final ranking.
        """
        # Placeholder: sort by distance (semantic) ascending
        return sorted(candidates, key=lambda x: x.get('distance', 0))[:k]

if __name__ == "__main__":
    config = Config()
    store = VectorStore(config.index_dir, config.embedding_dim, index_type="ivf_pq", use_pq=True)
    
    # Test adding embeddings
    embeddings = np.random.random((1000, config.embedding_dim)).astype(np.float32)  # Larger test
    metadata = [{"id": f"chunk_{i}", "doc_id": "test_doc", "page": i % 10} for i in range(1000)]
    store.add(embeddings, metadata)
    
    # Test search
    query_embedding = np.random.random((config.embedding_dim,)).astype(np.float32)
    results = store.search(query_embedding, k=5)
    for result in results[0]:
        print(f"Result: {result['id']} (Distance: {result['distance']})")
    
    # Test save and load
    store.save_index()
    store = VectorStore(config.index_dir, config.embedding_dim, index_type="ivf_pq", use_pq=True)
    store.load_index()
    results = store.search(query_embedding, k=5)
    print("After reload:")
    for result in results[0]:
        print(f"Result: {result['id']} (Distance: {result['distance']})")