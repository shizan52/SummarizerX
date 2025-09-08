import hashlib
import logging
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import lmdb
from sentence_transformers import SentenceTransformer
import pickle
import sys
import tempfile
import os
import time
import gc
import unicodedata
from cryptography.fernet import Fernet

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config

# Configure logging with privacy
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Embedder:
    """Class to handle text embedding with privacy and offline support."""
    
    def __init__(self, model_name: Optional[str] = None, cache_dir: Path = None, lang: str = 'multi', batch_size: Optional[int] = None):
        config = Config()
        self.model_name = model_name or getattr(config, 'embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.cache_dir = Path(cache_dir) if cache_dir else Path(getattr(config, 'embeddings_dir', './embeddings'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lang = lang or getattr(config, 'language', 'multi')
        # Encryption key
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        # Initialize LMDB
        self.env = None
        self._init_lmdb(map_size=1024*1024*1024 * 5)
        # Load model offline
        try:
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir / "models")
            logger.info(f"Loaded model: {self.model_name} from local cache")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
        # Batch size configurable (default 64, can override)
        self.batch_size = batch_size or getattr(config, 'embedding_batch_size', 64)

    def _init_lmdb(self, map_size: int):
        try:
            self.env = lmdb.open(str(self.cache_dir / "embeddings.lmdb"), 
                                 map_size=map_size, max_dbs=1)
            logger.info(f"Initialized LMDB cache with map_size {map_size / (1024*1024*1024)} GB")
        except Exception as e:
            logger.error(f"Failed to initialize LMDB: {str(e)}")
            raise

    def _normalize_text(self, text: str) -> str:
        """Normalize text for multilang support."""
        return unicodedata.normalize('NFKC', text).strip()

    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data for storage."""
        return self.cipher.encrypt(data)

    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data from storage."""
        return self.cipher.decrypt(encrypted_data)

    def embed_and_cache(self, texts: List[str], force_recompute: bool = False, batch_size: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
        """
        Embed texts and cache results. Optionally force recompute and set batch size per call.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        batch_size = batch_size or self.batch_size
        normalized_texts = [self._normalize_text(t) for t in texts]
        chunk_ids = [f"chunk_{i}_{hashlib.sha256(self._encrypt_data(t.encode())).hexdigest()}" for i, t in enumerate(normalized_texts)]
        embeddings = []
        to_compute = []
        to_compute_ids = []
        with self.env.begin() as txn:
            for cid, text in zip(chunk_ids, normalized_texts):
                cached = txn.get(cid.encode())
                if cached and not force_recompute:
                    decrypted = self._decrypt_data(cached)
                    embeddings.append(np.frombuffer(decrypted, dtype=np.float32))
                else:
                    to_compute.append(text)
                    to_compute_ids.append(cid)
        if to_compute:
            try:
                new_embeddings = []
                for i in range(0, len(to_compute), batch_size):
                    batch = to_compute[i:i + batch_size]
                    batch_emb = self.model.encode(
                        batch,
                        batch_size=len(batch),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    ).astype(np.float16)
                    new_embeddings.extend(batch_emb)
                    gc.collect()
                with self.env.begin(write=True) as txn:
                    for cid, emb in zip(to_compute_ids, new_embeddings):
                        encrypted_emb = self._encrypt_data(emb.tobytes())
                        txn.put(cid.encode(), encrypted_emb)
                embeddings.extend(new_embeddings)
            except Exception as e:
                logger.error(f"Failed to compute embeddings: {str(e)}")
                raise
        return chunk_ids, np.array(embeddings).astype(np.float32)

    def clear_cache(self, retries: int = 3, timeout: int = 10):
        start_time = time.time()
        for attempt in range(retries):
            try:
                if self.env:
                    self.env.close()
                lmdb_path = self.cache_dir / "embeddings.lmdb"
                if lmdb_path.exists():
                    lock_file = lmdb_path / "lock.mdb"
                    waited = 0
                    while lock_file.exists() and waited < timeout:
                        logger.warning("LMDB lock detected. Waiting...")
                        time.sleep(1)
                        waited += 1
                    if lock_file.exists():
                        raise RuntimeError("LMDB lock file still present after timeout. Please close other processes using the cache.")
                    import shutil
                    shutil.rmtree(lmdb_path)
                    logger.info(f"Cleared LMDB cache at {lmdb_path}")
                self._init_lmdb(map_size=1024*1024*1024 * 5)
                return
            except Exception as e:
                logger.warning(f"Clear cache attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2)
        raise RuntimeError("Failed to clear LMDB cache after retries or lock timeout")

    def test_offline_embedding(self, text: str = "Test text") -> np.ndarray:
        """Test embedding generation in offline mode."""
        try:
            embedding = self.model.encode([text], batch_size=1, convert_to_numpy=True)
            logger.info("Successfully generated embedding in offline mode")
            return embedding
        except Exception as e:
            logger.error(f"Offline embedding test failed: {str(e)}")
            raise

if __name__ == "__main__":
    config = Config()
    embedder = Embedder(model_name=config.embedding_model, cache_dir=config.embeddings_dir, batch_size=getattr(config, 'embedding_batch_size', 64))
    # Test offline embedding
    embedding = embedder.test_offline_embedding("এটি একটি বাংলা বাক্য।")
    logger.info(f"Offline embedding shape: {embedding.shape}")
    # Test batch embedding (multilingual)
    texts = ["This is a test.", "এটি একটি পরীক্ষা।", "यह एक परीक्षा है।", "Ceci est un test."]
    chunk_ids, embeddings = embedder.embed_and_cache(texts, force_recompute=True, batch_size=64)
    logger.info(f"Embedded {len(chunk_ids)} texts with shape: {embeddings.shape}")
    # Test cache clearing
    try:
        embedder.clear_cache()
        logger.info("Cache cleared successfully.")
    except Exception as e:
        logger.error(f"Cache clear failed: {str(e)}")