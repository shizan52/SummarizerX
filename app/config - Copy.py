import json
import logging
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class with privacy and offline support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Configurable parameters for the application.
        - embedding_model: Default is 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual, recommended).
        - dark_mode: Boolean, controls dark mode in GUI (toggleable from GUI).
        - search_alpha, search_beta, search_gamma: Search weights (should sum to 1.0), for GUI sliders.
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.index_dir = self.project_root / "data" / "indexes"
        self.embeddings_dir = self.project_root / "data" / "embeddings"
        self.db_path = self.project_root / "data" / "index.db"
        self.models_dir = self.project_root / "data" / "models"
        # Encryption key
        self.encryption_key = Fernet.generate_key()
        # Embedding model: default is multilingual MiniLM-L12-v2
        self.embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
        self.embedding_dim = 384
        self.language = "multi"
        # Search weights for hybrid search (should sum to 1.0, exposed as sliders in GUI)
        self.search_alpha = 0.5  # Semantic weight
        self.search_beta = 0.4   # Keyword weight
        self.search_gamma = 0.1  # Recency/other weight
        self.summary_sentences = 3
        # Abstractive summarizer max length (in tokens/words depending on model)
        self.abstractive_max_length = 150
        self.chunk_size = 512
        self.chunk_overlap = 50
        # Dark mode toggle for GUI
        self.dark_mode = False
        if config_path:
            self.load_config(config_path)
        self.validate_config()

    def load_config(self, config_path: str):
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with config_path.open("r") as f:
                config_data = json.load(f)
            
            self.index_dir = Path(config_data.get("index_dir", self.index_dir))
            self.embeddings_dir = Path(config_data.get("embeddings_dir", self.embeddings_dir))
            self.db_path = Path(config_data.get("db_path", self.db_path))
            self.models_dir = Path(config_data.get("models_dir", self.models_dir))
            self.encryption_key = config_data.get("encryption_key", self.encryption_key)
            self.embedding_model = config_data.get("embedding_model", self.embedding_model)
            self.embedding_dim = config_data.get("embedding_dim", self.embedding_dim)
            self.language = config_data.get("language", self.language)
            self.search_alpha = config_data.get("search_alpha", self.search_alpha)
            self.search_beta = config_data.get("search_beta", self.search_beta)
            self.search_gamma = config_data.get("search_gamma", self.search_gamma)
            self.summary_sentences = config_data.get("summary_sentences", self.summary_sentences)
            self.abstractive_max_length = config_data.get("abstractive_max_length", self.abstractive_max_length)
            self.chunk_size = config_data.get("chunk_size", self.chunk_size)
            self.chunk_overlap = config_data.get("chunk_overlap", self.chunk_overlap)
            self.dark_mode = config_data.get("dark_mode", self.dark_mode)
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def validate_config(self):
        try:
            for path in [self.index_dir, self.embeddings_dir, self.models_dir]:
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {path}")
                if not path.is_dir():
                    raise ValueError(f"Path is not a directory: {path}")
            
            if not self.db_path.parent.exists():
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory for database: {self.db_path.parent}")
            
            if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
                raise ValueError(f"Invalid embedding_dim: {self.embedding_dim}")
            
            total_weight = self.search_alpha + self.search_beta + self.search_gamma
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(f"Search weights must sum to 1.0, got {total_weight}")
            if any(w < 0 for w in [self.search_alpha, self.search_beta, self.search_gamma]):
                raise ValueError("Search weights must be non-negative")
            
            if not isinstance(self.summary_sentences, int) or self.summary_sentences <= 0:
                raise ValueError(f"Invalid summary_sentences: {self.summary_sentences}")
            
            if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
                raise ValueError(f"Invalid chunk_size: {self.chunk_size}")
            if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
                raise ValueError(f"Invalid chunk_overlap: {self.chunk_overlap}")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
            
            logger.info("Configuration validated successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

    def save_config(self, config_path: str):
        """
        Save configuration to a JSON file. GUI should call this after user changes (e.g., dark_mode toggle, search weights sliders).
        """
        config_data = {
            "index_dir": str(self.index_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "db_path": str(self.db_path),
            "models_dir": str(self.models_dir),
            "encryption_key": self.encryption_key.decode(),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "language": self.language,
            "search_alpha": self.search_alpha,
            "search_beta": self.search_beta,
            "search_gamma": self.search_gamma,
            "summary_sentences": self.summary_sentences,
            "abstractive_max_length": self.abstractive_max_length,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "dark_mode": self.dark_mode
        }
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w") as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
            raise

if __name__ == "__main__":
    config = Config()
    logger.info("Configuration loaded with defaults:")
    logger.info(f"Embedding Model: {config.embedding_model}")
    logger.info(f"Language: {config.language}")
    logger.info(f"Embedding Dim: {config.embedding_dim}")