import sqlite3
import importlib.util
import shutil
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class StartupChecks:
    def __init__(self, data_dir: str = "data"):
        """Initialize with path to data directory."""
        self.data_dir = Path(data_dir)
        self.checks: List[Tuple[bool, str, str]] = []  # (status, message, tip)

    def run_all_checks(self, min_free_space_gb: float = 1.0) -> List[Tuple[bool, str, str]]:
        """
        Run all startup checks and return results.
        min_free_space_gb: Minimum required free space in GB (default 1.0, set higher for large corpus).
        """
        self.checks = []
        self._check_sqlite_fts5()
        self._check_vector_store()
        self._check_tesseract()
        self._check_sentence_transformers()
        self._check_nltk()
        self._check_data_dir(min_free_space_gb=min_free_space_gb)
        return self.checks
    def _check_nltk(self) -> None:
        """Check if NLTK and required resources (punkt) are installed."""
        try:
            import importlib
            if importlib.util.find_spec("nltk") is None:
                raise ImportError("nltk not installed")
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
                self.checks.append((True, "NLTK: punkt tokenizer available", ""))
            except LookupError:
                self.checks.append((False, "NLTK: punkt tokenizer missing", "Run: python -c \"import nltk; nltk.download('punkt')\"") )
        except Exception as e:
            self.checks.append((False, f"NLTK: not available ({str(e)})", "Install nltk (`pip install nltk`) and download punkt tokenizer."))

    def _check_sqlite_fts5(self) -> None:
        """Check if SQLite FTS5 extension is enabled."""
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE VIRTUAL TABLE test USING fts5(content)")
            self.checks.append((True, "SQLite FTS5: enabled", ""))
            conn.close()
        except sqlite3.OperationalError as e:
            self.checks.append((
                False,
                f"SQLite FTS5: not enabled ({str(e)})",
                "Ensure SQLite is compiled with FTS5 support. Check your SQLite version or install a compatible version."
            ))

    def _check_vector_store(self) -> None:
        """Check if either faiss-cpu or hnswlib is installed."""
        faiss_installed = importlib.util.find_spec("faiss") is not None
        hnswlib_installed = importlib.util.find_spec("hnswlib") is not None

        if faiss_installed:
            self.checks.append((True, "faiss-cpu: installed", ""))
        elif hnswlib_installed:
            self.checks.append((True, "hnswlib: installed", ""))
        else:
            self.checks.append((
                False,
                "Vector store: neither faiss-cpu nor hnswlib installed",
                "Install either faiss-cpu (`pip install faiss-cpu`) or hnswlib (`pip install hnswlib`)."
            ))

    def _check_tesseract(self) -> None:
        """Check if Tesseract executable is available in PATH."""
        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            self.checks.append((True, f"Tesseract: found at {tesseract_path}", ""))
        else:
            self.checks.append((
                False,
                "Tesseract: not found in PATH",
                "Install Tesseract OCR and ensure it's added to your system PATH. Download from https://github.com/tesseract-ocr/tesseract."
            ))

    def _check_sentence_transformers(self) -> None:
        """Check if sentence-transformers and torch are installed and model loads."""
        try:
            if importlib.util.find_spec("sentence_transformers") is None:
                raise ImportError("sentence-transformers not installed")
            if importlib.util.find_spec("torch") is None:
                raise ImportError("PyTorch not installed")

            # Import lazily to avoid hard dependency at module import time
            st_mod = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(st_mod, "SentenceTransformer")

            # Try loading the configured embedding model to verify functionality
            # Use a resilient import so running the module as a script (where
            # sys.path[0] may be the package dir) still finds the top-level
            # `app` package. If the direct import fails, add the project root
            # to sys.path and retry.
            try:
                from app.config import Config
            except Exception:
                project_root = Path(__file__).resolve().parents[1]
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from app.config import Config
            cfg = Config()
            model = SentenceTransformer(cfg.embedding_model)
            test_text = ["This is a test sentence."]
            # encode may return numpy array or torch tensor depending on backend
            _ = model.encode(test_text)
            self.checks.append((True, "sentence-transformers: model loaded", ""))
        except Exception as e:
            self.checks.append((
                False,
                f"sentence-transformers: failed to load ({str(e)})",
                "Install sentence-transformers (`pip install sentence-transformers`) and PyTorch (`pip install torch`). Ensure enough memory is available."
            ))

    def _check_data_dir(self, min_free_space_gb: float = 1.0) -> None:
        """Check if data directory is writable and has enough disk space. Warn if <5GB, error if <min_free_space_gb."""
        try:
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True)
            # Check writability
            test_file = self.data_dir / ".test_write"
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()
            # Check disk space using shutil.disk_usage (stdlib)
            disk = shutil.disk_usage(self.data_dir)
            free_space_gb = disk.free / (1024 ** 3)  # Convert to GB
            if free_space_gb < min_free_space_gb:
                self.checks.append((
                    False,
                    f"Data dir: insufficient disk space ({free_space_gb:.2f}GB free)",
                    f"Free up at least {min_free_space_gb}GB in {self.data_dir} or change the data directory."
                ))
            elif free_space_gb < 5.0:
                self.checks.append((
                    True,
                    f"Data dir: low disk space warning ({free_space_gb:.2f}GB free)",
                    "Consider freeing up disk space for large corpus support."
                ))
            else:
                self.checks.append((True, f"Data dir: writable, {free_space_gb:.2f}GB free", ""))
        except Exception as e:
            self.checks.append((
                False,
                f"Data dir: not writable ({str(e)})",
                f"Ensure {self.data_dir} is writable and has sufficient disk space."
            ))
    def get_checks_for_gui(self) -> List[dict]:
        """
        Return check results as a list of dicts for GUI display.
        Each dict: {'status': bool, 'message': str, 'tip': str}
        """
        return [
            {'status': status, 'message': message, 'tip': tip}
            for status, message, tip in self.checks
        ]

    def print_report(self) -> None:
        """Print a human-readable report of all checks."""
        for status, message, tip in self.checks:
            symbol = "[✓]" if status else "[✗]"
            logger.info(f"{symbol} {message}")
            if not status and tip:
                logger.info(f"    Tip: {tip}")

    def has_critical_failure(self) -> bool:
        """Return True if any check failed."""
        return any(not status for status, _, _ in self.checks)

    def all_checks_passed(self) -> bool:
        """Return True if all checks passed."""
        return not self.has_critical_failure()

    def get_error_messages(self) -> List[str]:
        """Return list of error messages for failed checks."""
        return [message for status, message, _ in self.checks if not status]

if __name__ == "__main__":
    checker = StartupChecks(data_dir="data")
    checker.run_all_checks()
    checker.print_report()
    if checker.has_critical_failure():
        sys.exit(1)