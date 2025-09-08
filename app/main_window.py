from platform import processor
import sys
import os
from pathlib import Path
import logging
import hashlib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QListWidget, QFileDialog,
    QMessageBox, QProgressBar, QListWidgetItem, QDialog
)
from PyQt6.QtWidgets import QComboBox, QSpinBox, QLabel
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from torch import layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure project root on sys.path when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _import_config_and_checks():
    """Resiliently import Config and StartupChecks from the app package."""
    try:
        from app.startup_checks import StartupChecks
        from app.config import Config
        return StartupChecks, Config
    except Exception:
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from app.startup_checks import StartupChecks
        from app.config import Config
        return StartupChecks, Config

from app.pdf_extractor import PDFExtractor
from app.normalizer import Normalizer
from app.chunker import Chunker
from app.extractive import ExtractiveSummarizer
from app.hybrid_search import HybridSearch
from app.indexer.sqlite_store import SQLiteStore
from app.indexer.vector_store import VectorStore
from app.embeddings.embedder import Embedder
from app.result_item import ResultItemWidget
from app.config import Config
from app.topic_model import TopicModel
from app.exporter import Exporter

class FileProcessor(QThread):
    progress = pyqtSignal(int)
    file_processed = pyqtSignal(str, bool, str)
    finished = pyqtSignal()

    def __init__(self, file_path, parent=None, summarizer_mode='extractive', summary_sentences=3, abstractive_max_length=150):
        super().__init__(parent)
        self.file_path = file_path
        # summarization preferences
        self.summarizer_mode = summarizer_mode
        self.summary_sentences = summary_sentences
        self.abstractive_max_length = abstractive_max_length
        logger.info(f"Starting FileProcessor for {file_path}")

    def run(self):
        try:
            logger.info(f"Processing file: {self.file_path}")
            if not os.path.exists(self.file_path):
                raise ValueError(f"File not found: {self.file_path}")
            
            file_ext = self.file_path.lower().split('.')[-1]
            if file_ext not in ['pdf', 'docx']:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            extractor = PDFExtractor(enable_ocr=True)
            normalizer = Normalizer()
            chunker = Chunker()
            # Choose summarizer based on mode
            if self.summarizer_mode == 'abstractive':
                try:
                    from app.abstractive import AbstractiveSummarizer
                    summarizer = AbstractiveSummarizer()
                except Exception as e:
                    logger.warning(f"Abstractive summarizer not available ({e}), falling back to extractive")
                    summarizer = ExtractiveSummarizer(language="english", summary_sentences=self.summary_sentences)
            else:
                summarizer = ExtractiveSummarizer(language="english", summary_sentences=self.summary_sentences)
            
            if file_ext == "pdf":
                pages = extractor.extract_text(self.file_path)
            elif file_ext == "docx":
                from app.docx_extractor import DocxExtractor
                extractor = DocxExtractor()
                pages = extractor.extract_text(self.file_path)
            
            if not pages:
                raise ValueError("No content extracted from file")
            
            logger.info(f"Extracted {len(pages)} pages from {self.file_path}")
            normalized_chunks = normalizer.normalize_chunks(pages)
            doc_id = hashlib.sha256(self.file_path.encode("utf-8")).hexdigest()
            chunks = chunker.chunk_document(normalized_chunks, doc_id=doc_id)
            logger.info(f"Chunked into {len(chunks)} chunks")
            
            # summarizer.summarize expects a list of chunk dicts; pass chunks
            # Auto-detect language (simple heuristic on Unicode range for Bengali)
            try:
                combined_text = "\n".join(ch.get('text', '') for ch in chunks if ch.get('text'))
                bengali_chars = sum(1 for c in combined_text if '\u0980' <= c <= '\u09FF')
                lang = 'bengali' if (bengali_chars > 0 and bengali_chars / max(1, len(combined_text)) > 0.001) else 'english'
                logger.info(f"Auto-detected language: {lang} (bengali_chars={bengali_chars}, total_chars={len(combined_text)})")
            except Exception:
                lang = 'english'

            # If abstractive requested but language is Bengali, fallback to extractive (abstractive models may not support Bengali well)
            use_abstractive = (self.summarizer_mode == 'abstractive') and lang != 'bengali'

            # Abstractive summarizer returns dict with 'summary' key; extractive too
            if use_abstractive and getattr(summarizer, 'summarize', None):
                try:
                    result = summarizer.summarize(chunks, max_length=self.abstractive_max_length)
                except Exception as e:
                    logger.warning(f"Abstractive summarizer failed, falling back to extractive: {e}")
                    summarizer = ExtractiveSummarizer(language=lang, summary_sentences=self.summary_sentences)
                    result = summarizer.summarize(chunks)
            else:
                # ensure extractive summarizer uses detected language
                if not isinstance(summarizer, ExtractiveSummarizer):
                    summarizer = ExtractiveSummarizer(language=lang, summary_sentences=self.summary_sentences)
                else:
                    try:
                        summarizer.language = lang
                        summarizer.summary_sentences = self.summary_sentences
                    except Exception:
                        pass
                result = summarizer.summarize(chunks)
            if isinstance(result, dict):
                summary = result.get("summary", "") or "No significant summary available."
            else:
                summary = str(result) or "No significant summary available."
            
            for i in range(10, 101, 10):
                self.progress.emit(i)
                self.msleep(100)
            
            self.file_processed.emit(self.file_path, True, summary)
            logger.info(f"File processing completed for {self.file_path}")
            
        except ValueError as e:
            error_msg = f"Invalid file or content: {str(e)}"
            logger.error(f"Validation error for {self.file_path}: {error_msg}")
            self.file_processed.emit(self.file_path, False, error_msg)
        except RuntimeError as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Runtime error for {self.file_path}: {error_msg}")
            self.file_processed.emit(self.file_path, False, error_msg)
        except Exception as e:
            if "nltk" in str(e).lower() and "punkt" in str(e).lower():
                error_msg = ("NLTK tokenizers are missing or the language is not supported.\n"
                             "Please run: python -c \"import nltk; nltk.download('punkt')\" "
                             "to download the required resource.")
                logger.error(f"NLTK error for {self.file_path}: {error_msg}")
                self.file_processed.emit(self.file_path, False, error_msg)
            else:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error for {self.file_path}: {error_msg}")
                self.file_processed.emit(self.file_path, False, error_msg)
        finally:
            self.finished.emit()
            logger.info(f"FileProcessor finished for {self.file_path}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from PyQt6.QtGui import QIcon
        import os
        logo_path = os.path.join(os.path.dirname(__file__), 'App_logo.png')
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        self.setWindowTitle("Instant Document Summarizer & Search")
        self.setGeometry(100, 100, 900, 650)

        StartupChecks, Config = _import_config_and_checks()
        self.config = Config()

        # Run startup checks (constructor runs checks)
        try:
            startup_checks = StartupChecks(str(self.config.project_root / "data"))
            # If any checks failed, show detailed messages and exit
            if not startup_checks.all_checks_passed():
                error_messages = startup_checks.get_error_messages()
                QMessageBox.critical(self, "Startup Error",
                                     "Some dependencies are missing or misconfigured:\n" + "\n".join(error_messages))
                sys.exit(1)
            else:
                # Optionally print a short report to logs
                logger.info("Startup checks passed")
        except Exception as e:
            logger.exception("Failed running startup checks: %s", str(e))
            QMessageBox.critical(self, "Startup Error", f"Failed to run startup checks: {str(e)}")
            sys.exit(1)

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.upload_button = QPushButton("Upload File")
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button)

        # Summarizer controls: mode selector and length controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Summarizer:"))
        self.summarizer_selector = QComboBox()
        self.summarizer_selector.addItems(["extractive", "abstractive"])
        control_layout.addWidget(self.summarizer_selector)

        control_layout.addWidget(QLabel("Sentences:"))
        self.sentences_spin = QSpinBox()
        self.sentences_spin.setRange(1, 10)
        self.sentences_spin.setValue(3)
        control_layout.addWidget(self.sentences_spin)

        control_layout.addWidget(QLabel("Abstractive max len:"))
        self.abstractive_len = QSpinBox()
        self.abstractive_len.setRange(50, 512)
        # load default from config if present
        try:
            self.abstractive_len.setValue(getattr(self.config, 'abstractive_max_length', 150))
        except Exception:
            self.abstractive_len.setValue(150)
        control_layout.addWidget(self.abstractive_len)

        layout.addLayout(control_layout)

        # Drag-and-drop area for files
        self.setAcceptDrops(True)
        self.drag_label = QTextEdit("Drag and drop files here")
        self.drag_label.setReadOnly(True)
        self.drag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.drag_label)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Type to search...")
        self.search_bar.textChanged.connect(self.update_search_suggestions)
        self.search_bar.returnPressed.connect(self.show_search_results)
        layout.addWidget(self.search_bar)

        self.suggestion_list = QListWidget()
        self.suggestion_list.setVisible(False)
        self.suggestion_list.itemClicked.connect(self.select_suggestion)
        layout.addWidget(self.suggestion_list)

        self.summary_display = QTextEdit()
        self.summary_display.setReadOnly(True)
        layout.addWidget(self.summary_display)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.recent_files_button = QPushButton("Recent Files")
        self.recent_files_button.clicked.connect(self.show_recent_files)
        layout.addWidget(self.recent_files_button)

        self.save_button = QPushButton("Save Summary")
        self.save_button.clicked.connect(self.save_summary)
        layout.addWidget(self.save_button)

        self.about_button = QPushButton("About")
        self.about_button.clicked.connect(self.show_about)
        layout.addWidget(self.about_button)

        # Topic list (populated from TopicModel)
        self.topic_list = QListWidget()
        self.topic_list.setObjectName("topicList")
        layout.addWidget(self.topic_list)

        # Export button (export summary or search results)
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        layout.addWidget(self.export_button)

        # Dark mode toggle button for runtime switching
        self.dark_toggle_button = QPushButton("Toggle Dark Mode")
        self.dark_toggle_button.clicked.connect(self.toggle_dark_mode)
        layout.addWidget(self.dark_toggle_button)

        # Initialize stores and helpers (lazy-created)
        self.sqlite_store = None
        self.vector_store = None
        self.embedder = None
        self.hybrid_search = None

        # Track active FileProcessor threads so we can wait for them on exit
        self.active_threads = []

        # TopicModel and Exporter (lazy init)
        self.topic_model = None
        self.exporter = None

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Documents (*.pdf *.docx)")
        if file_path:
            self.start_processing(file_path)

    def start_processing(self, file_path):
        """Start processing a file programmatically (used by upload_file and CLI testing)."""
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Invalid file path: {file_path}")
            QMessageBox.critical(self, "Error", f"Invalid or non-existent file path: {file_path}")
            return
        # Check if a thread for this file is already active
        if any(t.file_path == file_path for t in self.active_threads):
            logger.warning(f"Processing already in progress for {file_path}")
            return
        self.progress_bar.setVisible(True)
        # read summarizer preferences from UI
        summarizer_mode = self.summarizer_selector.currentText()
        summary_sentences = int(self.sentences_spin.value())
        abstractive_max_length = int(self.abstractive_len.value())
        # persist into config
        try:
            self.config.summary_sentences = summary_sentences
            self.config.abstractive_max_length = abstractive_max_length
            self.config.save_config(str(self.config.project_root / 'data' / 'app_config.json'))
        except Exception:
            pass

        processor = FileProcessor(file_path, parent=self, summarizer_mode=summarizer_mode, summary_sentences=summary_sentences, abstractive_max_length=abstractive_max_length)
        processor.progress.connect(self.progress_bar.setValue)
        processor.file_processed.connect(self.file_processed)
        processor.finished.connect(lambda p=processor: self._cleanup_thread(p))
        self.active_threads.append(processor)
        processor.start()
        logger.info("Started processing %s", file_path)

    def _cleanup_thread(self, thread):
        """Remove the thread from the active_threads list when it finishes."""
        try:
            if thread in self.active_threads:
                self.active_threads.remove(thread)
        except Exception:
            pass

    def file_processed(self, file_path, success, summary):
        self.progress_bar.setVisible(False)
        if success:
            self.display_summary_with_highlight(summary)
            QMessageBox.information(self, "Success", f"Summary generated for {Path(file_path).name}")
        else:
            QMessageBox.critical(self, "Error", f"Failed to process {Path(file_path).name}: {summary}")

    def display_summary_with_highlight(self, summary):
        """
        Display summary and highlight search terms if present in the search bar.
        """
        self.summary_display.clear()
        self.summary_display.setPlainText(summary)
        query = self.search_bar.text().strip()
        if query:
            cursor = self.summary_display.textCursor()
            fmt = cursor.charFormat()
            fmt.setBackground(Qt.GlobalColor.yellow)
            pattern = query
            doc = self.summary_display.document()
            cursor.beginEditBlock()
            pos = 0
            while True:
                cursor = doc.find(pattern, pos)
                if cursor.isNull():
                    break
                cursor.mergeCharFormat(fmt)
                pos = cursor.position()
            cursor.endEditBlock()

    def update_search_suggestions(self, text):
        if not text or not self.sqlite_store:
            self.suggestion_list.clear()
            self.suggestion_list.setVisible(False)
        # Get suggestions from SQLiteStore (implement get_suggestions)
        try:
            suggestions = self.sqlite_store.get_suggestions(text)
        except Exception:
            suggestions = []
        self.suggestion_list.clear()
        if suggestions:
            self.suggestion_list.addItems(suggestions)
            self.suggestion_list.setVisible(True)
        else:
            self.suggestion_list.setVisible(False)

    def select_suggestion(self, item):
        self.search_bar.setText(item.text())
        self.suggestion_list.setVisible(False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file_path in files:
            if file_path.lower().endswith(('.pdf', '.docx')):
                self.start_processing(file_path)
            else:
                QMessageBox.warning(self, "Unsupported File", f"Unsupported file type: {file_path}")

    def export_results(self):
        # Export summary or search results to file
        summary = self.summary_display.toPlainText()
        if not summary:
            QMessageBox.warning(self, "Export", "No summary to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Text Files (*.txt);;Word Documents (*.docx);;PDF Files (*.pdf)")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                QMessageBox.information(self, "Export", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def show_search_results(self):
        query = self.search_bar.text().strip()
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a search query.")
            return
        if not self.sqlite_store:
            self.sqlite_store = SQLiteStore(db_path=self.config.db_path)
            self.vector_store = VectorStore(index_dir=self.config.index_dir, embedding_dim=self.config.embedding_dim)
            self.embedder = Embedder(model_name=self.config.embedding_model)
            self.hybrid_search = HybridSearch(self.sqlite_store, self.vector_store, self.embedder)
        results = self.hybrid_search.search(query, top_k=5)
        dialog = QDialog(self)
        dialog.setWindowTitle("Search Results")
        layout = QVBoxLayout(dialog)
        for result in results:
            widget = ResultItemWidget(result)
            layout.addWidget(widget)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        close_button.clicked.connect(self.search_bar.clear)
        layout.addWidget(close_button)
        dialog.exec()

    def save_summary(self):
        summary = self.summary_display.toPlainText()
        if not summary:
            QMessageBox.warning(self, "Error", "No summary to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Summary", "", "Text Files (*.txt);;Word Documents (*.docx);;PDF Files (*.pdf)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(summary)
            QMessageBox.information(self, "Success", f"Summary saved to {file_path}")

    def show_recent_files(self):
        QMessageBox.information(self, "Recent Files", "Recent files functionality not implemented yet.")

    def show_about(self):
        QMessageBox.information(self, "About", "Instant Document Summarizer & Search\nVersion 1.0\nDeveloped with PyQt6.")

    def toggle_dark_mode(self):
        """Toggle dark mode property on the window, all children, and config."""
        try:
            current = bool(self.property('darkMode'))
            new = not current
            self.setProperty('darkMode', new)
            if hasattr(self, 'config'):
                self.config.dark_mode = new
                try:
                    self.config.save_config(str(self.config.project_root / 'data' / 'app_config.json'))
                except Exception:
                    pass
            for w in self.findChildren(QWidget):
                try:
                    w.setProperty('darkMode', new)
                    w.style().unpolish(w)
                    w.style().polish(w)
                except Exception:
                    pass
            QApplication.instance().setStyleSheet(QApplication.instance().styleSheet())
        except Exception:
            logger.exception("Failed to toggle dark mode")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            if self.width() < 600:
                self.setProperty('screenSize', 'small')
            else:
                self.setProperty('screenSize', 'normal')
            # re-polish to apply any size-based selectors
            self.style().unpolish(self)
            self.style().polish(self)
        except Exception:
            pass

    def closeEvent(self, event):
        # Wait for all active threads to finish before exiting
        logger.info("Shutting down: %d active FileProcessor threads", len(self.active_threads))
        for thread in list(self.active_threads):
            try:
                logger.info("Thread %s running=%s", repr(thread), thread.isRunning())
                if thread.isRunning():
                    # First try a cooperative wait with timeout
                    finished = thread.wait(3000)  # wait up to 3s
                    if not finished and thread.isRunning():
                        logger.warning("Thread %s still running after wait, attempting terminate()", repr(thread))
                        try:
                            thread.terminate()
                        except Exception:
                            logger.exception("Failed to terminate thread %s", repr(thread))
                        # allow a moment for termination
                        thread.wait(1000)
            except Exception:
                logger.exception("Error while waiting for thread %s", repr(thread))
            finally:
                self._cleanup_thread(thread)

        # Clean up resources
        if self.sqlite_store:
            self.sqlite_store.close()
        if self.vector_store:
            try:
                self.vector_store.close()
            except Exception:
                pass
        if self.embedder:
            try:
                self.embedder.clear_cache()
            except Exception:
                pass

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Load stylesheet if available. Prefer app_style.qss, fallback to styles.qss
    try:
        qss_paths = [
            Path(__file__).resolve().parents[0] / "app_style.qss",
            Path(__file__).resolve().parents[0] / "styles.qss",
        ]
        qss_content = None
        for p in qss_paths:
            if p.exists():
                qss_content = p.read_text(encoding='utf-8')
                break
        if qss_content:
            app.setStyleSheet(qss_content)
    except Exception:
        logger.exception("Failed to load QSS styles")
    window = MainWindow()
    # If the config or environment indicates dark mode, set property on main window
    try:
        if hasattr(window, 'config') and getattr(window.config, 'dark_mode', False):
            window.setProperty('darkMode', True)
            # re-apply style to allow property selectors to take effect
            app.setStyleSheet(app.styleSheet())
    except Exception:
        pass
    window.show()
    # Handle test file from CLI
    # Usage: python app\main_window.py --test-file "C:\\path\\to\\file.pdf"
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--test-file" and i < len(sys.argv) - 1:
            test_path = sys.argv[i + 1]
            if test_path and os.path.exists(test_path):
                window.start_processing(test_path)
            else:
                logger.error(f"Invalid test file path: {test_path}")
                QMessageBox.critical(window, "Error", f"Invalid or non-existent test file path: {test_path}")
            break  # Process only the first valid --test-file argument

    sys.exit(app.exec())