from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt6.QtCore import Qt, pyqtSignal
import logging

logger = logging.getLogger(__name__)

class ResultWidget(QWidget):
    """A widget for displaying a single search result with metadata and clickable action."""
    result_clicked = pyqtSignal(dict)  # Emits search result dictionary

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result
        self.setProperty("darkMode", False)
        self.init_ui()
        logger.info("ResultWidget initialized")

    def init_ui(self):
        """Initialize the widget layout and components."""
        layout = QVBoxLayout(self)
        
        # Scrollable area for larger results
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Score and snippet
        score_label = QLabel(f"Score: {self.result.get('score', 0.0):.2f}")
        score_label.setObjectName("scoreLabel")
        scroll_layout.addWidget(score_label)
        
        # Highlight search terms in snippet
        snippet_text = self.result.get('snippet', 'No snippet available')
        search_term = self.result.get('search_term', '')
        if search_term:
            import re
            pattern = re.escape(search_term)
            snippet_html = re.sub(f"({pattern})", r'<span style="background:yellow">\\1</span>', snippet_text, flags=re.IGNORECASE)
        else:
            snippet_html = snippet_text
        snippet_label = QLabel()
        snippet_label.setTextFormat(Qt.TextFormat.RichText)
        snippet_label.setText(snippet_html)
        snippet_label.setWordWrap(True)
        snippet_label.setObjectName("snippetLabel")
        scroll_layout.addWidget(snippet_label)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # View in PDF button
        view_btn = QPushButton("View in PDF")
        view_btn.setObjectName("viewPdfButton")
        view_btn.clicked.connect(self.open_pdf_at_page)
        layout.addWidget(view_btn)
        
        # View details button
        details_button = QPushButton("View Details")
        details_button.setObjectName("detailsButton")
        details_button.clicked.connect(self.emit_clicked)
        layout.addWidget(details_button)
        
        self.setLayout(layout)

    def open_pdf_at_page(self):
        pdf_path = self.result.get("pdf_path")
        page = self.result.get("page")
        if pdf_path and page:
            import os
            import platform
            try:
                if platform.system() == "Windows":
                    os.startfile(pdf_path)
                elif platform.system() == "Darwin":
                    os.system(f"open '{pdf_path}'")
                else:
                    os.system(f"xdg-open '{pdf_path}'")
            except Exception:
                pass

    def emit_clicked(self):
        self.result_clicked.emit(self.result)

    def set_dark_mode(self, enabled):
        self.setProperty("darkMode", enabled)
        self.style().unpolish(self)
        self.style().polish(self)
        for child in self.findChildren(QWidget):
            try:
                child.setProperty("darkMode", enabled)
                child.style().unpolish(child)
                child.style().polish(child)
            except Exception:
                pass
import logging
from pathlib import Path
from typing import Dict, Any
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging to match other modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ResultWidget(QWidget):
    """
    A widget for displaying a single search result with metadata and clickable action.
    """
    result_clicked = pyqtSignal(dict)  # Emits search result dictionary

    def __init__(self, result: Dict[str, Any], parent=None):
        import logging
        from pathlib import Path
        from typing import Dict, Any
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
        from PyQt6.QtCore import Qt, pyqtSignal
        import sys

        # Ensure project root is on sys.path
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        # Configure logging to match other modules
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)


        class ResultWidget(QWidget):
            """
            A widget for displaying a single search result with metadata and clickable action.
            """
            result_clicked = pyqtSignal(dict)  # Emits search result dictionary

            def __init__(self, result: Dict[str, Any], parent=None):
                super().__init__(parent)
                self.result = result
                self.setProperty("darkMode", False)  # ডার্ক মোড প্রপার্টি
                self.init_ui()
                logger.info("ResultWidget initialized")

            def init_ui(self):
                """Initialize the widget layout and components."""
                layout = QVBoxLayout(self)
                # Score and snippet
                score_label = QLabel(f"Score: {self.result.get('score', 0.0):.2f}")
                score_label.setObjectName("scoreLabel")
                layout.addWidget(score_label)
                # Highlight search terms in snippet
                snippet_text = self.result.get('snippet', 'No snippet available')
                search_term = self.result.get('search_term', '')
                if search_term:
                    import re
                    pattern = re.escape(search_term)
                    snippet_html = re.sub(f"({pattern})", r'<span style="background:yellow">\\1</span>', snippet_text, flags=re.IGNORECASE)
                else:
                    snippet_html = snippet_text
                snippet_label = QLabel()
                snippet_label.setTextFormat(Qt.TextFormat.RichText)
                snippet_label.setText(snippet_html)
                snippet_label.setWordWrap(True)
                snippet_label.setObjectName("snippetLabel")
                layout.addWidget(snippet_label)
                # View in PDF button
                view_btn = QPushButton("View in PDF")
                view_btn.setObjectName("viewPdfButton")
                view_btn.clicked.connect(self.open_pdf_at_page)
                layout.addWidget(view_btn)
                # View details button
                details_button = QPushButton("View Details")
                details_button.setObjectName("detailsButton")
                details_button.clicked.connect(self.emit_clicked)
                layout.addWidget(details_button)
                self.setLayout(layout)

            def open_pdf_at_page(self):
                pdf_path = self.result.get("pdf_path")
                page = self.result.get("page")
                if pdf_path and page:
                    import os
                    import platform
                    try:
                        if platform.system() == "Windows":
                            os.startfile(pdf_path)
                        elif platform.system() == "Darwin":
                            os.system(f"open '{pdf_path}'")
                        else:
                            os.system(f"xdg-open '{pdf_path}'")
                    except Exception:
                        pass

            def set_dark_mode(self, enabled):
                self.setProperty("darkMode", enabled)
                self.style().unpolish(self)
                self.style().polish(self)
                for child in self.findChildren(QWidget):
                    try:
                        child.setProperty("darkMode", enabled)
                        child.style().unpolish(child)
                        child.style().polish(child)
                    except Exception:
                        pass

            def emit_clicked(self):
                self.result_clicked.emit(self.result)

            # ডার্ক মোড প্রপার্টি সেটার
            def set_dark_mode(self, enabled):
                self.setProperty("darkMode", enabled)
                self.style().unpolish(self)
                self.style().polish(self)