"""Simple widget to render a single search result with title, meta and snippet."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class ResultItemWidget(QWidget):
    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result
        self.setProperty("darkMode", False)  # ডার্ক মোড প্রপার্টি
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel(self.result.get("doc_title", self.result.get("doc_id", "Unknown")))
        title.setObjectName("resultTitle")
        layout.addWidget(title)
        meta = QLabel(f"Score: {self.result.get('score', 0):.3f} | Page: {self.result.get('page', '')}")
        meta.setObjectName("resultMeta")
        layout.addWidget(meta)
        # Highlight search terms in snippet
        snippet_text = self.result.get("snippet", self.result.get("text", ""))
        search_term = self.result.get("search_term", "")
        if search_term:
            import re
            pattern = re.escape(search_term)
            snippet_html = re.sub(f"({pattern})", r'<span style="background:yellow">\\1</span>', snippet_text, flags=re.IGNORECASE)
        else:
            snippet_html = snippet_text
        snippet = QLabel()
        snippet.setTextFormat(Qt.TextFormat.RichText)
        snippet.setText(snippet_html)
        snippet.setWordWrap(True)
        snippet.setObjectName("resultSnippet")
        layout.addWidget(snippet)
        # View in PDF button
        from PyQt6.QtWidgets import QPushButton
        view_btn = QPushButton("View in PDF")
        view_btn.clicked.connect(self.open_pdf_at_page)
        layout.addWidget(view_btn)
        self.setLayout(layout)

    def open_pdf_at_page(self):
        """Open the PDF at the result's page (requires external handler)."""
        pdf_path = self.result.get("pdf_path")
        page = self.result.get("page")
        if pdf_path and page:
            import os
            import platform
            # Try to open PDF at page (platform-specific)
            try:
                if platform.system() == "Windows":
                    os.startfile(pdf_path)
                elif platform.system() == "Darwin":
                    os.system(f"open '{pdf_path}'")
                else:
                    os.system(f"xdg-open '{pdf_path}'")
            except Exception:
                pass
        # Optionally emit a signal for app-level handling

    def set_dark_mode(self, enabled):
        self.setProperty("darkMode", enabled)
        self.style().unpolish(self)
        self.style().polish(self)
        # Propagate to children
        for child in self.findChildren(QWidget):
            try:
                child.setProperty("darkMode", enabled)
                child.style().unpolish(child)
                child.style().polish(child)
            except Exception:
                pass

    # ডার্ক মোড প্রপার্টি সেটার
    def set_dark_mode(self, enabled):
        self.setProperty("darkMode", enabled)
        self.style().unpolish(self)
        self.style().polish(self)