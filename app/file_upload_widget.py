from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QListWidget, QFileDialog, QMessageBox
from PyQt6.QtCore import Qt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FileUploadWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setAcceptDrops(True)

    def init_ui(self):
        """Initialize UI components."""
        self.layout = QVBoxLayout()
        
        self.upload_button = QPushButton("Upload Files")
        self.upload_button.clicked.connect(self.upload_files)
        
        self.file_list = QListWidget()
        
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.file_list)
        self.setLayout(self.layout)

    def upload_files(self):
        """Open file dialog to upload files."""
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select Files", "", "Documents (*.pdf *.docx *.txt)"
            )
            if not files:
                return  # User cancelled
            
            for file in files:
                if not os.path.exists(file):
                    QMessageBox.warning(self, "File Not Found", f"File does not exist: {file}")
                    continue
                file_ext = file.lower().split('.')[-1]
                if file_ext not in ['pdf', 'docx', 'txt']:
                    QMessageBox.warning(self, "Unsupported File", f"Unsupported file type: {file}")
                    continue
                self.file_list.addItem(file)
            logger.info(f"Uploaded {len(files)} files")
        except Exception as e:
            logger.error(f"Failed to upload files: {e}")
            QMessageBox.critical(self, "Upload Error", f"Failed to upload files: {str(e)}")

    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle drop event."""
        try:
            urls = event.mimeData().urls()
            if not urls:
                return
            
            for url in urls:
                file_path = url.toLocalFile()
                if not file_path:
                    continue
                    
                if not os.path.exists(file_path):
                    QMessageBox.warning(self, "File Not Found", f"File does not exist: {file_path}")
                    continue
                    
                file_ext = file_path.lower().split('.')[-1]
                if file_ext not in ['pdf', 'docx', 'txt']:
                    QMessageBox.warning(self, "Unsupported File", f"Unsupported file type: {file_path}")
                    continue
                    
                self.file_list.addItem(file_path)
            logger.info(f"Dropped {self.file_list.count()} files")
        except Exception as e:
            logger.error(f"Failed to process drop event: {e}")
            QMessageBox.critical(self, "Drop Error", f"Failed to process dropped files: {str(e)}")

    def get_uploaded_files(self) -> list:
        """Return list of uploaded file paths."""
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    widget = FileUploadWidget()
    widget.show()
    sys.exit(app.exec())
import logging
from pathlib import Path
from typing import Callable
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QProgressBar, QFileDialog
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
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


class FileUploadWidget(QWidget):
    """
    A widget for uploading PDF/DOCX files via drag-and-drop or file dialog.
    Emits signals for file processing and progress updates.
    """
    file_selected = pyqtSignal(str)  # Emits file path
    progress_updated = pyqtSignal(str)  # Emits progress message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setProperty("darkMode", False)  # ডার্ক মোড প্রপার্টি ইনিশিয়ালাইজ
        self.init_ui()
        logger.info("FileUploadWidget initialized")

    def init_ui(self):
        """Initialize the widget layout and components."""
        layout = QVBoxLayout(self)

        self.label = QLabel("Drag and drop PDF/DOCX files here or click to browse")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("uploadLabel")
        layout.addWidget(self.label)

        self.browse_button = QPushButton("Browse Files")
        self.browse_button.clicked.connect(self.browse_files)
        self.browse_button.setObjectName("browseButton")
        layout.addWidget(self.browse_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setObjectName("progressBar")
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for file drops."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".pdf", ".docx")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle file drop events."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".pdf", ".docx")):
                self.file_selected.emit(file_path)
                self.status_label.setText(f"Selected: {Path(file_path).name}")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                logger.info(f"Dropped file: {file_path}")
            else:
                self.status_label.setText("Unsupported file type")
                logger.warning(f"Unsupported file dropped: {file_path}")

    def browse_files(self):
        """Open file dialog for selecting files."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "Documents (*.pdf *.docx)"
        )
        if file_path:
            self.file_selected.emit(file_path)
            self.status_label.setText(f"Selected: {Path(file_path).name}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            logger.info(f"Browsed file: {file_path}")

    def update_progress(self, message: str):
        """Update progress bar and status label."""
        self.progress_updated.emit(message)
        self.status_label.setText(message)
        # Simulate progress (actual progress handled by IngestionWorker)
        current_value = self.progress_bar.value()
        if current_value < 90:
            self.progress_bar.setValue(current_value + 10)
        else:
            self.progress_bar.setValue(100)

    # ডার্ক মোড প্রপার্টি সেটার (প্যারেন্ট থেকে কল করুন যদি দরকার)
    def set_dark_mode(self, enabled):
        self.setProperty("darkMode", enabled)
        self.style().unpolish(self)
        self.style().polish(self)