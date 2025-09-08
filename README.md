Instant Document Summarizer & Search
A privacy-first, offline-capable desktop application for processing bulk documents (PDF, DOCX, TXT) to generate extractive summaries, perform hybrid search (keyword + semantic), and cluster/tag documents based on topics. Designed to run efficiently on CPU with minimal resource usage, ensuring fast performance and complete data privacy.

Features

Document Processing: Extract text from PDF, DOCX, and TXT files, with OCR support for scanned PDFs using pytesseract.
Extractive Summarization: Generate concise summaries using LexRank or TextRank algorithms, with configurable compression ratios (default: 20%).
Hybrid Search: Combines keyword-based search (SQLite FTS5) with semantic search (sentence-transformers) for precise and context-aware results.
Topic Clustering & Tagging: Automatically cluster documents or chunks into topics using BERTopic or KMeans, with multilingual support (Bengali, English, Hindi, etc.).
Privacy-Focused: All data processing is performed locally, with no reliance on cloud APIs.
Modern UI: Built with PyQt6, featuring a responsive and visually appealing interface with dark mode support and Bengali font fallbacks.
Export & Persistence: Export summaries and search results to JSON/CSV, with encrypted storage using cryptography.fernet.
Offline Capability: Runs entirely offline, with pre-downloaded models and local indexing (FAISS/HNSWlib, SQLite).


Requirements

Python: 3.8 or higher
Operating System: Windows, macOS, or Linux
Dependencies (listed in requirements.txt):
pdfminer.six or pdfplumber (PDF extraction)
python-docx (DOCX extraction)
pytesseract (OCR for scanned PDFs)
sentence-transformers (semantic embeddings)
faiss-cpu or hnswlib (vector indexing)
sqlite3 (keyword search, metadata storage)
sumy (extractive summarization)
bertopic, umap-learn, hdbscan (topic clustering, optional)
pyqt6 or pyside6 (GUI)
nltk (multilingual tokenization)
cryptography (encryption)


External Tools:
Tesseract OCR (tesseract executable must be in PATH)


Disk Space: At least 1GB free (5GB recommended for large corpora)


Installation

Clone the Repository:
git clone https://github.com/shizan52/SummarizerX
cd instant-summarizer


Set Up a Virtual Environment:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Install Tesseract OCR:

Windows: Download and install from Tesseract OCR.
Linux: sudo apt-get install tesseract-ocr
macOS: brew install tesseract


Download NLTK Data:
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"


Download Pre-trained Models:Run the model download script to cache the embedding model locally:
python scripts/download_model.py


Run Startup Checks:Verify all dependencies and configurations:
python app/startup_checks.py




Usage

Launch the Application:
python -m app.main_window


Upload Documents:

Use the file upload widget to add PDF, DOCX, or TXT files.
The app processes files in the background, extracting text, generating embeddings, and indexing them.


Search:

Enter a query in the search bar.
Results combine keyword matches (exact terms) and semantic matches (paraphrased meaning).
Click "View in PDF" to open the source document at the relevant page.


Summarize:

Select a document or chunk to generate a concise summary.
Summaries are extractive, selecting the most informative sentences.


Cluster & Tag:

Run topic modeling to group similar content into topics.
View topic labels in the results panel.


Export Results:

Export summaries or search results to JSON or CSV files via the export button.
Exported files are saved in plain text (decrypted) for easy access.




Project Structure
instant-summarizer/
├── .venv/                           # Python virtual environment
├── app/                             # Main application code
│   ├── __init__.py
│   ├── App_logo.png                 # Application logo
│   ├── app_style.qss                # Qt stylesheet
│   ├── chunker.py                   # Text chunking
│   ├── config.py                    # Configuration management
│   ├── docx_extractor.py            # DOCX processing
│   ├── embeddings/                  # Embedding generation
│   │   ├── embedder.py
│   ├── exporter.py                  # Export functionality
│   ├── extractive.py                # Extractive summarization
│   ├── file_upload_widget.py        # File upload GUI
│   ├── hybrid_search.py             # Hybrid search implementation
│   ├── indexer/                     # Database and vector indexing
│   │   ├── sqlite_store.py
│   │   ├── vector_store.py
│   ├── main_window.py               # Main GUI window
│   ├── normalizer.py                # Text normalization
│   ├── ocr.py                       # OCR processing
│   ├── pdf_extractor.py             # PDF text extraction
│   ├── result_item.py               # Search result item
│   ├── result_widget.py             # Search results display
│   ├── startup_checks.py            # Startup validation
│   ├── styles.qss                   # Qt styles
│   └── topic_model.py               # Topic modeling and clustering
├── data/                            # Data storage
│   ├── embeddings/                  # Stored embeddings
│   └── indexes/                     # Vector indexes
├── requirements.txt                 # Python dependencies
├── scripts/                         # Utility scripts
│   └── download_model.py            # Model download script
└── tools/                           # Development tools


Configuration
The application is configured via app/config.py. Key settings include:

embedding_model: Default is paraphrase-multilingual-MiniLM-L12-v2 for multilingual support.
embeddings_dir: Path to store cached embeddings (data/embeddings).
db_path: SQLite database path (data/indexes/db.sqlite).
index_dir: Vector index storage (data/indexes).
search_alpha, search_beta, search_gamma: Weights for semantic, lexical, and recency scores in hybrid search (default: 0.7, 0.25, 0.05).

To customize, edit config.py or override via environment variables.

Technical Details

Text Extraction: Uses pdfminer.six/pdfplumber for PDFs, python-docx for DOCX, and pytesseract for OCR.
Semantic Embeddings: sentence-transformers with paraphrase-multilingual-MiniLM-L12-v2, optimized for CPU.
Vector Indexing: FAISS (IVF-PQ or Flat) or HNSWlib for efficient semantic search.
Keyword Search: SQLite with FTS5 for fast, full-text search.
Summarization: Extractive using sumy (LexRank/TextRank) with multilingual tokenization.
Clustering: BERTopic (UMAP + HDBSCAN) or KMeans for topic modeling.
UI: PyQt6 with responsive design, dark mode, and Bengali font support (Noto Sans Bengali, SolaimanLipi).


Troubleshooting

Startup Check Failures:

Run python app/startup_checks.py to diagnose issues.
Ensure Tesseract is in PATH and NLTK data (punkt, stopwords) is downloaded.
Install missing dependencies: pip install faiss-cpu hnswlib sentence-transformers sumy bertopic umap-learn hdbscan.


Export Issues:

If exported files fail to open, check file encoding (UTF-8) and ensure no other process is locking the file.
Verify cryptography is installed for decryption (pip install cryptography).


Performance:

For large corpora, increase cache_size in sqlite_store.py or use a larger map_size in embedder.py.
Ensure at least 5GB free disk space for embeddings and indexes.




Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.


License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Built with inspiration from modern desktop applications and privacy-first principles.
Thanks to the open-source community for libraries like sentence-transformers, faiss, sumy, and bertopic.
