import logging
from typing import List, Dict, Any, Optional
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractiveSummarizer:
    """Extractive summarization with multilingual support."""

    def __init__(self, language: str = 'bengali', summarizer_type: str = 'lexrank', compression_ratio: float = 0.2):
        self.language = language
        self.summarizer_type = summarizer_type
        self.compression_ratio = max(0.1, min(1.0, compression_ratio))
        self.tokenizer = Tokenizer(self.language)
        if self.summarizer_type == 'lexrank':
            self.summarizer = LexRankSummarizer()
        else:
            self.summarizer = TextRankSummarizer()

    def summarize(self, text: str, max_sentences: Optional[int] = None) -> str:
        """Generate extractive summary."""
        if not text:
            return ""
        try:
            parser = PlaintextParser.from_string(text, self.tokenizer)
            sentence_count = len(parser.document.sentences)
            summary_sentences = int(sentence_count * self.compression_ratio)
            if max_sentences:
                summary_sentences = min(max_sentences, summary_sentences)
            summary = self.summarizer(parser.document, summary_sentences)
            return ' '.join(str(s) for s in summary)
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return text[:500] + "..."  # Fallback truncate

    def summarize_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize multiple chunks."""
        summaries = {}
        for chunk in chunks:
            chunk_id = chunk.get('id')
            text = chunk.get('text', '')
            summaries[chunk_id] = self.summarize(text)
        return summaries

if __name__ == "__main__":
    summarizer = ExtractiveSummarizer(language='bengali')
    test_text = "এটি একটি দীর্ঘ টেক্সট যা সংক্ষিপ্ত করতে হবে। এখানে অনেক তথ্য আছে।"
    print(summarizer.summarize(test_text))
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExtractiveSummarizer:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialize extractive summarizer with a multilingual sentence-transformers model."""
        try:
            self.model = SentenceTransformer(model_name, cache_folder='./models')
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise

        # TF-IDF is lightweight and CPU-friendly
        try:
            self.tfidf = TfidfVectorizer(max_features=5000)
        except Exception as e:
            logger.error(f"Failed to initialize TF-IDF vectorizer: {e}")
            raise

    def summarize(self, sentences: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """Generate extractive summary from a list of sentences.

        Returns a list of dicts: {"text": str, "score": float, "index": int}
        """
        try:
            if not sentences:
                return []

            # TF-IDF scoring (sentence-level)
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=1)

            # Semantic scoring via cosine similarity to centroid
            embeddings = self.model.encode(sentences, batch_size=16, show_progress_bar=False)
            centroid = np.mean(embeddings, axis=0)
            # avoid division by zero
            emb_norms = np.linalg.norm(embeddings, axis=1)
            centroid_norm = np.linalg.norm(centroid) if np.linalg.norm(centroid) > 0 else 1.0
            semantic_scores = np.dot(embeddings, centroid) / (emb_norms * centroid_norm + 1e-12)

            # Normalize scores
            tfidf_scores = (tfidf_scores - np.min(tfidf_scores))
            if np.max(tfidf_scores) > 0:
                tfidf_scores = tfidf_scores / (np.max(tfidf_scores) + 1e-12)

            if np.max(semantic_scores) > 0:
                semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-12)

            combined_scores = 0.6 * tfidf_scores + 0.4 * semantic_scores

            top_n = min(top_n, len(sentences))
            top_indices = np.argsort(combined_scores)[-top_n:]

            summary = [
                {"text": sentences[i], "score": float(combined_scores[i]), "index": int(i)}
                for i in sorted(top_indices)
            ]
            logger.info(f"Generated summary with {len(summary)} sentences")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return []


if __name__ == "__main__":
    test_sentences = [
        "এটি একটি পরীক্ষামূলক বাক্য।",
        "আমরা ডকুমেন্ট সামারাইজেশন নিয়ে কাজ করছি।",
        "এই অ্যাপটি বাংলা টেক্সট সাপোর্ট করে।",
        "আমরা অফলাইন মোডে কাজ করি।",
        "প্রাইভেসি আমাদের প্রধান লক্ষ্য।"
    ]
    summarizer = ExtractiveSummarizer()
    print(summarizer.summarize(test_sentences, top_n=3))
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    import nltk
    SUMY_IMPORT_ERROR = None
except Exception as e:
    SUMY_IMPORT_ERROR = e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.indexer.sqlite_store import SQLiteStore
from app.config import Config
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    import nltk
    SUMY_IMPORT_ERROR = None
except Exception as e:
    SUMY_IMPORT_ERROR = e

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.indexer.sqlite_store import SQLiteStore
from app.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

if SUMY_IMPORT_ERROR is None:
    try:
        nltk.download('punkt', quiet=True)
        SUMY_AVAILABLE = True
    except Exception:
        SUMY_AVAILABLE = False
        logger.warning("nltk punkt download failed - Extractive summarization may be limited")
else:
    SUMY_AVAILABLE = False
    logger.warning(f"sumy or nltk not available - Extractive summarization disabled: {SUMY_IMPORT_ERROR}")


class ExtractiveSummarizer:
    """
    Performs extractive summarization using TextRank (via sumy) on document chunks.
    Integrates with sqlite_store.py for chunk retrieval and summary storage.
    """

    def __init__(
        self,
        language: str = "english",
        summary_sentences: int = 2,  # Reduced to 2 for shorter summaries
        sqlite_store: Optional[SQLiteStore] = None,
        db_path: Optional[str] = None
    ):
        """
        Initialize the extractive summarizer.

        Args:
            language: Language for tokenization and stop words (e.g., 'english' or 'bengali')
            summary_sentences: Number of sentences in the summary
            sqlite_store: SQLiteStore instance for chunk retrieval/storage
            db_path: Path to SQLite database (used if sqlite_store is None)
        """
        if not SUMY_AVAILABLE:
            raise RuntimeError("sumy and nltk are required for summarization but not installed")

        self.language = language
        self.summary_sentences = max(1, int(summary_sentences))  # Ensure at least 1 sentence
        self.sqlite_store = sqlite_store

        if self.sqlite_store is None and db_path:
            self.sqlite_store = SQLiteStore(db_path=db_path)

        self.tokenizer = Tokenizer(self.language)
        self.stemmer = Stemmer(self.language)
        self.stop_words = set(get_stop_words(self.language))
        if self.language == "bengali":
            # Add a small set of Bengali stop-words if desired
            self.stop_words.update(["এবং", "যে", "যা", "আমি", "তুমি"])  # extend as needed

        logger.info(f"ExtractiveSummarizer initialized for {self.language} with {self.summary_sentences} sentences")

    def summarize(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize a list of chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' key

        Returns:
            Dictionary with summary and metadata
        """
        if not chunks or not any(chunk.get("text") for chunk in chunks):
            return {"summary": "", "sentence_count": 0, "source_chunk_ids": [], "error": "No text to summarize"}

        try:
            # Join chunk texts but keep some separators to avoid run-ons
            all_text = "\n".join(chunk["text"].strip() for chunk in chunks if chunk.get("text"))
            parser = PlaintextParser.from_string(all_text, Tokenizer(self.language))
            summarizer = TextRankSummarizer(Stemmer(self.language))
            summarizer.stop_words = self.stop_words

            # Request up to summary_sentences sentences from the summarizer
            summary_sentences_iter = summarizer(parser.document, self.summary_sentences)
            # Convert to strings and filter out very short sentences
            significant_sentences = [str(s).strip() for s in summary_sentences_iter if len(str(s).split()) > 3]
            # Limit to the requested number
            final_sentences = significant_sentences[: self.summary_sentences]
            summary = " ".join(final_sentences) if final_sentences else "No significant summary available."

            return {
                "summary": summary,
                "sentence_count": len(final_sentences),
                "source_chunk_ids": [chunk.get("id") for chunk in chunks if chunk.get("id")],
                "error": None,
            }

        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return {"summary": "", "sentence_count": 0, "source_chunk_ids": [], "error": str(e)}

    def summarize_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Summarize all chunks for a given document ID.

        Args:
            doc_id: Document ID to summarize

        Returns:
            Dictionary with summary and metadata
        """
        if not self.sqlite_store:
            return {"summary": "", "sentence_count": 0, "source_chunk_ids": [], "error": "SQLiteStore not initialized"}

        try:
            chunks = self.sqlite_store.get_chunks(doc_id=doc_id)
            return self.summarize(chunks)
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for doc_id={doc_id}: {str(e)}")
            return {"summary": "", "sentence_count": 0, "source_chunk_ids": [], "error": str(e)}

    def __del__(self):
        if self.sqlite_store:
            try:
                self.sqlite_store.close()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extractive Summarizer")
    parser.add_argument("--doc-id", help="Document ID to summarize (requires SQLite DB)")
    parser.add_argument("--text-file", help="Text file to summarize (bypasses SQLite)")
    parser.add_argument("--output", "-o", help="Output file for summary")
    parser.add_argument("--sentences", type=int, default=2, help="Number of sentences in summary")
    parser.add_argument("--language", default="english", help="Language for summarization")
    args = parser.parse_args()

    try:
        config = Config()
        summarizer = ExtractiveSummarizer(
            language=args.language,
            summary_sentences=args.sentences,
            db_path=config.db_path,
        )

        if args.doc_id:
            result = summarizer.summarize_document(args.doc_id)
        elif args.text_file:
            text_path = Path(args.text_file)
            if not text_path.exists():
                logger.error(f"Text file not found: {text_path}")
                sys.exit(1)

            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = [
                {
                    "id": "test_chunk",
                    "doc_id": "test_doc",
                    "text": text,
                    "page": 1,
                    "start_offset": 0,
                    "end_offset": len(text),
                }
            ]
            result = summarizer.summarize(chunks)
        else:
            logger.error("Must provide --doc-id or --text-file")
            sys.exit(1)

        print(f"\n--- Summary ---")
        print(f"Summary: {result.get('summary','')[:500] + '...' if len(result.get('summary','')) > 500 else result.get('summary','')}")
        print(f"Sentence Count: {result.get('sentence_count', 0)}")
        print(f"Source Chunks: {', '.join([str(x) for x in result.get('source_chunk_ids', [])])}")
        if result.get("error"):
            print(f"Error: {result.get('error')}")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.get("summary", ""))
            logger.info(f"Summary saved to {output_path}")

        if summarizer.sqlite_store:
            try:
                summarizer.sqlite_store.close()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        sys.exit(1)