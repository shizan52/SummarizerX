import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import re
import unicodedata

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Normalizer:
    """
    Normalizes extracted text with multilang support.
    """
    
    def __init__(
        self,
        remove_punctuation: bool = True,
        normalize_unicode: bool = True,
        lowercase: bool = True,
        max_length: Optional[int] = None,
        lang: str = 'multi'  # 'ben', 'hin', 'eng', 'multi'
    ):
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
        self.max_length = max_length
        self.lang = lang
        self.punct_regex = re.compile(r'[^\w\s\u0980-\u09FF\u0900-\u097F]')  # Bengali/Hindi/English punct
        
        logger.info(
            f"Normalizer initialized: remove_punctuation={remove_punctuation}, "
            f"normalize_unicode={normalize_unicode}, lowercase={lowercase}, "
            f"max_length={max_length}, lang={lang}"
        )
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize a single text string.
        - Unicode normalization (NFKC)
        - Bengali-specific normalization (optional, if lang is 'ben' or 'multi')
        - Punctuation removal (optional)
        - Lowercasing (optional)
        - Truncation (optional)
        """
        try:
            if self.normalize_unicode:
                text = unicodedata.normalize('NFKC', text)
            # Bengali normalization step
            if self.lang in ('ben', 'multi'):
                text = self._normalize_bengali(text)
            if self.remove_punctuation:
                # Note: For search, consider keeping punctuation for better keyword match
                text = self.punct_regex.sub('', text)
            if self.lowercase:
                text = text.lower()
            if self.max_length is not None:
                text = text[:self.max_length]
            return text.strip()
        except Exception as e:
            logger.error(f"Normalization failed: {str(e)}")
            return text

    def _normalize_bengali(self, text: str) -> str:
        """
        Bengali-specific normalization: fix common ligature, nukta, reph, and ZWNJ/ZWJ issues.
        """
        # Remove Zero Width Non-Joiner/Joiner
        text = text.replace('\u200c', '').replace('\u200d', '')
        # Normalize nukta (dot below)
        text = text.replace('\u09bc', '')
        # Normalize reph (রেফ) - move RA+HALANT to start if needed (simplified)
        # This is a placeholder for more advanced Bengali normalization
        # Add more rules as needed for your corpus
        return text

    def clean_for_ingestion(self, text: str) -> str:
        """
        Clean text for ingestion: Unicode, Bengali, and whitespace normalization.
        Use this in chunker/extractor before chunking/indexing.
        """
        text = self.normalize_text(text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize a list of chunks."""
        try:
            normalized_chunks = []
            for chunk in chunks:
                original_text = chunk.get("text", "")
                normalized_text = self.normalize_text(original_text)
                
                if normalized_text:  
                    normalized_chunk = chunk.copy()
                    normalized_chunk["text"] = normalized_text
                    normalized_chunks.append(normalized_chunk)
            
            logger.info(f"Normalized {len(normalized_chunks)} chunks")
            return normalized_chunks
        
        except Exception as e:
            logger.error(f"Chunk normalization failed: {str(e)}")
            return chunks

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Normalizer for Document Processing")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument("--output", "-o", help="Output file path for normalized text")
    parser.add_argument("--remove-punctuation", action="store_true", help="Remove punctuation")
    parser.add_argument("--no-unicode", action="store_false", dest="normalize_unicode", help="Disable Unicode normalization")
    parser.add_argument("--lowercase", action="store_true", help="Convert to lowercase")
    parser.add_argument("--max-length", type=int, help="Truncate text to max length")
    parser.add_argument("--lang", default="multi", help="Language: ben, hin, eng, multi")
    args = parser.parse_args()
    
    try:
        normalizer = Normalizer(
            remove_punctuation=args.remove_punctuation,
            normalize_unicode=args.normalize_unicode,
            lowercase=args.lowercase,
            max_length=args.max_length,
            lang=args.lang
        )
        
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        # Read input file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Normalize as a single chunk with mock metadata
        chunk = {
            "text": text,
            "page_number": 1,
            "start_offset": 0,
            "end_offset": len(text)
        }
        normalized_chunks = normalizer.normalize_chunks([chunk])
        
        # Print result
        if normalized_chunks:
            normalized_text = normalized_chunks[0]["text"]
            print(f"\n--- Normalized Text ---")
            print(normalized_text[:500] + "..." if len(normalized_text) > 500 else normalized_text)
        
        # Save to output file if specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in normalized_chunks:
                    f.write(chunk["text"])
            logger.info(f"Normalized text saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Normalization failed: {str(e)}")
        sys.exit(1)