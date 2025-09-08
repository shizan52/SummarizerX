import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import re
import json
import hashlib
import gc
import unicodedata

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Chunker:
    """
    Splits text into chunks with multilang support.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = r'\n\s*\n|[।\.؟!]\s+',  # Multilang punctuation
        min_chunk_size: int = 50
    ):
        self.chunk_size = max(100, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size // 2))
        self.separator = re.compile(separator)
        self.min_chunk_size = max(10, int(min_chunk_size))
        
        logger.info(
            f"Chunker initialized: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, min_chunk_size={self.min_chunk_size}"
        )
    
    def _normalize_text(self, text: str) -> str:
        return unicodedata.normalize('NFKC', text)

    def _count_words(self, text: str) -> int:
        return len(re.findall(r'\w+|\S', text))  # Multilang word count (includes non-Latin)
    
    def chunk_text(self, text: str, doc_id: str, page_number: int = 1) -> List[Dict[str, Any]]:
        text = self._normalize_text(text)
        if not text.strip():
            logger.warning("Empty text provided")
            return []
        
        try:
            segments = self.separator.split(text)
            segments = [seg.strip() for seg in segments if seg.strip()]
            
            chunks = []
            current_chunk = []
            current_word_count = 0
            current_offset = 0
            batch_size = 100
            
            for i in range(0, len(segments), batch_size):
                batch_segments = segments[i:i + batch_size]
                for segment in batch_segments:
                    segment_words = self._count_words(segment)
                try:
                    # Updated regex: supports Bengali (।, ॥), English (.!?) and Arabic (؟)
                    sentence_splitter = re.compile(r'(?<=[.!?।॥؟])\s+')
                    segments = self.separator.split(text)
                    segments = [seg.strip() for seg in segments if seg.strip()]
                    chunks = []
                    current_chunk = []
                    current_word_count = 0
                    current_offset = 0
                    batch_size = 100
                    for i in range(0, len(segments), batch_size):
                        batch_segments = segments[i:i + batch_size]
                        for segment in batch_segments:
                            segment_words = self._count_words(segment)
                            segment_length = len(segment)
                            if segment_words > self.chunk_size:
                                sentences = sentence_splitter.split(segment)
                                for sentence in sentences:
                                    sentence = sentence.strip()
                                    if not sentence:
                                        continue
                                    sentence_words = self._count_words(sentence)
                                    if current_word_count + sentence_words > self.chunk_size and current_chunk:
                                        chunk_text = " ".join(current_chunk)
                                        if self._count_words(chunk_text) >= self.min_chunk_size:
                                            # Hash includes page_number for less collision
                                            chunk_id = hashlib.sha256((chunk_text + f"|{page_number}").encode("utf-8")).hexdigest()
                                            chunks.append({
                                                "id": chunk_id,
                                                "doc_id": doc_id,
                                                "text": chunk_text,
                                                "page": page_number,
                                                "start_offset": current_offset,
                                                "end_offset": current_offset + len(chunk_text)
                                            })
                                        current_chunk = []
                                        current_word_count = 0
                                        current_offset += len(chunk_text) + 1
                                    current_chunk.append(sentence)
                                    current_word_count += sentence_words
                            else:
                                if current_word_count + segment_words > self.chunk_size and current_chunk:
                                    chunk_text = " ".join(current_chunk)
                                    if self._count_words(chunk_text) >= self.min_chunk_size:
                                        chunk_id = hashlib.sha256((chunk_text + f"|{page_number}").encode("utf-8")).hexdigest()
                                        chunks.append({
                                            "id": chunk_id,
                                            "doc_id": doc_id,
                                            "text": chunk_text,
                                            "page": page_number,
                                            "start_offset": current_offset,
                                            "end_offset": current_offset + len(chunk_text)
                                        })
                                    current_chunk = []
                                    current_word_count = 0
                                    current_offset += len(chunk_text) + 1
                                current_chunk.append(segment)
                                current_word_count += segment_words
                            current_offset += segment_length + 1
                        gc.collect()
                    # Finalize last chunk
                    if current_chunk:
                        chunk_text = " ".join(current_chunk)
                        if self._count_words(chunk_text) >= self.min_chunk_size:
                            chunk_id = hashlib.sha256((chunk_text + f"|{page_number}").encode("utf-8")).hexdigest()
                            chunks.append({
                                "id": chunk_id,
                                "doc_id": doc_id,
                                "text": chunk_text,
                                "page": page_number,
                                "start_offset": current_offset,
                                "end_offset": current_offset + len(chunk_text)
                            })
                    # Handle overlap
                    if self.chunk_overlap > 0 and len(chunks) > 1:
                        overlapped_chunks = []
                        for i in range(len(chunks)):
                            chunk = chunks[i]
                            chunk_text = chunk["text"]
                            if i < len(chunks) - 1:
                                next_chunk = chunks[i + 1]
                                words = re.findall(r'\w+|\S', chunk_text)
                                if len(words) > self.chunk_overlap:
                                    overlap_text = " ".join(words[-self.chunk_overlap:]) + " " + next_chunk["text"]
                                    overlap_id = hashlib.sha256((overlap_text + f"|{chunk['page']}").encode("utf-8")).hexdigest()
                                    overlapped_chunks.append({
                                        "id": overlap_id,
                                        "doc_id": doc_id,
                                        "text": overlap_text,
                                        "page": chunk["page"],
                                        "start_offset": chunk["start_offset"],
                                        "end_offset": next_chunk["end_offset"]
                                    })
                            overlapped_chunks.append(chunk)
                        chunks = overlapped_chunks
                    logger.info(f"Created {len(chunks)} chunks for doc_id={doc_id}")
                    return chunks
                except Exception as e:
                    logger.error(f"Chunking failed for doc_id={doc_id}: {str(e)}")
                    return []
            return chunks
        
        except Exception as e:
            logger.error(f"Document chunking failed for doc_id={doc_id}: {str(e)}")
            return []

    def chunk_document(self, input_data: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
            try:
                chunks = []
                offset_map = {}
                for segment in input_data:
                    text = segment.get("text", "")
                    page_number = segment.get("page_number") or segment.get("paragraph_number", 1)
                    if not text.strip():
                        continue
                    # Track offset for each page
                    page_offset = offset_map.get(page_number, 0)
                    segment_chunks = self.chunk_text(text, doc_id, page_number)
                    # Adjust offsets to be page-aware
                    for chunk in segment_chunks:
                        chunk["start_offset"] += page_offset
                        chunk["end_offset"] += page_offset
                    offset_map[page_number] = page_offset + len(text) + 1
                    chunks.extend(segment_chunks)
                logger.info(f"Chunked document {doc_id} into {len(chunks)} chunks")
                return chunks
            except Exception as e:
                logger.error(f"Document chunking failed for doc_id={doc_id}: {str(e)}")
                return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Chunker for Document Processing")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument("--output", "-o", help="Output file path for chunks (JSON)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target words per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Words to overlap between chunks")
    parser.add_argument("--min-chunk-size", type=int, default=50, help="Minimum words per chunk")
    args = parser.parse_args()
    
    try:
        chunker = Chunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_size=args.min_chunk_size
        )
        
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        # Read input file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Create mock document ID
        doc_id = hashlib.sha256(str(input_path).encode("utf-8")).hexdigest()
        
        # Chunk text
        chunks = chunker.chunk_text(text, doc_id=doc_id)
        
        # Print results
        print(f"\n--- Chunking Results ---")
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(f"ID: {chunk['id']}")
            print(f"Doc ID: {chunk['doc_id']}")
            print(f"Page: {chunk['page']}")
            print(f"Text: {chunk['text'][:100] + '...' if len(chunk['text']) > 100 else chunk['text']}")
            print(f"Offsets: {chunk['start_offset']} - {chunk['end_offset']}")
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Chunks saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        sys.exit(1)