import logging
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import io
import gc
import unicodedata
from cryptography.fernet import Fernet
import shutil

# Make imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")

try:
    import pdfminer
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger.warning("pdfminer not available")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract/PIL not available")

class PDFExtractor:
    """PDF text extraction with privacy and offline support."""
    
    def __init__(self, enable_ocr: bool = True, ocr_language: str = 'ben+eng+hin', max_pages: Optional[int] = None, ocr_dpi: int = 200):
        self.enable_ocr = enable_ocr and TESSERACT_AVAILABLE
        self.ocr_language = ocr_language
        self.max_pages = max_pages
        self.ocr_dpi = ocr_dpi
        self.cipher = Fernet(Fernet.generate_key())
        
        if not PDFPLUMBER_AVAILABLE and not PDFMINER_AVAILABLE:
            logger.error("No PDF extractor available")
        
        logger.info(f"PDF Extractor initialized: OCR={self.enable_ocr}, language={self.ocr_language}")

    def _normalize_text(self, text: str) -> str:
        return unicodedata.normalize('NFKC', text).strip()

    def _encrypt_text(self, text: str) -> bytes:
        return self.cipher.encrypt(text.encode())

    def extract_text(self, file_path: str, progress_callback=None) -> List[Dict[str, Any]]:
        """
        Extract text from PDF or DOCX. Uses OCRProcessor for scanned PDFs.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() == '.docx':
            try:
                from app.docx_extractor import DocxExtractor
                extractor = DocxExtractor()
                return extractor.extract_text(str(file_path))
            except Exception as e:
                logger.error(f"DOCX extraction failed: {str(e)}")
                raise
        # PDF extraction
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    raise ValueError(f"File is not a valid PDF: {file_path}")
        except (OSError, IOError) as e:
            raise RuntimeError(f"Cannot read PDF file {file_path}: {str(e)}")
        logger.info(f"Extracting text from PDF: {file_path.name}")
        if PDFPLUMBER_AVAILABLE:
            try:
                return self._extract_with_pdfplumber(file_path)
            except Exception as e:
                logger.warning(f"pdfplumber failed: {str(e)}")
        if PDFMINER_AVAILABLE:
            try:
                return self._extract_with_pdfminer(file_path)
            except Exception as e:
                logger.warning(f"pdfminer failed: {str(e)}")
        if self.enable_ocr:
            try:
                from app.ocr import OCRProcessor
                ocr = OCRProcessor(language=self.ocr_language, dpi=self.ocr_dpi, preprocess=True)
                return ocr.extract_from_pdf(str(file_path), progress_callback=progress_callback)
            except Exception as e:
                logger.error(f"OCR failed: {str(e)}")
        raise RuntimeError("All extraction methods failed")
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        pages_text = []
        batch_size = 20
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    raise ValueError(f"PDF has no pages: {pdf_path}")
                
                for start in range(0, total_pages, batch_size):
                    batch_pages = pdf.pages[start:start + batch_size]
                    for page_num, page in enumerate(batch_pages, start + 1):
                        try:
                            text = page.extract_text()
                            normalized_text = self._normalize_text(text)
                            encrypted_text = self._encrypt_text(normalized_text)
                            if not normalized_text:
                                if self.enable_ocr:
                                    logger.info(f"Page {page_num} scanned, attempting OCR")
                                    ocr_text = self._ocr_page(page, page_num)
                                    normalized_ocr = self._normalize_text(ocr_text)
                                    encrypted_ocr = self._encrypt_text(normalized_ocr)
                                    pages_text.append({
                                        "page_number": page_num,
                                        "text": normalized_ocr,
                                        "encrypted_text": encrypted_ocr,
                                        "method": "pdfplumber+ocr",
                                        "is_scanned": True
                                    })
                                    continue
                            
                            pages_text.append({
                                "page_number": page_num,
                                "text": normalized_text,
                                "encrypted_text": encrypted_text,
                                "method": "pdfplumber",
                                "is_scanned": False
                            })
                            
                        except Exception as e:
                            logger.error(f"Error extracting page {page_num}: {str(e)}")
                            pages_text.append({
                                "page_number": page_num,
                                "text": "",
                                "encrypted_text": b"",
                                "method": "error",
                                "is_scanned": False,
                                "error": str(e)
                            })
                    gc.collect()
        except Exception as e:
            logger.error(f"Failed to open PDF with pdfplumber: {str(e)}")
            raise RuntimeError(f"PDF extraction failed: {str(e)}")
        
        return pages_text
    
    def _extract_with_pdfminer(self, pdf_path: Path) -> List[Dict[str, Any]]:
        from io import StringIO
        from pdfminer.converter import TextConverter
        from pdfminer.layout import LAParams
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfparser import PDFParser
        
        pages_text = []
        
        try:
            with open(pdf_path, 'rb') as file:
                parser = PDFParser(file)
                doc = PDFDocument(parser)
                rsrcmgr = PDFResourceManager()
                
                for page_num, page in enumerate(PDFPage.create_pages(doc), 1):
                    try:
                        output_string = StringIO()
                        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
                        interpreter = PDFPageInterpreter(rsrcmgr, device)
                        
                        interpreter.process_page(page)
                        text = output_string.getvalue()
                        normalized_text = self._normalize_text(text)
                        encrypted_text = self._encrypt_text(normalized_text)
                        
                        pages_text.append({
                            "page_number": page_num,
                            "text": normalized_text,
                            "encrypted_text": encrypted_text,
                            "method": "pdfminer",
                            "is_scanned": False
                        })
                        
                        output_string.close()
                        device.close()
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {str(e)}")
                        pages_text.append({
                            "page_number": page_num,
                            "text": "",
                            "encrypted_text": b"",
                            "method": "error",
                            "is_scanned": False,
                            "error": str(e)
                        })
                    gc.collect()
        
        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {str(e)}")
            raise
        
        return pages_text
    
    # _extract_with_ocr is now handled by OCRProcessor from app.ocr
    
    # _ocr_page is now handled by OCRProcessor from app.ocr
    
    def is_scanned_pdf(self, pdf_path: str, threshold: float = 0.1) -> bool:
        """
        Heuristic: Returns True if more than `threshold` fraction of first 10 pages are likely scanned (little/no text).
        """
        if not PDFPLUMBER_AVAILABLE:
            return False
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    return False
                scanned_like_pages = 0
                sample_pages = min(10, total_pages)
                for i in range(sample_pages):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    normalized_text = self._normalize_text(text)
                    if not normalized_text or len(normalized_text) < 50:
                        scanned_like_pages += 1
                scanned_ratio = scanned_like_pages / sample_pages
                return scanned_ratio > threshold
        except Exception as e:
            logger.error(f"Error checking scanned PDF: {str(e)}")
            return False

    def test_offline_extraction(self, pdf_path: str = "test.pdf") -> List[Dict[str, Any]]:
        """Test PDF extraction in offline mode."""
        try:
            results = self.extract_text(pdf_path)
            logger.info(f"Offline PDF extraction test returned {len(results)} pages")
            return results
        except Exception as e:
            logger.error(f"Offline PDF extraction test failed: {str(e)}")
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Text Extractor")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output text file path")
    parser.add_argument("--ocr", action="store_true", help="Force OCR extraction")
    args = parser.parse_args()
    
    extractor = PDFExtractor(enable_ocr=True, ocr_language='ben+eng+hin')
    
    try:
        if args.ocr and TESSERACT_AVAILABLE:
            pages = extractor._extract_with_ocr(Path(args.pdf_path))
        else:
            pages = extractor.extract_text(args.pdf_path)
        
        logger.info(f"Extracted {len(pages)} pages from {args.pdf_path}")
        for page in pages:
            logger.info(f"Page {page['page_number']}: {page['text'][:50]}...")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for page in pages:
                    f.write(f"\n--- Page {page['page_number']} ---\n")
                    f.write(page["text"])
            logger.info(f"Text saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Failed to extract text: {str(e)}")