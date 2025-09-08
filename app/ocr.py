
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import sys
import gc
import unicodedata
from cryptography.fernet import Fernet
import shutil

# Ensure project root on sys.path
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
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available")


class OCRProcessor:
    """Handles OCR with privacy and offline support."""
    
    def __init__(self, language: str = "ben+eng+hin", dpi: int = 200, temp_dir: Optional[str] = None, preprocess: bool = False):
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract required for OCR")
        
        self.language = language
        self.dpi = max(100, min(int(dpi), 300))
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.cipher = Fernet(Fernet.generate_key())
        self.preprocess = preprocess
        if self.temp_dir and not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        # Ensure Tesseract data is local
        self._ensure_tesseract_data()
        logger.info(f"OCRProcessor initialized: language={self.language}, dpi={self.dpi}, preprocess={self.preprocess}")

    def _preprocess_image(self, img):
        """Optional: Preprocess image with OpenCV for better OCR accuracy."""
        try:
            import cv2
            import numpy as np
            img_np = np.array(img)
            if len(img_np.shape) == 3:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np
            # Binarization
            img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 31, 11)
            # Denoise
            img_bin = cv2.fastNlMeansDenoising(img_bin, None, 30, 7, 21)
            return Image.fromarray(img_bin)
        except Exception as e:
            logger.warning(f"OpenCV preprocessing failed: {e}")
            return img

    def _ensure_tesseract_data(self):
        """Ensure Tesseract language data is available locally."""
        try:
            pytesseract.get_languages()
            logger.info("Tesseract language data available locally")
        except Exception as e:
            logger.error(f"Tesseract language data not found: {str(e)}")
            raise

    def _normalize_text(self, text: str) -> str:
        return unicodedata.normalize('NFKC', text).strip()

    def _encrypt_text(self, text: str) -> bytes:
        return self.cipher.encrypt(text.encode())

    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Validate image file
        try:
            with Image.open(image_path) as img:
                img.verify()  # Check if image is corrupted
            with Image.open(image_path) as img:  # Re-open after verify
                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Invalid image dimensions: {img.size}")
                if self.preprocess:
                    img = self._preprocess_image(img)
                text = pytesseract.image_to_string(img, lang=self.language)
            
            normalized_text = self._normalize_text(text)
            encrypted_text = self._encrypt_text(normalized_text)
            logger.info(f"OCR extracted from image: {image_path.name}")
            return {
                "text": normalized_text,  # Return decrypted for use
                "encrypted_text": encrypted_text,
                "method": "tesseract",
                "error": None
            }
        except (OSError, IOError) as e:
            error_msg = f"Invalid or corrupted image file: {str(e)}"
            logger.error(f"Image validation failed for {image_path}: {error_msg}")
            return {
                "text": "",
                "encrypted_text": b"",
                "method": "tesseract",
                "error": error_msg
            }
        except Exception as e:
            logger.error(f"OCR failed for image {image_path}: {str(e)}")
            return {
                "text": "",
                "encrypted_text": b"",
                "method": "tesseract",
                "error": str(e)
            }

    def extract_from_pdf_page(self, pdf_path: str, page_num: int, progress_callback=None) -> Dict[str, Any]:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF required for PDF OCR")
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        temp_file = None
        try:
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                raise ValueError(f"Invalid page {page_num}")
            page = doc.load_page(page_num - 1)
            mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
            pix = page.get_pixmap(matrix=mat)
            with tempfile.NamedTemporaryFile(
                suffix=".png", dir=str(self.temp_dir) if self.temp_dir else None, delete=False
            ) as tmp:
                temp_file = tmp.name
                pix.save(temp_file)
            with Image.open(temp_file) as img:
                if self.preprocess:
                    img = self._preprocess_image(img)
                text = pytesseract.image_to_string(img, lang=self.language)
            normalized_text = self._normalize_text(text)
            encrypted_text = self._encrypt_text(normalized_text)
            return {
                "page_number": page_num,
                "text": normalized_text,
                "encrypted_text": encrypted_text,
                "method": "pymupdf+tesseract",
                "is_scanned": True,
                "error": None if normalized_text else "OCR failed"
            }
        except Exception as e:
            logger.error(f"OCR failed for PDF {pdf_path} page {page_num}: {str(e)}")
            return {
                "page_number": page_num,
                "text": "",
                "encrypted_text": b"",
                "method": "pymupdf+tesseract",
                "is_scanned": True,
                "error": str(e)
            }
        finally:
            try:
                doc.close()
            except Exception:
                pass
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            gc.collect()
        if progress_callback:
            try:
                progress_callback(page_num)
            except Exception:
                pass

    def extract_from_pdf(self, pdf_path: str, max_pages: Optional[int] = None, progress_callback=None) -> List[Dict[str, Any]]:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF required for PDF OCR")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Starting OCR for PDF: {pdf_path.name}")
        results = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            page_range = range(total_pages)
            if max_pages is not None and max_pages > 0:
                page_range = range(min(total_pages, max_pages))
            batch_size = 5
            for i in range(0, len(page_range), batch_size):
                batch_pages = page_range[i:i + batch_size]
                for page_num in batch_pages:
                    result = self.extract_from_pdf_page(pdf_path, page_num + 1, progress_callback=progress_callback)
                    results.append(result)
                    if progress_callback:
                        try:
                            progress_callback(page_num + 1)
                        except Exception:
                            pass
                gc.collect()
            doc.close()
            logger.info(f"Completed OCR for {len(results)} pages in {pdf_path.name}")
            return results
        except Exception as e:
            logger.error(f"OCR extraction failed for PDF {pdf_path}: {str(e)}")
            raise

    def test_offline_ocr(self, image_path: str = "test.png") -> Dict[str, Any]:
        """Test OCR in offline mode."""
        try:
            result = self.extract_from_image(image_path)
            logger.info("Offline OCR test successful")
            return result
        except Exception as e:
            logger.error(f"Offline OCR test failed: {str(e)}")
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR Processor for Images and PDFs")
    parser.add_argument("input_path", help="Path to image or PDF file")
    parser.add_argument("--language", default="ben+eng+hin", help="Tesseract language code")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF-to-image")
    parser.add_argument("--max-pages", type=int, help="Maximum PDF pages to process")
    parser.add_argument("--output", "-o", help="Output text file path")
    args = parser.parse_args()
    
    try:
        processor = OCRProcessor(language=args.language, dpi=args.dpi)
        input_path = Path(args.input_path)
        
        if input_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff"):
            result = processor.extract_from_image(input_path)
            logger.info(f"Image OCR Result: {result['text'][:50]}...")
            if result["error"]:
                logger.error(f"Error: {result['error']}")
        
        elif input_path.suffix.lower() == ".pdf":
            results = processor.extract_from_pdf(input_path, max_pages=args.max_pages)
            logger.info(f"Extracted {len(results)} pages from {input_path}")
            for page in results:
                logger.info(f"Page {page['page_number']}: {page['text'][:50]}...")
                if page["error"]:
                    logger.error(f"Error: {page['error']}")
        
        else:
            logger.error(f"Unsupported file type: {input_path.suffix}")
            sys.exit(1)
        
        # Save to file if specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                if input_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff"):
                    f.write(result["text"])
                else:
                    for page in results:
                        f.write(f"\n--- Page {page['page_number']} ---\n")
                        f.write(page["text"])
            logger.info(f"Text saved to {output_path}")
    
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        sys.exit(1)