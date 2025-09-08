from typing import List, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


class AbstractiveSummarizer:
    """Light wrapper around HuggingFace summarization pipeline.

    This is optional: if `transformers` is not installed, the class will raise
    a RuntimeError when initialized. For large documents we chunk by
    characters, summarize each chunk, then summarize the concatenation.
    """

    def __init__(self, model_name: str = 'sshleifer/distilbart-cnn-12-6'):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError('transformers not installed; install via `pip install transformers`')
        try:
            # lazy-load tokenizer/model to avoid long startup by default
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.summarizer = pipeline('summarization', model=self.model, tokenizer=self.tokenizer)
            logger.info(f'AbstractiveSummarizer loaded model {model_name}')
        except Exception as e:
            logger.error(f'Failed to load abstractive model: {e}')
            raise

    def _chunk_text(self, text: str, max_chars: int = 1000) -> List[str]:
        # simple character-based chunking on sentence/paragraph boundaries
        if len(text) <= max_chars:
            return [text]
        parts = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            # try to cut at nearest newline or period before end
            cut = text.rfind('\n', start, end)
            if cut <= start:
                cut = text.rfind('. ', start, end)
            if cut <= start:
                cut = end
            parts.append(text[start:cut].strip())
            start = cut
        return [p for p in parts if p]

    def summarize(self, chunks: List[Dict[str, Any]], max_length: int = 150, min_length: int = 40) -> Dict[str, Any]:
        if not chunks:
            return {"summary": "", "error": "No text to summarize"}

        full_text = "\n\n".join(ch.get('text', '') for ch in chunks if ch.get('text'))
        if not full_text.strip():
            return {"summary": "", "error": "No text to summarize"}

        # chunk for large documents
        parts = self._chunk_text(full_text, max_chars=1000)
        summaries = []
        try:
            for part in parts:
                out = self.summarizer(part, max_length=max_length, min_length=min_length, do_sample=False)
                if isinstance(out, list) and out:
                    summaries.append(out[0]['summary_text'])

            # if multiple summaries, combine and compress once more
            if len(summaries) > 1:
                combined = ' '.join(summaries)
                out = self.summarizer(combined, max_length=max_length, min_length=min_length, do_sample=False)
                final = out[0]['summary_text'] if isinstance(out, list) and out else combined
            else:
                final = summaries[0]

            return {"summary": final, "error": None}
        except Exception as e:
            logger.error(f'Abstractive summarization failed: {e}')
            return {"summary": "", "error": str(e)}
