from sentence_transformers import SentenceTransformer
from pathlib import Path

model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
print('Attempting to load', model_name)
model = SentenceTransformer(model_name)
# Save local copy
out_dir = Path('data/embeddings/models') / model_name
out_dir.parent.mkdir(parents=True, exist_ok=True)
print('Saving model to', out_dir)
model.save(str(out_dir))
print('Done')
