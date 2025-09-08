from transformers import pipeline

class QAAnswerer:
    def __init__(self, model_name="deepset/xlm-roberta-base-squad2", device=-1):
        self.qa = pipeline("question-answering", model=model_name, tokenizer=model_name, device=device)

    def answer(self, question, context, top_k=1):
        result = self.qa(question=question, context=context, top_k=top_k)
        if isinstance(result, list):
            return [r['answer'] for r in result]
        return [result['answer']]
