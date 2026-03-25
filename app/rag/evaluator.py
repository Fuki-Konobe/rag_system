from ragas import aevaluate
from datasets import Dataset

# 0.4.3 以降の正しいインポートパス
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy # クラス名は Relevancy です
from ragas.metrics._context_precision import ContextPrecision

class RAGEvaluator:
    def __init__(self, llm):
        self.metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm),
            # ContextPrecision(llm=llm)
        ]
        self.llm = llm

    async def evaluate_response(self, question, answer, contexts):
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        
        # evaluate実行
        result = await aevaluate(
            dataset,
            metrics=self.metrics
        )
        return result.to_pandas()