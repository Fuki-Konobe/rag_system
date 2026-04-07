"""自動評価スクリプト: RAGシステムの性能を包括的に評価するためのツール
実行方法: docker-compose exec rag_api python scripts/auto_evaluator.py "評価の備考"
"""

import asyncio
import json
import os
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score_fn
from dotenv import load_dotenv
load_dotenv()

from app.api.main import ask_question 

# --- 1. LLM Judgeの出力構造定義 ---
class EvaluationResult(BaseModel):
    score: int = Field(description="1から5までの評価スコア")
    reason: str = Field(description="そのスコアをつけた具体的な理由（簡潔に）")

class RAGEvaluator:
    def __init__(self, dataset_path: str, output_dir: str = "./eval_results", remarks: str = ""):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.remarks = remarks
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 評価用LLM
        self.judge_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        ).with_structured_output(EvaluationResult)

        # 定量的類似度計算用（OpenAIの埋め込みモデルを使用）
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは公平で厳密なAI評価者です。
提供された【ユーザーの質問】に対する【AIの回答】を、【模範解答】を基準にして1〜5点の5段階で評価してください。

[評価基準]
5点: 模範解答の重要な情報がすべて含まれており、過不足なく正確である。
4点: 模範解答の主要な意味は捉えているが、微細な情報が欠けている。
3点: 部分的に正しいが、重要な情報の一部が欠落しているか、少し曖昧である。
2点: 関連する話題には触れているが、的はずれな回答である。
1点: 全く無関係、または誤った情報である。"""),
            ("human", "[質問]\n{query}\n\n[模範解答]\n{reference_answer}\n\n[AI回答]\n{actual_answer}")
        ])
        self.judge_chain = self.eval_prompt | self.judge_llm

    async def get_rag_response(self, query: str) -> dict:
        """RAGシステムの呼び出し"""
        return await ask_question(query)

    def calculate_semantic_similarity(self, ref: str, act: str) -> float:
        """ベクトル類似度 (Cosine Similarity) の計算"""
        if not act or act.strip() == "": return 0.0
        vecs = self.embeddings.embed_documents([ref, act])
        return float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])

    async def evaluate_all(self):
        print(f"🚀 評価開始: {self.remarks}")
        results = []

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # BERTScore用のリスト
        all_refs = []
        all_acts = []

        for line in tqdm(lines, desc="Running RAG & LLM Judge"):
            if not line.strip(): continue
            data = json.loads(line)
            
            # RAG実行（時間計測）
            start_time = time.time()
            rag_output = await self.get_rag_response(data["query"])
            response_time = time.time() - start_time
            
            actual_answer = rag_output.get("answer", "")
            retrieved_sources = rag_output.get("sources", [])

            # 1. 検索精度 (Recall)
            is_hit = any(
                str(src.get("file_name")) == str(data["evidence_source"]) and 
                int(src.get("page")) == int(data["evidence_page"])
                for src in retrieved_sources
            )

            # 2. LLM Judge評価 (修正済み)
            try:
                eval_res = self.judge_chain.invoke({
                    "query": data["query"],
                    "reference_answer": data["reference_answer"],
                    "actual_answer": actual_answer
                })
                # dict型で返ってきた場合とPydantic型で返ってきた場合の両方に対応
                score = eval_res.score if hasattr(eval_res, 'score') else eval_res.get('score', 0)
                reason = eval_res.reason if hasattr(eval_res, 'reason') else eval_res.get('reason', "parse error")
            except Exception as e:
                score, reason = 0, f"Error: {e}"

            # 3. セマンティック類似度
            sem_sim = self.calculate_semantic_similarity(data["reference_answer"], actual_answer)

            results.append({
                "Category": data.get("category"),
                "Query": data["query"],
                "Is_Hit": is_hit,
                "Actual_Answer": actual_answer,
                "Judge_Score": score,
                "Judge_Reason": reason,
                "Semantic_Similarity": sem_sim,
                "Reference": data["reference_answer"],
                "Response_time": response_time
            })
            all_refs.append(data["reference_answer"])
            all_acts.append(actual_answer if actual_answer else " ")

        # 4. BERTScore の計算 (一括処理で高速化)
        print("💡 BERTScoreを計算中...")
        P, R, F1 = bert_score_fn(all_acts, all_refs, lang="ja", rescale_with_baseline=True)
        for i, res in enumerate(results):
            res["BERTScore_F1"] = float(F1[i])

        self._export_results(results)

    def _export_results(self, results: list):
        df = pd.DataFrame(results)
        
        # 保存パスの設定(eval_YYYYMMDD_HHMMSS.xlsx(日本時間) と evaluation_history.csv)
        timestamp_japan = (datetime.utcnow() + pd.Timedelta(hours=9)).strftime("%Y%m%d_%H%M%S")
        file_base = f"eval_{timestamp_japan}"
        excel_path = os.path.join(self.output_dir, f"{file_base}.xlsx")
        history_path = os.path.join(self.output_dir, "evaluation_history.csv")

        # 統計
        summary = {
            "datetime": timestamp_japan,
            "remarks": self.remarks,
            "total": len(df),
            "hit_rate": df["Is_Hit"].mean(),
            "avg_judge_score": df["Judge_Score"].mean(),
            "avg_semantic_sim": df["Semantic_Similarity"].mean(),
            "avg_bert_f1": df["BERTScore_F1"].mean(),
            "avg_time": df["Response_time"].mean()
        }

        # 個別詳細（Excel）の保存
        df.to_excel(excel_path, index=False)

        # 履歴への追記 (CSV)
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(history_path, mode='a', header=not os.path.exists(history_path), index=False, encoding="utf-8-sig")

        print(f"\n=== 📊 完了: {self.remarks} ===")
        print(f"検索ヒット率: {summary['hit_rate']:.2%}")
        print(f"平均 Judge Score: {summary['avg_judge_score']:.2f}")
        print(f"平均 BERTScore: {summary['avg_bert_f1']:.4f}")
        print(f"平均応答時間: {summary['avg_time']:.2f}秒")
        print(f"詳細: {excel_path}")

if __name__ == "__main__":
    import sys
    # 実行時に備考を入力可能にする
    comment = sys.argv[1] if len(sys.argv) > 1 else "No comment"
    
    evaluator = RAGEvaluator(
        dataset_path="./eval_dataset.jsonl", 
        remarks=comment
    )
    asyncio.run(evaluator.evaluate_all())