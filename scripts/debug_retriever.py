from app.rag.vectorstore import VectorStoreManager
import os
import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from app.rag.loader import PDFProcessor
from app.rag.vectorstore import VectorStoreManager
from app.rag.generator import RAGGenerator

def debug_search(query: str, k: int = 5):
    print(f"\n検証クエリ: {query}")
    print(f"取得件数 (k): {k}")
    print("-" * 50)
    processor = PDFProcessor()
    documents = processor.process_directory("data/raw")

    vdb_manager = VectorStoreManager()
    # 検索器を単体で取得
    retriever = vdb_manager.get_hybrid_retriever(documents=documents, search_kwargs={"k": k})
    
    # LLMを通さず、ベクトル検索の結果（Documentオブジェクトのリスト）を直接取得
    docs = retriever.invoke(query)

    if not docs:
        print("結果: 該当するドキュメントが一つも見つかりませんでした。")
        return

    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '不明')
        page = doc.metadata.get('page', '-')
        # 類似度スコアを表示したい場合は、vectorstore.similarity_search_with_score を使う必要がありますが、
        # まずは内容と出典を確認します。
        print(f"【結果 {i+1}】")
        print(f"出典: {source} (P.{page})")
        print(f"内容抜粋:\n{doc.page_content[:300]}...") # 最初の300文字を表示
        print("-" * 50)

if __name__ == "__main__":
    # 1. 精度が低いと感じる「描画処理」に関する具体的な質問を入力してください
    test_query = "Shift_JISはどこで使われている？" 
    
    debug_search(test_query)