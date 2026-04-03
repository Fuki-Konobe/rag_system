import os
import logging
from pathlib import Path
from app.rag.vectorstore import VectorStoreManager
from app.rag.loader import PDFProcessor

# ログ設定
logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.ERROR)

def debug_rent_issue():
    print("=== 🛠️ 収支計画デバッグ開始 ===")
    
    # 初期化
    manager = VectorStoreManager()
    processor = PDFProcessor(chunk_size=500, chunk_overlap=100)
    
    # ターゲット単語
    target_word = "賃金"
    target_file = "収支計画.pdf"
    
    # データディレクトリ（コンテナ環境に対応）
    data_dir = Path("/src/data/raw") if Path("/src/data/raw").exists() else Path("data/raw")

    # --- Step 1: 生テキストの抽出状態を確認 ---
    print(f"\n[Step 1: Raw Text Check - {target_file}]")
    print(f"データディレクトリ: {data_dir}")
    
    # PDFを読み込み
    docs = processor.process_directory(str(data_dir)) 
    
    target_chunks = [d for d in docs if target_word in d.page_content]
    
    if not target_chunks:
        print(f"❌ 致命的: '{target_word}' という文字列が、抽出後のテキスト内に存在しません。")
        # 似たパターンの検索
        all_text = "".join([d.page_content for d in docs if target_file in d.metadata.get("file_name", "")])
        print(f"🔍 ファイル内の全テキスト抜粋 (最初の300文字): \n{repr(all_text[:300])}")
    else:
        print(f"✅ 発見: {len(target_chunks)} 個のチャンクに含まれています。")
        for i, chunk in enumerate(target_chunks):
            print(f"\n--- チャンク {i+1} の生データ (repr形式) ---")
            # reprを使うことで、改行 \n や空白がどう入っているか可視化します
            print(repr(chunk.page_content))

    # --- Step 2: 類似度スコアの確認 ---
    print(f"\n[Step 2: Similarity Score Analysis]")
    # ドキュメントをベクトルストアに追加
    manager.add_documents(docs)
    
    # Chromaベクトルストアをロード
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from chromadb.config import Settings
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    chroma_settings = Settings(anonymized_telemetry=False, is_persistent=True)
    
    vectorstore = Chroma(
        persist_directory=manager.persist_directory,
        embedding_function=embeddings,
        client_settings=chroma_settings
    )
    
    # 距離（スコア）付きで検索
    results_with_scores = vectorstore.similarity_search_with_score(target_word, k=5)
    
    print(f"\n検索クエリ: 「{target_word}」")
    if results_with_scores:
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n--- 結果 {i} ---")
            print(f"Score: {score:.4f} (低いほど近い)")
            print(f"File: {doc.metadata.get('file_name')} | Page: {doc.metadata.get('page_number')}")
            print(f"Content Preview: {doc.page_content[:100]}...")
    else:
        print("❌ ベクトル検索で結果が見つかりません")

    # --- Step 3: 分かち書きの確認 ---
    print(f"\n[Step 3: Tokenization Test]")
    try:
        tokens = manager.japanese_tokenizer(target_word)
        print(f"「{target_word}」の分割結果: {tokens}")
    except Exception as e:
        print(f"⚠️  分かち書きのエラー: {e}")
        print("（MeCabがインストールされていない可能性があります）")

    # --- Step 4: BM25検索の確認 ---
    print(f"\n[Step 4: BM25 Search Test]")
    try:
        from langchain_community.retrievers import BM25Retriever
        bm25_retriever = BM25Retriever.from_documents(
            docs,
            tokenizer=manager.japanese_tokenizer
        )
        bm25_results = bm25_retriever.invoke(target_word)
        
        print(f"BM25検索で {len(bm25_results)} 件の結果を取得")
        for i, doc in enumerate(bm25_results[:3], 1):
            print(f"\n--- BM25結果 {i} ---")
            print(f"File: {doc.metadata.get('file_name')} | Page: {doc.metadata.get('page_number')}")
            print(f"Content Preview: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"⚠️  BM25検索エラー: {e}")

if __name__ == "__main__":
    debug_rent_issue()