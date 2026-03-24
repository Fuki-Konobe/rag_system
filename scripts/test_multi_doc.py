import os
from app.rag.loader import PDFProcessor
from app.rag.vectorstore import VectorStoreManager
from app.rag.generator import RAGGenerator

def main():
    print("=== Phase 1: ドキュメントの読み込みと分割 ===")
    # data/raw ディレクトリ内の全PDFを処理
    processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
    documents = processor.process_directory("data/raw")
    
    if not documents:
        print("エラー: data/raw にPDFファイルが見つかりません。")
        return

    print(f"\n=== Phase 2: VectorDBへの保存（ベクトル化） ===")
    vdb_manager = VectorStoreManager()
    vdb_manager.add_documents(documents)
    print("データベースの更新が完了しました。")

    print(f"\n=== Phase 3: RAGチェインによる回答生成 ===")
    # 検索器（Retriever）の準備
    retriever = vdb_manager.get_retriever(search_kwargs={"k": 3})
    
    # 生成器（Generator）の準備と実行
    generator = RAGGenerator()
    rag_chain = generator.get_chain(retriever)

    # テスト用の質問（徳島のナレッジに合わせて調整してください）
    query = "可視化描画処理の方法について教えてください。" # PDFの内容に合わせた質問に変えてみてください
    
    print(f"質問: {query}")
    print("AIが思考中...\n")
    
    # RAG実行
    response = rag_chain.invoke(query)

    print("=" * 50)
    print("【AIからの回答】")
    print(response)
    print("=" * 50)

if __name__ == "__main__":
    main()