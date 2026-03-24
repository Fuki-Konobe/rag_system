from app.rag.loader import PDFProcessor
from app.rag.vectorstore import VectorStoreManager
import os

def main():
    # 1. PDFの読み込みと分割
    pdf_path = "data/raw/test.pdf"
    processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
    documents = processor.process_pdf(pdf_path)
    
    # 2. VectorDBへの保存
    vdb_manager = VectorStoreManager()
    print(f"{len(documents)}件のチャンクをベクトル化して保存中...")
    vdb_manager.add_documents(documents)
    print("保存完了。")

    # 3. 検索テスト（Retrieverの動作確認）
    retriever = vdb_manager.get_retriever(search_kwargs={"k": 2})
    query = "井出さんは何の農家？" # PDFの内容に合わせた質問に変えてみてください
    results = retriever.invoke(query)

    print(f"\n質問: {query}")
    print("-" * 30)
    for i, doc in enumerate(results):
        print(f"結果 {i+1}:")
        print(doc.page_content[:100] + "...")
        print(f"参照元: {doc.metadata}\n")

if __name__ == "__main__":
    main()