from app.rag.loader import PDFProcessor
import os

def main():
    # テスト用PDFのパス（事前にファイルを置いてください）
    pdf_path = "data/raw/test.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} が見当たりません。PDFを配置してください。")
        return

    processor = PDFProcessor(chunk_size=300, chunk_overlap=30)
    documents = processor.process_pdf(pdf_path)

    print(f"分割後のチャンク数: {len(documents)}")
    if documents:
        print("--- 最初のチャンクの内容 ---")
        print(documents[0].page_content)
        print("--- メタデータ ---")
        print(documents[0].metadata)

if __name__ == "__main__":
    main()