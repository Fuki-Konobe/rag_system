import os
import pdfplumber
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "---", "。", "！", "？", "|", " ", ""]
        )

    def _extract_page_with_tables(self, file_path: str) -> List[Document]:
        """pdfplumberを使用して、テキストと表（Markdown）を抽出する"""
        documents = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # 1. 通常テキストの抽出
                    text = page.extract_text() or ""
                    
                    # 2. 表の抽出とMarkdown変換
                    tables = page.extract_tables()
                    table_content = ""
                    if tables:
                        for table in tables:
                            # None値を空文字に置換してDataFrame化
                            df = pd.DataFrame(table).fillna("")
                            # 1行目をヘッダーとして扱う（表として認識しやすくする）
                            if not df.empty:
                                table_content += "\n\n" + df.to_markdown(index=False) + "\n\n"
                    
                    # テキストと構造化された表データを結合
                    # これにより「賃金」の横に金額がある構造を維持
                    combined_content = f"{text}\n\n{table_content}"
                    
                    documents.append(Document(
                        page_content=combined_content,
                        metadata={
                            "source": file_path,
                            "page": i
                        }
                    ))
        except Exception as e:
            print(f"ファイル読み込みエラー ({file_path}): {e}")
            
        return documents

    def process_directory(self, directory_path: str) -> List[Document]:
        all_raw_documents = []
        
        # 指定ディレクトリ内のPDFを走査
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    all_raw_documents.extend(self._extract_page_with_tables(file_path))
        
        print(f"読込完了: {len(all_raw_documents)} ページ ")

        # チャンク分割
        split_docs = self.text_splitter.split_documents(all_raw_documents)
        print(f"分割完了: {len(split_docs)} チャンク")

        # メタデータ処理
        for doc in split_docs:
            if "source" in doc.metadata:
                fname = os.path.basename(doc.metadata["source"])
                doc.metadata["file_name"] = fname
                # 検索時に資料名を意識させるための接頭辞
                prefix = f"【資料名：{fname}】\n"
                doc.page_content = prefix + doc.page_content

            current_page = doc.metadata.get("page", 0)
            doc.metadata["page_number"] = current_page + 1
        
        print(f"メタデータ付与完了")
        return split_docs