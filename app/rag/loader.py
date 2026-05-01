import os
import pdfplumber
import pandas as pd
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "---", "。", "！", "？", "|", " ", ""]
        )
        # 要約用の高速LLM
        self.summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _get_doc_summary(self, sample_text: str) -> str:
        """資料の冒頭テキストから、その資料の全体像を1〜2文で要約する"""
        if not sample_text.strip():
            return "概要不明な資料"
        
        prompt = f"""
以下の資料の冒頭部分を読み、この資料が「何について（プロジェクト名、年度、対象、カテゴリなど）」書かれたものか、1~2文で記述してください。

（例）
- これは2025年のプロジェクトXの収支計画に関する資料です。年間の支出や収入の見込み、その内訳が記載されています。
- これはプロジェクトYのに関する2025年4月1日の会議の議事録です。主にプロジェクトの進捗状況と今後の課題について議論されています。

資料冒頭:
{sample_text[:3000]}  # 最初の3000文字程度をコンテキストとして利用
"""
        try:
            response = self.summary_llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"要約生成エラー: {e}")
            return "概要の取得に失敗しました"

    def _extract_page_with_tables(self, file_path: str) -> List[Document]:
        """テキストと構造化された表を抽出し、資料全体の要約をメタデータに付帯させる"""
        documents = []
        fname = os.path.basename(file_path)
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # 1. 資料全体の「コンテキスト（要約）」を生成
                # 最初の数ページのテキストを使って資料の正体を特定する
                full_text_for_summary = ""
                for i in range(min(3, len(pdf.pages))): # 最初の3ページ程度
                    full_text_for_summary += pdf.pages[i].extract_text() or ""
                
                doc_summary = self._get_doc_summary(full_text_for_summary)
                print(f"   [Summary] {fname}: {doc_summary}")

                # 2. 各ページの抽出
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    # 表の抽出とMarkdown的構造化
                    tables = page.extract_tables()
                    table_content = ""
                    if tables:
                        for table in tables:
                            if not table or len(table) < 2: continue
                            df = pd.DataFrame(table).fillna(" ")
                            headers = [str(c).replace("\n", "") for c in df.iloc[0]]
                            
                            formatted_rows = []
                            for _, row in df.iloc[1:].iterrows():
                                newline_char = '\n'
                                cells = [f"{h}: {str(v).replace(newline_char, '')}" for h, v in zip(headers, row)]
                                formatted_rows.append(" / ".join(cells))
                            
                            table_content += "\n\n【表データ】\n" + "\n".join(formatted_rows) + "\n\n"
                    
                    combined_content = f"{text}\n\n{table_content}"
                    
                    documents.append(Document(
                        page_content=combined_content,
                        metadata={
                            "source": file_path,
                            "file_name": fname,
                            "page": i,
                            "doc_summary": doc_summary # 全チャンクに共通のコンテキストを持たせる
                        }
                    ))
        except Exception as e:
            print(f"ファイル読み込みエラー ({file_path}): {e}")
            
        return documents

    def process_directory(self, directory_path: str) -> List[Document]:
        all_raw_documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    print(f"Processing: {file}")
                    all_raw_documents.extend(self._extract_page_with_tables(os.path.join(root, file)))
        
        # チャンク分割（metadataは自動的に引き継がれる）
        split_docs = self.text_splitter.split_documents(all_raw_documents)
        
        # 【実装の肝】各チャンクの本文を「文脈注入型」に書き換える
        for doc in split_docs:
            # 1. メタデータの更新（ここが抜けていました）
            raw_page_idx = doc.metadata.get("page", 0)
            page_num = raw_page_idx + 1
            
            # メタデータに「人間が読めるページ番号」を明示的にセット
            doc.metadata["page_number"] = page_num
            # （念のためUI側が "page" キーを参照している場合にも対応）
            doc.metadata["page"] = page_num 

            # 2. 本文への文脈注入
            fname = doc.metadata.get("file_name", "不明な資料")
            doc_summary = doc.metadata.get("doc_summary", "")

            contextual_header = (
                f"【資料名: {fname} (P.{page_num})】\n"
                f"【資料概要: {doc_summary}】\n"
                "--- 以下、本文 ---\n"
            )
            doc.page_content = contextual_header + doc.page_content

        print(f"全 {len(split_docs)} チャンクのメタデータ同期と文脈注入が完了しました。")
        return split_docs