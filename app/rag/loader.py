import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200): # サイズを拡大
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def process_directory(self, directory_path: str) -> List[Document]:
        # PyMuPDFLoaderを指定
        loader = DirectoryLoader(
            directory_path, 
            glob="**/*.pdf", 
            loader_cls=PyMuPDFLoader
        )
        
        raw_documents = loader.load()
        print(f"読み込み完了: {len(raw_documents)} ページ (全ファイル合計)")

        # 2. まとめてチャンク分割
        split_docs = self.text_splitter.split_documents(raw_documents)
        print(f"分割完了: {len(split_docs)} チャンク")
        
        return split_docs