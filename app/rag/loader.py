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
        print(f"読込完了: {len(raw_documents)} ページ ")

        # 2. まとめてチャンク分割
        split_docs = self.text_splitter.split_documents(raw_documents)
        print(f"分割完了: {len(split_docs)} チャンク")

        for doc in split_docs:
            # 1. sourceフルパスからファイル名のみを抽出して新設
            if "source" in doc.metadata:
                doc.metadata["file_name"] = os.path.basename(doc.metadata["source"])
            
            # 2. ページ番号を1始まりに補正 (表示用)
            # PyMuPDFLoaderはデフォルトで 'page' を持っています
            current_page = doc.metadata.get("page", 0)
            doc.metadata["page_number"] = current_page + 1
        
        print(f"メタデータ付与完了")
        return split_docs