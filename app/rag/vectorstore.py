import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class VectorStoreManager:
    def __init__(self, persist_directory: str = "/src/data/vectordb"): # コンテナ内の絶対パスに変更
        self.persist_directory = persist_directory
        
        # フォルダが存在しない場合に自動作成
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def add_documents(self, documents):
        # 既存のデータを上書き・更新する
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vectorstore.persist()
        return vectorstore

    def get_retriever(self, search_kwargs: dict = {"k": 3}):
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs=search_kwargs)