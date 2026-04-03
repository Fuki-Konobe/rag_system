import os
import MeCab
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

from langchain.globals import set_verbose

set_verbose(True)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "/src/data/vectordb"): # コンテナ内の絶対パスに変更
        self.persist_directory = persist_directory
        self.chroma_settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
        
        # フォルダが存在しない場合に自動作成
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def add_documents(self, documents):
        # 既存のデータを上書き・更新する
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            client_settings=self.chroma_settings
        )
        vectorstore.persist()
        return vectorstore

    def japanese_tokenizer(self, text):
        tagger = MeCab.Tagger("-Owakati")
        return tagger.parse(text).split()

    def get_hybrid_retriever(self, documents, search_kwargs: dict = {"k": 3}):
        # 1. ベクトル検索器の準備
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            client_settings=self.chroma_settings
        )
        vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        # 2. BM25検索器の準備（日本語の分かち書きが必要）
        # documents は分割済みのチャンク（List[Document]）を想定
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            tokenizer=self.japanese_tokenizer
        )
        bm25_retriever.k = search_kwargs.get("k", 3)

        # 3. 統合（比率は 0.5:0.5 が一般的）
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )

        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever,
            llm=self.llm
        )
        
        return mq_retriever