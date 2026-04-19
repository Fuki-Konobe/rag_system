import os
import MeCab
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.prompts import PromptTemplate
from chromadb.config import Settings

class VectorStoreManager:
    def __init__(self, persist_directory: str = "/src/data/vectordb"):
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

    def get_hybrid_retriever(self, documents, initial_k: int = 15, final_k: int = 5):
        """
        initial_k: 最初のハイブリッド検索で広く取得する件数（デフォルト15）
        final_k: リランク後に最終的にLLMに渡す件数（デフォルト5）
        """
        # 1. ベクトル検索器の準備（広く浅く拾うために initial_k を設定）
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            client_settings=self.chroma_settings
        )
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})

        # 2. BM25検索器の準備（同様に initial_k を設定）
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            tokenizer=self.japanese_tokenizer
        )
        bm25_retriever.k = initial_k

        # 3. アンサンブル統合（比率は 0.6:0.4 を維持）
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )

        CUSTOM_QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""あなたは行政資料や事業資料を検索する専門アシスタントです。
        ユーザーの質問に対して、ベクトルデータベースから関連文書を検索するために、意味が同じで異なる表現の検索クエリを3つ生成してください。
        行政用語は極力そのまま残し、簡潔なキーワードベースのクエリを含めてください。
        各クエリは改行で区切って出力してください。

        オリジナル質問: {question}"""
        )

        # 4. Multi-Queryによる質問の多角化
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever,
            llm=self.llm,
            prompt=CUSTOM_QUERY_PROMPT,
        )

        # 5. FlashRankによるリランキング (Contextual Compressionの実装)
        # PyTorch非依存で高速に動作する多言語対応モデルを指定
        compressor = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2",
            top_n=final_k
        )
        
        # 最終的なリトリーバーパイプラインの構築
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=mq_retriever
        )
        
        return compression_retriever