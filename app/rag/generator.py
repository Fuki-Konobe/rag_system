from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGGenerator:
    def __init__(self):
        # 性能とコストのバランスが良い gpt-3.5-turbo (または最新の gpt-4o-mini)
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            streaming=True,
            temperature=0)
        self.output_parser = StrOutputParser()

    def get_chain(self, retriever):
        """
        検索結果（Context）に含まれるメタデータ（source, page）を
        LLMが認識し、回答に引用を含めるためのチェインを構築。
        """
        
        # 指示（System Prompt）の最適化
        template = """あなたはスタートアップ企業のナレッジ担当アシスタントです。
提供された【コンテキスト】の情報のみを使用して、ユーザーの質問に正確に答えてください。

■ 回答のルール:
1. 回答は必ず日本語で行ってください。
2. コンテキストに情報が含まれていない場合は、「資料には該当する記載がありません」と伝えてください。
3. 複数の資料にまたがる場合は、それらを統合して整理して回答してください。
4. 連続したページの情報は関連している可能性が高いので、まとめて回答に活用してください。

【コンテキスト】:
{context}

【ユーザーの質問】:
{question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        # ドキュメントオブジェクトからテキストとメタデータを抽出してプロンプトに渡す補助関数
        def format_docs(docs):
            formatted = []
            for d in docs:
                source = d.metadata.get('source', '不明')
                page = d.metadata.get('page', '-')
                content = f"--- [出典: {source} P.{page}] ---\n{d.page_content}"
                formatted.append(content)
            return "\n\n".join(formatted)

        # RAGパイプラインの構築
        chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt 
            | self.llm 
            | self.output_parser
        )
        return chain