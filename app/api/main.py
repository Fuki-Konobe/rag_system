from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag.vectorstore import VectorStoreManager
from app.rag.generator import RAGGenerator

app = FastAPI(title="Tokushima RAG API")

# 1. 起動時に一度だけインスタンスを生成（効率化）
vdb_manager = VectorStoreManager()
retriever = vdb_manager.get_retriever()
generator = RAGGenerator()
rag_chain = generator.get_chain(retriever)

# 2. リクエストの型定義
class QueryRequest(BaseModel):
    question: str

# 3. レスポンスの型定義
class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # RAG実行
        answer = rag_chain.invoke(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}