import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from app.rag.loader import PDFProcessor
from langchain_community.document_loaders import PyMuPDFLoader
from app.rag.vectorstore import VectorStoreManager
from app.rag.generator import RAGGenerator
from app.rag.evaluator import RAGEvaluator
from contextlib import asynccontextmanager

# グローバルな「状態ホルダー」
class AppState:
    def __init__(self):
        self.hybrid_retriever = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 【起動時処理】BM25インデックスの初回作成
    print("Initializing Hybrid Retriever...")
    processor = PDFProcessor()
    documents = processor.process_directory("data/raw")
    state.hybrid_retriever = vdb_manager.get_hybrid_retriever(documents)
    yield
    # 【終了時処理】必要ならここでリソース解放
    print("Shutting down...")

app = FastAPI(title="RAG System", lifespan=lifespan)

# 保存先ディレクトリの準備
UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# インスタンスの初期化
vdb_manager = VectorStoreManager()
generator = RAGGenerator()

@app.post("/upload", summary="PDFファイルをアップロードして学習させる")
async def upload_file(file: UploadFile = File(...)):
    # 1. ファイルのバリデーション
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDFファイルのみ受け付けています。")

    file_path = UPLOAD_DIR / file.filename

    try:
        # 2. ファイルをローカルに保存
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. 保存したファイルを即座にロード・分割
        processor = PDFProcessor()
        loader = PyMuPDFLoader(str(file_path))
        documents = loader.load()
        split_docs = processor.text_splitter.split_documents(documents)

        # 4. VectorDBに追加
        vdb_manager.add_documents(split_docs)

        # 5. Hybrid Retrieverを最新状態に更新
        new_documents = processor.process_directory(str(UPLOAD_DIR))
        state.hybrid_retriever = vdb_manager.get_hybrid_retriever(documents=new_documents)

        return {
            "message": f"ファイル '{file.filename}' の学習が完了しました。",
            "chunks": len(split_docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: str):
    # 最新のDB状態でリトリーバーを取得
    retriever = vdb_manager.get_hybrid_retriever()
    rag_chain = generator.get_chain(retriever)
    
    answer = rag_chain.invoke(question)
    return {"answer": answer}

@app.post("/ask_stream")
async def ask_stream(question: str):
    # 毎回読み直さず、起動時に作ったものを使い回す
    if not state.hybrid_retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    rag_chain = generator.get_chain(state.hybrid_retriever)
    
    async def event_generator():
        async for chunk in rag_chain.astream(question):
            yield chunk
            await asyncio.sleep(0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/evaluate")
async def run_evaluation(question: str):
    # 1. 検索実行
    retriever = state.hybrid_retriever
    docs = retriever.get_relevant_documents(question)
    contexts = [doc.page_content for doc in docs]

    # 2. 回答生成（ストリーミングなしの通常生成）
    rag_chain = generator.get_chain(retriever)
    answer = rag_chain.invoke(question)

    # 3. 評価実行
    evaluator = RAGEvaluator(generator.llm)
    report = await evaluator.evaluate_response(question, answer, contexts)

    # Pandasの結果を辞書形式で返す
    return report.to_dict(orient="records")[0]