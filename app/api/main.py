import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.rag.evaluator import RAGEvaluator
from app.rag.generator import RAGGenerator
from app.rag.loader import PDFProcessor
from app.rag.vectorstore import VectorStoreManager

import logging
import warnings

# 1. すべてのロガーから 'chromadb' と 'posthog' を黙らせる
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('posthog').setLevel(logging.ERROR)

# 2. それでも標準出力(stdout)に直接書き込まれるログを抑制する
# (特にライブラリ内部で print や直書きされている場合)
warnings.filterwarnings("ignore", message="Failed to send telemetry event")

# ===== 設定定数 =====
DATA_DIR = Path("data/raw")
APP_TITLE = "RAG System"

# 初期化時のメッセージ
MSG_INITIALIZING = "Initializing Hybrid Retriever..."
MSG_SHUTTING_DOWN = "Shutting down..."
MSG_BACKGROUND_UPDATE = "Background Task: Updating Hybrid Index..."
MSG_UPDATE_COMPLETED = "Background Task: Index update completed."

# エラーメッセージ
ERR_RETRIEVER_UNINITIALIZED = "Retriever not initialized"
ERR_FILE_EXTENSION = "PDFファイルのみ受け付けています。"


# ===== グローバルステート管理 =====
class AppState:
    """アプリケーションのグローバル状態を管理するクラス."""
    def __init__(self) -> None:
        self.hybrid_retriever = None
        self.is_indexing = False


state = AppState()
vdb_manager = VectorStoreManager()
generator = RAGGenerator()


# ===== ライフサイクル管理 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションの起動時と終了時の処理を統括.
    
    起動時：BM25インデックスの初期化
    終了時：リソースの解放
    """
    print(MSG_INITIALIZING)
    processor = PDFProcessor()
    documents = processor.process_directory(str(DATA_DIR))
    state.hybrid_retriever = vdb_manager.get_hybrid_retriever(documents)
    
    if documents:
        # 2. 空になったDBにデータを追加（永続化）
        vdb_manager.add_documents(documents)
        # 3. リトリーバーを構成
        state.hybrid_retriever = vdb_manager.get_hybrid_retriever(documents)
        print(f"System initialized with {len(documents)} chunks.")
    else:
        print("Warning: No documents found in DATA_DIR. DB remains empty.")
    
    yield

    yield
    
    print(MSG_SHUTTING_DOWN)


# ===== FastAPIアプリの初期化 =====
app = FastAPI(title=APP_TITLE, lifespan=lifespan)

# 保存先ディレクトリの準備
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ===== バックグラウンドタスク =====
def update_index_task() -> None:
    """全PDFファイルをスキャンして検索インデックスを更新する重い処理.
    
    バックグラウンドタスクとして実行され、レスポンス返却を阻害しない.
    """
    state.is_indexing = True  # 開始時にTrue
    try:
        print("Background Task: Updating Index...")
        processor = PDFProcessor()
        new_documents = processor.process_directory("data/raw")
        state.hybrid_retriever = vdb_manager.get_hybrid_retriever(new_documents)
        print("Background Task: Completed.")
    finally:
        state.is_indexing = False  # 成功・失敗に関わらず最後はFalse


# ===== APIエンドポイント =====
@app.post("/ask", summary="質問に対して回答を生成")
async def ask_question(question: str) -> dict:
    """質問に対してRAGシステムで回答を生成.
    
    Args:
        question: ユーザーからの質問文
        
    Returns:
        回答を含む辞書
    """
    retriever = vdb_manager.get_hybrid_retriever()
    rag_chain = generator.get_chain(retriever)
    answer = rag_chain.invoke(question)
    
    return {"answer": answer}


@app.post("/ask_stream", summary="質問に対してストリーミングで回答を生成")
async def ask_stream(question: str):
    """質問に対してストリーミング形式で回答を生成.
    
    起動時に初期化したretrieverを再利用し、高速応答を実現.
    
    Args:
        question: ユーザーからの質問文
        
    Raises:
        HTTPException: Retrieverが未初期化の場合
        
    Yields:
        ストリーミング形式の回答チャンク
    """
    if not state.hybrid_retriever:
        raise HTTPException(status_code=503, detail=ERR_RETRIEVER_UNINITIALIZED)
    
    rag_chain = generator.get_chain(state.hybrid_retriever)
    
    async def event_generator():
        async for chunk in rag_chain.astream(question):
            yield chunk
            await asyncio.sleep(0)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/evaluate", summary="回答品質を評価")
async def run_evaluation(question: str) -> dict:
    """質問に対する回答を生成し、品質を評価.
    
    以下の処理を順序実行：
    1. 関連ドキュメントの検索
    2. RAGによる回答生成
    3. 回答品質の評価
    
    Args:
        question: ユーザーからの質問文
        
    Returns:
        評価結果を含む辞書
    """
    retriever = state.hybrid_retriever
    docs = retriever.get_relevant_documents(question)
    contexts = [doc.page_content for doc in docs]
    
    rag_chain = generator.get_chain(retriever)
    answer = rag_chain.invoke(question)
    
    evaluator = RAGEvaluator(generator.llm)
    report = await evaluator.evaluate_response(question, answer, contexts)
    
    return report.to_dict(orient="records")[0]


@app.post("/upload", summary="PDFファイルをアップロードしてインデックスを更新")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> dict:
    """PDFファイルをアップロードしてシステムに学習させる.
    
    重いインデックス更新処理はバックグラウンドタスクで実行し、
    ユーザーへの応答を高速化.
    
    Args:
        background_tasks: バックグラウンドタスク队列
        file: アップロードされたPDFファイル
        
    Returns:
        ファイル保存受理メッセージ
    """
    file_path = DATA_DIR / file.filename
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # インデックス更新をバックグラウンドで実行
        background_tasks.add_task(update_index_task)
        
        return {
            "status": "accepted",
            "message": f"{file.filename} の保存が完了しました。検索インデックスは数秒後に自動更新されます。"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/status")
async def get_status():
    return {"is_indexing": state.is_indexing}