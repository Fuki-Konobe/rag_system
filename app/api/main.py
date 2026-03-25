import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from app.rag.loader import PDFProcessor
from app.rag.vectorstore import VectorStoreManager
from app.rag.generator import RAGGenerator

app = FastAPI(title="RAG API")

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
        # ここではアップロードした特定のファイルのみを処理
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(str(file_path))
        documents = loader.load()
        split_docs = processor.text_splitter.split_documents(documents)

        # 4. VectorDBに追加
        vdb_manager.add_documents(split_docs)

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
    processor = PDFProcessor()
    documents = processor.process_directory("data/raw")

    retriever = vdb_manager.get_hybrid_retriever(documents=documents)
    rag_chain = generator.get_chain(retriever)

    # 非同期ジェネレータを作成
    async def generate_answer():
        # astream() を使うことで、生成されたトークンを逐次取得
        async for chunk in rag_chain.astream(question):
            yield chunk
            # ネットワークのバッファリングを考慮し、わずかな待機を入れる場合もあります
            await asyncio.sleep(0.01)

    return StreamingResponse(generate_answer(), media_type="text/plain")