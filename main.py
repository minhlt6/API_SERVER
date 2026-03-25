#Import các thư viện cần thiết
import os
import logging
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncpg
from starlette.concurrency import iterate_in_threadpool
from qdrant_client import QdrantClient
#Import các model và các hàm cần thiết từ core 
from core.config import QDRANT_URL, QDRANT_API_KEY, DATABASE_URL
from core.vectorstore import build_vectorstore_improved, load_vectorstore_improved
from core.retriever import HybridRetriever
from core.qa_pipeline import ask_ai_improved, ask_ai_stream_delta
# Hàm log lỗi an toàn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))

# Khởi tạo database để lưu lịch sử trò chuyện
async def init_db_asyncpg(pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        ''')
        await conn.execute('''
            ALTER TABLE history
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_history_session_id_id
            ON history(session_id, id)
        ''')

async def get_history_async(pool: asyncpg.Pool, session_id: str):
    try:
        query = """
            SELECT role, content FROM (
                SELECT id, role, content FROM history
                WHERE session_id = $1
                ORDER BY id DESC LIMIT $2
            ) sub
            ORDER BY id ASC
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, session_id, MAX_HISTORY_MESSAGES)
        return [{"role": row["role"], "content": row["content"]} for row in rows]
    except Exception:
        logger.exception("Lỗi khi truy vấn lịch sử trò chuyện:", exc_info=True)
        return []

async def save_turn_async(pool: asyncpg.Pool, session_id: str, user_msg: str, assistant_msg: str):
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO history (session_id, role, content) VALUES ($1, $2, $3)",
                    session_id,
                    "user",
                    user_msg,
                )
                await conn.execute(
                    "INSERT INTO history (session_id, role, content) VALUES ($1, $2, $3)",
                    session_id,
                    "assistant",
                    assistant_msg,
                )
    except Exception:
        logger.exception("Lỗi khi lưu lượt hội thoại:", exc_info=True)


#Khởi tạo hệ thống khi start server 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Đang khởi tạo API SERVER ...")
    pool = None
    try:
        pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
        )
        app.state.db_pool = pool
        await init_db_asyncpg(pool)

        client = QdrantClient(url = QDRANT_URL, api_key=QDRANT_API_KEY)
        collection_name= "quy_che_db"
        if not client.collection_exists(collection_name):
            logger.warning(f"Chưa có collection {collection_name} trên Qdrant Cloud. Đang xây dựng vectorstore mới...")
            db, all_chunks= build_vectorstore_improved()
        else :
            logger.info(f"Đã tìm thấy collection {collection_name} trên Qdrant Cloud. Đang tải vectorstore...")
            db, all_chunks = load_vectorstore_improved()

        if db is None or not all_chunks:
            raise RuntimeError("Không thể khởi tạo vectorstore. Kiểm tra log để biết chi tiết.")
        logger.info("Đang khởi tạo retriever ...")
        app.state.retriever = HybridRetriever(db, all_chunks)
        logger.info("API SERVER đã sẵn sàng!")
        yield
    except Exception :
        logger.exception("Lỗi khởi tạo hệ thống!", exc_info=True)
        raise RuntimeError("Lỗi khởi tạo hệ thống. Kiểm tra log để biết chi tiết.")
    finally :
        app.state.retriever = None
        if pool is not None:
            await pool.close()
        app.state.db_pool = None


def get_runtime_components(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    db_pool = getattr(request.app.state, "db_pool", None)
    if retriever is None or db_pool is None:
        raise HTTPException(status_code=503, detail="Hệ thống đang khởi động")
    return retriever, db_pool

#Cấu hình FastAPI với middleware CORS và lifespan để quản lý trạng thái hệ thống
app = FastAPI(lifespan=lifespan, title= "RAG API SERVER")
#Cho phép truy cập từ mọi nguồn 
allow_origins = [origin.strip() for origin in os.getenv("ALLOW_ORIGINS", "*").split(",") if origin.strip()]
if not allow_origins:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Định nghĩa Endpoint 
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str


@app.get("/healthz")
async def health_check(request: Request):
    ready = bool(getattr(request.app.state, "retriever", None) and getattr(request.app.state, "db_pool", None))
    return {"status": "ok" if ready else "starting", "ready": ready}

# Endpoint JSON thường (non-streaming) - trả toàn bộ câu trả lời một lúc
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request):
    """Endpoint chat thông thường - trả JSON response đầy đủ"""
    retriever, db_pool = get_runtime_components(request)
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Bạn chưa nhập câu hỏi")

    session_id = payload.session_id
    history = await get_history_async(db_pool, session_id)
    
    # Tập hợp toàn bộ response từ generator
    full_response = ""
    try:
        async for chunk in iterate_in_threadpool(ask_ai_improved(user_msg, history, retriever)):
            full_response = chunk
    except Exception:
        logger.exception("Lỗi khi xử lý phản hồi từ AI:", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi khi xử lý yêu cầu")
    
    # Lưu lịch sử sau khi có response đầy đủ
    await save_turn_async(db_pool, session_id, user_msg, full_response)
    
    return ChatResponse(response=full_response)

# Endpoint SSE streaming - trả chunk delta theo time real
@app.post("/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest, request: Request):
    """Endpoint chat streaming - trả SSE (Server-Sent Events) cho web frontend"""
    retriever, db_pool = get_runtime_components(request)
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Bạn chưa nhập câu hỏi")

    session_id = payload.session_id
    history = await get_history_async(db_pool, session_id)
    
    async def event_stream_generator():
        """Generator SSE - yield mỗi delta chunk và cuối cùng done=true"""
        full_response = ""
        try:
            # ask_ai_stream_delta yield từng delta chunk (không cumulative)
            async for delta_chunk in iterate_in_threadpool(ask_ai_stream_delta(user_msg, history, retriever)):
                full_response += delta_chunk
                # Gửi SSE event với delta chunk
                sse_data = json.dumps({"delta": delta_chunk, "done": False}, ensure_ascii=False)
                yield f"data: {sse_data}\n\n"
            
            # Gửi tín hiệu kết thúc
            yield 'data: {"delta": "", "done": true}\n\n'
            
            # Lưu lịch sử sau khi stream xong
            await save_turn_async(db_pool, session_id, user_msg, full_response)
            
        except Exception:
            logger.exception("Lỗi khi stream phản hồi từ AI:", exc_info=True)
            error_data = json.dumps({"error": "Lỗi khi xử lý yêu cầu", "done": True}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
