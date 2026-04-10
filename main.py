#Import các thư viện cần thiết
import asyncio
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
from core.config import (
    COLLECTION_ROUTER_TOP_N,
    DATABASE_URL,
    QDRANT_API_KEY,
    QDRANT_URL,
    SUPABASE_ADMIN_SYNC_TOKEN,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_STARTUP_SYNC_WAIT_SECONDS,
    SUPABASE_STORAGE_BUCKET,
    SUPABASE_SYNC_ENABLED,
    SUPABASE_SYNC_INTERVAL_SECONDS,
    SUPABASE_SYNC_SNAPSHOT_FILE,
    SUPABASE_URL,
)
from core.document_db import init_document_db
from core.supabase_sync_service import SupabaseStorageSyncService, SupabaseSyncCoordinator
from core.collection_router_retriever import CollectionRouterRetriever
from core.vectorstore import build_vectorstore_improved, load_vectorstore_improved
from core.models import embeddings
from core.qa_pipeline import ask_ai_improved, ask_ai_stream_delta
from api.admin_sync_router import router as admin_sync_router

# Hàm log lỗi an toàn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
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
                user_id TEXT, 
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                title TEXT, 
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        ''')
        await conn.execute('''
            ALTER TABLE history
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        ''')
        # 2 lệnh ALTER TABLE để cập nhật bảng cũ nếu đã tồn tại
        await conn.execute('ALTER TABLE history ADD COLUMN IF NOT EXISTS user_id TEXT')
        await conn.execute('ALTER TABLE history ADD COLUMN IF NOT EXISTS title TEXT')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_history_session_id_id
            ON history(session_id, id)
        ''')

# Hàm lấy danh sách phiên chat theo user_id
async def get_user_sessions_async(pool: asyncpg.Pool, user_id: str):
    query = """
        SELECT DISTINCT ON (session_id) 
            session_id, title, created_at 
        FROM history 
        WHERE user_id = $1 
        ORDER BY session_id, created_at DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, user_id)
    return [{"session_id": r["session_id"], "title": r["title"] or "Cuộc trò chuyện mới", "created_at": r["created_at"]} for r in rows]

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

#  Hàm lưu lượt chat để hỗ trợ title và user_id
async def save_turn_async(pool: asyncpg.Pool, session_id: str, user_msg: str, assistant_msg: str, user_id: str = None):
    try:
        async with pool.acquire() as conn:
            # Kiểm tra xem session này đã có tiêu đề chưa
            existing_title = await conn.fetchval("SELECT title FROM history WHERE session_id = $1 LIMIT 1", session_id)
            
            # Nếu chưa có, lấy 40 ký tự đầu làm tiêu đề
            title = existing_title if existing_title else user_msg[:40] + "..."

            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO history (session_id, user_id, role, content, title) VALUES ($1, $2, $3, $4, $5)",
                    session_id, user_id, "user", user_msg, title
                )
                await conn.execute(
                    "INSERT INTO history (session_id, user_id, role, content, title) VALUES ($1, $2, $3, $4, $5)",
                    session_id, user_id, "assistant", assistant_msg, title
                )
    except Exception:
        logger.exception("Lỗi khi lưu lượt hội thoại:", exc_info=True)


#Khởi tạo hệ thống khi start server 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Đang khởi tạo API SERVER ...")
    pool = None
    app.state.supabase_sync_service = None
    app.state.supabase_sync_coordinator = None
    app.state.supabase_startup_sync_task = None
    app.state.supabase_sync_stop_event = None
    app.state.supabase_sync_task = None
    try:
        init_document_db()

        pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
        )
        app.state.db_pool = pool
        await init_db_asyncpg(pool)

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()

        logger.info("Đang khởi tạo retriever (Qdrant collection router)...")
        app.state.retriever = CollectionRouterRetriever(
            base_retriever=None,
            qdrant_client=client,
            embeddings_model=embeddings,
            top_n_collections=COLLECTION_ROUTER_TOP_N,
        )

        if SUPABASE_SYNC_ENABLED:
            try:
                sync_service = SupabaseStorageSyncService(
                    supabase_url=SUPABASE_URL,
                    service_role_key=SUPABASE_SERVICE_ROLE_KEY,
                    bucket=SUPABASE_STORAGE_BUCKET,
                    snapshot_file=SUPABASE_SYNC_SNAPSHOT_FILE,
                )

                sync_coordinator = SupabaseSyncCoordinator(
                    sync_service=sync_service,
                    poll_interval_seconds=SUPABASE_SYNC_INTERVAL_SECONDS,
                )

                app.state.supabase_sync_service = sync_service
                app.state.supabase_sync_coordinator = sync_coordinator

                build_result = await build_vectorstore_improved(
                    sync_coordinator=sync_coordinator,
                    startup_wait_seconds=SUPABASE_STARTUP_SYNC_WAIT_SECONDS,
                )

                startup_sync_task = build_result.get("task")
                app.state.supabase_startup_sync_task = startup_sync_task
                initial_sync = build_result.get("initial_sync")
                timed_out = bool(build_result.get("timed_out"))

                if timed_out:
                    if SUPABASE_STARTUP_SYNC_WAIT_SECONDS > 0:
                        logger.warning(
                            "Supabase initial sync is still running after %ss. API startup continues and sync will finish in background.",
                            SUPABASE_STARTUP_SYNC_WAIT_SECONDS,
                        )
                    else:
                        logger.info("Supabase initial sync chạy nền, không chặn startup (SUPABASE_STARTUP_SYNC_WAIT_SECONDS=0).")
                elif isinstance(initial_sync, dict) and initial_sync.get("status") == "failed":
                    logger.warning(
                        "Supabase initial sync failed at startup. service will continue and retry in scheduler. error=%s",
                        initial_sync.get("error"),
                    )
                else:
                    summary = initial_sync.get("result") if isinstance(initial_sync, dict) and isinstance(initial_sync.get("result"), dict) else {}
                    logger.info(
                        "Supabase initial sync completed. added=%s updated=%s deleted=%s failed=%s total_objects=%s",
                        summary.get("added", 0),
                        summary.get("updated", 0),
                        summary.get("deleted", 0),
                        summary.get("failed", 0),
                        summary.get("total_objects", 0),
                    )

                sync_state = load_vectorstore_improved(sync_coordinator)
                if sync_state:
                    logger.info(
                        "Supabase sync state loaded. running=%s queued_events=%s last_sync_at=%s",
                        sync_state.get("running"),
                        sync_state.get("queued_events"),
                        sync_state.get("last_sync_at"),
                    )

                sync_stop_event = asyncio.Event()
                sync_task = asyncio.create_task(
                    sync_coordinator.run_polling_loop(stop_event=sync_stop_event)
                )

                app.state.supabase_sync_stop_event = sync_stop_event
                app.state.supabase_sync_task = sync_task
                logger.info(
                    "Supabase sync scheduler enabled. interval=%ss bucket=%s token_configured=%s",
                    SUPABASE_SYNC_INTERVAL_SECONDS,
                    SUPABASE_STORAGE_BUCKET,
                    bool(SUPABASE_ADMIN_SYNC_TOKEN),
                )
            except Exception:
                logger.exception("Không thể khởi động Supabase sync scheduler", exc_info=True)
        else:
            logger.info("Supabase sync scheduler đang tắt do thiếu cấu hình SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY/SUPABASE_STORAGE_BUCKET")

        logger.info("API SERVER đã sẵn sàng!")
        yield
    except Exception :
        logger.exception("Lỗi khởi tạo hệ thống!", exc_info=True)
        raise RuntimeError("Lỗi khởi tạo hệ thống. Kiểm tra log để biết chi tiết.")
    finally :
        startup_sync_task = getattr(app.state, "supabase_startup_sync_task", None)
        sync_stop_event = getattr(app.state, "supabase_sync_stop_event", None)
        sync_task = getattr(app.state, "supabase_sync_task", None)

        if startup_sync_task is not None and not startup_sync_task.done():
            startup_sync_task.cancel()
            try:
                await startup_sync_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Supabase startup sync task dừng với lỗi", exc_info=True)

        if sync_stop_event is not None:
            sync_stop_event.set()

        if sync_task is not None:
            try:
                await sync_task
            except Exception:
                logger.exception("Supabase sync scheduler dừng với lỗi", exc_info=True)

        app.state.supabase_sync_service = None
        app.state.supabase_sync_coordinator = None
        app.state.supabase_startup_sync_task = None
        app.state.supabase_sync_stop_event = None
        app.state.supabase_sync_task = None

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
app.include_router(admin_sync_router)

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
    user_id: str = None
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Chatbot API đang chạy!"}
    
@app.get("/healthz")
async def health_check(request: Request):
    ready = bool(getattr(request.app.state, "retriever", None) and getattr(request.app.state, "db_pool", None))
    return {"status": "ok" if ready else "starting", "ready": ready}

#Endpoint lấy danh sách session ở Sidebar
@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str, request: Request):
    _, db_pool = get_runtime_components(request)
    sessions = await get_user_sessions_async(db_pool, user_id)
    return {"sessions": sessions}

@app.get("/chat/history/{session_id}")
async def get_session_history(session_id: str, request: Request):
    """API để lấy toàn bộ nội dung tin nhắn cũ của một phiên chat cụ thể"""
    _, db_pool = get_runtime_components(request)
    history = await get_history_async(db_pool, session_id)
    if not history:
        return {"messages": []}
    return {"messages": history}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request):
    """Endpoint chat thông thường - trả JSON response đầy đủ"""
    retriever, db_pool = get_runtime_components(request)
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Bạn chưa nhập câu hỏi")

    session_id = payload.session_id
    user_id = payload.user_id # Lấy user_id từ request
    history = await get_history_async(db_pool, session_id)
    
    # Tập hợp toàn bộ response từ generator
    full_response = ""
    try:
        async for chunk in iterate_in_threadpool(ask_ai_improved(user_msg, history, retriever)):
            full_response = chunk
    except Exception:
        logger.exception("Lỗi khi xử lý phản hồi từ AI:", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi khi xử lý yêu cầu")
    
    # Lưu lịch sử sau khi có response đầy đủ (Kèm theo user_id)
    await save_turn_async(db_pool, session_id, user_msg, full_response, user_id)
    
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
    user_id = payload.user_id # Lấy user_id từ request
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
            
            # Lưu lịch sử sau khi stream xong (Kèm theo user_id)
            await save_turn_async(db_pool, session_id, user_msg, full_response, user_id)
            
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