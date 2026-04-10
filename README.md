---
title: M Chatbot
emoji: 📚
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# M_chatbot - RAG Chatbot quy che dao tao

Du an nay la API chatbot RAG (Retrieval-Augmented Generation) cho bai toan tra cuu quy che dao tao. He thong dung FastAPI + Qdrant + BM25 + Cross-Encoder rerank + LLM (Groq/Gemini), co ho tro ca response JSON va streaming SSE.

## 1) Luong tong quan

```text
Client (web/mobile)
	|
	| POST /chat or /chat/stream
	v
main.py (FastAPI)
	|
	| lay lich su hoi thoai tu PostgreSQL
	v
core/qa_pipeline.py
	|-- generate_standalone_query (chuan hoa cau hoi theo ngu canh)
	|-- analyze_and_expand_query (phan loai + tao truy van mo rong)
	|-- HybridRetriever.search (BM25 + Vector Search)
	|-- advanced_rerank (Cross-Encoder)
	|-- create_advanced_prompt (tao prompt cuoi)
	|-- goi LLM Groq (fallback Gemini)
	v
Tra ve ket qua cho main.py
	|
	| luu luot chat vao PostgreSQL
	v
Tra ket qua ve Client (JSON hoac SSE delta)
```

## 2) Luong khoi dong he thong

Khi server bat dau, `lifespan` trong `main.py` chay theo thu tu:

1. Doc bien moi truong tu `core/config.py`.
2. Tao pool ket noi PostgreSQL (`asyncpg`) va dam bao bang `history` ton tai.
3. Ket noi Qdrant Cloud.
4. Khoi tao `CollectionRouterRetriever` de tim theo cac collection dang active tren Qdrant.
5. Khoi tao Supabase sync coordinator va chay `startup:initial_sync` thong qua `build_vectorstore_improved` trong `core/vectorstore.py` (co the cho toi da theo `SUPABASE_STARTUP_SYNC_WAIT_SECONDS` hoac chay nen).
6. Bat polling sync dinh ky de dong bo thay doi add/update/delete tu Supabase.
7. Danh dau app san sang (endpoint `/healthz` se bao `ready=true`).

## 3) Luong ingest tai lieu (Supabase-only)

Luong nay duoc kich hoat boi scheduler/event sync trong `core/supabase_sync_service.py` va ingest trong `core/document_ingest_service.py`:

1. Lay danh sach object tu Supabase Storage va diff voi snapshot de xac dinh `added/updated/deleted`.
2. Download tung file can ingest ve file tam.
3. Trich xuat noi dung tai lieu bang bo ham cu (`load_documents_from_file` trong `core/vectorstore.py`).
4. Lam sach text bang bo ham cu (`clean_text` trong `core/text_utils.py`).
5. Chunk van ban bang bo ham cu (`smart_chunking` trong `core/chunking.py`).
6. Embedding chunks bang model trong `core/models.py`.
7. Upsert vector len Qdrant collection theo folder nam hoc.
8. Xoa/ghi de theo `object_path` de dam bao incremental sync va tranh duplicate.

## 3.1) Hoi va tra loi theo nam hoc

He thong ho tro tu dong nhan dien nam hoc khi nguoi dung hoi, vi du:

- `Hoc phi nam 2022-2023 nhu the nao?`
- `Quy dinh thi truc tuyen nam 2021-2022`
- `Quy che hoc bong nam 2023`

Co che xu ly:

1. Pipeline phat hien nam (dang `YYYY-YYYY` hoac nam le `YYYY`) trong cau hoi.
2. Retriever loc tai lieu theo `academic_year` phu hop.
3. Prompt bat buoc LLM uu tien tra loi dung pham vi nam duoc hoi.
4. Neu khong co du lieu cho nam do, chatbot se thong bao ro khong tim thay thong tin phu hop theo nam.

## 4) Luong chat khong streaming (`/chat`)

1. Nhan request (`session_id`, `user_id`, `message`) tai `main.py`.
2. Lay lich su gan day cua session tu PostgreSQL.
3. Goi `ask_ai_improved(...)` trong `core/qa_pipeline.py`.
4. Ben trong pipeline:
   - Tai tao cau hoi doc lap theo ngu canh lich su.
   - Phan loai va mo rong truy van tim kiem.
   - Tim kiem lai voi Hybrid Retriever (BM25 + Vector).
   - Rerank bang Cross-Encoder.
   - Tao prompt cuoi theo mau nghiep vu.
   - Goi LLM tao cau tra loi.
5. Nhan full response va luu ca user/assistant message vao bang `history`.
6. Tra ve JSON: `{ "response": "..." }`.

## 5) Luong chat streaming SSE (`/chat/stream`)

Tuong tu luong `/chat`, khac o cho:

1. `ask_ai_stream_delta(...)` sinh tung doan text nho (delta).
2. `main.py` dong goi tung delta thanh SSE event: `data: {"delta": "...", "done": false}`.
3. Khi xong, gui event ket thuc: `done=true`.
4. Luu full response vao DB sau khi stream hoan tat.

## 6) Giai thich tung file trong luong

### Entry va API layer

- `main.py`: Diem vao cua he thong. Quan ly startup/shutdown, DB pool, retriever, va toan bo endpoint (`/`, `/healthz`, `/sessions/{user_id}`, `/chat/history/{session_id}`, `/chat`, `/chat/stream`).
- `api/chat_api_routers.py`: File router du phong, hien tai de trong.

### Core pipeline

- `core/config.py`: Tap trung bien cau hinh (model names, chunking, retrieval, Qdrant, DB, gioi han context/output).
- `core/qa_pipeline.py`: Nguoi dieu phoi chinh cua luong hoi-dap; bao gom phan tich cau hoi, truy hoi tai lieu, rerank, tao prompt, goi LLM va fallback provider.
- `core/analyze_and_expand.py`: Phan loai cau hoi va tao danh sach truy van mo rong de tim kiem chinh xac hon.
- `core/prompting.py`: Sinh prompt nghiep vu co guardrail de ep cau tra loi bam sat tai lieu.
- `core/retriever.py`: Hybrid retrieval ket hop BM25 va vector similarity bang RRF.
- `core/rerank.py`: Rerank tap tai lieu lay duoc bang Cross-Encoder.

### Du lieu va vector

- `core/vectorstore.py`: Cung cap bo ham xu ly tai lieu cu (doc PDF/DOCX, metadata nam hoc) duoc tai su dung trong luong Supabase ingest, dong thoi chua `build_vectorstore_improved`/`load_vectorstore_improved` cho luong Supabase.
- `core/chunking.py`: Cat van ban thong minh (uu tien giu cau truc bang/danh sach).
- `core/text_utils.py`: Lam sach va chuan hoa noi dung text truoc khi embedding.
- `core/models.py`: Khoi tao embedding model va cross-encoder model.

### Ho tro van hanh

- `core/llm_utils.py`: Ham utility goi LLM an toan (stream/invoke co retry/timeout), hien chua duoc su dung dong nhat tren toan bo pipeline.
- `client_demo.html`: Trang web demo de thu nhanh 2 endpoint `/chat` va `/chat/stream`.
- `Dockerfile`: Cau hinh dong goi va chay API bang Docker/Uvicorn tren cong `7860`.
- `requirements.txt`: Danh sach dependency Python cua du an.

## 7) Bien moi truong quan trong

Toi thieu can co:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `DATABASE_URL`
- `GROQ_API_KEYS` (hoac `GROQ_API_KEY`)

De bat dong bo Supabase Storage (scheduler quet dinh ky):

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_STORAGE_BUCKET` (mac dinh: `file`)
- `SUPABASE_SYNC_INTERVAL_SECONDS` (khuyen nghi 120; he thong tu gioi han trong khoang 60-180 giay)
- `SUPABASE_STARTUP_SYNC_WAIT_SECONDS` (mac dinh: `5`; dat `0` de khong cho initial sync luc khoi dong, giup API len nhanh hon)
- `SUPABASE_ADMIN_SYNC_TOKEN` (du phong cho endpoint admin sync o cac giai doan tiep theo)
- `SUPABASE_SYNC_SNAPSHOT_FILE` (mac dinh: `supabase_sync_snapshot.json`)
- `SUPABASE_SYNC_ALLOWED_IPS` (danh sach IP duoc phep goi endpoint admin sync, cach nhau boi dau phay)
- `SUPABASE_SYNC_ALLOW_PRIVATE_NETWORK` (mac dinh `true`; cho phep IP private/loopback)
- `COLLECTION_ROUTER_TOP_N` (so collection active se tim khi query khong chi dinh nam hoc)

Tuy chon:

- `GEMINI_API_KEYS`
- `ALLOW_ORIGINS`
- `MAX_HISTORY_MESSAGES`, `MAX_CONTEXT_CHARS`, `MAX_OUT_CHARS`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_RESULTS`, `FINAL_TOP_K`

## 8) Chay nhanh local

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Sau khi chay, kiem tra:

- `GET /healthz`
- Dung `client_demo.html` de test ca non-streaming va streaming.

## 9) Ghi chu

- Trong Hugging Face Spaces, frontmatter o dau file README can duoc giu nguyen.
- Lan chay dau co the cham do qua trinh initial sync tu Supabase (download + chunk + embedding + upsert Qdrant).
