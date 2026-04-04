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
4. Neu collection `quy_che_db` chua ton tai: build vectorstore moi bang `core/vectorstore.py`.
5. Neu da ton tai: tai vectorstore va chunks da luu.
6. Khoi tao `HybridRetriever` trong `core/retriever.py`.
7. Danh dau app san sang (endpoint `/healthz` se bao `ready=true`).

## 3) Luong ingest tai lieu (xay dung vectorstore)

Luong nay nam trong `core/vectorstore.py`:

1. Quet de quy file trong thu muc `data/` (`.pdf`, `.doc`, `.docx`), bao gom ca cac thu muc nam hoc nhu `So tay sinh vien 2022-2023/`.
2. Trich xuat noi dung (giu bang tu PDF/DOCX).
3. Lam sach text bang `core/text_utils.py`.
4. Gan metadata `academic_year` cho tung tai lieu/chunk (neu tim thay mau nam hoc `YYYY-YYYY` trong duong dan hoac ten file).
5. Chunk van ban thong minh bang `core/chunking.py`.
6. Embedding chunks bang model trong `core/models.py`.
7. Day vector len Qdrant collection `quy_che_db`.
8. Luu ban sao chunks local vao `vectorstore/chunks.pkl` de startup nhanh hon.
9. Neu phat hien file moi trong `data/` ma chunks cache chua co, he thong tu dong rebuild de dong bo du lieu.

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

- `core/vectorstore.py`: Load tai lieu, tien xu ly, chunking, embedding, tao/tai Qdrant vector store, luu chunks local.
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
- Lan chay dau co the cham do qua trinh doc tai lieu, chunk, embedding va day vector len Qdrant.
